import copy
import threading
import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import BacktestRequest
from app.data.errors import DataError
from app.data.market import load, synthetic
from app.main import app
from app.ml import jobs, search, service, storage
from app.ml.features import chronological_plan, make_samples
from app.ml.schemas import ResearchRequest, ResearchResult, SearchRequest, SearchResult


@pytest.fixture
def research(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "JOBS", {})
    config = ResearchRequest(
        backtest=BacktestRequest(asset="BTC/USD", start=date(2024, 1, 1), end=date(2025, 1, 1)),
        n_estimators=20,
        folds=2,
    )
    frame = synthetic(config.backtest)
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns, UTC]")
    _, provenance = load(config.backtest)
    monkeypatch.setattr(service, "load", lambda request, refresh=False: (frame, provenance))
    return config, frame, provenance


def test_search_candidates_include_current_are_unique_and_keep_comparison_fixed(research):
    config, _, _ = research
    request = SearchRequest(research=config, trials=48)
    first, second = search.candidates(request), search.candidates(request)
    assert first == second
    assert first[0] == config
    assert len({tuple(getattr(candidate, key) for key in search.SPACE) for candidate in first}) == 48
    fixed = config.model_dump(exclude=set(search.SPACE))
    assert all(candidate.model_dump(exclude=set(search.SPACE)) == fixed for candidate in first)
    other = search.candidates(request.model_copy(update={"research": config.model_copy(update={"seed": 99})}))
    assert [c.max_depth for c in first[1:]] != [c.max_depth for c in other[1:]]


@pytest.mark.parametrize(
    "objective,expected",
    [
        ("sharpe", [1, 2, 3, 5, 4]),
        ("net_return", [2, 1, 3, 5, 4]),
    ],
)
def test_search_ranking_hand_calculated(objective, expected):
    def row(index, net, sharpe, eligible, drawdown):
        return {
            "trial": index,
            "status": "completed",
            "metrics": {"net_return": net, "sharpe": sharpe, "max_drawdown": drawdown, "turnover": 3},
            "prediction": {"mae": 0.01},
            "promotion": {"eligible": eligible},
            "risk_passed": drawdown >= -0.2,
        }

    # $10,000 portfolios: $400 vs $1,000 net profit. Sharpe ranking selects
    # the lower-profit .5-Sharpe candidate; return ranking selects $1,000.
    # A 40% drawdown ($4,000 below the peak) fails the 20% guardrail.
    rows = [
        row(1, 0.04, 0.5, True, -0.08),
        row(2, 0.1, 0.25, True, -0.1),
        row(3, 0.4, 3, False, -0.1),
        row(4, 0.9, 8, False, -0.4),
        row(5, 0, None, False, 0),
    ]
    ranked = search.rank_trials(rows, objective)
    assert [trial["trial"] for trial in ranked] == expected
    assert [trial["rank"] for trial in ranked] == list(range(1, 6))
    tied = copy.deepcopy(rows[0])
    tied["trial"] = 6
    assert search.rank_trials([tied, rows[0]], objective)[0]["trial"] == 1


@pytest.mark.parametrize("mode", ["random", "coarse_to_fine", "local"])
def test_search_only_tests_selected_winner_and_heldout_changes_cannot_change_ranking(
    research, monkeypatch, mode
):
    config, frame, provenance = research
    samples = make_samples(frame, config.backtest, config.horizon)
    _, _, holdout = chronological_plan(samples, config.holdout_fraction, config.folds)
    boundary = holdout.timestamp.iloc[0]
    real_evaluate = service.evaluate
    calls, downloads = [], []

    def tracked(frame, features, rows, model, request, training):
        final_test = rows.timestamp.iloc[0] >= boundary
        calls.append(final_test)
        assert training.label_end.max() < boundary
        if final_test:
            identifier = ("a" if len(downloads) == 1 else "b") * 24
            selected = storage.search_result(identifier)
            assert selected["selected_trial"] is not None
            assert selected["winner_id"] is None
        else:
            assert rows.label_end.max() < boundary
        return real_evaluate(frame, features, rows, model, request, training)

    def provider(request, refresh=False):
        downloads.append(refresh)
        return frame, provenance

    monkeypatch.setattr(service, "evaluate", tracked)
    monkeypatch.setattr(service, "load", provider)
    request = SearchRequest(research=config, trials=6, mode=mode)
    first = search.execute(request, "a" * 24)
    assert calls == [False] * (request.trials * (config.folds + 1)) + [True]
    assert len(downloads) == 1
    winner = ResearchResult.model_validate(storage.result(first["winner_id"]))
    assert winner.search["id"] == first["id"]
    assert winner.search["selected_trial"] == first["selected_trial"]
    selected = next(t for t in first["trials"] if t["trial"] == first["selected_trial"])
    assert all(getattr(winner.config, key) == value for key, value in selected["parameters"].items())
    assert selected["metrics"]["excess_return"] == pytest.approx(
        selected["metrics"]["net_return"]
        - max(value["net_return"] for value in selected["baselines"].values())
    )
    assert len(storage.versions()) == 1
    assert storage.registry()["active"] == {}

    # Change only prices at and after the reserved test boundary. None of these
    # prices may influence candidates' validation scores or parameter selection.
    frame = frame.copy(deep=True)
    mask = frame.timestamp >= boundary
    for column in ("open", "high", "low", "close"):
        frame.loc[mask, column] *= 3
    calls.clear()
    second = search.execute(request, "b" * 24)
    assert first["trials"] == second["trials"]
    assert first["selected_trial"] == second["selected_trial"]
    assert calls == [False] * (request.trials * (config.folds + 1)) + [True]
    assert len(downloads) == 2
    assert len(storage.versions()) == 2
    assert len(storage.searches()) == 2
    assert storage.result(second["winner_id"])["holdout"] != winner.holdout.model_dump(mode="json")


@pytest.mark.parametrize("all_fail", [False, True])
def test_search_records_trial_failure_and_never_tests_failed_candidates(research, monkeypatch, all_fail):
    config, _, _ = research
    original = service.validate
    attempts = []

    def validate(request, prepared, progress):
        attempts.append(request)
        if all_fail or len(attempts) == 1:
            raise DataError("Candidate could not train")
        return original(request, prepared, progress)

    monkeypatch.setattr(service, "validate", validate)
    request = SearchRequest(research=config, trials=2)
    if all_fail:
        with pytest.raises(DataError, match="Every parameter trial failed"):
            search.execute(request, "c" * 24)
        result = storage.search_result("c" * 24)
        assert result["status"] == "failed"
        assert result["winner_id"] is None
        assert len(storage.versions()) == 0
    else:
        result = search.execute(request, "c" * 24)
        assert result["selected_trial"] == 2
        assert result["trials"][0]["rank"] == 1
        assert len(storage.versions()) == 1
    failed = [t for t in result["trials"] if t["status"] == "failed"]
    assert failed
    assert all(t["error"] == "Candidate could not train" and t["rank"] is None for t in failed)


def test_search_api_job_concurrency_persistence_and_input_errors(research, monkeypatch):
    config, _, _ = research
    client = TestClient(app)
    for update in ({"trials": 49}, {"trials": 1}, {"objective": "accuracy"}, {"mode": "test_return"}):
        assert (
            client.post(
                "/api/research/searches",
                json={**SearchRequest(research=config).model_dump(mode="json"), **update},
            ).status_code
            == 422
        )
    assert (
        client.post(
            "/api/research/searches",
            json=SearchRequest(research=config).model_dump(mode="json")
            | {"research": config.model_copy(update={"algorithm": "q_learning"}).model_dump(mode="json")},
        ).status_code
        == 422
    )
    assert client.get("/api/research/searches/invalid").status_code == 404
    started, release = threading.Event(), threading.Event()
    original = search.execute

    def delayed(request, identifier, progress):
        started.set()
        assert release.wait(timeout=10)
        return original(request, identifier, progress)

    monkeypatch.setattr(search, "execute", delayed)
    request = SearchRequest(research=config, trials=2)
    response = client.post("/api/research/searches", json=request.model_dump(mode="json"))
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "search"
    assert job["search_id"] == job["id"]
    assert started.wait(timeout=5)
    try:
        assert client.post("/api/research/jobs", json=config.model_dump(mode="json")).status_code == 409
        assert client.post("/api/research/searches", json=request.model_dump(mode="json")).status_code == 409
    finally:
        release.set()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/research/jobs/" + job["id"]).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert job["status"] == "completed", job
    result = SearchResult.model_validate(client.get("/api/research/searches/" + job["search_id"]).json())
    assert result.winner_id == job["result_id"]
    assert result.trials[0]["trial"] == result.selected_trial
    assert client.get("/api/research/searches").json()[0]["id"] == result.id
    assert client.get("/api/research/versions/" + result.winner_id).status_code == 200
    jobs.JOBS.clear()
    jobs.initialize()
    assert jobs.retrieve(job["id"])["kind"] == "search"
    assert jobs.retrieve(job["id"])["status"] == "completed"


@pytest.mark.parametrize("entry_rule", ["return_and_probability", "probability_only"])
def test_local_candidates_are_bounded_unique_and_preserve_data_and_model_family(research, entry_rule):
    config, _, _ = research
    config = config.model_copy(update={"entry_rule": entry_rule, "subsample": 0.8, "colsample_bytree": 0.8})
    request = SearchRequest(research=config, trials=48, mode="local")
    first = list(search.candidate_plan(request, list))
    assert first == list(search.candidate_plan(request, list))
    assert first[0] == (config, "current", None)
    assert len(first) == len({search.signature(candidate) for candidate, _, _ in first}) == 48
    fixed = config.model_dump(exclude=set(search.LOCAL))
    assert all(candidate.model_dump(exclude=set(search.LOCAL)) == fixed for candidate, _, _ in first)
    for candidate, phase, origin in first[1:]:
        assert phase == "refine" and origin == 1
        changed = {key for key in search.LOCAL if getattr(candidate, key) != getattr(config, key)}
        assert len(changed) == 1 or changed == {"min_edge_bps", "probability_threshold"}
        for key, (_, low, high) in search.LOCAL.items():
            assert low <= getattr(candidate, key) <= high
    assert first[1][0].probability_threshold == pytest.approx(0.54)
    assert first[2][0].probability_threshold == pytest.approx(0.56)
    if entry_rule == "probability_only":
        assert all(candidate.min_edge_bps == config.min_edge_bps for candidate, _, _ in first)


def test_refinement_centers_follow_broad_validation_rank_and_are_frozen(research):
    config, _, _ = research
    request = SearchRequest(research=config, trials=12, mode="coarse_to_fine", objective="net_return")
    recorded = []
    plan = search.candidate_plan(request, lambda: recorded)
    broad = []
    for index in range(1, 7):
        candidate, phase, origin = next(plan)
        broad.append(candidate)
        assert phase == ("current" if index == 1 else "broad") and origin is None
        # Returns 1..6% on $10,000 mean profits $100..$600. Trial 6's 30%
        # drawdown fails the guardrail, so centers are trials 5 and 4.
        recorded.append(
            {
                "trial": index,
                "status": "completed",
                "parameters": {key: getattr(candidate, key) for key in search.SPACE},
                "metrics": {
                    "net_return": index / 100,
                    "sharpe": 1,
                    "max_drawdown": -0.3 if index == 6 else -0.1,
                    "turnover": 2,
                },
                "prediction": {"mae": 0.01},
                "promotion": {"eligible": False},
                "risk_passed": index != 6,
            }
        )
    candidate, phase, origin = next(plan)
    assert phase == "refine" and origin == 5
    changed = [key for key in search.SPACE if getattr(candidate, key) != getattr(broad[4], key)]
    assert changed == ["probability_threshold"]
    # A later refinement's enormous score must not redefine the frozen broad centers.
    recorded.append({**recorded[0], "trial": 7, "metrics": {**recorded[0]["metrics"], "net_return": 100}})
    remaining = list(plan)
    assert {origin for _, _, origin in remaining} == {4, 5}
    assert len(remaining) == 5
