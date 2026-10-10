import math
import threading
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gymnasium.utils.env_checker import check_env

from app.backtesting.engine import run_engine
from app.core.schemas import BacktestRequest
from app.data.errors import DataError
from app.data.market import closing_times, load, synthetic
from app.main import app
from app.ml import jobs, service, storage
from app.ml.diagnostics import distribution_reference, drift, prediction_metrics
from app.ml.evaluation import LookupPolicy, evaluate, evaluation_frame
from app.ml.features import FEATURES, chronological_plan, make_features, make_samples
from app.ml.models import BoostModel
from app.ml.rl import QModel, TradingEnvironment, equity_reward
from app.ml.schemas import ResearchRequest, ResearchResult


def config(**kwargs):
    return ResearchRequest(
        backtest=BacktestRequest(asset="BTC/USD", start=date(2024, 1, 1), end=date(2025, 1, 1)),
        n_estimators=30,
        folds=2,
        episodes=2,
        **kwargs,
    )


@pytest.fixture
def dataset():
    req = config()
    frame = synthetic(req.backtest)
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns, UTC]")
    features = make_features(frame, req.backtest)
    samples = make_samples(frame, req.backtest, req.horizon)
    return req, frame, features, samples


def test_features_do_not_change_when_future_prices_or_actions_change(dataset):
    req, frame, _, _ = dataset
    prefix = frame.iloc[:170].copy()
    reference = make_features(prefix, req.backtest)
    extended = frame.copy()
    for column in ("open", "high", "low", "close"):
        extended.loc[170:, column] *= 100
    ex_date = extended.iloc[200].timestamp.tz_convert("America/New_York").date().isoformat()
    extended.attrs["corporate_actions"] = [
        {"id": "future", "kind": "split", "asset": "BTC/USD", "ex_date": ex_date, "ratio": 5}
    ]
    actual = make_features(extended, req.backtest).iloc[: len(reference)]
    pd.testing.assert_frame_equal(reference, actual.reset_index(drop=True))


def test_next_open_labels_split_and_dividend_manual():
    req = BacktestRequest(start=date(2022, 1, 3), end=date(2024, 1, 1))
    frame = synthetic(req)
    for column in ("open", "high", "low", "close"):
        frame[column] = 20.0
        frame.loc[62:, column] = 10.0
    ex_date = frame.iloc[62].timestamp.tz_convert("America/New_York").date().isoformat()
    entry_date = frame.iloc[61].timestamp.tz_convert("America/New_York").date().isoformat()
    frame.attrs["corporate_actions"] = [
        {"id": "split", "kind": "split", "asset": "AAPL", "ex_date": ex_date, "ratio": 2},
        {
            "id": "income",
            "kind": "cash_dividend",
            "asset": "AAPL",
            "ex_date": ex_date,
            "amount": 1,
            "payable_date": ex_date,
        },
        {
            "id": "entry-income",
            "kind": "cash_dividend",
            "asset": "AAPL",
            "ex_date": entry_date,
            "amount": 9,
            "payable_date": entry_date,
        },
    ]
    sample = make_samples(frame, req, 1).iloc[0]
    assert sample.bar_index == 60
    assert sample.entry_time == frame.iloc[61].timestamp
    assert sample.label_end == frame.iloc[62].timestamp
    # Buy at $20, split into 2 shares, exit at $10 each, plus $2 entitlement.
    # Entering on the prior dividend ex-date does not earn that $9.
    assert sample.target == pytest.approx(0.1)


@pytest.mark.parametrize("horizon", [1, 5])
def test_purged_walk_forward_and_held_out_boundaries(dataset, horizon):
    req, frame, _, _ = dataset
    samples = make_samples(frame, req.backtest, horizon)
    plan, training, holdout = chronological_plan(samples, 0.2, 2)
    previous = set()
    for train, evaluation in plan:
        assert train.signal_time.max() < evaluation.signal_time.min()
        assert train.label_end.max() < evaluation.timestamp.min()
        assert evaluation.label_end.max() < holdout.timestamp.min()
        assert previous.isdisjoint(set(evaluation.bar_index))
        previous.update(evaluation.bar_index)
    assert training.label_end.max() < holdout.timestamp.min()
    validation_rows = pd.concat([block for _, block in plan], ignore_index=True)
    validation_frame = evaluation_frame(frame, validation_rows)
    assert closing_times(validation_frame, req.backtest.asset, "1d")[-1] <= holdout.timestamp.min()
    assert previous.isdisjoint(set(holdout.bar_index))


def test_short_data_and_invalid_research_requests(dataset):
    req, _, _, samples = dataset
    with pytest.raises(DataError, match="250"):
        chronological_plan(samples.iloc[:100], 0.2, 2)
    with pytest.raises(ValueError, match="daily"):
        ResearchRequest(backtest=req.backtest.model_copy(update={"timeframe": "1h"}))
    with pytest.raises(ValueError, match="one-bar"):
        config(algorithm="q_learning", horizon=2)
    with pytest.raises(ValueError):
        config(parent_id="../../etc/passwd")


def test_prediction_metrics_manual():
    result = prediction_metrics([0.1, -0.1, 0.2, -0.2], [0.05, -0.05, -0.1, 0.1], [0.8, 0.2, 0.4, 0.6])
    assert result["mae"] == pytest.approx(0.175)
    assert result["rmse"] == pytest.approx(math.sqrt(0.04625))
    assert result["directional_accuracy"] == 0.5
    assert result["brier_score"] == pytest.approx(0.2)
    assert result["log_loss"] == pytest.approx(-0.5 * (math.log(0.8) + math.log(0.4)))
    assert result["calibration_error"] == pytest.approx(0.4)
    assert result["rank_correlation"] is None
    assert sum(bin["count"] for bin in result["reliability"]) == 4
    assert prediction_metrics(np.arange(30), np.arange(30))["rank_correlation"] == pytest.approx(1)
    assert prediction_metrics(np.arange(30), np.zeros(30))["rank_correlation"] is None


def test_boost_training_deterministic_chronological_calibration_and_reload(dataset, tmp_path):
    req, _, _, samples = dataset
    _, training, holdout = chronological_plan(samples, 0.2, 2)
    model = BoostModel.fit(training, req)
    again = BoostModel.fit(training.copy(), req)
    for first, second in zip(model.predict(holdout), again.predict(holdout)):
        np.testing.assert_array_equal(first, second)
    assert pd.Timestamp(model.metadata["core_end"]) < pd.Timestamp(model.metadata["early_start"])
    assert pd.Timestamp(model.metadata["early_end"]) < pd.Timestamp(model.metadata["calibration_start"])
    assert pd.Timestamp(model.metadata["trained_through"]) < holdout.signal_time.min()
    model.save(tmp_path)
    for first, second in zip(model.predict(holdout), BoostModel.load(tmp_path).predict(holdout)):
        np.testing.assert_array_equal(first, second)
    changed = samples.copy()
    changed.loc[holdout.index, "target"] = 100
    _, same_training, _ = chronological_plan(changed, 0.2, 2)
    np.testing.assert_array_equal(training.target, same_training.target)
    untouched = BoostModel.fit(same_training, req)
    np.testing.assert_array_equal(model.predict(holdout)[0], untouched.predict(holdout)[0])


def test_constant_direction_and_uncalibratable_period(dataset):
    req, _, _, samples = dataset
    training = samples.iloc[:200].copy()
    training["target"] = 0.01
    model = BoostModel.fit(training, req)
    _, probability = model.predict(samples.tail(20))
    assert (probability == 1).all()
    assert model.metadata["warnings"]


def test_predictions_align_with_broker_and_baseline_windows(dataset):
    req, frame, features, samples = dataset
    req = req.model_copy(
        update={
            "min_edge_bps": 0,
            "backtest": req.backtest.model_copy(
                update={"commission_bps": 0, "spread_bps": 0, "slippage_bps": 0, "max_participation": 1}
            ),
        }
    )
    rows = samples.iloc[200:230]

    class AlwaysPositive:
        def predict(self, data):
            return np.full(len(data), 0.1), np.ones(len(data))

    report = evaluate(frame, features, rows, AlwaysPositive(), req, samples.iloc[:200])
    fill = report["strategy"]["trades"][0]
    assert fill["signal_timestamp"] == rows.signal_time.iloc[0].isoformat()
    assert fill["timestamp"] == rows.entry_time.iloc[0].isoformat()
    assert fill["price"] == frame.iloc[int(rows.bar_index.iloc[0]) + 1].open
    for baseline in report["baselines"].values():
        assert [p["timestamp"] for p in baseline["equity"]] == [
            p["timestamp"] for p in report["strategy"]["equity"]
        ]
    assert report["baselines"]["cash"]["metrics"]["net_return"] == 0
    assert all(r["prediction"]["samples"] <= len(rows) for r in report["regimes"])


def test_reward_and_rl_execution_manual():
    req = config().backtest.model_copy(
        update={
            "initial_cash": 1000,
            "allocation": 0.5,
            "cash_reserve": 0,
            "commission_bps": 100,
            "spread_bps": 0,
            "slippage_bps": 0,
        }
    )
    frame = synthetic(req).iloc[:3].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = [10.0, 10.0, 12.0]
    frame["volume"] = 1000000
    features = pd.DataFrame({feature: [0.0] * 3 for feature in FEATURES})
    features["timestamp"] = frame.timestamp
    env = TradingEnvironment(frame, features, req)
    try:
        observation, _ = env.reset(seed=42)
        assert env.observation_space.contains(observation)
        _, reward1, done, _, _ = env.step(1)
        assert not done
        quantity = 500 / 10.1
        equity1 = 500 + quantity * 10
        assert reward1 == pytest.approx(math.log(equity1 / 1000))
        _, reward2, done, _, info = env.step(0)
        equity2 = 500 + quantity * 12 * 0.99
        assert done
        assert info["equity"] == pytest.approx(equity2)
        assert reward2 == pytest.approx(math.log(equity2 / equity1))
        mapping = {frame.iloc[0].timestamp.isoformat(): 1, frame.iloc[1].timestamp.isoformat(): 0}
        independent = run_engine(frame, req, LookupPolicy(mapping))
        assert env.result["equity"] == independent["equity"]
        assert env.result["trades"] == independent["trades"]
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(0)
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    assert equity_reward(100, 90, 0, -0.1, 0.5) == pytest.approx(math.log(0.9) - 0.05)
    assert equity_reward(90, 95, -0.1, -0.05, 0.5) == pytest.approx(math.log(95 / 90))


def test_q_training_frozen_reproducible_and_reload(dataset, tmp_path):
    req, frame, features, samples = dataset
    req = req.model_copy(update={"algorithm": "q_learning"})
    train = samples.iloc[:120]
    model = QModel.fit(frame, features, train, req)
    again = QModel.fit(frame, features, train, req)
    np.testing.assert_array_equal(model.q, again.q)
    assert model.metadata["trained_through"] == train.signal_time.iloc[-1].isoformat()
    model.save(tmp_path)
    restored = QModel.load(tmp_path)
    np.testing.assert_array_equal(model.q, restored.q)
    before = model.q.copy()
    report = evaluate(frame, features, samples.iloc[120:160], model, req, train)
    np.testing.assert_array_equal(before, model.q)
    assert report["prediction"]["mae"] is None
    assert report["prediction"]["brier_score"] is None
    assert all(row["target"] in (0, 1) for row in report["predictions"])


def test_drift_detects_distribution_shift_without_refitting(dataset):
    _, _, _, samples = dataset
    reference = distribution_reference(samples.iloc[:150])
    unchanged = drift(reference, samples.iloc[:150], [])
    assert all(row["psi"] == pytest.approx(0) for row in unchanged["features"])
    shifted = samples.iloc[:100].copy()
    shifted["return_1"] = 100
    report = drift(reference, shifted, [])
    assert next(row for row in report["features"] if row["feature"] == "return_1")["flagged"]


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "JOBS", {})
    return tmp_path


def test_end_to_end_background_api_and_promotion_rejection(dataset, isolated_store, monkeypatch):
    req, frame, _, _ = dataset
    _, provenance = load(req.backtest)
    monkeypatch.setattr(service, "load", lambda request, refresh=False: (frame, provenance))
    client = TestClient(app)
    response = client.post("/api/research/jobs", json=req.model_dump(mode="json"))
    assert response.status_code == 202
    identifier = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get("/api/research/jobs/" + identifier).json()
        if job["status"] in ("failed", "completed"):
            break
        time.sleep(0.02)
    assert job["status"] == "completed", job
    result = client.get("/api/research/versions/" + job["result_id"])
    assert result.status_code == 200
    saved = ResearchResult.model_validate(result.json())
    assert saved.provenance.synthetic
    assert saved.holdout.prediction.samples > 40
    assert saved.training["trained_through"] < saved.training["holdout_start"]
    assert client.get("/api/research/versions").json()[0]["id"] == saved.id
    assert client.get("/api/research/versions/invalid").status_code == 404
    assert client.get("/api/research/jobs/invalid").status_code == 404
    if saved.promotion["eligible"]:
        assert client.post(f"/api/research/versions/{saved.id}/promote").status_code == 200
        # A stale previously evaluated candidate cannot replace the active model.
        assert client.post(f"/api/research/versions/{saved.id}/promote").status_code == 409
    else:
        assert client.post(f"/api/research/versions/{saved.id}/promote").status_code == 409
    repeat = service.execute(req)
    assert repeat["id"] == saved.id


def test_registry_preserves_previous_version_and_rollback(isolated_store):
    req = config()
    first, second = "a" * 24, "b" * 24
    base = {
        "created_at": "2024-01-01",
        "config": req.model_dump(mode="json"),
        "promotion": {"eligible": True, "active_model_at_training": None},
        "holdout": {"strategy": {"metrics": {"net_return": 0.1}}},
    }
    storage.atomic_json(storage.folder() / first / "result.json", {**base, "id": first})
    storage.promote(first)
    base["promotion"] = {"eligible": True, "active_model_at_training": first}
    storage.atomic_json(storage.folder() / second / "result.json", {**base, "id": second})
    storage.promote(second)
    assert storage.registry()["active"][storage.key(req)] == second
    storage.rollback(first)
    assert storage.registry()["active"][storage.key(req)] == first
    assert len(storage.versions()) == 2
    assert all(row["rollback_available"] for row in storage.versions())
    other = "c" * 24
    storage.atomic_json(storage.folder() / other / "result.json", {**base, "id": other})
    with pytest.raises(ValueError, match="previously active"):
        storage.rollback(other)


def test_job_failure_and_concurrency_limit(isolated_store, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def failing(request, progress):
        started.set()
        release.wait(timeout=10)
        raise DataError("Verified data unavailable")

    monkeypatch.setattr(jobs, "execute", failing)
    job = jobs.submit(config())
    assert started.wait(timeout=5)
    try:
        with pytest.raises(jobs.ResearchBusy):
            jobs.submit(config())
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while jobs.retrieve(job["id"])["status"] not in ("failed", "completed") and time.monotonic() < deadline:
        time.sleep(0.01)
    result = jobs.retrieve(job["id"])
    assert result["status"] == "failed"
    assert result["error"] == "Verified data unavailable"


def test_retraining_preserves_snapshot_and_requires_new_forward_data(dataset, isolated_store, monkeypatch):
    req, frame, _, _ = dataset
    _, provenance = load(req.backtest)
    refreshes = []

    def provider(request, refresh=False):
        refreshes.append(refresh)
        return frame, provenance

    monkeypatch.setattr(service, "load", provider)
    parent = service.execute(req)
    snapshot = storage.folder() / parent["id"] / "dataset.parquet"
    before = snapshot.read_bytes()
    candidate = service.execute(req.model_copy(update={"parent_id": parent["id"], "refresh_data": True}))
    assert candidate["id"] != parent["id"]
    assert refreshes == [False, True]
    assert candidate["promotion"]["incumbent_comparison"]["samples"] == 0
    assert not candidate["promotion"]["eligible"]
    assert snapshot.read_bytes() == before
    pd.testing.assert_frame_equal(pd.read_parquet(snapshot), frame)
    assert (storage.folder() / parent["id"] / "dataset-actions.json").exists()
    assert len(storage.versions()) == 2


def test_manual_retrain_route_refreshes_and_preserves_configuration(isolated_store, monkeypatch):
    from datetime import UTC, datetime

    req = config()
    identifier = "d" * 24
    storage.atomic_json(
        storage.folder() / identifier / "result.json", {"config": req.model_dump(mode="json")}
    )
    captured = []

    def submit(request):
        captured.append(request)
        return {
            "id": "e" * 24,
            "status": "queued",
            "progress": "Queued",
            "created_at": datetime.now(UTC).isoformat(),
            "result_id": None,
            "error": None,
        }

    monkeypatch.setattr(jobs, "submit", submit)
    response = TestClient(app).post(f"/api/research/versions/{identifier}/retrain")
    assert response.status_code == 202
    candidate = captured[0]
    assert candidate.refresh_data
    assert candidate.parent_id == identifier
    assert candidate.backtest.end == datetime.now(UTC).date()
    assert candidate.seed == req.seed
    assert candidate.backtest.commission_bps == req.backtest.commission_bps


def test_restarted_jobs_are_explicitly_interrupted(isolated_store):
    identifier = "f" * 24
    storage.atomic_json(
        storage.folder() / "jobs" / f"{identifier}.json",
        {
            "id": identifier,
            "status": "running",
            "progress": "Training",
            "created_at": "2024-01-01",
            "error": None,
            "result_id": None,
        },
    )
    jobs.initialize()
    assert jobs.retrieve(identifier)["status"] == "interrupted"
    assert "restart" in jobs.retrieve(identifier)["error"]


def test_promotion_criteria_reject_complexity_without_baseline_improvement():
    from app.ml.service import acceptance

    validation = {
        "strategy": {"metrics": {"net_return": 0.05, "sharpe": 1.0, "max_drawdown": -0.03, "turnover": 3}},
        "prediction": {"samples": 100, "mae": 0.005},
        "prediction_baselines": {"zero": {"mae": 0.01}, "mean": {"mae": 0.009}},
        "baselines": {"cash": {"metrics": {"net_return": 0}}, "buy_hold": {"metrics": {"net_return": 0.1}}},
        "regimes": [{"sufficient": True, "net_return": 0.02}, {"sufficient": True, "net_return": 0.03}],
    }
    result = acceptance(validation, "xgboost", None)
    assert not result["eligible"]
    assert not next(c for c in result["criteria"] if "Net return exceeds" in c["name"])["passed"]
