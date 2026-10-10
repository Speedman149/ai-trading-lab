from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backtesting.engine import run_engine
from app.core.schemas import BacktestRequest
from app.data.market import load, synthetic
from app.main import app
from app.ml import search, service, storage
from app.ml.features import make_samples
from app.ml.grid import grid_candidates, grid_space
from app.ml.models import BoostModel, recency_weights
from app.ml.planning import for_candidate, plan
from app.ml.schemas import ResearchRequest, SearchRequest


def stop_fixture(prices, **changes):
    req = BacktestRequest(
        asset="AAPL",
        start=date(2024, 1, 2),
        end=date(2024, 1, 16),
        strategy="buy_hold",
        initial_cash=1000,
        allocation=0.5,
        cash_reserve=0,
        commission_bps=0,
        spread_bps=0,
        slippage_bps=0,
        stop_loss_pct=10,
        **changes,
    )
    frame = synthetic(req).iloc[: len(prices)].copy()
    for col in ("open", "high", "low", "close"):
        frame[col] = np.array(prices, dtype=float)
    frame["volume"] = 1000000
    return req, frame


@pytest.mark.parametrize(
    "entry_day,gap,expected", [(False, False, 950), (True, False, 950), (False, True, 900)]
)
def test_stop_touch_entry_day_and_gap_manual(entry_day, gap, expected):
    # $500 cash + 5 shares at $100. A touch exits at $90 -> $950;
    # an $80 gap exits at $80 -> $900, despite a nominal 10% stop.
    req, frame = stop_fixture([100, 100, 80 if gap else 100])
    frame.loc[1 if entry_day else 2, "low"] = 80
    result = run_engine(frame, req)
    assert [trade["side"] for trade in result["trades"]] == ["buy", "sell"]
    assert result["trades"][1]["reason"] == "stop_loss"
    assert result["metrics"]["final_equity"] == pytest.approx(expected)
    assert result["metrics"]["transaction_costs"] == 0
    assert result["metrics"]["realized_pnl"] == pytest.approx(expected - 1000)


def test_stop_costs_are_not_price_loss():
    req, frame = stop_fixture([100, 100, 100])
    req = req.model_copy(
        update={"commission_bps": 100, "fixed_fee": 2, "spread_bps": 100, "slippage_bps": 50}
    )
    frame.loc[2, "low"] = 80
    result = run_engine(frame, req)
    qty = 498 / (101 * 1.01)
    stop = 101 * 0.9
    sell = stop * 0.99
    expected = 500 + qty * sell * 0.99 - 2
    assert result["metrics"]["final_equity"] == pytest.approx(expected)
    trade = result["trades"][1]
    assert trade["reference_price"] == pytest.approx(stop)
    assert trade["spread_slippage_cost"] == pytest.approx(qty * stop * 0.01)
    assert trade["commission"] == pytest.approx(qty * sell * 0.01 + 2)


def test_stop_split_dividend_and_remaining_exit_manual():
    req, frame = stop_fixture([100, 100, 50, 50])
    frame.loc[2, "low"] = 40
    ex = frame.timestamp.iloc[2].tz_convert("America/New_York").date().isoformat()
    pay = frame.timestamp.iloc[3].tz_convert("America/New_York").date().isoformat()
    frame.attrs["corporate_actions"] = [
        {"id": "s", "kind": "split", "asset": "AAPL", "ex_date": ex, "ratio": 2},
        {
            "id": "d",
            "kind": "cash_dividend",
            "asset": "AAPL",
            "ex_date": ex,
            "amount": 1,
            "payable_date": pay,
        },
    ]
    result = run_engine(frame, req)
    # 10 split-adjusted shares sell at $45 plus $10 entitlement.
    assert result["trades"][1]["quantity"] == 10
    assert result["trades"][1]["price"] == 45
    assert result["metrics"]["final_equity"] == 960
    assert result["metrics"]["dividend_income"] == 10


def test_stop_partial_exit_latches_and_uses_prior_volume_only():
    req, frame = stop_fixture([100, 100, 100, 120])
    # Buy 5 shares using day 0's volume. Day 2 can sell only 2 shares,
    # then the remaining 3 exit at day 3's $120 open, even after a bounce.
    frame.loc[1, "volume"] = 200
    frame.loc[2, "low"] = 80
    frame.loc[2, "volume"] = 300
    result = run_engine(frame, req)
    assert [t["quantity"] for t in result["trades"]] == [5, 2, 3]
    assert [t["price"] for t in result["trades"]] == [100, 90, 120]
    assert result["metrics"]["final_equity"] == 1040
    changed = frame.copy()
    changed.loc[2, "volume"] = 30000
    other = run_engine(changed, req)
    assert result["trades"][:2] == other["trades"][:2]


def test_weights_calendar_half_life_manual():
    req = ResearchRequest(recency_half_life_days=90)
    rows = pd.DataFrame({"signal_time": pd.to_datetime(["2024-01-01", "2024-03-31", "2024-06-29"], utc=True)})
    # Ages 180, 90, 0 days -> raw weights 1/4, 1/2, 1.
    weights = recency_weights(rows, req)
    np.testing.assert_allclose(weights, np.array([0.25, 0.5, 1]) / (1.75 / 3))
    np.testing.assert_allclose(
        recency_weights(rows, req.model_copy(update={"recency_half_life_days": 0})), [1, 1, 1]
    )


def research_fixture(monkeypatch, tmp_path, **changes):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    req = ResearchRequest(
        backtest=BacktestRequest(asset="BTC/USD", start=date(2024, 1, 1), end=date(2025, 7, 1)),
        n_estimators=20,
        folds=2,
        **changes,
    )
    frame = synthetic(req.backtest)
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns, UTC]")
    _, provenance = load(req.backtest)
    monkeypatch.setattr(service, "load", lambda request, refresh=False: (frame, provenance))
    return req, frame


def test_monthly_three_month_window_purges_and_recent_refit(monkeypatch, tmp_path):
    req, frame = research_fixture(
        monkeypatch,
        tmp_path,
        training_window_months=3,
        validation_scheme="monthly",
        recency_half_life_days=90,
    )
    samples = make_samples(frame, req.backtest, 1)
    blocks, training, holdout = plan(samples, req)
    assert len(blocks) > req.folds
    for train, evaluation in blocks:
        assert train.signal_time.min() >= evaluation.timestamp.iloc[0] - pd.DateOffset(months=3)
        assert train.label_end.max() < evaluation.timestamp.iloc[0]
        assert evaluation.label_end.max() < holdout.timestamp.iloc[0]
        assert evaluation.timestamp.dt.month.nunique() == 1
    model = BoostModel.fit(training, req)
    assert model.metadata["regression_fit_rows"] == len(training)
    assert model.metadata["regression_fit_through"] == training.label_end.max().isoformat()
    assert pd.Timestamp(model.metadata["direction_fit_through"]) < pd.Timestamp(
        model.metadata["calibration_start"]
    )
    assert model.metadata["recency"]["newest_weight"] > model.metadata["recency"]["oldest_weight"]
    predicted = model.predict(holdout)
    model.save(tmp_path / "model")
    restored = BoostModel.load(tmp_path / "model")
    for a, b in zip(predicted, restored.predict(holdout)):
        np.testing.assert_array_equal(a, b)


def test_decimal_grid_all_combinations_and_integer_validation():
    req = SearchRequest(
        research=ResearchRequest(),
        mode="grid",
        grid={
            "probability_threshold": {"minimum": 0.55, "maximum": 0.552, "step": 0.001},
            "stop_loss_pct": {"minimum": 2.9, "maximum": 3.1, "step": 0.1},
            "max_depth": {"minimum": 2, "maximum": 3, "step": 1},
        },
    )
    assert grid_space(req)["stop_loss_pct"] == [2.9, 3.0, 3.1]
    candidates = list(grid_candidates(req))
    assert len(candidates) == 18
    assert len({(c.probability_threshold, c.backtest.stop_loss_pct, c.max_depth) for c in candidates}) == 18
    for invalid in (
        {"max_depth": {"minimum": 2, "maximum": 3, "step": 0.1}},
        {"seed": {"minimum": 0, "maximum": 1000000, "step": 1}},
        {"probability_threshold": {"minimum": 0.5, "maximum": 0.9, "step": 0.001}},
        {"unknown": {"minimum": 0, "maximum": 1, "step": 0.1}},
    ):
        with pytest.raises(ValueError):
            SearchRequest(research=ResearchRequest(), mode="grid", grid=invalid)
    client = TestClient(app)
    assert (
        client.post(
            "/api/research/searches",
            json={
                **req.model_dump(mode="json"),
                "grid": {"max_depth": {"minimum": 2, "maximum": 3, "step": 0.1}},
            },
        ).status_code
        == 422
    )


def test_grid_horizon_folds_seed_shared_dates_final_once_no_leakage(monkeypatch, tmp_path):
    req, frame = research_fixture(monkeypatch, tmp_path, recency_half_life_days=90, training_window_months=6)
    request = SearchRequest(
        research=req,
        mode="grid",
        grid={
            "horizon": {"minimum": 1, "maximum": 2, "step": 1},
            "folds": {"minimum": 2, "maximum": 3, "step": 1},
            "seed": {"minimum": 42, "maximum": 43, "step": 1},
            "stop_loss_pct": {"minimum": 3, "maximum": 3, "step": 0.1},
        },
    )
    anchor = service.prepare(req.model_copy(update={"horizon": 2}))
    anchor["reference_samples"] = make_samples(frame, req.backtest, 2)
    spans = []
    for candidate in grid_candidates(request):
        prepared = for_candidate(anchor, candidate)
        blocks = prepared["plan"]
        assert len(blocks) == candidate.folds
        spans.append(
            (
                blocks[0][1].timestamp.iloc[0],
                blocks[-1][1].attrs["simulation_end"],
                prepared["holdout"].timestamp.iloc[0],
                prepared["holdout"].attrs["simulation_end"],
            )
        )
    assert len(set(spans)) == 1
    final_calls = []
    original = service.complete

    def complete(*args, **kwargs):
        final_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "complete", complete)
    first = search.execute(request, "a" * 24)
    assert len(first["trials"]) == 8
    assert all(t["status"] == "completed" for t in first["trials"])
    assert len(final_calls) == 1
    saved_config = ResearchRequest.model_validate(storage.result(first["winner_id"])["config"])
    assert saved_config.evaluation_anchor_horizon == 2
    # Reopening/retraining a grid winner keeps its canonical comparison dates.
    reopened = service.prepare(saved_config)
    assert reopened["plan"][0][1].timestamp.iloc[0] == spans[0][0]
    assert reopened["plan"][-1][1].attrs["simulation_end"] == spans[0][1]
    assert reopened["holdout"].timestamp.iloc[0] == spans[0][2]
    boundary = spans[0][2]
    for col in ("open", "high", "low", "close"):
        frame.loc[frame.timestamp >= boundary, col] *= 3
    second = search.execute(request, "b" * 24)
    assert first["trials"] == second["trials"]
    assert first["selected_trial"] == second["selected_trial"]
    winner = storage.result(first["winner_id"])
    assert winner["config"]["backtest"]["stop_loss_pct"] == 3
    assert winner["holdout"] != storage.result(second["winner_id"])["holdout"]
    # Buy-and-hold remains a true no-stop benchmark across all trials.
    assert len({t["baselines"]["buy_hold"]["net_return"] for t in first["trials"]}) == 1


def test_stop_does_not_oversell_after_partial_strategy_exit():
    req, frame = stop_fixture([100, 100, 100, 120])
    frame.loc[1, "volume"] = 200
    frame.loc[2, "low"] = 80
    frame.loc[2, "volume"] = 300

    class Exit:
        def target(self, history):
            return float(len(history) == 1)

    result = run_engine(frame, req, Exit())
    # The regular exit sells 2 at $100 using all of the day's sell allowance.
    # The latched stop waits for the next opening and sells the remaining 3.
    assert [t["quantity"] for t in result["trades"]] == [5, 2, 3]
    assert [t["price"] for t in result["trades"]] == [100, 100, 120]
    assert [t["reason"] for t in result["trades"]] == ["signal", "signal", "stop_loss"]
    assert result["metrics"]["final_equity"] == 1060
    assert all(point["quantity"] >= 0 for point in result["equity"])


def test_disabled_stop_ignores_intrabar_low():
    req, frame = stop_fixture([100, 100, 110])
    req = req.model_copy(update={"stop_loss_pct": 0})
    frame.loc[1, "low"] = 50
    result = run_engine(frame, req)
    assert len(result["trades"]) == 1
    assert result["metrics"]["final_equity"] == 1050


def test_grid_candidate_uses_matching_incumbent_family(monkeypatch, tmp_path):
    req, frame = research_fixture(monkeypatch, tmp_path)
    prepared = service.prepare(req)
    prepared["reference_samples"] = make_samples(frame, req.backtest, 1)
    candidate = req.model_copy(update={"backtest": req.backtest.model_copy(update={"stop_loss_pct": 3})})
    prepared["active_by_family"] = {storage.key(req): "a" * 24, storage.key(candidate): "b" * 24}
    prepared["active"] = "a" * 24
    assert for_candidate(prepared, candidate)["active"] == "b" * 24


def test_weighted_refit_learns_latest_rows_manual():
    # Constant features force a constant return forecast. Thirty rows each
    # at ages 180/90/0 carry weights 1/4, 1/2, 1 and returns .1/.2/.3.
    # The full weighted refit must predict (.025 + .1 + .3)/1.75.
    from app.ml.features import FEATURES

    training = pd.DataFrame({feature: np.zeros(90) for feature in FEATURES})
    training["timestamp"] = pd.to_datetime(
        ["2024-01-01"] * 30 + ["2024-03-31"] * 30 + ["2024-06-29"] * 30, utc=True
    )
    training["signal_time"] = training.timestamp
    training["label_end"] = training.timestamp + pd.Timedelta(days=1)
    training["target"] = [0.1] * 30 + [0.2] * 30 + [0.3] * 30
    cfg = ResearchRequest(n_estimators=20, max_depth=1, recency_half_life_days=90)
    weighted = BoostModel.fit(training, cfg)
    equal = BoostModel.fit(training, cfg.model_copy(update={"recency_half_life_days": 0}))
    assert weighted.predict(training.tail(1))[0][0] == pytest.approx(0.425 / 1.75, abs=1e-6)
    assert equal.predict(training.tail(1))[0][0] == pytest.approx(0.2, abs=1e-6)
    assert weighted.metadata["recency"]["effective_samples"] == pytest.approx(70)


def test_disabled_stop_preserves_legacy_registry_key():
    import hashlib
    import json

    config = ResearchRequest()
    legacy = config.backtest.model_dump(mode="json", exclude={"stop_loss_pct"})
    costs = {k: v for k, v in legacy.items() if k not in ("start", "end", "strategy", "fast", "slow")}
    expected = hashlib.sha256(
        json.dumps({"backtest": costs, "algorithm": "xgboost", "horizon": 1}, sort_keys=True).encode()
    ).hexdigest()[:24]
    assert storage.key(config) == expected
