import copy
import json
import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backtesting.engine import run_engine
from app.core.schemas import BacktestRequest
from app.data.market import synthetic
from app.main import app
from app.ml import jobs, service, storage
from app.ml.evaluation import activity_diagnostics
from app.ml.features import FEATURES, TECHNICAL, make_features, make_samples
from app.ml.models import BoostModel, PredictionPolicy
from app.ml.schemas import ResearchRequest


def request(**changes):
    return ResearchRequest(
        backtest=BacktestRequest(asset="BTC/USD", start=date(2024, 1, 1), end=date(2025, 1, 1)),
        n_estimators=20,
        folds=2,
        **changes,
    )


def test_expanded_features_hand_calculated_and_causal():
    config = request()
    frame = synthetic(config.backtest).iloc[:80].copy()
    prices = np.arange(1, 81, dtype=float)
    frame["open"] = frame["close"] = prices
    frame["high"], frame["low"] = prices + 1, prices - 0.5
    frame["volume"] = 1000000
    actual = make_features(frame, config.backtest).iloc[-1]
    assert actual.return_21 == pytest.approx(80 / 59 - 1)
    assert actual.ma_gap_50 == pytest.approx(80 / 55.5 - 1)
    # Last twenty prices are 61..80: mean 70.5, sample variance 35.
    assert actual.bollinger_percent_b == pytest.approx(0.5 + 9.5 / (4 * math.sqrt(35)))
    # True range is 2 on every warmed-up day (high minus previous close).
    assert actual.atr_ratio_14 == pytest.approx(2 / 80)
    fast = slow = 1.0
    signal = 0.0
    for price in range(2, 81):
        fast += (price - fast) * (2 / 13)
        slow += (price - slow) * (2 / 27)
        signal += ((fast - slow) - signal) * (2 / 10)
    assert actual.ppo_histogram == pytest.approx((fast - slow - signal) / slow)
    prefix = make_features(frame.iloc[:70], config.backtest)
    changed = frame.copy()
    for column in ("open", "high", "low", "close"):
        changed.loc[70:, column] *= 100
    pd.testing.assert_frame_equal(
        prefix, make_features(changed, config.backtest).iloc[: len(prefix)].reset_index(drop=True)
    )


def test_sampling_regularization_feature_selection_and_legacy_model_reload(tmp_path):
    config = request(
        feature_set="technical", subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1
    )
    frame = synthetic(config.backtest)
    samples = make_samples(frame, config.backtest, 1)
    training, unseen = samples.iloc[:150], samples.iloc[170:190]
    model = BoostModel.fit(training, config)
    assert model.features == FEATURES + TECHNICAL
    for key in ("subsample", "colsample_bytree", "min_child_weight", "reg_lambda"):
        assert model.regressor.get_params()[key] == getattr(config, key)
    repeated = BoostModel.fit(training, config)
    expected = model.predict(unseen)
    for first, second in zip(expected, repeated.predict(unseen)):
        np.testing.assert_array_equal(first, second)
    model.save(tmp_path)
    metadata_path = tmp_path / "model.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("features")  # Older files do not have this field.
    metadata_path.write_text(json.dumps(metadata))
    restored = BoostModel.load(tmp_path)
    assert restored.features == model.features
    for first, second in zip(expected, restored.predict(unseen)):
        np.testing.assert_array_equal(first, second)
    assert [f["feature"] for f in model.importance()] == [f["feature"] for f in restored.importance()]


def test_probability_policy_costs_and_activity_hand_calculated():
    base = request(min_edge_bps=0)
    config = base.model_copy(
        update={
            "backtest": base.backtest.model_copy(
                update={
                    "strategy": "buy_hold",
                    "initial_cash": 1000,
                    "allocation": 0.5,
                    "cash_reserve": 0,
                    "commission_bps": 100,
                    "spread_bps": 0,
                    "slippage_bps": 0,
                    "max_participation": 1,
                }
            )
        }
    )
    frame = synthetic(config.backtest).iloc[:3].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = [10.0, 10.0, 12.0]
    frame["volume"] = 1000000
    rows = frame.iloc[:2]
    predicted, probability = np.array([-0.05, 0.05]), np.array([0.8, 0.2])
    joint = run_engine(frame, config.backtest, PredictionPolicy(rows, predicted, probability, config))
    assert joint["metrics"]["fills"] == 0
    direction = config.model_copy(update={"entry_rule": "probability_only"})
    simulated = run_engine(frame, config.backtest, PredictionPolicy(rows, predicted, probability, direction))
    # Buy at next open $10, paying 1%; sell at the following open $12, paying 1%.
    quantity = 500 / 10.1
    final_cash = 500 + quantity * 12 * 0.99
    assert simulated["metrics"]["final_cash"] == pytest.approx(final_cash)
    assert [fill["side"] for fill in simulated["trades"]] == ["buy", "sell"]
    baselines = {"buy_hold": run_engine(frame, config.backtest)}
    activity = activity_diagnostics(simulated, baselines, predicted, probability, direction)
    assert activity["entry_hurdle"] == pytest.approx(0.02)
    assert activity["return_passes"] == activity["probability_passes"] == activity["qualifying_signals"] == 1
    assert activity["time_in_market"] == pytest.approx(1 / 3)
    assert activity["closed_trades"] == 1
    assert activity["excess_vs_buy_hold"] == pytest.approx(-quantity * 12 * 0.01 / 1000)
    no_entries = activity_diagnostics(joint, baselines, predicted, probability, config)
    assert no_entries["qualifying_signals"] == 0
    assert no_entries["time_in_market"] == 0
    assert any("No prediction" in note for note in no_entries["notes"])


@pytest.mark.parametrize("mode", ["latest", "same_dates"])
def test_retrain_modes_preserve_every_model_and_broker_setting(mode, monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    config = request(
        feature_set="technical",
        entry_rule="probability_only",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        min_child_weight=1,
    )
    identifier = "a" * 24
    storage.atomic_json(
        storage.folder() / identifier / "result.json", {"config": config.model_dump(mode="json")}
    )
    captured = []

    def submit(candidate):
        captured.append(candidate)
        return {
            "id": "b" * 24,
            "status": "queued",
            "progress": "Queued",
            "created_at": datetime.now(UTC).isoformat(),
            "result_id": None,
            "error": None,
        }

    monkeypatch.setattr(jobs, "submit", submit)
    response = TestClient(app).post(f"/api/research/versions/{identifier}/retrain", json={"mode": mode})
    assert response.status_code == 202
    candidate = captured[0]
    assert candidate.model_dump(exclude={"parent_id", "refresh_data", "backtest"}) == config.model_dump(
        exclude={"parent_id", "refresh_data", "backtest"}
    )
    assert candidate.backtest.model_dump(exclude={"start", "end"}) == config.backtest.model_dump(
        exclude={"start", "end"}
    )
    assert candidate.backtest.start == config.backtest.start
    assert candidate.backtest.end == (datetime.now(UTC).date() if mode == "latest" else config.backtest.end)
    assert candidate.refresh_data == (mode == "latest")
    assert candidate.parent_id == identifier
    assert (
        TestClient(app)
        .post(f"/api/research/versions/{identifier}/retrain", json={"mode": "invalid"})
        .status_code
        == 422
    )


def test_legacy_report_annotations_preserve_scores_files_and_recover_retrain_lineage(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    config = request()
    parent = service.execute(config)
    child = copy.deepcopy(parent)
    child_id = "c" * 24
    child.update(id=child_id, search=None, retraining=None)
    child["config"]["parent_id"] = parent["id"]
    child["config"]["backtest"]["end"] = "2026-01-01"
    child["training"]["holdout_start"] = "2025-01-01T00:00:00+00:00"
    child["holdout"]["strategy"]["equity"][0]["timestamp"] = "2025-01-01T00:00:00+00:00"
    for name in ("validation", "holdout"):
        child[name].pop("activity")
    path = storage.folder() / child_id / "result.json"
    storage.atomic_json(path, child)
    before = path.read_bytes()
    response = TestClient(app).get(f"/api/research/versions/{child_id}")
    assert response.status_code == 200
    enriched = response.json()
    assert enriched["retraining"]["parameters_preserved"]
    assert not enriched["retraining"]["same_holdout_dates"]
    assert enriched["retraining"]["previous_end"] == "2025-01-01"
    assert enriched["retraining"]["current_end"] == "2026-01-01"
    for name in ("validation", "holdout"):
        assert enriched[name]["activity"] is not None
        assert enriched[name]["strategy"]["metrics"] == child[name]["strategy"]["metrics"]
    assert path.read_bytes() == before


def test_probability_policy_acceptance_uses_probability_baseline_not_unused_return_head():
    validation = {
        "strategy": {"metrics": {"net_return": 0.2, "sharpe": 1, "max_drawdown": -0.1, "turnover": 2}},
        "prediction": {"samples": 60, "mae": 0.03, "brier_score": 0.15},
        "baselines": {"cash": {"metrics": {"net_return": 0}}, "buy_hold": {"metrics": {"net_return": 0.1}}},
        "prediction_baselines": {"constant": {"mae": 0.01, "brier_score": 0.25}},
        "regimes": [{"sufficient": True, "net_return": 0.1}, {"sufficient": True, "net_return": 0.05}],
    }
    joint = service.acceptance(validation, "xgboost", None)
    probability = service.acceptance(validation, "xgboost", None, "probability_only")
    assert not joint["eligible"]
    assert probability["eligible"]
    assert probability["criteria"][-1]["value"] == pytest.approx(-0.1)
    validation["prediction"]["brier_score"] = 0.3
    assert not service.acceptance(validation, "xgboost", None, "probability_only")["eligible"]


def test_fine_entry_threshold_pairs_and_real_costs_hand_calculated():
    config = request(min_edge_bps=5, probability_threshold=0.55)
    config = config.model_copy(
        update={
            "backtest": config.backtest.model_copy(
                update={
                    "initial_cash": 1000,
                    "allocation": 0.5,
                    "cash_reserve": 0,
                    "commission_bps": 10,
                    "spread_bps": 0,
                    "slippage_bps": 0,
                    "fixed_fee": 0,
                    "max_participation": 1,
                }
            )
        }
    )
    frame = synthetic(config.backtest).iloc[:3].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = [10.0, 10.0, 12.0]
    frame["volume"] = 1000000
    rows = frame.iloc[:2]
    predicted, probability = np.array([0.0024, -0.01]), np.array([0.545, 0.2])

    def simulate(candidate):
        return run_engine(
            frame, candidate.backtest, PredictionPolicy(rows, predicted, probability, candidate)
        )

    # Round-trip commission is 20 bps. 5 extra bps requires >0.25%,
    # while 2.5 extra bps requires >0.225%. The forecast is 0.24%.
    assert simulate(config)["metrics"]["fills"] == 0
    assert simulate(config.model_copy(update={"min_edge_bps": 2.5}))["metrics"]["fills"] == 0
    assert simulate(config.model_copy(update={"probability_threshold": 0.54}))["metrics"]["fills"] == 0
    fine = config.model_copy(update={"min_edge_bps": 2.5, "probability_threshold": 0.54})
    traded = simulate(fine)
    # Both gates now pass. Buy $500 inclusive of 0.1% commission, then sell
    # at $12 paying the same 0.1%. Refinement must never erase broker costs.
    quantity = 500 / 10.01
    expected = 500 + quantity * 12 * 0.999
    assert traded["metrics"]["final_cash"] == pytest.approx(expected)
    assert traded["metrics"]["transaction_costs"] == pytest.approx(quantity * (10 + 12) * 0.001)
    assert traded["metrics"]["closed_trades"] == 1
