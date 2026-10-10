import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime

import numpy as np

from app.data.errors import DataError
from app.data.market import load
from app.experiments.service import CODE_VERSION as ENGINE_VERSION
from app.ml import storage
from app.ml.diagnostics import distribution_reference, drift
from app.ml.evaluation import evaluate
from app.ml.features import FEATURE_VERSION, FEATURES, chronological_plan, make_features, make_samples
from app.ml.models import BoostModel
from app.ml.rl import QModel

CODE_VERSION = "research-v4-" + ENGINE_VERSION


def load_model(identifier):
    path = storage.folder() / storage.safe_id(identifier)
    metadata = json.loads((path / "model.json").read_text())
    return BoostModel.load(path) if metadata["algorithm"] == "xgboost" else QModel.load(path)


class WalkForwardModel:
    def __init__(self, blocks, algorithm):
        self.blocks, self.algorithm = blocks, algorithm

    def predict(self, rows):
        if self.algorithm == "q_learning":
            return None, None
        forecasts = {}
        for model, evaluation in self.blocks:
            predicted, probability = model.predict(evaluation)
            forecasts.update(
                {
                    int(index): (float(p), float(q))
                    for index, p, q in zip(evaluation.bar_index, predicted, probability)
                }
            )
        values = [forecasts[int(index)] for index in rows.bar_index]
        return np.array([v[0] for v in values]), np.array([v[1] for v in values])

    def policy(self, rows):
        children, route = [], {}
        for model, evaluation in self.blocks:
            child = model.policy(evaluation)
            children.append(child)
            route.update({row.timestamp.isoformat(): child for _, row in evaluation.iterrows()})

        class Router:
            def __init__(self):
                self.last = 0.0
                self.decisions = {}

            def observe(self, point, terminal):
                for child in children:
                    child.observe(point, terminal)

            def target(self, history):
                timestamp = history.iloc[-1].timestamp.isoformat()
                if timestamp in route:
                    self.last = route[timestamp].target(history)
                    self.decisions[timestamp] = self.last
                return self.last

        return Router()


def acceptance(validation, algorithm, active):
    stats = validation["strategy"]["metrics"]
    predictive = validation["prediction"]
    baseline = max(s["metrics"]["net_return"] for s in validation["baselines"].values())
    reliable_regimes = [r for r in validation["regimes"] if r["sufficient"]]
    criteria = [
        {
            "name": "At least 40 validation predictions",
            "value": predictive["samples"],
            "passed": predictive["samples"] >= 40,
        },
        {
            "name": "Net return exceeds cash, buy-and-hold and MA",
            "value": stats["net_return"] - baseline,
            "passed": stats["net_return"] > baseline,
        },
        {
            "name": "Positive validation Sharpe",
            "value": stats["sharpe"],
            "passed": stats["sharpe"] is not None and stats["sharpe"] > 0,
        },
        {
            "name": "Maximum drawdown no worse than 20%",
            "value": stats["max_drawdown"],
            "passed": stats["max_drawdown"] >= -0.2,
        },
        {
            "name": "Turnover no greater than 20x initial capital",
            "value": stats["turnover"],
            "passed": stats["turnover"] <= 20,
        },
        {
            "name": "At least two regimes with 20 intervals",
            "value": len(reliable_regimes),
            "passed": len(reliable_regimes) >= 2,
        },
        {
            "name": "No supported regime loses more than 10%",
            "value": min((r["net_return"] for r in reliable_regimes), default=None),
            "passed": bool(reliable_regimes) and all(r["net_return"] >= -0.1 for r in reliable_regimes),
        },
    ]
    if algorithm == "xgboost":
        error = min(p["mae"] for p in validation["prediction_baselines"].values())
        criteria.append(
            {
                "name": "MAE beats constant prediction baselines",
                "value": predictive["mae"] - error,
                "passed": predictive["mae"] < error,
            }
        )
    return {
        "eligible": all(c["passed"] for c in criteria),
        "criteria": criteria,
        "active_model_at_training": active,
        "incumbent_comparison": None,
        "policy": "Fixed v1 criteria use walk-forward validation only. Holdout results are reporting only; repeated user research consumes that holdout.",
    }


def execute(config, progress=lambda message: None):
    import pandas as pd

    progress("Downloading and validating daily bars")
    frame, provenance = load(config.backtest, refresh=config.refresh_data)
    identity = {
        "config": config.model_dump(mode="json"),
        "provenance": provenance,
        "code_version": CODE_VERSION,
        "feature_version": FEATURE_VERSION,
        "libraries": {p: importlib.metadata.version(p) for p in ("xgboost-cpu", "scikit-learn", "gymnasium")},
    }
    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = storage.folder() / identifier
    if (path / "result.json").exists():
        progress("Loaded the identical saved model version")
        return storage.result(identifier)
    features = make_features(frame, config.backtest)
    samples = make_samples(frame, config.backtest, config.horizon)
    plan, training, holdout = chronological_plan(samples, config.holdout_fraction, config.folds)
    active = storage.registry()["active"].get(storage.key(config))
    if config.parent_id is not None:
        parent = storage.result(config.parent_id)
        from app.ml.schemas import ResearchRequest

        if storage.key(ResearchRequest.model_validate(parent["config"])) != storage.key(config):
            raise DataError(
                "Parent model must use the same asset, provider, algorithm, horizon and execution settings."
            )
    blocks, walk_forward = [], []
    for index, (train, evaluation) in enumerate(plan):
        progress(f"Training walk-forward fold {index + 1}/{len(plan)}")
        model = (
            BoostModel.fit(train, config)
            if config.algorithm == "xgboost"
            else QModel.fit(frame, features, train, config)
        )
        report = evaluate(frame, features, evaluation, model, config, train)
        blocks.append((model, evaluation))
        walk_forward.append(
            {
                "fold": index + 1,
                "training_rows": len(train),
                "training_label_end": train.label_end.max().isoformat(),
                "evaluation_start": evaluation.signal_time.iloc[0].isoformat(),
                "evaluation_end": evaluation.label_end.iloc[-1].isoformat(),
                "training": model.metadata,
                "prediction": report["prediction"],
                "strategy": report["strategy"]["metrics"],
                "baselines": {k: v["metrics"] for k, v in report["baselines"].items()},
            }
        )
    progress("Evaluating combined out-of-sample validation")
    validation_rows = pd.concat([b[1] for b in blocks], ignore_index=True)
    oof = WalkForwardModel(blocks, config.algorithm)
    validation = evaluate(frame, features, validation_rows, oof, config, plan[0][0])
    progress("Fitting final model without held-out labels")
    final = (
        BoostModel.fit(training, config)
        if config.algorithm == "xgboost"
        else QModel.fit(frame, features, training, config)
    )
    progress("Evaluating frozen model on final held-out period")
    test = evaluate(frame, features, holdout, final, config, training)
    promotion = acceptance(validation, config.algorithm, active)
    comparison_id = active or config.parent_id
    if comparison_id:
        incumbent = storage.result(comparison_id)
        cutoff = pd.Timestamp(incumbent["training"]["trained_through"])
        forward = validation_rows.loc[validation_rows.signal_time > cutoff]
        comparison = {"samples": len(forward), "incumbent": comparison_id, "sufficient": len(forward) >= 40}
        passed = False
        if len(forward) >= 40 and config.parent_id == comparison_id:
            challenger_report = evaluate(frame, features, forward, oof, config, plan[0][0])
            incumbent_report = evaluate(
                frame,
                features,
                forward,
                load_model(comparison_id),
                ResearchRequest.model_validate(incumbent["config"]),
                plan[0][0],
            )
            delta = (
                challenger_report["strategy"]["metrics"]["net_return"]
                - incumbent_report["strategy"]["metrics"]["net_return"]
            )
            comparison.update(
                candidate=challenger_report["strategy"]["metrics"],
                incumbent_metrics=incumbent_report["strategy"]["metrics"],
                return_improvement=delta,
            )
            passed = (
                delta > 0
                and challenger_report["strategy"]["metrics"]["max_drawdown"]
                >= incumbent_report["strategy"]["metrics"]["max_drawdown"]
            )
        promotion["incumbent_comparison"] = comparison
        promotion["criteria"].append(
            {
                "name": "Beats reference model on 40 new forward validation samples with no worse drawdown",
                "value": comparison.get("return_improvement"),
                "passed": passed,
            }
        )
        promotion["eligible"] = all(c["passed"] for c in promotion["criteria"])
    reference = distribution_reference(training)
    latest = features.tail(1)
    predicted, probability = final.predict(latest)
    latest_prediction = {
        "signal_time": latest.signal_time.iloc[0].isoformat(),
        "predicted_return": None if predicted is None else float(predicted[0]),
        "probability_up": None if probability is None else float(probability[0]),
        "context": "Forecast from the final frozen model using the latest completed bar. This is a research observation.",
    }
    warnings = [
        "SYNTHETIC demonstration data: learned results do not establish real-market performance."
        if provenance["synthetic"]
        else "Historical data and action records can be revised. Features use ex-date split economics, never future announcements.",
        "Single-asset daily research with limited samples; repeated experiments can overfit the reported holdout.",
        "Regime statistics attribute intervals by the prior completed bar. Compounded regime returns are diagnostic subsets, not independently tradable portfolios.",
        *final.metadata.get("warnings", []),
    ]
    if config.algorithm == "q_learning":
        latest_prediction["context"] = (
            "Q-learning outputs cash/long actions from portfolio state; it does not estimate return probabilities."
        )
    result = dict(
        id=identifier,
        created_at=datetime.now(UTC).isoformat(),
        **identity,
        features=FEATURES,
        training={
            **final.metadata,
            "rows": len(training),
            "holdout_start": holdout.signal_time.iloc[0].isoformat(),
            "distribution_reference": reference,
        },
        walk_forward=walk_forward,
        validation=validation,
        holdout=test,
        importance=final.importance(),
        promotion=promotion,
        latest_prediction=latest_prediction,
        drift=drift(reference, holdout, test["predictions"]),
        warnings=warnings,
        assumptions=[
            "Features use completed daily bars, a 60-bar warm-up, and dimensionless causal split adjustment.",
            f"Return labels run from the next open to the open {config.horizon} bar(s) later, including split/dividend entitlement economics.",
            "Expanding training folds purge every label whose endpoint reaches the next prediction boundary; all hyperparameters and hurdles are fixed before evaluation.",
            "XGBoost uses separate chronological core, early-stopping, and calibration blocks; fitted final-model components see no held-out labels.",
            "Cash, buy-and-hold, MA and AI use identical evaluation windows, fresh initial capital, broker costs and constraints. MA uses 10/30 warmed-up causal features.",
            "Probability diagnostics include Brier, log loss, reliability bins and expected calibration error. Rank correlation is temporal Spearman within one asset, not cross-sectional ranking.",
            "Regime thresholds come from training volatility only; small regime groups are flagged.",
            "RL trains repeated historical episodes on training bars only, then freezes its table for unseen validation/test periods. Costs enter reward through actual net equity.",
            "Manual retraining refreshes provider data and creates a separate candidate. Promotion is explicit and gated; previous versions remain available.",
        ],
    )
    progress("Saving model, provenance and evaluation")
    final.save(path)
    frame.to_parquet(path / "dataset.parquet", index=False)
    storage.atomic_json(path / "dataset-actions.json", frame.attrs.get("corporate_actions", []))
    storage.atomic_json(path / "result.json", result)
    return result
