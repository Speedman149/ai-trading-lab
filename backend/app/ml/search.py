"""Reproducible parameter selection using pre-test validation only."""

import itertools
import logging
import math
import random
from datetime import UTC, datetime

from app.data.errors import DataError
from app.ml import service, storage
from app.ml.schemas import ResearchRequest

SPACE = {
    "n_estimators": [60, 120, 240, 400],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "min_edge_bps": [0, 5, 10, 20],
    "probability_threshold": [0.5, 0.55, 0.6, 0.65],
}
POLICY = (
    "Rank candidates passing all promotion checks first, then candidates within "
    "20% maximum drawdown and 20x turnover limits, then the selected validation metric. "
    "Ties use the other trading metric, smaller drawdown, lower turnover, lower MAE, "
    "and earlier trial. Undefined Sharpe ranks below every defined Sharpe. "
    "Final held-out results never enter ranking; only the selected winner is tested."
)


def candidates(request):
    base = request.research
    current = {key: getattr(base, key) for key in SPACE}
    options = [
        dict(zip(SPACE, values))
        for values in itertools.product(*SPACE.values())
        if dict(zip(SPACE, values)) != current
    ]
    sampled = random.Random(base.seed).sample(options, request.trials - 1)
    return [ResearchRequest.model_validate({**base.model_dump(), **params}) for params in [current, *sampled]]


def ranking_key(trial, objective):
    stats, prediction = trial["metrics"], trial["prediction"]
    sharpe = stats["sharpe"] if stats["sharpe"] is not None else -math.inf
    first, second = (sharpe, stats["net_return"]) if objective == "sharpe" else (stats["net_return"], sharpe)
    return (
        trial["promotion"]["eligible"],
        trial["risk_passed"],
        first,
        second,
        stats["max_drawdown"],
        -stats["turnover"],
        -prediction["mae"] if prediction["mae"] is not None else -math.inf,
        -trial["trial"],
    )


def rank_trials(trials, objective):
    completed = sorted(
        (trial for trial in trials if trial["status"] == "completed"),
        key=lambda trial: ranking_key(trial, objective),
        reverse=True,
    )
    failed = [trial for trial in trials if trial["status"] == "failed"]
    return [{**trial, "rank": index + 1} for index, trial in enumerate(completed)] + [
        {**trial, "rank": None} for trial in failed
    ]


def execute(request, identifier, progress=lambda message: None):
    prepared = service.prepare(request.research, progress)
    result = {
        "id": storage.safe_id(identifier),
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "config": request.model_dump(mode="json"),
        "space": SPACE,
        "selection_policy": POLICY,
        "code_version": service.CODE_VERSION,
        "provenance": prepared["provenance"],
        "validation_start": prepared["plan"][0][1].signal_time.iloc[0].isoformat(),
        "validation_end": prepared["plan"][-1][1].label_end.iloc[-1].isoformat(),
        "holdout_start": prepared["holdout"].signal_time.iloc[0].isoformat(),
        "winner_id": None,
        "selected_trial": None,
        "trials": [],
        "error": None,
        "warnings": [
            "Best found in this sampled search, not a proven optimum. Searching more choices can overfit validation.",
            "Only the selected winner is tested on the held-out period. Repeated searches or previously viewed reports consume its independence.",
        ],
    }
    target = storage.folder() / "searches" / f"{identifier}.json"
    storage.atomic_json(target, result)
    best = None
    try:
        for index, config in enumerate(candidates(request), 1):
            trial = {
                "trial": index,
                "rank": None,
                "parameters": {key: getattr(config, key) for key in SPACE},
                "status": "completed",
                "error": None,
                "metrics": None,
                "prediction": None,
                "baselines": {},
                "folds": [],
                "promotion": None,
                "risk_passed": False,
            }
            try:
                validated = service.validate(
                    config,
                    prepared,
                    lambda message, i=index: progress(f"Trial {i}/{request.trials}: {message}"),
                )
                validation = validated["validation"]
                metrics = validation["strategy"]["metrics"]
                trial.update(
                    metrics={
                        **metrics,
                        "excess_return": metrics["net_return"]
                        - max(value["metrics"]["net_return"] for value in validation["baselines"].values()),
                    },
                    prediction=validation["prediction"],
                    baselines={key: value["metrics"] for key, value in validation["baselines"].items()},
                    folds=[{"fold": fold["fold"], **fold["strategy"]} for fold in validated["walk_forward"]],
                    promotion=validated["promotion"],
                    risk_passed=metrics["max_drawdown"] >= -0.2 and metrics["turnover"] <= 20,
                )
                if best is None or ranking_key(trial, request.objective) > ranking_key(
                    best[0], request.objective
                ):
                    best = (trial, config, validated)
            except Exception as exc:
                logging.getLogger(__name__).exception("Parameter-search trial failed")
                trial.update(
                    status="failed",
                    error=str(exc)
                    if isinstance(exc, (DataError, ValueError))
                    else "Candidate training failed; see backend logs.",
                )
            result["trials"].append(trial)
            result["trials"] = rank_trials(result["trials"], request.objective)
            storage.atomic_json(target, result)
        if best is None:
            raise DataError("Every parameter trial failed. Review the search's recorded trial errors.")
        result["selected_trial"] = best[0]["trial"]
        # Persist selection before any final-test evaluation.
        storage.atomic_json(target, result)
        progress(f"Selected trial {best[0]['trial']}; fitting and testing only the winner")
        winner = service.complete(
            best[1],
            prepared,
            best[2],
            progress,
            selection={
                "id": identifier,
                "selected_trial": best[0]["trial"],
                "trials": request.trials,
                "objective": request.objective,
                "selection_policy": POLICY,
            },
        )
        result.update(status="completed", winner_id=winner["id"])
        storage.atomic_json(target, result)
        return result
    except Exception:
        result.update(status="failed", error="Search did not finish; review the job and trial errors.")
        storage.atomic_json(target, result)
        raise
