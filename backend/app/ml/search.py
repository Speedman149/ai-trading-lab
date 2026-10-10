"""Reproducible parameter selection using pre-test validation only."""

import logging
import math
import random
from datetime import UTC, datetime

from app.data.errors import DataError
from app.ml import service, storage
from app.ml.features import make_samples
from app.ml.grid import anchor_horizon, grid_candidates, grid_space, updated, value
from app.ml.planning import for_candidate
from app.ml.schemas import ResearchRequest

SPACE = {
    "n_estimators": [60, 120, 240, 400],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "min_edge_bps": [0, 5, 10, 20],
    "probability_threshold": [0.5, 0.52, 0.55, 0.6],
    "feature_set": ["basic", "technical"],
    "entry_rule": ["return_and_probability", "probability_only"],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1, 5, 10],
}
# Step, lower bound, upper bound. Local trials keep feature set and entry rule fixed.
LOCAL = {
    "probability_threshold": (0.01, 0.5, 0.9),
    "min_edge_bps": (2.5, 0, 200),
    "learning_rate": (0.01, 0.01, 0.3),
    "max_depth": (1, 1, 6),
    "min_child_weight": (1, 1, 20),
    "reg_lambda": (1, 0, 50),
    "subsample": (0.05, 0.5, 1),
    "colsample_bytree": (0.05, 0.5, 1),
    "n_estimators": (40, 20, 600),
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
    # Sample directly instead of materializing the growing Cartesian product.
    rng = random.Random(base.seed)
    parameters, seen = [current], {tuple(current.values())}
    while len(parameters) < request.trials:
        sample = {key: rng.choice(values) for key, values in SPACE.items()}
        if sample["entry_rule"] == "probability_only":
            sample["min_edge_bps"] = 0  # The return hurdle does not enter this policy.
        signature = tuple(sample.values())
        if signature not in seen:
            parameters.append(sample)
            seen.add(signature)
    return [ResearchRequest.model_validate({**base.model_dump(), **params}) for params in parameters]


def signature(config):
    return tuple(
        0 if key == "min_edge_bps" and config.entry_rule == "probability_only" else getattr(config, key)
        for key in SPACE
    )


def neighbors(center):
    # Start with single-setting changes, then paired entry gates at the same radius.
    # Increase the radius only after testing the nearest changes in every setting.
    for radius in range(1, 51):
        for key, (step, lower, upper) in LOCAL.items():
            if key == "min_edge_bps" and center.entry_rule == "probability_only":
                continue
            for direction in (-1, 1):
                value = round(min(upper, max(lower, getattr(center, key) + direction * radius * step)), 8)
                if key in ("max_depth", "n_estimators"):
                    value = int(value)
                if value != getattr(center, key):
                    yield ResearchRequest.model_validate({**center.model_dump(), key: value})
        # Joint gates interact: changing either alone can still leave every entry blocked.
        if center.entry_rule == "return_and_probability":
            for probability_direction in (-1, 1):
                for edge_direction in (-1, 1):
                    yield ResearchRequest.model_validate(
                        {
                            **center.model_dump(),
                            "probability_threshold": round(
                                min(
                                    0.9,
                                    max(
                                        0.5,
                                        center.probability_threshold + probability_direction * radius * 0.01,
                                    ),
                                ),
                                8,
                            ),
                            "min_edge_bps": round(
                                min(200, max(0, center.min_edge_bps + edge_direction * radius * 2.5)), 8
                            ),
                        }
                    )


def candidate_plan(request, recorded_trials):
    """Adapt only to completed validation scores; never read a winner's test report."""
    if request.mode == "grid":
        for config in grid_candidates(request):
            yield config, "grid", None
        return
    if request.mode == "random":
        for index, config in enumerate(candidates(request)):
            yield config, "current" if index == 0 else "broad", None
        return
    broad_count = 1 if request.mode == "local" else max(1, request.trials // 2)
    seen = set()
    for index, config in enumerate(candidates(request.model_copy(update={"trials": broad_count})), 1):
        seen.add(signature(config))
        yield config, "current" if index == 1 else "broad", None
    completed = [
        trial for trial in rank_trials(recorded_trials(), request.objective) if trial["status"] == "completed"
    ]
    # Freeze the two broad leaders before refinement. Local mode starts at the submitted settings.
    centers = (
        [
            (
                ResearchRequest.model_validate({**request.research.model_dump(), **trial["parameters"]}),
                trial["trial"],
            )
            for trial in completed[:2]
        ]
        if request.mode == "coarse_to_fine"
        else []
    )
    if not centers:
        centers = [(request.research, 1)]
    queues = [(iter(neighbors(config)), trial) for config, trial in centers]
    remaining = request.trials - broad_count
    while remaining:
        advanced = False
        for queue, origin in queues:
            for config in queue:
                key = signature(config)
                if key not in seen:
                    seen.add(key)
                    remaining -= 1
                    advanced = True
                    yield config, "refine", origin
                    break
            if remaining == 0:
                return
        if not advanced:
            raise ValueError("No unique local candidates remain; use a smaller trial budget.")


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
    space = grid_space(request) if request.mode == "grid" else ({} if request.mode == "local" else SPACE)
    total = math.prod(map(len, space.values())) if request.mode == "grid" else request.trials
    reference_config = request.research
    if request.mode == "grid":
        reference_config = updated(
            reference_config,
            {
                "horizon": anchor_horizon(request, space),
                "evaluation_anchor_horizon": anchor_horizon(request, space),
                "folds": min(space.get("folds", [reference_config.folds])),
            },
        )
    prepared = service.prepare(reference_config, progress)
    if request.mode == "grid":
        prepared["reference_samples"] = make_samples(
            prepared["frame"], reference_config.backtest, reference_config.horizon
        )
    result = {
        "id": storage.safe_id(identifier),
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "config": request.model_dump(mode="json"),
        "space": space,
        "total_combinations": total,
        "selection_policy": POLICY,
        "refinement": None
        if request.mode in ("random", "grid")
        else {
            "steps": {
                key: {"step": step, "minimum": low, "maximum": high}
                for key, (step, low, high) in LOCAL.items()
            },
            "broad_trials": 1 if request.mode == "local" else max(1, request.trials // 2),
            "note": "Numeric changes and joint probability/hurdle pairs around the submitted configuration (local) or the two highest-ranked broad validation candidates (broad then refine). Nearest neighbors are tried first; feature set and entry rule stay fixed within each neighborhood. Final-test scores never choose refinement centers.",
        },
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
            "Every combination in the declared finite ranges is attempted; best within this grid is not a proven market optimum. More choices can overfit validation."
            if request.mode == "grid"
            else "Best found in this sampled search, not a proven optimum. Searching more choices can overfit validation.",
            "Only the selected winner is tested on the held-out period. Repeated searches or previously viewed reports consume its independence.",
        ],
    }
    target = storage.folder() / "searches" / f"{identifier}.json"
    storage.atomic_json(target, result)
    best = None
    try:
        for index, (config, phase, origin) in enumerate(candidate_plan(request, lambda: result["trials"]), 1):
            parameter_keys = list(SPACE)
            if request.mode == "grid":
                parameter_keys += [
                    key
                    for key in (
                        "horizon",
                        "folds",
                        "seed",
                        "stop_loss_pct",
                        "training_window_months",
                        "recency_half_life_days",
                    )
                    if key not in parameter_keys
                ]
            trial = {
                "trial": index,
                "phase": phase,
                "origin_trial": origin,
                "rank": None,
                "parameters": {key: value(config, key) for key in parameter_keys},
                "status": "completed",
                "error": None,
                "metrics": None,
                "prediction": None,
                "baselines": {},
                "folds": [],
                "promotion": None,
                "risk_passed": False,
                "activity": None,
            }
            try:
                trial_prepared = for_candidate(prepared, config) if request.mode == "grid" else prepared
                validated = service.validate(
                    config,
                    trial_prepared,
                    lambda message, i=index, stage=phase: progress(f"Trial {i}/{total} ({stage}): {message}"),
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
                    activity=validation["activity"],
                    baselines={key: value["metrics"] for key, value in validation["baselines"].items()},
                    folds=[{"fold": fold["fold"], **fold["strategy"]} for fold in validated["walk_forward"]],
                    promotion=validated["promotion"],
                    risk_passed=metrics["max_drawdown"] >= -0.2 and metrics["turnover"] <= 20,
                )
                if best is None or ranking_key(trial, request.objective) > ranking_key(
                    best[0], request.objective
                ):
                    best = (trial, config, validated, trial_prepared)
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
            best[3],
            best[2],
            progress,
            selection={
                "id": identifier,
                "selected_trial": best[0]["trial"],
                "trials": total,
                "objective": request.objective,
                "mode": request.mode,
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
