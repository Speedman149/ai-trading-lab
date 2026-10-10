"""Finite, decimal-exact Cartesian grids. Reject oversize jobs before training."""

import itertools
import math
from decimal import Decimal

from app.ml.schemas import ResearchRequest

LIMIT = 256
# All numeric model controls plus the protective stop. Data/test boundaries and
# categorical feature/entry choices stay fixed at the submitted configuration.
BOUNDS = {
    "horizon": (1, 5, True),
    "max_depth": (1, 6, True),
    "n_estimators": (20, 600, True),
    "learning_rate": (0.01, 0.3, False),
    "probability_threshold": (0.5, 0.9, False),
    "min_edge_bps": (0, 200, False),
    "folds": (2, 5, True),
    "seed": (0, 2147483647, True),
    "stop_loss_pct": (0, 50, False),
    "training_window_months": (0, 60, True),
    "recency_half_life_days": (0, 730, False),
    "subsample": (0.5, 1, False),
    "colsample_bytree": (0.5, 1, False),
    "min_child_weight": (1, 20, False),
    "reg_lambda": (0, 50, False),
}


def value(config, key):
    return config.backtest.stop_loss_pct if key == "stop_loss_pct" else getattr(config, key)


def updated(base, parameters):
    data = base.model_dump()
    for key, setting in parameters.items():
        if key == "stop_loss_pct":
            data["backtest"][key] = setting
        else:
            data[key] = setting
    return ResearchRequest.model_validate(data)


def grid_space(request):
    if not request.grid:
        raise ValueError("Specify at least one parameter range for grid search.")
    result, counts = {}, {}
    for key, interval in request.grid.items():
        if key not in BOUNDS:
            raise ValueError(f"Unsupported grid parameter: {key}")
        lower, upper, integer = BOUNDS[key]
        lo, hi, step = map(Decimal, map(str, (interval.minimum, interval.maximum, interval.step)))
        if lo > hi or lo < Decimal(str(lower)) or hi > Decimal(str(upper)):
            raise ValueError(f"{key}: minimum/maximum must be ordered within {lower}–{upper}.")
        if integer and any(v != v.to_integral_value() for v in (lo, hi, step)):
            raise ValueError(f"{key} uses whole numbers; its range and step must be integers.")
        count = int((hi - lo) // step) + 1
        counts[key] = count
        if count > LIMIT:
            raise ValueError(f"{key} has {count} values; the complete grid limit is {LIMIT} combinations.")
        choices = [int(lo + i * step) if integer else float(lo + i * step) for i in range(count)]
        if key == "training_window_months" and any(v in (1, 2) for v in choices):
            raise ValueError("Training windows must be 0 or at least 3 months.")
        if key == "folds" and request.research.validation_scheme == "monthly" and count > 1:
            raise ValueError(
                "Monthly validation determines its folds from calendar months. Switch to fixed blocks to search fold counts."
            )
        if key == "min_edge_bps" and request.research.entry_rule == "probability_only" and count > 1:
            raise ValueError(
                "Probability-only entry ignores the return hurdle. Use return + probability to search it."
            )
        result[key] = choices
    total = math.prod(counts.values())
    if total > LIMIT:
        raise ValueError(
            f"Your grid contains {total:,} combinations; limit {LIMIT}. Narrow ranges or increase steps. No combinations have been sampled or trained."
        )
    return result


def anchor_horizon(request, space):
    return max(
        request.research.evaluation_anchor_horizon or 0, max(space.get("horizon", [request.research.horizon]))
    )


def grid_candidates(request):
    space = grid_space(request)
    base = request.research.model_copy(update={"evaluation_anchor_horizon": anchor_horizon(request, space)})
    for choices in itertools.product(*space.values()):
        yield updated(base, dict(zip(space, choices)))
