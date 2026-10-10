"""Rolling windows with purged labels and shared search calendar boundaries."""

import itertools

import numpy as np
import pandas as pd

from app.core.schemas import ASSETS
from app.data.errors import DataError
from app.ml import storage
from app.ml.features import chronological_plan, make_samples, purge_before


def training_before(samples, boundary, config):
    training = purge_before(samples, boundary)
    if config.training_window_months:
        start = boundary - pd.DateOffset(months=config.training_window_months)
        training = training.loc[training.signal_time >= start].copy()
    minimum = 30 if config.training_window_months else 100
    if len(training) < minimum:
        raise DataError(
            f"Only {len(training)} known training labels before {boundary.date()}; need {minimum}. Extend the history or training window."
        )
    return training


def plan(samples, config, anchor=None):
    # Legacy configurations retain their previous calendar splits.
    if anchor is None and not config.training_window_months and config.validation_scheme == "blocks":
        return chronological_plan(samples, config.holdout_fraction, config.folds)
    reference = samples if anchor is None else anchor
    if len(reference) < 250:
        raise DataError(
            "Need at least 250 labeled daily bars after feature warm-up; use about 2+ years of stocks or 1+ year of crypto."
        )
    test_start = int(len(reference) * (1 - config.holdout_fraction))
    reserved = reference.iloc[test_start:].copy()
    boundary = reserved.timestamp.iloc[0]
    validation = reference.iloc[int(len(reference) * 0.5) : test_start].copy()
    validation = validation.loc[validation.label_end < boundary]
    timezone = "America/New_York" if ASSETS[config.backtest.asset] == "stock" else "UTC"
    if config.validation_scheme == "monthly":
        local = validation.timestamp.dt.tz_convert(timezone)
        months = local.dt.strftime("%Y-%m")
        # Start at the first full validation month, not halfway through one.
        if local.iloc[0].day > 3:
            validation = validation.loc[months != months.iloc[0]].copy()
        groups = [
            part
            for _, part in validation.groupby(
                validation.timestamp.dt.tz_convert(timezone).dt.strftime("%Y-%m"), sort=True
            )
        ]
    else:
        edges = np.linspace(0, len(validation), config.folds + 1, dtype=int)
        groups = [validation.iloc[a:b].copy() for a, b in itertools.pairwise(edges)]
    blocks = []
    for reference_rows in groups:
        if len(reference_rows) < (1 if config.validation_scheme == "monthly" else 20):
            raise DataError(
                "Not enough predictions in an evaluation block; extend the date range or reduce folds."
            )
        evaluation = samples.loc[samples.bar_index.isin(reference_rows.bar_index)].copy()
        if len(evaluation) != len(reference_rows):
            raise DataError("A candidate cannot cover the shared search window.")
        evaluation.attrs = {"simulation_end": reference_rows.label_end.iloc[-1]}
        blocks.append((training_before(samples, evaluation.timestamp.iloc[0], config), evaluation))
    holdout = samples.loc[samples.bar_index.isin(reserved.bar_index)].copy()
    holdout.attrs = {"simulation_end": reserved.label_end.iloc[-1]}
    return blocks, training_before(samples, boundary, config), holdout


def for_candidate(prepared, config):
    samples = make_samples(prepared["frame"], config.backtest, config.horizon)
    blocks, training, holdout = plan(samples, config, prepared["reference_samples"])
    return {
        **prepared,
        "plan": blocks,
        "training": training,
        "holdout": holdout,
        "active": prepared["active_by_family"].get(storage.key(config)),
    }
