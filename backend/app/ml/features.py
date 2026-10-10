"""Causal, dimensionless features; labels describe next-open executable total returns."""

import itertools

import numpy as np
import pandas as pd

from app.data.actions import CorporateAction
from app.data.market import closing_times

FEATURE_VERSION = "daily-features-v2"
FEATURES = [
    "return_1",
    "return_5",
    "return_20",
    "volatility_20",
    "volatility_60",
    "ma_gap_10",
    "ma_gap_30",
    "range",
    "body",
    "opening_gap",
    "volume_ratio",
    "rsi_14",
]


TECHNICAL = ["return_21", "ma_gap_50", "bollinger_percent_b", "ppo_histogram", "atr_ratio_14"]


def feature_names(feature_set="basic"):
    return FEATURES + TECHNICAL if feature_set == "technical" else FEATURES.copy()


def action_arrays(frame):
    split = np.ones(len(frame))
    dividend = np.zeros(len(frame))
    dates = frame.timestamp.dt.tz_convert("America/New_York").dt.date
    for record in frame.attrs.get("corporate_actions", []):
        action = CorporateAction.model_validate(record)
        mask = np.asarray(dates == action.ex_date)
        if action.kind == "split":
            split[mask] *= action.ratio
        else:
            dividend[mask] += action.amount
    return split, dividend


def make_features(frame, req):
    split, _ = action_arrays(frame)
    # Forward adjustment uses only splits effective by this row. Ratios/windows are
    # invariant to price units, and future splits never rewrite earlier features.
    factor = pd.Series(np.cumprod(split), index=frame.index)
    close = frame.close * factor
    volume = frame.volume / factor
    returns = close.pct_change()
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = (up / (up + down)).fillna(0.5)
    result = pd.DataFrame(
        {
            "return_1": returns,
            "return_5": close.pct_change(5),
            "return_20": close.pct_change(20),
            "volatility_20": returns.rolling(20).std(),
            "volatility_60": returns.rolling(60).std(),
            "ma_gap_10": close / close.rolling(10).mean() - 1,
            "ma_gap_30": close / close.rolling(30).mean() - 1,
            "range": (frame.high - frame.low) / frame.close,
            "body": frame.close / frame.open - 1,
            "opening_gap": frame.open * factor / close.shift(1) - 1,
            "volume_ratio": volume / volume.rolling(20).mean().replace(0, np.nan) - 1,
            "rsi_14": rsi,
        }
    )
    mean20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    fast, slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    adjusted_high, adjusted_low = frame.high * factor, frame.low * factor
    true_range = pd.concat(
        [
            adjusted_high - adjusted_low,
            (adjusted_high - close.shift()).abs(),
            (adjusted_low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["return_21"] = close.pct_change(21)
    result["ma_gap_50"] = close / close.rolling(50).mean() - 1
    # Flat bands have no directional information; use the neutral midpoint.
    result["bollinger_percent_b"] = ((close - (mean20 - 2 * std20)) / (4 * std20).replace(0, np.nan)).fillna(
        0.5
    )
    result["ppo_histogram"] = (macd - macd.ewm(span=9, adjust=False).mean()) / slow
    result["atr_ratio_14"] = true_range.rolling(14).mean() / close
    result["bar_index"] = np.arange(len(frame))
    result["timestamp"] = frame.timestamp.to_numpy()
    result["signal_time"] = closing_times(frame, req.asset, req.timeframe)
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).reset_index(drop=True)


def make_samples(frame, req, horizon):
    features = make_features(frame, req)
    split, dividend = action_arrays(frame)
    labels, exits, entries = [], [], []
    for index in features.bar_index:
        entry, exit_index = int(index) + 1, int(index) + 1 + horizon
        if exit_index >= len(frame):
            labels.append(np.nan)
            exits.append(pd.NaT)
            entries.append(pd.NaT)
            continue
        units, income = 1.0, 0.0
        # Entry on ex-date has no prior overnight dividend entitlement.
        for j in range(entry + 1, exit_index + 1):
            units *= split[j]
            income += units * dividend[j]
        labels.append(
            (units * float(frame.iloc[exit_index].open) + income) / float(frame.iloc[entry].open) - 1
        )
        entries.append(frame.iloc[entry].timestamp)
        exits.append(frame.iloc[exit_index].timestamp)
    features["target"] = labels
    features["entry_time"] = pd.to_datetime(entries, utc=True)
    features["label_end"] = pd.to_datetime(exits, utc=True)
    return features.dropna(subset=["target"]).reset_index(drop=True)


def purge_before(samples, boundary):
    return samples.loc[(samples.signal_time < boundary) & (samples.label_end < boundary)].copy()


def chronological_plan(samples, fraction, folds):
    from app.data.errors import DataError

    n = len(samples)
    if n < 250:
        raise DataError(
            "AI research needs at least 250 labeled daily bars after the 60-bar warm-up. Choose roughly 2+ years of stocks or 1+ year of crypto."
        )
    test_start = int(n * (1 - fraction))
    validation_start = int(n * 0.5)
    holdout = samples.iloc[test_start:].copy()
    validation = samples.iloc[validation_start:test_start].copy()
    validation = validation.loc[validation.label_end < holdout.timestamp.iloc[0]]
    edges = np.linspace(0, len(validation), folds + 1, dtype=int)
    plan = []
    for start, end in itertools.pairwise(edges):
        evaluation = validation.iloc[start:end].copy()
        training = purge_before(samples, evaluation.timestamp.iloc[0])
        if len(training) < 100 or len(evaluation) < 20:
            raise DataError("Not enough samples per fold; increase the date range or reduce folds.")
        plan.append((training, evaluation))
    training = purge_before(samples, holdout.timestamp.iloc[0])
    return plan, training, holdout
