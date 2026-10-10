import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from app.ml.features import FEATURES


def prediction_metrics(actual, predicted=None, probability=None):
    y = np.asarray(actual, dtype=float)
    result = {
        "samples": len(y),
        "mae": None,
        "rmse": None,
        "directional_accuracy": None,
        "rank_correlation": None,
        "brier_score": None,
        "log_loss": None,
        "calibration_error": None,
        "reliability": [],
    }
    if predicted is not None and len(y):
        p = np.asarray(predicted, dtype=float)
        result.update(
            mae=float(np.mean(np.abs(y - p))),
            rmse=float(np.sqrt(np.mean((y - p) ** 2))),
            directional_accuracy=float(np.mean(np.sign(y) == np.sign(p))),
        )
        if len(y) >= 20 and np.std(y) > 1e-12 and np.std(p) > 1e-12:
            result["rank_correlation"] = float(pd.Series(y).rank().corr(pd.Series(p).rank()))
    if probability is not None and len(y):
        p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        binary = (y > 0).astype(int)
        result["brier_score"] = float(brier_score_loss(binary, p))
        result["log_loss"] = float(log_loss(binary, p, labels=[0, 1]))
        error = 0.0
        bins = np.minimum((p * 10).astype(int), 9)
        for index in range(10):
            mask = bins == index
            if not mask.any():
                continue
            mean, frequency = float(p[mask].mean()), float(binary[mask].mean())
            error += mask.mean() * abs(mean - frequency)
            result["reliability"].append(
                {"bin": index, "count": int(mask.sum()), "probability": mean, "observed_frequency": frequency}
            )
        result["calibration_error"] = float(error)
    return result


def distribution_reference(training, names=None):
    result = {}
    for name in FEATURES if names is None else names:
        edges = np.unique(np.quantile(training[name], np.arange(0.1, 1, 0.1)))
        counts = np.bincount(np.digitize(training[name], edges), minlength=len(edges) + 1)
        result[name] = {"edges": edges.tolist(), "frequency": (counts / counts.sum()).tolist()}
    return result


def drift(reference, recent, prediction_rows):
    results = []
    for name, bins in reference.items():
        counts = np.bincount(np.digitize(recent[name], bins["edges"]), minlength=len(bins["edges"]) + 1)
        current = np.clip(counts / counts.sum(), 1e-6, 1)
        old = np.clip(bins["frequency"], 1e-6, 1)
        psi = float(np.sum((current - old) * np.log(current / old)))
        results.append({"feature": name, "psi": psi, "flagged": psi > 0.25})
    ratio = None
    eligible = [p for p in prediction_rows if p.get("predicted_return") is not None]
    if len(eligible) >= 40:
        midpoint = len(eligible) // 2
        errors = np.array([abs(p["actual_return"] - p["predicted_return"]) for p in eligible])
        first = float(errors[:midpoint].mean())
        ratio = float(errors[midpoint:].mean() / first) if first > 1e-12 else None
    return {
        "features": sorted(results, key=lambda r: -r["psi"]),
        "recent_mae_ratio": ratio,
        "deterioration": ratio is not None and ratio > 1.5,
        "sample_count": len(recent),
        "note": "PSI > 0.25 and recent/earlier MAE > 1.5 are descriptive alerts, not proof of model failure or automatic retraining triggers.",
    }
