"""CPU-only XGBoost; chronological early stopping and disjoint sigmoid calibration."""

import json

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from app.ml.features import FEATURES, feature_names, purge_before


def recency_weights(rows, config, reference=None):
    """Calendar-age weights normalized to mean 1 for stable regularization."""
    reference = rows.signal_time.max() if reference is None else reference
    if not config.recency_half_life_days:
        return np.ones(len(rows))
    age = np.maximum(0, (reference - rows.signal_time).dt.total_seconds().to_numpy() / 86400)
    # A shared age offset cancels when normalizing. Shift before exponentiation
    # so each block's most recent row remains finite for short half-lives.
    with np.errstate(over="ignore"):
        weights = np.exp2(-(age - age.min()) / config.recency_half_life_days)
    return weights / weights.mean()


class BoostModel:
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.constant_probability = None
        self.calibration = None
        self.metadata = {}
        self.features = FEATURES.copy()

    @classmethod
    def fit(cls, training, config):
        obj = cls()
        obj.features = feature_names(config.feature_set)
        n = len(training)
        calibration_start = n - min(21, max(6, int(n * 0.2)))
        early_start = int(calibration_start * 0.75)
        calibration = training.iloc[calibration_start:]
        early = purge_before(training.iloc[early_start:calibration_start], calibration.timestamp.iloc[0])
        core = purge_before(training.iloc[:early_start], training.timestamp.iloc[early_start])
        direction_training = purge_before(training, calibration.timestamp.iloc[0])
        if len(core) < 10 or len(early) < 5 or len(calibration) < 5:
            raise ValueError(
                "Too few purged rows for core, early stopping and calibration. Increase the training window."
            )
        reference = training.signal_time.max()
        weights = lambda rows: recency_weights(rows, config, reference)
        parameters = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "reg_lambda": config.reg_lambda,
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
            "random_state": config.seed,
        }
        selector = XGBRegressor(
            **parameters, early_stopping_rounds=15, objective="reg:squarederror", eval_metric="mae"
        )
        selector.fit(
            core[obj.features],
            core.target,
            sample_weight=weights(core),
            eval_set=[(early[obj.features], early.target)],
            sample_weight_eval_set=[weights(early)],
            verbose=False,
        )
        regression_trees = int(selector.best_iteration) + 1
        # Refit the chosen tree count on all known labels. Otherwise recent rows
        # would only stop/calibrate trees whose patterns came from older history.
        obj.regressor = XGBRegressor(
            **{**parameters, "n_estimators": regression_trees},
            objective="reg:squarederror",
            eval_metric="mae",
        )
        obj.regressor.fit(training[obj.features], training.target, sample_weight=weights(training))
        binary = (direction_training.target > 0).astype(int)
        warnings, direction_trees = [], None
        if binary.nunique() < 2 or (core.target > 0).nunique() < 2:
            obj.constant_probability = float(np.average(binary, weights=weights(direction_training)))
            warnings.append(
                "Training direction has one class in a required fitting block; probabilities use the weighted training frequency."
            )
        else:
            selector = XGBClassifier(
                **parameters, early_stopping_rounds=15, objective="binary:logistic", eval_metric="logloss"
            )
            selector.fit(
                core[obj.features],
                (core.target > 0).astype(int),
                sample_weight=weights(core),
                eval_set=[(early[obj.features], (early.target > 0).astype(int))],
                sample_weight_eval_set=[weights(early)],
                verbose=False,
            )
            direction_trees = int(selector.best_iteration) + 1
            obj.classifier = XGBClassifier(
                **{**parameters, "n_estimators": direction_trees},
                objective="binary:logistic",
                eval_metric="logloss",
            )
            obj.classifier.fit(
                direction_training[obj.features], binary, sample_weight=weights(direction_training)
            )
            if (calibration.target > 0).nunique() == 2:
                margin = obj.classifier.predict(calibration[obj.features], output_margin=True)
                calibrator = LogisticRegression(C=1, random_state=config.seed)
                calibrator.fit(
                    margin.reshape(-1, 1),
                    (calibration.target > 0).astype(int),
                    sample_weight=weights(calibration),
                )
                obj.calibration = {
                    "coefficient": float(calibrator.coef_[0, 0]),
                    "intercept": float(calibrator.intercept_[0]),
                }
            else:
                warnings.append(
                    "Calibration period has one class; raw classifier probabilities are reported."
                )
        all_weights = weights(training)
        obj.metadata = {
            "core_rows": len(core),
            "early_stopping_rows": len(early),
            "calibration_rows": len(calibration),
            "regression_fit_rows": len(training),
            "direction_fit_rows": len(direction_training),
            "best_iteration": regression_trees - 1,
            "regression_trees": regression_trees,
            "direction_trees": direction_trees,
            "features": obj.features,
            "parameters": parameters,
            "core_end": core.label_end.max().isoformat(),
            "early_start": early.signal_time.iloc[0].isoformat(),
            "early_end": early.label_end.max().isoformat(),
            "calibration_start": calibration.signal_time.iloc[0].isoformat(),
            "regression_fit_through": training.label_end.max().isoformat(),
            "direction_fit_through": direction_training.label_end.max().isoformat(),
            "trained_through": training.label_end.max().isoformat(),
            "training_start": training.signal_time.iloc[0].isoformat(),
            "recency": {
                "half_life_days": config.recency_half_life_days,
                "training_window_months": config.training_window_months,
                "reference_time": reference.isoformat(),
                "oldest_weight": float(all_weights[0]),
                "newest_weight": float(all_weights[-1]),
                "effective_samples": float(all_weights.sum() ** 2 / np.square(all_weights).sum()),
                "note": "Weights affect fitting, early stopping and calibration; reported performance metrics remain unweighted.",
            },
            "warnings": warnings,
        }
        return obj

    def predict(self, rows):
        regression = self.regressor.predict(rows[self.features]).astype(float)
        if self.classifier is None:
            probability = np.full(len(rows), self.constant_probability)
        elif self.calibration is None:
            probability = self.classifier.predict_proba(rows[self.features])[:, 1].astype(float)
        else:
            margin = self.classifier.predict(rows[self.features], output_margin=True)
            probability = expit(self.calibration["coefficient"] * margin + self.calibration["intercept"])
        return regression, probability

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        self.regressor.save_model(folder / "returns.json")
        if self.classifier is not None:
            self.classifier.save_model(folder / "direction.json")
        (folder / "model.json").write_text(
            json.dumps(
                {
                    "algorithm": "xgboost",
                    "calibration": self.calibration,
                    "constant_probability": self.constant_probability,
                    "metadata": self.metadata,
                    "features": self.features,
                },
                allow_nan=False,
            )
        )

    @classmethod
    def load(cls, folder):
        obj = cls()
        metadata = json.loads((folder / "model.json").read_text())
        obj.regressor = XGBRegressor()
        obj.regressor.load_model(folder / "returns.json")
        obj.features = (
            metadata.get("features") or obj.regressor.get_booster().feature_names or FEATURES.copy()
        )
        if metadata["constant_probability"] is None:
            obj.classifier = XGBClassifier()
            obj.classifier.load_model(folder / "direction.json")
        obj.constant_probability = metadata["constant_probability"]
        obj.calibration = metadata["calibration"]
        obj.metadata = metadata["metadata"]
        return obj

    def importance(self):
        return sorted(
            [
                {"feature": f, "importance": float(v)}
                for f, v in zip(self.features, self.regressor.feature_importances_)
            ],
            key=lambda r: -r["importance"],
        )


class PredictionPolicy:
    def __init__(self, rows, predicted, probability, config):
        self.lookup = {
            row.timestamp.isoformat(): (float(p), float(q))
            for (_, row), p, q in zip(rows.iterrows(), predicted, probability)
        }
        self.config = config
        self.holding = False
        self.last_target = 0.0
        self.decisions = {}

    def observe(self, point, terminal):
        self.holding = point["quantity"] > 0

    def target(self, history):
        forecast = self.lookup.get(history.iloc[-1].timestamp.isoformat())
        if forecast is None:
            return self.last_target
        expected, probability = forecast
        hurdle = entry_hurdle(self.config)
        if self.config.entry_rule == "probability_only":
            self.last_target = float(probability >= self.config.probability_threshold)
        elif self.holding:
            self.last_target = float(expected >= 0 and probability >= 1 - self.config.probability_threshold)
        else:
            self.last_target = float(expected > hurdle and probability >= self.config.probability_threshold)
        self.decisions[history.iloc[-1].timestamp.isoformat()] = self.last_target
        return self.last_target


def entry_hurdle(config):
    req = config.backtest
    # Approximate round-trip fixed fees at configured entry notional. The broker
    # still applies actual costs and sizing independently for both entry rules.
    return (
        2 * req.commission_bps + req.spread_bps + 2 * req.slippage_bps + config.min_edge_bps
    ) / 10000 + 2 * req.fixed_fee / max(req.initial_cash * req.allocation, 1)
