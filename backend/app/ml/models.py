"""CPU-only XGBoost; chronological early stopping and disjoint sigmoid calibration."""

import json

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from app.ml.features import FEATURES, purge_before


class BoostModel:
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.constant_probability = None
        self.calibration = None
        self.metadata = {}

    @classmethod
    def fit(cls, training, config):
        obj = cls()
        n = len(training)
        early_start, calibration_start = int(n * 0.6), int(n * 0.8)
        early = training.iloc[early_start:calibration_start]
        calibration = training.iloc[calibration_start:]
        core = purge_before(training.iloc[:early_start], early.timestamp.iloc[0])
        early = purge_before(early, calibration.timestamp.iloc[0])
        parameters = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": 1,
            "colsample_bytree": 1,
            "min_child_weight": 5,
            "reg_lambda": 5,
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": 1,
            "random_state": config.seed,
            "early_stopping_rounds": 15,
        }
        obj.regressor = XGBRegressor(**parameters, objective="reg:squarederror", eval_metric="mae")
        obj.regressor.fit(
            core[FEATURES], core.target, eval_set=[(early[FEATURES], early.target)], verbose=False
        )
        binary = (core.target > 0).astype(int)
        warnings = []
        if binary.nunique() < 2:
            obj.constant_probability = float(binary.mean())
            warnings.append(
                "Training direction has one class; direction probabilities use the constant training frequency."
            )
        else:
            obj.classifier = XGBClassifier(**parameters, objective="binary:logistic", eval_metric="logloss")
            obj.classifier.fit(
                core[FEATURES],
                binary,
                eval_set=[(early[FEATURES], (early.target > 0).astype(int))],
                verbose=False,
            )
            if (calibration.target > 0).nunique() == 2:
                margin = obj.classifier.predict(calibration[FEATURES], output_margin=True)
                calibrator = LogisticRegression(C=1, random_state=config.seed)
                calibrator.fit(margin.reshape(-1, 1), (calibration.target > 0).astype(int))
                obj.calibration = {
                    "coefficient": float(calibrator.coef_[0, 0]),
                    "intercept": float(calibrator.intercept_[0]),
                }
            else:
                warnings.append(
                    "Calibration period has one class; raw classifier probabilities are reported."
                )
        obj.metadata = {
            "core_rows": len(core),
            "early_stopping_rows": len(early),
            "calibration_rows": len(calibration),
            "best_iteration": int(obj.regressor.best_iteration),
            "core_end": core.label_end.iloc[-1].isoformat(),
            "early_start": early.signal_time.iloc[0].isoformat(),
            "early_end": early.label_end.iloc[-1].isoformat(),
            "calibration_start": calibration.signal_time.iloc[0].isoformat(),
            "trained_through": training.label_end.max().isoformat(),
            "warnings": warnings,
        }
        return obj

    def predict(self, rows):
        regression = self.regressor.predict(rows[FEATURES]).astype(float)
        if self.classifier is None:
            probability = np.full(len(rows), self.constant_probability)
        elif self.calibration is None:
            probability = self.classifier.predict_proba(rows[FEATURES])[:, 1].astype(float)
        else:
            margin = self.classifier.predict(rows[FEATURES], output_margin=True)
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
                for f, v in zip(FEATURES, self.regressor.feature_importances_)
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
        req = self.config.backtest
        # Conservative modeled round-trip hurdle. Fixed fees approximate entry
        # notional; actual broker costs/cash/volume caps still govern every fill.
        hurdle = (
            2 * req.commission_bps + req.spread_bps + 2 * req.slippage_bps + self.config.min_edge_bps
        ) / 10000
        hurdle += 2 * req.fixed_fee / max(req.initial_cash * req.allocation, 1)
        if self.holding:
            self.last_target = float(expected >= 0 and probability >= 1 - self.config.probability_threshold)
        else:
            self.last_target = float(expected > hurdle and probability >= self.config.probability_threshold)
        self.decisions[history.iloc[-1].timestamp.isoformat()] = self.last_target
        return self.last_target
