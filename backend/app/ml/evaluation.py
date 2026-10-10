import numpy as np

from app.backtesting.engine import run_engine
from app.core.schemas import ASSETS
from app.ml.diagnostics import prediction_metrics
from app.ml.models import PredictionPolicy, entry_hurdle


class LookupPolicy:
    def __init__(self, mapping):
        self.mapping, self.last = mapping, 0.0

    def target(self, history):
        self.last = self.mapping.get(history.iloc[-1].timestamp.isoformat(), self.last)
        return self.last


def regime(row, volatility_threshold):
    trend = "uptrend" if row.return_20 >= 0 else "downtrend"
    volatility = "high volatility" if row.volatility_20 > volatility_threshold else "low volatility"
    return f"{trend} / {volatility}"


def evaluation_frame(frame, rows):
    start = int(rows.bar_index.iloc[0])
    last_exit = rows.attrs.get("simulation_end", rows.label_end.iloc[-1])
    end = int(frame.index[frame.timestamp == last_exit][0])
    sliced = frame.iloc[start : end + 1].copy().reset_index(drop=True)
    # Remove pre-window actions: this portfolio starts with cash, with no past entitlement.
    first_day = sliced.timestamp.iloc[0].tz_convert("America/New_York").date().isoformat()
    sliced.attrs = {
        "corporate_actions": [
            a for a in frame.attrs.get("corporate_actions", []) if a["ex_date"] >= first_day
        ]
    }
    return sliced


def regime_diagnostics(
    simulation, eval_frame, features, rows, predicted, probability, volatility_threshold, req
):
    lookup = {row.timestamp.isoformat(): row for _, row in features.iterrows()}
    returns = np.asarray([p["equity"] for p in simulation["equity"]])
    returns = returns / np.r_[req.initial_cash, returns[:-1]] - 1
    attributed = {}
    # Attribute interval P&L using the preceding completed bar's known regime.
    for i in range(1, len(eval_frame)):
        previous = lookup.get(eval_frame.iloc[i - 1].timestamp.isoformat())
        if previous is None:
            continue
        name = regime(previous, volatility_threshold)
        attributed.setdefault(name, []).append(float(returns[i]))
    result = []
    names = [regime(row, volatility_threshold) for _, row in rows.iterrows()]
    for name in sorted(set(names) | set(attributed)):
        mask = np.asarray([n == name for n in names])
        interval_returns = np.asarray(attributed.get(name, []))
        vol = float(np.std(interval_returns, ddof=1)) if len(interval_returns) > 1 else 0
        curve = np.r_[1, np.cumprod(1 + interval_returns)]
        dd = curve / np.maximum.accumulate(curve) - 1
        annual = 365 if ASSETS[req.asset] == "crypto" else 252
        result.append(
            {
                "regime": name,
                "intervals": len(interval_returns),
                "sufficient": len(interval_returns) >= 20,
                "prediction": prediction_metrics(
                    rows.target.to_numpy()[mask],
                    None if predicted is None else predicted[mask],
                    None if probability is None else probability[mask],
                ),
                "net_return": float(curve[-1] - 1),
                "max_drawdown": float(dd.min()),
                "sharpe": float(interval_returns.mean() / vol * np.sqrt(annual)) if vol > 1e-12 else None,
            }
        )
    return result


def evaluate(frame, features, rows, model, config, training):
    predicted, probability = model.predict(rows)
    policy = (
        model.policy(rows)
        if config.algorithm == "q_learning"
        else PredictionPolicy(rows, predicted, probability, config)
    )
    test_frame = evaluation_frame(frame, rows)
    simulation = run_engine(test_frame, config.backtest, policy)
    dates = [row.timestamp.isoformat() for _, row in rows.iterrows()]
    baseline_request = config.backtest.model_copy(update={"stop_loss_pct": 0})
    baselines = {
        "cash": run_engine(test_frame, baseline_request, LookupPolicy({d: 0.0 for d in dates})),
        "buy_hold": run_engine(test_frame, baseline_request, LookupPolicy({d: 1.0 for d in dates})),
        "ma_crossover": run_engine(
            test_frame,
            baseline_request,
            LookupPolicy(
                {
                    row.timestamp.isoformat(): float(row.ma_gap_10 < row.ma_gap_30)
                    for _, row in rows.iterrows()
                }
            ),
        ),
    }
    mean, frequency = float(training.target.mean()), float((training.target > 0).mean())
    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        predictions.append(
            {
                "signal_time": row.signal_time.isoformat(),
                "entry_time": row.entry_time.isoformat(),
                "label_end": row.label_end.isoformat(),
                "actual_return": float(row.target),
                "predicted_return": None if predicted is None else float(predicted[i]),
                "probability_up": None if probability is None else float(probability[i]),
                "target": getattr(policy, "decisions", {}).get(row.timestamp.isoformat()),
            }
        )
    return {
        "activity": activity_diagnostics(simulation, baselines, predicted, probability, config),
        "prediction": prediction_metrics(rows.target, predicted, probability),
        "prediction_baselines": {
            "zero_return": prediction_metrics(
                rows.target, np.zeros(len(rows)), np.full(len(rows), frequency)
            ),
            "training_mean": prediction_metrics(
                rows.target, np.full(len(rows), mean), np.full(len(rows), frequency)
            ),
        },
        "strategy": simulation,
        "baselines": baselines,
        "predictions": predictions,
        "regimes": regime_diagnostics(
            simulation,
            test_frame,
            features,
            rows,
            predicted,
            probability,
            float(training.volatility_20.median()),
            config.backtest,
        ),
    }


def activity_diagnostics(simulation, baselines, predicted, probability, config):
    metrics = simulation["metrics"]
    closes = simulation["equity"]
    invested = sum(point["quantity"] > 0 for point in closes)
    result = {
        "entry_rule": config.entry_rule if config.algorithm == "xgboost" else "q_learning",
        "time_in_market": invested / len(closes) if closes else 0,
        "average_exposure": float(np.mean([point["exposure"] for point in closes])) if closes else 0,
        "fills": metrics["fills"],
        "closed_trades": metrics["closed_trades"],
        "rejected_orders": len(simulation["rejected_orders"]),
        "buy_hold_return": baselines["buy_hold"]["metrics"]["net_return"],
        "excess_vs_buy_hold": metrics["net_return"] - baselines["buy_hold"]["metrics"]["net_return"],
        "entry_hurdle": None,
        "signals": None,
        "return_passes": None,
        "probability_passes": None,
        "qualifying_signals": None,
        "predicted_return_min": None,
        "predicted_return_max": None,
        "probability_min": None,
        "probability_max": None,
        "notes": [],
    }
    if predicted is not None:
        predicted, probability = np.asarray(predicted), np.asarray(probability)
        hurdle = entry_hurdle(config)
        returns_pass = predicted > hurdle
        probabilities_pass = probability >= config.probability_threshold
        qualifying = (
            probabilities_pass
            if config.entry_rule == "probability_only"
            else returns_pass & probabilities_pass
        )
        result.update(
            entry_hurdle=hurdle,
            signals=len(predicted),
            return_passes=int(returns_pass.sum()),
            probability_passes=int(probabilities_pass.sum()),
            qualifying_signals=int(qualifying.sum()),
            predicted_return_min=float(predicted.min()),
            predicted_return_max=float(predicted.max()),
            probability_min=float(probability.min()),
            probability_max=float(probability.max()),
        )
        if not qualifying.any():
            result["notes"].append(
                "No prediction met this strategy's entry conditions. It stayed in cash; check the forecast range and probability threshold below."
            )
        elif not metrics["fills"]:
            result["notes"].append(
                "Some signals qualified but no order filled. Review cash, minimum order size, and volume limits."
            )
    if metrics["closed_trades"] < 5:
        result["notes"].append(
            f"Only {metrics['closed_trades']} completed trades; there is little independent trade evidence, even if the return is positive."
        )
    if result["excess_vs_buy_hold"] < 0:
        result["notes"].append(
            "The AI earned less than buy-and-hold over the same dates and execution settings."
        )
    result["notes"].append(
        "Qualifying signals count each prediction as a possible entry; they are not a count of orders. Time invested uses daily-close holdings and can miss intraday risk."
    )
    return result
