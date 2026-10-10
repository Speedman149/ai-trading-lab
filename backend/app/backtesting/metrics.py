import numpy as np
import pandas as pd


def metrics(curve, trades, closed, initial, asset_class):
    df = pd.DataFrame(curve)
    equity = df.equity
    peak = equity.cummax().clip(lower=initial)
    drawdown = equity / peak - 1
    for row, dd in zip(curve, drawdown):
        row["drawdown"] = float(dd)
    daily = (
        pd.Series(equity.to_numpy(), index=pd.to_datetime(df.timestamp, utc=True))
        .resample("1D")
        .last()
        .dropna()
    )
    returns = daily.pct_change().dropna()
    annual = 365 if asset_class == "crypto" else 252
    vol = float(returns.std(ddof=1)) if len(returns) >= 2 else 0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2))) if len(returns) else 0
    duration = (
        pd.to_datetime(df.timestamp.iloc[-1]) - pd.to_datetime(df.timestamp.iloc[0])
    ).total_seconds() / 86400
    costs = sum(t["total_cost"] for t in trades)
    wins = [t["net_pnl"] for t in closed if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in closed if t["net_pnl"] < 0]
    net = equity.iloc[-1] / initial - 1
    result = {
        "net_return": float(net),
        "gross_return_same_fills": float((equity.iloc[-1] + costs) / initial - 1),
        "annualized_return": float((1 + net) ** (365.25 / duration) - 1) if duration >= 30 else None,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": vol * np.sqrt(annual) if vol else None,
        "sharpe": float(returns.mean() / vol * np.sqrt(annual)) if vol > 1e-12 else None,
        "sortino": float(returns.mean() / downside * np.sqrt(annual)) if downside > 1e-12 else None,
        "fills": len(trades),
        "closed_trades": len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "average_win": float(np.mean(wins)) if wins else None,
        "average_loss": float(np.mean(losses)) if losses else None,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "transaction_costs": costs,
        "turnover": sum(abs(t["quantity"] * t["price"]) for t in trades) / initial,
        "final_equity": float(equity.iloc[-1]),
        "final_cash": curve[-1]["cash"],
        "realized_pnl": sum(t["net_pnl"] for t in closed),
        "unrealized_pnl": curve[-1]["unrealized_pnl"],
    }
    return {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in result.items()}
