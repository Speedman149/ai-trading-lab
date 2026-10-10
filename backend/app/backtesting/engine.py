"""Backtrader service: close-time decisions, next-open modeled fills, cash only."""

import math

import backtrader as bt

from app.backtesting.metrics import metrics
from app.core.schemas import ASSETS
from app.data.market import closing_times, validate
from app.strategies.base import strategy_for


class Fees(bt.CommInfoBase):
    params = (
        ("commission", 0.0),
        ("fixed", 0.0),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission + (self.p.fixed if size else 0)


class ResearchStrategy(bt.Strategy):
    params = (("frame", None), ("request", None), ("logic", None))

    def __init__(self):
        self.curve, self.fills, self.closed, self.rejected = [], [], [], []
        self.closes = closing_times(self.p.frame, self.p.request.asset, self.p.request.timeframe)
        self.pending = None
        self.recorded = {}

    def notify_order(self, order):
        if order.status in (order.Completed, order.Partial):
            quantity = order.executed.size - self.recorded.get(order.ref, 0)
            self.recorded[order.ref] = order.executed.size
            if quantity:
                i = len(self.data) - 1
                raw = float(self.p.frame.iloc[i].open)
                cost = abs(quantity) * abs(order.executed.price - raw)
                self.fills.append(
                    {
                        "timestamp": self.p.frame.iloc[i].timestamp.isoformat(),
                        "signal_timestamp": order.info.signal_timestamp,
                        "side": "buy" if quantity > 0 else "sell",
                        "quantity": abs(quantity),
                        "price": order.executed.price,
                        "reference_open": raw,
                        "commission": order.executed.comm,
                        "spread_slippage_cost": cost,
                        "total_cost": cost + order.executed.comm,
                    }
                )
        elif order.status in (order.Margin, order.Rejected):
            self.rejected.append(
                {
                    "timestamp": self.p.frame.iloc[len(self.data) - 1].timestamp.isoformat(),
                    "reason": order.getstatusname(),
                }
            )

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed.append(
                {
                    "timestamp": self.p.frame.iloc[len(self.data) - 1].timestamp.isoformat(),
                    "gross_pnl": trade.pnl,
                    "net_pnl": trade.pnlcomm,
                }
            )

    def next(self):
        # A participation-capped partial order expires after its first executable bar.
        if self.pending and self.pending.alive():
            self.cancel(self.pending)
            self.rejected.append(
                {
                    "timestamp": self.p.frame.iloc[len(self.data) - 1].timestamp.isoformat(),
                    "reason": "Unfilled remainder expired after one bar",
                }
            )
        i = len(self.data) - 1
        cash, equity = self.broker.getcash(), self.broker.getvalue()
        qty = self.position.size
        self.curve.append(
            {
                "timestamp": self.closes[i].isoformat(),
                "equity": equity,
                "cash": cash,
                "quantity": qty,
                "exposure": qty * self.data.close[0] / equity if equity else 0,
                "unrealized_pnl": qty * (self.data.close[0] - self.position.price),
            }
        )
        if i == len(self.p.frame) - 1:
            return
        target = self.p.logic.target(self.p.frame.iloc[: i + 1])
        # Buy once and retain shares, rather than continually rebalance the benchmark.
        if target and qty > 0:
            return
        if not target and qty > 0:
            self.pending = self.sell(size=qty)
        elif target and qty == 0:
            req = self.p.request
            budget = min(equity * req.allocation, cash - equity * req.cash_reserve)
            impact = (req.spread_bps / 2 + req.slippage_bps) / 10000
            size = max(
                0,
                (budget - req.fixed_fee)
                / (self.data.close[0] * (1 + impact) * (1 + req.commission_bps / 10000)),
            )
            if not req.fractional:
                size = math.floor(size)
            if size * self.data.close[0] < req.min_notional or size <= 0:
                self.rejected.append(
                    {
                        "timestamp": self.p.frame.iloc[i].timestamp.isoformat(),
                        "reason": "Below minimum notional or insufficient cash",
                    }
                )
                return
            self.pending = self.buy(size=size)
        else:
            return
        self.pending.addinfo(signal_timestamp=self.closes[i].isoformat())


def run_engine(frame, req, logic=None):
    validate(frame, req.asset, req.timeframe)
    cerebro = bt.Cerebro(stdstats=False)
    feed = frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
    feed.index = feed.index.tz_localize(None)  # Backtrader internal UTC floats only.
    cerebro.adddata(bt.feeds.PandasData(dataname=feed))
    cerebro.broker.setcash(req.initial_cash)
    cerebro.broker.addcommissioninfo(Fees(commission=req.commission_bps / 10000, fixed=req.fixed_fee))
    impact = (req.spread_bps / 2 + req.slippage_bps) / 10000
    # Costs may move beyond candle high/low: this is an explicit modeled assumption.
    cerebro.broker.set_slippage_perc(impact, slip_open=True, slip_match=True, slip_out=True)

    def filler(order, price, ago):
        remaining = abs(order.executed.remsize)
        # Prior completed volume only; never use the future bar's total volume.
        volume = float(order.data.volume[-1])
        size = min(remaining, volume * req.max_participation)
        if order.isbuy():
            cash = cerebro.broker.getcash()
            equity_open = cash + cerebro.broker.getposition(order.data).size * float(order.data.open[0])
            budget = min(cash - equity_open * req.cash_reserve, equity_open * req.allocation)
            size = min(size, max(0, (budget - req.fixed_fee) / (price * (1 + req.commission_bps / 10000))))
        if not req.fractional:
            size = math.floor(size)
        return size if size * price >= req.min_notional else 0

    cerebro.broker.set_filler(filler)
    cerebro.addstrategy(ResearchStrategy, frame=frame, request=req, logic=logic or strategy_for(req))
    strategy = cerebro.run()[0]
    if min(row["cash"] for row in strategy.curve) < -1e-7:
        raise RuntimeError("Cash invariant violated")
    return {
        "metrics": metrics(
            strategy.curve, strategy.fills, strategy.closed, req.initial_cash, ASSETS[req.asset]
        ),
        "equity": strategy.curve,
        "trades": strategy.fills,
        "closed_trades": strategy.closed,
        "rejected_orders": strategy.rejected,
    }
