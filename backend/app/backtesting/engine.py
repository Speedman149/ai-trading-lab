"""Backtrader service: close-time decisions, next-open modeled fills, cash only."""

import math

import backtrader as bt

from app.backtesting.actions_broker import ActionsBroker
from app.backtesting.metrics import metrics
from app.core.schemas import ASSETS
from app.data.actions import CorporateAction
from app.data.errors import DataError
from app.data.market import closing_times, validate
from app.portfolio.ledger import Ledger
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
        self.closed = self.broker.ledger.closed

    def notify_order(self, order):
        if order.status in (order.Completed, order.Partial):
            quantity = order.executed.size - self.recorded.get(order.ref, 0)
            self.recorded[order.ref] = order.executed.size
            if quantity:
                i = len(self.data) - 1
                raw = float(self.p.frame.iloc[i].open)
                cost = abs(quantity) * abs(order.executed.price - raw)
                self.broker.ledger.fill(
                    "buy" if quantity > 0 else "sell",
                    abs(quantity),
                    order.executed.price,
                    order.executed.comm,
                    self.p.frame.iloc[i].timestamp.isoformat(),
                )
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
        cash = self.broker.getcash()
        equity = self.broker.getvalue() + self.broker.receivable
        qty = self.position.size
        self.curve.append(
            {
                "timestamp": self.closes[i].isoformat(),
                "equity": equity,
                "cash": cash,
                "quantity": qty,
                "exposure": qty * self.data.close[0] / equity if equity else 0,
                "unrealized_pnl": self.broker.ledger.unrealized(self.data.close[0]),
                "dividend_receivable": self.broker.receivable,
            }
        )
        terminal = i == len(self.p.frame) - 1
        if hasattr(self.p.logic, "observe"):
            self.p.logic.observe(self.curve[-1].copy(), terminal)
        if terminal:
            return
        history = self.p.frame.iloc[: i + 1]
        history.attrs = {}  # Do not expose future action records through DataFrame metadata.
        if self.broker.effective_splits:
            history = history.copy()
            for split_index, ratio in self.broker.effective_splits:
                for column in ("open", "high", "low", "close"):
                    history.loc[history.index[:split_index], column] /= ratio
                history.loc[history.index[:split_index], "volume"] *= ratio
        target = self.p.logic.target(history)
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
    actions = [CorporateAction.model_validate(action) for action in frame.attrs.get("corporate_actions", [])]
    if req.provider == "alpaca" and ASSETS[req.asset] == "stock":
        if "corporate_actions_verified" not in frame or not frame.corporate_actions_verified.all():
            raise DataError("Real stock simulation needs verified corporate-action metadata")
        if "corporate_actions" not in frame.attrs:
            raise DataError("Real stock simulation is missing corporate-action records")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker = ActionsBroker(frame=frame, actions=actions, ledger=Ledger(), req=req)
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
        volume = float(order.data.volume[-1]) * cerebro.broker.split_factors[len(order.data) - 1]
        size = min(remaining, volume * req.max_participation)
        if order.isbuy():
            cash = cerebro.broker.getcash()
            equity_open = (
                cash
                + cerebro.broker.getposition(order.data).size * float(order.data.open[0])
                + cerebro.broker.receivable
            )
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
    stats = metrics(strategy.curve, strategy.fills, strategy.closed, req.initial_cash, ASSETS[req.asset])
    stats["dividend_income"] = cerebro.broker.ledger.dividend_income
    stats["dividend_receivable"] = cerebro.broker.receivable
    return {
        "metrics": stats,
        "corporate_actions": cerebro.broker.events,
        "equity": strategy.curve,
        "trades": strategy.fills,
        "closed_trades": strategy.closed,
        "rejected_orders": strategy.rejected,
    }
