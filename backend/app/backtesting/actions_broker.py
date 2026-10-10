"""Apply supported corporate actions before next-open market executions."""

import math

import backtrader as bt

from app.data.errors import DataError


class ActionsBroker(bt.brokers.BackBroker):
    def __init__(self, frame, actions, ledger, req):
        super().__init__()
        self.frame, self.actions, self.ledger, self.req = frame, actions, ledger, req
        self.applied = set()
        self.receivables = []
        self.events = []
        self.split_factors = [1.0] * len(frame)
        self.effective_splits = []

    @property
    def receivable(self):
        return sum(item["amount"] for item in self.receivables)

    def next(self):
        if self.cerebro.datas and len(self.cerebro.datas[0]):
            data = self.cerebro.datas[0]
            index = len(data) - 1
            timestamp = self.frame.iloc[index].timestamp
            today = timestamp.tz_convert("America/New_York").date()
            # Pay existing entitlements before today's executions. Income is not external capital.
            unpaid = []
            for item in self.receivables:
                if item["payable_date"] <= today:
                    self.cash += item["amount"]
                    self.events.append(
                        {
                            "id": item["id"],
                            "kind": "dividend_payment",
                            "timestamp": timestamp.isoformat(),
                            "quantity": item["quantity"],
                            "amount": item["amount"],
                            "ratio": 1.0,
                        }
                    )
                else:
                    unpaid.append(item)
            self.receivables = unpaid
            for action in self.actions:
                if action.id in self.applied or action.ex_date > today:
                    continue
                self.applied.add(action.id)
                position = self.getposition(data)
                quantity = position.size
                if action.kind == "split":
                    adjusted = quantity * action.ratio
                    if not self.req.fractional and not math.isclose(adjusted, round(adjusted), abs_tol=1e-8):
                        raise DataError(
                            "This split creates fractional shares. Enable fractional quantities; cash-in-lieu settlement is not modeled."
                        )
                    position.fix(adjusted, position.price / action.ratio if quantity else 0)
                    self.ledger.split(action.ratio)
                    self.split_factors[index] *= action.ratio
                    self.effective_splits.append((index, action.ratio))
                    # Outstanding unfilled market instructions keep the same intended economics.
                    orders = {
                        order.ref: order
                        for order in list(self.submitted) + list(self.pending)
                        if order is not None and order.alive()
                    }
                    for order in orders.values():
                        if order.executed.size:
                            raise DataError("A split encountered an unexpected multi-bar partial order")
                        order.created.size *= action.ratio
                        order.created.price /= action.ratio
                        order.created.pclose /= action.ratio
                        order.size *= action.ratio
                        order.executed.remsize *= action.ratio
                    self.events.append(
                        {
                            "id": action.id,
                            "kind": "split",
                            "timestamp": timestamp.isoformat(),
                            "quantity": adjusted,
                            "amount": 0.0,
                            "ratio": action.ratio,
                        }
                    )
                elif quantity:
                    amount = quantity * action.amount
                    self.ledger.dividend(amount)
                    self.events.append(
                        {
                            "id": action.id,
                            "kind": "dividend_entitlement",
                            "timestamp": timestamp.isoformat(),
                            "quantity": quantity,
                            "amount": amount,
                            "ratio": 1.0,
                        }
                    )
                    if action.payable_date <= today:
                        self.cash += amount
                        self.events.append(
                            {
                                "id": action.id,
                                "kind": "dividend_payment",
                                "timestamp": timestamp.isoformat(),
                                "quantity": quantity,
                                "amount": amount,
                                "ratio": 1.0,
                            }
                        )
                    else:
                        self.receivables.append(
                            {
                                "id": action.id,
                                "quantity": quantity,
                                "amount": amount,
                                "payable_date": action.payable_date,
                            }
                        )
        super().next()
