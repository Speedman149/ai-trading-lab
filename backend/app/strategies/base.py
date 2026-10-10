from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    def target(self, history: pd.DataFrame) -> float: ...


class BuyHold:
    def target(self, history):
        return 1.0


class MovingAverage:
    def __init__(self, fast, slow):
        self.fast, self.slow = fast, slow

    def target(self, history):
        if len(history) < self.slow:
            return 0.0
        closes = history.close
        return float(closes.iloc[-self.fast :].mean() > closes.iloc[-self.slow :].mean())


def strategy_for(req):
    return BuyHold() if req.strategy == "buy_hold" else MovingAverage(req.fast, req.slow)
