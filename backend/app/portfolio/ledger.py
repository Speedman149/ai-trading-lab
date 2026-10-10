"""Long-only accounting independent of Backtrader's unadjusted trade objects."""

from app.data.errors import DataError


class Ledger:
    def __init__(self):
        self.quantity = 0.0
        self.basis = 0.0
        self.trade_costs = 0.0
        self.trade_profit = 0.0
        self.trade_income = 0.0
        self.dividend_income = 0.0
        self.closed = []

    def split(self, ratio):
        self.quantity *= ratio
        # Total cost basis is preserved, so price per share changes inversely.

    def dividend(self, amount):
        self.dividend_income += amount
        self.trade_income += amount

    def fill(self, side, quantity, price, commission, timestamp):
        if side == "buy":
            if self.quantity == 0:
                self.trade_costs = self.trade_profit = self.trade_income = 0.0
            self.quantity += quantity
            self.basis += quantity * price
            self.trade_costs += commission
        else:
            if quantity > self.quantity + 1e-8:
                raise DataError("Sell quantity exceeds corporate-action-adjusted holdings")
            share = min(1.0, quantity / self.quantity)
            cost = self.basis * share
            self.basis -= cost
            self.quantity -= quantity
            self.trade_profit += quantity * price - cost
            self.trade_costs += commission
            if self.quantity <= 1e-8:
                self.quantity = self.basis = 0.0
                self.closed.append(
                    {
                        "timestamp": timestamp,
                        "gross_pnl": self.trade_profit + self.trade_income,
                        "net_pnl": self.trade_profit + self.trade_income - self.trade_costs,
                    }
                )

    def unrealized(self, close):
        return self.quantity * close - self.basis
