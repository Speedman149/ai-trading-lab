from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ASSETS = {"AAPL": "stock", "SPY": "stock", "BTC/USD": "crypto", "ETH/USD": "crypto"}
TIMEFRAMES = {"1d": 1440, "1h": 60, "15m": 15, "5m": 5}


class BacktestRequest(BaseModel):
    asset: Literal["AAPL", "SPY", "BTC/USD", "ETH/USD"] = "AAPL"
    timeframe: Literal["1d", "1h", "15m", "5m"] = "1d"
    provider: Literal["synthetic", "alpaca"] = "synthetic"
    start: date = date(2024, 1, 2)
    end: date = date(2024, 6, 30)
    strategy: Literal["buy_hold", "ma_crossover"] = "ma_crossover"
    fast: int = Field(10, ge=2, le=200)
    slow: int = Field(30, ge=3, le=500)
    initial_cash: float = Field(10000, ge=100, le=1e9, allow_inf_nan=False)
    commission_bps: float = Field(10, ge=0, le=500, allow_inf_nan=False)
    fixed_fee: float = Field(0, ge=0, le=100, allow_inf_nan=False)
    spread_bps: float = Field(4, ge=0, le=500, allow_inf_nan=False)
    slippage_bps: float = Field(2, ge=0, le=500, allow_inf_nan=False)
    allocation: float = Field(0.9, gt=0, le=1, allow_inf_nan=False)
    cash_reserve: float = Field(0.05, ge=0, lt=1, allow_inf_nan=False)
    max_participation: float = Field(0.01, gt=0, le=1, allow_inf_nan=False)
    stop_loss_pct: float = Field(0, ge=0, le=50, allow_inf_nan=False)
    fractional: bool = True
    min_notional: float = Field(1, ge=0, le=1e6, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_range(self):
        if self.start >= self.end:
            raise ValueError("Start must precede end (end date is exclusive)")
        if (self.end - self.start).days > 3650:
            raise ValueError("Requests are limited to 3650 days")
        if self.fast >= self.slow:
            raise ValueError("Fast window must be smaller than slow window")
        return self


class EquityPoint(BaseModel):
    timestamp: str
    equity: float
    cash: float
    quantity: float
    exposure: float
    unrealized_pnl: float
    drawdown: float
    dividend_receivable: float = 0


class FillRecord(BaseModel):
    timestamp: str
    signal_timestamp: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    reference_open: float
    reference_price: float | None = None
    reason: str = "signal"
    commission: float
    spread_slippage_cost: float
    total_cost: float


class ClosedTrade(BaseModel):
    timestamp: str
    gross_pnl: float
    net_pnl: float


class RejectedOrder(BaseModel):
    timestamp: str
    reason: str


class CorporateActionEvent(BaseModel):
    id: str
    kind: Literal["split", "dividend_entitlement", "dividend_payment"]
    timestamp: str
    quantity: float
    amount: float
    ratio: float


class SimulationResult(BaseModel):
    corporate_actions: list[CorporateActionEvent] = Field(default_factory=list)
    metrics: dict[str, float | int | None]
    equity: list[EquityPoint]
    trades: list[FillRecord]
    closed_trades: list[ClosedTrade]
    rejected_orders: list[RejectedOrder]


class DatasetProvenance(BaseModel):
    asset: str
    timeframe: str
    provider: str
    start: str
    end: str
    version: str
    feed: str
    dataset_hash: str
    rows: int
    synthetic: bool
    processing_version: str
    corporate_actions_hash: str
    corporate_actions_count: int


class BacktestResult(SimulationResult):
    id: str
    config: BacktestRequest
    provenance: DatasetProvenance
    code_version: str
    benchmark: SimulationResult
    warnings: list[str]
    assumptions: list[str]
