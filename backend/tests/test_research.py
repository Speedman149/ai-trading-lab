from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backtesting.engine import run_engine
from app.backtesting.metrics import metrics
from app.core.schemas import BacktestRequest
from app.data.market import DataError, expected_times, load, synthetic, validate
from app.main import app
from app.strategies.base import MovingAverage


def request(**kwargs):
    kwargs.setdefault("strategy", "buy_hold")
    return BacktestRequest(asset="BTC/USD", start=date(2024, 1, 2), end=date(2024, 1, 8), **kwargs)


def bars(prices):
    frame = synthetic(request()).iloc[: len(prices)].copy()
    for col in ["open", "high", "low", "close"]:
        frame[col] = prices
    frame["volume"] = 1000000
    return frame


def test_next_open_manual_accounting():
    req = request(
        strategy="buy_hold",
        initial_cash=1000,
        allocation=0.5,
        cash_reserve=0,
        commission_bps=0,
        spread_bps=0,
        slippage_bps=0,
    )
    result = run_engine(bars([10, 20, 30]), req)
    trade = result["trades"][0]
    assert trade["price"] == 20
    assert trade["quantity"] == 25
    assert trade["timestamp"] >= trade["signal_timestamp"]
    assert result["metrics"]["final_cash"] == 500
    assert result["metrics"]["final_equity"] == 1250
    assert result["metrics"]["unrealized_pnl"] == 250


def test_fees_spread_slippage_manual():
    req = request(
        strategy="buy_hold",
        initial_cash=1000,
        allocation=0.5,
        cash_reserve=0,
        commission_bps=100,
        fixed_fee=2,
        spread_bps=100,
        slippage_bps=50,
    )
    result = run_engine(bars([10, 10, 10]), req)
    fill = result["trades"][0]
    quantity = 498 / (10.1 * 1.01)
    assert fill["price"] == pytest.approx(10.1)
    assert fill["quantity"] == pytest.approx(quantity)
    assert fill["commission"] == pytest.approx(quantity * 10.1 * 0.01 + 2)
    assert result["metrics"]["final_cash"] == pytest.approx(500)
    assert result["metrics"]["final_equity"] == pytest.approx(500 + quantity * 10)
    assert result["metrics"]["transaction_costs"] == pytest.approx(1000 - result["metrics"]["final_equity"])


class RoundTrip:
    def target(self, history):
        return float(len(history) == 1)


def test_sell_and_realized_pnl():
    req = request(
        initial_cash=1000, allocation=0.5, cash_reserve=0, commission_bps=0, spread_bps=0, slippage_bps=0
    )
    result = run_engine(bars([10, 10, 12]), req, RoundTrip())
    assert [t["side"] for t in result["trades"]] == ["buy", "sell"]
    assert result["metrics"]["final_cash"] == 1100
    assert result["metrics"]["realized_pnl"] == 100
    assert result["metrics"]["closed_trades"] == 1


def test_cash_reserve_gap_and_integer():
    req = request(
        initial_cash=1000,
        allocation=1,
        cash_reserve=0.2,
        fractional=False,
        commission_bps=10,
        spread_bps=10,
        slippage_bps=10,
    )
    result = run_engine(bars([10, 100, 100]), req)
    assert result["metrics"]["final_cash"] >= 200
    assert result["trades"][0]["quantity"] == int(result["trades"][0]["quantity"])


def test_insufficient_cash_fixed_fee_no_fill():
    req = request(initial_cash=100, allocation=0.1, fixed_fee=20)
    result = run_engine(bars([10, 10, 10]), req)
    assert result["trades"] == []
    assert result["rejected_orders"]
    assert result["metrics"]["final_cash"] == 100


def test_no_trades_undefined_metrics():
    result = run_engine(bars([10, 9, 8]), request(strategy="ma_crossover"))
    assert result["metrics"]["net_return"] == 0
    assert result["metrics"]["sharpe"] is None
    assert result["metrics"]["profit_factor"] is None


def test_drawdown_manual():
    curve = [
        {"timestamp": f"2024-01-0{i + 1}T00:00:00+00:00", "equity": value, "cash": value, "unrealized_pnl": 0}
        for i, value in enumerate([100, 120, 90, 110])
    ]
    result = metrics(curve, [], [], 100, "crypto")
    assert result["max_drawdown"] == -0.25
    assert result["annualized_return"] is None


@pytest.mark.parametrize(
    "mutation",
    ["duplicates", "order", "price", "volume", "ohlc", "incomplete", "symbol", "timezone", "nan", "gap"],
)
def test_invalid_data(mutation):
    req = request()
    frame = synthetic(req)
    if mutation == "duplicates":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    if mutation == "order":
        frame = frame.iloc[::-1]
    if mutation == "price":
        frame.loc[0, "open"] = -1
    if mutation == "volume":
        frame.loc[0, "volume"] = -1
    if mutation == "ohlc":
        frame.loc[0, "high"] = 1
    if mutation == "incomplete":
        frame.loc[len(frame) - 1, "complete"] = False
    if mutation == "symbol":
        frame.loc[0, "asset"] = "ETH/USD"
    if mutation == "timezone":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    if mutation == "nan":
        frame.loc[0, "close"] = np.nan
    if mutation == "gap":
        frame = frame.drop(2)
    with pytest.raises(DataError):
        validate(frame, req.asset, req.timeframe, req.start, req.end)


def test_calendars_holiday_early_close_dst():
    stocks = expected_times("AAPL", "1d", date(2024, 7, 3), date(2024, 7, 6))
    crypto = expected_times("BTC/USD", "1d", date(2024, 7, 3), date(2024, 7, 6))
    assert len(stocks) == 2 and len(crypto) == 3
    early = expected_times("AAPL", "1h", date(2024, 7, 3), date(2024, 7, 4))
    assert len(early) == 4
    dst = expected_times("AAPL", "1d", date(2024, 3, 8), date(2024, 3, 12))
    assert [t.hour for t in dst] == [14, 13]


@pytest.mark.parametrize("timeframe", ["1d", "1h", "15m", "5m"])
def test_timeframes(timeframe):
    ts = expected_times("BTC/USD", timeframe, date(2024, 1, 2), date(2024, 1, 3))
    assert len(ts) == {"1d": 1, "1h": 24, "15m": 96, "5m": 288}[timeframe]


def test_features_do_not_access_future():
    frame = bars([10, 10, 10, 10, 20, 1])
    logic = MovingAverage(2, 3)
    assert logic.target(frame.iloc[:4]) == 0
    changed = frame.copy()
    changed.loc[4:, "close"] = 1e9
    assert logic.target(frame.iloc[:4]) == logic.target(changed.iloc[:4])


def test_liquidity_uses_prior_volume():
    frame = bars([10, 10, 10])
    frame.loc[0, "volume"], frame.loc[1, "volume"] = 100, 1e9
    result = run_engine(frame, request(strategy="buy_hold", max_participation=0.01))
    assert result["trades"][0]["quantity"] == 1


def test_cache_and_reproducibility():
    req = request()
    a, provenance_a = load(req)
    b, provenance_b = load(req)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert provenance_a == provenance_b
    assert run_engine(a, req) == run_engine(b, req)


def test_api_end_to_end_and_benchmark():
    client = TestClient(app)
    assert client.get("/health").json()["live_trading"] is False
    for asset in ["AAPL", "BTC/USD"]:
        response = client.post("/api/backtests", json={"asset": asset})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["provenance"]["synthetic"]
        identifier = result["id"]
        assert client.get(f"/api/backtests/{identifier}").json() == result
        assert client.get(f"/api/backtests/{identifier}/trades").json() == result["trades"]
        assert client.get(f"/api/backtests/{identifier}/equity").json() == result["equity"]
        req = BacktestRequest(asset=asset, strategy="buy_hold")
        frame, _ = load(req)
        assert result["benchmark"] == run_engine(frame, req)
    assert client.get("/api/backtests/invalid").status_code == 404
    assert client.post("/api/backtests", json={"fast": 30, "slow": 10}).status_code == 422
    assert client.post("/api/backtests", json={"asset": "../secret"}).status_code == 422
    assert client.post("/api/backtests", json={"initial_cash": -1}).status_code == 422
