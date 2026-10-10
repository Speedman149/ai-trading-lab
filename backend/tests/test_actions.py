from datetime import date

import httpx
import pandas as pd
import pytest

from app.backtesting.engine import run_engine
from app.core.config import settings
from app.core.schemas import BacktestRequest
from app.data import market
from app.data.actions import parse_actions
from app.data.market import AlpacaProvider, DataError, synthetic
from app.strategies.base import BuyHold, MovingAverage


def fixture(prices, **kwargs):
    req = BacktestRequest(
        asset="AAPL",
        start=date(2024, 1, 2),
        end=date(2024, 1, 12),
        strategy="buy_hold",
        initial_cash=1000,
        allocation=0.5,
        cash_reserve=0,
        commission_bps=0,
        spread_bps=0,
        slippage_bps=0,
        **kwargs,
    )
    frame = synthetic(req).iloc[: len(prices)].copy()
    for col in ("open", "high", "low", "close"):
        frame[col] = [float(p) for p in prices]
    return req, frame


def action(frame, index, kind="split", **kwargs):
    return {
        "id": f"event-{index}",
        "asset": "AAPL",
        "kind": kind,
        "ex_date": frame.timestamp.iloc[index].date().isoformat(),
        **kwargs,
    }


def test_split_hold_preserves_equity_and_cost_basis():
    req, frame = fixture([100, 100, 50, 60])
    frame.attrs["corporate_actions"] = [action(frame, 2, ratio=2)]
    result = run_engine(frame, req)
    assert [p["quantity"] for p in result["equity"]] == [0, 5, 10, 10]
    assert [p["equity"] for p in result["equity"]] == [1000, 1000, 1000, 1100]
    assert result["metrics"]["unrealized_pnl"] == 100
    assert len(result["trades"]) == 1


class ExitOnSplit:
    def target(self, history):
        return float(len(history) < 2)


def test_outstanding_sell_scales_on_split_and_roundtrip_pnl():
    req, frame = fixture([100, 100, 60])
    frame.attrs["corporate_actions"] = [action(frame, 2, ratio=2)]
    result = run_engine(frame, req, ExitOnSplit())
    assert result["trades"][1]["quantity"] == 10
    assert result["metrics"]["final_cash"] == 1100
    assert result["metrics"]["realized_pnl"] == 100


def test_outstanding_buy_scales_on_split():
    req, frame = fixture([100, 50, 50])
    frame.attrs["corporate_actions"] = [action(frame, 1, ratio=2)]
    result = run_engine(frame, req)
    assert result["trades"][0]["quantity"] == 10
    assert result["metrics"]["final_equity"] == 1000


def test_reverse_split():
    req, frame = fixture([100, 100, 200, 220])
    frame.attrs["corporate_actions"] = [action(frame, 2, ratio=0.5)]
    result = run_engine(frame, req)
    assert result["equity"][-1]["quantity"] == 2.5
    assert result["metrics"]["final_equity"] == 1050


def test_reverse_fraction_cash_in_lieu_rejected():
    req, frame = fixture([100, 100, 200], fractional=False)
    frame.attrs["corporate_actions"] = [action(frame, 2, ratio=0.5)]
    with pytest.raises(DataError, match="fractional shares"):
        run_engine(frame, req)


def test_dividend_receivable_and_later_cash_after_exit():
    req, frame = fixture([100, 100, 98, 98, 98])
    frame.attrs["corporate_actions"] = [
        action(
            frame, 2, kind="cash_dividend", amount=2, payable_date=frame.timestamp.iloc[4].date().isoformat()
        )
    ]
    result = run_engine(frame, req, ExitOnSplit())
    assert result["equity"][2]["quantity"] == 0
    assert result["equity"][2]["cash"] == 990
    assert result["equity"][2]["dividend_receivable"] == 10
    assert result["equity"][2]["equity"] == 1000
    assert result["metrics"]["final_cash"] == 1000
    assert result["metrics"]["dividend_income"] == 10
    assert result["metrics"]["realized_pnl"] == 0


def test_ex_date_buyer_is_not_entitled():
    req, frame = fixture([100, 98, 98])
    frame.attrs["corporate_actions"] = [
        action(
            frame, 1, kind="cash_dividend", amount=2, payable_date=frame.timestamp.iloc[2].date().isoformat()
        )
    ]
    result = run_engine(frame, req)
    assert result["metrics"]["dividend_income"] == 0


def test_dividend_remains_receivable_past_backtest_end():
    req, frame = fixture([100, 100, 98])
    frame.attrs["corporate_actions"] = [
        action(frame, 2, kind="cash_dividend", amount=2, payable_date="2024-02-01")
    ]
    result = run_engine(frame, req)
    assert result["metrics"]["final_equity"] == 1000
    assert result["metrics"]["final_cash"] == 500
    assert result["metrics"]["dividend_receivable"] == 10


def test_split_features_adjust_only_after_effective_date():
    req, frame = fixture([100, 100, 50, 50])
    frame.attrs["corporate_actions"] = [action(frame, 2, ratio=2)]
    prefixes = []

    class Observe:
        def target(self, history):
            assert history.attrs == {}
            prefixes.append(list(history.close))
            return BuyHold().target(history)

    run_engine(frame, req, Observe())
    assert prefixes[0] == [100]
    assert prefixes[1] == [100, 100]
    assert prefixes[2] == [50, 50, 50]
    assert MovingAverage(2, 3).target(pd.DataFrame({"close": prefixes[2]})) == 0


def test_missing_verified_actions_blocks_raw_stock_simulation():
    req, frame = fixture([100, 100, 100])
    req = req.model_copy(update={"provider": "alpaca"})
    with pytest.raises(DataError, match="verified"):
        run_engine(frame, req)


@pytest.mark.parametrize(
    "mutation", ["missing_pay", "negative_amount", "duplicate", "unsupported", "missing_ex"]
)
def test_bad_actions_fail_closed(mutation):
    req, _ = fixture([100, 100, 100])
    event = {
        "id": "x",
        "symbol": "AAPL",
        "ex_date": "2024-01-04",
        "process_date": "2024-01-05",
        "rate": 2,
        "payable_date": "2024-01-08",
        "special": False,
        "foreign": False,
    }
    group = "cash_dividends"
    rows = [event]
    if mutation == "missing_pay":
        event.pop("payable_date")
    if mutation == "negative_amount":
        event["rate"] = -2
    if mutation == "duplicate":
        rows.append(event.copy())
    if mutation == "unsupported":
        group = "stock_dividends"
    if mutation == "missing_ex":
        event.pop("ex_date")
    with pytest.raises(DataError):
        parse_actions({group: rows}, req)


def test_actions_pagination_uses_ex_date_not_late_process_date(monkeypatch):
    original = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        if "page_token" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "corporate_actions": {
                        "forward_splits": [
                            {
                                "id": "split",
                                "symbol": "AAPL",
                                "old_rate": 1,
                                "new_rate": 2,
                                "ex_date": "2024-01-04",
                                "process_date": "2024-03-01",
                            }
                        ]
                    },
                    "next_page_token": "second",
                },
            )
        return httpx.Response(200, json={"corporate_actions": {}})

    monkeypatch.setattr(
        market.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    req, _ = fixture([100, 100, 100])
    actions = AlpacaProvider().load_actions(req)
    assert len(actions) == 1 and actions[0].ratio == 2
    assert len(calls) == 2
    assert calls[0].url.params["start"] == "2015-01-01"
    assert calls[0].url.params["data_quality"] == "all"
    assert "types" not in calls[0].url.params


def test_public_crypto_needs_no_credentials(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(settings, "alpaca_api_key", "")
    monkeypatch.setattr(settings, "alpaca_secret_key", "")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "bars": {
                    "BTC/USD": [
                        {"t": f"2024-01-0{i}T00:00:00Z", "o": 10, "h": 10, "l": 10, "c": 10, "v": 100}
                        for i in (2, 3)
                    ]
                }
            },
        )

    monkeypatch.setattr(
        market.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    req = BacktestRequest(asset="BTC/USD", provider="alpaca", start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert len(AlpacaProvider().load(req)) == 2
