from datetime import date

import httpx
import pytest

from app.core.config import settings
from app.core.schemas import BacktestRequest
from app.data import market
from app.data.market import AlpacaProvider, DataError
from app.experiments.service import execute


def configure(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(settings, "alpaca_api_key", "test-key")
    monkeypatch.setattr(settings, "alpaca_secret_key", "test-secret")
    monkeypatch.setattr(
        market.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    monkeypatch.setattr(market.time, "sleep", lambda seconds: None)


def row(timestamp):
    return {"t": timestamp, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}


def test_crypto_pagination(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        if "page_token" not in req.url.params:
            return httpx.Response(
                200, json={"bars": {"BTC/USD": [row("2024-01-02T00:00:00Z")]}, "next_page_token": "second"}
            )
        return httpx.Response(
            200, json={"bars": {"BTC/USD": [row("2024-01-03T00:00:00Z")]}, "next_page_token": None}
        )

    configure(monkeypatch, handler)
    req = BacktestRequest(asset="BTC/USD", provider="alpaca", start=date(2024, 1, 2), end=date(2024, 1, 4))
    data = AlpacaProvider().load(req)
    assert len(data) == 2 and str(data.timestamp.dt.tz) == "UTC"
    assert len(calls) == 2
    assert calls[0].headers["APCA-API-KEY-ID"] == "test-key"
    assert all(call.method == "GET" for call in calls)


def test_equity_raw_ingestion_and_simulation_gate(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(
            200, json={"bars": {"AAPL": [row("2024-01-02T05:00:00Z"), row("2024-01-03T05:00:00Z")]}}
        )

    configure(monkeypatch, handler)
    req = BacktestRequest(asset="AAPL", provider="alpaca", start=date(2024, 1, 2), end=date(2024, 1, 4))
    data = AlpacaProvider().load(req)
    assert calls[0].url.path == "/v2/stocks/bars"
    assert calls[0].url.params["adjustment"] == "raw"
    assert data.timestamp.iloc[0].hour == 14
    assert not data.corporate_actions_verified.any()
    with pytest.raises(DataError, match="split/dividend"):
        execute(req)


@pytest.mark.parametrize("status", [429, 503, 401])
def test_provider_failures_and_retries(monkeypatch, status):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(status, json={"secret": "must never appear in errors"})

    configure(monkeypatch, handler)
    with pytest.raises(DataError) as error:
        AlpacaProvider().load(BacktestRequest(asset="BTC/USD", provider="alpaca"))
    assert "secret" not in str(error.value)
    assert len(calls) == (1 if status == 401 else 4)


def test_missing_provider_bars_rejected(monkeypatch):
    def handler(req):
        return httpx.Response(
            200, json={"bars": {"BTC/USD": [row("2024-01-02T00:00:00Z"), row("2024-01-04T00:00:00Z")]}}
        )

    configure(monkeypatch, handler)
    req = BacktestRequest(asset="BTC/USD", provider="alpaca", start=date(2024, 1, 2), end=date(2024, 1, 5))
    with pytest.raises(DataError, match="missing"):
        AlpacaProvider().load(req)


def test_no_credentials_explicit_error(monkeypatch):
    monkeypatch.setattr(settings, "alpaca_api_key", "")
    monkeypatch.setattr(settings, "alpaca_secret_key", "")
    with pytest.raises(DataError, match="ALPACA_API_KEY"):
        AlpacaProvider().load(BacktestRequest(asset="BTC/USD", provider="alpaca"))
