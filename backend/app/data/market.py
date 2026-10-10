"""UTC opening-time bars; strict validation, exchange calendars, no gap filling."""

import hashlib
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import httpx
import numpy as np
import pandas as pd

from app.core.config import ROOT, settings
from app.core.schemas import ASSETS, TIMEFRAMES
from app.data.actions import CorporateAction, parse_actions
from app.data.errors import DataError

VERSION = "synthetic-v2"
PROCESSING_VERSION = "market-v3-actions"


def expected_times(asset, timeframe, start, end):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    minutes = TIMEFRAMES[timeframe]
    if ASSETS[asset] == "crypto":
        return pd.date_range(start, end, freq=f"{minutes}min", inclusive="left")
    schedule = xcals.get_calendar("XNYS").schedule.loc[start.date().isoformat() : end.date().isoformat()]
    points = []
    for _, session in schedule.iterrows():
        opening, closing = session["open"], session["close"]
        points.extend(
            [opening]
            if timeframe == "1d"
            else pd.date_range(opening, closing, freq=f"{minutes}min", inclusive="left")
        )
    return pd.DatetimeIndex([p for p in points if start <= p < end])


def validate(df, asset, timeframe, start=None, end=None):
    required = {"timestamp", "open", "high", "low", "close", "volume", "asset", "complete"}
    if not required.issubset(df.columns) or len(df) < 2:
        raise DataError("Dataset needs at least two bars and all normalized fields")
    ts = pd.to_datetime(df.timestamp)
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise DataError("Timestamps must be timezone-aware UTC")
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise DataError("Duplicate or unsorted timestamps")
    values = df[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any():
        raise DataError("Prices must be finite and positive")
    if (df.volume < 0).any() or (df.high < df[["open", "close", "low"]].max(axis=1)).any():
        raise DataError("Invalid volume or OHLC relationships")
    if (df.low > df[["open", "close", "high"]].min(axis=1)).any():
        raise DataError("Invalid OHLC relationships")
    if not (df.asset == asset).all() or not df.complete.all():
        raise DataError("Mixed symbols or incomplete bars")
    if start is not None:
        expected = expected_times(asset, timeframe, start, end)
        missing = expected.difference(pd.DatetimeIndex(ts))
        extra = pd.DatetimeIndex(ts).difference(expected)
        if len(missing) or len(extra):
            raise DataError(
                f"Calendar mismatch: {len(missing)} missing, {len(extra)} unexpected bars; no filling"
            )
    return df


def synthetic(req):
    ts = expected_times(req.asset, req.timeframe, req.start, req.end)
    if len(ts) > 100000:
        raise DataError("MVP experiments are limited to 100,000 bars; select a shorter range")
    t = ts.as_unit("ns").asi8 / 1e9 / 86400
    base = 180 if ASSETS[req.asset] == "stock" else 40000
    phase = int(hashlib.sha256(req.asset.encode()).hexdigest()[:6], 16) % 100
    opening = base * np.exp(0.0003 * (t - 19723) + 0.07 * np.sin(t / 13 + phase))
    close = opening * (1 + 0.004 * np.sin(t * 5 + phase))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opening,
            "close": close,
            "high": np.maximum(opening, close) * 1.008,
            "low": np.minimum(opening, close) * 0.992,
            "volume": np.full(len(ts), 1000000 if ASSETS[req.asset] == "stock" else 1000),
            "asset": req.asset,
            "asset_class": ASSETS[req.asset],
            "provider": "synthetic",
            "market": "XNYS" if ASSETS[req.asset] == "stock" else "24/7",
            "timeframe": req.timeframe,
            "adjustment": "synthetic-no-corporate-actions",
            "complete": True,
        }
    )
    return validate(df, req.asset, req.timeframe, req.start, req.end)


class AlpacaProvider:
    def load(self, req):
        crypto = ASSETS[req.asset] == "crypto"
        if not crypto and (not settings.alpaca_api_key or not settings.alpaca_secret_key):
            raise DataError(
                "Stock data needs an Alpaca account. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to the local .env file, then restart the backend. Bitcoin/Ethereum history works without keys."
            )
        if not crypto and req.timeframe != "1d":
            raise DataError(
                "Real equity ingestion initially supports daily bars; intraday session aggregation is pending"
            )
        url = (
            "https://data.alpaca.markets/v1beta3/crypto/us/bars"
            if crypto
            else "https://data.alpaca.markets/v2/stocks/bars"
        )
        params = {
            "symbols": req.asset,
            "timeframe": {"1d": "1Day", "1h": "1Hour", "15m": "15Min", "5m": "5Min"}[req.timeframe],
            "start": f"{req.start}T00:00:00Z",
            "end": f"{req.end}T00:00:00Z",
            "limit": 10000,
            "sort": "asc",
        }
        if not crypto:
            params.update(adjustment="raw", feed=settings.alpaca_feed)
        rows = []
        with httpx.Client(
            timeout=30,
            headers={
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
            },
        ) as client:
            for _ in range(100):
                payload = self.request(client, url, params)
                rows.extend(payload.get("bars", {}).get(req.asset, []))
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
            else:
                raise DataError("Provider pagination limit exceeded")
        if not rows:
            raise DataError("Alpaca returned no bars")
        df = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
        expected = expected_times(req.asset, req.timeframe, req.start, req.end)
        if not crypto:
            by_date = {p.tz_convert("America/New_York").date(): p for p in expected}
            df["timestamp"] = df.timestamp.map(
                lambda p: by_date.get(p.tz_convert("America/New_York").date(), pd.NaT)
            )
        df = df[df.timestamp.isin(expected)].copy()
        df["complete"] = closing_times(df, req.asset, req.timeframe) <= pd.Timestamp.now(tz="UTC")
        df["asset"], df["asset_class"], df["provider"] = req.asset, ASSETS[req.asset], "alpaca"
        df["market"], df["timeframe"], df["adjustment"] = "us" if crypto else "XNYS", req.timeframe, "raw"
        actions = [] if crypto else self.load_actions(req)
        df.attrs["corporate_actions"] = [action.model_dump(mode="json") for action in actions]
        df["corporate_actions_verified"] = True
        return validate(df, req.asset, req.timeframe, req.start, req.end)

    def request(self, client, url, params):
        for attempt in range(4):
            try:
                response = client.get(url, params=params)
            except httpx.RequestError:
                if attempt == 3:
                    raise DataError("Alpaca network failure; try again later") from None
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise DataError("Alpaca unavailable or rate limited; try again later")
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise DataError(
                    f"Alpaca request failed (HTTP {response.status_code}); check account credentials and data access"
                )
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError()
                return payload
            except (ValueError, TypeError):
                raise DataError("Alpaca returned invalid JSON") from None
        raise DataError("Alpaca request did not complete")

    def load_actions(self, req):
        # API filters process_date, not ex_date. Read all available action history through today,
        # then filter economic ex-dates locally, including delayed processing outside the bar range.
        params = {
            "symbols": req.asset,
            "start": "2015-01-01",
            "end": datetime.now(UTC).date().isoformat(),
            "limit": 1000,
            "sort": "asc",
            "data_quality": "all",
        }
        groups = {}
        with httpx.Client(
            timeout=30,
            headers={
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
            },
        ) as client:
            for _ in range(100):
                payload = self.request(client, "https://data.alpaca.markets/v1/corporate-actions", params)
                page = payload.get("corporate_actions")
                if not isinstance(page, dict):
                    raise DataError("Alpaca did not provide verifiable corporate-action records")
                for group, rows in page.items():
                    if not isinstance(rows, list):
                        raise DataError("Malformed corporate-action records")
                    groups.setdefault(group, []).extend(rows)
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
            else:
                raise DataError("Corporate-action pagination limit exceeded")
        actions = parse_actions(groups, req)
        sessions = {
            timestamp.tz_convert("America/New_York").date()
            for timestamp in expected_times(req.asset, req.timeframe, req.start, req.end)
        }
        if any(action.ex_date not in sessions for action in actions):
            raise DataError("Corporate-action ex-date does not match an exchange session")
        return actions


def load(req, refresh=False):
    if len(expected_times(req.asset, req.timeframe, req.start, req.end)) > 100000:
        raise DataError("MVP experiments are limited to 100,000 bars; select a shorter range")
    key_fields = {
        k: req.model_dump(mode="json")[k] for k in ("asset", "timeframe", "provider", "start", "end")
    }
    key_fields.update(version=VERSION, feed=settings.alpaca_feed, processing_version=PROCESSING_VERSION)
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()
    folder = ROOT / "data" / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.parquet"
    if path.exists() and not refresh:
        with duckdb.connect() as conn:
            conn.execute("SET TimeZone = 'UTC'")
            df = conn.execute("SELECT * FROM read_parquet(?) ORDER BY timestamp", [str(path)]).df()
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True).astype("datetime64[ns, UTC]")
        validate(df, req.asset, req.timeframe, req.start, req.end)
        try:
            metadata = json.loads(path.with_suffix(".json").read_text())
            actions = [CorporateAction.model_validate(action) for action in metadata["corporate_actions"]]
        except (OSError, ValueError, KeyError):
            raise DataError(
                "Cached action metadata is missing or invalid; remove this cache entry to download again"
            ) from None
        df.attrs["corporate_actions"] = [action.model_dump(mode="json") for action in actions]
    else:
        df = synthetic(req) if req.provider == "synthetic" else AlpacaProvider().load(req)
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        df.to_parquet(temporary, index=False)
        temporary.replace(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    **key_fields,
                    "downloaded_at": datetime.now(UTC).isoformat(),
                    "rows": len(df),
                    "corporate_actions": df.attrs.get("corporate_actions", []),
                }
            )
        )
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True).astype("datetime64[ns, UTC]")
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    action_json = json.dumps(df.attrs.get("corporate_actions", []), sort_keys=True)
    action_hash = hashlib.sha256(action_json.encode()).hexdigest()
    return df, {
        **key_fields,
        "dataset_hash": digest,
        "rows": len(df),
        "corporate_actions_hash": action_hash,
        "corporate_actions_count": len(df.attrs.get("corporate_actions", [])),
        "synthetic": req.provider == "synthetic",
    }


def closing_times(frame, asset, timeframe):
    opening = pd.DatetimeIndex(frame.timestamp)
    if ASSETS[asset] == "crypto":
        return opening + pd.Timedelta(minutes=TIMEFRAMES[timeframe])
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule
    closes = {idx.date(): row["close"] for idx, row in schedule.iterrows()}
    result = []
    for timestamp in opening:
        close = closes[timestamp.tz_convert("America/New_York").date()]
        result.append(
            close
            if timeframe == "1d"
            else min(close, timestamp + pd.Timedelta(minutes=TIMEFRAMES[timeframe]))
        )
    return pd.DatetimeIndex(result)
