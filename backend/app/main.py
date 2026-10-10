import json
import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import ROOT, settings
from app.core.schemas import ASSETS, TIMEFRAMES, BacktestRequest, BacktestResult, EquityPoint, FillRecord
from app.data.market import DataError
from app.experiments.service import execute

app = FastAPI(title="AI Trading Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "research", "live_trading": False}


@app.get("/api/assets")
def assets():
    return [{"symbol": k, "asset_class": v} for k, v in ASSETS.items()]


@app.get("/api/timeframes")
def timeframes():
    return list(TIMEFRAMES)


@app.get("/api/strategies")
def strategies():
    return [
        {"id": "buy_hold", "name": "Buy and hold"},
        {"id": "ma_crossover", "name": "Moving average crossover"},
    ]


@app.get("/api/data/status")
def status():
    return {
        "synthetic_available": True,
        "public_crypto_available": True,
        "alpaca_equity_timeframes": ["1d"],
        "alpaca_configured": bool(settings.alpaca_api_key and settings.alpaca_secret_key),
        "alpaca_equities_enabled": True,
        "cached_datasets": len(list((ROOT / "data" / "processed").glob("*.parquet"))),
    }


@app.post("/api/backtests", response_model=BacktestResult)
def create(req: BacktestRequest):
    try:
        return execute(req)
    except DataError as exc:
        raise HTTPException(422, detail={"code": "data_error", "message": str(exc)}) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Backtest failed")
        raise HTTPException(
            500, detail={"code": "backtest_error", "message": "Backtest failed; see backend logs"}
        ) from exc


def retrieve(identifier):
    if not re.fullmatch(r"[0-9a-f]{24}", identifier):
        raise HTTPException(404, "Unknown experiment")
    path = ROOT / "artifacts" / f"{identifier}.json"
    if not path.exists():
        raise HTTPException(404, "Unknown experiment")
    return json.loads(path.read_text())


@app.get("/api/backtests/{identifier}", response_model=BacktestResult)
def result(identifier: str):
    return retrieve(identifier)


@app.get("/api/backtests/{identifier}/trades", response_model=list[FillRecord])
def trades(identifier: str):
    return retrieve(identifier)["trades"]


@app.get("/api/backtests/{identifier}/equity", response_model=list[EquityPoint])
def equity(identifier: str):
    return retrieve(identifier)["equity"]
