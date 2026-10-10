import json
import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import ROOT, settings
from app.core.schemas import ASSETS, TIMEFRAMES, BacktestRequest, BacktestResult, EquityPoint, FillRecord
from app.data.market import DataError
from app.experiments.service import execute

app = FastAPI(title="AI Trading Lab", version="0.2.0")
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


# Model research is deliberately separate from conventional backtests.
from app.ml import jobs, storage
from app.ml.schemas import ResearchJob, ResearchRequest, ResearchResult, SearchRequest, SearchResult

jobs.initialize()


@app.post("/api/research/jobs", response_model=ResearchJob, status_code=202)
def train_model(req: ResearchRequest):
    try:
        return jobs.submit(req)
    except jobs.ResearchBusy as exc:
        raise HTTPException(409, detail={"code": "research_busy", "message": str(exc)}) from exc


@app.get("/api/research/jobs", response_model=list[ResearchJob])
def research_jobs():
    with storage.LOCK:
        return sorted([j.copy() for j in jobs.JOBS.values()], key=lambda j: j["created_at"], reverse=True)[
            :20
        ]


@app.get("/api/research/jobs/{identifier}", response_model=ResearchJob)
def research_job(identifier: str):
    try:
        return jobs.retrieve(identifier)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Unknown training job") from exc


@app.get("/api/research/versions")
def model_versions():
    return storage.versions()


@app.get("/api/research/versions/{identifier}", response_model=ResearchResult)
def model_result(identifier: str):
    try:
        return storage.result(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown model version") from exc


@app.post("/api/research/versions/{identifier}/retrain", response_model=ResearchJob, status_code=202)
def retrain_model(identifier: str):
    from datetime import UTC, datetime, timedelta

    try:
        saved = storage.result(identifier)
        req = ResearchRequest.model_validate(saved["config"])
        end = datetime.now(UTC).date()
        start = max(req.backtest.start, end - timedelta(days=3650))
        req = req.model_copy(
            update={
                "parent_id": identifier,
                "refresh_data": True,
                "backtest": req.backtest.model_copy(update={"start": start, "end": end}),
            }
        )
        req = ResearchRequest.model_validate(req.model_dump())
        return jobs.submit(req)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown model version") from exc
    except jobs.ResearchBusy as exc:
        raise HTTPException(409, detail={"code": "research_busy", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(422, detail={"code": "research_error", "message": str(exc)}) from exc


@app.post("/api/research/versions/{identifier}/promote")
def promote_model(identifier: str):
    try:
        return storage.promote(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown model version") from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"code": "promotion_rejected", "message": str(exc)}) from exc


@app.post("/api/research/versions/{identifier}/rollback")
def rollback_model(identifier: str):
    try:
        return storage.rollback(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown model version") from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"code": "rollback_rejected", "message": str(exc)}) from exc


@app.post("/api/research/searches", response_model=ResearchJob, status_code=202)
def search_parameters(req: SearchRequest):
    try:
        return jobs.submit_search(req)
    except jobs.ResearchBusy as exc:
        raise HTTPException(409, detail={"code": "research_busy", "message": str(exc)}) from exc


@app.get("/api/research/searches")
def parameter_searches():
    return storage.searches()


@app.get("/api/research/searches/{identifier}", response_model=SearchResult)
def parameter_search_result(identifier: str):
    try:
        return storage.search_result(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown parameter search") from exc
