import hashlib
import json
import tempfile

from app.backtesting.engine import run_engine
from app.core.config import ROOT
from app.core.schemas import BacktestRequest
from app.data.market import DataError, load

CODE_VERSION = "backtest-v2"


def execute(req: BacktestRequest):
    if req.provider == "alpaca" and req.asset in ("AAPL", "SPY"):
        raise DataError(
            "Real equity backtests require verified split/dividend accounting, which is pending. Daily raw equity ingestion is available through app.data.download."
        )
    frame, provenance = load(req)
    config = req.model_dump(mode="json")
    identity = {"config": config, "provenance": provenance, "code_version": CODE_VERSION}
    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    result = {
        "id": identifier,
        **identity,
        **run_engine(frame, req),
        "benchmark": run_engine(frame, req.model_copy(update={"strategy": "buy_hold"})),
        "warnings": ["SYNTHETIC demonstration data: no historical investment conclusions."]
        if provenance["synthetic"]
        else [],
        "assumptions": [
            "Completed-bar signals fill at next available open; final positions marked to close, not liquidated.",
            "Modeled half-spread plus additional slippage, not measured quotes.",
            "Participation cap uses previous completed volume; unfilled remainder expires after one bar.",
            "Buy-and-hold enters after first completed bar with the same costs and constraints.",
            "Gross return adds costs back to the same fills; it is not a separate zero-cost strategy.",
            "Daily-close risk metrics; 252 equity sessions / 365 crypto days; zero risk-free rate.",
            "Single long-only asset; allocation/reserve limits apply at entry, exposure can drift thereafter.",
        ],
    }
    serialized = json.dumps(result, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, suffix=".tmp", delete=False) as handle:
        handle.write(serialized)
        temporary = handle.name
    from pathlib import Path

    Path(temporary).replace(folder / f"{identifier}.json")
    return result
