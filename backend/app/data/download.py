"""Read-only Alpaca ingestion; raw equity bars are not cleared for simulation."""

import argparse
import json

from app.core.schemas import BacktestRequest
from app.data.market import load

parser = argparse.ArgumentParser()
parser.add_argument("--asset", default="AAPL")
parser.add_argument("--start", default="2024-01-02")
parser.add_argument("--end", default="2024-06-30")
parser.add_argument("--timeframe", default="1d")
args = parser.parse_args()
_, metadata = load(BacktestRequest(provider="alpaca", **vars(args)))
print(json.dumps(metadata, indent=2))
