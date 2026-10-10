import argparse
import json

from app.core.schemas import BacktestRequest
from app.experiments.service import execute

parser = argparse.ArgumentParser(description="Credential-free research demonstration")
parser.add_argument("--asset", default="AAPL")
parser.add_argument("--timeframe", default="1d")
args = parser.parse_args()
result = execute(BacktestRequest(asset=args.asset, timeframe=args.timeframe))
print(
    json.dumps(
        {"id": result["id"], "provenance": result["provenance"], "metrics": result["metrics"]}, indent=2
    )
)
