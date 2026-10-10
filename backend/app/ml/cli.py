import argparse
import json
from datetime import date

from app.core.schemas import BacktestRequest
from app.ml.schemas import ResearchRequest
from app.ml.service import execute


def main():
    parser = argparse.ArgumentParser(description="Chronological daily model research")
    parser.add_argument("--asset", default="AAPL")
    parser.add_argument("--provider", default="synthetic", choices=["synthetic", "alpaca"])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 3))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--algorithm", default="xgboost", choices=["xgboost", "q_learning"])
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    request = ResearchRequest(
        backtest=BacktestRequest(asset=args.asset, provider=args.provider, start=args.start, end=args.end),
        algorithm=args.algorithm,
        episodes=args.episodes,
    )
    result = execute(request, progress=print)
    print(
        json.dumps(
            {
                "id": result["id"],
                "prediction": result["holdout"]["prediction"],
                "strategy": result["holdout"]["strategy"]["metrics"],
                "promotion": result["promotion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
