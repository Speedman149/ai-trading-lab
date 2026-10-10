# Development roadmap

Foundation: FastAPI, Next.js, dependency locks, data validation/storage, deterministic samples, Backtrader wrapper, two strategies, cost/risk accounting, metrics, API, and browser workflow are implemented.

The first demo milestone supports stocks and crypto without credentials. Daily real equity accounting now supports splits and ordinary cash dividends, with validated metadata and rejection of unsupported actions. Session-aligned real intraday equity ingestion is pending. Public crypto and credentialed AAPL/SPY stock/action requests have been verified live, including dividend and split periods.

Next milestone:
1. Expand real-data verification to additional supported periods and revision scenarios.
2. Verify live split/dividend metadata and expand support for advanced actions and provider revisions.
3. Aggregate minute equity data into session-aligned hourly bars and validate late revisions/completion status.
4. Add experiment browsing and explicit cache refresh, then measure backtest runtime limits.

Only after those checks: supervised XGBoost features/labels, chronological splits with purging, expanding walk-forward evaluation, a held-out final period, and cost-aware decisions. Do not install heavy ML/NLP dependencies yet.

Later: multi-asset accounting, news timestamps, manual retraining/version promotion, optional RL, and simulated paper trading. Real-money execution needs new explicit authorization.