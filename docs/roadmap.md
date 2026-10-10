# Development roadmap

Foundation: FastAPI, Next.js, dependency locks, data validation/storage, deterministic samples, Backtrader wrapper, two strategies, cost/risk accounting, metrics, API, and browser workflow are implemented.

The first demo milestone supports stocks and crypto without credentials. Phase 1 remains incomplete for full real equity research: corporate actions and session-aligned real intraday equity ingestion are pending. Alpaca integration is verified with mocks, not a credentialed live request.

Next milestone:
1. Verify read-only Alpaca requests with user-supplied credentials and record real dataset provenance.
2. Add complete split/dividend metadata and portfolio adjustments with hand-calculated fixtures; then enable real equity simulation.
3. Aggregate minute equity data into session-aligned hourly bars and validate late revisions/completion status.
4. Add experiment browsing and explicit cache refresh, then measure backtest runtime limits.

Only after those checks: supervised XGBoost features/labels, chronological splits with purging, expanding walk-forward evaluation, a held-out final period, and cost-aware decisions. Do not install heavy ML/NLP dependencies yet.

Later: multi-asset accounting, news timestamps, manual retraining/version promotion, optional RL, and simulated paper trading. Real-money execution needs new explicit authorization.