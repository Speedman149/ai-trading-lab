# AI Trading Lab

Local trading research software: Python/FastAPI calculations, Backtrader execution, and a Next.js/React/TypeScript dashboard. Parquet caches and DuckDB queries keep data local. No brokerage order-placement code is included.

## Run in WSL

The development environment is already installed in this checkout. Use two WSL terminals:

```bash
cd ~/projects/ai-trading-lab
source .venv/bin/activate
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd ~/projects/ai-trading-lab
source ~/.nvm/nvm.sh
nvm use 22
cd frontend
npm run dev
```

Open **http://localhost:3000**. API documentation: **http://localhost:8000/docs**.

Alternatively, `bash scripts/dev.sh` starts both servers in one terminal and stops both on Ctrl+C.

For a fresh checkout: Python 3.11+, Node 22+, `python3 -m venv .venv`, `.venv/bin/pip install -r backend/requirements.lock`, then `cd frontend && npm ci`. Ubuntu may need `sudo apt install python3-venv`. This machine lacked ensurepip, so its isolated environment was bootstrapped with the official pip installer; no global Python packages were modified.

## What works

- Credential-free, explicitly **synthetic** AAPL, SPY, BTC/USD, and ETH/USD data.
- Daily, hourly, 15-minute, and 5-minute bars. Stock sessions include holidays, early closes, and daylight saving changes; crypto runs 24/7.
- Buy-and-hold and moving-average crossover with configurable windows.
- Long-only simulation, fractional or integer units, entry allocation/cash reserve, previous-bar volume participation, minimum notional, commissions, fixed fees, full spread, and additional slippage.
- Equity, drawdown, exposure, cash, fills, closed-trade P&L, modeled costs, return/risk metrics, and an identical-assumptions buy-and-hold benchmark.
- Responsive dashboard, validated configuration, loading/empty/error states, and backend-derived charts and tables.
- Deterministic experiment IDs, dataset hashes, strategy/service versions, and local result persistence.
- Alpaca read-only stock/crypto data adapter with pagination, timeouts, limited retry/backoff, and strict missing-bar detection. See the restriction below.

## Run without the website

```bash
cd ~/projects/ai-trading-lab/backend
../.venv/bin/python -m app.cli --asset AAPL
../.venv/bin/python -m app.cli --asset BTC/USD --timeframe 1h
```

## Alpaca configuration

Copy `.env.example` to `.env` in the repository root and set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Restart the backend. Never place these values in frontend variables. Choose Alpaca and a crypto asset in the dashboard. Crypto historical data works even without these keys. No automatic fallback disguises a provider error as real data.

Daily raw equity data and validated corporate-action records can be ingested with:

```bash
cd backend
../.venv/bin/python -m app.data.download --asset AAPL --start 2024-01-02 --end 2024-06-30
```

**Daily real equity backtests now support forward/reverse splits and ordinary USD cash dividends**, provided the corporate-action endpoint is accessible and its records validate. Adjusted prices are not used as executable prices. Unsupported actions and missing action metadata fail explicitly. Real equity intraday ingestion awaits session-aligned aggregation. Stock integration is tested with mocked responses and live AAPL/SPY requests using the local account configuration. Live AAPL split and AAPL/SPY dividend periods were verified. Public Bitcoin/Ethereum history was downloaded and simulated without requiring keys. Crypto gaps fail explicitly rather than inventing missing prices. Feed entitlements may affect availability. `ALPACA_FEED=iex` configures daily stock ingestion.

The official [Alpaca historical stock bars reference](https://docs.alpaca.markets/us/reference/stockbars) documents the raw data endpoint. The frontend uses Node 22, which satisfies the [Next.js 16 runtime requirements](https://nextjs.org/docs/app/guides/upgrading/version-16).

## Verify

```bash
cd ~/projects/ai-trading-lab/backend
../.venv/bin/pytest -q
../.venv/bin/ruff check app tests
../.venv/bin/ruff format --check app tests
cd ../frontend
source ~/.nvm/nvm.sh
nvm use 22
npm run build
npm run typecheck
npm run lint
npm run format:check
npx playwright install chromium
npm run test:e2e
```

The browser tests start the production frontend and backend themselves; build first. Tests cover stock and crypto browser backtests, settings propagation, exact display/backend agreement, actual chart paths, API errors, connection failure, and mobile overflow. Backend tests include hand-calculated fills, fees and P&L, next-open execution, UTC cache round trips, calendar rules, invalid data, liquidity, deterministic results, and provider failure handling. Browser screenshots are saved under ignored `artifacts/`.

## API

`GET /health`, `/api/assets`, `/api/timeframes`, `/api/strategies`, `/api/data/status`; `POST /api/backtests`; `GET /api/backtests/{id}` and its `/trades` and `/equity` subroutes. End dates are exclusive. Request/response schemas are documented by FastAPI. Unknown experiment identifiers cannot access arbitrary filesystem paths.

## Main source files

- `backend/app/data/market.py`: calendars, synthetic generation, Alpaca ingestion, validation, and caching.
- `backend/app/strategies/base.py`: provider/engine-independent strategies.
- `backend/app/backtesting/engine.py`: Backtrader wrapper and portfolio constraints.
- `backend/app/backtesting/metrics.py`: performance accounting and statistics.
- `backend/app/experiments/service.py`: benchmark comparisons and reproducible artifacts.
- `backend/app/core/schemas.py`, `backend/app/main.py`: validated API contract and routes.
- `backend/tests/`: deterministic backend regression tests.
- `frontend/app/page.tsx`, `globals.css`: integrated research interface.
- `frontend/tests/research.spec.ts`: browser integration tests.

Read [research methodology](docs/methodology.md) for exact simulation assumptions and [roadmap](docs/roadmap.md) for remaining work. This is a foundation, not a completed AI trading system. The next milestone is credentialed real equity verification and session-aligned intraday aggregation, before machine learning.
Verification evidence is recorded in [docs/verification.md](docs/verification.md).

## Connect real market data

**Try real crypto now:** select Alpaca historical data, BTC/USD or ETH/USD, and Run Backtest. No account or API keys are needed for this public endpoint.

**Stocks/ETFs:** create an account at https://app.alpaca.markets/signup, then open its dashboard and generate API keys. Use the paper environment credentials for research. Follow the [official account/key guide](https://docs.alpaca.markets/us/docs/getting-started-with-alpaca-market-data). This application's adapter only calls data endpoints, never order-placement endpoints.

The repository root has a local .env template. Open it in VS Code and fill these values locally:

```dotenv
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_FEED=iex
```

Do not send keys in chat or commit this file. After saving, restart with bash scripts/dev.sh, select Alpaca, AAPL/SPY, and **1d**. Access to corporate actions is required too; if your account lacks access, the app reports that error instead of presenting an unchecked stock result. Provider entitlement issues cannot be solved by bypassing action validation.

See [docs/corporate-actions.md](docs/corporate-actions.md) for the supported event rules and limitations.