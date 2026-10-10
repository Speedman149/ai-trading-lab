# Verified foundation — 2026-10-09

Executed inside WSL Ubuntu using Python 3.12.3 and Node 22.23.3.

- Backend Pytest: **33 passed**, no failures; one Starlette/httpx deprecation warning.
- Backend Ruff lint and format checks: passed.
- Frontend production build: passed (Next.js 16.4.0).
- TypeScript check: passed.
- Frontend ESLint and Prettier checks: passed.
- npm dependency audit: zero vulnerabilities after replacing a lint-only dependency chain.
- Chromium Playwright: **5 passed**, no failures (AAPL flow, BTC/USD flow, validation error, unavailable backend, mobile overflow).
- Browser tests checked requested inputs, backend/display agreement, fill-row counts, synthetic disclosure, and generated chart paths.
- Full-page AAPL screenshot visually inspected; chart rendering and date input layout verified.
- Documented development startup/shutdown smoke test: backend and frontend returned HTTP 200; both owned server groups stopped afterward.
- CLI AAPL daily and BTC/USD hourly simulations completed using synthetic data.

Initial regression failures were fixed and rerun. They included fixture liquidity/strategy assumptions, cached timezone conversion, browser error selectors, and screenshot-time chart animation. No current test failures remain.

Live Alpaca requests were not verified: credentials were not provided. Pagination, retries, error privacy, missing bars, daily raw equity normalization, and the equity simulation gate were tested with mocked HTTP responses. Real equity simulation remains unavailable until corporate-action accounting is implemented.

This verifies a local demonstration and research foundation. It does not verify ML, news, real-data equity strategy performance, multi-asset portfolios, or paper trading; those features are not implemented.
## Real-data milestone — 2026-10-10

This update supersedes the previous equity-accounting limitation.

- Backend: **50 tests passed**, including 17 new action/public-data tests. One existing Starlette/httpx deprecation warning remains.
- Backend Ruff lint and formatting: passed.
- Frontend build, TypeScript, and ESLint: passed.
- Chromium browser integration: **5 tests passed**.
- Live Alpaca BTC/USD and ETH/USD: 30 real daily bars each, January 2024, successfully simulated through the running API.
- Live AAPL and SPY: 21 real January 2024 bars each, successfully simulated with an empty validated action set.
- Live AAPL and SPY January–June 2024: 124 real bars each and two cash-dividend records each. Earned income and payment/receivable events appeared in results.
- Live AAPL August–September 2020: 36 real bars and a validated four-for-one split. Holdings and pending instructions were adjusted successfully.
- Updated dashboard and backend relaunched and verified reachable from Windows.
- Credentials stayed in local configuration. The tracked template contains no credential values.

Supported stock actions are ordinary cash dividends and forward/reverse splits. Advanced events remain rejected. These checks verify accounting and provider integration, not strategy profitability or exhaustive vendor coverage.
## ML, RL and manual retraining milestone — 2026-10-10

- Full backend suite: **69 passed**. Covers causal features, hand-calculated split/dividend labels and RL fees/reward, horizon-1/5 purge boundaries, disjoint calibration, held-out market isolation, deterministic training, JSON model reload, frozen RL evaluation, jobs, snapshots, refresh, acceptance checks and rollback.
- Ruff lint and formatting pass. Python compilation and the CPU-only dependency installation succeed.
- Next production build, TypeScript, ESLint and Prettier pass.
- Browser suite: **8 passed**, including original stock/crypto simulations, XGBoost job/evaluation agreement, calibration, saved-version reload, RL N/A prediction diagnostics, short-history errors and mobile layout.
- Screenshots visually inspected. Mobile chart overflow found during QA was corrected and verified by the browser suite.
- Final chronological protocol is research-v4: training labels finish before the first evaluation candle opens; validation labels and portfolio marks stop before the held-out market period, including multi-bar horizons.
- Real AAPL 2022-01-03 to 2025-01-01 trained successfully. Final saved version 28595236071eb075408da694 has 139 held-out predictions, MAE about 1.076 percentage points and directional accuracy about 56.1%. The trading rule stayed in cash in that test and failed validation acceptance; this is a reported result, not a model-success claim.
- Manual API retraining refreshed real Alpaca history through the current exclusive end date, preserving the prior version. Version b98efd84a771633b97710fdd uses 1,197 real daily bars and compares 339 forward validation samples against the reference model. It remains ineligible for promotion. Exact provider input/action snapshots and reports are saved locally.
- Research page responds HTTP 200, with both development servers running locally.
- Remaining warnings: existing Starlette/httpx TestClient deprecation and Gymnasium's informational warnings about unbounded observation-space limits. These checks pass; the Q agent discretizes finite observed features.
- Generated datasets/models, API credentials and QA images are excluded from Git. Automatic scheduling, deep PPO, multi-asset research and ongoing paper/live trading are outside this milestone.
