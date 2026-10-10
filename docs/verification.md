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