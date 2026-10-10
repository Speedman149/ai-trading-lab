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