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

## Parameter recovery, retraining explanations and expanded comparisons — 2026-10-10

- Backend: **83 tests passed**, with hand-calculated new feature values, causal prefix invariance, deterministic sampled models and model reload, cost-bearing probability-only execution, activity counts/exposure, both retrain date modes, same-period reproduction, read-only legacy annotations, and strategy-specific predictive acceptance.
- Ruff lint/format, Next production build, TypeScript, ESLint and Prettier passed.
- Browser integration: **12 tests passed** on the production build, including all previous workflows plus JSON settings export, restore of complete model configuration, automatic winner-to-form propagation and identical saved-date rerun results.
- Real-data verification: 24 validation-only AAPL trials on the exact cached 2022-01-03 to 2025-01-01 input snapshot used by the existing saved search. The selected expanded-technical configuration returned **4.9568%** on validation versus **8.3758%** for buy-and-hold; eight completed trades, 2.5933% maximum drawdown, 0.9271 Sharpe and $210.40 modeled costs. The frozen final model stayed in cash (0%) versus 18.7950% for buy-and-hold. It failed net-baseline and return-MAE acceptance checks. It was not promoted. These results verify the workflow and show no demonstrated profitable advantage.
- Desktop/mobile screenshots of the selected real parameters, activity and legacy retrain comparison were inspected. The user's earlier 2026-10-10 retrain correctly restores its 240-tree maximum and Alpaca settings and explains that its history/test windows differ from the parent ending 2025-01-01. No horizontal mobile overflow or browser script errors.
- Exact code identity advances to research-v6.1 after finalizing probability-only acceptance, avoiding reuse of earlier development artifacts with a different acceptance policy. Stored historical reports and input snapshots remain untouched.
- Generated research files/screenshots remain local and ignored by Git. No live or paper order-placement functionality was enabled.

A final development-server run exposed a pre-hydration date-edit race: the browser could accept a native input before React attached its handler, then restore the default start date. The form now stays disabled until hydration finishes. A delayed-script regression checks that the exact entered start/end reach the backend and receive the expected short-history error. This adds a twelfth browser test.

The hydration correction also passed all seven research checks against the development server. Completed and reopened results scroll to their saved parameter panel; applying a saved configuration returns to the form.

After adding result navigation, the final production build, lint, TypeScript, Prettier and all four focused search/retrain/hydration browser checks passed.


## Research v7: precise search and low-return diagnosis

Final checks: 89 backend tests, Ruff lint/format, production frontend build, TypeScript, ESLint, Prettier and 15 production Playwright tests passed. Search regressions cover all three modes, validation-only frozen refinement centers, bounded unique neighbors, and unchanged ranking when final-test prices are altered. Hand-calculated paired-gate coverage confirms a 0.24% forecast only trades when both the return hurdle and probability checks pass, with actual 0.1% per-side commissions still charged. Browser coverage includes fractional restored parameters reaching the backend, winning-setting recovery, error recovery and mobile layout.

A predefined 32-trial local search used the exact saved AAPL Alpaca dataset/actions snapshot from the previous 4.9568% validation run. Trial 1 reproduced 4.9568% exactly. The selected trial 5 changed only the extra return hurdle from 5 to 7.5 bps: validation net 9.8000% versus buy-and-hold 8.3758%, drawdown 2.4086%, Sharpe 1.7203, seven completed trades and $186.21 modeled costs. Only one of three folds was profitable. It still failed the predictive-MAE baseline check. The selected final frozen model used two return trees and one direction tree; no final-test forecast met both entry gates, giving 0% versus buy-and-hold 18.7950%. No candidate was promoted. These previously inspected dates are exploratory, not a fresh independent test.

Saved search 005698b46ac4487ba2a2f615 and model f0028d0a7935a4b70f70901a remain local ignored artifacts. The real saved report was checked on desktop and at 390px width, with no page errors or horizontal page overflow. Restoring its fractional 7.5 bps hurdle passes native form validation and preserves the 10 bps commission assumption. The source comparison is not rewritten.
