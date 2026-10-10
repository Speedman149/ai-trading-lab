# Real market data and supported corporate actions

Raw prices remain the execution/valuation prices. The adapter requests all corporate-action types (without a types filter), paginates the response, and validates the relevant events. Source: [Alpaca corporate-actions API](https://docs.alpaca.markets/us/reference/corporateactions-1).

The API filters by process_date, which can differ from ex_date. The adapter reads available records from 2015 through download day, then selects economic ex-dates within the requested bar range. Processing/version metadata and the normalized action-content hash are persisted with each dataset. Old caches from before action accounting have a different processing version and cannot silently pass the new verification gate.

Alpaca warns that creation times may be delayed. Successful retrieval/validation means the returned records satisfy this application's accounting contract; it is not proof that the vendor's history is exhaustive. Results include this limitation. The action feed is used for retrospective bookkeeping, not announcement-time prediction.

## Splits

Forward and reverse splits multiply holdings by new_rate / old_rate and divide their average cost per share by the same ratio. Total cost basis is preserved. Outstanding unfilled market orders scale quantity and signal reference price before broker execution; prior-bar participation volume scales to the new share units. Applied events appear in the result log.

MA history is adjusted only after the economic split occurs: earlier prices divide by the effective ratio and volumes multiply by it. Future splits are never used to adjust earlier decision prefixes. Raw fills remain unchanged. A reverse split that creates fractional holdings is rejected if fractional quantities are disabled; cash-in-lieu settlement is not modeled.

## Ordinary cash dividends

Entitlement is captured from holdings before ex-date opening execution. Ex-date buyers receive no entitlement; ex-date sellers retain it. Entitlement is marked as a receivable so the ex-dividend price drop is not falsely reported as an uncompensated loss. Cash is paid at the first available market bar on or after payable_date. It is not spendable before then.

A dividend payable beyond the backtest remains a receivable in final equity. Entitlements survive position exits. Closed-trade net P&L includes attributable earned dividend income, fill-price gains/losses, and commissions. A separate ledger is used because unadjusted Backtrader trade objects do not account for share splits. Total dividend income and unpaid receivables are also reported separately; don't add them again to equity.

No dividend reinvestment, withholding taxes, interest on receivables, or settlement lag beyond payable-date session availability is modeled. Price features remain price-only apart from causal split adjustment.

## Rejection rules

Missing/invalid ex-dates, payable dates, IDs, amounts, split ratios, unexpected symbols, duplicate relevant IDs, non-session ex-dates, incomplete action payloads, unsupported corporate-action types, special/foreign/due-bill dividends, and multiple actions on one ex-date fail closed. Mergers, spin-offs, stock dividends, capital-gain distributions, and cash-in-lieu need additional accounting. Missing bars are never fabricated.

Real stocks initially support daily bars only. Public crypto has no equity corporate-action ledger requirements. All endpoints are read-only market-data calls.

## Verification

Hand-calculated tests cover unchanged value across splits, adjusted basis, pending buy/sell quantities, reverse splits, fractional-settlement rejection, dividend eligibility, payment after selling, unpaid final entitlements, causal feature adjustment, invalid records, action pagination with delayed process_date, and raw-stock metadata rejection.

Live public BTC/USD and ETH/USD daily backtests were verified. After the local account configuration became available, live AAPL/SPY backtests were verified for January–June 2024 dividend periods, and AAPL August–September 2020 for its four-for-one split. Deterministic tests use mocked providers and do not require network access.