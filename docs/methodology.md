# Research methodology and simulation contract

## Data

All normalized bars identify asset, asset class, provider, market, timeframe, UTC opening timestamp, OHLCV, adjustment metadata, and completion status. No missing price observations are fabricated. Duplicates, ordering failures, invalid/nonfinite prices or volumes, incomplete bars, wrong symbols, and unexpected calendar gaps fail validation.

Demo data is a deterministic mathematical function of UTC time and asset identity, versioned as synthetic-v2. It is not observed market data and has no predictive evidence. Overlapping sample ranges share identical prices. XNYS calendars model US equity sessions; all sample stocks/ETFs use that shared regular-session calendar. Equity hourly samples have a shortened final session bar, including early closes. Crypto uses continuous UTC intervals. Sample volume is simplified. Synthetic data has no corporate actions; real raw equities are simulated only with verified supported-action metadata; see corporate-actions.md.

Caches store Parquet plus JSON source/range/download metadata. DuckDB queries set UTC explicitly, and returned timestamps normalize to nanosecond UTC for stable hashing. Provider/feed/range/timeframe/generator version identify cache entries. Data content hashes identify research results. Invalid cached data is rejected. To refresh historical provider revisions, remove the specific cache entry (a refresh UI is pending).

## Decision and execution timing

Strategies receive only a history prefix ending with the current completed candle. Decisions become available at its closing time. Market orders execute at the next available candle open, with zero modeled processing delay at adjacent bar boundaries. Backtrader's cheat-on-close and cheat-on-open are not enabled. Overnight/weekend gaps use the next actual session. Equity values and drawdowns are recorded at bar close; order logs use fill/open time.

Buy-and-hold observes the first candle then buys on the second. MA crossover waits for its slow window; it holds while the fast average exceeds the slow average and exits otherwise. Strategies buy once per entry and retain their quantity while invested. No continuous target-weight rebalancing is implied. Splits update holdings and pending instructions before ex-date open; ordinary dividends accrue to overnight holders as receivables and become cash on payable date. Final holdings remain marked at the final close, with no artificial end-of-test liquidation or closing commission. A signal on the last bar cannot fill.

## Costs and liquidity

Commission is `abs(quantity) * fill_price * commission_bps / 10000 + fixed_fee` per nonzero execution. The full spread input contributes half-spread per side. Additional slippage contributes separate basis points. Buys pay an increased price; sells receive a decreased price. These inputs are modeled assumptions, not measured historical quotes or order-book evidence. Fills may exceed candle high/low because costs are modeled on top of the open. Fixed fees are charged for completed fill slices, not rejected orders.

Quantity is estimated using signal-time close and capped at execution by remaining cash, entry allocation, reserve, and **previous completed bar's volume** times maximum participation. The future bar's total volume is never read for sizing. A partial fill's remainder expires after its first executable bar. An incomplete exit may continue with a fresh order after the next close, incurring another fee. The strategy does not top up partial entries. This is a conservative simplified liquidity model, not a reconstruction of actual liquidity.

## Portfolio and metrics

Single asset, long-only, no leverage. Entry allocation/reserve apply at purchase; exposure may drift with prices, and risk limits are not continuously rebalanced. Negative cash is rejected by the broker and checked after simulation. Currency is USD. Fractional units are optional for both asset classes.

Net return uses marked portfolio equity including fees. Gross same-fills return adds cumulative commission and modeled spread/slippage costs back to final equity: a diagnostic, not a distinct zero-cost backtest (costs can change quantities). Closed round-trip P&L includes execution prices, commissions, and earned dividend income; final open positions have no closed-trade outcome. Unrealized P&L is price-only and does not separately deduct entry commissions already reflected in cash. Transaction costs are logged on every fill, including open positions.

Drawdown uses an initial-capital floor for the running peak. Turnover is total absolute executed notional divided by initial capital. Daily-close returns underpin volatility, Sharpe, and Sortino, using 252 equity sessions or 365 crypto days. Zero risk-free rate. Intraday results aggregate to daily closes; missing days are not filled. Sharpe needs at least two nonconstant daily returns; Sortino uses downside root mean square relative to zero. Undefined metrics are null. Annualized return uses elapsed calendar time and is suppressed for ranges shorter than 30 days. It remains particularly misleading on synthetic or short data and is not evidence of profitability.

## Reproducibility and limitations

Every saved experiment includes inputs, dataset provenance/content hash, code version, modeled assumptions, metrics, fill/trade history, rejection records, equity/exposure, and benchmark. Identical inputs/data/service version yield the same experiment identifier and deterministic results. Artifacts persist as local JSON and are atomically replaced; they are not an experiment database. Version constants must change when simulation semantics change.

No broad-universe research, survivorship-bias correction, advanced corporate-action types, multi-asset portfolio, measured spreads, market impact, asynchronous jobs, model training, news, or paper/live trading is implemented. Requests are limited to 730 days and 100,000 bars. API is intended for loopback local use, without authentication. Before deployment, add authentication, resource quotas, and job execution. No performance shown on synthetic data is a market claim.