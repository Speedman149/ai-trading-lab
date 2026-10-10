# Comparing the pasted AAPL scripts with the research app

The supplied scripts are useful model experiments, but their headline returns do not establish a tradable advantage. They do not use the same history, target, portfolio allocation or execution assumptions as the app.

| Item | Pasted scripts | Research app |
| --- | --- | --- |
| History | 2018–2023 with a 2023 test, or 2016–2023 with expanding blocks | Configured history; expanding validation and a separately reserved newest period |
| Forecast target | Today's close to tomorrow's close | Next open to a later open, including supported split/dividend economics |
| Entry timing | Uses the finished closing candle's indicators, then buys at that exact same close | Signal after a completed candle; fills at the following opening plus costs |
| Costs | No commissions, spread or slippage | Configured commissions, spread, slippage and size limits |
| Allocation | Entire capital per positive prediction | Configured entry allocation, reserve, minimum order and participation limits |
| Holding | One close-to-close trade per positive prediction | Holds while the chosen policy remains long; exits at a subsequent open |
| Stop | Assumes exactly the stop-loss percentage whenever the next low breaches it | No stop-order strategy; the scripts' optimistic stop semantics are not copied |
| Parameter selection | Second script sorts returns from the very periods used to compare all combinations | Selects using validation only; only the winner receives the final test |
| Unavailable targets | Final comparison against a missing next close becomes false rather than missing | Rows without observable future label endpoints are excluded |

The same-close entry needs information unavailable before that close was set. A stop also cannot guarantee its trigger price: buying at $100 with a $98 stop and opening at $90 can lose about 10%, rather than the script's fixed 2%, before costs. See [SEC order guidance](https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/types-orders). Searching and then reporting performance on the same evaluation period selects for noise; see [scikit-learn model-selection guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

## Supported comparisons

Research v6 exposes the previously fixed XGBoost subsample, feature sampling, minimum child weight and L2 regularization. Their defaults preserve legacy values (1, 1, 5, 5). Both are evaluated with the same chronological early-stopping/calibration boundaries.

- Basic features retain the original 12 inputs.
- Expanded technical adds 21-bar return, 50-bar moving-average distance, Bollinger percent B, normalized PPO histogram, and normalized 14-bar ATR. They use only completed bars and splits already effective. The pasted scripts' SPY market-context features are not included in this set.
- Return + probability retains the original economic-hurdle and probability entry conditions, with its existing holding rule.
- Probability only holds long while calibrated positive-return probability meets the entry threshold, otherwise cash. It ignores the return hurdle but pays the same actual modeled trading costs. It does not imply an expected profit after those costs. Its predictive acceptance check uses Brier score against the constant probability baseline; joint strategies retain the return MAE check. Both must still beat economic baselines and pass risk/regime checks.

Automatic search varies these choices together with the original tree/hurdle/threshold settings. Data, horizon, splits and broker assumptions remain fixed; selection does not inspect final-test performance. More choices can overfit validation, and repeatedly viewed final tests are no longer independent evidence.

## Recovering parameters and understanding retraining

Every report shows its saved parameters, actual tree counts when recorded, source search and a JSON export. Restore any saved version's complete form settings using **Use settings** or **Use this model's settings**. Completed searches automatically load the selected settings into the form. History lists can reveal all stored runs.

**Re-run saved dates** preserves the saved range and parameters, uses cached provider history and the current code, and saves a separate candidate. It is reproducible when the data, libraries and code match; it is not a replay of an older code version. **Retrain through today** refreshes and extends history while preserving model parameters. Expanding the date range shifts the percentage-based validation and final-test boundaries, so old and new headline returns cover different dates. Parent reports and data snapshots remain unchanged. The comparison panel states whether windows, inputs and code match.

Activity diagnostics show potential entry counts for each gate, forecast/probability ranges, daily-close time invested, average exposure, fills, completed trades, rejections and return relative to buy-and-hold on the same dates. Counts of qualifying signals are not counts of trades, and daily-close observations do not capture all intraday risk. Legacy reports receive these explanations on read without rewriting their saved scores or files.
