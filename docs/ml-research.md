# Daily model research

The AI Research page is at http://localhost:3000/research. It trains candidates locally, evaluates actual forecasts and broker results, and saves each version. Supported assets are AAPL, SPY, BTC/USD and ETH/USD; this milestone uses daily bars. Synthetic mode is a workflow demonstration, not evidence of investment performance.

## Settings in plain language

| Setting | Meaning |
| --- | --- |
| Forecast horizon | How many daily opening-to-opening intervals the model forecasts, starting at the next opening after a completed bar. It is not a mandatory holding period. |
| Tree depth | Maximum number of decision levels in a tree. Greater depth adds complexity and overfitting risk. |
| Maximum trees | Upper limit per XGBoost model; early stopping may select fewer. |
| Learning rate | Size of each added tree's contribution to the model, not retraining frequency. |
| Extra return hurdle | Predicted return required above modeled round-trip costs. 5 basis points equals 0.05%. |
| Entry probability | Minimum calibrated estimate of a positive economic return. 0.55 means 55%; joint entry also requires the return hurdle. |
| Walk-forward folds | Number of later evaluation blocks, each tested by a fresh model trained on preceding data. |
| Random seed | Controls reproducible randomness. The numeric value does not indicate quality. |
| Final test fraction | Share of the newest usable labeled history reserved for final testing. 0.20 means 20%. |
| Initial capital | Simulated starting portfolio cash in dollars. |

With roughly 1,000 usable labeled samples and a 20% final test, the first 500 initialize training, the next roughly 300 form the walk-forward blocks, and the newest 200 are held out. Boundary purging can reduce these counts. Each walk-forward fold expands or rolls the available training history according to the window setting; the final held-out period uses one frozen model trained only on preceding history. Its results report unseen-period performance and do not decide promotion.

Promotion marks a saved version as the active research model; it does not place brokerage orders. An ineligible candidate shows the number of failed validation checks and stays disabled. An already active version says "Already active." A waiting cursor and "Promoting..." appear only during the actual promotion request. A server rejection appears as an error and releases the button.

## Training protocol

Use at least 250 labeled samples after a 60-bar warm-up: roughly two years of stocks or one year of crypto. Requests can span up to 3,650 days. Missing bars and unsupported equity actions still fail rather than being filled or ignored.

Features are dimensionless: 1/5/20-bar returns, 20/60-bar volatility, 10/30-bar moving-average gaps, candle range/body, opening gap, relative volume and RSI. Only completed candles enter a row. Split factors take effect on ex-date and adjust analytical price/volume units forward; future splits never rewrite older feature values. Cash dividend announcements and future action metadata are not model inputs.

A decision at close i predicts the economic return from open i+1 to open i+1+horizon. The label includes splits and ordinary dividend entitlements held overnight, excludes entitlements on the entry ex-date, and values an unpaid entitlement without assuming spendable cash. This is an economic forecast target, not a net-return guarantee or a mandatory holding period.

The first 50% of labeled rows initializes the available training history. The following rows, up to the reserved holdout, form 2–5 non-overlapping fixed evaluation blocks, or calendar-month blocks when monthly scheduling is selected. Each fold trains a fresh model. Training labels must finish strictly before that fold's first candle opens. Validation labels and portfolio marks also stop before the final held-out market period; multi-bar labels crossing that boundary are removed. The last 15–30% is a held-out period, evaluated once per candidate by a final frozen model. No threshold or hyperparameter search reads this period.

XGBoost return regression and binary direction classification use CPU histogram trees with a fixed seed and one training thread. The newest 20% of each window, capped at 21 rows with a minimum of six, is reserved for sigmoid direction calibration. The preceding rows divide approximately 75%/25% into core fitting and early stopping. Labels crossing either internal boundary are purged. Early stopping selects the tree count. The return model then refits that count on **all known training rows**, including the most recent rows; the direction model refits on all rows whose labels finish before the calibration block. Direction calibration remains disjoint from classifier fitting and tree selection. If a required classifier block/calibration has only one direction class, the recorded fallback applies. Older saved reports retain their older fitting protocol.

The website defaults new experiments to a 12-month window and a 90-calendar-day weight half-life. Backend defaults and restored older reports retain expanding, equal-weight history unless explicitly changed. **Training window (months)** = 0 uses all earlier history, otherwise 3–60 calendar months before the prediction boundary. Features still use earlier completed prices for indicator warm-up. **Recency half-life (days)** = 0 disables weighting. Otherwise raw row weight is 2 ** (-age_days / half_life_days): at 90 days, rows aged 0/90/180 days have relative weights 1/0.5/0.25. Each fitting/evaluation/calibration component normalizes its weights to mean one, preserving within-block age ratios without arbitrarily scaling regularization. These weights affect fitting, internal early-stopping loss and sigmoid calibration; reported portfolio returns and evaluation errors remain unweighted. Saved metadata records weight reference time, oldest/newest normalized weights, effective sample size, actual refit rows and label endpoints.

**Use last 3 months → next month** sets a three-month rolling window, 90-day half-life and monthly validation. Validation starts at the first full calendar month after the initial chronological training region; each month gets a fresh model from preceding known labels. Stocks use New York session months; crypto uses UTC months. The last block can be a partial month before the reserved final test. Training windows may overlap, evaluated prediction rows do not. Monthly scheduling determines fold count from dates, so the fixed fold-count setting is ignored in this mode. **The final held-out test still uses one frozen model for its entire reserved period**; monthly refits apply to validation and must not be confused with training on final-test labels. Short windows contain fewer independent examples and may train less reliably.

The default joint entry rule requires predicted economic return above modeled round-trip commission, spread, slippage, approximate fixed fees and the additional return hurdle, plus the configured up-probability threshold. Holding uses a zero-return exit boundary and the complementary probability threshold. The broker still applies actual modeled costs and affordability/volume caps. The optional probability-only rule holds long at or above its entry threshold and cash below it, ignoring the regression hurdle while still paying modeled trading costs. There is no leverage or short selling.

## Evaluation

Prediction results include return MAE/RMSE, directional accuracy of the regression sign, temporal Spearman rank correlation when at least 20 nonconstant samples exist, Brier score, binary log loss, expected calibration error, and reliability-bin counts. MAE/RMSE appear in percentage points of return. Constant forecasts have undefined rank correlation, not invented zeros.

Trading reports include net return, maximum drawdown, turnover, Sharpe, fees, fills and the equity curve. Cash, buy-and-hold and warmed-up 10/30 moving averages use the same evaluation dates, initial cash, costs and broker constraints. Prediction baselines are zero return and a training-mean forecast, with training-frequency probability diagnostics. For combined walk-forward validation these constant baselines and the regime volatility cutoff are frozen from the first training fold; final-test baselines use pre-test training data.

Regimes combine the preceding known 20-bar trend with a training-only median volatility cutoff. Interval returns are attributed to the preceding completed bar. Compounding noncontiguous intervals is a diagnostic decomposition, not a separately executable regime strategy. Counts below 20 are explicitly marked small. Fold reports also expose changing results over time.

Seeing a holdout consumes it for human research. Rerunning many parameter choices and selecting the winner from the test table would overfit; reserve genuinely later data before drawing conclusions. Historical provider data/action revisions and survivorship in this small fixed asset set remain limitations.

## Automatic parameter search

Select XGBoost, choose a search method and trial budget (2–48; the website defaults to 32), then click **Find best parameters**. Choose validation Sharpe (default) or validation net return. For sampled/refinement methods, trial 1 contains your current settings. Grid order follows the declared ranges and may exclude the current setting.

- **Broad then refine** (website default): spend half the budget on seeded broad combinations, freeze the two highest-ranked successful broad validation candidates, then alternate nearby changes around those centers. Ranking uses the same promotion/risk/metric policy throughout. If only one broad trial succeeds, refine that center; if all fail, try neighbors of the submitted settings.
- **Refine current settings**: evaluate the submitted configuration, then nearby numeric choices with the same feature set and entry rule. **Refine winning settings** restores a saved winner's complete data/cost/model configuration and prepares this mode; click Find best parameters to start it.
- **Broad random search**: use the original seeded sampling across feature sets, rules and numeric choices.
- **Exhaustive range grid**: enter minimum, maximum and step for each numeric parameter. Every Cartesian combination is attempted, including horizon, depth, maximum trees, learning rate, entry probability, hurdle, fold count, seed, stop loss, window, weight half-life, row/column sampling, child weight and L2. A single-value range fixes a parameter. Feature set, entry rule, schedule, dates, costs, sizing and final-test fraction stay fixed. The backend uses Decimal stepping, includes an aligned maximum, and excludes an unaligned maximum. The preview shows the full count; grids exceeding 256 combinations are rejected before downloading/training, never sampled under an exhaustive label. Failed combinations are recorded with their errors.

For example, probability 0.550–0.552 in steps of 0.001 (0.1 percentage point), stop 2.9–3.1% in steps of 0.1%, and depths 2–3 in whole steps generate exactly **3 × 3 × 2 = 18** combinations. Integer settings cannot move by 0.1. Learning-rate 0.1 means a different unit from a 0.1% probability change. More dimensions multiply the count quickly: two values for eight settings already yield 256 combinations.

Fold counts are an evaluation/refitting choice, rather than a tree learning parameter. Grid candidates with different folds retain the same aggregate validation period. Monthly scheduling cannot vary fold counts. Seeds change training randomness; selecting the best seed on validation is allowed as an exploratory grid but is not evidence of stability across seeds. Probability-only policies ignore the hurdle, so a varying hurdle grid is rejected in that mode.

Refinement starts with ±0.01 entry probability, ±2.5 bps extra hurdle, ±0.01 learning rate, ±1 depth/child-weight/L2, ±0.05 row/feature fractions and ±40 maximum trees. Parameters are bounded by the existing request limits. Try single-setting neighbors and joint probability/hurdle pairs at each radius, increasing the radius only after that neighborhood. Probability-only policies skip the unused hurdle. Deduplication treats ignored hurdles as equivalent. Every trial records its phase and originating broad trial. The report saves the steps and bounds; old random-search reports remain readable. Short budgets may only cover the first settings in this order; this is not an exhaustive or Bayesian optimizer.

| Parameter | Sampled values |
| --- | --- |
| Maximum trees | 60, 120, 240, 400 |
| Tree depth | 2, 3, 4, 5 |
| Learning rate | 0.02, 0.05, 0.10, 0.20 |
| Extra return hurdle (bps) | 0, 5, 10, 20 |
| Entry probability | 0.50, 0.52, 0.55, 0.60 |
| Feature set | Basic, expanded technical |
| Entry rule | Return + probability, probability only |
| Training row fraction | 0.8, 1.0 |
| Feature fraction per tree | 0.8, 1.0 |
| Minimum child weight | 1, 5, 10 |
| L2 regularization | 1, 5, 10 |

For sampled/refinement modes, asset, provider, date range, horizon, folds, final-test fraction, seed, stop loss and recent-data controls stay fixed. Grids may vary declared numeric settings while keeping a common calendar protocol: before running any candidate, the maximum requested horizon fixes the shared labeled signal dates and held-out boundary. Each candidate recomputes its own labels and purged training window, but portfolio starts/ends and the final-test reservation remain identical. Portfolio simulation ends use the reference horizon's endpoint so shorter horizons do not receive a shorter comparison period. Each search downloads/validates one data snapshot and reuses it across trials. Every trial uses the same chronological purging, separate training/early-stopping/calibration blocks, causal features and broker simulation as manual training. These are strategy-aware searches; predictive accuracy alone does not choose the winner.

Ranking is fixed before evaluation: candidates passing all promotion checks first, then candidates within the 20% drawdown and 20x turnover limits, then the chosen validation metric. Ties use the other trading metric, smaller drawdown, lower turnover, lower MAE and earlier trial. Undefined Sharpe ranks below all defined values. Failed trials remain visible with an error and no rank. The leaderboard includes returns, Sharpe, drawdown, turnover, trading costs, prediction error, directional accuracy, baseline-relative return and individual fold returns.

Selection is persisted **before** final-test evaluation. Only the selected winner is fitted on pre-test history and evaluated on the reserved final test. Losing trials never evaluate that period. The winner is saved as a normal research version with a link to its search; no model is promoted automatically. **Use best-found settings** copies the winner and the matching data/cost configuration into the form. Saved searches and jobs survive page reloads; interrupted backend jobs are explicitly marked.

"Best found" means best under this ranking among the sampled choices, not a proven optimum. Searching many combinations can overfit validation. Previously viewed holdouts and repeated searches of the same dates do not become independent again. A genuinely later period is needed for another independent assessment. This separation follows [scikit-learn's model-selection guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

API: POST /api/research/searches accepts {"research": <ResearchRequest>, "trials": 32, "objective": "sharpe", "mode": "coarse_to_fine"}. API clients omitting mode retain the original random search and default 12-trial budget. The endpoint returns the usual background job with kind="search" and search_id. Poll GET /api/research/jobs/{id}; after completion result_id identifies the winner. GET /api/research/searches lists saved runs, and GET /api/research/searches/{search_id} returns the leaderboard and selection record. Q-learning parameter search is not included in this milestone.

## Reinforcement learning

The initial RL experiment is tabular Q-learning, not PPO. Its Gymnasium environment drives the same Backtrader broker through completed-bar callbacks. It does not use a duplicate fill/cash simulator. Observations contain causal market features, cash ratio, exposure, equity ratio and drawdown. Actions target cash or long using existing entry sizing and execution constraints.

A 32-state table discretizes five booleans: positive 5-bar momentum, positive 30-bar MA gap, high training-relative volatility, current holdings and drawdown below -5%. Training explores with seeded epsilon decay; learning rate is 0.1 and discount factor is 0.95. Every fold resets the table and portfolio, trains only on that fold's training decision bars, then freezes the policy for evaluation. Training rewards are not unseen-market performance.

Reward = log(net equity now / previous net equity) minus drawdown_penalty × max(0, absolute drawdown now - previous absolute drawdown).

Net equity includes realized/unrealized holdings and earned dividend receivables, with commissions/spread/slippage already deducted. They are not subtracted twice. The default penalty is zero. Changing it is a new research configuration, not permission to tune against held-out results.

An action policy has no return/probability prediction output, so prediction/calibration fields are N/A. RL is judged by the same economic baselines, unseen windows and regime diagnostics. Sparse state coverage, seed sensitivity, repeated historical episodes and reward exploitation remain research limitations. This small environment is a starting experiment, not evidence that RL improves trading.

## Retraining, drift and version control

Refresh data & retrain is manual. It extends the saved configuration through newly completed daily history, refreshes validated provider bars/actions, creates a separate candidate and retains the previous model. The final end date is today's UTC date, exclusive, so the still-current daily bar is excluded. Histories longer than 3,650 days roll their start forward. No automatic scheduler or trading process runs.

Each version stores XGBoost JSON models or the JSON Q table, configuration, library/code/feature versions, provider hashes, predictions, diagnostics and evaluation. Starting with research-v3, dataset.parquet and dataset-actions.json preserve the exact input snapshot, even if the shared provider cache is refreshed. Generated data/models live under ignored artifacts/research; no pickle deserialization or user-supplied executable model path is used.

Distribution monitoring compares training quantile-bin frequencies with final-test features using PSI; >0.25 is an informational alert. With 40+ predictions, later-half/earlier-half test MAE >1.5 alerts possible deterioration. These thresholds are heuristics, not statistical proof or automatic retraining commands.

Promotion is an explicit research-registry action. Fixed acceptance criteria use walk-forward validation, not the reported final test:

- At least 40 validation predictions.
- Net return greater than each of cash, buy-and-hold and moving averages.
- Positive defined Sharpe.
- Maximum drawdown no worse than -20%.
- Turnover at most 20 times initial capital.
- At least two regimes with 20 intervals.
- No sufficiently sampled regime loses more than 10%.
- Joint XGBoost MAE beats both constant forecast baselines; probability-only Brier score beats the constant probability baseline.
- For a retrained/reference model: at least 40 forward validation samples strictly after that reference's last training information; candidate return improves and drawdown is no worse.

Reference comparisons use the saved model's own decision thresholds on identical dates and broker settings. If an active model exists, it is the reference and the candidate must explicitly be retrained against it. Promotion rejects stale evaluations if the active pointer changed. Former active versions remain on disk and can be restored. Synthetic/real providers and execution/model families have distinct registry keys. An active research version is a saved pointer, not a live trading deployment.

Training is a persisted background job, with one concurrent trainer to bound local resource use. Jobs report progress/failure; a backend restart marks unfinished jobs interrupted. The UI reconnects to running jobs and can browse finished versions.

## API and CLI

POST /api/research/jobs starts training; GET /api/research/jobs/{id} reports progress. GET /api/research/versions lists models; GET /api/research/versions/{id} returns the complete validated report. POST suffixes /retrain, /promote and /rollback perform the corresponding version operation. API keys remain in the backend.

```bash
cd ~/projects/ai-trading-lab/backend
../.venv/bin/python -m app.ml.cli --provider alpaca --asset AAPL --start 2022-01-03 --end 2025-01-01
../.venv/bin/python -m app.ml.cli --algorithm q_learning --episodes 20 --asset BTC/USD
```

Official research references: [XGBoost estimator and early stopping](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [CPU-only installation](https://xgboost.readthedocs.io/en/stable/install.html), [chronological splits and gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), and [Gymnasium agent training](https://gymnasium.farama.org/introduction/train_agent/). Our purging uses actual label-availability timestamps rather than shuffled cross-validation.

## Research v6: feature and strategy comparisons

The parameter search additionally varies feature set (basic/technical), entry rule (return + probability/probability only), subsample and colsample_bytree (0.8/1), min_child_weight and reg_lambda (1/5/10). Probability thresholds are 0.50, 0.52, 0.55 and 0.60; the current configuration remains the first trial even outside these ranges. Probability-only trials set the ignored extra return hurdle to zero. Direct seeded sampling avoids materializing the expanded Cartesian product. Full validation and promotion ranking, data windows, prediction horizon and costs remain fixed.

Model files record their feature names so older trained models remain readable. New activity and retraining explanations are optional fields; legacy reports are enriched on read without altering their saved results. Details and the audit of the supplied scripts are in [script-comparison.md](script-comparison.md).

Acceptance policy v2 uses the return MAE baseline for joint entry and the Brier baseline for probability-only entry, since the latter never uses the regression forecast to trade. Economic, risk and regime gates remain required. Earlier saved versions retain their original evaluated acceptance policy.

The final service tag is research-v6.1 to invalidate development artifacts created before the strategy-specific acceptance policy was finalized. Old reports remain readable with their saved policies.


## Research v7: diagnose low returns before increasing the search

A reported net return covers the displayed evaluation dates. It is not an annual or monthly rate. Compare the benchmark on those same dates and costs, plus exposure, drawdown, trade count and performance in every fold. The report now shows this window explicitly, gross return on the same fills, costs, final tree-fitting/early-stopping/calibration sample counts, and actual trees in each fold. Gross return keeps the actual executed trades and sizing; it does not simulate a lower-cost policy, whose entry hurdle and trades could differ.

The saved AAPL comparison before refinement has 4.9568% validation net return from 2023-08-15 to 2024-06-10 versus buy-and-hold's 8.3758%. The AI makes eight completed trades; only one of three folds has a positive return. Its drawdown is 2.5933% versus 15.0805% for buy-and-hold. Gross return on the same fills is 7.0608%, with $210.40 modeled costs on $10,000. The final frozen model uses two return trees and one direction tree, and stays in cash for the viewed final test. These facts do not establish a reliable forecasting advantage. Refinement of these dates is exploratory because the validation scores and final test have already been inspected.

Early stopping deliberately selects fewer trees when additional trees fail to improve the internal score. A high maximum does not force a complex model, and a one-tree result is not a failed training job. The 60/20/20 split deliberately reserves later history for tree-count selection and probability calibration. No fitted component sees the final held-out labels. See the [official XGBoost estimator guidance](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).

The default 10 bps per-side commission is a simulation assumption, not a quote from Alpaca. Alpaca describes commission-free trading for qualifying self-directed individual API brokerage accounts, with possible regulatory fees; account plans and execution arrangements matter. See [Alpaca disclosures](https://alpaca.markets/disclosures) and [regulatory fees](https://docs.alpaca.markets/us/docs/regulatory-fees). Match modeled assumptions to the intended account and observed execution costs in a separate documented experiment; do not silently lower costs to improve a score. This engine has configurable generic per-side bps/fixed fees, not a dated Alpaca regulatory-fee schedule.

More precise optimization can fit the validation period more closely without producing a useful future strategy. Prespecify each experiment and keep genuinely later data for a fresh assessment; changing settings after viewing the final test does not restore its independence. This follows [scikit-learn's explanation of model-selection bias](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html). Additional market context, broader asset/history coverage, and longer forecast horizons are separate research hypotheses, not guarantees of improved returns. Forecast horizon remains fixed within a search so candidates share exactly the same labels and evaluation boundaries.

## Protective stops (engine v5 / research v8)

**backtest.stop_loss_pct** is a fixed percentage below actual entry execution price; zero disables it. Its reference adjusts with stock splits through position cost basis. It is active on the entry bar. A daily low at/below the threshold triggers a market exit modeled at the threshold unless the opening price is already lower; gaps execute at that opening price. Half-spread, additional slippage and commission/fixed fee apply on top. The price decline to the stop is trading P&L, not transaction cost.

Open-time strategy instructions execute first. A risk exit then cancels their unfilled remainder, observes the preceding completed volume, and subtracts shares already sold on that bar from the stop's participation allowance. A stop's unfilled remainder expires but its risk-exit state remains latched: remaining shares attempt to exit at subsequent openings even if price bounces. No new entry is placed at the stop-triggered close or while the risk exit remains incomplete. Final partial holdings are still marked to close. Ordinary dividend entitlement/payment accounting survives stops and splits.

OHLC cannot reconstruct intrabar order books, queueing or the exact trigger time. Intrabar fill records therefore retain the bar-opening timestamp as a **bar identifier**, not a claim that the stop was hit at the opening. The fields reference_open, reference_price (gap open or stop level) and reason=stop_loss distinguish execution assumptions; losses may exceed the configured stop. Buy-and-hold, cash and the MA baseline have **no stop overlay**, retaining genuine simple baselines with the same dates/costs/sizing. See the [SEC's order explanation](https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/types-orders) for why stop prices do not guarantee execution prices.

Weight fitting follows the [XGBoost sample-weight API](https://xgboost.readthedocs.io/en/stable/python/python_api.html). Training/evaluation isolation follows chronological validation principles, including bounded prior training history and gaps described in [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Neither recency weighting, finer grids nor stop losses guarantees stronger future returns.

Saved grid winners retain their shared maximum-horizon date anchor in their configuration, so restoring or retraining them reproduces the same split protocol. Editing the forecast horizon in the form resets that anchor for a new experiment. Disabled stops retain existing active-model family keys.

The finalized service tag is research-v8.1, including saved grid date anchors and compatible disabled-stop registry keys. Earlier v8 development reports remain readable with their original boundaries.
