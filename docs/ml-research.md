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
| Entry probability | Minimum calibrated estimate of a positive economic return. 0.55 means 55%; the return hurdle must also pass. |
| Walk-forward folds | Number of later evaluation blocks, each tested by a fresh model trained on preceding data. |
| Random seed | Controls reproducible randomness. The numeric value does not indicate quality. |
| Final test fraction | Share of the newest usable labeled history reserved for final testing. 0.20 means 20%. |
| Initial capital | Simulated starting portfolio cash in dollars. |

With roughly 1,000 usable labeled samples and a 20% final test, the first 500 initialize training, the next roughly 300 form the walk-forward blocks, and the newest 200 are held out. Boundary purging can reduce these counts. Each walk-forward fold expands the available training history; the final held-out period uses one frozen model trained only on preceding history. Its results report unseen-period performance and do not decide promotion.

Promotion marks a saved version as the active research model; it does not place brokerage orders. An ineligible candidate shows the number of failed validation checks and stays disabled. An already active version says "Already active." A waiting cursor and "Promoting..." appear only during the actual promotion request. A server rejection appears as an error and releases the button.

## Training protocol

Use at least 250 labeled samples after a 60-bar warm-up: roughly two years of stocks or one year of crypto. Requests can span up to 3,650 days. Missing bars and unsupported equity actions still fail rather than being filled or ignored.

Features are dimensionless: 1/5/20-bar returns, 20/60-bar volatility, 10/30-bar moving-average gaps, candle range/body, opening gap, relative volume and RSI. Only completed candles enter a row. Split factors take effect on ex-date and adjust analytical price/volume units forward; future splits never rewrite older feature values. Cash dividend announcements and future action metadata are not model inputs.

A decision at close i predicts the economic return from open i+1 to open i+1+horizon. The label includes splits and ordinary dividend entitlements held overnight, excludes entitlements on the entry ex-date, and values an unpaid entitlement without assuming spendable cash. This is an economic forecast target, not a net-return guarantee or a mandatory holding period.

The first 50% of labeled rows initializes expanding training. The following rows, up to the reserved holdout, form 2–5 non-overlapping walk-forward evaluation blocks. Each fold trains a fresh model. Training labels must finish strictly before that fold's first candle opens. Validation labels and portfolio marks also stop before the final held-out market period; multi-bar labels crossing that boundary are removed. The last 15–30% is a held-out period, evaluated once per candidate by a final frozen model. No threshold or hyperparameter search reads this period.

XGBoost return regression and binary direction classification use CPU histogram trees with a fixed seed and one training thread. Each training window is split chronologically into approximately 60% fitting, 20% early stopping, and 20% sigmoid probability calibration. Labels crossing either internal boundary are purged too. The calibration model sees disjoint later data. If fitting/calibration has only one direction class, the fallback and its limitation are recorded.

Entry requires predicted economic return above modeled round-trip commission, spread, slippage, approximate fixed fees and the additional return hurdle, plus the configured up-probability threshold. Holding uses a zero-return exit boundary and the complementary probability threshold. The broker still applies actual modeled costs and affordability/volume caps. There is no leverage or short selling.

## Evaluation

Prediction results include return MAE/RMSE, directional accuracy of the regression sign, temporal Spearman rank correlation when at least 20 nonconstant samples exist, Brier score, binary log loss, expected calibration error, and reliability-bin counts. MAE/RMSE appear in percentage points of return. Constant forecasts have undefined rank correlation, not invented zeros.

Trading reports include net return, maximum drawdown, turnover, Sharpe, fees, fills and the equity curve. Cash, buy-and-hold and warmed-up 10/30 moving averages use the same evaluation dates, initial cash, costs and broker constraints. Prediction baselines are zero return and a training-mean forecast, with training-frequency probability diagnostics. For combined walk-forward validation these constant baselines and the regime volatility cutoff are frozen from the first training fold; final-test baselines use pre-test training data.

Regimes combine the preceding known 20-bar trend with a training-only median volatility cutoff. Interval returns are attributed to the preceding completed bar. Compounding noncontiguous intervals is a diagnostic decomposition, not a separately executable regime strategy. Counts below 20 are explicitly marked small. Fold reports also expose changing results over time.

Seeing a holdout consumes it for human research. Rerunning many parameter choices and selecting the winner from the test table would overfit; reserve genuinely later data before drawing conclusions. Historical provider data/action revisions and survivorship in this small fixed asset set remain limitations.

## Automatic parameter search

Select XGBoost, choose a trial count (2–48, default 12), and click **Find best parameters**. Choose validation Sharpe (default) or validation net return as the ranking metric. The search always includes your current settings as trial 1, then samples unique combinations reproducibly with the research random seed.

| Parameter | Sampled values |
| --- | --- |
| Maximum trees | 60, 120, 240, 400 |
| Tree depth | 2, 3, 4, 5 |
| Learning rate | 0.02, 0.05, 0.10, 0.20 |
| Extra return hurdle (bps) | 0, 5, 10, 20 |
| Entry probability | 0.50, 0.55, 0.60, 0.65 |

The asset, provider, date range, forecast horizon, folds, final-test fraction, random seed and execution costs remain fixed. Each search downloads/validates one data snapshot and reuses it across trials. Every trial uses the same chronological purging, separate training/early-stopping/calibration blocks, causal features and broker simulation as manual training. This is a strategy-aware random search rather than an accuracy-only classifier grid search.

Ranking is fixed before evaluation: candidates passing all promotion checks first, then candidates within the 20% drawdown and 20x turnover limits, then the chosen validation metric. Ties use the other trading metric, smaller drawdown, lower turnover, lower MAE and earlier trial. Undefined Sharpe ranks below all defined values. Failed trials remain visible with an error and no rank. The leaderboard includes returns, Sharpe, drawdown, turnover, trading costs, prediction error, directional accuracy, baseline-relative return and individual fold returns.

Selection is persisted **before** final-test evaluation. Only the selected winner is fitted on pre-test history and evaluated on the reserved final test. Losing trials never evaluate that period. The winner is saved as a normal research version with a link to its search; no model is promoted automatically. **Use best-found settings** copies the winner and the matching data/cost configuration into the form. Saved searches and jobs survive page reloads; interrupted backend jobs are explicitly marked.

"Best found" means best under this ranking among the sampled choices, not a proven optimum. Searching many combinations can overfit validation. Previously viewed holdouts and repeated searches of the same dates do not become independent again. A genuinely later period is needed for another independent assessment. This separation follows [scikit-learn's model-selection guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

API: POST /api/research/searches accepts {"research": <ResearchRequest>, "trials": 12, "objective": "sharpe"} and returns the usual background job with kind="search" and search_id. Poll GET /api/research/jobs/{id}; after completion result_id identifies the winner. GET /api/research/searches lists saved runs, and GET /api/research/searches/{search_id} returns the leaderboard and selection record. Q-learning parameter search is not included in this milestone.

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
- XGBoost MAE beats both constant forecast baselines.
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
