# Development roadmap

Foundation and real daily accounting are verified: FastAPI/Next.js, strict calendar-aware data, Backtrader next-open execution, costs and constraints, split/dividend support, deterministic tests and the original dashboard.

The supervised research milestone is implemented: causal daily features, next-open return labels, CPU XGBoost regression/classification, chronological early stopping and probability calibration, purged expanding walk-forward validation, a final held-out period, broker evaluation, forecast/economic baselines, regimes, drift, background jobs and saved model versions.

Manual continuous research is implemented: provider refresh, independent retrained candidates, forward reference comparisons, fixed acceptance criteria, explicit promotion and restoration of previous active versions. Input snapshots preserve version provenance. Automatic scheduling is deferred.

An experimental Gymnasium Q-learning workflow is implemented against the same broker and unseen evaluation windows. PPO/deep RL, multi-seed statistical research and richer state/action spaces remain future work.

Next meaningful steps:
1. Collect a larger genuinely unseen history and use fixed research hypotheses; compare multiple seeds and uncertainty intervals before interpreting small differences.
2. Expand asset coverage and add multi-asset portfolios/cross-sectional ranking with point-in-time universes.
3. Complete session-aligned intraday equity ingestion and provider revision verification.
4. Add historical news only when point-in-time publication timestamps can be validated.
5. Add a bounded optional retraining scheduler and monitoring only after the manual workflow has enough operational evidence.
6. Add simulated ongoing paper portfolios and portfolio-state inference.

No live brokerage execution or production deployment is enabled. Model complexity is not evidence of improvement; losing/no-trade results and failed acceptance checks are part of the research output.
