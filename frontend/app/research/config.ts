export const defaults = {
  asset: "AAPL",
  provider: "synthetic",
  start: "2022-01-03",
  end: "2025-01-01",
  algorithm: "xgboost",
  horizon: 1,
  evaluation_anchor_horizon: null as number | null,
  folds: 3,
  holdout_fraction: 0.2,
  n_estimators: 160,
  max_depth: 3,
  learning_rate: 0.05,
  min_edge_bps: 5,
  probability_threshold: 0.55,
  feature_set: "basic",
  entry_rule: "return_and_probability",
  subsample: 1,
  colsample_bytree: 1,
  min_child_weight: 5,
  reg_lambda: 5,
  training_window_months: 12,
  recency_half_life_days: 90,
  validation_scheme: "blocks",
  stop_loss_pct: 0,
  seed: 42,
  episodes: 20,
  drawdown_penalty: 0,
  initial_cash: 10000,
  commission_bps: 10,
  fixed_fee: 0,
  spread_bps: 4,
  slippage_bps: 2,
  allocation: 0.9,
  cash_reserve: 0.05,
  max_participation: 0.01,
  fractional: true,
  min_notional: 1,
  strategy: "ma_crossover",
  fast: 10,
  slow: 30,
};

export type ResearchConfig = {
  algorithm: string;
  horizon: number;
  evaluation_anchor_horizon?: number | null;
  folds: number;
  holdout_fraction: number;
  n_estimators: number;
  max_depth: number;
  learning_rate: number;
  min_edge_bps: number;
  probability_threshold: number;
  seed: number;
  feature_set?: string;
  entry_rule?: string;
  subsample?: number;
  colsample_bytree?: number;
  min_child_weight?: number;
  reg_lambda?: number;
  parent_id?: string | null;
  backtest: {
    asset: string;
    start: string;
    end: string;
    provider: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

export const parameterLabels: Record<string, string> = {
  horizon: "Forecast horizon (bars)",
  folds: "Walk-forward folds",
  holdout_fraction: "Final test fraction",
  n_estimators: "Maximum trees",
  max_depth: "Tree depth",
  learning_rate: "Learning rate",
  min_edge_bps: "Extra return hurdle (bps)",
  probability_threshold: "Entry probability",
  training_window_months: "Training window (months)",
  recency_half_life_days: "Recency half-life (days)",
  validation_scheme: "Validation schedule",
  stop_loss_pct: "Stop loss (%)",
  seed: "Random seed",
  feature_set: "Feature set",
  entry_rule: "Entry rule",
  subsample: "Training row fraction",
  colsample_bytree: "Feature fraction per tree",
  min_child_weight: "Minimum child weight",
  reg_lambda: "L2 regularization",
};
export function settingsFromConfig(config: ResearchConfig) {
  const flat = { ...config, ...config.backtest };
  return Object.fromEntries(
    Object.entries(defaults).map(([key, fallback]) => [
      key,
      flat[key] ??
        (key === "training_window_months" || key === "recency_half_life_days"
          ? 0
          : fallback),
    ]),
  ) as typeof defaults;
}
export function settingValue(key: string, value: unknown) {
  if (key === "validation_scheme")
    return value === "monthly" ? "Next calendar month" : "Fixed blocks";
  if (key === "training_window_months" && Number(value) === 0)
    return "All earlier history";
  if (key === "recency_half_life_days" && Number(value) === 0)
    return "Equal weights";
  if (key === "stop_loss_pct")
    return Number(value) === 0 ? "Disabled" : String(value) + "%";
  if (key === "feature_set")
    return value === "technical" ? "Expanded technical (17)" : "Basic (12)";
  if (key === "entry_rule")
    return value === "probability_only"
      ? "Probability only"
      : "Return + probability";
  if (
    key === "probability_threshold" ||
    key === "holdout_fraction" ||
    key === "subsample" ||
    key === "colsample_bytree"
  )
    return (
      (Number(value) * 100).toLocaleString("en-US", {
        maximumFractionDigits: 3,
      }) + "%"
    );
  return String(value);
}
