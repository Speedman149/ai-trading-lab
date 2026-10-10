"use client";
import Link from "next/link";
import SearchResults, {
  type SearchReport,
  type SearchSummary,
} from "./search-results";
import { useEffect, useState } from "react";
import { defaults, settingsFromConfig, type ResearchConfig } from "./config";
import ModelSettings, { type Retraining } from "./model-settings";
import Activity, { type ActivityData } from "./activity";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
type Stats = Record<string, number | null>;
type Prediction = {
  samples: number;
  mae: number | null;
  rmse: number | null;
  directional_accuracy: number | null;
  rank_correlation: number | null;
  brier_score: number | null;
  log_loss: number | null;
  calibration_error: number | null;
  reliability: {
    probability: number;
    observed_frequency: number;
    count: number;
  }[];
};
type Simulation = {
  metrics: Stats;
  equity: { timestamp: string; equity: number; drawdown: number }[];
  trades: {
    timestamp: string;
    side: string;
    quantity: number;
    price: number;
    total_cost: number;
  }[];
};
type Evaluation = {
  prediction: Prediction;
  activity?: ActivityData | null;
  prediction_baselines: Record<string, Prediction>;
  strategy: Simulation;
  baselines: Record<string, Simulation>;
  regimes: {
    regime: string;
    intervals: number;
    sufficient: boolean;
    prediction: Prediction;
    net_return: number;
    max_drawdown: number;
    sharpe: number | null;
  }[];
  predictions: {
    signal_time: string;
    entry_time: string;
    actual_return: number;
    predicted_return: number | null;
    probability_up: number | null;
    target: number | null;
  }[];
};
type Job = {
  id: string;
  kind?: "train" | "search";
  search_id?: string | null;
  status: string;
  progress: string;
  result_id: string | null;
  error: string | null;
};
type Version = {
  id: string;
  created_at: string;
  asset: string;
  algorithm: string;
  provider: string;
  net_return: number;
  active: boolean;
  eligible: boolean;
  rollback_available: boolean;
};
type Report = {
  id: string;
  search?: { id: string; selected_trial?: number } | null;
  created_at: string;
  config: ResearchConfig;
  retraining?: Retraining | null;
  provenance: { synthetic: boolean; rows: number; dataset_hash: string };
  training: {
    trained_through: string;
    holdout_start: string;
    rows: number;
    episode_rewards?: number[];
    regression_trees?: number;
    direction_trees?: number | null;
    best_iteration?: number;
  };
  validation: Evaluation;
  holdout: Evaluation;
  walk_forward: {
    fold: number;
    training_rows: number;
    training_label_end: string;
    evaluation_start: string;
    prediction: Prediction;
    strategy: Stats;
  }[];
  importance: { feature: string; importance: number }[];
  drift: {
    features: { feature: string; psi: number; flagged: boolean }[];
    recent_mae_ratio: number | null;
    deterioration: boolean;
    note: string;
  };
  promotion: {
    eligible: boolean;
    criteria: { name: string; value: number | null; passed: boolean }[];
    incumbent_comparison: {
      samples: number;
      sufficient: boolean;
      return_improvement?: number;
    } | null;
  };
  latest_prediction: {
    signal_time: string;
    predicted_return: number | null;
    probability_up: number | null;
    context: string;
  };
  warnings: string[];
  assumptions: string[];
};

const settingHelp: Partial<Record<keyof typeof defaults, string>> = {
  horizon:
    "Daily bars ahead: 1 forecasts from the next market open to the following open. This does not force a one-day holding period.",
  max_depth:
    "Maximum decision levels in each tree. Deeper trees learn more complex patterns and can overfit.",
  n_estimators:
    "Maximum trees per XGBoost model. Early stopping may use fewer trees.",
  learning_rate:
    "How much each new tree changes the forecast. Smaller values make smaller learning steps.",
  min_edge_bps:
    "Extra predicted return required above modeled round-trip trading costs. 5 bps = 0.05%.",
  probability_threshold:
    "Minimum estimated chance of a positive return to enter. 0.55 = 55%; the return hurdle must also pass.",
  folds:
    "Train on earlier history, test the next block, then expand training and repeat this many times.",
  seed: "Controls randomness so the same data and settings can reproduce a run. A higher number does not mean better predictions.",
  holdout_fraction:
    "Newest share of usable history reserved for the final test. 0.20 = 20%; it never trains the model.",
  initial_cash:
    "Starting money in the simulated portfolio. No money is deposited or traded.",
};
const percent = (v: number | null | undefined) =>
  v == null ? "N/A" : (v * 100).toFixed(2) + "%";
const decimal = (v: number | null | undefined, digits = 3) =>
  v == null ? "N/A" : v.toFixed(digits);
const names: Record<string, string> = {
  cash: "Cash",
  buy_hold: "Buy and hold",
  ma_crossover: "Moving average 10/30",
  zero_return: "Zero return",
  training_mean: "Training mean",
};
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API + path, options);
  } catch {
    throw new Error("Cannot reach the backend. Start the app and try again.");
  }
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item: { msg: string }) => item.msg).join("; ")
          : detail?.message || "Research request failed",
    );
  }
  return data;
}
export default function ResearchPage() {
  const [form, setForm] = useState(defaults);
  const [ready, setReady] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [period, setPeriod] = useState<"holdout" | "validation">("holdout");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submissionKind, setSubmissionKind] = useState<"train" | "search">(
    "train",
  );
  const [searchTrials, setSearchTrials] = useState(12);
  const [searchObjective, setSearchObjective] = useState<
    "sharpe" | "net_return"
  >("sharpe");
  const [searchReport, setSearchReport] = useState<SearchReport | null>(null);
  const [showAllVersions, setShowAllVersions] = useState(false);
  const [showAllSearches, setShowAllSearches] = useState(false);
  const [searchRuns, setSearchRuns] = useState<SearchSummary[]>([]);
  const [pendingAction, setPendingAction] = useState<{
    id: string;
    operation: "retrain" | "promote" | "rollback";
  } | null>(null);
  const training =
    submitting ||
    pendingAction?.operation === "retrain" ||
    job?.status === "running" ||
    job?.status === "queued";
  const running = training || pendingAction !== null;
  const searching =
    (submitting && submissionKind === "search") ||
    (job?.kind === "search" && ["queued", "running"].includes(job.status));
  const reloadVersions = async () =>
    setVersions(await request<Version[]>("/api/research/versions"));
  useEffect(() => {
    setReady(true);
    let mounted = true;
    Promise.all([
      request<Version[]>("/api/research/versions"),
      request<Job[]>("/api/research/jobs"),
      request<SearchSummary[]>("/api/research/searches"),
    ])
      .then(([saved, jobs, searches]) => {
        if (mounted) {
          setVersions(saved);
          setSearchRuns(searches);
          const current = jobs.find(
            (item) => item.status === "running" || item.status === "queued",
          );
          if (current) setJob(current);
        }
      })
      .catch((e: Error) => {
        if (mounted) setError(e.message);
      });
    return () => {
      mounted = false;
    };
  }, []);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await request<Job>("/api/research/jobs/" + job!.id);
        if (cancelled) return;
        if (next.status === "completed" && next.result_id) {
          const result = await request<Report>(
            "/api/research/versions/" + next.result_id,
          );
          const [saved, searches, found] = await Promise.all([
            request<Version[]>("/api/research/versions"),
            request<SearchSummary[]>("/api/research/searches"),
            next.search_id
              ? request<SearchReport>(
                  "/api/research/searches/" + next.search_id,
                )
              : Promise.resolve(null),
          ]);
          if (!cancelled) {
            setReport(result);
            setForm(settingsFromConfig(result.config));
            setMessage(
              found
                ? "The winning settings are now loaded in the training form."
                : "The form now shows the settings and dates used for this model.",
            );
            setVersions(saved);
            setJob(next);
            setSearchRuns(searches);
            setSearchReport(found);
            setPeriod(found ? "validation" : "holdout");
          }
        } else if (["failed", "interrupted"].includes(next.status)) {
          setJob(next);
          setError(next.error || next.progress);
        } else {
          setJob(next);
          timer = setTimeout(poll, 1000);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Unable to check training");
          setJob(null);
        }
      }
    }
    timer = setTimeout(poll, 500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [job]);
  useEffect(() => {
    const target = report
      ? "model-settings"
      : searchReport
        ? "search-results"
        : null;
    if (target)
      document.getElementById(target)?.scrollIntoView({ block: "start" });
  }, [report, searchReport]);
  const change = (
    key: keyof typeof defaults,
    value: string | number | boolean,
  ) => setForm((old) => ({ ...old, [key]: value }));
  const numeric = (
    key: keyof typeof defaults,
    label: string,
    min: number,
    max: number,
    step = 1,
  ) => (
    <div>
      <label htmlFor={key}>{label}</label>
      <input
        id={key}
        type="number"
        required
        min={min}
        max={max}
        step={step}
        value={form[key] as string | number}
        aria-describedby={settingHelp[key] ? key + "-help" : undefined}
        onChange={(e) => change(key, Number(e.target.value))}
      />
      {settingHelp[key] && (
        <p id={key + "-help"} className="subtle field-hint">
          {settingHelp[key]}
        </p>
      )}
    </div>
  );
  function researchConfig() {
    const {
      asset,
      provider,
      start,
      end,
      initial_cash,
      commission_bps,
      fixed_fee,
      spread_bps,
      slippage_bps,
      allocation,
      cash_reserve,
      max_participation,
      fractional,
      min_notional,
      strategy,
      fast,
      slow,
      ...model
    } = form;
    return {
      ...model,
      backtest: {
        asset,
        provider,
        start,
        end,
        timeframe: "1d",
        initial_cash,
        commission_bps,
        fixed_fee,
        spread_bps,
        slippage_bps,
        allocation,
        cash_reserve,
        max_participation,
        fractional,
        min_notional,
        strategy,
        fast,
        slow,
      },
    };
  }
  async function submitResearch(kind: "train" | "search") {
    setError("");
    setMessage("");
    setSubmissionKind(kind);
    setSubmitting(true);
    setReport(null);
    setSearchReport(null);
    try {
      setJob(
        await request<Job>(
          kind === "search" ? "/api/research/searches" : "/api/research/jobs",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
              kind === "search"
                ? {
                    research: researchConfig(),
                    trials: searchTrials,
                    objective: searchObjective,
                  }
                : researchConfig(),
            ),
          },
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Research request failed");
    } finally {
      setSubmitting(false);
    }
  }
  async function train(e: React.FormEvent) {
    e.preventDefault();
    await submitResearch("train");
  }
  async function findParameters() {
    const formElement = document.getElementById(
      "research-form",
    ) as HTMLFormElement;
    if (formElement.reportValidity()) await submitResearch("search");
  }
  async function loadSearch(id: string) {
    setError("");
    setMessage("");
    try {
      const found = await request<SearchReport>("/api/research/searches/" + id);
      const winner = found.winner_id
        ? await request<Report>("/api/research/versions/" + found.winner_id)
        : null;
      setSearchReport(found);
      setReport(winner);
      setPeriod("validation");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load search");
    }
  }
  function applyWinner() {
    if (!searchReport) return;
    const winner = searchReport.trials.find(
      (trial) => trial.trial === searchReport.selected_trial,
    );
    if (!winner) return;
    const base = searchReport.config.research;
    setForm(
      settingsFromConfig({ ...base, ...winner.parameters } as ResearchConfig),
    );
    document
      .getElementById("research-form")
      ?.scrollIntoView({ block: "start" });
    setSearchTrials(searchReport.config.trials);
    setSearchObjective(searchReport.config.objective);
    setMessage(
      "Best-found settings and their data/cost configuration copied into the training form.",
    );
  }
  async function load(id: string, apply = false) {
    setError("");
    setMessage("");
    try {
      const saved = await request<Report>("/api/research/versions/" + id);
      const found = saved.search
        ? await request<SearchReport>(
            "/api/research/searches/" + saved.search.id,
          )
        : null;
      setReport(saved);
      if (apply) useSettings(saved.config);
      setSearchReport(found);
      setPeriod(found ? "validation" : "holdout");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load model");
    }
  }
  function useSettings(config: ResearchConfig) {
    setForm(settingsFromConfig(config));
    document
      .getElementById("research-form")
      ?.scrollIntoView({ block: "start" });
    setMessage(
      "Saved model settings, history dates and execution costs copied into the training form.",
    );
  }
  async function action(
    id: string,
    operation: "retrain" | "promote" | "rollback",
    mode: "latest" | "same_dates" = "latest",
  ) {
    setError("");
    setMessage("");
    setPendingAction({ id, operation });
    try {
      const response = await request<Job>(
        "/api/research/versions/" + id + "/" + operation,
        {
          method: "POST",
          ...(operation === "retrain"
            ? {
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode }),
              }
            : {}),
        },
      );
      if (operation === "retrain") {
        setJob(response);
      } else {
        await reloadVersions();
        setMessage(
          operation === "promote"
            ? "Candidate is now the active research model."
            : "Previous research model restored.",
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Model operation failed");
    } finally {
      setPendingAction(null);
    }
  }
  const evaluation = report?.[period];
  const chart = evaluation?.strategy.equity.map((point, i) => ({
    ...point,
    date: point.timestamp.slice(0, 10),
    buy_hold: evaluation.baselines.buy_hold.equity[i]?.equity,
    ma: evaluation.baselines.ma_crossover.equity[i]?.equity,
  }));
  const reliability = evaluation?.prediction.reliability.map((bin) => ({
    ...bin,
    ideal: bin.probability,
  }));
  const calibrationChart = [
    { probability: 0, ideal: 0 },
    ...(reliability || []),
    { probability: 1, ideal: 1 },
  ];
  const active = Boolean(
    versions.find((version) => version.id === report?.id)?.active,
  );
  const promoting =
    pendingAction?.operation === "promote" && pendingAction.id === report?.id;
  const failedChecks =
    report?.promotion.criteria.filter((criterion) => !criterion.passed)
      .length || 0;
  const promotionReason = promoting
    ? "Promoting this candidate..."
    : active
      ? "This version is already active; no promotion is needed."
      : training
        ? "Promotion unavailable while training is in progress."
        : pendingAction
          ? "Wait for the current model update to finish."
          : report && !report.promotion.eligible
            ? `Promotion unavailable: ${failedChecks} validation check${failedChecks === 1 ? "" : "s"} failed. Review the red Fail rows above.`
            : "Promotion marks this saved version as the active research model. It does not place brokerage orders.";
  return (
    <main className="shell">
      <header>
        <div className="brand">
          <span>◈</span> AI Trading Lab
        </div>
        <div className="pill">Model research · Daily bars</div>
      </header>
      <nav className="workspace-nav">
        <Link href="/">Historical Backtesting</Link>
        <Link href="/research" aria-current="page">
          AI Research
        </Link>
      </nav>
      <div className="tag">Research workspace / Model training</div>
      <h1>Train. Test. Compare.</h1>
      <p>
        Measure prediction quality and the portfolio it produces, after trading
        costs.
      </p>
      <div className="notice">
        {form.provider === "synthetic"
          ? "Synthetic demonstration — use this to test the workflow. Learned performance is not real-market evidence."
          : "Alpaca daily market history — your local backend reads the data using the existing connection."}
      </div>
      <div className="grid">
        <div>
          <form className="panel" id="research-form" onSubmit={train}>
            <fieldset className="research-fields" disabled={!ready || running}>
              <h2>Train a candidate model</h2>
              <label htmlFor="provider">Data source</label>
              <select
                id="provider"
                value={form.provider}
                onChange={(e) => change("provider", e.target.value)}
              >
                <option value="synthetic">Synthetic sample data</option>
                <option value="alpaca">Alpaca historical data</option>
              </select>
              <label htmlFor="asset">Asset</label>
              <select
                id="asset"
                value={form.asset}
                onChange={(e) => change("asset", e.target.value)}
              >
                {["AAPL", "SPY", "BTC/USD", "ETH/USD"].map((asset) => (
                  <option key={asset}>{asset}</option>
                ))}
              </select>
              <div className="pair">
                <div>
                  <label htmlFor="start">History start</label>
                  <input
                    id="start"
                    type="date"
                    required
                    value={form.start}
                    onChange={(e) => change("start", e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor="end">End (exclusive)</label>
                  <input
                    id="end"
                    type="date"
                    required
                    value={form.end}
                    onChange={(e) => change("end", e.target.value)}
                  />
                </div>
              </div>
              <p className="subtle">
                Use at least 2 years for stocks. The final{" "}
                {percent(form.holdout_fraction)} is reserved for the held-out
                test.
              </p>
              <label htmlFor="algorithm">Model</label>
              <select
                id="algorithm"
                value={form.algorithm}
                onChange={(e) =>
                  setForm((old) => ({
                    ...old,
                    algorithm: e.target.value,
                    horizon: e.target.value === "q_learning" ? 1 : old.horizon,
                  }))
                }
              >
                <option value="xgboost">XGBoost · Return + direction</option>
                <option value="q_learning">Q-learning · Experimental RL</option>
              </select>
              {form.algorithm === "xgboost" ? (
                <>
                  <label htmlFor="feature_set">Features to learn from</label>
                  <select
                    id="feature_set"
                    value={form.feature_set}
                    onChange={(e) => change("feature_set", e.target.value)}
                  >
                    <option value="basic">
                      Basic: returns, trend, volatility, RSI
                    </option>
                    <option value="technical">
                      Expanded: also Bollinger bands, PPO, ATR, 50-day trend
                    </option>
                  </select>
                  <label htmlFor="entry_rule">Trading decision rule</label>
                  <select
                    id="entry_rule"
                    value={form.entry_rule}
                    onChange={(e) => change("entry_rule", e.target.value)}
                  >
                    <option value="return_and_probability">
                      Return forecast + direction probability
                    </option>
                    <option value="probability_only">
                      Direction probability only
                    </option>
                  </select>
                  <div className="pair">
                    {numeric("horizon", "Forecast horizon (bars)", 1, 5)}
                    {numeric("max_depth", "Tree depth", 1, 6)}
                    {numeric("n_estimators", "Maximum trees", 20, 600)}
                    {numeric("learning_rate", "Learning rate", 0.01, 0.3, 0.01)}
                  </div>
                  <div className="pair">
                    {numeric(
                      "min_edge_bps",
                      "Extra return hurdle (bps)",
                      0,
                      200,
                    )}
                    {numeric(
                      "probability_threshold",
                      "Entry probability",
                      0.5,
                      0.9,
                      0.01,
                    )}
                  </div>
                  <p className="subtle">
                    {form.entry_rule === "probability_only"
                      ? "Hold while the calibrated up-probability meets the entry threshold; otherwise hold cash. The return hurdle is ignored. Trading costs still apply."
                      : "Enter only when the forecast exceeds modeled costs and the probability threshold."}
                  </p>
                  <details>
                    <summary>XGBoost sampling & regularization</summary>
                    <div className="pair">
                      {numeric(
                        "subsample",
                        "Training row fraction",
                        0.5,
                        1,
                        0.1,
                      )}
                      {numeric(
                        "colsample_bytree",
                        "Feature fraction per tree",
                        0.5,
                        1,
                        0.1,
                      )}
                      {numeric(
                        "min_child_weight",
                        "Minimum child weight",
                        1,
                        20,
                      )}
                      {numeric("reg_lambda", "L2 regularization", 0, 50)}
                    </div>
                    <p className="subtle">
                      Sampling uses a subset of training rows or features for
                      each tree. Higher child weight and regularization restrict
                      fitting; stronger settings do not automatically improve
                      performance.
                    </p>
                  </details>
                </>
              ) : (
                <>
                  {numeric("episodes", "Training episodes", 2, 60)}
                  {numeric(
                    "drawdown_penalty",
                    "Drawdown increase penalty",
                    0,
                    2,
                    0.1,
                  )}
                  <p className="subtle">
                    Actions: cash or long. Reward: net log equity change,
                    including actual broker costs. A penalty of 0 uses net
                    equity alone. Each fold trains a fresh policy.
                  </p>
                </>
              )}
              <div className="pair">
                {numeric("folds", "Walk-forward folds", 2, 5)}
                {numeric("seed", "Random seed", 0, 2147483647)}
                {numeric(
                  "holdout_fraction",
                  "Final test fraction",
                  0.15,
                  0.3,
                  0.05,
                )}
                {numeric(
                  "initial_cash",
                  "Initial capital ($)",
                  100,
                  1000000000,
                  100,
                )}
              </div>
              <details>
                <summary>Execution costs & position sizing</summary>
                <div className="pair">
                  {numeric("commission_bps", "Commission (bps)", 0, 500)}
                  {numeric("fixed_fee", "Fixed fee ($)", 0, 100, 0.01)}
                  {numeric("spread_bps", "Full spread (bps)", 0, 500)}
                  {numeric("slippage_bps", "Slippage (bps)", 0, 500)}
                  {numeric("min_notional", "Minimum order ($)", 0, 1000000, 1)}
                  {numeric("allocation", "Entry allocation", 0.01, 1, 0.01)}
                  {numeric("cash_reserve", "Cash reserve", 0, 0.99, 0.01)}
                  {numeric(
                    "max_participation",
                    "Prior-volume fraction",
                    0.001,
                    1,
                    0.001,
                  )}
                </div>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.fractional}
                    onChange={(e) => change("fractional", e.target.checked)}
                  />{" "}
                  Allow fractional shares
                </label>
              </details>
              <button className="run" disabled={running} aria-busy={training}>
                {training
                  ? searching
                    ? "Parameter search in progress…"
                    : "Training in progress…"
                  : pendingAction
                    ? "Model update in progress…"
                    : "Train & Evaluate"}
              </button>
              {form.algorithm === "xgboost" && (
                <fieldset className="search-settings" disabled={running}>
                  <legend>Automatic parameter search</legend>
                  <label htmlFor="search-trials">Parameter trials (2–48)</label>
                  <input
                    id="search-trials"
                    type="number"
                    min={2}
                    max={48}
                    step={1}
                    required
                    value={searchTrials}
                    onChange={(e) => setSearchTrials(Number(e.target.value))}
                  />
                  <label htmlFor="search-objective">Rank candidates by</label>
                  <select
                    id="search-objective"
                    value={searchObjective}
                    onChange={(e) =>
                      setSearchObjective(
                        e.target.value as "sharpe" | "net_return",
                      )
                    }
                  >
                    <option value="sharpe">Validation Sharpe</option>
                    <option value="net_return">Validation net return</option>
                  </select>
                  <p className="subtle" id="search-help">
                    Compare feature sets, entry rules, tree settings, sampling,
                    regularization, return hurdles and entry probabilities.
                    Passing validation checks rank first, then candidates within
                    risk limits, then your chosen metric. Only the winner is
                    evaluated on the final test.
                  </p>
                  <button
                    type="button"
                    className="run secondary"
                    disabled={running}
                    aria-busy={searching}
                    aria-describedby="search-help"
                    onClick={findParameters}
                  >
                    {searching
                      ? "Searching parameters…"
                      : "Find best parameters"}
                  </button>
                </fieldset>
              )}
              <p className="subtle">
                Chronological splits · Purged labels · Next-open fills
              </p>
            </fieldset>
          </form>
          <div className="panel charts saved-models">
            <h2>Saved model versions</h2>
            {versions.length === 0 && (
              <p>Your trained candidates will appear here.</p>
            )}
            {(showAllVersions ? versions : versions.slice(0, 12)).map(
              (version) => (
                <div
                  className="version"
                  key={version.id}
                  data-version={version.id}
                >
                  <strong>
                    {version.asset} ·{" "}
                    {version.algorithm === "xgboost" ? "XGBoost" : "Q-learning"}
                  </strong>
                  <p>
                    {version.provider} ·{" "}
                    {version.created_at.slice(0, 16).replace("T", " ")}
                    <br />
                    Version {version.id.slice(0, 8)} · Test return{" "}
                    {percent(version.net_return)} ·{" "}
                    {version.active ? "Active" : "Candidate"}
                  </p>
                  <div className="button-row">
                    <button
                      type="button"
                      onClick={() => load(version.id)}
                      disabled={running}
                    >
                      View results
                    </button>
                    <button
                      type="button"
                      onClick={() => load(version.id, true)}
                      disabled={running}
                    >
                      Use settings
                    </button>
                    {version.rollback_available && !version.active && (
                      <button
                        type="button"
                        onClick={() => action(version.id, "rollback")}
                        disabled={running}
                        aria-busy={
                          pendingAction?.operation === "rollback" &&
                          pendingAction.id === version.id
                        }
                      >
                        {pendingAction?.operation === "rollback" &&
                        pendingAction.id === version.id
                          ? "Restoring…"
                          : "Restore version"}
                      </button>
                    )}
                  </div>
                </div>
              ),
            )}
          </div>
          {versions.length > 12 && (
            <button
              type="button"
              onClick={() => setShowAllVersions(!showAllVersions)}
            >
              {showAllVersions
                ? "Show fewer models"
                : `Show all ${versions.length} models`}
            </button>
          )}
          <div className="panel charts saved-searches">
            <h2>Saved parameter searches</h2>
            {searchRuns.length === 0 && (
              <p>Your search history will appear here.</p>
            )}
            {(showAllSearches ? searchRuns : searchRuns.slice(0, 6)).map(
              (run) => (
                <div className="version" key={run.id}>
                  <strong>
                    {run.asset} ·{" "}
                    {run.provider === "synthetic" ? "SYNTHETIC" : "Alpaca"}
                  </strong>
                  <p>
                    {run.trials} trials ·{" "}
                    {run.objective === "sharpe" ? "Sharpe" : "Net return"} ·{" "}
                    {run.status}
                    <br />
                    {run.created_at.slice(0, 16).replace("T", " ")}
                  </p>
                  {run.parameters && (
                    <p className="best-parameters">
                      Chosen: depth {run.parameters.max_depth},{" "}
                      {run.parameters.n_estimators} trees, rate{" "}
                      {run.parameters.learning_rate}, entry{" "}
                      {percent(run.parameters.probability_threshold)}
                    </p>
                  )}
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => loadSearch(run.id)}
                  >
                    View search
                  </button>
                </div>
              ),
            )}
            {searchRuns.length > 6 && (
              <button
                type="button"
                onClick={() => setShowAllSearches(!showAllSearches)}
              >
                {showAllSearches
                  ? "Show fewer searches"
                  : `Show all ${searchRuns.length} searches`}
              </button>
            )}
          </div>
        </div>
        <section aria-live="polite">
          {error && (
            <div className="notice error" role="alert">
              {error}
            </div>
          )}
          {message && <div className="notice">{message}</div>}
          {training && (
            <div className="panel empty">
              <div>
                <h2>
                  {searching ? "Searching parameters" : "Training and testing"}
                </h2>
                <p role="status">
                  {job && ["queued", "running"].includes(job.status)
                    ? job.progress
                    : searching
                      ? "Submitting parameter search…"
                      : "Submitting training job…"}
                </p>
                <p className="subtle">
                  The backend keeps working if you leave this page.
                </p>
              </div>
            </div>
          )}
          {report && (
            <ModelSettings
              config={report.config}
              id={report.id}
              training={report.training}
              search={report.search}
              retraining={report.retraining}
              validationReturn={
                report.validation.strategy.metrics.net_return ?? 0
              }
              testReturn={report.holdout.strategy.metrics.net_return ?? 0}
              running={running}
              onApply={() => useSettings(report.config)}
              onViewParent={(id) => load(id)}
              onViewSearch={(id) => loadSearch(id)}
            />
          )}
          {searchReport && (
            <SearchResults
              result={searchReport}
              running={running}
              onApply={applyWinner}
            />
          )}
          {!report && !searchReport && !running && (
            <div className="panel empty">
              <div>
                <div className="mark">⌁</div>
                <h2>Predictive accuracy is only the start</h2>
                <p>
                  Train a model to see forecast errors, calibration, trading
                  performance,
                  <br />
                  market regimes and baseline comparisons.
                </p>
              </div>
            </div>
          )}
          {report && evaluation && (
            <>
              <div className="panel charts">
                <h2>
                  {report.config.backtest.asset} ·{" "}
                  {report.config.algorithm === "xgboost"
                    ? "XGBoost"
                    : "Experimental Q-learning"}{" "}
                  ·{" "}
                  {report.provenance.synthetic ? "SYNTHETIC" : "Alpaca history"}
                </h2>
                <p>
                  {report.config.backtest.start} to {report.config.backtest.end}{" "}
                  · {report.provenance.rows} bars
                  <br />
                  Trained through {report.training.trained_through.slice(
                    0,
                    10,
                  )}{" "}
                  · Final test begins{" "}
                  {report.training.holdout_start.slice(0, 10)}
                </p>
                <div className="button-row">
                  <button
                    type="button"
                    aria-pressed={period === "holdout"}
                    aria-describedby="evaluation-period-help"
                    onClick={() => setPeriod("holdout")}
                  >
                    Final held-out test
                  </button>
                  <button
                    type="button"
                    aria-pressed={period === "validation"}
                    aria-describedby="evaluation-period-help"
                    onClick={() => setPeriod("validation")}
                  >
                    Walk-forward validation
                  </button>
                </div>
                <p className="subtle" id="evaluation-period-help">
                  {period === "holdout"
                    ? "The newest reserved data tests one frozen model trained only on earlier history. These results do not decide promotion. Repeatedly changing settings based on this test uses up its independence."
                    : "Each fold trains a fresh model on earlier history and tests the next unseen block. Training history expands between folds; the combined results are used for the promotion checks."}
                </p>
              </div>
              <h2>
                {period === "holdout"
                  ? "Final held-out test"
                  : "Walk-forward validation"}{" "}
                · {evaluation.prediction.samples} predictions
              </h2>
              <div className="stats">
                {[
                  [
                    "Net return",
                    percent(evaluation.strategy.metrics.net_return),
                  ],
                  [
                    "Maximum drawdown",
                    percent(evaluation.strategy.metrics.max_drawdown),
                  ],
                  ["Sharpe ratio", decimal(evaluation.strategy.metrics.sharpe)],
                  ["Forecast MAE", percent(evaluation.prediction.mae)],
                ].map(([label, value]) => (
                  <div className="stat" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
              {evaluation.activity && <Activity data={evaluation.activity} />}
              <div className="panel charts">
                <h2>AI portfolio vs simple baselines</h2>
                <div className="legend">
                  <b>● AI strategy</b>● Buy and hold · ● Moving average
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chart}>
                      <CartesianGrid stroke="#233146" vertical={false} />
                      <XAxis
                        dataKey="date"
                        minTickGap={80}
                        tick={{ fill: "#8495af", fontSize: 10 }}
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        tick={{ fill: "#8495af", fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#172235",
                          border: "1px solid #35435a",
                        }}
                      />
                      <Line
                        isAnimationActive={false}
                        name="AI strategy"
                        dataKey="equity"
                        stroke="#64dec2"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        isAnimationActive={false}
                        name="Buy and hold"
                        dataKey="buy_hold"
                        stroke="#7288b0"
                        dot={false}
                      />
                      <Line
                        isAnimationActive={false}
                        name="Moving average"
                        dataKey="ma"
                        stroke="#d3b680"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Net return</th>
                        <th>Drawdown</th>
                        <th>Sharpe</th>
                        <th>Turnover</th>
                        <th>Costs ($)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries({
                        AI: evaluation.strategy,
                        ...evaluation.baselines,
                      }).map(([name, simulation]) => (
                        <tr key={name}>
                          <td>{names[name] || name}</td>
                          <td>{percent(simulation.metrics.net_return)}</td>
                          <td>{percent(simulation.metrics.max_drawdown)}</td>
                          <td>{decimal(simulation.metrics.sharpe)}</td>
                          <td>{decimal(simulation.metrics.turnover, 2)}×</td>
                          <td>
                            {decimal(simulation.metrics.transaction_costs, 2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="panel charts">
                <h2>Prediction quality</h2>
                {report.config.algorithm === "q_learning" ? (
                  <p>
                    Q-learning chooses portfolio actions. MAE, direction
                    prediction and probability calibration do not apply.
                    Evaluate its net returns, risk and baseline comparisons.
                  </p>
                ) : (
                  <>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Forecast</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>Directional accuracy</th>
                            <th>Rank correlation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries({
                            XGBoost: evaluation.prediction,
                            ...evaluation.prediction_baselines,
                          }).map(([name, p]) => (
                            <tr key={name}>
                              <td>{names[name] || name}</td>
                              <td>{percent(p.mae)}</td>
                              <td>{percent(p.rmse)}</td>
                              <td>{percent(p.directional_accuracy)}</td>
                              <td>{decimal(p.rank_correlation)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="subtle">
                      MAE/RMSE measure return error in percentage points. Rank
                      correlation measures temporal ordering within this asset.
                    </p>
                    <h2>Probability calibration</h2>
                    <p>
                      Brier score: {decimal(evaluation.prediction.brier_score)}{" "}
                      · Log loss: {decimal(evaluation.prediction.log_loss)} ·
                      Calibration error:{" "}
                      {percent(evaluation.prediction.calibration_error)}
                    </p>
                    <div className="chart" style={{ height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={calibrationChart}>
                          <CartesianGrid stroke="#233146" />
                          <XAxis
                            dataKey="probability"
                            type="number"
                            domain={[0, 1]}
                            tick={{ fill: "#8495af", fontSize: 10 }}
                          />
                          <YAxis
                            domain={[0, 1]}
                            tick={{ fill: "#8495af", fontSize: 10 }}
                          />
                          <Tooltip contentStyle={{ background: "#172235" }} />
                          <Line
                            isAnimationActive={false}
                            name="Observed up frequency"
                            dataKey="observed_frequency"
                            stroke="#64dec2"
                          />
                          <Line
                            isAnimationActive={false}
                            name="Perfect calibration"
                            dataKey="ideal"
                            stroke="#7288b0"
                            dot={false}
                            strokeDasharray="5 4"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Mean predicted probability</th>
                            <th>Observed up frequency</th>
                            <th>Samples</th>
                          </tr>
                        </thead>
                        <tbody>
                          {reliability?.map((bin, i) => (
                            <tr key={i}>
                              <td>{percent(bin.probability)}</td>
                              <td>{percent(bin.observed_frequency)}</td>
                              <td>{bin.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
              <div className="panel charts">
                <h2>Stability across market regimes</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Regime</th>
                        <th>Intervals</th>
                        <th>MAE</th>
                        <th>Direction</th>
                        <th>Attributed return</th>
                        <th>Drawdown</th>
                        <th>Sharpe</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.regimes.map((r) => (
                        <tr key={r.regime}>
                          <td>
                            {r.regime}
                            {!r.sufficient && " · small sample"}
                          </td>
                          <td>{r.intervals}</td>
                          <td>{percent(r.prediction.mae)}</td>
                          <td>{percent(r.prediction.directional_accuracy)}</td>
                          <td>{percent(r.net_return)}</td>
                          <td>{percent(r.max_drawdown)}</td>
                          <td>{decimal(r.sharpe)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="subtle">
                  Regimes use past trend and a training-only volatility
                  threshold. Returns describe attributed intervals, not separate
                  portfolios.
                </p>
              </div>
              <div className="panel charts">
                <h2>Walk-forward folds</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Fold</th>
                        <th>Train rows</th>
                        <th>Last label known</th>
                        <th>First prediction</th>
                        <th>MAE</th>
                        <th>Net return</th>
                        <th>Drawdown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.walk_forward.map((f) => (
                        <tr key={f.fold}>
                          <td>{f.fold}</td>
                          <td>{f.training_rows}</td>
                          <td>{f.training_label_end.slice(0, 10)}</td>
                          <td>{f.evaluation_start.slice(0, 10)}</td>
                          <td>{percent(f.prediction.mae)}</td>
                          <td>{percent(f.strategy.net_return)}</td>
                          <td>{percent(f.strategy.max_drawdown)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              {!!report.importance.length && (
                <div className="panel charts">
                  <h2>Feature importance</h2>
                  {report.importance.map((f) => (
                    <div className="feature-row" key={f.feature}>
                      <span>{f.feature.replaceAll("_", " ")}</span>
                      <meter min={0} max={1} value={f.importance} />
                      <span>{percent(f.importance)}</span>
                    </div>
                  ))}
                  <p className="subtle">
                    Tree gain importance explains model usage; it does not
                    establish causation or future profitability.
                  </p>
                </div>
              )}
              <div className="panel charts">
                <h2>Model drift</h2>
                <p>
                  Recent / earlier test MAE:{" "}
                  {decimal(report.drift.recent_mae_ratio)}
                  {report.drift.deterioration && " · deterioration alert"}
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Distribution shift (PSI)</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.drift.features.map((f) => (
                        <tr key={f.feature}>
                          <td>{f.feature.replaceAll("_", " ")}</td>
                          <td>{decimal(f.psi)}</td>
                          <td>
                            {f.flagged
                              ? "Shift detected"
                              : "Below alert threshold"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="subtle">{report.drift.note}</p>
              </div>
              <div className="panel charts">
                <h2>Retraining & model acceptance</h2>
                <p>
                  {active
                    ? "This is the active research model."
                    : report.promotion.eligible
                      ? "This candidate passed the validation criteria."
                      : "This candidate has not passed every validation criterion."}
                </p>
                <ul className="criteria">
                  {report.promotion.criteria.map((c) => (
                    <li
                      key={c.name}
                      className={c.passed ? "positive" : "negative"}
                    >
                      {c.passed ? "Pass" : "Fail"} · {c.name}
                    </li>
                  ))}
                </ul>
                {report.promotion.incumbent_comparison && (
                  <p>
                    Active-model comparison:{" "}
                    {report.promotion.incumbent_comparison.samples} new forward
                    validation samples
                    {!report.promotion.incumbent_comparison.sufficient &&
                      " · more new data required"}
                    .
                  </p>
                )}
                <div className="button-row">
                  <button
                    type="button"
                    disabled={running}
                    aria-busy={
                      pendingAction?.operation === "retrain" &&
                      pendingAction.id === report.id
                    }
                    onClick={() => action(report.id, "retrain")}
                  >
                    {pendingAction?.operation === "retrain"
                      ? "Submitting retraining…"
                      : "Retrain through today"}
                  </button>
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => action(report.id, "retrain", "same_dates")}
                  >
                    Re-run saved dates
                  </button>
                  <button
                    type="button"
                    disabled={running || !report.promotion.eligible || active}
                    aria-busy={promoting}
                    aria-describedby="promotion-status"
                    onClick={() => action(report.id, "promote")}
                  >
                    {promoting
                      ? "Promoting…"
                      : active
                        ? "Already active"
                        : "Promote candidate"}
                  </button>
                </div>
                <p className="subtle" id="promotion-status" role="status">
                  {promotionReason}
                </p>
                <p className="subtle">
                  Both buttons use this model's saved parameters, rather than
                  unsaved form edits. Re-run saved dates keeps{" "}
                  {report.config.backtest.start} to {report.config.backtest.end}{" "}
                  and uses cached history with the current code. Retrain through
                  today refreshes history and extends the end date, which moves
                  the validation and final-test windows. Both preserve this
                  version; a changed period can produce a loss without any
                  parameters changing.
                </p>
              </div>
              {report.config.algorithm === "xgboost" && (
                <div className="panel charts">
                  <h2>Latest saved forecast</h2>
                  <p>
                    {report.latest_prediction.signal_time
                      .slice(0, 16)
                      .replace("T", " ")}{" "}
                    UTC · Expected return:{" "}
                    {percent(report.latest_prediction.predicted_return)} ·
                    Probability of an up move:{" "}
                    {percent(report.latest_prediction.probability_up)}
                  </p>
                  <p className="subtle">{report.latest_prediction.context}</p>
                </div>
              )}
              <div className="panel charts">
                <h2>Recent out-of-sample predictions</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Signal date (UTC)</th>
                        <th>Earliest entry</th>
                        <th>Actual return</th>
                        <th>Forecast</th>
                        <th>Up probability</th>
                        <th>Target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.predictions.slice(-12).map((p) => (
                        <tr key={p.signal_time}>
                          <td>{p.signal_time.slice(0, 10)}</td>
                          <td>{p.entry_time.slice(0, 10)}</td>
                          <td>{percent(p.actual_return)}</td>
                          <td>{percent(p.predicted_return)}</td>
                          <td>{percent(p.probability_up)}</td>
                          <td>
                            {p.target == null
                              ? "N/A"
                              : p.target
                                ? "Long"
                                : "Cash"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <details>
                  <summary>Fills · {evaluation.strategy.trades.length}</summary>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Fill (UTC)</th>
                          <th>Side</th>
                          <th>Quantity</th>
                          <th>Price ($)</th>
                          <th>Cost ($)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evaluation.strategy.trades.map((t, i) => (
                          <tr key={i}>
                            <td>{t.timestamp.slice(0, 16)}</td>
                            <td>{t.side}</td>
                            <td>{decimal(t.quantity, 6)}</td>
                            <td>{decimal(t.price, 2)}</td>
                            <td>{decimal(t.total_cost, 2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
                <details>
                  <summary>Assumptions & reproducibility</summary>
                  <p>
                    Model version: {report.id}
                    <br />
                    Dataset: {report.provenance.dataset_hash}
                  </p>
                  {[...report.warnings, ...report.assumptions].map((text) => (
                    <p key={text}>{text}</p>
                  ))}
                </details>
              </div>
            </>
          )}
        </section>
      </div>
      <footer>
        AI Trading Lab · Local model research · Saved candidates and simulated
        portfolios
      </footer>
    </main>
  );
}
