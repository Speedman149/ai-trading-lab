"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
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
  created_at: string;
  config: {
    algorithm: string;
    horizon: number;
    backtest: { asset: string; start: string; end: string; provider: string };
  };
  provenance: { synthetic: boolean; rows: number; dataset_hash: string };
  training: {
    trained_through: string;
    holdout_start: string;
    rows: number;
    episode_rewards?: number[];
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
const defaults = {
  asset: "AAPL",
  provider: "synthetic",
  start: "2022-01-03",
  end: "2025-01-01",
  algorithm: "xgboost",
  horizon: 1,
  folds: 3,
  holdout_fraction: 0.2,
  n_estimators: 160,
  max_depth: 3,
  learning_rate: 0.05,
  min_edge_bps: 5,
  probability_threshold: 0.55,
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
  const [report, setReport] = useState<Report | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [period, setPeriod] = useState<"holdout" | "validation">("holdout");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const running =
    submitting || job?.status === "running" || job?.status === "queued";
  const reloadVersions = async () =>
    setVersions(await request<Version[]>("/api/research/versions"));
  useEffect(() => {
    let mounted = true;
    Promise.all([
      request<Version[]>("/api/research/versions"),
      request<Job[]>("/api/research/jobs"),
    ])
      .then(([saved, jobs]) => {
        if (mounted) {
          setVersions(saved);
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
          const saved = await request<Version[]>("/api/research/versions");
          if (!cancelled) {
            setReport(result);
            setVersions(saved);
            setJob(next);
            setPeriod("holdout");
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
  const change = (key: keyof typeof defaults, value: string | number) =>
    setForm((old) => ({ ...old, [key]: value }));
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
        value={form[key]}
        onChange={(e) => change(key, Number(e.target.value))}
      />
    </div>
  );
  async function train(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMessage("");
    setSubmitting(true);
    setReport(null);
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
      ...model
    } = form;
    try {
      setJob(
        await request<Job>("/api/research/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
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
            },
          }),
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setSubmitting(false);
    }
  }
  async function load(id: string) {
    setError("");
    setMessage("");
    try {
      setReport(await request<Report>("/api/research/versions/" + id));
      setPeriod("holdout");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load model");
    }
  }
  async function action(
    id: string,
    operation: "retrain" | "promote" | "rollback",
  ) {
    setError("");
    setMessage("");
    setSubmitting(true);
    try {
      const response = await request<Job>(
        "/api/research/versions/" + id + "/" + operation,
        { method: "POST" },
      );
      if (operation === "retrain") {
        setJob(response);
        setReport(null);
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
      setSubmitting(false);
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
  const active = versions.find((version) => version.id === report?.id)?.active;
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
          <form className="panel" onSubmit={train}>
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
                <div className="pair">
                  {numeric("horizon", "Forecast horizon (bars)", 1, 5)}
                  {numeric("max_depth", "Tree depth", 1, 6)}
                  {numeric("n_estimators", "Maximum trees", 20, 600)}
                  {numeric("learning_rate", "Learning rate", 0.01, 0.3, 0.01)}
                </div>
                <div className="pair">
                  {numeric("min_edge_bps", "Extra return hurdle (bps)", 0, 200)}
                  {numeric(
                    "probability_threshold",
                    "Entry probability",
                    0.5,
                    0.9,
                    0.01,
                  )}
                </div>
                <p className="subtle">
                  Enter only when the forecast exceeds modeled costs and the
                  probability threshold.
                </p>
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
                  including actual broker costs. A penalty of 0 uses net equity
                  alone. Each fold trains a fresh policy.
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
            </details>
            <button className="run" disabled={running}>
              {running ? "Training in progress…" : "Train & Evaluate"}
            </button>
            <p className="subtle">
              Chronological splits · Purged labels · Next-open fills
            </p>
          </form>
          <div className="panel charts saved-models">
            <h2>Saved model versions</h2>
            {versions.length === 0 && (
              <p>Your trained candidates will appear here.</p>
            )}
            {versions.slice(0, 12).map((version) => (
              <div className="version" key={version.id}>
                <strong>
                  {version.asset} ·{" "}
                  {version.algorithm === "xgboost" ? "XGBoost" : "Q-learning"}
                </strong>
                <p>
                  {version.provider} ·{" "}
                  {version.created_at.slice(0, 16).replace("T", " ")}
                  <br />
                  Test return {percent(version.net_return)} ·{" "}
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
                  {version.rollback_available && !version.active && (
                    <button
                      type="button"
                      onClick={() => action(version.id, "rollback")}
                      disabled={running}
                    >
                      Restore version
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <section aria-live="polite">
          {error && (
            <div className="notice error" role="alert">
              {error}
            </div>
          )}
          {message && <div className="notice">{message}</div>}
          {running && (
            <div className="panel empty">
              <div>
                <h2>Training and testing</h2>
                <p role="status">
                  {job?.progress || "Submitting training job…"}
                </p>
                <p className="subtle">
                  The backend keeps working if you leave this page.
                </p>
              </div>
            </div>
          )}
          {!report && !running && (
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
                    onClick={() => setPeriod("holdout")}
                  >
                    Final held-out test
                  </button>
                  <button
                    type="button"
                    aria-pressed={period === "validation"}
                    onClick={() => setPeriod("validation")}
                  >
                    Walk-forward validation
                  </button>
                </div>
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
                    onClick={() => action(report.id, "retrain")}
                  >
                    Refresh data & retrain
                  </button>
                  <button
                    type="button"
                    disabled={running || !report.promotion.eligible || active}
                    onClick={() => action(report.id, "promote")}
                  >
                    Promote candidate
                  </button>
                </div>
                <p className="subtle">
                  Retraining uses newly completed daily bars, preserves this
                  version and trains a separate candidate. Promotion uses fixed
                  validation criteria. Previously active versions can be
                  restored from the saved list.
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
