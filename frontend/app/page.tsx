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
  AreaChart,
  Area,
} from "recharts";
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
type Point = {
  timestamp: string;
  equity: number;
  cash: number;
  drawdown: number;
};
type Fill = {
  timestamp: string;
  signal_timestamp: string;
  side: string;
  quantity: number;
  price: number;
  commission: number;
  total_cost: number;
};
type Result = {
  id: string;
  metrics: Record<string, number | null>;
  equity: Point[];
  trades: Fill[];
  benchmark: { equity: Point[]; metrics: Record<string, number | null> };
  provenance: { synthetic: boolean; rows: number; dataset_hash: string };
  warnings: string[];
  assumptions: string[];
  config: Record<string, unknown>;
  rejected_orders: { timestamp: string; reason: string }[];
  corporate_actions: {
    id: string;
    kind: string;
    timestamp: string;
    quantity: number;
    amount: number;
    ratio: number;
  }[];
};
const defaults = {
  asset: "AAPL",
  timeframe: "1d",
  provider: "synthetic",
  start: "2024-01-02",
  end: "2024-06-30",
  strategy: "ma_crossover",
  fast: 10,
  slow: 30,
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
};
const money = (v: number | null | undefined) =>
  v == null
    ? "Undefined"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      }).format(v);
const pct = (v: number | null | undefined) =>
  v == null ? "Undefined" : `${(v * 100).toFixed(2)}%`;
export default function Page() {
  const [form, setForm] = useState(defaults),
    [result, setResult] = useState<Result | null>(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [connected, setConnected] = useState(false);
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setConnected(r.ok))
      .catch(() => setConnected(false));
  }, []);
  const change = (
    key: keyof typeof defaults,
    value: string | number | boolean,
  ) => setForm((old) => ({ ...old, [key]: value }));
  const number = (
    key: keyof typeof defaults,
    label: string,
    step = "1",
    min = "0",
    max?: string,
  ) => (
    <div>
      <label htmlFor={key}>{label}</label>
      <input
        id={key}
        type="number"
        required
        step={step}
        min={min}
        max={max}
        value={String(form[key])}
        onChange={(e) => change(key, Number(e.target.value))}
      />
    </div>
  );
  async function run(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API}/api/backtests`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (!response.ok) {
        const d = data.detail;
        throw new Error(
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d.map((x: { msg: string }) => x.msg).join("; ")
              : d?.message || "Backtest request failed",
        );
      }
      setResult(data);
      setConnected(true);
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "Cannot reach the backend. Start FastAPI on http://localhost:8000 and try again."
          : e instanceof Error
            ? e.message
            : "Unable to run experiment",
      );
    } finally {
      setLoading(false);
    }
  }
  const chart = result?.equity.map((p, i) => ({
    ...p,
    label: p.timestamp.slice(0, 16).replace("T", " "),
    benchmark: result.benchmark.equity[i]?.equity,
  }));
  return (
    <main className="shell">
      <header>
        <div className="brand">
          <span>◈</span> AI Trading Lab
        </div>
        <div className="pill">
          {connected ? "Backend connected" : "Backend unavailable"} · Research
          only
        </div>
      </header>
      <nav className="workspace-nav">
        <Link href="/" aria-current="page">
          Historical Backtesting
        </Link>
        <Link href="/research">AI Research</Link>
      </nav>
      <div className="tag">Research workspace / Historical simulation</div>
      <h1>Test ideas. Measure outcomes.</h1>
      <p>
        Compare strategies under explicit execution costs, with reproducible
        experiments.
      </p>
      <div className="notice">
        {form.provider === "synthetic"
          ? "Synthetic demonstration mode — generated prices, not real market history or investment performance."
          : "Real market history — Bitcoin/Ethereum need no account. Daily stocks need Alpaca API keys and supported split/dividend records."}
      </div>
      <div className="grid">
        <form className="panel" onSubmit={run}>
          <h2>Configure experiment</h2>
          <label htmlFor="provider">Data source</label>
          <select
            id="provider"
            value={form.provider}
            onChange={(e) => change("provider", e.target.value)}
          >
            <option value="synthetic">Synthetic sample data</option>
            <option value="alpaca">Alpaca historical data</option>
          </select>
          <div className="pair">
            <div>
              <label htmlFor="asset">Asset</label>
              <select
                id="asset"
                value={form.asset}
                onChange={(e) => change("asset", e.target.value)}
              >
                <optgroup label="Stocks / ETFs">
                  <option>AAPL</option>
                  <option>SPY</option>
                </optgroup>
                <optgroup label="Crypto · 24/7">
                  <option>BTC/USD</option>
                  <option>ETH/USD</option>
                </optgroup>
              </select>
            </div>
            <div>
              <label htmlFor="timeframe">Timeframe</label>
              <select
                id="timeframe"
                value={form.timeframe}
                onChange={(e) => change("timeframe", e.target.value)}
              >
                {["1d", "1h", "15m", "5m"].map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="pair">
            <div>
              <label htmlFor="start">Start date</label>
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
          <label htmlFor="strategy">Strategy</label>
          <select
            id="strategy"
            value={form.strategy}
            onChange={(e) => change("strategy", e.target.value)}
          >
            <option value="ma_crossover">Moving average crossover</option>
            <option value="buy_hold">Buy and hold</option>
          </select>
          {form.strategy === "ma_crossover" && (
            <div className="pair">
              {number("fast", "Fast window", "1", "2", "200")}
              {number("slow", "Slow window", "1", "3", "500")}
            </div>
          )}
          {number(
            "initial_cash",
            "Initial capital ($)",
            "100",
            "100",
            "1000000000",
          )}
          <div className="pair">
            {number("commission_bps", "Commission (bps)", "1", "0", "500")}
            {number("fixed_fee", "Fixed fee ($)", ".01", "0", "100")}
            {number("spread_bps", "Full spread (bps)", "1", "0", "500")}
            {number("slippage_bps", "Slippage (bps)", "1", "0", "500")}
          </div>
          <details>
            <summary>Position sizing & liquidity</summary>
            <div className="pair">
              {number("allocation", "Max entry allocation", ".01", ".01", "1")}
              {number("cash_reserve", "Cash reserve", ".01", "0", ".99")}
              {number(
                "max_participation",
                "Prior-volume fraction",
                ".001",
                ".001",
                "1",
              )}
              {number("min_notional", "Minimum order ($)", "1", "0", "1000000")}
            </div>
            <label>
              <input
                type="checkbox"
                checked={form.fractional}
                onChange={(e) => change("fractional", e.target.checked)}
              />
              Allow fractional shares / units
            </label>
          </details>
          <button className="run" disabled={loading} aria-busy={loading}>
            {loading ? "Running simulation…" : "Run Backtest →"}
          </button>
          <p className="subtle">
            Long-only · No leverage · Next-bar open execution
          </p>
          {form.provider === "alpaca" && (
            <details open>
              <summary>Connect real stock data</summary>
              <p>
                Bitcoin and Ethereum history work without API keys. For stocks,
                create an Alpaca account, generate API keys, and add them to the
                local .env file. Restart the backend after saving. Choose 1d for
                real stocks.
              </p>
              <p>
                <a
                  href="https://docs.alpaca.markets/us/docs/getting-started-with-alpaca-market-data"
                  target="_blank"
                  rel="noreferrer"
                >
                  Alpaca account and API-key guide ↗
                </a>
              </p>
              <p>
                Your keys stay in the Python backend. This app only reads market
                data.
              </p>
            </details>
          )}
        </form>
        <section aria-live="polite">
          {error && (
            <div role="alert" className="notice error">
              {error}
            </div>
          )}
          {loading && (
            <div className="panel empty">
              <p>Computing portfolio accounting and benchmark…</p>
            </div>
          )}
          {!result && !loading && (
            <div className="panel empty">
              <div>
                <div className="mark">⌁</div>
                <h2>Your next experiment starts here</h2>
                <p>
                  Choose your settings and run a backtest.
                  <br />
                  Equity, costs, and completed fills will appear here.
                </p>
              </div>
            </div>
          )}
          {result && (
            <>
              <div className="stats">
                {[
                  ["Net return", pct(result.metrics.net_return)],
                  ["Maximum drawdown", pct(result.metrics.max_drawdown)],
                  [
                    "Transaction costs",
                    money(result.metrics.transaction_costs),
                  ],
                  ["Final equity", money(result.metrics.final_equity)],
                ].map(([label, value]) => (
                  <div className="stat" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
              <div className="panel charts">
                <h2>Portfolio equity</h2>
                <div className="legend">
                  <b>● Strategy</b>● Buy and hold ·{" "}
                  {pct(result.benchmark.metrics.net_return)}
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chart}>
                      <CartesianGrid stroke="#233146" vertical={false} />
                      <XAxis
                        dataKey="label"
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
                        name="Strategy"
                        dataKey="equity"
                        stroke="#64dec2"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        isAnimationActive={false}
                        name="Buy and hold"
                        dataKey="benchmark"
                        stroke="#7288b0"
                        dot={false}
                        strokeDasharray="5 4"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="panel charts">
                <h2>Drawdown</h2>
                <div style={{ height: 160 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chart}>
                      <XAxis dataKey="label" hide />
                      <YAxis
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                        tick={{ fill: "#8495af", fontSize: 10 }}
                      />
                      <Tooltip contentStyle={{ background: "#172235" }} />
                      <Area
                        isAnimationActive={false}
                        dataKey="drawdown"
                        stroke="#ee939e"
                        fill="#ee939e"
                        fillOpacity={0.12}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="panel charts">
                <h2>Execution history · {result.trades.length} fills</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Fill time (UTC)</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Fill price</th>
                        <th>Total modeled cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.trades.map((t, i) => (
                        <tr key={i}>
                          <td>{t.timestamp.slice(0, 16).replace("T", " ")}</td>
                          <td
                            className={
                              t.side === "buy" ? "positive" : "negative"
                            }
                          >
                            {t.side.toUpperCase()}
                          </td>
                          <td>{t.quantity.toFixed(6)}</td>
                          <td>{money(t.price)}</td>
                          <td>{money(t.total_cost)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!result.trades.length && (
                    <p>No completed fills for these settings.</p>
                  )}
                </div>
                <details>
                  <summary>More statistics & reproducibility</summary>
                  <p>
                    Sharpe: {result.metrics.sharpe?.toFixed(3) ?? "Undefined"} ·
                    Sortino: {result.metrics.sortino?.toFixed(3) ?? "Undefined"}{" "}
                    · Closed trades: {result.metrics.closed_trades} · Win rate:{" "}
                    {pct(result.metrics.win_rate)}
                  </p>
                  <p>
                    Gross same-fills return:{" "}
                    {pct(result.metrics.gross_return_same_fills)} · Annualized
                    return: {pct(result.metrics.annualized_return)} ·
                    Volatility: {pct(result.metrics.annualized_volatility)} ·
                    Turnover: {result.metrics.turnover?.toFixed(2)}
                  </p>
                  <p>
                    Experiment: {result.id}
                    <br />
                    Dataset: {result.provenance.dataset_hash}
                    <br />
                    {result.provenance.rows} bars ·{" "}
                    {result.provenance.synthetic
                      ? "Synthetic"
                      : "Alpaca historical"}
                  </p>
                  <pre style={{ whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(result.config, null, 2)}
                  </pre>
                  <p>
                    Dividend income: {money(result.metrics.dividend_income)} ·
                    Unpaid dividends:{" "}
                    {money(result.metrics.dividend_receivable)}
                  </p>
                  {result.corporate_actions.map((event, i) => (
                    <p key={i}>
                      {event.timestamp.slice(0, 10)} ·{" "}
                      {event.kind.replaceAll("_", " ")} ·{" "}
                      {event.kind === "split"
                        ? event.ratio + ":1 share ratio"
                        : money(event.amount)}
                    </p>
                  ))}
                  {result.assumptions.map((a) => (
                    <p key={a}>{a}</p>
                  ))}
                  {result.warnings.map((w) => (
                    <p key={w}>{w}</p>
                  ))}
                  {result.rejected_orders.map((o, i) => (
                    <p key={i}>
                      {o.timestamp}: {o.reason}
                    </p>
                  ))}
                </details>
              </div>
            </>
          )}
        </section>
      </div>
      <footer>
        AI Trading Lab · Research software · All trading calculations run in the
        Python backend.
      </footer>
    </main>
  );
}
