"use client";
import { parameterLabels, settingValue } from "./config";
import type { ActivityData } from "./activity";
export const searchMethodLabels = {
  random: "Broad random search",
  coarse_to_fine: "Broad then refine",
  local: "Refine current settings",
};
export type SearchParameters = {
  n_estimators: number;
  max_depth: number;
  learning_rate: number;
  min_edge_bps: number;
  probability_threshold: number;
  feature_set?: string;
  entry_rule?: string;
  subsample?: number;
  colsample_bytree?: number;
  min_child_weight?: number;
  reg_lambda?: number;
};
export type SearchTrial = {
  trial: number;
  phase?: "current" | "broad" | "refine";
  origin_trial?: number | null;
  rank: number | null;
  parameters: SearchParameters;
  status: string;
  error: string | null;
  metrics: Record<string, number | null> | null;
  prediction: {
    mae: number | null;
    directional_accuracy: number | null;
  } | null;
  folds: { fold: number; net_return: number; sharpe: number | null }[];
  promotion: {
    eligible: boolean;
    criteria: { name: string; passed: boolean }[];
  } | null;
  risk_passed: boolean;
  activity?: ActivityData | null;
  baselines?: Record<string, { net_return: number }>;
};
export type SearchReport = {
  id: string;
  status: string;
  created_at: string;
  config: {
    research: Record<string, unknown> & {
      backtest: {
        asset: string;
        provider: string;
        start: string;
        end: string;
        [key: string]: unknown;
      };
    };
    trials: number;
    objective: "sharpe" | "net_return";
    mode?: "random" | "coarse_to_fine" | "local";
  };
  space: Record<string, (string | number)[]>;
  selection_policy: string;
  refinement?: {
    steps: Record<string, { step: number; minimum: number; maximum: number }>;
    broad_trials: number;
    note: string;
  } | null;
  provenance: { synthetic: boolean };
  validation_start: string;
  validation_end: string;
  holdout_start: string;
  winner_id: string | null;
  selected_trial: number | null;
  trials: SearchTrial[];
  warnings: string[];
  error: string | null;
};
export type SearchSummary = {
  id: string;
  created_at: string;
  asset: string;
  provider: string;
  status: string;
  trials: number;
  objective: string;
  mode?: keyof typeof searchMethodLabels;
  winner_id: string | null;
  parameters?: SearchParameters | null;
};
const percent = (value: number | null | undefined) =>
  value == null ? "N/A" : (value * 100).toFixed(2) + "%";
const decimal = (value: number | null | undefined, digits = 3) =>
  value == null ? "N/A" : value.toFixed(digits);
export default function SearchResults({
  result,
  running,
  onApply,
  onRefine,
}: {
  result: SearchReport;
  running: boolean;
  onApply: () => void;
  onRefine: () => void;
}) {
  const winner = result.trials.find(
    (trial) => trial.trial === result.selected_trial,
  );
  return (
    <div className="panel charts search-results" id="search-results">
      <h2>{winner ? "Best candidate found" : "Parameter search results"}</h2>
      <p>
        {result.config.research.backtest.asset} ·{" "}
        {result.provenance.synthetic
          ? "SYNTHETIC demonstration"
          : "Alpaca history"}
        <br />
        {searchMethodLabels[result.config.mode || "random"]} ·{" "}
        {result.trials.length} / {result.config.trials} trials · Ranked by
        validation{" "}
        {result.config.objective === "sharpe" ? "Sharpe" : "net return"}
        <br />
        Validation: {result.validation_start.slice(0, 10)} to{" "}
        {result.validation_end.slice(0, 10)} · Final test begins{" "}
        {result.holdout_start.slice(0, 10)}
      </p>
      {winner && (
        <>
          <p className="best-parameters">
            Chosen parameters · Trial {winner.trial}
          </p>
          <dl className="parameter-grid">
            {Object.entries(winner.parameters).map(([key, value]) => (
              <div key={key}>
                <dt>{parameterLabels[key] || key}</dt>
                <dd>{settingValue(key, value)}</dd>
              </div>
            ))}
          </dl>
          <div className="stats">
            <div className="stat">
              <span>Best validation net return</span>
              <strong>{percent(winner.metrics?.net_return)}</strong>
            </div>
            <div className="stat">
              <span>Best validation Sharpe</span>
              <strong>{decimal(winner.metrics?.sharpe)}</strong>
            </div>
            <div className="stat">
              <span>Best validation drawdown</span>
              <strong>{percent(winner.metrics?.max_drawdown)}</strong>
            </div>
          </div>
          <p>
            {winner.promotion?.eligible
              ? "The winner passed all promotion checks. Its final-test report is below."
              : "This is the best candidate found under the search ranking, but it has not passed every promotion check."}
          </p>
          <p>
            Validation buy-and-hold:{" "}
            {percent(
              (
                winner as SearchTrial & {
                  baselines?: Record<string, { net_return: number }>;
                }
              ).baselines?.buy_hold?.net_return,
            )}{" "}
            · Completed trades: {winner.metrics?.closed_trades ?? "N/A"}
          </p>
          <p className="subtle">
            Gross return on the same fills:{" "}
            {percent(winner.metrics?.gross_return_same_fills)} · Modeled trading
            costs: ${winner.metrics?.transaction_costs?.toFixed(2) ?? "N/A"}.
            Profitable folds:{" "}
            {winner.folds.filter((fold) => fold.net_return > 0).length} /{" "}
            {winner.folds.length}. The net return covers the validation dates
            above, not a monthly or annual rate. Folds start with fresh capital
            and cannot be added together.
          </p>
          {!result.trials.some((trial) => trial.promotion?.eligible) && (
            <p className="notice">
              No candidate passed all validation checks. The selected settings
              are a research candidate, not a validated improvement over the
              baselines.
            </p>
          )}
          <div className="button-row">
            <button type="button" disabled={running} onClick={onApply}>
              Use best-found settings
            </button>
            <button type="button" disabled={running} onClick={onRefine}>
              Refine winning settings
            </button>
          </div>
        </>
      )}
      <p className="subtle" id="search-ranking-help">
        {result.selection_policy}
      </p>
      <p className="subtle">
        Only the winner receives a final held-out test. That test is not used to
        rank these trials.
      </p>
      <p className="subtle">
        Scroll the table horizontally to see all metrics and parameter choices.
      </p>
      <div className="table-wrap">
        <table aria-label="Parameter search leaderboard">
          <thead>
            <tr>
              <th>Rank / trial</th>
              <th>Validation net return</th>
              <th>Sharpe</th>
              <th>Drawdown</th>
              <th>Turnover</th>
              <th>Costs ($)</th>
              <th>Forecast MAE</th>
              <th>Direction accuracy</th>
              <th>Above best baseline</th>
              <th>Fold returns</th>
              <th>Features / entry rule</th>
              <th>Sampling / regularization</th>
              <th>Completed trades</th>
              <th>Depth / trees</th>
              <th>Learning rate</th>
              <th>Hurdle / entry</th>
              <th>Validation checks</th>
              <th>Search phase</th>
            </tr>
          </thead>
          <tbody>
            {result.trials.map((trial) => (
              <tr
                key={trial.trial}
                data-trial={trial.trial}
                aria-selected={trial.trial === result.selected_trial}
              >
                <td>
                  {trial.rank ?? "Failed"} / {trial.trial}
                  {trial.trial === result.selected_trial && " · Best"}
                </td>
                <td>{percent(trial.metrics?.net_return)}</td>
                <td>{decimal(trial.metrics?.sharpe)}</td>
                <td>{percent(trial.metrics?.max_drawdown)}</td>
                <td>
                  {trial.metrics?.turnover == null
                    ? "N/A"
                    : decimal(trial.metrics.turnover) + "x"}
                </td>
                <td>{decimal(trial.metrics?.transaction_costs, 2)}</td>
                <td>{percent(trial.prediction?.mae)}</td>
                <td>{percent(trial.prediction?.directional_accuracy)}</td>
                <td>{percent(trial.metrics?.excess_return)}</td>
                <td>
                  {trial.folds
                    .map((fold) => percent(fold.net_return))
                    .join(" / ") || "N/A"}
                </td>
                <td>
                  {settingValue("feature_set", trial.parameters.feature_set)} /{" "}
                  {settingValue("entry_rule", trial.parameters.entry_rule)}
                </td>
                <td>
                  Rows {trial.parameters.subsample ?? 1} / features{" "}
                  {trial.parameters.colsample_bytree ?? 1}
                  <br />
                  Child {trial.parameters.min_child_weight ?? 5} / L2{" "}
                  {trial.parameters.reg_lambda ?? 5}
                </td>
                <td>{trial.metrics?.closed_trades ?? "N/A"}</td>
                <td>
                  {trial.parameters.max_depth} / {trial.parameters.n_estimators}
                </td>
                <td>{trial.parameters.learning_rate}</td>
                <td>
                  {trial.parameters.min_edge_bps} bps /{" "}
                  {percent(trial.parameters.probability_threshold)}
                </td>
                <td>
                  {trial.error ||
                    (trial.promotion?.eligible
                      ? "All passed"
                      : String(
                          trial.promotion?.criteria.filter(
                            (criterion) => !criterion.passed,
                          ).length ?? 0,
                        ) + " failed")}
                </td>
                <td>
                  {trial.phase === "refine"
                    ? "Refine from trial " + trial.origin_trial
                    : trial.phase || "Broad"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {Object.keys(result.space).length > 0 && (
        <details>
          <summary>Search ranges</summary>
          <ul>
            {Object.entries(result.space).map(([key, values]) => (
              <li key={key}>
                {parameterLabels[key] || key}: {values.join(", ")}
              </li>
            ))}
          </ul>
          <p className="subtle">
            Your current settings are always included as trial 1. Broad
            combinations use the same random seed. The data, horizon, costs and
            validation split stay fixed. Refinement changes numeric settings
            around validation leaders in smaller steps.
          </p>
        </details>
      )}
      {result.refinement && (
        <details>
          <summary>Refinement steps</summary>
          <p className="subtle">{result.refinement.note}</p>
          <p>
            {result.refinement.broad_trials} initial trials ·{" "}
            {result.config.trials - result.refinement.broad_trials} refinement
            trials.
          </p>
          <ul>
            {Object.entries(result.refinement.steps).map(([key, range]) => (
              <li key={key}>
                {parameterLabels[key] || key}: step {range.step}, bounded{" "}
                {range.minimum} to {range.maximum}
              </li>
            ))}
          </ul>
        </details>
      )}
      <p className="subtle">{result.warnings.join(" ")}</p>
      {result.error && <p className="negative">{result.error}</p>}
    </div>
  );
}
