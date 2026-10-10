"use client";
export type SearchParameters = {
  n_estimators: number;
  max_depth: number;
  learning_rate: number;
  min_edge_bps: number;
  probability_threshold: number;
};
export type SearchTrial = {
  trial: number;
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
  };
  space: Record<string, number[]>;
  selection_policy: string;
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
  winner_id: string | null;
};
const percent = (value: number | null | undefined) =>
  value == null ? "N/A" : (value * 100).toFixed(2) + "%";
const decimal = (value: number | null | undefined, digits = 3) =>
  value == null ? "N/A" : value.toFixed(digits);
const labels: Record<keyof SearchParameters, string> = {
  n_estimators: "Maximum trees",
  max_depth: "Tree depth",
  learning_rate: "Learning rate",
  min_edge_bps: "Extra return hurdle (bps)",
  probability_threshold: "Entry probability",
};
export default function SearchResults({
  result,
  running,
  onApply,
}: {
  result: SearchReport;
  running: boolean;
  onApply: () => void;
}) {
  const winner = result.trials.find(
    (trial) => trial.trial === result.selected_trial,
  );
  return (
    <div className="panel charts search-results">
      <h2>{winner ? "Best candidate found" : "Parameter search results"}</h2>
      <p>
        {result.config.research.backtest.asset} ·{" "}
        {result.provenance.synthetic
          ? "SYNTHETIC demonstration"
          : "Alpaca history"}
        <br />
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
            Trial {winner.trial}: depth {winner.parameters.max_depth} ·{" "}
            {winner.parameters.n_estimators} trees · Learning rate{" "}
            {winner.parameters.learning_rate} · Extra hurdle{" "}
            {winner.parameters.min_edge_bps} bps · Entry probability{" "}
            {percent(winner.parameters.probability_threshold)}
          </p>
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
          <div className="button-row">
            <button type="button" disabled={running} onClick={onApply}>
              Use best-found settings
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
              <th>Depth / trees</th>
              <th>Learning rate</th>
              <th>Hurdle / entry</th>
              <th>Validation checks</th>
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
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary>Search ranges</summary>
        <ul>
          {Object.entries(result.space).map(([key, values]) => (
            <li key={key}>
              {labels[key as keyof SearchParameters]}: {values.join(", ")}
            </li>
          ))}
        </ul>
        <p className="subtle">
          Your current settings are always included as trial 1. Additional
          combinations are sampled with the same random seed. The data, horizon,
          costs and validation split stay fixed.
        </p>
      </details>
      <p className="subtle">{result.warnings.join(" ")}</p>
      {result.error && <p className="negative">{result.error}</p>}
    </div>
  );
}
