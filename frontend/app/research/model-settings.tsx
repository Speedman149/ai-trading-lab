"use client";
import {
  defaults,
  parameterLabels,
  settingsFromConfig,
  settingValue,
  type ResearchConfig,
} from "./config";
export type Retraining = {
  parent_id: string;
  source_search_id?: string | null;
  previous_start: string;
  previous_end: string;
  current_start: string;
  current_end: string;
  previous_holdout_start: string;
  current_holdout_start: string;
  parameters_preserved: boolean;
  same_validation_dates: boolean;
  same_holdout_dates: boolean;
  previous_validation_return: number;
  previous_test_return: number;
  data_changed: boolean;
  code_changed: boolean;
};
const percent = (value: number) => (value * 100).toFixed(2) + "%";
export default function ModelSettings({
  config,
  id,
  training,
  search,
  retraining,
  validationReturn,
  testReturn,
  running,
  onApply,
  onViewParent,
  onViewSearch,
}: {
  config: ResearchConfig;
  id: string;
  training: {
    regression_trees?: number;
    direction_trees?: number | null;
    best_iteration?: number;
    core_rows?: number;
    early_stopping_rows?: number;
    calibration_rows?: number;
  };
  search?: { id: string; selected_trial?: number } | null;
  retraining?: Retraining | null;
  validationReturn: number;
  testReturn: number;
  running: boolean;
  onApply: () => void;
  onViewParent: (id: string) => void;
  onViewSearch: (id: string) => void;
}) {
  const used = settingsFromConfig(config);
  const keys =
    config.algorithm === "xgboost"
      ? Object.keys(parameterLabels)
      : ["horizon", "folds", "holdout_fraction", "seed"];
  const sourceSearch = search?.id || retraining?.source_search_id;
  function download() {
    const blob = new Blob(
      [
        JSON.stringify(
          { format: "ai-trading-lab-settings-v1", research: config },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ai-trading-lab-${id}-settings.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <div className="panel charts model-settings" id="model-settings">
      <h2>Parameters used for this model</h2>
      <p>
        {config.backtest.asset} ·{" "}
        {config.backtest.provider === "synthetic" ? "SYNTHETIC" : "Alpaca"} ·{" "}
        {config.backtest.start} to {config.backtest.end} (end exclusive)
        <br />
        {search
          ? `Chosen by search${search.selected_trial ? `, trial ${search.selected_trial}` : ""}.`
          : retraining
            ? "Retrained from a saved model."
            : "Trained with the submitted settings."}
      </p>
      <dl className="parameter-grid">
        {keys.map((key) => (
          <div key={key}>
            <dt>{parameterLabels[key]}</dt>
            <dd>{settingValue(key, used[key as keyof typeof defaults])}</dd>
          </div>
        ))}
        {config.algorithm === "xgboost" && (
          <>
            <div>
              <dt>Return trees actually used</dt>
              <dd>
                {training.regression_trees ??
                  (training.best_iteration == null
                    ? "N/A"
                    : training.best_iteration + 1)}
              </dd>
            </div>
            <div>
              <dt>Direction trees actually used</dt>
              <dd>{training.direction_trees ?? "N/A"}</dd>
            </div>
          </>
        )}
      </dl>
      <p className="subtle">
        These are this saved model's settings. Editing the form does not change
        this model. Early stopping can use fewer trees than the configured
        maximum.
      </p>
      {config.algorithm === "xgboost" && training.core_rows != null && (
        <p className="subtle" id="training-sample-help">
          Final-model training uses {training.core_rows} rows to fit trees,{" "}
          {training.early_stopping_rows} later rows to choose the tree count,
          and {training.calibration_rows} separate rows to calibrate
          probabilities. A one- or two-tree model means more trees did not
          improve its internal validation score. Raising the maximum alone may
          leave its predictions unchanged.
        </p>
      )}
      <div className="button-row">
        <button type="button" disabled={running} onClick={onApply}>
          Use this model's settings
        </button>
        <button type="button" onClick={download}>
          Download settings JSON
        </button>
        {sourceSearch && (
          <button
            type="button"
            disabled={running}
            onClick={() => onViewSearch(sourceSearch)}
          >
            View original parameter search
          </button>
        )}
      </div>
      <details>
        <summary>All saved settings & execution costs</summary>
        <pre className="settings-json">{JSON.stringify(config, null, 2)}</pre>
      </details>
      {retraining && (
        <div className="retraining-comparison">
          <h3>What changed when retraining?</h3>
          <p>
            {retraining.parameters_preserved
              ? "Model parameters were preserved."
              : "Model parameters changed."}{" "}
            {!retraining.same_validation_dates || !retraining.same_holdout_dates
              ? "The evaluation dates changed; these returns do not measure performance on the same period."
              : "The evaluation dates match."}
          </p>
          <div className="table-wrap">
            <table aria-label="Retraining comparison">
              <thead>
                <tr>
                  <th>Setting / result</th>
                  <th>Previous version</th>
                  <th>This version</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>History start</td>
                  <td>{retraining.previous_start}</td>
                  <td>{retraining.current_start}</td>
                </tr>
                <tr>
                  <td>History end (exclusive)</td>
                  <td>{retraining.previous_end}</td>
                  <td>{retraining.current_end}</td>
                </tr>
                <tr>
                  <td>Final test begins</td>
                  <td>{retraining.previous_holdout_start.slice(0, 10)}</td>
                  <td>{retraining.current_holdout_start.slice(0, 10)}</td>
                </tr>
                <tr>
                  <td>Validation return</td>
                  <td>{percent(retraining.previous_validation_return)}</td>
                  <td>{percent(validationReturn)}</td>
                </tr>
                <tr>
                  <td>Final test return</td>
                  <td>{percent(retraining.previous_test_return)}</td>
                  <td>{percent(testReturn)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="subtle">
            {retraining.data_changed
              ? "The input data snapshot differs."
              : "The input data snapshot matches."}{" "}
            {retraining.code_changed
              ? "The research code version also changed."
              : "The research code version matches."}{" "}
            Retraining can improve or worsen performance.
          </p>
          <button
            type="button"
            disabled={running}
            onClick={() => onViewParent(retraining.parent_id)}
          >
            View previous model
          </button>
        </div>
      )}
    </div>
  );
}
