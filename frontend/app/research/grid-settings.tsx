"use client";
import { defaults, parameterLabels } from "./config";

export type Range = { minimum: number; maximum: number; step: number };
export type Ranges = Record<string, Range>;
const limits: Record<string, [number, number, number]> = {
  horizon: [1, 5, 1],
  max_depth: [1, 6, 1],
  n_estimators: [20, 600, 1],
  learning_rate: [0.01, 0.3, 0.01],
  probability_threshold: [0.5, 0.9, 0.001],
  min_edge_bps: [0, 200, 0.1],
  folds: [2, 5, 1],
  seed: [0, 2147483647, 1],
  stop_loss_pct: [0, 50, 0.1],
  training_window_months: [0, 60, 1],
  recency_half_life_days: [0, 730, 1],
  subsample: [0.5, 1, 0.01],
  colsample_bytree: [0.5, 1, 0.01],
  min_child_weight: [1, 20, 0.1],
  reg_lambda: [0, 50, 0.1],
};
export function fixedGrid(form: typeof defaults): Ranges {
  return Object.fromEntries(
    Object.entries(limits).map(([key, bounds]) => {
      const setting = Number(form[key as keyof typeof form]);
      return [key, { minimum: setting, maximum: setting, step: bounds[2] }];
    }),
  );
}
export function gridCount(ranges: Ranges) {
  let count = 1;
  for (const range of Object.values(ranges)) {
    if (
      !Number.isFinite(range.step) ||
      range.step <= 0 ||
      range.maximum < range.minimum
    )
      return null;
    count *=
      Math.floor((range.maximum - range.minimum) / range.step + 1e-8) + 1;
  }
  return count;
}
export default function GridSettings({
  ranges,
  onChange,
  form,
}: {
  ranges: Ranges;
  onChange: (ranges: Ranges) => void;
  form: typeof defaults;
}) {
  const count = gridCount(ranges);
  return (
    <div className="grid-search-controls">
      <p>
        Every combination in these finite ranges will be attempted. Minimum =
        maximum keeps a setting fixed. Values advance by Step up to Maximum; an
        unaligned maximum is excluded.
      </p>
      <p id="grid-count" role="status">
        <strong>
          {count === null
            ? "Invalid range"
            : count.toLocaleString() + " combinations"}
        </strong>{" "}
        · limit 256
      </p>
      <p className="subtle">
        Probability 0.001 = 0.1 percentage point. Stop loss 0.1 = 0.1%. Horizon,
        depth, trees, folds and seed use whole numbers. Seeds measure
        sensitivity to randomness; a winning seed is weak evidence of a
        repeatable edge.
      </p>
      <button
        type="button"
        className="secondary"
        onClick={() => onChange(fixedGrid(form))}
      >
        Fix all ranges at current settings
      </button>
      <button
        type="button"
        className="secondary"
        onClick={() => {
          const next = fixedGrid(form);
          const center = form.probability_threshold;
          next.probability_threshold = {
            minimum: Number(Math.max(0.5, center - 0.001).toFixed(3)),
            maximum: Number(Math.min(0.9, center + 0.001).toFixed(3)),
            step: 0.001,
          };
          const stop = form.stop_loss_pct || 3;
          next.stop_loss_pct = {
            minimum: Number(Math.max(0, stop - 0.1).toFixed(2)),
            maximum: Number(Math.min(50, stop + 0.1).toFixed(2)),
            step: 0.1,
          };
          onChange(next);
        }}
      >
        Example: 0.1% entry and stop sweep
      </button>
      {Object.entries(ranges).map(([key, range]) => (
        <details key={key}>
          <summary>
            {parameterLabels[key]}: {range.minimum} → {range.maximum}, step{" "}
            {range.step}
          </summary>
          <div className="grid-range-fields">
            {(["minimum", "maximum", "step"] as const).map((field) => (
              <div key={field}>
                <label htmlFor={"grid-" + key + "-" + field}>
                  {field === "minimum"
                    ? "Minimum"
                    : field === "maximum"
                      ? "Maximum"
                      : "Step"}
                </label>
                <input
                  id={"grid-" + key + "-" + field}
                  type="number"
                  required
                  min={field === "step" ? 0.000001 : limits[key][0]}
                  max={field === "step" ? undefined : limits[key][1]}
                  step="any"
                  value={range[field]}
                  onChange={(event) =>
                    onChange({
                      ...ranges,
                      [key]: { ...range, [field]: Number(event.target.value) },
                    })
                  }
                />
              </div>
            ))}
          </div>
        </details>
      ))}
      {form.validation_scheme === "monthly" && (
        <p className="subtle">
          Monthly validation derives folds from calendar months. Keep the fold
          range fixed; select fixed blocks to compare different fold counts.
        </p>
      )}
      {form.entry_rule === "probability_only" && (
        <p className="subtle">
          Probability-only entry ignores the return hurdle, so keep that range
          fixed.
        </p>
      )}
      <p className="subtle">
        Feature set, entry rule, validation schedule, data dates, costs, sizing
        and final test fraction stay fixed. All candidates share the same
        portfolio evaluation dates even when horizon or folds change.
      </p>
    </div>
  );
}
