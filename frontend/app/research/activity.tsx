export type ActivityData = {
  entry_rule: string;
  time_in_market: number;
  average_exposure: number;
  fills: number;
  closed_trades: number;
  rejected_orders: number;
  buy_hold_return: number;
  excess_vs_buy_hold: number;
  entry_hurdle: number | null;
  signals: number | null;
  return_passes: number | null;
  probability_passes: number | null;
  qualifying_signals: number | null;
  predicted_return_min: number | null;
  predicted_return_max: number | null;
  probability_min: number | null;
  probability_max: number | null;
  notes: string[];
};
const percent = (value: number | null) =>
  value == null ? "N/A" : (value * 100).toFixed(2) + "%";
export default function Activity({ data }: { data: ActivityData }) {
  return (
    <div className="panel charts activity-diagnostics">
      <h2>What drove these results?</h2>
      <dl className="parameter-grid">
        <div>
          <dt>Buy-and-hold net return</dt>
          <dd>{percent(data.buy_hold_return)}</dd>
        </div>
        <div>
          <dt>AI minus buy-and-hold</dt>
          <dd className={data.excess_vs_buy_hold < 0 ? "negative" : "positive"}>
            {(data.excess_vs_buy_hold * 100).toFixed(2)} percentage points
          </dd>
        </div>
        <div>
          <dt>Time invested (daily closes)</dt>
          <dd>{percent(data.time_in_market)}</dd>
        </div>
        <div>
          <dt>Average portfolio exposure</dt>
          <dd>{percent(data.average_exposure)}</dd>
        </div>
        <div>
          <dt>Completed trades / fills</dt>
          <dd>
            {data.closed_trades} / {data.fills}
          </dd>
        </div>
        <div>
          <dt>Rejected / expired orders</dt>
          <dd>{data.rejected_orders}</dd>
        </div>
      </dl>
      {data.signals != null && (
        <>
          <div className="table-wrap">
            <table aria-label="Entry signal diagnostics">
              <thead>
                <tr>
                  <th>Entry check</th>
                  <th>Predictions passing</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    Return above modeled hurdle ({percent(data.entry_hurdle)})
                    {data.entry_rule === "probability_only" &&
                      " — not required in probability-only mode"}
                  </td>
                  <td>
                    {data.return_passes} / {data.signals}
                  </td>
                </tr>
                <tr>
                  <td>Probability above entry threshold</td>
                  <td>
                    {data.probability_passes} / {data.signals}
                  </td>
                </tr>
                <tr>
                  <td>Meets this strategy's entry conditions</td>
                  <td>
                    {data.qualifying_signals} / {data.signals}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            Return forecast range: {percent(data.predicted_return_min)} to{" "}
            {percent(data.predicted_return_max)}
            <br />
            Up-probability range: {percent(data.probability_min)} to{" "}
            {percent(data.probability_max)}
          </p>
        </>
      )}
      <ul className="criteria">
        {data.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  );
}
