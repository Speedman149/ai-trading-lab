import { test, expect } from "@playwright/test";

test("monthly recent training, exact decimal grid, stop settings and saved restore", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  await page.goto("/research");
  await page.locator("#asset").selectOption("BTC/USD");
  await page.locator("#start").fill("2024-01-01");
  await page.locator("#end").fill("2025-07-01");
  await page.locator("#n_estimators").fill("20");
  await page
    .getByRole("button", { name: "Use last 3 months → next month" })
    .click();
  await expect(page.locator("#training_window_months")).toHaveValue("3");
  await expect(page.locator("#validation_scheme")).toHaveValue("monthly");
  await page.locator("#search-mode").selectOption("grid");
  await page
    .getByRole("button", { name: "Fix all ranges at current settings" })
    .click();
  await expect(page.locator("#grid-count")).toContainText("1 combinations");
  await page
    .locator(".grid-search-controls details")
    .filter({ hasText: "Stop loss (%)" })
    .locator("summary")
    .click();
  await page.locator("#grid-stop_loss_pct-minimum").fill("2.9");
  await page.locator("#grid-stop_loss_pct-maximum").fill("3.1");
  await page.locator("#grid-stop_loss_pct-step").fill("0.1");
  await page
    .locator(".grid-search-controls details")
    .filter({ hasText: "Forecast horizon (bars):" })
    .locator("summary")
    .click();
  await page.locator("#grid-horizon-maximum").fill("2");
  await expect(page.locator("#grid-count")).toContainText("6 combinations");
  const completed = page.waitForResponse(
    (response) =>
      /\/api\/research\/searches\/[0-9a-f]{24}$/.test(response.url()),
    { timeout: 150000 },
  );
  await page
    .getByRole("button", { name: "Find best parameters", exact: true })
    .click();
  const report = await (await completed).json();
  expect(report.status).toBe("completed");
  expect(report.total_combinations).toBe(6);
  expect(report.space.stop_loss_pct).toEqual([2.9, 3, 3.1]);
  expect(report.trials).toHaveLength(6);
  expect(
    report.trials.every(
      (trial: { status: string }) => trial.status === "completed",
    ),
  ).toBeTruthy();
  const model = await (
    await request.get(
      "http://127.0.0.1:8000/api/research/versions/" + report.winner_id,
    )
  ).json();
  expect(model.config.training_window_months).toBe(3);
  expect(model.config.recency_half_life_days).toBe(90);
  expect(model.walk_forward.length).toBeGreaterThan(3);
  expect(model.training.regression_fit_rows).toBe(model.training.rows);
  expect(model.training.recency.newest_weight).toBeGreaterThan(
    model.training.recency.oldest_weight,
  );
  await expect(page.locator("#recent-fit-help")).toContainText(
    "effective weighted sample size",
  );
  await expect(page.locator("#stop_loss_pct")).toHaveValue(
    String(model.config.backtest.stop_loss_pct),
  );
  await page
    .getByRole("button", { name: "Use best-found settings", exact: true })
    .click();
  await expect(page.locator("#search-mode")).toHaveValue("grid");
  await expect(page.locator("#grid-stop_loss_pct-step")).toHaveValue("0.1");
  await expect(page.locator("#grid-count")).toContainText("6 combinations");
  expect(model.config.evaluation_anchor_horizon).toBe(2);
  const resubmitted = page.waitForRequest(
    (request) =>
      request.url().endsWith("/api/research/jobs") &&
      request.method() === "POST",
  );
  const rerunResponse = page.waitForResponse(
    (response) =>
      new RegExp("/api/research/versions/[0-9a-f]{24}$").test(response.url()),
    { timeout: 100000 },
  );
  await page
    .getByRole("button", { name: "Train & Evaluate", exact: true })
    .click();
  expect((await resubmitted).postDataJSON().evaluation_anchor_horizon).toBe(2);
  const rerun = await (await rerunResponse).json();
  expect(
    rerun.validation.strategy.equity.map(
      (point: { timestamp: string }) => point.timestamp,
    ),
  ).toEqual(
    model.validation.strategy.equity.map(
      (point: { timestamp: string }) => point.timestamp,
    ),
  );
  expect(
    rerun.holdout.strategy.equity.map(
      (point: { timestamp: string }) => point.timestamp,
    ),
  ).toEqual(
    model.holdout.strategy.equity.map(
      (point: { timestamp: string }) => point.timestamp,
    ),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBeFalsy();
});

test("oversized grids show their complete count and do not submit jobs", async ({
  page,
}) => {
  await page.goto("/research");
  await page.locator("#search-mode").selectOption("grid");
  await page
    .locator(".grid-search-controls details")
    .filter({ hasText: "Entry probability:" })
    .locator("summary")
    .click();
  await page.locator("#grid-probability_threshold-minimum").fill("0.5");
  await page.locator("#grid-probability_threshold-maximum").fill("0.9");
  await page.locator("#grid-probability_threshold-step").fill("0.001");
  await expect(page.locator("#grid-count")).toContainText("401 combinations");
  let submitted = false;
  page.on("request", (request) => {
    if (request.method() === "POST") submitted = true;
  });
  await page
    .getByRole("button", { name: "Find best parameters", exact: true })
    .click();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "1–256 combinations",
  );
  expect(submitted).toBeFalsy();
});
