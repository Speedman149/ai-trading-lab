import { test, expect } from "@playwright/test";

test("parameter search ranks validation, tests one winner, applies settings and reloads history", async ({
  page,
  request,
}) => {
  test.setTimeout(120000);
  await page.goto("/research");
  await page.locator("#asset").selectOption("BTC/USD");
  await page.locator("#start").fill("2024-01-01");
  await page.locator("#end").fill("2025-01-01");
  await page.locator("#folds").fill("2");
  await page.locator("#n_estimators").fill("30");
  await page.locator("#search-trials").fill("6");
  await page.locator("#search-objective").selectOption("net_return");
  const submitted = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/research/searches") &&
      response.request().method() === "POST",
  );
  const completed = page.waitForResponse(
    (response) =>
      /\/api\/research\/searches\/[0-9a-f]{24}$/.test(response.url()),
    { timeout: 100000 },
  );
  await page
    .getByRole("button", { name: "Find best parameters", exact: true })
    .click();
  const submission = await submitted;
  expect(submission.status()).toBe(202);
  const job = await submission.json();
  expect(job.kind).toBe("search");
  await expect(
    page.getByRole("heading", { name: "Searching parameters", exact: true }),
  ).toBeVisible();
  const response = await completed;
  const report = await response.json();
  expect(report.status).toBe("completed");
  expect(report.config.objective).toBe("net_return");
  expect(report.config.mode).toBe("coarse_to_fine");
  expect(
    report.trials.filter(
      (trial: { phase: string }) => trial.phase === "refine",
    ),
  ).toHaveLength(3);
  expect(report.trials).toHaveLength(6);
  const selectedParameters = report.trials.find(
    (trial: { trial: number }) => trial.trial === report.selected_trial,
  ).parameters;
  for (const [key, value] of Object.entries(selectedParameters)) {
    await expect(page.locator("#" + key)).toHaveValue(String(value));
  }
  await expect(
    page.getByRole("heading", {
      name: "Parameters used for this model",
      exact: true,
    }),
  ).toBeVisible();
  expect(report.trials[0].trial).toBe(report.selected_trial);
  await expect(
    page.getByRole("heading", { name: "Best candidate found", exact: true }),
  ).toBeVisible();
  const table = page.getByRole("table", {
    name: "Parameter search leaderboard",
  });
  await expect(table.locator("tbody tr")).toHaveCount(6);
  await expect(table.locator("tbody tr").first()).toHaveAttribute(
    "aria-selected",
    "true",
  );
  const percent = (value: number) => (value * 100).toFixed(2) + "%";
  await expect(table.locator("tbody tr").first()).toContainText(
    percent(report.trials[0].metrics.net_return),
  );
  await expect(
    table.locator("tbody tr").first().locator("td").nth(5),
  ).toHaveText(report.trials[0].metrics.transaction_costs.toFixed(2));
  await expect(page.locator(".search-results")).toContainText(
    "SYNTHETIC demonstration",
  );
  const winner = await (
    await request.get(
      "http://127.0.0.1:8000/api/research/versions/" + report.winner_id,
    )
  ).json();
  expect(winner.search.id).toBe(report.id);
  expect(winner.search.selected_trial).toBe(report.selected_trial);
  await expect(
    page.locator(".stat").filter({ hasText: "Net return" }).last(),
  ).toContainText(percent(winner.validation.strategy.metrics.net_return));
  const versions = await (
    await request.get("http://127.0.0.1:8000/api/research/versions")
  ).json();
  expect(
    versions.find((version: { id: string }) => version.id === report.winner_id)
      .active,
  ).toBeFalsy();
  await page.locator("#feature_set").selectOption("basic");
  await page.locator("#entry_rule").selectOption("return_and_probability");
  await page.locator("#asset").selectOption("AAPL");
  await page.locator("#max_depth").fill("6");
  await page.locator("#initial_cash").fill("54321");
  await page.getByRole("button", { name: "Use best-found settings" }).click();
  await expect(page.locator("#asset")).toHaveValue("BTC/USD");
  await expect(page.locator("#initial_cash")).toHaveValue("10000");
  for (const [key, value] of Object.entries(report.trials[0].parameters)) {
    await expect(page.locator("#" + key)).toHaveValue(String(value));
  }
  await page.getByRole("button", { name: "Refine winning settings" }).click();
  await expect(page.locator("#search-mode")).toHaveValue("local");
  await expect(page.locator("#search-trials")).toHaveValue("32");
  await expect(page.locator("#initial_cash")).toHaveValue("10000");
  await expect(page.locator("#return-window-help")).toContainText(
    "whole window",
  );
  await expect(page.locator("#training-sample-help")).toContainText(
    "separate rows",
  );
  await expect(page.locator(".search-results")).toContainText(
    "Profitable folds:",
  );
  await page
    .locator(".search-results")
    .screenshot({ path: "../artifacts/parameter-search-desktop.png" });
  await page.reload();
  await page
    .locator(".saved-searches .version")
    .first()
    .getByRole("button", { name: "View search" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Best candidate found", exact: true }),
  ).toBeVisible();
  await expect(table.locator("tbody tr")).toHaveCount(6);
  await page
    .getByRole("button", { name: "Final held-out test", exact: true })
    .click();
  await expect(
    page.locator(".stat").filter({ hasText: "Net return" }).last(),
  ).toContainText(percent(winner.holdout.strategy.metrics.net_return));
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page
    .locator(".search-results")
    .screenshot({ path: "../artifacts/parameter-search-mobile.png" });
});

test("failed parameter search reports an error and releases search controls", async ({
  page,
}) => {
  await page.goto("/research");
  await page.locator("#start").fill("2024-01-02");
  await page.locator("#end").fill("2024-06-30");
  await page.locator("#search-trials").fill("2");
  await page
    .getByRole("button", { name: "Find best parameters", exact: true })
    .click();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "250 labeled daily bars",
    { timeout: 20000 },
  );
  await expect(
    page.getByRole("button", { name: "Find best parameters", exact: true }),
  ).toBeEnabled();
});

for (const mode of ["local", "random"]) {
  test(
    "search method " +
      mode +
      " is submitted with the selected budget and costs",
    async ({ page }) => {
      await page.route("**/api/research/searches", async (route) => {
        if (route.request().method() !== "POST") return route.continue();
        const submitted = route.request().postDataJSON();
        expect(submitted.mode).toBe(mode);
        expect(submitted.trials).toBe(8);
        expect(submitted.research.backtest.commission_bps).toBe(10);
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Search temporarily unavailable" }),
        });
      });
      await page.goto("/research");
      await page.locator("#search-mode").selectOption(mode);
      await page.locator("#search-trials").fill("8");
      await page
        .getByRole("button", { name: "Find best parameters", exact: true })
        .click();
      await expect(page.locator(".error[role=alert]")).toContainText(
        "Search temporarily unavailable",
      );
      await expect(page.locator("#search-mode")).toBeEnabled();
    },
  );
}

test("fractional restored model parameters can start another search", async ({
  page,
}) => {
  const parameters = {
    min_edge_bps: 7.5,
    probability_threshold: 0.535,
    learning_rate: 0.075,
    subsample: 0.85,
    colsample_bytree: 0.95,
    min_child_weight: 2.5,
    reg_lambda: 4.5,
  };
  await page.route("**/api/research/searches", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    expect(route.request().postDataJSON().research).toMatchObject(parameters);
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Fractional settings reached backend" }),
    });
  });
  await page.goto("/research");
  await page
    .getByText("XGBoost sampling & regularization", { exact: true })
    .click();
  for (const [key, value] of Object.entries(parameters))
    await page.locator("#" + key).fill(String(value));
  await page
    .getByRole("button", { name: "Find best parameters", exact: true })
    .click();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "Fractional settings reached backend",
  );
});
