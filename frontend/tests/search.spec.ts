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
  await page.locator("#search-trials").fill("3");
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
  expect(report.trials).toHaveLength(3);
  expect(report.trials[0].trial).toBe(report.selected_trial);
  await expect(
    page.getByRole("heading", { name: "Best candidate found", exact: true }),
  ).toBeVisible();
  const table = page.getByRole("table", {
    name: "Parameter search leaderboard",
  });
  await expect(table.locator("tbody tr")).toHaveCount(3);
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
  await page.locator("#asset").selectOption("AAPL");
  await page.locator("#max_depth").fill("6");
  await page.locator("#initial_cash").fill("54321");
  await page.getByRole("button", { name: "Use best-found settings" }).click();
  await expect(page.locator("#asset")).toHaveValue("BTC/USD");
  await expect(page.locator("#initial_cash")).toHaveValue("10000");
  for (const [key, value] of Object.entries(report.trials[0].parameters)) {
    await expect(page.locator("#" + key)).toHaveValue(String(value));
  }
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
  await expect(table.locator("tbody tr")).toHaveCount(3);
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
