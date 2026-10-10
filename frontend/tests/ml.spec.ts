import { test, expect } from "@playwright/test";

test("XGBoost training shows backend evaluation, calibration and saved versions", async ({
  page,
}) => {
  test.setTimeout(90000);
  await page.goto("/research");
  await expect(
    page.getByRole("heading", { name: "Train. Test. Compare." }),
  ).toBeVisible();
  await page.locator("#asset").selectOption("BTC/USD");
  await page.locator("#start").fill("2024-01-01");
  await page.locator("#end").fill("2025-01-01");
  await page.locator("#folds").fill("2");
  await page.locator("#n_estimators").fill("30");
  const submitted = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/research/jobs") && r.request().method() === "POST",
  );
  const reportResponse = page.waitForResponse(
    (r) =>
      /\/api\/research\/versions\/[0-9a-f]{24}$/.test(r.url()) &&
      r.request().method() === "GET",
    { timeout: 70000 },
  );
  await page.getByRole("button", { name: "Train & Evaluate" }).click();
  expect((await submitted).status()).toBe(202);
  const response = await reportResponse;
  expect(response.ok()).toBeTruthy();
  const report = await response.json();
  expect(report.config.algorithm).toBe("xgboost");
  expect(report.config.backtest.asset).toBe("BTC/USD");
  expect(report.provenance.synthetic).toBeTruthy();
  const percent = (v: number) => (v * 100).toFixed(2) + "%";
  await expect(page.locator(".stat").first()).toContainText(
    percent(report.holdout.strategy.metrics.net_return),
  );
  await expect(page.locator(".stat").nth(3)).toContainText(
    percent(report.holdout.prediction.mae),
  );
  await expect(
    page.getByRole("heading", { name: "Probability calibration" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Stability across market regimes" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Retraining & model acceptance" }),
  ).toBeVisible();
  await expect(page.locator(".notice").first()).toContainText(
    "Synthetic demonstration",
  );
  const firstPath = await page
    .locator(".recharts-line-curve")
    .first()
    .getAttribute("d");
  expect(firstPath!.length).toBeGreaterThan(100);
  await page
    .getByRole("button", { name: "Walk-forward validation", exact: true })
    .click();
  await expect(page.locator(".stat").first()).toContainText(
    percent(report.validation.strategy.metrics.net_return),
  );
  await page
    .getByRole("button", { name: "Final held-out test", exact: true })
    .click();
  await page.screenshot({
    path: "../artifacts/ml-desktop.png",
    fullPage: true,
  });
  await page.reload();
  const saved = page
    .locator(".version")
    .filter({ hasText: "BTC/USD · XGBoost" })
    .first();
  await saved.getByRole("button", { name: "View results" }).click();
  await expect(page.locator(".stat").first()).toContainText(
    percent(report.holdout.strategy.metrics.net_return),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({ path: "../artifacts/ml-mobile.png", fullPage: true });
});

test("RL trains as a job and reports strategy metrics without fabricated prediction accuracy", async ({
  page,
}) => {
  test.setTimeout(90000);
  await page.goto("/research");
  await page.locator("#algorithm").selectOption("q_learning");
  await page.locator("#asset").selectOption("BTC/USD");
  await page.locator("#start").fill("2024-01-01");
  await page.locator("#end").fill("2025-01-01");
  await page.locator("#episodes").fill("2");
  await page.locator("#folds").fill("2");
  await page.locator("#seed").fill("45");
  const responsePromise = page.waitForResponse(
    (r) => /\/api\/research\/versions\/[0-9a-f]{24}$/.test(r.url()),
    { timeout: 70000 },
  );
  await page.getByRole("button", { name: "Train & Evaluate" }).click();
  const response = await responsePromise;
  const report = await response.json();
  expect(report.config.algorithm).toBe("q_learning");
  expect(report.holdout.prediction.mae).toBeNull();
  await expect(page.locator(".stat").nth(3)).toContainText("N/A");
  await expect(
    page.getByText("Q-learning chooses portfolio actions.", { exact: false }),
  ).toBeVisible();
  if (!report.promotion.eligible)
    await expect(
      page.getByRole("button", { name: "Promote candidate" }),
    ).toBeDisabled();
  await page.screenshot({
    path: "../artifacts/rl-desktop.png",
    fullPage: true,
  });
});

test("short training history gives an actionable error", async ({ page }) => {
  await page.goto("/research");
  await page.locator("#start").fill("2024-01-02");
  await page.locator("#end").fill("2024-06-30");
  await page.getByRole("button", { name: "Train & Evaluate" }).click();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "250 labeled daily bars",
    { timeout: 20000 },
  );
  await expect(
    page.getByRole("button", { name: "Train & Evaluate" }),
  ).toBeEnabled();
});
