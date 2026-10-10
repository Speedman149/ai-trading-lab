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

  // Exercise eligibility, a real pending request, failure recovery and success
  // without changing the local model registry.
  let active = false;
  let eligible = false;
  let attempts = 0;
  let finishPromotion: (() => void) | undefined;
  const savedVersion = {
    id: report.id,
    created_at: report.created_at,
    asset: report.config.backtest.asset,
    algorithm: report.config.algorithm,
    provider: report.config.backtest.provider,
    net_return: report.holdout.strategy.metrics.net_return,
    eligible: true,
    rollback_available: false,
  };
  await page.route("**/api/research/jobs", (route) =>
    route.fulfill({ json: [] }),
  );
  await page.route("**/api/research/versions", (route) =>
    route.fulfill({ json: [{ ...savedVersion, active, eligible }] }),
  );
  await page.route("**/api/research/versions/" + report.id, (route) =>
    route.fulfill({
      json: {
        ...report,
        promotion: {
          ...report.promotion,
          eligible,
          criteria: report.promotion.criteria.map(
            (criterion: { name: string }, index: number) => ({
              ...criterion,
              passed: eligible || index !== 0,
            }),
          ),
        },
      },
    }),
  );
  await page.route(
    "**/api/research/versions/" + report.id + "/promote",
    async (route) => {
      attempts += 1;
      await new Promise<void>((resolve) => {
        finishPromotion = resolve;
      });
      finishPromotion = undefined;
      if (attempts === 1) {
        await route.fulfill({
          status: 409,
          json: {
            detail: "Active model changed since this candidate was trained.",
          },
        });
      } else {
        active = true;
        await route.fulfill({ json: { active: report.id } });
      }
    },
  );
  await page.reload();
  await page.getByRole("button", { name: "View results" }).click();
  const promote = page.getByRole("button", {
    name: "Promote candidate",
    exact: true,
  });
  await expect(promote).toBeDisabled();
  await expect(promote).toHaveCSS("cursor", "not-allowed");
  await expect(promote).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#promotion-status")).toContainText(
    "1 validation check failed",
  );
  expect(attempts).toBe(0);
  await expect(page.locator("#horizon")).toHaveAccessibleDescription(
    /from the next market open to the following open/,
  );
  await expect(page.locator("#evaluation-period-help")).toContainText(
    "one frozen model",
  );
  eligible = true;
  await page.reload();
  await page.getByRole("button", { name: "View results" }).click();
  await expect(promote).toBeEnabled();
  await promote.click();
  const promoting = page.getByRole("button", { name: "Promoting…" });
  await expect(promoting).toBeDisabled();
  await expect(promoting).toHaveAttribute("aria-busy", "true");
  await expect(promoting).toHaveCSS("cursor", "wait");
  await expect(
    page.getByRole("heading", { name: "Training and testing", exact: true }),
  ).toHaveCount(0);
  await expect.poll(() => Boolean(finishPromotion)).toBeTruthy();
  finishPromotion!();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "Active model changed",
  );
  await expect(promote).toBeEnabled();
  await expect(promote).toHaveCSS("cursor", "pointer");
  await promote.click();
  await expect(promoting).toBeVisible();
  await expect.poll(() => Boolean(finishPromotion)).toBeTruthy();
  finishPromotion!();
  const alreadyActive = page.getByRole("button", { name: "Already active" });
  await expect(alreadyActive).toBeDisabled();
  await expect(alreadyActive).toHaveCSS("cursor", "not-allowed");
  await expect(page.locator("#promotion-status")).toContainText(
    "already active; no promotion is needed",
  );
  await expect(
    page.getByText("Candidate is now the active research model."),
  ).toBeVisible();
  expect(attempts).toBe(2);
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
