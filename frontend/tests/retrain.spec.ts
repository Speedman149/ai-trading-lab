import { test, expect } from "@playwright/test";

test("saved-date rerun preserves settings, dates, results and previous version", async ({
  page,
}) => {
  test.setTimeout(90000);
  await page.goto("/research");
  await page.locator("#asset").selectOption("BTC/USD");
  await page.locator("#start").fill("2024-01-01");
  await page.locator("#end").fill("2025-01-01");
  await page.locator("#folds").fill("2");
  await page.locator("#n_estimators").fill("20");
  const firstResponse = page.waitForResponse(
    (response) =>
      /\/api\/research\/versions\/[0-9a-f]{24}$/.test(response.url()),
    { timeout: 60000 },
  );
  await page
    .getByRole("button", { name: "Train & Evaluate", exact: true })
    .click();
  const parent = await (await firstResponse).json();
  await expect(
    page.getByRole("button", { name: "Re-run saved dates", exact: true }),
  ).toBeEnabled();
  // Unsaved form edits must never overwrite the saved model's retrain config.
  await page.locator("#max_depth").fill("6");
  await page.locator("#end").fill("2026-01-01");
  const submitted = page.waitForRequest(
    (request) =>
      request.url().endsWith("/" + parent.id + "/retrain") &&
      request.method() === "POST",
  );
  const candidateResponse = page.waitForResponse(
    (response) =>
      /\/api\/research\/versions\/[0-9a-f]{24}$/.test(response.url()),
    { timeout: 60000 },
  );
  await page
    .getByRole("button", { name: "Re-run saved dates", exact: true })
    .click();
  expect((await submitted).postDataJSON()).toEqual({ mode: "same_dates" });
  const candidate = await (await candidateResponse).json();
  expect(candidate.id).not.toBe(parent.id);
  expect(candidate.config.parent_id).toBe(parent.id);
  expect(candidate.config.max_depth).toBe(parent.config.max_depth);
  expect(candidate.config.backtest).toEqual(parent.config.backtest);
  expect(candidate.validation).toEqual(parent.validation);
  expect(candidate.holdout).toEqual(parent.holdout);
  await expect(
    page.getByRole("table", { name: "Retraining comparison", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".retraining-comparison")).toContainText(
    "Model parameters were preserved.",
  );
  await expect(page.locator(".retraining-comparison")).toContainText(
    "The evaluation dates match.",
  );
  await expect(page.locator("#end")).toHaveValue("2025-01-01");
  await expect(page.locator("#max_depth")).toHaveValue(
    String(parent.config.max_depth),
  );
  await page
    .getByRole("button", { name: "View previous model", exact: true })
    .click();
  await expect(page.locator(".model-settings")).toContainText(
    "2024-01-01 to 2025-01-01",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page
    .locator(".model-settings")
    .screenshot({ path: "../artifacts/saved-model-settings-mobile.png" });
});

test("form waits for hydration and submits the dates the user entered", async ({
  page,
}) => {
  let releaseScripts!: () => void;
  const scriptsReady = new Promise<void>((resolve) => {
    releaseScripts = resolve;
  });
  await page.route(/\/_next\/static\/.*\.js(?:\?.*)?$/, async (route) => {
    await scriptsReady;
    await route.continue();
  });
  try {
    await page.goto("/research", { waitUntil: "commit" });
    await expect(page.locator("#start")).toBeDisabled();
    const fill = page.locator("#start").fill("2024-01-02");
    releaseScripts();
    await fill;
    await page.locator("#end").fill("2024-06-30");
    const submitted = page.waitForRequest(
      (request) =>
        request.url().endsWith("/api/research/jobs") &&
        request.method() === "POST",
    );
    await page
      .getByRole("button", { name: "Train & Evaluate", exact: true })
      .click();
    const config = (await submitted).postDataJSON();
    expect(config.backtest.start).toBe("2024-01-02");
    expect(config.backtest.end).toBe("2024-06-30");
    await expect(page.locator(".error[role=alert]")).toContainText(
      "250 labeled daily bars",
      { timeout: 20000 },
    );
  } finally {
    releaseScripts();
  }
});
