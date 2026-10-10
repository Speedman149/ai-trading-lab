import { test, expect } from "@playwright/test";
for (const asset of ["AAPL", "BTC/USD"]) {
  test(`backtest ${asset} from browser matches Python results`, async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Test ideas. Measure outcomes." }),
    ).toBeVisible();
    await page.locator("#asset").selectOption(asset);
    await page.locator("#initial_cash").fill("12000");
    const responsePromise = page.waitForResponse(
      (r) =>
        r.url().endsWith("/api/backtests") && r.request().method() === "POST",
    );
    await page.getByRole("button", { name: "Run Backtest" }).click();
    const response = await responsePromise;
    expect(response.ok()).toBeTruthy();
    const result = await response.json();
    expect(result.config.asset).toBe(asset);
    expect(result.config.initial_cash).toBe(12000);
    const net = (result.metrics.net_return * 100).toFixed(2) + "%";
    await expect(page.locator(".stat").first()).toContainText(net);
    await expect(
      page.getByRole("heading", { name: "Portfolio equity" }),
    ).toBeVisible();
    await expect(page.locator("tbody tr")).toHaveCount(result.trades.length);
    await expect(page.locator(".notice").first()).toContainText(
      "Synthetic demonstration",
    );
    await expect(page.locator(".recharts-line-curve")).toHaveCount(2);
    const path = await page
      .locator(".recharts-line-curve")
      .first()
      .getAttribute("d");
    expect(path!.length).toBeGreaterThan(100);
    await page.screenshot({
      path: `../artifacts/browser-${asset.replace("/", "-")}.png`,
      fullPage: true,
    });
  });
}
test("validation error appears", async ({ page }) => {
  await page.goto("/");
  await page.locator("#fast").fill("40");
  await page.locator("#slow").fill("20");
  await page.getByRole("button", { name: "Run Backtest" }).click();
  await expect(page.locator(".error[role=alert]")).toContainText(
    "Fast window must be smaller",
  );
});
test("backend unavailable is actionable", async ({ page }) => {
  await page.route("**/api/backtests", (route) => route.abort());
  await page.goto("/");
  await page.getByRole("button", { name: "Run Backtest" }).click();
  await expect(page.locator(".error[role=alert]")).toBeVisible();
});
test("mobile layout fits viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Run Backtest" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});
