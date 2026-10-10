import { defineConfig, globalIgnores } from "eslint/config";
import js from "@eslint/js";
import ts from "typescript-eslint";
import globals from "globals";
export default defineConfig([
  globalIgnores([
    ".next/**",
    "test-results/**",
    "playwright-report/**",
    "next-env.d.ts",
  ]),
  js.configs.recommended,
  ...ts.configs.recommended,
  { languageOptions: { globals: { ...globals.browser, ...globals.node } } },
]);
