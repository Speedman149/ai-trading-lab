# Repository development instructions

Build the foundation before advanced ML, news, RL, deployment, or paper trading. Follow docs/methodology.md for execution semantics. Never enable real-money order placement.

Run commands inside WSL Ubuntu. Python virtual environment is .venv at the repository root; backend imports need backend as the working directory or PYTHONPATH. Use Node 22 (source ~/.nvm/nvm.sh && nvm use 22) for Next.js.

Keep all portfolio/trading calculations in Python. Strategies receive history prefixes only. Signals use completed bars, fill at next opening price, and never read future volume. Every financial change needs hand-calculated regression coverage. Synthetic results must remain visibly labeled.

Real equity simulation requires validated supported-action metadata; keep the engine verification gate. Splits and ordinary dividends are supported and tested; unsupported actions must fail closed. Do not substitute adjusted prices for execution prices.

Backend checks: .venv/bin/pytest from backend, Ruff lint/format. Frontend checks: npm run lint, npm run typecheck, npm run build. Run browser tests after meaningful interface or integration changes (npm run test:e2e, after build). Avoid simultaneous builds and edits to the same frontend files.

Update README and methodology after substantial changes. Bump generator/service version constants when data or simulation semantics change. Keep .env, credentials, caches, model files, node_modules, virtual environments, screenshots, and test output out of Git. Do not change files in neighboring projects.