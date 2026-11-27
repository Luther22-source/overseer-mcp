# Overseer Repository Instructions (`AGENTS.md`)

This repository is the home of the **OpenRouter Worker Pool MCP Server ("IDE-as-Boss Overseer")** (`overseer-mcp`).

## 1. Pick Up Context First

Whenever you start working in this repository:

1. Read [README.md](README.md) for the installation flow, folder map, and development commands.
2. Read [docs/designs/openrouter-worker-pool-mcp.md](docs/designs/openrouter-worker-pool-mcp.md) for the architecture specification, tool schemas, limits, and module contracts.
3. Read [.agents/skills/overseer-boss/SKILL.md](.agents/skills/overseer-boss/SKILL.md) if you are orchestrating or tuning worker runs.

## 2. Hygiene & Documentation Rules

* **Keep the architecture spec synced:** Update [docs/designs/openrouter-worker-pool-mcp.md](docs/designs/openrouter-worker-pool-mcp.md) whenever tool schemas, environment variables, limits, or module contracts change.
* **Keep the root clean:** Do not leave temporary notes, audit logs, or generated benchmark outputs in the repository root.
* **Never read or print raw API keys:** Never output the contents of `<data_dir>/.env` (you may inspect variable names only).

## 3. Code & Verification Rules

* **Always verify after changes:** Run the full offline verification suite from the repository root before reporting work complete (`pyproject.toml` sets `testpaths = ["tests"]` to exclude `overseer-bench/`):
  * `python -m compileall -q overseer tests`
  * `ruff check --isolated --select E9,F63,F7,F82,F,I overseer`
  * `pytest -v`
  * If `overseer/ui/` changed: `npm --prefix overseer/ui run typecheck`, `npm --prefix overseer/ui test`, and rebuild `overseer/ui/dist/app.html` (`npm --prefix overseer/ui run build`).
* **Style & honesty:** Use plain, concise English. Report only verified file paths, symbols, and command results directly observed after your final edit.
