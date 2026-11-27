---
name: overseer-boss
description: >-
  Orchestration playbook for delegating bounded file tasks to Overseer MCP's OpenRouter worker pool
  (`start_run`, `get_report`, `cancel_run`, `rollback`, `get_config`, `update_config`, `list_models`,
  `open_overseer_monitor`). Use whenever calling Overseer or planning how to decompose coding,
  refactoring, documentation, or test-writing tasks for free/cheap/heavy OpenRouter worker models
  so they finish cleanly in minimal steps with high accuracy.
---

# Overseer Boss — Teaching & Driving OpenRouter Workers

Overseer workers (`free`, `cheap`, and `heavy` tiers) are **folder-jailed, no-shell** agents with 8 tools (`view_file`, `file_outline`, `list_dir`, `search_dir`, `find_file`, `replace_block`, `write_new_file`, `finish`). Their success depends directly on the clarity and constraints of your `start_run` brief.

## Pre-Flight & Setup

1. **First-time setup / diagnostics:** Users can initialize keys (`<data_dir>/.env`) and check their installation with `overseer-mcp init` and `overseer-mcp doctor`.
2. **`project_root` restrictions:** Must be an existing absolute directory outside `~/.gemini/` and outside the Overseer data directory (`%LOCALAPPDATA%\overseer` on Windows or `~/.local/share/overseer` on macOS/Linux).
3. **Pre-existing non-overlapping folders:** Every task's `folder` MUST already exist on disk before `start_run`. In multi-worker runs (`len(tasks) > 1`), folders cannot be `.` and cannot overlap or nest.
4. **Never poll `get_report` in a loop:** After `start_run`, do other work or return control/wait for the user before calling `get_report`. When the user asks for hands-free auto-collection, you may set a single one-shot `schedule(DurationSeconds=...)` wakeup (never a recurring polling loop). Overseer automatically fires an OS completion notification when a run finishes (unless `OVERSEER_DISABLE_TOASTS=1`) and exposes `open_overseer_monitor` for live side-pane monitoring.

## Model Discovery, Config & Custom Instructions

* **`list_models(free_only, tools_only, search, limit)`:** Search OpenRouter's `/models` catalog before switching models so you pick valid tool-calling slugs (and `:free` slugs for the `free` tier).
* **`get_config` / `update_config`:** Inspect or update `free`, `cheap`, and `heavy` primary/fallback model chains, per-tier USD spend caps (`cheap`, `heavy`), `max_concurrent_workers` (`1..10`), live `key_quotas` (`GET /api/v1/key` credits/rate limits), and OpenRouter `provider` routing rules (`sort`, `order`, `only`, `allow_fallbacks`, `zdr`, `quantizations`).
* **Custom Instructions:** Global instructions live in `<data_dir>/custom_instructions.txt` (editable in the Monitor UI) and repo-local instructions can be placed in `<project_root>/.overseer/instructions.md` (capped at 10 KB). Both are appended to worker prompts automatically.

## Writing Briefs That Workers Execute Cleanly

### 1. Seed `target_files` Explicitly
Overseer injects line counts / `(new file - does not exist yet)` for every path in `target_files` straight into the worker's system prompt.
- Include **both** compact reference files the worker should read (even outside its writable folder, such as shared types or `REFERENCES.md`) **and** files inside its writable folder that it will edit or create.
- **Do NOT seed giant design specs (>300 lines / >20 KB)** when the exact signatures and rules are already in `shared_contract`—loading huge docs trips `max_history_chars` pruning and causes free models to loop re-reading pruned chunks.
- Workers can also call `file_outline` on large files (`> 300` lines) before paginating with `view_file`, and `search_dir` supports `case_insensitive` and `file_glob` filtering to narrow matches quickly.

### 2. Pre-Seed Live Docs
Workers have no web or MCP access. Before calling `start_run` on fast-moving frameworks or SDKs, query active docs MCPs once yourself and embed exact API signatures, import paths, and version-specific requirements directly in `shared_contract`.

### 3. Write a Concrete `shared_contract`
Workers cannot talk to each other. Put the cross-worker interface truth in `shared_contract`:
- Exact file paths and module import paths.
- Exact exported symbol names, function signatures, and data shapes.
- Positive language standards (e.g. `"Start every .luau file with --!strict"`, `"Use os.clock(), not time()"`). Avoid vague advice like `"write clean code"`.

### 4. Craft Each Worker's `goal` With Positive Completion Criteria
Structure every `goal` as:
1. **Inputs to read first:** `"Read <file1> and <file2> first with view_file."`
2. **Exact deliverables & acceptance bar:** Name every file to create/modify, its responsibility, and edge cases.
3. **Scope boundary:** `"Only write inside <folder>. If an edit outside <folder> is needed, record it in finish(summary, notes=[...])."`
4. **Completion gate:** `"Write/edit the deliverables once and call finish(summary, notes) immediately—do not re-read files you just wrote to verify them (finish() runs static syntax checks automatically; Boss runs full linters/tests)."`

### 5. Wave Parallel Dependencies
Never launch Task B in parallel with Task A if Task B needs to `view_file` files that Task A has not created yet—either:
- **Wave 1 -> Wave 2:** Run shared/types workers first, collect `get_report`, then run consumers.
- **Or inline contracts:** Put exact exported types/signatures in `shared_contract` so Task B does not wait or search for missing files.

### 6. Right-Size Model Tier & Step Limits
- **Free (`model: null` / `"free"`):** Ideal for isolated modules, tests, documentation, schema files, and well-contracted CRUD/UI controllers (`$0` cost, enforces `:free` slugs).
- **Cheap (`model: "cheap"`):** Use when a task involves complex multi-block refactoring inside a large file (`replace_block` precision) or tricky debugging where free models hit `consecutive_tool_errors`.
- **Heavy (`model: "heavy"`):** Use for complex cross-module reasoning or difficult tasks requiring flagship models (respecting `heavy` tier spend limits).
- **`max_steps_per_worker`:** `12–15` for 1–2 files; `20–25` for 3+ files or edits requiring exploration. For large new modules (`> 300` lines each), keep each worker to 1–2 files so message history stays below pruning thresholds.

## Post-Run Boss Verification (Mandatory)

Workers have **no shell** and only run static syntax checks on `.py`, `.json`, `.toml`, `.js`/`.jsx`/`.ts`/`.tsx`, `.yaml`/`.yml`, `.luau`, `.ps1`, and `.cpp`/`.h` during `finish()` and on any escalation when files were touched. Always:
1. Call `get_report(project_root, run_id)` once ready (inspect `spend_usd`, `cached_tokens`, `diffstat`, and `static_check`).
2. Inspect `notes`, `escalation_reason`, and `stop_reason` for cross-folder follow-ups or escalated workers:
   - **`stalled_after_writes` / `max_steps_reached`:** Do **not** blindly call `rollback`. Inspect `diffstat` and `static_check` first — `stalled_after_writes` can contain valid, syntax-checked deliverables (`static_check.status == "passed"`) where the worker looped on reads after writing. Keep valid outputs or patch minor gaps instead of discarding good work.
   - **`read_loop_detected`:** The worker looped on redundant read-only calls before making writes (`diffstat` empty, `static_check.status == "not_run"`). Tighten `target_files` / `shared_contract` or switch to `"cheap"`/`"heavy"` before retrying.
3. Read or diff the produced files directly — verify types, imports, and edge cases.
4. Run the project's actual compiler, linter, or test suite (`pytest`, `ruff`, `tsc`, `lune`, `selene`, etc.) before declaring completion.
5. If a run produced broken or off-spec edits after inspecting `diffstat` and `static_check`, call `rollback(project_root, run_id)` (or `confirm=True` to skip the interactive MCP elicitation prompt on supporting clients) to cleanly revert created/modified files.
