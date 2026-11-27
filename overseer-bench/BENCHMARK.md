# Overseer Usage & Speed Benchmark (`overseer-bench`)

## Rules
- **Do not read `harness.py`** during either run. Only work inside `run_normal/rounds/round_XXX` (Run A) or dispatch workers scoped to `run_overseer/rounds/round_XXX` (Run B).
- Both runs solve the exact same deterministic sequence (`round_001, round_002, ...`).

## Run A: Normal Solo Loop
Repeat indefinitely until the user interrupts or tells you to stop:
1. Prepare the next round: `python tally.py prepare run_normal --start <N> --count 1`
2. Read `run_normal/rounds/round_<NNN>/README.md` and write `run_normal/rounds/round_<NNN>/solution.py`.
3. Verify and log tally: `python tally.py verify run_normal <N>`
4. Fix `solution.py` and re-verify until `PASS`, then immediately advance to `N + 1`.

## Run B: Overseer MCP (Batches of 10, Default Free Model)
1. Prepare the next 10 rounds: `python tally.py prepare run_overseer --start <N> --count 10`
2. Call `overseer:start_run` with:
   - `project_root`: the absolute path to this `overseer-bench` directory (must NOT be inside `~/.gemini/`)
   - `tasks`: 10 tasks (`worker_id`: `"round_<NNN>"`, `folder`: `"run_overseer/rounds/round_<NNN>"`, `target_files`: `["solution.py"]`, `model`: `null`, `goal`: `"Read README.md and implement solution.py so that python -m unittest test_solution.py passes completely."`)
3. **Do not poll `get_report`.** Acknowledge that the batch is running and wait for the user to ask to collect results.
4. When the user asks for results: call `overseer:get_report`, run `python tally.py verify run_overseer <N..N+9>`, report the `tally.md` summary, and immediately prepare + start the next batch of 10 (unless told to stop).
