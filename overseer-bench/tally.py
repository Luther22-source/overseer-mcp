"""Tally & verification runner for overseer-bench."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from harness import generate_round


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tally(run_dir: Path) -> dict:
    tally_path = run_dir / "tally.json"
    if tally_path.exists():
        return json.loads(tally_path.read_text(encoding="utf-8"))
    return {
        "run_name": run_dir.name,
        "started_at_epoch": None,
        "started_at_iso": None,
        "rounds": {},
    }


def _save_tally(run_dir: Path, data: dict) -> None:
    (run_dir / "tally.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    rounds = data["rounds"]
    passed = sum(1 for r in rounds.values() if r["status"] == "PASS")
    failed = sum(1 for r in rounds.values() if r["status"] == "FAIL")
    total = len(rounds)
    started = data["started_at_epoch"]
    last_epoch = max((r["verified_at_epoch"] for r in rounds.values()), default=started)
    elapsed = round((last_epoch - started), 2) if started and last_epoch else 0.0

    lines = [
        f"# Tally: {data['run_name']}",
        "",
        f"- **Started**: {data['started_at_iso'] or 'Not started'}",
        f"- **Completed / Passed**: **{passed}** / {total} attempted ({failed} failed)",
        f"- **Elapsed (start -> latest verify)**: {elapsed:.2f}s",
        "",
        "| Round | Family | Status | Test Runtime (s) | Cumulative Elapsed (s) |",
        "| :--- | :--- | :--- | ---: | ---: |",
    ]
    for r_key in sorted(rounds.keys(), key=int):
        r = rounds[r_key]
        cum = round(r["verified_at_epoch"] - started, 2) if started else 0.0
        lines.append(
            f"| {r_key} | {r['family']} | {r['status']} | {r['test_seconds']:.3f} | {cum:.2f} |"
        )
    (run_dir / "tally.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _load_tally(run_dir)
    if data["started_at_epoch"] is None:
        data["started_at_epoch"] = time.time()
        data["started_at_iso"] = _now_iso()
        _save_tally(run_dir, data)

    created = []
    for rid in range(args.start, args.start + args.count):
        r_dir = run_dir / "rounds" / f"round_{rid:03d}"
        fam = generate_round(rid, r_dir)
        created.append((rid, fam, str(r_dir)))
    print(f"Prepared {len(created)} rounds in {run_dir / 'rounds'} (start={args.start}, count={args.count})")
    for rid, fam, path in created:
        print(f"  round_{rid:03d} [{fam}] -> {path}")


def cmd_verify(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    data = _load_tally(run_dir)
    if data["started_at_epoch"] is None:
        data["started_at_epoch"] = time.time()
        data["started_at_iso"] = _now_iso()

    for rid in args.rounds:
        r_dir = run_dir / "rounds" / f"round_{rid:03d}"
        if not (r_dir / "test_solution.py").exists():
            print(f"round_{rid:03d}: MISSING test_solution.py", file=sys.stderr)
            continue
        fam = ["vm", "route", "dsl", "scheduler"][(rid - 1) % 4]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "test_solution.py"],
                cwd=str(r_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            status = "PASS" if proc.returncode == 0 else "FAIL"
            if status == "PASS":
                detail = ""
            else:
                out_lines = (proc.stderr or proc.stdout).strip().splitlines()
                detail = out_lines[-1] if out_lines else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "FAIL"
            detail = "TIMEOUT (>10s)"
        dt = time.perf_counter() - t0
        now_e = time.time()
        data["rounds"][str(rid)] = {
            "family": fam,
            "status": status,
            "test_seconds": round(dt, 4),
            "verified_at_epoch": now_e,
            "verified_at_iso": _now_iso(),
            "detail": detail,
        }
        print(f"round_{rid:03d} [{fam}]: {status} ({dt:.3f}s) {detail}")

    _save_tally(run_dir, data)
    passed = sum(1 for r in data["rounds"].values() if r["status"] == "PASS")
    total = len(data["rounds"])
    elapsed = time.time() - data["started_at_epoch"]
    print(f"TALLY [{data['run_name']}]: {passed} PASS / {total} total | elapsed {elapsed:.2f}s")


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    data = _load_tally(run_dir)
    passed = sum(1 for r in data["rounds"].values() if r["status"] == "PASS")
    total = len(data["rounds"])
    started = data["started_at_epoch"]
    elapsed = (time.time() - started) if started else 0.0
    print(f"TALLY [{data['run_name']}]: {passed} PASS / {total} total | elapsed {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tally runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("run_dir")
    p_prep.add_argument("--start", type=int, default=1)
    p_prep.add_argument("--count", type=int, default=10)
    p_prep.set_defaults(func=cmd_prepare)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("run_dir")
    p_ver.add_argument("rounds", type=int, nargs="+")
    p_ver.set_defaults(func=cmd_verify)

    p_stat = sub.add_parser("status")
    p_stat.add_argument("run_dir")
    p_stat.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
