"""CLI subcommands for Overseer (`overseer-mcp init`, `overseer-mcp doctor`)."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx

from overseer.config import (
    MODEL_SLUG_ALIASES,
    _usable_key,
    config_file_path,
    get_cheap_default_model,
    get_cheap_fallback_models,
    get_config_value,
    get_configured_tier_keys,
    get_default_model,
    get_fallback_models,
    get_heavy_default_model,
    get_heavy_fallback_models,
    get_openrouter_base_url,
    load_config_file_sync,
    load_external_dotenv,
)
from overseer.guardrails import atomic_write_bytes_sync, get_data_dir, redact_secrets
from overseer.openrouter import fetch_openrouter_key_quotas_sync

_KEY_NAMES: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_FREE_API_KEY",
    "OPENROUTER_CHEAP_API_KEY",
    "OPENROUTER_HEAVY_API_KEY",
)


def _format_env_value(value: str) -> str:
    cleaned = value.replace("\r", "").replace("\n", "").strip()
    if any(ch in cleaned for ch in (" ", "#", "'", '"')):
        escaped = cleaned.replace('"', '\\"')
        return f'"{escaped}"'
    return cleaned


def _write_dotenv_keys(updates: dict[str, str]) -> Path:
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / ".env"

    merged: dict[str, str] = load_external_dotenv()
    for key, val in updates.items():
        if val.strip():
            merged[key] = val.strip()

    lines = [
        "# Overseer OpenRouter configuration (kept outside any code repository)",
    ]
    for key in sorted(merged):
        lines.append(f"{key}={_format_env_value(merged[key])}")
    content = ("\n".join(lines) + "\n").encode("utf-8")
    atomic_write_bytes_sync(env_path, content, mode=0o600)
    return env_path


def _verify_openrouter_connectivity(base_url: str, api_key: str) -> tuple[bool, str]:
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return False, redact_secrets(f"HTTP {resp.status_code} from {base_url}/models")
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return False, "Unexpected /models response format."
        return True, f"Connected to {base_url}/models ({len(data)} models visible)."
    except Exception as exc:  # noqa: BLE001
        return False, redact_secrets(f"Connectivity check failed: {exc}")


def _build_mcp_client_snippets(python_exe: str) -> dict[str, dict[str, Any]]:
    overseer_bin = shutil.which("overseer-mcp")
    uvx_path = None if overseer_bin else shutil.which("uvx")
    if overseer_bin:
        command = overseer_bin
        args: list[str] = []
    elif uvx_path:
        command = uvx_path
        args = ["--from", "git+https://github.com/Luther22-source/overseer-mcp", "overseer-mcp"]
    else:
        command = python_exe
        args = ["-m", "overseer.server"]
    server_entry: dict[str, Any] = {
        "command": command,
        "args": args,
        "env": {"PYTHONUNBUFFERED": "1"},
    }
    return {
        "Antigravity (mcp_config.json)": {"mcpServers": {"overseer": server_entry}},
        "Claude Desktop (claude_desktop_config.json)": {"mcpServers": {"overseer": server_entry}},
        "Cursor (.cursor/mcp.json)": {"mcpServers": {"overseer": server_entry}},
        "Windsurf (mcp_config.json)": {"mcpServers": {"overseer": server_entry}},
        "Cline (cline_mcp_settings.json)": {"mcpServers": {"overseer": server_entry}},
    }


def _prompt_secret(prompt_text: str) -> str:
    try:
        return getpass.getpass(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def run_init(argv: list[str] | None = None) -> int:
    """Interactive or non-interactive setup for <data_dir>/.env and MCP client config snippets."""
    parser = argparse.ArgumentParser(prog="overseer-mcp init", description="Initialize Overseer OpenRouter keys.")
    parser.add_argument("--api-key", default=None, help="Shared OPENROUTER_API_KEY value.")
    parser.add_argument("--free-key", default=None, help="Free-tier OPENROUTER_FREE_API_KEY value.")
    parser.add_argument("--cheap-key", default=None, help="Cheap-tier OPENROUTER_CHEAP_API_KEY value.")
    parser.add_argument("--heavy-key", default=None, help="Heavy-tier OPENROUTER_HEAVY_API_KEY value.")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt for missing keys.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip live OpenRouter /models check.")
    args = parser.parse_args(argv)

    updates: dict[str, str] = {}
    if args.api_key:
        updates["OPENROUTER_API_KEY"] = args.api_key.strip()
    if args.free_key:
        updates["OPENROUTER_FREE_API_KEY"] = args.free_key.strip()
    if args.cheap_key:
        updates["OPENROUTER_CHEAP_API_KEY"] = args.cheap_key.strip()
    if args.heavy_key:
        updates["OPENROUTER_HEAVY_API_KEY"] = args.heavy_key.strip()

    if not args.non_interactive and not updates:
        print("Overseer setup: enter at least one OpenRouter API key (input hidden, press Enter to skip).")
        shared = _prompt_secret("OPENROUTER_API_KEY (shared fallback): ")
        if shared:
            updates["OPENROUTER_API_KEY"] = shared
        free_k = _prompt_secret("OPENROUTER_FREE_API_KEY (optional, free tier): ")
        if free_k:
            updates["OPENROUTER_FREE_API_KEY"] = free_k
        cheap_k = _prompt_secret("OPENROUTER_CHEAP_API_KEY (optional, cheap tier): ")
        if cheap_k:
            updates["OPENROUTER_CHEAP_API_KEY"] = cheap_k
        heavy_k = _prompt_secret("OPENROUTER_HEAVY_API_KEY (optional, heavy tier): ")
        if heavy_k:
            updates["OPENROUTER_HEAVY_API_KEY"] = heavy_k

    existing = load_external_dotenv()
    if not updates and not any(existing.get(k) for k in _KEY_NAMES):
        print("Error: no OpenRouter API key provided.", file=sys.stderr)
        return 1

    if updates:
        env_path = _write_dotenv_keys(updates)
        print(f"Wrote Overseer environment file: {env_path}")
    else:
        env_path = get_data_dir() / ".env"
        print(f"Using existing Overseer environment file: {env_path}")

    test_key = (
        updates.get("OPENROUTER_FREE_API_KEY")
        or updates.get("OPENROUTER_API_KEY")
        or updates.get("OPENROUTER_CHEAP_API_KEY")
        or updates.get("OPENROUTER_HEAVY_API_KEY")
        or next((existing[k] for k in _KEY_NAMES if existing.get(k)), "")
    )
    if not args.skip_verify and test_key:
        base_url = get_openrouter_base_url()
        ok, msg = _verify_openrouter_connectivity(base_url, test_key)
        print(f"OpenRouter check: {'OK' if ok else 'WARN'} - {msg}")
        if not ok:
            return 1

    print("\nCopy-paste MCP client snippets:")
    snippets = _build_mcp_client_snippets(sys.executable)
    for label, snippet in snippets.items():
        print(f"\n--- {label} ---")
        print(json.dumps(snippet, indent=2))
    return 0


def _resolve_app_html_path() -> Path:
    override = get_config_value("OVERSEER_APP_HTML")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "ui" / "dist" / "app.html"


def run_doctor(argv: list[str] | None = None) -> int:
    """Verify Python version, key presence, config.json, app.html, ruff, and tier models."""
    parser = argparse.ArgumentParser(prog="overseer-mcp doctor", description="Diagnose Overseer setup.")
    parser.add_argument("--offline", action="store_true", help="Skip live OpenRouter /models verification.")
    args = parser.parse_args(argv)

    failures = 0

    py_ok = sys.version_info >= (3, 11)
    print(f"[{'PASS' if py_ok else 'FAIL'}] Python version: {sys.version.split()[0]} (requires >= 3.11)")
    if not py_ok:
        failures += 1

    key_status = {k: bool(_usable_key(k)) for k in _KEY_NAMES}
    any_key = any(key_status.values())
    present_names = [k for k, present in key_status.items() if present]
    print(
        f"[{'PASS' if any_key else 'FAIL'}] API keys configured: "
        f"{', '.join(present_names) if present_names else 'none'}"
    )
    if not any_key:
        failures += 1

    cfg_path = config_file_path()
    try:
        load_config_file_sync(strict=True)
        cfg_state = "valid" if cfg_path.is_file() else "absent (using defaults)"
        print(f"[PASS] config.json ({cfg_path}): {cfg_state}")
    except ValueError as exc:
        print(f"[FAIL] config.json ({cfg_path}): {exc}")
        failures += 1

    app_html = _resolve_app_html_path()
    if app_html.is_file():
        size_kb = app_html.stat().st_size / 1024.0
        print(f"[PASS] Monitor bundle ({app_html}): {size_kb:.1f} KiB")
    else:
        print(f"[FAIL] Monitor bundle missing: {app_html}")
        failures += 1

    ruff_path = shutil.which("ruff")
    print(f"[{'PASS' if ruff_path else 'INFO'}] ruff: {ruff_path or 'not found (optional)'}")

    if not args.offline:
        tier_keys = get_configured_tier_keys()
        tier_specs: list[tuple[str, str, list[tuple[str, str]]]] = [
            (
                "free",
                tier_keys["free"],
                [("free.primary", get_default_model()), *[("free.fallback", m) for m in get_fallback_models()]],
            ),
            (
                "cheap",
                tier_keys["cheap"],
                [
                    ("cheap.primary", get_cheap_default_model()),
                    *[("cheap.fallback", m) for m in get_cheap_fallback_models()],
                ],
            ),
        ]
        heavy_primary = get_heavy_default_model()
        heavy_fallbacks = get_heavy_fallback_models()
        heavy_slugs: list[tuple[str, str]] = []
        if heavy_primary:
            heavy_slugs.append(("heavy.primary", heavy_primary))
        heavy_slugs.extend(("heavy.fallback", m) for m in heavy_fallbacks)
        if heavy_slugs:
            tier_specs.append(
                (
                    "heavy",
                    tier_keys["heavy"],
                    heavy_slugs,
                )
            )

        if not any(key for _, key, _ in tier_specs):
            print("[FAIL] OpenRouter models check: no API key available.")
            failures += 1
        else:
            try:
                base_url = get_openrouter_base_url()
                catalog_by_key: dict[str, set[str]] = {}
                missing: list[tuple[str, str]] = []
                checked_count = 0
                for tier_name, tier_key, slugs in tier_specs:
                    if not tier_key:
                        print(f"[WARN] Skipping {tier_name} model check: no {tier_name} API key configured.")
                        continue
                    if tier_key not in catalog_by_key:
                        resp = httpx.get(
                            f"{base_url}/models",
                            headers={
                                "Authorization": f"Bearer {tier_key}",
                                "X-OpenRouter-Title": "Overseer MCP",
                            },
                            timeout=15.0,
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                        catalog_by_key[tier_key] = {
                            item["id"]
                            for item in (payload.get("data") or [])
                            if isinstance(item, dict) and isinstance(item.get("id"), str)
                        }
                    ids_for_key = catalog_by_key[tier_key]
                    for label, slug in slugs:
                        checked_count += 1
                        normalized_slug = MODEL_SLUG_ALIASES.get(slug, slug)
                        if normalized_slug not in ids_for_key:
                            missing.append((label, normalized_slug))

                if missing:
                    missing_desc = ", ".join(f"{label}={slug}" for label, slug in missing)
                    print(f"[FAIL] Configured tier models missing on OpenRouter: {missing_desc}")
                    failures += 1
                else:
                    print(f"[PASS] Configured tier models verified on OpenRouter ({checked_count} slugs).")
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] OpenRouter models check: {redact_secrets(str(exc))}")
                failures += 1

        quota_info = fetch_openrouter_key_quotas_sync(timeout=5.0)
        for tier_name in ("free", "cheap", "heavy"):
            tier_q = (quota_info.get("tiers") or {}).get(tier_name) or {}
            if not tier_q.get("configured"):
                continue
            if tier_q.get("status") == "ok":
                rem = tier_q.get("limit_remaining")
                lim = tier_q.get("limit")
                usage = tier_q.get("usage")
                if rem is not None and lim is not None:
                    credits_desc = f"${rem:.4f} / ${lim:.4f} remaining"
                elif rem is not None:
                    credits_desc = f"${rem:.4f} remaining"
                else:
                    credits_desc = "unlimited credits"
                usage_desc = f", usage=${usage:.4f}" if usage is not None else ""
                rl = tier_q.get("rate_limit")
                rl_desc = ""
                if isinstance(rl, dict) and rl.get("requests") is not None:
                    interval = rl.get("interval") or "window"
                    rl_desc = f", rate_limit={rl['requests']}/{interval}"
                print(f"[INFO] OpenRouter key ({tier_name}): {credits_desc}{usage_desc}{rl_desc}")
            else:
                err = tier_q.get("error") or "unavailable"
                print(f"[WARN] OpenRouter key ({tier_name}) quota check: {redact_secrets(str(err))}")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint dispatching `init` and `doctor` subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: overseer-mcp {init,doctor} [options]")
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == "init":
        return run_init(rest)
    if cmd == "doctor":
        return run_doctor(rest)
    print(f"Unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
