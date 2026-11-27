"""Regression and feature tests for Overseer v1.1 hardening, CLI, UI app tools, and cross-platform support."""

from __future__ import annotations

import asyncio
import json
import os
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from overseer import cli, notify
from overseer.config import (
    DEFAULT_CHEAP_MAX_RUN_USD,
    data_dir_file_lock,
    get_effective_config_snapshot,
    get_max_concurrent_workers,
    resolve_effective_max_run_usd,
    save_config_file_sync,
)
from overseer.guardrails import (
    extract_file_outline,
    get_project_data_dir,
    is_denied_rel_path,
    redact_secrets,
    run_static_precheck,
    sha256_bytes,
)
from overseer.models import (
    CreatedFileRecord,
    FileEditRecord,
    ListDirArgs,
    OverseerConfigFile,
    ProviderRoutingRule,
    RunLimits,
    RunManifest,
    SearchDirArgs,
    StartRunRequest,
    TierConfig,
    WorkerManifestState,
    WorkerTaskInput,
)
from overseer.openrouter import (
    WORKER_TOOL_SCHEMAS,
    ModelCatalog,
    _call_openrouter_with_failover,
    fetch_openrouter_key_quotas,
    parse_rate_limit_wait_seconds,
)
from overseer.server import (
    APP_URI,
    OverseerServerState,
    _MissingApiKeySchema,
    _RollbackConfirmSchema,
    create_server,
)
from overseer.state import (
    RunRegistry,
    active_lock_path_for,
    collect_run_report,
    list_all_runs_snapshot,
    manifest_path_for,
    rollback_run,
    save_manifest_atomic,
    write_worker_patch_sync,
)
from overseer.worker import (
    WorkerToolContext,
    _build_worker_system_prompt,
    _list_dir_sync,
    _load_repo_instructions_sync,
    _search_dir_sync,
    execute_worker_tool,
    prepare_and_launch_run,
)


def _extract_tool_payload(res: Any) -> dict[str, Any]:
    if isinstance(res, dict):
        return res
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        return res[1]
    if isinstance(res, list) and res:
        first = res[0]
        if hasattr(first, "text"):
            return json.loads(first.text)
    if hasattr(res, "content") and res.content:
        return json.loads(res.content[0].text)
    raise AssertionError(f"Unexpected tool result shape: {res!r}")


# ---------------------------------------------------------------------------
# 1. Config, budget cap, file lock, and env_overrides
# ---------------------------------------------------------------------------


def test_resolve_effective_max_run_usd_mixed_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    save_config_file_sync(
        OverseerConfigFile(
            tiers={
                "cheap": TierConfig(primary="deepseek/deepseek-v4.1-flash-1017", max_run_usd=0.40),
                "heavy": TierConfig(primary="anthropic/claude-sonnet-4.6", max_run_usd=3.50),
            }
        )
    )

    assert resolve_effective_max_run_usd(["free", "free"], None) == DEFAULT_CHEAP_MAX_RUN_USD
    assert resolve_effective_max_run_usd(["free", "cheap"], None) == 0.40
    assert resolve_effective_max_run_usd(["free", "heavy"], None) == 3.50
    assert resolve_effective_max_run_usd(["free", "cheap"], 0.15) == 0.15


def test_data_dir_file_lock_token_ownership_and_pid_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".cfgtest.lock"

    # Stale lock owned by a live PID must NOT be reclaimed
    lock_path.write_text(f"{os.getpid()}:other-uuid", encoding="ascii")
    old = time.time() - 60.0
    os.utime(lock_path, (old, old))
    monkeypatch.setattr("overseer.config.is_pid_alive", lambda pid: pid == os.getpid())
    with pytest.raises(TimeoutError):
        with data_dir_file_lock("cfgtest", timeout_sec=0.06, stale_sec=1.0):
            pass
    assert lock_path.exists()

    # Stale lock owned by a dead PID IS reclaimed, and finally only unlinks own token
    lock_path.write_text("999999:dead-uuid", encoding="ascii")
    os.utime(lock_path, (old, old))
    monkeypatch.setattr("overseer.config.is_pid_alive", lambda pid: False)
    with data_dir_file_lock("cfgtest", timeout_sec=0.5, stale_sec=1.0):
        assert lock_path.read_text(encoding="ascii").startswith(f"{os.getpid()}:")
        # Simulate another process replacing the lock file before our finally runs
        lock_path.write_text("12345:foreign-token", encoding="ascii")
    assert lock_path.read_text(encoding="ascii") == "12345:foreign-token"


def test_max_concurrent_workers_precedence_and_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    (tmp_path / "appdata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "appdata" / ".env").write_text("OVERSEER_MAX_CONCURRENT_WORKERS=7\n", encoding="utf-8")
    assert get_max_concurrent_workers() == 7

    save_config_file_sync(OverseerConfigFile(max_concurrent_workers=5))
    assert get_max_concurrent_workers() == 5

    monkeypatch.setenv("OVERSEER_MAX_CONCURRENT_WORKERS", "99")
    assert get_max_concurrent_workers() == 10

    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "qwen/qwen3-coder:free")
    snap = get_effective_config_snapshot()
    assert snap["max_concurrent_workers"] == 10
    assert snap["env_overrides"]["max_concurrent_workers"] == "OVERSEER_MAX_CONCURRENT_WORKERS"
    assert snap["env_overrides"]["tiers.free.primary"] == "OVERSEER_FREE_DEFAULT_MODEL"


# ---------------------------------------------------------------------------
# 2. prepare_and_launch_run failure cleanup ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_and_launch_run_failure_cleanup_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-test-free")
    (tmp_path / "proj" / "src").mkdir(parents=True)
    proj = tmp_path / "proj"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "particle-ai/particle-1:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected HTTP call: {request.url}")

    events: list[str] = []
    orig_await_commits = RunRegistry.await_commits
    orig_mark_finished = RunRegistry.mark_run_finished

    async def tracked_await_commits(self: RunRegistry, project_root: Path, run_id: str) -> None:
        events.append("await_commits")
        await orig_await_commits(self, project_root, run_id)

    def tracked_mark_finished(self: RunRegistry, project_root: Path, run_id: str) -> None:
        events.append("mark_run_finished")
        orig_mark_finished(self, project_root, run_id)

    async def failing_latest_pointer(project_data_dir: Path, run_id: str) -> None:
        raise OSError("simulated pointer write failure")

    monkeypatch.setattr(RunRegistry, "await_commits", tracked_await_commits)
    monkeypatch.setattr(RunRegistry, "mark_run_finished", tracked_mark_finished)
    monkeypatch.setattr("overseer.worker.write_latest_run_pointer", failing_latest_pointer)

    registry = RunRegistry()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        req = StartRunRequest(
            project_root=str(proj),
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="Do work")],
        )
        with pytest.raises(OSError, match="simulated pointer write failure"):
            await prepare_and_launch_run(req, registry, client, ModelCatalog(), proj)

        assert events == ["await_commits", "mark_run_finished"]
        assert registry.has_active_run(proj) is None
        assert not active_lock_path_for(get_project_data_dir(proj)).exists()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 3. OpenRouter ModelCatalog retry, list_models, suggest_free_tool_models, 429=0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_catalog_retry_list_and_suggest_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    attempts = 0
    seen_titles: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        seen_titles.append(request.headers.get("X-OpenRouter-Title"))
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        if attempts == 2:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "z-org/free-small:free",
                        "name": "Free Small",
                        "context_length": 32768,
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {
                        "id": "a-org/free-big:free",
                        "name": "Free Big",
                        "context_length": 131072,
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {
                        "id": "paid/coder",
                        "name": "Paid Coder",
                        "context_length": 65536,
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    },
                ]
            },
        )

    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await catalog.ensure_loaded(client, "sk-or-test")

    assert attempts == 3
    assert seen_titles == ["Overseer MCP"] * 3
    assert catalog.suggest_free_tool_models(limit=2) == [
        "a-org/free-big:free",
        "z-org/free-small:free",
    ]
    free_list = catalog.list_models(free_only=True, tools_only=True, search="big")
    assert len(free_list) == 1
    assert free_list[0]["id"] == "a-org/free-big:free"

    with pytest.raises(ValueError, match="Suggested active :free tool models: a-org/free-big:free"):
        catalog.validate_tool_support(["retired/model:free"])


@pytest.mark.asyncio
async def test_openrouter_zero_429_wait_headers_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-free-key")

    # X-RateLimit-Reset epoch ms parsing
    future_ms = str(int((time.time() + 5.0) * 1000))
    parsed_wait = parse_rate_limit_wait_seconds({"X-RateLimit-Reset": future_ms})
    assert parsed_wait is not None
    assert 3.0 <= parsed_wait <= 6.0

    sleep_calls: list[float] = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await orig_sleep(0)

    monkeypatch.setattr("overseer.openrouter.asyncio.sleep", fake_sleep)

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-OpenRouter-Title") == "Overseer MCP"
        body = json.loads(request.content.decode("utf-8"))
        model = body["model"]
        calls.append(model)
        if model == "m1:free":
            return httpx.Response(429, headers={"Retry-After": "10"})
        if model == "m2:free":
            return httpx.Response(200, json={"error": {"code": 429, "message": "rate limited"}})
        if model == "m3:free" and calls.count("m3:free") == 1:
            return httpx.Response(200, text="not-json")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, used_model, idx = await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-free-key",
            model_chain=["m1:free", "m2:free", "m3:free"],
            start_model_idx=0,
            messages=[{"role": "user", "content": "hi"}],
            limits=RunLimits(max_429_wait_seconds=0),
            tier="free",
        )
    assert used_model == "m3:free"
    assert idx == 2
    assert data["choices"][0]["message"]["content"] == "ok"
    # m1 and m2 failed over immediately without sleeping; m3 invalid JSON used safe >=0.05 floor
    assert calls == ["m1:free", "m2:free", "m3:free", "m3:free"]
    assert sleep_calls == [0.05]

    # Network error redaction
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "failed connecting to https://user:superpass@proxy.local/?api_key=secret123",
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing_handler)) as client:
        with pytest.raises(RuntimeError) as exc_info:
            await _call_openrouter_with_failover(
                client=client,
                api_key="sk-or-free-key",
                model_chain=["m1:free"],
                start_model_idx=0,
                messages=[{"role": "user", "content": "hi"}],
                limits=RunLimits(max_429_wait_seconds=0),
                tier="free",
            )
    err_str = str(exc_info.value)
    assert "superpass" not in err_str
    assert "secret123" not in err_str
    assert "[REDACTED]" in err_str


# ---------------------------------------------------------------------------
# 4. Guardrails: redaction, DENY_GLOBS, outlines, static precheck, Luau strict
# ---------------------------------------------------------------------------


def test_redact_secrets_url_basic_auth_and_query_params() -> None:
    sample = (
        "Bearer sk-or-v1-abcdef1234567890 "
        "https://alice:mysecretpass@example.com/v1?api_key=topsecret&token=tok123&ok=1"
    )
    redacted = redact_secrets(sample)
    assert "mysecretpass" not in redacted
    assert "topsecret" not in redacted
    assert "tok123" not in redacted
    assert "sk-or-v1-abcdef1234567890" not in redacted
    assert "https://[REDACTED]@example.com/v1?api_key=[REDACTED]&token=[REDACTED]&ok=1" in redacted


def test_deny_globs_new_patterns() -> None:
    for denied in (
        ".overseer/instructions.md",
        ".github/workflows/ci.yml",
        ".vscode/settings.json",
        ".idea/workspace.xml",
        ".gemini/config.json",
        ".claude/settings.json",
        ".cursor/mcp.json",
        "config/my-credentials.yaml",
        "deploy/app-secret.toml",
        ".git-credentials",
        "id_ed25519",
    ):
        assert is_denied_rel_path(denied), f"Expected denied: {denied}"


def test_outlines_and_static_precheck_jsts_yaml_go_rust_luau(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))

    ts_src = "export interface User { id: number }\nexport async function getUser(id: number) {\n  return { id };\n}\n"
    assert "L1: export interface User" in extract_file_outline(ts_src, "src/user.ts")
    assert "L2: export async function getUser(id: number)" in extract_file_outline(ts_src, "src/user.ts")

    yaml_src = "name: overseer\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
    assert "L1: name:" in extract_file_outline(yaml_src, "ci.yaml")
    assert "L2: jobs:" in extract_file_outline(yaml_src, "ci.yaml")

    go_src = "package main\ntype Server struct {\n\tPort int\n}\nfunc (s *Server) Start() error {\n\treturn nil\n}\n"
    assert "L2: type Server struct" in extract_file_outline(go_src, "main.go")
    assert "L5: func (s *Server) Start() error" in extract_file_outline(go_src, "main.go")

    rs_src = "pub struct Pool {}\nimpl Pool {\n    fn new() -> Self { Self {} }\n}\npub async fn run() {}\n"
    assert "L1: pub struct Pool" in extract_file_outline(rs_src, "lib.rs")
    assert "L2: impl Pool" in extract_file_outline(rs_src, "lib.rs")
    assert "L5: pub async fn run()" in extract_file_outline(rs_src, "lib.rs")

    (tmp_path / "ok.ts").write_text(ts_src, encoding="utf-8")
    (tmp_path / "bad.ts").write_text("export function broken() {\n  return (1 + 2;\n}\n", encoding="utf-8")
    (tmp_path / "ok.yaml").write_text(yaml_src, encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("a: {\n  b: 1\n", encoding="utf-8")
    (tmp_path / "script.luau").write_text("local x = 1\n", encoding="utf-8")

    assert run_static_precheck(tmp_path, ["ok.ts", "ok.yaml", "script.luau"]).status == "passed"
    bad_ts = run_static_precheck(tmp_path, ["bad.ts"])
    assert bad_ts.status == "failed"
    assert any("Unclosed '('" in e for e in bad_ts.errors)

    bad_yaml = run_static_precheck(tmp_path, ["bad.yaml"])
    assert bad_yaml.status == "failed"

    monkeypatch.setenv("OVERSEER_LUAU_REQUIRE_STRICT", "1")
    luau_res = run_static_precheck(tmp_path, ["script.luau"])
    assert luau_res.status == "failed"
    assert any("--!strict" in e for e in luau_res.errors)


# ---------------------------------------------------------------------------
# 5. Worker tools: prompt root omission, .overseer/instructions.md, search_dir, symlinks, error redaction
# ---------------------------------------------------------------------------


def test_worker_prompt_omits_abs_root_and_caps_repo_instructions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    proj = tmp_path / "my-secret-workspace"
    (proj / ".overseer").mkdir(parents=True)
    instr_path = proj / ".overseer" / "instructions.md"
    instr_path.write_bytes(b"A" * 15_000)

    loaded = _load_repo_instructions_sync(proj)
    assert len(loaded) == 10_240

    prompt = _build_worker_system_prompt(
        proj,
        WorkerTaskInput(worker_id="w1", folder=".", goal="Check prompt"),
        shared_contract=None,
    )
    assert "Read-only project root: my-secret-workspace" in prompt
    assert str(proj) not in prompt
    assert "A" * 100 in prompt


def test_search_dir_utf8_sig_case_insensitive_and_file_glob(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.ts").write_bytes(b"\xef\xbb\xbfexport const HelloWorld = 1;\n")
    (tmp_path / "src" / "notes.md").write_text("helloworld in markdown\n", encoding="utf-8")

    res = _search_dir_sync(
        tmp_path,
        SearchDirArgs(query="helloworld", path="src", case_insensitive=True, file_glob="*.ts"),
    )
    assert "src/app.ts:1:" in res
    assert "notes.md" not in res


def test_list_dir_filters_symlink_targets(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "ok.txt").write_text("ok", encoding="utf-8")
    (proj / ".env").write_text("SECRET=1", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    try:
        (proj / "link_outside.txt").symlink_to(outside)
        (proj / "link_env.txt").symlink_to(proj / ".env")
        (proj / "link_ok.txt").symlink_to(proj / "ok.txt")
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this host")

    listing = _list_dir_sync(proj, ListDirArgs(path="."))
    assert "ok.txt" in listing
    assert "link_ok.txt" in listing
    assert "link_outside.txt" not in listing
    assert "link_env.txt" not in listing


@pytest.mark.asyncio
async def test_execute_worker_tool_redacts_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    (tmp_path / "src").mkdir()
    manifest = RunManifest(
        version="0.1.0",
        run_id="run_20260101_000000_abcdef",
        server_pid=os.getpid(),
        created_at_iso="2026-01-01T00:00:00Z",
        project_root=str(tmp_path),
        limits=RunLimits(),
        patch_dir=str(tmp_path / "appdata"),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="src",
                goal="g",
                requested_model="m:free",
            )
        },
    )
    ctx = WorkerToolContext(
        project_root=tmp_path,
        folder_abs=tmp_path / "src",
        worker=manifest.workers["w1"],
        manifest=manifest,
        manifest_path=tmp_path / "appdata" / "manifest.json",
        run_lock=asyncio.Lock(),
        backups_dir=tmp_path / "appdata" / "backups",
        run_dir=tmp_path / "appdata" / "run",
        registry=RunRegistry(),
        cancel_event=asyncio.Event(),
    )
    out, is_err = await execute_worker_tool(
        "view_file",
        {"path": "https://user:secretpass@host/file?api_key=sk-or-secretkey123456"},
        ctx,
    )
    assert is_err is True
    assert "secretpass" not in out
    assert "sk-or-secretkey123456" not in out


# ---------------------------------------------------------------------------
# 6. State: backup_path containment + pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backup_path_containment_in_patch_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    target = proj / "src" / "a.py"
    target.write_text("new = 1\n", encoding="utf-8")

    outside_backup = tmp_path / "outside_backup.txt"
    outside_backup.write_text("old = 0\n", encoding="utf-8")

    p_data = get_project_data_dir(proj)
    run_id = "run_20260101_000001_aaaaaa"
    run_dir = p_data / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(p_data, run_id)

    worker = WorkerManifestState(
        worker_id="w1",
        folder="src",
        goal="g",
        requested_model="m:free",
        status="completed",
        modified_files={
            "src/a.py": FileEditRecord(
                backup_path=str(outside_backup),
                initial_hash=sha256_bytes(b"old = 0\n"),
                worker_last_hash=sha256_bytes(b"new = 1\n"),
            )
        },
    )
    patch_file = write_worker_patch_sync(proj, run_dir, worker)
    assert patch_file is not None
    assert "outside backups directory" in Path(patch_file).read_text(encoding="utf-8")

    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=os.getpid(),
        created_at_iso="2026-01-01T00:00:01Z",
        project_root=str(proj),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={"w1": worker},
    )
    await save_manifest_atomic(m_path, manifest, asyncio.Lock())

    rb = await rollback_run(proj, run_id, RunRegistry())
    assert rb.skipped_conflicts == ["src/a.py"]
    assert target.read_text(encoding="utf-8") == "new = 1\n"


@pytest.mark.asyncio
async def test_list_all_runs_snapshot_pagination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    proj = tmp_path / "proj"
    proj.mkdir()
    p_data = get_project_data_dir(proj)
    for idx in range(3):
        rid = f"run_20260101_00000{idx}_00000{idx}"
        rdir = p_data / "runs" / rid
        rdir.mkdir(parents=True, exist_ok=True)
        m = RunManifest(
            version="0.1.0",
            run_id=rid,
            server_pid=os.getpid(),
            created_at_iso=f"2026-01-01T00:00:0{idx}Z",
            project_root=str(proj),
            status="completed",
            limits=RunLimits(),
            patch_dir=str(rdir),
            workers={},
        )
        await save_manifest_atomic(manifest_path_for(p_data, rid), m, asyncio.Lock())

    reg = RunRegistry()
    page1 = await list_all_runs_snapshot(reg, project_root=proj, limit=2, offset=0)
    assert page1["total_count"] == 3
    assert page1["has_more"] is True
    assert len(page1["runs"]) == 2

    page2 = await list_all_runs_snapshot(reg, project_root=proj, limit=2, offset=2)
    assert page2["total_count"] == 3
    assert page2["has_more"] is False
    assert len(page2["runs"]) == 1


# ---------------------------------------------------------------------------
# 7. Server Boss & App tools: list_models, app_*, ToolAnnotations, summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_new_boss_and_app_tools_and_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-free-test")
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    created_file = proj / "src" / "new.py"
    created_bytes = b"x = 1\n"
    created_file.write_bytes(created_bytes)

    p_data = get_project_data_dir(proj)
    rid = "run_20260101_000010_abcdef"
    rdir = p_data / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "w1.patch").write_text("+++ b/src/new.py\n+x = 1\n", encoding="utf-8")
    m = RunManifest(
        version="0.1.0",
        run_id=rid,
        server_pid=os.getpid(),
        created_at_iso="2026-01-01T00:00:10Z",
        project_root=str(proj),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(rdir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="src",
                goal="create file",
                requested_model="a/free-model:free",
                status="completed",
                created_files={
                    "src/new.py": CreatedFileRecord(
                        worker_last_hash=sha256_bytes(created_bytes),
                        line_count=1,
                    )
                },
            )
        },
    )
    await save_manifest_atomic(manifest_path_for(p_data, rid), m, asyncio.Lock())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "a/free-model:free",
                            "name": "Free Model",
                            "context_length": 64000,
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        tools = await mcp.list_tools()
        by_name = {t.name: t for t in tools}
        for boss_tool in (
            "start_run",
            "get_report",
            "cancel_run",
            "rollback",
            "get_config",
            "update_config",
            "list_models",
            "open_overseer_monitor",
        ):
            assert by_name[boss_tool].annotations is not None

        boss_models = _extract_tool_payload(await mcp.call_tool("list_models", {"free_only": True}))
        assert boss_models["count"] == 1
        app_models = _extract_tool_payload(await mcp.call_tool("app_list_models", {"free_only": True}))
        assert app_models["count"] == 1

        monitor = _extract_tool_payload(
            await mcp.call_tool("open_overseer_monitor", {"project_root": str(proj), "run_id": rid})
        )
        assert rid in monitor["summary"]

        patch_res = _extract_tool_payload(
            await mcp.call_tool(
                "app_get_worker_patch",
                {"project_root": str(proj), "run_id": rid, "worker_id": "w1"},
            )
        )
        assert "src/new.py" in patch_res["patch"]

        with pytest.raises(ToolError, match="confirm=True"):
            await mcp.call_tool(
                "app_rollback_run",
                {"project_root": str(proj), "run_id": rid, "confirm": False},
            )

        rb_res = _extract_tool_payload(
            await mcp.call_tool(
                "app_rollback_run",
                {"project_root": str(proj), "run_id": rid, "confirm": True},
            )
        )
        assert rb_res["deleted_files"] == ["src/new.py"]
        assert not created_file.exists()
    finally:
        await state.aclose()


# ---------------------------------------------------------------------------
# 8. CLI (init, doctor) & cross-platform notify
# ---------------------------------------------------------------------------


def test_cli_init_and_doctor_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    rc = cli.run_init(
        [
            "--api-key",
            "sk-or-cli-secret-123456",
            "--non-interactive",
            "--skip-verify",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cline (cline_mcp_settings.json)" in out
    assert '"PYTHONUNBUFFERED": "1"' in out
    assert (tmp_path / "appdata" / ".env").is_file()

    fake_html = tmp_path / "app.html"
    fake_html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("OVERSEER_APP_HTML", str(fake_html))

    rc_doc = cli.run_doctor(["--offline"])
    assert rc_doc == 0
    doc_out = capsys.readouterr().out
    assert "sk-or-cli-secret-123456" not in doc_out
    assert "OPENROUTER_API_KEY" in doc_out

    # Re-running init --non-interactive with no key updates must preserve existing .env comments
    env_file = tmp_path / "appdata" / ".env"
    custom_env = "# keep my custom comment\nOPENROUTER_API_KEY=sk-or-cli-secret-123456\n"
    env_file.write_text(custom_env, encoding="utf-8")
    rc_noop = cli.run_init(["--non-interactive", "--skip-verify"])
    assert rc_noop == 0
    noop_out = capsys.readouterr().out
    assert "Using existing Overseer environment file:" in noop_out
    assert env_file.read_text(encoding="utf-8") == custom_env

    # Snippet builder prefers installed overseer-mcp binary over uvx over python -m
    with monkeypatch.context() as mp:
        mp.setattr("overseer.cli.shutil.which", lambda name: "/usr/local/bin/overseer-mcp" if name == "overseer-mcp" else "/usr/local/bin/uvx")
        entry = cli._build_mcp_client_snippets("/usr/bin/python3")["Cursor (.cursor/mcp.json)"]["mcpServers"]["overseer"]
        assert entry["command"] == "/usr/local/bin/overseer-mcp"
        assert entry["args"] == []
    with monkeypatch.context() as mp:
        mp.setattr("overseer.cli.shutil.which", lambda name: "/usr/local/bin/uvx" if name == "uvx" else None)
        entry = cli._build_mcp_client_snippets("/usr/bin/python3")["Cursor (.cursor/mcp.json)"]["mcpServers"]["overseer"]
        assert entry["command"] == "/usr/local/bin/uvx"
        assert entry["args"] == ["--from", "git+https://github.com/Luther22-source/overseer-mcp", "overseer-mcp"]
    with monkeypatch.context() as mp:
        mp.setattr("overseer.cli.shutil.which", lambda _name: None)
        entry = cli._build_mcp_client_snippets("/usr/bin/python3")["Cursor (.cursor/mcp.json)"]["mcpServers"]["overseer"]
        assert entry["command"] == "/usr/bin/python3"
        assert entry["args"] == ["-m", "overseer.server"]


@pytest.mark.asyncio
async def test_notify_cross_platform_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-leak-to-notify")
    captured: list[tuple[list[str], dict[str, str]]] = []

    async def fake_run(
        cmd: list[str],
        env: dict[str, str],
        timeout_sec: float,
        creationflags: int = 0,
    ) -> bool:
        captured.append((cmd, env))
        return True

    monkeypatch.setattr(notify, "_run_notify_subprocess", fake_run)

    monkeypatch.setattr(notify.sys, "platform", "win32")
    assert await notify.send_windows_toast("Title", "Body") is True
    assert captured[-1][0][0].lower().endswith("powershell.exe")
    assert "OPENROUTER_API_KEY" not in captured[-1][1]
    assert captured[-1][1]["OVERSEER_TOAST_TITLE"] == "Title"

    monkeypatch.setattr(notify.sys, "platform", "darwin")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/osascript" if name == "osascript" else None)
    assert await notify.send_completion_toast("T2", "B2") is True
    assert captured[-1][0][0] == "/usr/bin/osascript"
    assert "OPENROUTER_API_KEY" not in captured[-1][1]

    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/notify-send" if name == "notify-send" else None)
    assert await notify.send_completion_toast("T3", "B3") is True
    assert captured[-1][0] == ["/usr/bin/notify-send", "--", "T3", "B3"]
    assert "OPENROUTER_API_KEY" not in captured[-1][1]


# ---------------------------------------------------------------------------
# 9. Bucket B #1: Worker prompt caching & cached_tokens reporting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_caching_supported_unsupported_400_fallback_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-cache-test")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    seen_bodies: list[dict[str, Any]] = []
    fail_once = {"400": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "qwen/qwen3-coder:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        seen_bodies.append(body)
        if body["model"] == "anthropic/claude-sonnet-4.6" and fail_once["400"]:
            fail_once["400"] = False
            return httpx.Response(400, json={"error": {"message": "cache_control rejected"}})
        return httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_fin",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"summary": "done", "notes": []}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    orig_messages = [
        {"role": "system", "content": "You are a worker."},
        {"role": "user", "content": "Inspect src."},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # Supported model: attaches cache_control without mutating inputs
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-cache-test",
            model_chain=["qwen/qwen3-coder:free"],
            start_model_idx=0,
            messages=orig_messages,
            limits=RunLimits(max_429_wait_seconds=0),
            tier="free",
        )
        assert isinstance(orig_messages[0]["content"], str)
        assert "cache_control" not in WORKER_TOOL_SCHEMAS[-1]
        assert seen_bodies[-1]["messages"][0]["content"] == [
            {"type": "text", "text": "You are a worker.", "cache_control": {"type": "ephemeral"}}
        ]
        assert seen_bodies[-1]["tools"][-1]["cache_control"] == {"type": "ephemeral"}

        # Unsupported model: omits cache_control
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-cache-test",
            model_chain=["deepseek/deepseek-v4.1-flash-1017"],
            start_model_idx=0,
            messages=orig_messages,
            limits=RunLimits(max_429_wait_seconds=0),
            tier="cheap",
        )
        assert seen_bodies[-1]["messages"][0]["content"] == "You are a worker."
        assert "cache_control" not in seen_bodies[-1]["tools"][-1]

        # Supported model with HTTP 400 rejection: retries once without cache_control and succeeds
        before_400 = len(seen_bodies)
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-cache-test",
            model_chain=["anthropic/claude-sonnet-4.6"],
            start_model_idx=0,
            messages=orig_messages,
            limits=RunLimits(max_429_wait_seconds=0),
            tier="heavy",
        )
        assert len(seen_bodies) == before_400 + 2
        assert "cache_control" in seen_bodies[before_400]["tools"][-1]
        assert "cache_control" not in seen_bodies[before_400 + 1]["tools"][-1]
        assert seen_bodies[before_400 + 1]["messages"][0]["content"] == "You are a worker."

        # End-to-end worker run surfaces cached_tokens in get_report
        proj = tmp_path / "proj"
        (proj / "src").mkdir(parents=True)
        (proj / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "qwen/qwen3-coder:free")
        monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "")
        registry = RunRegistry()
        req = StartRunRequest(
            project_root=str(proj),
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="Finish")],
        )
        ack = await prepare_and_launch_run(req, registry, client, ModelCatalog(), proj)
        for _ in range(100):
            if not registry.is_run_active(proj, ack.run_id):
                break
            await asyncio.sleep(0.02)
        report = await collect_run_report(proj, ack.run_id, registry)
        assert report.cached_tokens == 64
        assert report.workers[0].cached_tokens == 64


# ---------------------------------------------------------------------------
# 10. Bucket B #2: GET /api/v1/key preflight & zdr/quantizations routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_quota_preflight_dedup_redaction_and_get_config_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    # Unconfigured
    unconf = await fetch_openrouter_key_quotas()
    assert unconf["status"] == "unconfigured"
    assert all(unconf["tiers"][t]["status"] == "unconfigured" for t in ("free", "cheap", "heavy"))

    # Distinct keys: free has keyA; cheap & heavy share keyB via OPENROUTER_API_KEY
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-free-keyA-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shared-keyB-secret")

    key_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/key")
        auth = request.headers.get("Authorization", "")
        key_calls.append(auth)
        if auth == "Bearer sk-or-free-keyA-secret":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "limit": 10.0,
                        "limit_remaining": 8.5,
                        "usage": 1.5,
                        "is_free_tier": True,
                        "rate_limit": {"requests": 20, "interval": "10s"},
                    }
                },
            )
        raise httpx.ConnectError(
            "connect failed to https://u:pw@proxy.local/?api_key=sk-or-shared-keyB-secret",
            request=request,
        )

    orig_async_client = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr("overseer.openrouter.httpx.AsyncClient", patched_client)

    quotas = await fetch_openrouter_key_quotas()
    assert len(key_calls) == 2  # Deduplicated across cheap & heavy
    assert quotas["status"] == "partial"
    assert quotas["tiers"]["free"]["status"] == "ok"
    assert quotas["tiers"]["free"]["limit_remaining"] == 8.5
    assert quotas["tiers"]["free"]["rate_limit"] == {"requests": 20, "interval": "10s"}
    assert quotas["tiers"]["cheap"]["status"] == "error"
    assert quotas["tiers"]["heavy"]["status"] == "error"

    serialized = json.dumps(quotas)
    assert "sk-or-free-keyA-secret" not in serialized
    assert "sk-or-shared-keyB-secret" not in serialized
    assert "u:pw" not in serialized
    assert "[REDACTED]" in quotas["tiers"]["cheap"]["error"]

    # get_config attaches key_quotas and degrades cleanly if preflight raises
    state = OverseerServerState(client=orig_async_client(transport=httpx.MockTransport(handler)))
    try:
        mcp, _ = create_server(state=state)
        cfg = _extract_tool_payload(await mcp.call_tool("get_config", {}))
        assert cfg["key_quotas"]["status"] == "partial"

        async def boom(timeout: float = 5.0, client: Any = None) -> dict[str, Any]:
            raise RuntimeError("boom sk-or-shared-keyB-secret")

        monkeypatch.setattr("overseer.server.fetch_openrouter_key_quotas", boom)
        cfg_fallback = _extract_tool_payload(await mcp.call_tool("get_config", {}))
        assert cfg_fallback["key_quotas"]["status"] == "unavailable"
        assert "sk-or-shared-keyB-secret" not in json.dumps(cfg_fallback["key_quotas"])
    finally:
        await state.aclose()


@pytest.mark.asyncio
async def test_provider_routing_zdr_and_quantizations_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-routing-test")

    rule = ProviderRoutingRule(zdr=True, quantizations=["fp8", "bf16"])
    assert rule.to_openrouter_dict() == {"zdr": True, "quantizations": ["fp8", "bf16"]}

    with pytest.raises(ValidationError):
        ProviderRoutingRule.model_validate({"zdr": True, "unknown_field": 1})

    save_config_file_sync(OverseerConfigFile(provider_tiers={"cheap": rule}))

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-routing-test",
            model_chain=["deepseek/deepseek-v4.1-flash-1017"],
            start_model_idx=0,
            messages=[{"role": "user", "content": "hi"}],
            limits=RunLimits(max_429_wait_seconds=0),
            tier="cheap",
        )
    assert captured[-1]["provider"] == {"zdr": True, "quantizations": ["fp8", "bf16"]}


# ---------------------------------------------------------------------------
# 11. Bucket B #3 & #4: server.json schema validation & ResourceCSP metadata
# ---------------------------------------------------------------------------


def test_server_json_manifest_validation() -> None:
    import overseer

    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "server.json"
    pyproject_path = root / "pyproject.toml"
    readme_path = root / "README.md"
    assert manifest_path.is_file()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    pkg_version = pyproject["project"]["version"]
    pkg_name = pyproject["project"]["name"]

    assert pkg_name == "overseer-mcp"
    assert overseer.__version__ == pkg_version
    assert data["$schema"] == "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
    assert data["name"] == "io.github.Luther22-source/overseer-mcp"
    assert readme_path.read_text(encoding="utf-8").splitlines()[0] == f"<!-- mcp-name: {data['name']} -->"
    assert data["title"] == "Overseer MCP"
    assert data["version"] == pkg_version
    assert data["repository"] == {
        "url": "https://github.com/Luther22-source/overseer-mcp",
        "source": "github",
    }
    assert isinstance(data["packages"], list) and len(data["packages"]) >= 1

    pkg = data["packages"][0]
    assert pkg["registryType"] == "pypi"
    assert pkg["identifier"] == "overseer-mcp"
    assert pkg["version"] == pkg_version
    assert pkg["runtimeHint"] == "uvx"
    assert "runtimeArguments" not in pkg
    assert "packageArguments" not in pkg
    assert pkg["transport"] == {"type": "stdio"}

    env_by_name = {e["name"]: e for e in pkg["environmentVariables"]}
    for secret_var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
    ):
        assert secret_var in env_by_name
        assert env_by_name[secret_var]["isSecret"] is True
        assert env_by_name[secret_var]["isRequired"] is False
    assert env_by_name["OVERSEER_DATA_DIR"]["isSecret"] is False


@pytest.mark.asyncio
async def test_app_resource_csp_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        resources = await mcp.list_resources()
        app_res = next(r for r in resources if str(r.uri) == APP_URI)
        meta = app_res.model_dump(by_alias=True)["_meta"]
        assert meta["ui"]["csp"] == {"connectDomains": [], "resourceDomains": []}
    finally:
        await state.aclose()


# ---------------------------------------------------------------------------
# 12. Bucket B #5: Interactive MCP Elicitations (rollback & missing API key)
# ---------------------------------------------------------------------------


def _make_fake_ctx(supports_elicitation: bool, elicit_result: Any = None) -> Any:
    class _FakeSession:
        def check_client_capability(self, _cap: Any) -> bool:
            return supports_elicitation

    class _FakeContext:
        def __init__(self) -> None:
            self.request_context = SimpleNamespace(session=_FakeSession())
            self.elicit_calls: list[tuple[str, Any]] = []

        async def elicit(self, message: str, schema: Any) -> Any:
            self.elicit_calls.append((message, schema))
            if isinstance(elicit_result, Exception):
                raise elicit_result
            return elicit_result

    return _FakeContext()


@pytest.mark.asyncio
async def test_rollback_elicitation_accept_decline_cancel_and_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)

    async def _seed_run(rid: str, filename: str) -> Path:
        created = proj / "src" / filename
        raw = b"val = 1\n"
        created.write_bytes(raw)
        p_data = get_project_data_dir(proj)
        rdir = p_data / "runs" / rid
        rdir.mkdir(parents=True, exist_ok=True)
        m = RunManifest(
            version="0.1.0",
            run_id=rid,
            server_pid=os.getpid(),
            created_at_iso="2026-01-01T00:00:10Z",
            project_root=str(proj),
            status="completed",
            limits=RunLimits(),
            patch_dir=str(rdir),
            workers={
                "w1": WorkerManifestState(
                    worker_id="w1",
                    folder="src",
                    goal="create",
                    requested_model="m:free",
                    status="completed",
                    created_files={
                        f"src/{filename}": CreatedFileRecord(
                            worker_last_hash=sha256_bytes(raw),
                            line_count=1,
                        )
                    },
                )
            },
        )
        await save_manifest_atomic(manifest_path_for(p_data, rid), m, asyncio.Lock())
        return created

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        rollback_fn = mcp._tool_manager.get_tool("rollback").fn

        # 1. Decline elicitation -> does NOT revert
        f1 = await _seed_run("run_20260101_000001_aaaaaa", "f1.py")
        ctx_decline = _make_fake_ctx(True, DeclinedElicitation())
        res_decline = await rollback_fn(
            project_root=str(proj),
            run_id="run_20260101_000001_aaaaaa",
            confirm=False,
            ctx=ctx_decline,
        )
        assert res_decline["status"] == "declined"
        assert res_decline["reverted"] is False
        assert f1.exists()

        # 2. Accept with confirm=False -> does NOT revert
        ctx_acc_false = _make_fake_ctx(
            True, AcceptedElicitation(data=_RollbackConfirmSchema(confirm=False))
        )
        res_acc_false = await rollback_fn(
            project_root=str(proj),
            run_id="run_20260101_000001_aaaaaa",
            confirm=False,
            ctx=ctx_acc_false,
        )
        assert res_acc_false["status"] == "declined"
        assert f1.exists()

        # 3. Cancel elicitation -> does NOT revert
        ctx_cancel = _make_fake_ctx(True, CancelledElicitation())
        res_cancel = await rollback_fn(
            project_root=str(proj),
            run_id="run_20260101_000001_aaaaaa",
            confirm=False,
            ctx=ctx_cancel,
        )
        assert res_cancel["status"] == "cancelled"
        assert res_cancel["reverted"] is False
        assert f1.exists()

        # 4. Accept with confirm=True -> reverts
        ctx_accept = _make_fake_ctx(
            True, AcceptedElicitation(data=_RollbackConfirmSchema(confirm=True))
        )
        res_accept = await rollback_fn(
            project_root=str(proj),
            run_id="run_20260101_000001_aaaaaa",
            confirm=False,
            ctx=ctx_accept,
        )
        assert res_accept["deleted_files"] == ["src/f1.py"]
        assert not f1.exists()

        # 5. Unsupported client fallback -> proceeds without blocking (non-breaking)
        f2 = await _seed_run("run_20260101_000002_bbbbbb", "f2.py")
        ctx_unsup = _make_fake_ctx(False)
        res_unsup = await rollback_fn(
            project_root=str(proj),
            run_id="run_20260101_000002_bbbbbb",
            confirm=False,
            ctx=ctx_unsup,
        )
        assert res_unsup["deleted_files"] == ["src/f2.py"]
        assert not f2.exists()
        assert ctx_unsup.elicit_calls == []
    finally:
        await state.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_elicitation_accept_decline_cancel_and_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "a/free-model:free",
                            "name": "Free Model",
                            "context_length": 64000,
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        list_models_fn = mcp._tool_manager.get_tool("list_models").fn

        # 1. Unsupported client -> ToolError without calling elicit
        ctx_unsup = _make_fake_ctx(False)
        with pytest.raises(ToolError, match="OpenRouter API key"):
            await list_models_fn(free_only=True, ctx=ctx_unsup)
        assert ctx_unsup.elicit_calls == []

        # 2. Decline -> ToolError
        ctx_decline = _make_fake_ctx(True, DeclinedElicitation())
        with pytest.raises(ToolError, match="OpenRouter API key"):
            await list_models_fn(free_only=True, ctx=ctx_decline)
        assert len(ctx_decline.elicit_calls) == 1

        # 3. Cancel -> ToolError
        ctx_cancel = _make_fake_ctx(True, CancelledElicitation())
        with pytest.raises(ToolError, match="OpenRouter API key"):
            await list_models_fn(free_only=True, ctx=ctx_cancel)
        assert len(ctx_cancel.elicit_calls) == 1

        # 4. Accept -> stores in os.environ and retries cleanly
        ctx_accept = _make_fake_ctx(
            True,
            AcceptedElicitation(
                data=_MissingApiKeySchema(
                    api_key="sk-or-elicited-session-key",
                    env_var="OPENROUTER_API_KEY",
                )
            ),
        )
        res = await list_models_fn(free_only=True, ctx=ctx_accept)
        assert res["count"] == 1
        assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-elicited-session-key"
    finally:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        await state.aclose()


# ---------------------------------------------------------------------------
# 13. Ponytail-Audit Regression Suite (#1-#5, #7-#13)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_1_worker_semaphore_resizes_between_runs_and_locks_while_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-sem-test")
    monkeypatch.setenv("OVERSEER_MAX_CONCURRENT_WORKERS", "2")
    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "a/free:free")
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "")

    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)

    registry = RunRegistry()
    sem_initial = registry.get_worker_semaphore()
    assert registry._semaphore_limit == 2

    # While a run is active, changing max_concurrent_workers must NOT resize the semaphore
    registry.register_run(proj, "run_20260101_000001_aaaaaa")
    monkeypatch.setenv("OVERSEER_MAX_CONCURRENT_WORKERS", "5")
    assert registry.get_worker_semaphore() is sem_initial
    assert registry._semaphore_limit == 2

    # Once the run finishes, calling prepare_and_launch_run resizes BEFORE registering the next run
    registry.mark_run_finished(proj, "run_20260101_000001_aaaaaa")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "a/free:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_sem",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "ok", "notes": []}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(proj),
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="Finish")],
        )
        ack = await prepare_and_launch_run(req, registry, client, ModelCatalog(), proj)
        assert registry._semaphore_limit == 5
        assert registry.worker_semaphore is not sem_initial
        for _ in range(100):
            if not registry.is_run_active(proj, ack.run_id):
                break
            await asyncio.sleep(0.01)


def test_audit_2_monitor_snapshot_includes_token_and_cached_metrics(tmp_path: Path) -> None:
    from overseer.state import _build_monitor_snapshot_sync

    proj = tmp_path / "proj"
    proj.mkdir()
    manifest = RunManifest(
        version="0.1.0",
        run_id="run_20260101_000002_bbbbbb",
        server_pid=os.getpid(),
        created_at_iso="2026-01-01T00:00:02Z",
        project_root=str(proj),
        status="completed",
        limits=RunLimits(),
        cached_tokens=320,
        patch_dir=str(tmp_path / "run"),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="src",
                goal="g",
                requested_model="m:free",
                status="completed",
                prompt_tokens=1000,
                completion_tokens=250,
                cached_tokens=320,
            )
        },
    )
    snap = _build_monitor_snapshot_sync(proj, manifest)
    assert snap["cached_tokens"] == 320
    assert len(snap["workers"]) == 1
    w = snap["workers"][0]
    assert w["prompt_tokens"] == 1000
    assert w["completion_tokens"] == 250
    assert w["cached_tokens"] == 320


@pytest.mark.asyncio
async def test_audit_3_patch_written_on_completion_and_on_demand_respecting_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.state import read_worker_patch_sync

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("OPENROUTER_FREE_API_KEY", "sk-or-patch-test")
    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "a/free:free")
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "")

    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "a/free:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        }
                    ]
                },
            )
        turn += 1
        if turn == 1:
            tc = {
                "id": "t_write",
                "type": "function",
                "function": {
                    "name": "write_new_file",
                    "arguments": json.dumps({"path": "src/created.py", "content": "VAL = 42\n"}),
                },
            }
        else:
            tc = {
                "id": "t_fin",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"summary": "created", "notes": []}'},
            }
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]}}],
            },
        )

    registry = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(proj),
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="create file")],
        )
        ack = await prepare_and_launch_run(req, registry, client, ModelCatalog(), proj)
        for _ in range(100):
            if not registry.is_run_active(proj, ack.run_id):
                break
            await asyncio.sleep(0.01)

    p_data = get_project_data_dir(proj)
    patch_file = p_data / "runs" / ack.run_id / "w1.patch"
    # 1. Written automatically on completion without calling get_report
    assert patch_file.is_file()
    assert "VAL = 42" in patch_file.read_text(encoding="utf-8")

    # 2. If missing on disk, read_worker_patch_sync generates it on demand
    patch_file.unlink()
    assert not patch_file.exists()
    on_demand = read_worker_patch_sync(proj, ack.run_id, "w1")
    assert "VAL = 42" in on_demand
    assert patch_file.is_file()

    # 3. After rollback, if patch file is absent, read_worker_patch_sync must NOT regenerate it
    await rollback_run(proj, ack.run_id, registry)
    patch_file.unlink()
    after_rb = read_worker_patch_sync(proj, ack.run_id, "w1")
    assert after_rb == ""
    assert not patch_file.exists()


def test_audit_4_jsts_regex_literals_and_division_vs_syntax_errors() -> None:
    from overseer.guardrails import _check_jsts_source

    valid_samples = [
        "const x = fn(s.replace(/'/g, ''));\n",
        "const ok = (/https?:\\/\\//.test(u));\n",
        "const x = fn(/[({]/g);\n",
        "const ratio = (a + b) / c;\nconst half = arr[0] / 2;\nconst inc = x++ / 2;\n",
    ]
    for src in valid_samples:
        assert _check_jsts_source("sample.ts", src) == [], f"Unexpected error for valid JS/TS: {src!r}"

    bad_samples = [
        "const broken = fn(a + b;\n",
        "export function f() { return [1, 2; }\n",
        "/* unclosed block comment\nconst x = 1;\n",
    ]
    for src in bad_samples:
        assert _check_jsts_source("bad.ts", src), f"Expected syntax error for: {src!r}"


def test_audit_5_yaml_fallback_allows_unquoted_apostrophe_and_catches_unclosed_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from overseer.guardrails import _check_yaml_source

    real_import = builtins.__import__

    def block_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml" or name.startswith("yaml."):
            raise ImportError("PyYAML blocked for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_yaml)

    assert _check_yaml_source("ok.yaml", "title: Don't break\nauthor: O'Connor\n") == []
    errs_single = _check_yaml_source("bad1.yaml", "title: 'unterminated\n")
    assert any("Unclosed quoted string" in e for e in errs_single)
    errs_double = _check_yaml_source("bad2.yaml", 'title: "unterminated\n')
    assert any("Unclosed quoted string" in e for e in errs_double)


def test_audit_7_shared_is_pid_alive_from_guardrails() -> None:
    from overseer import config as config_mod
    from overseer import guardrails as guardrails_mod
    from overseer import state as state_mod

    assert not hasattr(config_mod, "_is_pid_alive")
    assert config_mod.is_pid_alive is guardrails_mod.is_pid_alive
    assert state_mod.is_pid_alive is guardrails_mod.is_pid_alive
    assert guardrails_mod.is_pid_alive(os.getpid()) is True
    assert guardrails_mod.is_pid_alive(0) is False
    assert guardrails_mod.is_pid_alive(-42) is False


def test_audit_8_atomic_write_mode_0600_and_dotenv_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stat as stat_mod

    from overseer.guardrails import atomic_write_bytes_sync

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))

    recorded_modes: list[int | None] = []
    orig_atomic = cli.atomic_write_bytes_sync

    def spy_atomic(
        target_path: Path,
        data: bytes,
        retries: int = 3,
        backoff_sec: float = 0.05,
        mode: int | None = None,
    ) -> None:
        recorded_modes.append(mode)
        orig_atomic(target_path, data, retries=retries, backoff_sec=backoff_sec, mode=mode)

    monkeypatch.setattr(cli, "atomic_write_bytes_sync", spy_atomic)
    env_path = cli._write_dotenv_keys({"OPENROUTER_API_KEY": "sk-or-mode-test"})
    assert env_path.is_file()
    assert recorded_modes == [0o600]

    # Verify atomic_write_bytes_sync forwards mode=0o600 to _atomic_write_sync (and sets 0o600 on POSIX)
    target = tmp_path / "secret.bin"
    seen_inner: list[int | None] = []
    from overseer import guardrails as guardrails_mod

    orig_inner = guardrails_mod._atomic_write_sync

    def spy_inner(target_path: Path, data: bytes, mode: int | None = None) -> None:
        seen_inner.append(mode)
        orig_inner(target_path, data, mode=mode)

    monkeypatch.setattr(guardrails_mod, "_atomic_write_sync", spy_inner)
    atomic_write_bytes_sync(target, b"secret", mode=0o600)
    assert seen_inner == [0o600]
    if os.name != "nt":
        assert stat_mod.S_IMODE(target.stat().st_mode) == 0o600


def test_audit_9_cli_doctor_aliases_and_dotenv_app_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OVERSEER_APP_HTML", raising=False)

    custom_html = tmp_path / "custom_app.html"
    custom_html.write_text("<html>custom</html>", encoding="utf-8")
    (data_dir / ".env").write_text(
        f"OPENROUTER_API_KEY=sk-or-doc-alias-test\nOVERSEER_APP_HTML={custom_html}\n",
        encoding="utf-8",
    )
    assert cli._resolve_app_html_path() == custom_html

    # Configure a legacy alias slug and verify run_doctor resolves it via MODEL_SLUG_ALIASES
    from overseer.config import MODEL_SLUG_ALIASES

    legacy_slug, canonical_slug = next(iter(MODEL_SLUG_ALIASES.items()))
    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", legacy_slug)
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "")
    monkeypatch.setenv("OVERSEER_CHEAP_DEFAULT_MODEL", canonical_slug)
    monkeypatch.setenv("OVERSEER_CHEAP_FALLBACK_MODELS", "")
    monkeypatch.setenv("OVERSEER_HEAVY_DEFAULT_MODEL", "")
    monkeypatch.setenv("OVERSEER_HEAVY_FALLBACK_MODELS", "")

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": canonical_slug}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(cli.httpx, "get", fake_get)
    monkeypatch.setattr(
        cli,
        "fetch_openrouter_key_quotas_sync",
        lambda timeout=5.0: {"status": "ok", "tiers": {}},
    )

    rc = cli.run_doctor([])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[PASS] Configured tier models verified on OpenRouter" in out
    assert str(custom_html) in out


def test_audit_10_normcase_path_macos_in_project_data_dir_and_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import guardrails as guardrails_mod
    from overseer.state import _prune_empty_dirs

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setattr(guardrails_mod.sys, "platform", "darwin")

    upper = tmp_path / "MyRepo"
    lower = tmp_path / "myrepo"
    assert guardrails_mod._normcase_path(upper) == guardrails_mod._normcase_path(lower)
    assert get_project_data_dir(upper) == get_project_data_dir(lower)

    # _prune_empty_dirs stops at stop_root even when path casing differs on darwin
    stop_root = tmp_path / "CaseRoot"
    leaf = stop_root / "a" / "b"
    leaf.mkdir(parents=True)
    _prune_empty_dirs(leaf, tmp_path / "caseroot")
    assert stop_root.is_dir()
    assert not (stop_root / "a").exists()


@pytest.mark.asyncio
async def test_audit_11_rollback_resolves_latest_before_elicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.state import write_latest_run_pointer

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    created = proj / "src" / "f.py"
    raw = b"v = 1\n"
    created.write_bytes(raw)

    rid = "run_20260101_000011_111111"
    p_data = get_project_data_dir(proj)
    rdir = p_data / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        version="0.1.0",
        run_id=rid,
        server_pid=os.getpid(),
        created_at_iso="2026-01-01T00:00:11Z",
        project_root=str(proj),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(rdir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="src",
                goal="g",
                requested_model="m:free",
                status="completed",
                created_files={"src/f.py": CreatedFileRecord(worker_last_hash=sha256_bytes(raw), line_count=1)},
            )
        },
    )
    await save_manifest_atomic(manifest_path_for(p_data, rid), manifest, asyncio.Lock())
    await write_latest_run_pointer(p_data, rid)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        rollback_fn = mcp._tool_manager.get_tool("rollback").fn
        ctx_decline = _make_fake_ctx(True, DeclinedElicitation())
        res = await rollback_fn(project_root=str(proj), run_id="latest", confirm=False, ctx=ctx_decline)
        assert res["status"] == "declined"
        assert res["run_id"] == rid
        assert len(ctx_decline.elicit_calls) == 1
        assert rid in ctx_decline.elicit_calls[0][0]
        assert "'latest'" not in ctx_decline.elicit_calls[0][0]
    finally:
        await state.aclose()


@pytest.mark.asyncio
async def test_audit_12_prompt_cache_400_cached_so_second_step_skips_cache_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import openrouter as openrouter_mod

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    openrouter_mod._prompt_cache_unsupported.clear()

    seen_bodies: list[dict[str, Any]] = []
    model_id = "anthropic/claude-sonnet-4.6"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_bodies.append(body)
        if "cache_control" in body["tools"][-1]:
            return httpx.Response(400, json={"error": {"message": "cache_control unsupported"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # First call: sends cache_control -> 400 -> retries without cache_control (2 requests)
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-test",
            model_chain=[model_id],
            start_model_idx=0,
            messages=msgs,
            limits=RunLimits(max_429_wait_seconds=0),
            tier="heavy",
        )
        assert len(seen_bodies) == 2
        assert "cache_control" in seen_bodies[0]["tools"][-1]
        assert "cache_control" not in seen_bodies[1]["tools"][-1]

        # Second call: remembers (base_url, model_id) and skips cache_control immediately (1 request)
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-or-test",
            model_chain=[model_id],
            start_model_idx=0,
            messages=msgs,
            limits=RunLimits(max_429_wait_seconds=0),
            tier="heavy",
        )
        assert len(seen_bodies) == 3
        assert "cache_control" not in seen_bodies[2]["tools"][-1]
    openrouter_mod._prompt_cache_unsupported.clear()


@pytest.mark.asyncio
async def test_audit_13_key_quotas_reuses_passed_client_and_caches_within_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.openrouter import clear_key_quota_cache

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-cache-ttl-key")
    clear_key_quota_cache()

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"data": {"limit": 5.0, "limit_remaining": 4.25, "usage": 0.75, "is_free_tier": False}},
        )

    # Ensure no new httpx.AsyncClient is instantiated when a client is passed
    def forbid_new_client(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Should reuse passed client instead of creating a new AsyncClient")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared_client:
        monkeypatch.setattr("overseer.openrouter.httpx.AsyncClient", forbid_new_client)
        q1 = await fetch_openrouter_key_quotas(client=shared_client)
        assert q1["status"] == "ok"
        assert q1["tiers"]["free"]["limit_remaining"] == 4.25
        assert calls == 1

        # Second call within TTL serves cached result without another HTTP hit
        q2 = await fetch_openrouter_key_quotas(client=shared_client)
        assert q2["status"] == "ok"
        assert q2["tiers"]["free"]["limit_remaining"] == 4.25
        assert calls == 1

        # Clearing cache triggers a fresh fetch
        clear_key_quota_cache()
        q3 = await fetch_openrouter_key_quotas(client=shared_client)
        assert q3["status"] == "ok"
        assert calls == 2
    clear_key_quota_cache()
