"""Stage 2 async unit tests for worker.py using httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from overseer.guardrails import get_project_data_dir, sha256_bytes
from overseer.models import (
    ReplaceBlockArgs,
    RunLimits,
    RunManifest,
    StartRunRequest,
    StaticCheckResult,
    ViewFileArgs,
    WorkerManifestState,
    WorkerTaskInput,
    WriteNewFileArgs,
)
from overseer.state import (
    RunRegistry,
    cancel_flag_path_for,
    load_manifest_sync,
    manifest_path_for,
    save_manifest_atomic,
    utc_now_iso,
)
from overseer.worker import (
    MAX_POST_WRITE_READONLY_TURNS,
    MAX_REDUNDANT_READ_TURNS,
    POST_WRITE_NUDGE_TURN,
    WORKER_TOOL_SCHEMAS,
    ModelCatalog,
    WorkerToolContext,
    _build_worker_system_prompt,
    _canonical_tool_sig,
    _pending_folder_targets,
    _view_file_sync,
    build_model_chain,
    execute_worker_tool,
    load_external_dotenv,
    parse_retry_after_seconds,
    prepare_and_launch_run,
    prune_message_history,
    select_api_key_for_model,
)


@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "overseer_appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "model/fallback:free")

    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "frontend").mkdir(parents=True)
    (repo / "backend" / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    (repo / "frontend" / "index.ts").write_text("export const v = 1;\n", encoding="utf-8")
    return repo


def _models_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "model/primary:free",
                    "supported_parameters": ["tools", "temperature"],
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "model/fallback:free",
                    "supported_parameters": ["tools"],
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "model/paid-coder",
                    "supported_parameters": ["tools"],
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                },
                {
                    "id": "model/no-tools",
                    "supported_parameters": ["temperature"],
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )


def _tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


async def _wait_for_run(repo: Path, run_id: str, registry: RunRegistry, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while registry.is_run_active(repo, run_id):
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("Timed out waiting for background workers")
        await asyncio.sleep(0.01)
    m_path = manifest_path_for(get_project_data_dir(repo), run_id)
    return load_manifest_sync(m_path)


def test_prune_message_history_prioritizes_active_reads() -> None:
    messages: list[dict] = [{"role": "system", "content": "sys"}]
    # Build 9 tool turns:
    # 1: view_file(a.py 1-10) -> superseded by #4
    # 2: list_dir(.) -> exploratory
    # 3: view_file(b.py 1-10) -> ACTIVE non-superseded read!
    # 4: view_file(a.py 1-20) -> recent active
    # 5: search_dir("foo") -> exploratory
    # 6..9: recent tool turns
    seq = [
        ("c1", "view_file", {"path": "a.py", "start_line": 1, "end_line": 10}, "old a.py"),
        ("c2", "list_dir", {"path": "."}, "dir listing"),
        ("c3", "view_file", {"path": "b.py", "start_line": 1, "end_line": 10}, "ACTIVE b.py"),
        ("c4", "view_file", {"path": "a.py", "start_line": 1, "end_line": 20}, "new a.py"),
        ("c5", "search_dir", {"query": "foo", "path": "."}, "search hits"),
        ("c6", "replace_block", {"path": "a.py", "target": "1", "replacement": "2"}, "edited a"),
        ("c7", "write_new_file", {"path": "c.py", "content": "x=1"}, "created c"),
        ("c8", "find_file", {"pattern": "*.py", "path": "."}, "found py"),
        ("c9", "finish", {"summary": "done", "notes": []}, "finished"),
    ]
    for cid, name, args, resp in seq:
        messages.append(_tool_call_msg(cid, name, args))
        messages.append({"role": "tool", "tool_call_id": cid, "content": resp})

    pruned = prune_message_history(messages, keep_recent_tools=6)
    tool_contents = [m["content"] for m in pruned if m["role"] == "tool"]
    assert len(tool_contents) == 9
    # Exactly 3 oldest exploratory/superseded outputs pruned: c1 (superseded view_file), c2 (list_dir), c5 (search_dir)
    assert tool_contents[0].startswith("[pruned view_file a.py lines 1-10]")
    assert tool_contents[1].startswith("[pruned list_dir .]")
    assert tool_contents[2] == "ACTIVE b.py"
    assert tool_contents[4].startswith("[pruned search_dir .]")
    assert tool_contents[-1] == "finished"


def test_parse_retry_after_numeric_and_http_date() -> None:
    assert parse_retry_after_seconds("12.5") == 12.5
    assert parse_retry_after_seconds(None) is None
    assert parse_retry_after_seconds("invalid-date") is None
    future = datetime.now(UTC) + timedelta(seconds=20)
    http_date = email_fmt = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    parsed = parse_retry_after_seconds(email_fmt)
    assert parsed is not None and 10.0 <= parsed <= 25.0


async def test_model_catalog_caching_and_tool_validation(worker_env: Path) -> None:
    hits = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return _models_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        catalog = ModelCatalog(ttl_sec=3600.0)
        await catalog.ensure_loaded(client, "sk-test")
        await catalog.ensure_loaded(client, "sk-test")
        assert hits == 1
        catalog.validate_tool_support(["model/primary:free", "model/fallback:free"])
        with pytest.raises(ValueError, match="tools"):
            catalog.validate_tool_support(["model/no-tools"])
        with pytest.raises(ValueError, match="not found"):
            catalog.validate_tool_support(["model/unknown"])
        assert catalog.estimate_cost("model/paid-coder", 10, 5) == pytest.approx(0.02)


async def test_full_worker_loop_tools_syntax_repair_and_notes(worker_env: Path) -> None:
    repo = worker_env
    # Add binary and extra files for tool boundary tests
    (repo / "backend" / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
    (repo / "backend" / "huge_line.txt").write_text("A" * 70_000, encoding="utf-8")

    captured_system_prompts: list[str] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()

        body = json.loads(request.content.decode("utf-8"))
        assert body["usage"] == {"include": True}
        captured_system_prompts.append(body["messages"][0]["content"])
        turn += 1

        if turn == 1:
            # Batch multiple tool calls: list_dir, find_file, search_dir, view_file (huge + app)
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {"id": "t1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}},
                                    {"id": "t2", "type": "function", "function": {"name": "find_file", "arguments": '{"pattern": "*.py", "path": "."}'}},
                                    {"id": "t3", "type": "function", "function": {"name": "search_dir", "arguments": '{"query": "greet", "path": "."}'}},
                                    {"id": "t4", "type": "function", "function": {"name": "view_file", "arguments": '{"path": "backend/huge_line.txt"}'}},
                                    {"id": "t5", "type": "function", "function": {"name": "view_file", "arguments": '{"path": "backend/app.py"}'}},
                                ],
                            }
                        }
                    ],
                },
            )
        if turn == 2:
            # Write new nested file with syntax error + edit existing app.py
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "t6",
                                        "type": "function",
                                        "function": {
                                            "name": "write_new_file",
                                            "arguments": json.dumps({"path": "backend/pkg/mod.py", "content": "def bad(:\n    return 1\n"}),
                                        },
                                    },
                                    {
                                        "id": "t7",
                                        "type": "function",
                                        "function": {
                                            "name": "replace_block",
                                            "arguments": json.dumps({"path": "backend/app.py", "target": "return 'hi'", "replacement": "return 'hello'"}),
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                },
            )
        if turn == 3:
            # Call finish() before fixing bad syntax -> pre-check rejects, keeping worker alive
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "t8",
                                        "type": "function",
                                        "function": {"name": "finish", "arguments": json.dumps({"summary": "Done", "notes": []})},
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        if turn == 4:
            # Repair the created file with replace_block (must remain in created_files!)
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "t9",
                                        "type": "function",
                                        "function": {
                                            "name": "replace_block",
                                            "arguments": json.dumps({"path": "backend/pkg/mod.py", "target": "def bad(:", "replacement": "def good():"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        # Turn 5: finish succeeds with plain-text note
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "t10",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": json.dumps(
                                            {
                                                "summary": "Updated app.py and added backend/pkg/mod.py",
                                                "notes": ["Update frontend/index.ts to use new hello response"],
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            shared_contract="Backend greet returns 'hello'",
            tasks=[
                WorkerTaskInput(
                    worker_id="W1-Backend",
                    folder="backend",
                    goal="Update greeting and add mod.py",
                    target_files=["backend/app.py", "backend/pkg/mod.py"],
                )
            ],
        )
        start_resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        assert start_resp.warning and ":free" in start_resp.warning
        manifest = await _wait_for_run(repo, start_resp.run_id, registry)

    assert "backend/app.py (2 lines)" in captured_system_prompts[0]
    assert "backend/pkg/mod.py (new file - does not exist yet)" in captured_system_prompts[0]
    assert manifest.status == "completed"
    w = manifest.workers["W1-Backend"]
    assert w.steps_used == 5
    assert w.static_check.status == "passed"
    assert "backend/app.py" in w.modified_files
    assert "backend/pkg/mod.py" in w.created_files
    assert "backend/pkg/mod.py" not in w.modified_files
    mod_bytes = (repo / "backend" / "pkg" / "mod.py").read_bytes()
    assert w.created_files["backend/pkg/mod.py"].worker_last_hash == sha256_bytes(mod_bytes)
    assert w.notes == ["Update frontend/index.ts to use new hello response"]


async def test_429_failover_without_step_penalty_and_breakers(worker_env: Path) -> None:
    repo = worker_env
    # Test 1: W1 gets 429 (Retry-After > max_429_wait_seconds), fails over immediately with 0 step penalty
    # Test 2: W2 hits 3 consecutive tool errors -> escalated
    calls_per_worker: dict[str, int] = {"w1": 0, "w2": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        sys_msg = body["messages"][0]["content"]
        if "folder jail: backend" in sys_msg:
            calls_per_worker["w1"] += 1
            if body["model"] == "model/primary:free":
                return httpx.Response(429, headers={"Retry-After": "120"}, text="rate limited")
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
                                        "id": "w1_fin",
                                        "type": "function",
                                        "function": {"name": "finish", "arguments": '{"summary": "Failover ok", "notes": []}'},
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        # W2 on frontend hits 3 bad replace_block calls -> escalated
        calls_per_worker["w2"] += 1
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
                                    "id": f"bad_{calls_per_worker['w2']}",
                                    "type": "function",
                                    "function": {
                                        "name": "replace_block",
                                        "arguments": '{"path": "frontend/index.ts", "target": "x", "replacement": "y"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_429_wait_seconds=10,
            tasks=[
                WorkerTaskInput(worker_id="w1", folder="backend", goal="be"),
                WorkerTaskInput(worker_id="w2", folder="frontend", goal="fe"),
            ],
        )
        start_resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, start_resp.run_id, registry)

    assert manifest.status == "completed_with_escalations"
    assert manifest.workers["w1"].status == "completed"
    assert manifest.workers["w1"].steps_used == 1
    assert manifest.workers["w1"].models_used == ["model/fallback:free"]
    assert manifest.workers["w2"].status == "escalated"
    assert manifest.workers["w2"].stop_reason == "consecutive_tool_errors"


async def test_budget_halt_non_retryable_and_empty_turns(worker_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = worker_env
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    # 1. Non-retryable 402 -> failed
    def handler_402(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(402, text="Payment Required")

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_402)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="fail fast", model="model/paid-coder")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        m_fail = await _wait_for_run(repo, resp.run_id, registry)
    assert m_fail.status == "failed"
    assert m_fail.workers["w1"].status == "failed"
    assert resp.warning is None

    # 2. Empty plain-text turns + max_run_usd cutoff
    def handler_empty(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.01},
                "choices": [{"message": {"role": "assistant", "content": "Thinking aloud..."}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_empty)) as client:
        req2 = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="think")],
        )
        resp2 = await prepare_and_launch_run(req2, registry, client, catalog, repo)
        m_empty = await _wait_for_run(repo, resp2.run_id, registry)
    assert m_empty.workers["w1"].status == "escalated"
    assert m_empty.workers["w1"].stop_reason == "no_tool_calls"

    # 3. Budget halt + max_steps
    def handler_budget(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.30},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_budget)) as client:
        req3 = StartRunRequest(
            project_root=str(repo),
            max_run_usd=0.25,
            tasks=[
                WorkerTaskInput(worker_id="w1", folder="backend", goal="expensive"),
                WorkerTaskInput(worker_id="w2", folder="frontend", goal="expensive2"),
            ],
        )
        resp3 = await prepare_and_launch_run(req3, registry, client, catalog, repo)
        m_budget = await _wait_for_run(repo, resp3.run_id, registry)
    assert m_budget.status == "halted_budget"
    assert any(w.status == "halted_budget" for w in m_budget.workers.values())


@pytest.mark.asyncio
async def test_execute_worker_tool_error_and_edge_branches(worker_env: Path) -> None:
    repo = worker_env
    (repo / "backend" / "bin.dat").write_bytes(b"\x00\x01\x02")
    (repo / "backend" / "nonutf8.txt").write_bytes(b"\xff\xfe\xfa")
    (repo / "backend" / "huge_line.txt").write_text("x = '" + ("a" * 70_000) + "'\n", encoding="utf-8")
    (repo / "backend" / "many_lines.txt").write_text("\n".join("z" * 500 for _ in range(180)) + "\n", encoding="utf-8")
    (repo / "backend" / "unviewed.py").write_text("u = 1\n", encoding="utf-8")

    proj_data = get_project_data_dir(repo)
    run_id = "run_20261115_000010_cccccc"
    run_dir = proj_data / "runs" / run_id
    backups_dir = run_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="edges",
                requested_model="m:free",
            )
        },
    )
    lock = asyncio.Lock()
    await save_manifest_atomic(m_path, manifest, lock)
    ctx = WorkerToolContext(
        project_root=repo,
        folder_abs=repo / "backend",
        backups_dir=backups_dir,
        run_dir=run_dir,
        manifest_path=m_path,
        manifest=manifest,
        worker=manifest.workers["w1"],
        run_lock=lock,
    )

    msg, is_err = await execute_worker_tool("view_file", "{bad json", ctx)
    assert is_err and "JSON" in msg

    msg, is_err = await execute_worker_tool("no_such_tool", "{}", ctx)
    assert is_err and "Unknown tool" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend"}, ctx)
    assert is_err and "not a file" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/bin.dat"}, ctx)
    assert is_err and "Binary" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/nonutf8.txt"}, ctx)
    assert is_err and "Non-UTF-8" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/app.py", "start_line": 999, "end_line": 1005}, ctx)
    assert not is_err and "past EOF" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/huge_line.txt", "start_line": 1, "end_line": 5}, ctx)
    assert not is_err and "64 KB" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/many_lines.txt", "start_line": 1, "end_line": 200}, ctx)
    assert not is_err and "Output capped at" in msg

    for tool_name, payload in (
        ("list_dir", {"path": "backend/app.py"}),
        ("search_dir", {"path": "backend/app.py", "query": "greet"}),
        ("find_file", {"path": "backend/app.py", "pattern": "*.py"}),
    ):
        msg, is_err = await execute_worker_tool(tool_name, payload, ctx)
        assert is_err and "not a directory" in msg

    msg, is_err = await execute_worker_tool("search_dir", {"path": "backend", "query": "greet"}, ctx)
    assert not is_err and "backend/app.py" in msg

    msg, is_err = await execute_worker_tool("search_dir", {"path": "backend", "query": "definitely_absent_token"}, ctx)
    assert not is_err

    msg, is_err = await execute_worker_tool(
        "replace_block",
        {"path": "backend/unviewed.py", "target": "u = 1", "replacement": "u = 2"},
        ctx,
    )
    assert is_err

    msg, is_err = await execute_worker_tool(
        "write_new_file",
        {"path": "backend/app.py", "content": "x = 1\n"},
        ctx,
    )
    assert is_err and "refuses to overwrite" in msg

    msg, is_err = await execute_worker_tool("view_file", {"path": "backend/app.py"}, ctx)
    assert not is_err
    (repo / "backend" / "app.py").write_text("def greet():\n    return 'external'\n", encoding="utf-8")
    msg, is_err = await execute_worker_tool(
        "replace_block",
        {"path": "backend/app.py", "target": "return 'external'", "replacement": "return 'edited'"},
        ctx,
    )
    assert is_err and "hash mismatch" in msg


@pytest.mark.asyncio
async def test_worker_transport_failover_503_max_steps_and_consecutive_errors(worker_env: Path) -> None:
    repo = worker_env
    registry = RunRegistry()
    catalog = ModelCatalog()

    # 1. ConnectError on primary -> fails over to fallback -> succeeds with shared_contract & missing target_file
    def handler_transport_failover(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        if body["model"] == "model/primary:free":
            raise httpx.ConnectError("transient network drop", request=request)
        assert body["model"] == "model/fallback:free"
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
                                {"id": "f1", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Recovered", "notes": []}'}}
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_transport_failover)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            shared_contract="Keep signatures aligned.",
            max_429_wait_seconds=0,
            tasks=[
                WorkerTaskInput(
                    worker_id="w1",
                    folder="backend",
                    goal="finish via fallback",
                    target_files=["backend/app.py", "backend/future_file.py"],
                )
            ],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        m1 = await _wait_for_run(repo, resp.run_id, registry)
    assert m1.status == "completed"
    assert m1.workers["w1"].models_used == ["model/fallback:free"]

    # 2. HTTP 503 exhausted across all models -> failed (all_models_exhausted)
    def handler_503(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(503, text="Service Unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_503)) as client:
        req2 = StartRunRequest(
            project_root=str(repo),
            max_429_wait_seconds=0,
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="hit 503")],
        )
        resp2 = await prepare_and_launch_run(req2, registry, client, catalog, repo)
        m2 = await _wait_for_run(repo, resp2.run_id, registry)
    assert m2.status == "failed"
    assert m2.workers["w1"].stop_reason == "all_models_exhausted"

    # 3. max_steps_per_worker=1 without finish -> escalated (max_steps_reached)
    def handler_list_only(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "l1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_list_only)) as client:
        req3 = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=1,
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="run out of steps")],
        )
        resp3 = await prepare_and_launch_run(req3, registry, client, catalog, repo)
        m3 = await _wait_for_run(repo, resp3.run_id, registry)
    assert m3.workers["w1"].status == "escalated"
    assert m3.workers["w1"].stop_reason == "max_steps_reached"

    # 4. 3 consecutive tool errors -> escalated (consecutive_tool_errors)
    def handler_bad_tools(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "e1", "type": "function", "function": {"name": "bad_tool", "arguments": "{}"}}
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_bad_tools)) as client:
        req4 = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="trigger error breaker")],
        )
        resp4 = await prepare_and_launch_run(req4, registry, client, catalog, repo)
        m4 = await _wait_for_run(repo, resp4.run_id, registry)
    assert m4.workers["w1"].status == "escalated"
    assert m4.workers["w1"].stop_reason == "consecutive_tool_errors"


@pytest.mark.asyncio
async def test_external_dotenv_two_key_routing_and_model_chains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "ext_appdata"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OVERSEER_DEFAULT_MODEL",
        "OVERSEER_FALLBACK_MODELS",
        "OVERSEER_FREE_DEFAULT_MODEL",
        "OVERSEER_FREE_FALLBACK_MODELS",
        "OVERSEER_CHEAP_DEFAULT_MODEL",
        "OVERSEER_CHEAP_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)

    env_file = data_dir / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# External config with BOM/quotes/export",
                'OPENROUTER_FREE_API_KEY="sk-free-only-key"',
                "OPENROUTER_CHEAP_API_KEY=PASTE_YOUR_CHEAP_MODEL_KEY_HERE",
            ]
        ),
        encoding="utf-8-sig",
    )

    loaded = load_external_dotenv()
    assert loaded["OPENROUTER_FREE_API_KEY"] == "sk-free-only-key"

    assert build_model_chain(None) == [
        "particle-ai/particle-1:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-31b-it:free",
    ]
    assert build_model_chain("free") == build_model_chain(None)
    assert build_model_chain("cheap") == [
        "deepseek/deepseek-v4.1-flash-1017",
        "z-ai/glm-5.3-flash",
    ]
    assert build_model_chain("particle-ai/particle-1-20261027:free")[0] == "particle-ai/particle-1:free"
    assert build_model_chain("deepseek/deepseek-v4.1-flash-20261017")[0] == "deepseek/deepseek-v4.1-flash-1017"

    assert select_api_key_for_model("particle-ai/particle-1:free") == "sk-free-only-key"
    with pytest.raises(ValueError, match="OPENROUTER_CHEAP_API_KEY") as exc_info:
        select_api_key_for_model("deepseek/deepseek-v4.1-flash-1017")
    assert str(env_file) in str(exc_info.value)

    env_file.write_text(
        "\n".join(
            [
                "OPENROUTER_FREE_API_KEY=sk-free-only-key",
                "OPENROUTER_CHEAP_API_KEY='sk-cheap-only-key'",
            ]
        ),
        encoding="utf-8",
    )
    assert select_api_key_for_model("deepseek/deepseek-v4.1-flash-1017") == "sk-cheap-only-key"

    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "frontend").mkdir(parents=True)

    seen_auth_by_model: dict[str, str] = {}

    def two_key_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            auth = request.headers.get("Authorization", "")
            if auth == "Bearer sk-free-only-key":
                models_data = [
                    {
                        "id": "particle-ai/particle-1:free",
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {
                        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {
                        "id": "google/gemma-4-31b-it:free",
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                ]
            else:
                models_data = [
                    {
                        "id": "deepseek/deepseek-v4.1-flash-1017",
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0.000000036", "completion": "0.0000005"},
                    },
                    {
                        "id": "z-ai/glm-5.3-flash",
                        "supported_parameters": ["tools"],
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000005"},
                    },
                ]
            return httpx.Response(200, json={"data": models_data})
        body = json.loads(request.content.decode("utf-8"))
        seen_auth_by_model[body["model"]] = request.headers.get("Authorization", "")
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
                                {"id": "k1", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Done", "notes": []}'}}
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(two_key_handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[
                WorkerTaskInput(worker_id="w_free", folder="backend", goal="run free"),
                WorkerTaskInput(worker_id="w_cheap", folder="frontend", goal="run cheap", model="cheap"),
            ],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    assert manifest.status == "completed"
    assert seen_auth_by_model["particle-ai/particle-1:free"] == "Bearer sk-free-only-key"
    assert seen_auth_by_model["deepseek/deepseek-v4.1-flash-1017"] == "Bearer sk-cheap-only-key"


@pytest.mark.asyncio
async def test_startup_failure_unlocks_project_root(worker_env: Path) -> None:
    repo = worker_env
    registry = RunRegistry()
    catalog = ModelCatalog()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_ok",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "Done", "notes": []}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="test startup rollback")],
        )
        with patch("overseer.worker.write_latest_run_pointer", side_effect=OSError("disk error")):
            with pytest.raises(OSError, match="disk error"):
                await prepare_and_launch_run(req, registry, client, catalog, repo)
        assert registry.has_active_run(repo) is None

        # Subsequent run on the same repo succeeds without being blocked
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)
    assert manifest.status == "completed"


@pytest.mark.asyncio
async def test_post_finish_tool_skipped_and_view_file_edge_cases(worker_env: Path) -> None:
    repo = worker_env
    (repo / "backend" / "empty.py").write_text("", encoding="utf-8")
    (repo / "backend" / "long.py").write_text("\n".join(f"L{i} = {i}" for i in range(1, 350)) + "\n", encoding="utf-8")

    # 1. Inverted view_file range rejected; start_line > 300 with omitted end_line defaults end_line = start_line + 299
    with pytest.raises(Exception, match="end_line"):
        ViewFileArgs(path="backend/app.py", start_line=10, end_line=5)
    shifted = ViewFileArgs(path="backend/long.py", start_line=301)
    assert shifted.end_line == 600
    folder_arg = Path(str(repo / "backend").upper()) if sys.platform in ("win32", "darwin") else (repo / "backend")
    text_shifted, rel_key, rel_hash = _view_file_sync(repo, folder_arg, shifted)
    assert "301: L301 = 301" in text_shifted
    assert rel_key == "backend/long.py" and rel_hash is not None

    # Empty file: start_line=1 reads cleanly; start_line=2 is past EOF
    empty_ok, e_key, e_hash = _view_file_sync(repo, repo / "backend", ViewFileArgs(path="backend/empty.py"))
    assert "lines 1..0 of 0" in empty_ok and e_key == "backend/empty.py" and e_hash is not None
    empty_eof, eof_key, eof_hash = _view_file_sync(
        repo, repo / "backend", ViewFileArgs(path="backend/empty.py", start_line=2, end_line=10)
    )
    assert "past EOF" in empty_eof and eof_key is None and eof_hash is None

    # 2. Direct execute_worker_tool rejects inverted view_file and post-finish tools
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261116_000011_dddddd"
    run_dir = proj_data / "runs" / run_id
    backups_dir = run_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="edge", requested_model="m:free")},
    )
    lock = asyncio.Lock()
    await save_manifest_atomic(m_path, manifest, lock)
    ctx = WorkerToolContext(
        project_root=repo,
        folder_abs=repo / "backend",
        backups_dir=backups_dir,
        run_dir=run_dir,
        manifest_path=m_path,
        manifest=manifest,
        worker=manifest.workers["w1"],
        run_lock=lock,
    )
    err_msg, is_err = await execute_worker_tool("view_file", {"path": "backend/app.py", "start_line": 5, "end_line": 2}, ctx)
    assert is_err and "end_line" in err_msg
    assert "backend/app.py" not in ctx.viewed_hashes

    # 3. Worker batch emitting [finish(...), replace_block(...)] does not execute post-finish edit
    def handler_post_finish(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
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
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "view_file", "arguments": '{"path": "backend/app.py"}'},
                                },
                                {
                                    "id": "t2",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "Finished early", "notes": []}'},
                                },
                                {
                                    "id": "t3",
                                    "type": "function",
                                    "function": {
                                        "name": "replace_block",
                                        "arguments": '{"path": "backend/app.py", "target": "return \'hi\'", "replacement": "def broken(:"}',
                                    },
                                },
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_post_finish)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="test post-finish")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        final_manifest = await _wait_for_run(repo, resp.run_id, registry)

    assert final_manifest.status == "completed"
    assert (repo / "backend" / "app.py").read_text(encoding="utf-8") == "def greet():\n    return 'hi'\n"
    assert final_manifest.workers["w1"].modified_files == {}


def test_prune_message_history_handles_malformed_view_args_and_omitted_end_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    env_file = data_dir / ".env"
    env_file.write_text("OPENROUTER_FREE_API_KEY=sk-free-retried\n", encoding="utf-8")

    orig_read_bytes = Path.read_bytes
    env_calls = 0

    def flaky_env_read(self: Path) -> bytes:
        nonlocal env_calls
        if self == env_file:
            env_calls += 1
            if env_calls == 1:
                raise PermissionError("locked .env")
        return orig_read_bytes(self)

    with (
        patch.object(Path, "read_bytes", flaky_env_read),
        patch("overseer.guardrails.time.sleep"),
    ):
        assert load_external_dotenv()["OPENROUTER_FREE_API_KEY"] == "sk-free-retried"
    assert env_calls == 2

    messages: list[dict] = [{"role": "system", "content": "sys"}]
    seq = [
        ("c1", "view_file", {"path": "a.py", "start_line": "bad", "end_line": None}, "bad args error"),
        ("c1b", "view_file", {"path": "b.py", "start_line": None}, "null start line read"),
        ("c2", "view_file", {"path": "long.py", "start_line": 301}, "shifted read"),
        ("c3", "view_file", {"path": "long.py", "start_line": 301, "end_line": 600}, "newer overlapping read"),
        ("c4", "list_dir", {"path": "."}, "dir1"),
        ("c5", "list_dir", {"path": "."}, "dir2"),
        ("c6", "list_dir", {"path": "."}, "dir3"),
        ("c7", "list_dir", {"path": "."}, "dir4"),
        ("c8", "finish", {"summary": "done", "notes": []}, "finished"),
    ]
    for cid, name, args, resp in seq:
        messages.append(_tool_call_msg(cid, name, args))
        messages.append({"role": "tool", "tool_call_id": cid, "content": resp})

    pruned = prune_message_history(messages, keep_recent_tools=6)
    tool_contents = [m["content"] for m in pruned if m["role"] == "tool"]
    assert tool_contents[0] == "bad args error"
    assert tool_contents[1] == "null start line read"
    assert tool_contents[2] == "[pruned view_file long.py lines 301-600]"

    pruned_strict = prune_message_history(messages, keep_recent_tools=2)
    strict_contents = [m["content"] for m in pruned_strict if m["role"] == "tool"]
    assert "[pruned view_file b.py lines 1-300]" in strict_contents

    char_capped = prune_message_history(
        [
            {"role": "system", "content": "sys"},
            _tool_call_msg("a1", "view_file", {"path": "x.py"}),
            {"role": "tool", "tool_call_id": "a1", "content": "Z" * 5000},
            _tool_call_msg("a2", "write_new_file", {"path": "y.py", "content": "W" * 5000}),
            {"role": "tool", "tool_call_id": "a2", "content": "Created y.py"},
        ],
        keep_recent_tools=6,
        max_history_chars=4000,
    )
    assert str(char_capped[2]["content"]).startswith("[pruned view_file")


@pytest.mark.asyncio
async def test_seven_tool_caps_and_multi_segment_skip_and_view_refresh(worker_env: Path) -> None:
    repo = worker_env
    # 350-line file for 300-line view_file cap
    (repo / "backend" / "long350.py").write_text(
        "\n".join(f"LINE_{i} = {i}" for i in range(1, 351)) + "\n", encoding="utf-8"
    )
    # 210 files for list_dir (200) and find_file (100) caps, plus ignored/denied dirs
    many_dir = repo / "backend" / "many"
    many_dir.mkdir(parents=True)
    for i in range(210):
        (many_dir / f"item_{i:03d}.txt").write_text(f"needle_text line {i}\n", encoding="utf-8")
    (many_dir / "node_modules").mkdir()
    (many_dir / "node_modules" / "dep.txt").write_text("needle_text hidden\n", encoding="utf-8")
    (many_dir / ".env").write_text("SECRET=needle_text\n", encoding="utf-8")
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("run: needle_text\n", encoding="utf-8")

    # 1) view_file 300-line cap starting at line 1
    view_300, _, h1 = _view_file_sync(repo, repo / "backend", ViewFileArgs(path="backend/long350.py"))
    assert "lines 1..300 of 350" in view_300
    assert "300: LINE_300" in view_300
    assert "LINE_301" not in view_300

    # 2) execute_worker_tool refreshes viewed_hashes on unedited file re-read
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261116_000044_eeeeee"
    run_dir = proj_data / "runs" / run_id
    backups_dir = run_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="caps", requested_model="m:free")},
    )
    lock = asyncio.Lock()
    await save_manifest_atomic(m_path, manifest, lock)
    ctx = WorkerToolContext(
        project_root=repo,
        folder_abs=repo / "backend",
        backups_dir=backups_dir,
        run_dir=run_dir,
        manifest_path=m_path,
        manifest=manifest,
        worker=manifest.workers["w1"],
        run_lock=lock,
    )
    _, err = await execute_worker_tool("view_file", {"path": "backend/app.py"}, ctx)
    assert not err
    first_hash = ctx.viewed_hashes["backend/app.py"]
    (repo / "backend" / "app.py").write_text("def greet():\n    return 'external'\n", encoding="utf-8")
    _, err = await execute_worker_tool("view_file", {"path": "backend/app.py"}, ctx)
    assert not err
    assert ctx.viewed_hashes["backend/app.py"] != first_hash
    assert h1 is not None

    # 3) list_dir 200-entry cap + ignored/denied skips
    listed, err = await execute_worker_tool("list_dir", {"path": "backend/many"}, ctx)
    assert not err
    assert "[capped at 200 entries]" in listed
    assert "node_modules" not in listed and ".env" not in listed

    # 4) search_dir 50-match cap + multi-segment .github/workflows skip
    searched, err = await execute_worker_tool("search_dir", {"query": "needle_text", "path": "."}, ctx)
    assert not err
    assert "[capped at 50 matches]" in searched
    assert ".github/workflows" not in searched and "node_modules" not in searched

    # 5) find_file 100-path cap
    found, err = await execute_worker_tool("find_file", {"pattern": "item_*.txt", "path": "backend/many"}, ctx)
    assert not err
    assert "[capped at 100 files]" in found


@pytest.mark.asyncio
async def test_budget_cutoff_skips_over_budget_writes_and_404_failover(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = worker_env
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "model/fallback:free")

    # 1) Over-budget response must halt BEFORE executing write_new_file tool call
    def handler_overbudget_write(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.40},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "w_over",
                                    "type": "function",
                                    "function": {
                                        "name": "write_new_file",
                                        "arguments": json.dumps({"path": "backend/should_not_exist.py", "content": "X = 1\n"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_overbudget_write)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_run_usd=0.25,
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="stop before write")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)
    assert manifest.status == "halted_budget"
    assert not (repo / "backend" / "should_not_exist.py").exists()

    # 2) HTTP 404 on primary :free model fails over cleanly to fallback:free; dead fallback slug filtered at launch
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "model/delisted:free,model/fallback:free")

    def handler_404_failover(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        if body["model"] == "model/primary:free":
            return httpx.Response(404, text="Model not found sk-or-v1-secretkeyvalue")
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_ok",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "Done via fallback", "notes": []}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_404_failover)) as client:
        req2 = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="failover 404")],
        )
        resp2 = await prepare_and_launch_run(req2, registry, client, catalog, repo)
        m2 = await _wait_for_run(repo, resp2.run_id, registry)
    assert m2.status == "completed"
    assert m2.workers["w1"].models_used == ["model/fallback:free"]

    # 3) Non-existent target_file outside task.folder rejected; malformed JSON sets failed status
    from overseer.guardrails import PathSecurityError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_404_failover)) as client:
        bad_req = StartRunRequest(
            project_root=str(repo),
            tasks=[
                WorkerTaskInput(
                    worker_id="w1",
                    folder="backend",
                    goal="bad target",
                    target_files=["frontend/does_not_exist.ts"],
                )
            ],
        )
        with pytest.raises(PathSecurityError, match="writable folder"):
            await prepare_and_launch_run(bad_req, registry, client, catalog, repo)

    def handler_bad_json(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(200, text="not-valid-json{{{")

    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_bad_json)) as client:
        req3 = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="bad json")],
        )
        resp3 = await prepare_and_launch_run(req3, registry, client, catalog, repo)
        m3 = await _wait_for_run(repo, resp3.run_id, registry)
    assert m3.status == "failed"
    assert m3.workers["w1"].status == "failed"
    assert m3.workers["w1"].stop_reason == "openrouter_invalid_json"


async def test_worker_prompt_guidance_and_length_truncation_recovery(worker_env: Path) -> None:
    repo = worker_env
    captured_messages: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_messages.append(body["messages"])
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 20, "completion_tokens": 50},
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": "Long analysis truncated..."},
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 30, "completion_tokens": 10},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_len",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "Recovered", "notes": []}'},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=12,
            tasks=[WorkerTaskInput(worker_id="w_len", folder="backend", goal="Test length recovery")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    assert manifest.status == "completed"
    sys_prompt = captured_messages[0][0]["content"]
    assert "Step budget: 12 turns" in sys_prompt
    assert "Closest match" in sys_prompt
    assert "Writable folder jail: backend" in sys_prompt
    nudge_msg = captured_messages[1][-1]["content"]
    assert "output token limit" in nudge_msg


async def test_worker_two_step_budget_warning_and_expanded_read_window(worker_env: Path) -> None:
    repo = worker_env
    captured_turns: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        turn += 1
        if turn <= 2:
            return httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {"id": f"v{turn}", "type": "function", "function": {"name": "view_file", "arguments": '{"path": "backend/app.py"}'}}
                                ],
                            },
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "fin_step", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Wrapped up on budget notice", "notes": []}'}}
                            ],
                        },
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=4,
            tasks=[WorkerTaskInput(worker_id="w_warn", folder="backend", goal="Check 2-step warning")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    assert manifest.status == "completed"
    assert manifest.workers["w_warn"].steps_used == 3
    # Turn 3 received the 2-step budget notice after Turn 2 completed (steps_used == 2 == 4 - 2)
    assert "2 steps remain" in captured_turns[2][-1]["content"]


def test_custom_instructions_replace_hardcoded_domain_packs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.config import (
        MAX_CUSTOM_INSTRUCTIONS_CHARS,
        custom_instructions_path,
        load_custom_instructions_sync,
        save_custom_instructions_sync,
    )
    from overseer.worker import _build_worker_system_prompt

    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))

    # No hidden/personal domain packs are injected by default
    for folder, goal, targets in (
        ("src/server", "Implement Roblox ProcessReceipt handler", ["src/server/Monetization.luau"]),
        ("prac", "Implement C++ pointer sort", ["prac/main.cpp"]),
        ("scripts", "Audit NetQoS DPC policy", ["scripts/network_tuning.ps1"]),
        ("bot", "Compute Sharpe and max drawdown on OHLCV trading bars", ["bot/risk.py"]),
    ):
        prompt = _build_worker_system_prompt(
            tmp_path,
            WorkerTaskInput(worker_id="w1", folder=folder, goal=goal, target_files=targets),
            shared_contract=None,
        )
        assert "Domain pack (" not in prompt
        assert "Custom instructions:" not in prompt

    assert load_custom_instructions_sync() == ""

    # User-saved custom instructions are injected cleanly and stripped of NUL bytes
    saved = save_custom_instructions_sync("  Use --!strict in Luau\x00\r\nNo global wait()  ")
    assert saved == "Use --!strict in Luau\nNo global wait()"
    assert load_custom_instructions_sync() == saved
    assert custom_instructions_path().is_file()

    prompt_with_custom = _build_worker_system_prompt(
        tmp_path,
        WorkerTaskInput(worker_id="w1", folder="src", goal="Edit module"),
        shared_contract=None,
    )
    assert "Custom instructions:\nUse --!strict in Luau\nNo global wait()" in prompt_with_custom

    # Exceeding cap raises ValueError
    with pytest.raises(ValueError, match="Custom instructions exceed"):
        save_custom_instructions_sync("x" * (MAX_CUSTOM_INSTRUCTIONS_CHARS + 1))

    # Clearing removes the file and stops injection
    assert save_custom_instructions_sync("   ") == ""
    assert not custom_instructions_path().exists()
    prompt_after_clear = _build_worker_system_prompt(
        tmp_path,
        WorkerTaskInput(worker_id="w1", folder="src", goal="Edit module"),
        shared_contract=None,
    )
    assert "Custom instructions:" not in prompt_after_clear


@pytest.mark.asyncio
async def test_file_outline_worker_tool_security_and_tier1_pruning(
    worker_env: Path,
    tmp_path: Path,
) -> None:
    repo = worker_env
    assert len(WORKER_TOOL_SCHEMAS) == 8
    assert "file_outline" in {t["function"]["name"] for t in WORKER_TOOL_SCHEMAS}

    sys_prompt = _build_worker_system_prompt(
        repo,
        WorkerTaskInput(worker_id="w1", folder="backend", goal="inspect files"),
        shared_contract=None,
    )
    assert "file_outline" in sys_prompt

    (repo / "backend" / "sample.py").write_text(
        "class Service:\n    def run(self, count: int) -> bool:\n        return count > 0\n",
        encoding="utf-8",
    )
    (repo / "backend" / ".env").write_text("SECRET=abc\n", encoding="utf-8")
    (repo / "backend" / "bin.dat").write_bytes(b"\x00\x01\x02")
    (repo / "backend" / "badutf8.txt").write_bytes(b"\xff\xfe\xfd")
    (repo / "backend" / "oversized.py").write_bytes(b"# " + (b"x" * 1_048_600))
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("SECRET = 1\n", encoding="utf-8")

    proj_data = get_project_data_dir(repo)
    run_id = "run_20261121_000001_aaaaaa"
    run_dir = proj_data / "runs" / run_id
    backups_dir = run_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="outline", requested_model="m:free")},
    )
    lock = asyncio.Lock()
    await save_manifest_atomic(m_path, manifest, lock)
    ctx = WorkerToolContext(
        project_root=repo,
        folder_abs=repo / "backend",
        backups_dir=backups_dir,
        run_dir=run_dir,
        manifest_path=m_path,
        manifest=manifest,
        worker=manifest.workers["w1"],
        run_lock=lock,
    )

    # Happy path (also verifies read-only access across project_root, e.g. frontend)
    out, is_err = await execute_worker_tool("file_outline", {"path": "backend/sample.py"}, ctx)
    assert not is_err
    assert "L1-L3: class Service" in out
    assert "L2-L3: def run(self, count: int) -> bool" in out

    # Path traversal, outside-root, deny-glob, directory, binary, non-UTF-8, >1 MiB rejected
    for bad_path in ("../outside.py", str(outside_file), "backend/.env", "backend"):
        msg, err = await execute_worker_tool("file_outline", {"path": bad_path}, ctx)
        assert err and ("Tool error" in msg or "not a file" in msg or "Path" in msg)

    msg, err = await execute_worker_tool("file_outline", {"path": "backend/bin.dat"}, ctx)
    assert err and "Binary" in msg

    msg, err = await execute_worker_tool("file_outline", {"path": "backend/badutf8.txt"}, ctx)
    assert err and "Non-UTF-8" in msg

    msg, err = await execute_worker_tool("file_outline", {"path": "backend/oversized.py"}, ctx)
    assert err and "exceeds" in msg

    # Pruning places file_outline in tier1_exploratory (pruned before write_new_file/replace_block and active view_file)
    messages: list[dict] = [{"role": "system", "content": "sys"}]
    seq = [
        ("c1", "view_file", {"path": "backend/app.py", "start_line": 1, "end_line": 10}, "ACTIVE app.py"),
        ("c2", "replace_block", {"path": "backend/app.py", "target": "a", "replacement": "b"}, "edited app"),
        ("c3", "file_outline", {"path": "backend/sample.py"}, "outline output"),
        ("c4", "write_new_file", {"path": "backend/new.py", "content": "X = 1\n"}, "created new"),
        ("c5", "finish", {"summary": "done", "notes": []}, "finished"),
    ]
    for cid, name, args, resp in seq:
        messages.append(_tool_call_msg(cid, name, args))
        messages.append({"role": "tool", "tool_call_id": cid, "content": resp})

    pruned = prune_message_history(messages, keep_recent_tools=4)
    tool_contents = [m["content"] for m in pruned if m["role"] == "tool"]
    assert tool_contents[0] == "ACTIVE app.py"
    assert tool_contents[1] == "edited app"
    assert tool_contents[2] == "[pruned file_outline backend/sample.py]"
    assert tool_contents[3] == "created new"


@pytest.mark.asyncio
async def test_notify_send_windows_toast_guards_env_passing_and_timeout_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import notify

    # 1. Auto-toast suppression checks PYTEST_CURRENT_TEST and OVERSEER_DISABLE_TOASTS
    assert notify.should_suppress_auto_toast() is True
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
    assert notify.should_suppress_auto_toast() is False
    monkeypatch.setenv("OVERSEER_DISABLE_TOASTS", "true")
    assert notify.should_suppress_auto_toast() is True

    # 2. Non-win32 guard returns False without spawning subprocess
    monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
    monkeypatch.setattr(notify.sys, "platform", "linux")
    with patch("overseer.notify.asyncio.create_subprocess_exec") as mock_exec:
        assert await notify.send_windows_toast("Title", "Body") is False
        mock_exec.assert_not_called()

    # 3. OVERSEER_DISABLE_TOASTS guard on win32 returns False without spawning subprocess
    monkeypatch.setattr(notify.sys, "platform", "win32")
    monkeypatch.setenv("OVERSEER_DISABLE_TOASTS", "1")
    with patch("overseer.notify.asyncio.create_subprocess_exec") as mock_exec:
        assert await notify.send_windows_toast("Title", "Body") is False
        mock_exec.assert_not_called()

    # 4. Mocked win32 subprocess passes title/message strictly via env vars (no interpolation) and uses CREATE_NO_WINDOW
    monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
    captured: dict = {}

    class FakeProcSuccess:
        def __init__(self, rc: int = 0) -> None:
            self._rc = rc
            self.killed = False
            self.wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            return self._rc

        def kill(self) -> None:
            self.killed = True

    async def fake_exec_ok(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcSuccess(rc=0)

    unsafe_title = "Overseer $(calc.exe) <xml>"
    unsafe_msg = "run_1 & 'quote' \"double\""
    with patch("overseer.notify.asyncio.create_subprocess_exec", side_effect=fake_exec_ok):
        ok = await notify.send_windows_toast(unsafe_title, unsafe_msg, timeout_sec=1.0)
    assert ok is True
    cmd_args = captured["args"]
    assert cmd_args[0].lower().endswith("powershell.exe")
    assert cmd_args[1:4] == ("-NoProfile", "-NonInteractive", "-Command")
    ps_script = cmd_args[4]
    assert unsafe_title not in ps_script
    assert unsafe_msg not in ps_script
    assert "SecurityElement]::Escape" in ps_script
    kw = captured["kwargs"]
    assert kw["env"]["OVERSEER_TOAST_TITLE"] == unsafe_title
    assert kw["env"]["OVERSEER_TOAST_BODY"] == unsafe_msg
    assert kw["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert kw["stdin"] == asyncio.subprocess.DEVNULL
    assert kw["stdout"] == asyncio.subprocess.DEVNULL
    assert kw["stderr"] == asyncio.subprocess.DEVNULL

    # Non-zero exit code returns False
    async def fake_exec_fail(*_args, **_kwargs):
        return FakeProcSuccess(rc=1)

    with patch("overseer.notify.asyncio.create_subprocess_exec", side_effect=fake_exec_fail):
        assert await notify.send_windows_toast("T", "M") is False

    # 5. TimeoutError kills and awaits proc.wait() to reap stale powershell.exe
    hanging_proc = SimpleNamespace()
    hanging_proc.killed = False
    hanging_proc.wait_calls = 0

    async def _hanging_wait() -> int:
        hanging_proc.wait_calls += 1
        if not hanging_proc.killed:
            await asyncio.sleep(5.0)
        return -1

    def _kill_proc() -> None:
        hanging_proc.killed = True

    hanging_proc.wait = _hanging_wait
    hanging_proc.kill = _kill_proc

    async def fake_exec_hang(*_args, **_kwargs):
        return hanging_proc

    with patch("overseer.notify.asyncio.create_subprocess_exec", side_effect=fake_exec_hang):
        timed_out_ok = await notify.send_windows_toast("Slow", "Msg", timeout_sec=0.05)
    assert timed_out_ok is False
    assert hanging_proc.killed is True
    assert hanging_proc.wait_calls >= 2

    # 6. FileNotFoundError / OSError returns False cleanly
    with patch("overseer.notify.asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no powershell")):
        assert await notify.send_windows_toast("Missing", "Msg") is False


@pytest.mark.asyncio
async def test_coordinator_fires_auto_toast_only_when_not_suppressed(
    worker_env: Path,
) -> None:
    repo = worker_env
    toast_calls: list[tuple[str, str]] = []
    toast_done = asyncio.Event()

    async def fake_send_toast(title: str, message: str, timeout_sec: float = 5.0) -> bool:
        toast_calls.append((title, message))
        toast_done.set()
        return True

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_toast",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "Done", "notes": []}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="toast test")],
        )
        with (
            patch("overseer.worker.should_suppress_auto_toast", return_value=False),
            patch("overseer.worker.send_completion_toast", side_effect=fake_send_toast),
        ):
            resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
            manifest = await _wait_for_run(repo, resp.run_id, registry)
            await asyncio.wait_for(toast_done.wait(), timeout=2.0)

    assert manifest.status == "completed"
    assert len(toast_calls) == 1
    assert toast_calls[0][0] == "Overseer run finished"
    assert resp.run_id in toast_calls[0][1]
    assert "1/1 workers" in toast_calls[0][1]


def _assert_no_tool_to_user_alternation(messages: list[dict]) -> None:
    for prev, cur in zip(messages, messages[1:]):
        assert not (prev.get("role") == "tool" and cur.get("role") == "user"), (
            f"Unexpected tool -> user alternation: {prev!r} -> {cur!r}"
        )


def _assistant_tool_resp(tool_calls: list[dict], finish_reason: str = "tool_calls") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                }
            ],
        },
    )


async def test_write_new_file_progress_hints_and_post_write_read_loop_breaker(worker_env: Path) -> None:
    repo = worker_env
    captured_turns: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        turn += 1
        if turn == 1:
            tc = {
                "id": "w1",
                "type": "function",
                "function": {
                    "name": "write_new_file",
                    "arguments": json.dumps({"path": "backend/one.py", "content": "X = 1\n"}),
                },
            }
        elif turn == 2:
            tc = {
                "id": "w2",
                "type": "function",
                "function": {
                    "name": "write_new_file",
                    "arguments": json.dumps({"path": "backend/two.py", "content": "Y = 2\n"}),
                },
            }
        else:
            # Repeatedly re-read after all target files are already written
            tc = {
                "id": f"r{turn}",
                "type": "function",
                "function": {
                    "name": "view_file",
                    "arguments": json.dumps({"path": "backend/one.py", "start_line": 1, "end_line": 10}),
                },
            }
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]}}],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=20,
            tasks=[
                WorkerTaskInput(
                    worker_id="w_loop",
                    folder="backend",
                    goal="Create two files",
                    target_files=["backend/one.py", "backend/two.py"],
                )
            ],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    w = manifest.workers["w_loop"]
    assert w.status == "escalated"
    assert w.stop_reason == "stalled_after_writes"
    # Stopped at 2 writes + MAX_REDUNDANT_READ_TURNS (3) redundant self-written read turns = 5 steps
    assert w.steps_used == 2 + MAX_REDUNDANT_READ_TURNS
    assert w.static_check.status == "passed"
    # Turn 2 saw remaining target file hint after writing backend/one.py
    assert "Remaining new target files in your folder: backend/two.py" in captured_turns[1][-1]["content"]
    # Turn 3 saw all-targets-written finish hint after writing backend/two.py
    assert "All target files in your folder are now written; call finish(summary, notes) now" in captured_turns[2][-1]["content"]
    # Turn 5 (after 2nd identical post-write view_file) saw final read-only warning appended on last tool message
    assert captured_turns[4][-1]["role"] == "tool"
    assert "FINAL READ-ONLY WARNING:" in captured_turns[4][-1]["content"]
    assert "Stop re-reading and call finish(summary, notes) now" in captured_turns[4][-1]["content"]
    assert "One more read-only turn without an edit or finish() will halt execution." in captured_turns[4][-1]["content"]
    for turn_msgs in captured_turns:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_max_steps_reached_runs_static_precheck_and_final_step_notice(worker_env: Path) -> None:
    repo = worker_env
    captured_turns: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        turn += 1
        if turn == 1:
            tc = {
                "id": "w_bad",
                "type": "function",
                "function": {
                    "name": "write_new_file",
                    "arguments": json.dumps({"path": "backend/broken.py", "content": "def bad(:\n    pass\n"}),
                },
            }
        else:
            tc = {
                "id": f"l{turn}",
                "type": "function",
                "function": {"name": "list_dir", "arguments": json.dumps({"path": "backend"})},
            }
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]}}],
            },
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=3,
            tasks=[WorkerTaskInput(worker_id="w_max", folder="backend", goal="Write file and hit cap")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    w = manifest.workers["w_max"]
    assert w.status == "escalated"
    assert w.stop_reason == "max_steps_reached"
    assert w.steps_used == 3
    # Turn 3 received the FINAL STEP notice appended to the last tool message after Turn 2 (steps_used == 2 == 3 - 1)
    assert captured_turns[2][-1]["role"] == "tool"
    assert "FINAL STEP: 1 step remains" in captured_turns[2][-1]["content"]
    for turn_msgs in captured_turns:
        _assert_no_tool_to_user_alternation(turn_msgs)
    # Static precheck ran on touched files even though finish() was never called
    assert w.static_check.status == "failed"
    assert any("backend/broken.py" in err for err in w.static_check.errors)


async def test_mixed_new_and_existing_target_files_progress_hints_and_stagnation_guard(
    worker_env: Path,
) -> None:
    repo = worker_env
    folder_abs = repo / "backend"

    # Unit check _pending_folder_targets: deduplication, folder scoping, and existing vs new tracking
    has_t, missing_new, unedited_existing = _pending_folder_targets(
        repo,
        folder_abs,
        ["./backend/app.py", "backend/app.py", "backend\\app.py", "backend/new1.py", "frontend/index.ts"],
    )
    assert has_t is True
    assert missing_new == ["backend/new1.py"]
    assert unedited_existing == ["backend/app.py"]

    assert _pending_folder_targets(repo, folder_abs, []) == (False, [], [])
    assert _pending_folder_targets(repo, folder_abs, ["frontend/index.ts"]) == (False, [], [])

    captured_turns: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        turn += 1
        if turn == 1:
            tc = {
                "id": "m1",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/new1.py", "content": "A = 1\n"})},
            }
        elif turn == 2:
            tc = {
                "id": "m2",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/new2.py", "content": "B = 2\n"})},
            }
        elif turn in (3, 4):
            # View existing target file twice so turn 4 triggers a post-write stagnation nudge while app.py is unedited
            tc = {
                "id": f"m{turn}",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py", "start_line": 1, "end_line": 10})},
            }
        elif turn == 5:
            tc = {
                "id": "m5",
                "type": "function",
                "function": {
                    "name": "replace_block",
                    "arguments": json.dumps({"path": "backend/app.py", "target": "return 'hi'", "replacement": "return 'hello'"}),
                },
            }
        elif turn == 6:
            tc = {
                "id": "m6",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/extra.py", "content": "C = 3\n"})},
            }
        else:
            tc = {
                "id": "m7",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "All mixed targets handled", "notes": []})},
            }
        return _assistant_tool_resp([tc])

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=15,
            tasks=[
                WorkerTaskInput(
                    worker_id="w_mixed",
                    folder="backend",
                    goal="Create new1/new2 and edit app.py",
                    target_files=["backend/app.py", "backend/new1.py", "backend/new2.py", "frontend/index.ts"],
                )
            ],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    w = manifest.workers["w_mixed"]
    assert w.status == "completed"
    assert w.static_check.status == "passed"
    # After creating new1.py: still missing new2.py AND app.py is unedited
    t1_out = captured_turns[1][-1]["content"]
    assert "Remaining new target files in your folder: backend/new2.py." in t1_out
    assert "Existing folder target(s) not yet edited: backend/app.py." in t1_out
    assert "All target files in your folder are now written" not in t1_out
    # After creating new2.py: all NEW targets exist, but existing app.py is unedited -> MUST NOT claim all targets done
    t2_out = captured_turns[2][-1]["content"]
    assert "All new target files created; existing folder target(s) not yet edited: backend/app.py." in t2_out
    assert "All target files in your folder are now written; call finish(summary, notes) now" not in t2_out
    # After repeated read on turn 4: stagnation nudge lists unedited existing target instead of telling worker to finish
    t4_out = captured_turns[4][-1]["content"]
    assert captured_turns[4][-1]["role"] == "tool"
    assert "Existing folder target(s) not yet edited: backend/app.py." in t4_out
    assert "Stop re-reading and call finish(summary, notes) now" not in t4_out
    # After editing app.py (turn 5) and writing extra.py (turn 6): all targets handled -> emits finish hint
    t6_out = captured_turns[6][-1]["content"]
    assert "All target files in your folder are now written; call finish(summary, notes) now" in t6_out
    for turn_msgs in captured_turns:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_failed_finish_and_block_replace_error_reset_readonly_counter_for_repair_reads(
    worker_env: Path,
) -> None:
    repo = worker_env
    (repo / "frontend" / "ref.txt").write_text(
        "".join(f"line {i}\n" for i in range(1, 120)),
        encoding="utf-8",
    )

    # Sub-scenario A: failed finish() static check after 7 post-write novel reads followed by repair view_file
    captured_a: list[list[dict]] = []
    turn_a = 0

    def handler_a(request: httpx.Request) -> httpx.Response:
        nonlocal turn_a
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_a.append(body["messages"])
        turn_a += 1
        if turn_a == 1:
            tc = {
                "id": "a1",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/broken.py", "content": "def broken(:\n    return 1\n"})},
            }
        elif 2 <= turn_a <= 8:
            # 7 novel reads -> readonly_turns_since_write reaches MAX_POST_WRITE_READONLY_TURNS - 1 (7)
            idx = turn_a - 2
            tc = {
                "id": f"ar{idx}",
                "type": "function",
                "function": {
                    "name": "view_file",
                    "arguments": json.dumps({"path": "frontend/ref.txt", "start_line": 1 + idx * 10, "end_line": 10 + idx * 10}),
                },
            }
        elif turn_a == 9:
            tc = {
                "id": "af1",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "Attempt finish", "notes": []})},
            }
        elif turn_a == 10:
            # Repair read immediately after failed finish() must NOT kill the worker
            tc = {
                "id": "arep",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/broken.py", "start_line": 1, "end_line": 10})},
            }
        elif turn_a == 11:
            tc = {
                "id": "afix",
                "type": "function",
                "function": {
                    "name": "replace_block",
                    "arguments": json.dumps({"path": "backend/broken.py", "target": "def broken(:", "replacement": "def fixed():"}),
                },
            }
        else:
            tc = {
                "id": "af2",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "Fixed syntax", "notes": []})},
            }
        return _assistant_tool_resp([tc])

    registry_a = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_a)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=20,
            tasks=[WorkerTaskInput(worker_id="w_fin_fail", folder="backend", goal="Write and repair after failed finish")],
        )
        resp_a = await prepare_and_launch_run(req_a, registry_a, client, catalog, repo)
        manifest_a = await _wait_for_run(repo, resp_a.run_id, registry_a)

    wa = manifest_a.workers["w_fin_fail"]
    assert wa.status == "completed"
    assert wa.static_check.status == "passed"
    # Turn 10 (after failed finish on turn 9) must NOT append a false stagnation nudge
    failed_fin_msg = captured_a[9][-1]
    assert failed_fin_msg["role"] == "tool"
    assert failed_fin_msg["content"].startswith("Static pre-check failed:")
    assert "Progress reminder:" not in failed_fin_msg["content"]
    assert "FINAL READ-ONLY WARNING:" not in failed_fin_msg["content"]
    assert "Stop re-reading" not in failed_fin_msg["content"]
    for turn_msgs in captured_a:
        _assert_no_tool_to_user_alternation(turn_msgs)

    # Sub-scenario B: BlockReplaceError after 7 post-write novel reads followed by repair view_file
    captured_b: list[list[dict]] = []
    turn_b = 0

    def handler_b(request: httpx.Request) -> httpx.Response:
        nonlocal turn_b
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_b.append(body["messages"])
        turn_b += 1
        if turn_b == 1:
            tc = {
                "id": "b1",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py", "start_line": 1, "end_line": 10})},
            }
        elif turn_b == 2:
            tc = {
                "id": "b2",
                "type": "function",
                "function": {
                    "name": "replace_block",
                    "arguments": json.dumps({"path": "backend/app.py", "target": "return 'hi'", "replacement": "return 'v1'"}),
                },
            }
        elif 3 <= turn_b <= 9:
            idx = turn_b - 3
            tc = {
                "id": f"br{idx}",
                "type": "function",
                "function": {
                    "name": "view_file",
                    "arguments": json.dumps({"path": "frontend/ref.txt", "start_line": 1 + idx * 10, "end_line": 10 + idx * 10}),
                },
            }
        elif turn_b == 10:
            tc = {
                "id": "berr",
                "type": "function",
                "function": {
                    "name": "replace_block",
                    "arguments": json.dumps({"path": "backend/app.py", "target": "NONEXISTENT_LINE", "replacement": "x"}),
                },
            }
        elif turn_b == 11:
            tc = {
                "id": "brep",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py", "start_line": 1, "end_line": 10})},
            }
        elif turn_b == 12:
            tc = {
                "id": "bfix",
                "type": "function",
                "function": {
                    "name": "replace_block",
                    "arguments": json.dumps({"path": "backend/app.py", "target": "return 'v1'", "replacement": "return 'v2'"}),
                },
            }
        else:
            tc = {
                "id": "bfin",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "Repaired after BlockReplaceError", "notes": []})},
            }
        return _assistant_tool_resp([tc])

    registry_b = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_b)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=20,
            tasks=[WorkerTaskInput(worker_id="w_bre_fail", folder="backend", goal="Edit and recover from BlockReplaceError")],
        )
        resp_b = await prepare_and_launch_run(req_b, registry_b, client, catalog, repo)
        manifest_b = await _wait_for_run(repo, resp_b.run_id, registry_b)

    wb = manifest_b.workers["w_bre_fail"]
    assert wb.status == "completed"
    assert wb.static_check.status == "passed"
    # Turn 11 (after BlockReplaceError on turn 10) must NOT append a false stagnation nudge
    bre_msg = captured_b[10][-1]
    assert bre_msg["role"] == "tool"
    assert bre_msg["content"].startswith("Tool error (replace_block):")
    assert "Progress reminder:" not in bre_msg["content"]
    assert "FINAL READ-ONLY WARNING:" not in bre_msg["content"]
    assert "Stop re-reading" not in bre_msg["content"]
    for turn_msgs in captured_b:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_mixed_multi_tool_dup_read_and_write_suppresses_stagnation_nudge(worker_env: Path) -> None:
    repo = worker_env
    captured_turns: list[list[dict]] = []
    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        turn += 1
        if turn in (1, 2):
            tcs = [
                {
                    "id": f"pre_r{turn}",
                    "type": "function",
                    "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py", "start_line": 1, "end_line": 10})},
                }
            ]
        elif turn == 3:
            # Pre-write mixed batch: 3rd identical read + write_new_file -> must NOT emit stagnation nudge
            tcs = [
                {
                    "id": "pre_r3",
                    "type": "function",
                    "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py", "start_line": 1, "end_line": 10})},
                },
                {
                    "id": "pre_w",
                    "type": "function",
                    "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/fresh.py", "content": "V = 1\n"})},
                },
            ]
        elif turn == 4:
            tcs = [
                {
                    "id": "post_r1",
                    "type": "function",
                    "function": {"name": "view_file", "arguments": json.dumps({"path": "frontend/index.ts", "start_line": 1, "end_line": 10})},
                }
            ]
        elif turn == 5:
            # Post-write mixed batch: 2nd identical read + replace_block -> must NOT emit stagnation nudge
            tcs = [
                {
                    "id": "post_r2",
                    "type": "function",
                    "function": {"name": "view_file", "arguments": json.dumps({"path": "frontend/index.ts", "start_line": 1, "end_line": 10})},
                },
                {
                    "id": "post_w",
                    "type": "function",
                    "function": {
                        "name": "replace_block",
                        "arguments": json.dumps({"path": "backend/fresh.py", "target": "V = 1", "replacement": "V = 2"}),
                    },
                },
            ]
        else:
            tcs = [
                {
                    "id": "mix_fin",
                    "type": "function",
                    "function": {"name": "finish", "arguments": json.dumps({"summary": "Done", "notes": []})},
                }
            ]
        return _assistant_tool_resp(tcs)

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=12,
            tasks=[WorkerTaskInput(worker_id="w_mix", folder="backend", goal="Mixed read+write batch")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    assert manifest.workers["w_mix"].status == "completed"
    for idx in (3, 5):
        last_msg = captured_turns[idx][-1]
        assert last_msg["role"] == "tool"
        assert "Progress reminder:" not in last_msg["content"]
        assert "FINAL READ-ONLY WARNING:" not in last_msg["content"]
        assert "stop repeating" not in last_msg["content"]
    for turn_msgs in captured_turns:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_canonical_tool_sig_normalization_and_pre_write_read_loop_breaker(worker_env: Path) -> None:
    # Direct _canonical_tool_sig normalization checks
    base_vf = _canonical_tool_sig("view_file", {"path": "a.py"})
    assert _canonical_tool_sig("view_file", {"path": "./a.py"}) == base_vf
    assert _canonical_tool_sig("view_file", {"path": "sub/../a.py"}) == base_vf
    assert _canonical_tool_sig("view_file", {"path": "a.py", "start_line": 1, "end_line": 300}) == base_vf
    assert _canonical_tool_sig("view_file", {"path": "./a.py", "start_line": 1, "end_line": 999}) == base_vf
    assert _canonical_tool_sig("view_file", {"path": "a.py", "start_line": 2, "end_line": 300}) != base_vf
    assert _canonical_tool_sig("view_file", {"path": "sub\\a.py"}) == _canonical_tool_sig("view_file", {"path": "sub/a.py"})
    assert _canonical_tool_sig("view_file", {"path": ".gitignore"}) == _canonical_tool_sig("view_file", {"path": "./.gitignore"})
    assert _canonical_tool_sig("view_file", {"path": ".gitignore"}) != _canonical_tool_sig("view_file", {"path": "gitignore"})
    assert _canonical_tool_sig("file_outline", {"path": "./backend/app.py"}) == _canonical_tool_sig("file_outline", {"path": "backend\\app.py"})
    assert _canonical_tool_sig("list_dir", {}) == _canonical_tool_sig("list_dir", {"path": "./"})
    assert _canonical_tool_sig("search_dir", {"query": "hi"}) == _canonical_tool_sig("search_dir", {"query": "hi", "path": "."})
    assert _canonical_tool_sig("find_file", {"pattern": "*.py"}) == _canonical_tool_sig("find_file", {"pattern": "*.py", "path": "./"})

    # End-to-end pre-write redundant read loop breaker using differently formatted equivalent args
    repo = worker_env
    captured_turns: list[list[dict]] = []
    turn = 0
    equiv_args = [
        {"path": "./backend/app.py"},
        {"path": "backend/app.py", "start_line": 1, "end_line": 300},
        {"path": "backend\\app.py", "start_line": 1, "end_line": 999},
        {"path": "backend/./app.py", "start_line": 1},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_turns.append(body["messages"])
        args = equiv_args[min(turn, len(equiv_args) - 1)]
        turn += 1
        return _assistant_tool_resp(
            [{"id": f"pre_dup_{turn}", "type": "function", "function": {"name": "view_file", "arguments": json.dumps(args)}}]
        )

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=15,
            tasks=[WorkerTaskInput(worker_id="w_preloop", folder="backend", goal="Loop reading before writing")],
        )
        resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, registry)

    w = manifest.workers["w_preloop"]
    assert w.status == "escalated"
    assert w.stop_reason == "read_loop_detected"
    assert w.steps_used == 1 + MAX_REDUNDANT_READ_TURNS
    assert w.static_check.status == "not_run"
    # Turn 4 received FINAL READ-ONLY WARNING on the last tool message after 2nd redundant read (turn 3)
    warn_msg = captured_turns[3][-1]
    assert warn_msg["role"] == "tool"
    assert "FINAL READ-ONLY WARNING:" in warn_msg["content"]
    assert "stop repeating read-only calls" in warn_msg["content"]
    assert "One more read-only turn without an edit or finish() will halt execution." in warn_msg["content"]
    for turn_msgs in captured_turns:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_post_write_redundant_read_failed_precheck_and_novel_reads_up_to_cap(worker_env: Path) -> None:
    repo = worker_env
    (repo / "frontend" / "big_ref.txt").write_text(
        "".join(f"ref line {i}\n" for i in range(1, 200)),
        encoding="utf-8",
    )
    catalog = ModelCatalog()

    # Part A: Post-write redundant read loop breaker with static precheck FAILED
    captured_a: list[list[dict]] = []
    turn_a = 0

    def handler_a(request: httpx.Request) -> httpx.Response:
        nonlocal turn_a
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_a.append(body["messages"])
        turn_a += 1
        if turn_a == 1:
            tc = {
                "id": "sf1",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/syntax_err.py", "content": "def bad(:\n    pass\n"})},
            }
        else:
            tc = {
                "id": f"sr{turn_a}",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/syntax_err.py", "start_line": 1, "end_line": 10})},
            }
        return _assistant_tool_resp([tc])

    reg_a = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_a)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=15,
            tasks=[WorkerTaskInput(worker_id="w_stall_fail", folder="backend", goal="Write bad syntax then re-read")],
        )
        resp_a = await prepare_and_launch_run(req_a, reg_a, client, catalog, repo)
        manifest_a = await _wait_for_run(repo, resp_a.run_id, reg_a)

    wa = manifest_a.workers["w_stall_fail"]
    assert wa.status == "escalated"
    assert wa.stop_reason == "stalled_after_writes"
    assert wa.steps_used == 1 + MAX_REDUNDANT_READ_TURNS
    assert wa.static_check.status == "failed"
    assert any("backend/syntax_err.py" in err for err in wa.static_check.errors)
    for turn_msgs in captured_a:
        _assert_no_tool_to_user_alternation(turn_msgs)

    # Part B: 7 novel post-write reads (> old cap of 6) NOT prematurely killed, finishing on turn 9
    captured_b: list[list[dict]] = []
    turn_b = 0

    def handler_b(request: httpx.Request) -> httpx.Response:
        nonlocal turn_b
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_b.append(body["messages"])
        turn_b += 1
        if turn_b == 1:
            tc = {
                "id": "nv_w",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/ok.py", "content": "OK = True\n"})},
            }
        elif 2 <= turn_b <= MAX_POST_WRITE_READONLY_TURNS:
            idx = turn_b - 2
            tc = {
                "id": f"nv_r{idx}",
                "type": "function",
                "function": {
                    "name": "view_file",
                    "arguments": json.dumps({"path": "frontend/big_ref.txt", "start_line": 1 + idx * 10, "end_line": 10 + idx * 10}),
                },
            }
        else:
            tc = {
                "id": "nv_fin",
                "type": "function",
                "function": {"name": "finish", "arguments": json.dumps({"summary": "Finished after 7 novel reads", "notes": []})},
            }
        return _assistant_tool_resp([tc])

    reg_b = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_b)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=20,
            tasks=[WorkerTaskInput(worker_id="w_novel_ok", folder="backend", goal="Write then do 7 novel reads")],
        )
        resp_b = await prepare_and_launch_run(req_b, reg_b, client, catalog, repo)
        manifest_b = await _wait_for_run(repo, resp_b.run_id, reg_b)

    wb = manifest_b.workers["w_novel_ok"]
    assert wb.status == "completed"
    assert wb.steps_used == MAX_POST_WRITE_READONLY_TURNS + 1  # 1 write + 7 novel reads + 1 finish = 9
    assert wb.static_check.status == "passed"
    # Nudge at turn POST_WRITE_NUDGE_TURN (after 4th novel read -> visible on turn 6 = index 5)
    nudge_msg = captured_b[1 + POST_WRITE_NUDGE_TURN][-1]
    assert nudge_msg["role"] == "tool"
    assert "Progress reminder:" in nudge_msg["content"]
    assert "Make your next edit or call finish(summary, notes) now" in nudge_msg["content"]
    assert "stop repeating read-only calls" not in nudge_msg["content"]
    # Novel post-write reads do NOT increment post_write_stall_turns, so no FINAL READ-ONLY WARNING
    for turn_msgs in captured_b:
        _assert_no_tool_to_user_alternation(turn_msgs)

    # Part C: 8 redundant post-write reads hit MAX_POST_WRITE_READONLY_TURNS cap
    turn_c = 0

    def handler_c(request: httpx.Request) -> httpx.Response:
        nonlocal turn_c
        if request.url.path.endswith("/models"):
            return _models_response()
        turn_c += 1
        if turn_c == 1:
            tc = {
                "id": "cap_w",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/ok2.py", "content": "OK2 = True\n"})},
            }
        else:
            idx = turn_c - 2
            slice_idx = idx % 2
            tc = {
                "id": f"cap_r{idx}",
                "type": "function",
                "function": {
                    "name": "view_file",
                    "arguments": json.dumps({"path": "frontend/big_ref.txt", "start_line": 1 + slice_idx * 10, "end_line": 10 + slice_idx * 10}),
                },
            }
        return _assistant_tool_resp([tc])

    reg_c = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_c)) as client:
        req_c = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=20,
            tasks=[WorkerTaskInput(worker_id="w_novel_cap", folder="backend", goal="Write then exceed redundant read cap")],
        )
        resp_c = await prepare_and_launch_run(req_c, reg_c, client, catalog, repo)
        manifest_c = await _wait_for_run(repo, resp_c.run_id, reg_c)

    wc = manifest_c.workers["w_novel_cap"]
    assert wc.status == "escalated"
    assert wc.stop_reason == "stalled_after_writes"
    # 1 write + 2 initial novel reads + MAX_REDUNDANT_READ_TURNS redundant alternating reads
    assert wc.steps_used == 3 + MAX_REDUNDANT_READ_TURNS
    assert wc.static_check.status == "passed"


def test_prune_message_history_write_compaction_subsumption_and_failed_finish_tier3() -> None:
    # Part A: Compact large succeeded write_new_file/replace_block args (including last_assistant_idx)
    # with schema-valid placeholders BEFORE evicting tier3_active_views; leave failed writes intact.
    active_view_body = "ACTIVE REF VIEW " + ("V" * 400)
    failed_content = "F" * 2500
    messages_a: list[dict] = [
        {"role": "system", "content": "sys"},
        _tool_call_msg("c1", "view_file", {"path": "ref.py", "start_line": 1, "end_line": 50}),
        {"role": "tool", "tool_call_id": "c1", "content": active_view_body},
        _tool_call_msg("c2", "replace_block", {"path": "a.py", "target": "T" * 1500, "replacement": "R" * 1500}),
        {"role": "tool", "tool_call_id": "c2", "content": "Replaced block in a.py (+1/-1 lines)."},
        _tool_call_msg("c3", "write_new_file", {"path": "fail.py", "content": failed_content}),
        {"role": "tool", "tool_call_id": "c3", "content": "Tool error (write_new_file): already exists"},
        _tool_call_msg("c4", "write_new_file", {"path": "big.py", "content": "B" * 3500}),
        {"role": "tool", "tool_call_id": "c4", "content": "Created big.py (1 lines)."},
    ]
    pruned_a = prune_message_history(messages_a, keep_recent_tools=6, max_history_chars=4500)
    tool_by_id = {m["tool_call_id"]: m["content"] for m in pruned_a if m.get("role") == "tool"}
    args_by_id = {
        tc["id"]: json.loads(tc["function"]["arguments"])
        for m in pruned_a
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    # tier3_active_views preserved because write compaction freed enough space first
    assert tool_by_id["c1"] == active_view_body
    # Succeeded replace_block and succeeded write_new_file (on last_assistant_idx!) compacted with schema-valid payloads
    assert args_by_id["c2"] == {"path": "a.py", "target": "[compacted]", "replacement": "[compacted]"}
    ReplaceBlockArgs.model_validate(args_by_id["c2"])
    assert args_by_id["c4"] == {"path": "big.py", "content": "[written to disk; compacted]"}
    WriteNewFileArgs.model_validate(args_by_id["c4"])
    # Failed write_new_file NOT compacted
    assert args_by_id["c3"]["content"] == failed_content

    # Part B: Subsumption-only view_file supersession (1-line overlap 1-300 + 300-600 NOT pruned)
    messages_b: list[dict] = [{"role": "system", "content": "sys"}]
    seq_b = [
        ("b1", "view_file", {"path": "big.py", "start_line": 1, "end_line": 300}, "CHUNK 1-300"),
        ("b2", "view_file", {"path": "big.py", "start_line": 300, "end_line": 600}, "CHUNK 300-600"),
        ("b3", "view_file", {"path": "sub.py", "start_line": 10, "end_line": 50}, "OLD SUB"),
        ("b4", "view_file", {"path": "./sub.py", "start_line": 1, "end_line": 100}, "FULL SUB"),
        ("b5", "list_dir", {"path": "."}, "DIR LIST"),
        ("b6", "replace_block", {"path": "big.py", "target": "a", "replacement": "b"}, "EDITED BIG"),
        ("b7", "write_new_file", {"path": "z.py", "content": "Z=1"}, "CREATED Z"),
    ]
    for cid, name, args, resp in seq_b:
        messages_b.append(_tool_call_msg(cid, name, args))
        messages_b.append({"role": "tool", "tool_call_id": cid, "content": resp})
    pruned_b = prune_message_history(messages_b, keep_recent_tools=4)
    b_by_id = {m["tool_call_id"]: m["content"] for m in pruned_b if m.get("role") == "tool"}
    assert b_by_id["b1"] == "CHUNK 1-300"
    assert b_by_id["b2"] == "CHUNK 300-600"
    assert b_by_id["b3"].startswith("[pruned view_file sub.py lines 10-50]")
    assert b_by_id["b4"] == "FULL SUB"
    assert b_by_id["b5"].startswith("[pruned list_dir .]")
    assert b_by_id["b6"].startswith("[pruned replace_block big.py]")

    # Part C: Failed finish() static-check diagnostics placed in tier3 (pruned after tier1 and tier2)
    failed_diag = "Static pre-check failed:\nbackend/a.py:1: SyntaxError"
    messages_c: list[dict] = [{"role": "system", "content": "sys"}]
    seq_c = [
        ("f1", "list_dir", {"path": "."}, "DIR"),
        ("f2", "write_new_file", {"path": "a.py", "content": "def bad(:"}, "Created a.py"),
        ("f3", "finish", {"summary": "done", "notes": []}, failed_diag),
        ("f4", "replace_block", {"path": "a.py", "target": "x", "replacement": "y"}, "Edited a.py"),
        ("f5", "view_file", {"path": "a.py", "start_line": 1, "end_line": 20}, "VIEW A"),
    ]
    for cid, name, args, resp in seq_c:
        messages_c.append(_tool_call_msg(cid, name, args))
        messages_c.append({"role": "tool", "tool_call_id": cid, "content": resp})
    pruned_c = prune_message_history(messages_c, keep_recent_tools=2)
    c_by_id = {m["tool_call_id"]: m["content"] for m in pruned_c if m.get("role") == "tool"}
    assert c_by_id["f1"].startswith("[pruned list_dir .]")
    assert c_by_id["f2"].startswith("[pruned write_new_file a.py]")
    assert c_by_id["f4"].startswith("[pruned replace_block a.py]")
    assert c_by_id["f3"] == failed_diag
    assert c_by_id["f5"] == "VIEW A"


async def test_cancel_event_during_escalation_precheck_preserves_cancelled_and_halted_budget(
    worker_env: Path,
) -> None:
    repo = worker_env
    catalog = ModelCatalog()
    loop = asyncio.get_running_loop()

    # Sub-case A: User cancel flag written while run_static_precheck is running on max_steps_reached
    reg_a = RunRegistry()
    turn_a = 0

    def handler_a(request: httpx.Request) -> httpx.Response:
        nonlocal turn_a
        if request.url.path.endswith("/models"):
            return _models_response()
        turn_a += 1
        if turn_a == 1:
            tc = {
                "id": "ca1",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/c1.py", "content": "X = 1\n"})},
            }
        else:
            tc = {
                "id": "ca2",
                "type": "function",
                "function": {"name": "list_dir", "arguments": json.dumps({"path": "backend"})},
            }
        return _assistant_tool_resp([tc])

    def fake_precheck_user_cancel(project_root: Path, _touched: list[str]) -> StaticCheckResult:
        active_id = reg_a.has_active_run(project_root)
        assert active_id is not None
        run_dir = get_project_data_dir(project_root) / "runs" / active_id
        cancel_flag_path_for(run_dir).write_text("cancelled", encoding="utf-8")
        return StaticCheckResult(status="passed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_a)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=2,
            tasks=[WorkerTaskInput(worker_id="w_ca", folder="backend", goal="Hit max steps while cancelled")],
        )
        with patch("overseer.worker.run_static_precheck", side_effect=fake_precheck_user_cancel):
            resp_a = await prepare_and_launch_run(req_a, reg_a, client, catalog, repo)
            manifest_a = await _wait_for_run(repo, resp_a.run_id, reg_a)

    assert manifest_a.status == "cancelled"
    assert manifest_a.workers["w_ca"].status == "cancelled"
    assert manifest_a.workers["w_ca"].stop_reason == "cancelled_by_user"

    # Sub-case B: Budget halt (cancel_event set without user cancel) during stalled_after_writes precheck
    reg_b = RunRegistry()
    turn_b = 0

    def handler_b(request: httpx.Request) -> httpx.Response:
        nonlocal turn_b
        if request.url.path.endswith("/models"):
            return _models_response()
        turn_b += 1
        if turn_b == 1:
            tc = {
                "id": "cb1",
                "type": "function",
                "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/c2.py", "content": "Y = 2\n"})},
            }
        else:
            tc = {
                "id": f"cbr{turn_b}",
                "type": "function",
                "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/c2.py", "start_line": 1, "end_line": 10})},
            }
        return _assistant_tool_resp([tc])

    def fake_precheck_budget_halt(project_root: Path, _touched: list[str]) -> StaticCheckResult:
        active_id = reg_b.has_active_run(project_root)
        assert active_id is not None
        cancel_ev = reg_b.get_cancel_event(project_root, active_id)
        assert cancel_ev is not None

        async def _set_ev() -> None:
            cancel_ev.set()

        asyncio.run_coroutine_threadsafe(_set_ev(), loop).result(timeout=2.0)
        return StaticCheckResult(status="passed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_b)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=10,
            tasks=[WorkerTaskInput(worker_id="w_cb", folder="backend", goal="Stall while budget halts")],
        )
        with patch("overseer.worker.run_static_precheck", side_effect=fake_precheck_budget_halt):
            resp_b = await prepare_and_launch_run(req_b, reg_b, client, catalog, repo)
            manifest_b = await _wait_for_run(repo, resp_b.run_id, reg_b)

    assert manifest_b.status == "halted_budget"
    assert manifest_b.workers["w_cb"].status == "halted_budget"
    assert manifest_b.workers["w_cb"].stop_reason == "budget_halted"


async def test_empty_tool_turn_receives_step_budget_notices_and_zero_touch_max_steps_skips_precheck(
    worker_env: Path,
) -> None:
    repo = worker_env
    catalog = ModelCatalog()

    # Sub-case A: empty-tool turn on max_steps - 2 (step 2 of 4), then hit max_steps_reached with zero touched files
    captured_a: list[list[dict]] = []
    turn_a = 0

    def handler_a(request: httpx.Request) -> httpx.Response:
        nonlocal turn_a
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_a.append(body["messages"])
        turn_a += 1
        if turn_a == 1:
            return _assistant_tool_resp(
                [{"id": "e1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]
            )
        if turn_a == 2:
            # Empty tool call on steps_used == 2 == max_steps - 2
            return _assistant_tool_resp([], finish_reason="stop")
        if turn_a == 3:
            return _assistant_tool_resp(
                [{"id": "e3", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "backend"}'}}]
            )
        return _assistant_tool_resp(
            [{"id": "e4", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "frontend"}'}}]
        )

    reg_a = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_a)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=4,
            tasks=[WorkerTaskInput(worker_id="w_empty_2", folder="backend", goal="Empty turn on max-2 and zero-touch cap")],
        )
        resp_a = await prepare_and_launch_run(req_a, reg_a, client, catalog, repo)
        manifest_a = await _wait_for_run(repo, resp_a.run_id, reg_a)

    wa = manifest_a.workers["w_empty_2"]
    assert wa.status == "escalated"
    assert wa.stop_reason == "max_steps_reached"
    assert wa.steps_used == 4
    assert wa.created_files == {}
    assert wa.modified_files == {}
    assert wa.static_check.status == "not_run"
    # Turn 3 received the 2-step budget notice after the empty-tool Turn 2
    assert captured_a[2][-1]["role"] == "user"
    assert "2 steps remain" in captured_a[2][-1]["content"]
    # Turn 4 received FINAL STEP appended onto the Turn 3 tool message (no tool -> user alternation)
    assert captured_a[3][-1]["role"] == "tool"
    assert "FINAL STEP: 1 step remains" in captured_a[3][-1]["content"]
    for turn_msgs in captured_a:
        _assert_no_tool_to_user_alternation(turn_msgs)

    # Sub-case B: empty-tool turn (finish_reason="length") on max_steps - 1 (step 2 of 3)
    captured_b: list[list[dict]] = []
    turn_b = 0

    def handler_b(request: httpx.Request) -> httpx.Response:
        nonlocal turn_b
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_b.append(body["messages"])
        turn_b += 1
        if turn_b == 1:
            return _assistant_tool_resp(
                [{"id": "eb1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]
            )
        if turn_b == 2:
            return _assistant_tool_resp([], finish_reason="length")
        return _assistant_tool_resp(
            [{"id": "eb3", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Done on final step", "notes": []}'}}]
        )

    reg_b = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_b)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=3,
            tasks=[WorkerTaskInput(worker_id="w_empty_1", folder="backend", goal="Empty turn on max-1")],
        )
        resp_b = await prepare_and_launch_run(req_b, reg_b, client, catalog, repo)
        manifest_b = await _wait_for_run(repo, resp_b.run_id, reg_b)

    wb = manifest_b.workers["w_empty_1"]
    assert wb.status == "completed"
    assert wb.steps_used == 3
    assert "output token limit" in captured_b[2][-1]["content"]
    assert "FINAL STEP: 1 step remains" in captured_b[2][-1]["content"]
    for turn_msgs in captured_b:
        _assert_no_tool_to_user_alternation(turn_msgs)


async def test_modified_file_pagination_and_failed_finish_created_file_repair(worker_env: Path) -> None:
    repo = worker_env
    # Create a 1,000-line pre-existing file to verify multi-hunk edits across paginated slices
    big_lines = [f"VAR_{i} = {i}" for i in range(1, 1001)]
    (repo / "backend" / "large.py").write_text("\n".join(big_lines) + "\n", encoding="utf-8")

    turn = 0

    def handler_modified(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        turn += 1
        if turn == 1:
            tc = {"id": "m1", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/large.py", "start_line": 1, "end_line": 50})}}
        elif turn == 2:
            tc = {"id": "m2", "type": "function", "function": {"name": "replace_block", "arguments": json.dumps({"path": "backend/large.py", "target": "VAR_1 = 1", "replacement": "VAR_1 = 10"})}}
        elif turn == 3:
            tc = {"id": "m3", "type": "function", "function": {"name": "file_outline", "arguments": json.dumps({"path": "backend/large.py"})}}
        elif turn == 4:
            tc = {"id": "m4", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/large.py", "start_line": 301, "end_line": 600})}}
        elif turn == 5:
            tc = {"id": "m5", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/large.py", "start_line": 601, "end_line": 900})}}
        elif turn == 6:
            tc = {"id": "m6", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/large.py", "start_line": 901, "end_line": 1000})}}
        elif turn == 7:
            tc = {"id": "m7", "type": "function", "function": {"name": "replace_block", "arguments": json.dumps({"path": "backend/large.py", "target": "VAR_999 = 999", "replacement": "VAR_999 = 9990"})}}
        else:
            tc = {"id": "m8", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Edited both regions", "notes": []}'}}
        return _assistant_tool_resp([tc])

    catalog = ModelCatalog()
    reg_a = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_modified)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=15,
            tasks=[WorkerTaskInput(worker_id="w_mod", folder="backend", goal="Edit top and bottom", target_files=["backend/large.py"])],
        )
        resp_a = await prepare_and_launch_run(req_a, reg_a, client, catalog, repo)
        manifest_a = await _wait_for_run(repo, resp_a.run_id, reg_a)

    assert manifest_a.workers["w_mod"].status == "completed"
    assert manifest_a.workers["w_mod"].steps_used == 8

    # Part B: After finish() fails with syntax errors across 3 created files, reading all 3 created files
    # across 3 turns must NOT trigger redundant-read kill before repair.
    captured_b: list[list[dict]] = []
    turn_b = 0

    def handler_repair(request: httpx.Request) -> httpx.Response:
        nonlocal turn_b
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        captured_b.append(body["messages"])
        turn_b += 1
        if turn_b == 1:
            tcs = [
                {"id": "c1", "type": "function", "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/f1.py", "content": "def bad1(:\n    pass\n"})}},
                {"id": "c2", "type": "function", "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/f2.py", "content": "X = 2\n"})}},
                {"id": "c3", "type": "function", "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/f3.py", "content": "Y = 3\n"})}},
            ]
            return _assistant_tool_resp(tcs)
        if turn_b == 2:
            return _assistant_tool_resp([{"id": "fin_fail", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Try", "notes": []}'}}])
        if turn_b == 3:
            return _assistant_tool_resp([{"id": "r1", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/f1.py"})}}])
        if turn_b == 4:
            return _assistant_tool_resp([{"id": "r2", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/f2.py"})}}])
        if turn_b == 5:
            return _assistant_tool_resp([{"id": "r3", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/f3.py"})}}])
        if turn_b == 6:
            return _assistant_tool_resp([{"id": "r4", "type": "function", "function": {"name": "view_file", "arguments": json.dumps({"path": "backend/app.py"})}}])
        if turn_b == 7:
            return _assistant_tool_resp([{"id": "fix1", "type": "function", "function": {"name": "replace_block", "arguments": json.dumps({"path": "backend/f1.py", "target": "def bad1(:", "replacement": "def good1():"})}}])
        return _assistant_tool_resp([{"id": "fin_ok", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Fixed", "notes": []}'}}])

    reg_b = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_repair)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=15,
            tasks=[
                WorkerTaskInput(
                    worker_id="w_rep",
                    folder="backend",
                    goal="Create and repair",
                    target_files=["backend/f1.py", "backend/f2.py", "backend/f3.py"],
                )
            ],
        )
        resp_b = await prepare_and_launch_run(req_b, reg_b, client, catalog, repo)
        manifest_b = await _wait_for_run(repo, resp_b.run_id, reg_b)

    assert manifest_b.workers["w_rep"].status == "completed"
    # On Turn 7 (after 4 post-failure reads), nudge tells worker to fix static pre-check errors, NOT call finish() blindly
    turn7_tail = captured_b[6][-1]["content"]
    assert "Fix the reported static pre-check errors" in turn7_tail
    assert "Stop re-reading and call finish(summary, notes) now" not in turn7_tail


async def test_no_tool_calls_after_write_runs_precheck_and_respects_cancel(worker_env: Path) -> None:
    repo = worker_env
    catalog = ModelCatalog()
    turn = 0

    def handler_prose(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        if request.url.path.endswith("/models"):
            return _models_response()
        turn += 1
        if turn == 1:
            return _assistant_tool_resp(
                [{"id": "w1", "type": "function", "function": {"name": "write_new_file", "arguments": json.dumps({"path": "backend/ok.py", "content": "OK = 1\n"})}}]
            )
        return _assistant_tool_resp([], finish_reason="stop")

    reg = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_prose)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=10,
            tasks=[WorkerTaskInput(worker_id="w_prose", folder="backend", goal="Write then emit prose")],
        )
        resp = await prepare_and_launch_run(req, reg, client, catalog, repo)
        manifest = await _wait_for_run(repo, resp.run_id, reg)

    w = manifest.workers["w_prose"]
    assert w.status == "escalated"
    assert w.stop_reason == "no_tool_calls"
    assert w.static_check.status == "passed"


def test_prune_message_history_compacts_older_failed_writes_and_clamps_view_stub() -> None:
    active_view = "ACTIVE VIEW " + ("A" * 500)
    older_failed = "X" * 3000
    latest_failed = "Y" * 3000
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        _tool_call_msg("v1", "view_file", {"path": "oversized.py", "start_line": 1, "end_line": 1000}),
        {"role": "tool", "tool_call_id": "v1", "content": "OLD VIEW"},
        _tool_call_msg("v2", "view_file", {"path": "keep.py", "start_line": 1, "end_line": 50}),
        {"role": "tool", "tool_call_id": "v2", "content": active_view},
        _tool_call_msg("w_old_fail", "write_new_file", {"path": "exists.py", "content": older_failed}),
        {"role": "tool", "tool_call_id": "w_old_fail", "content": "Tool error (write_new_file): exists"},
        _tool_call_msg("w_new_fail", "write_new_file", {"path": "latest.py", "content": latest_failed}),
        {"role": "tool", "tool_call_id": "w_new_fail", "content": "Tool error (write_new_file): exists"},
    ]
    # Budget fits active_view + latest_failed once older_failed is compacted, without evicting v2
    pruned = prune_message_history(messages, keep_recent_tools=2, max_history_chars=5200)
    tool_by_id = {m["tool_call_id"]: m["content"] for m in pruned if m.get("role") == "tool"}
    args_by_id = {
        tc["id"]: json.loads(tc["function"]["arguments"])
        for m in pruned
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    # Older failed write compacted with schema-valid placeholder; latest failed write preserved; active view preserved
    assert args_by_id["w_old_fail"] == {"path": "exists.py", "content": "[failed write; compacted]"}
    WriteNewFileArgs.model_validate(args_by_id["w_old_fail"])
    assert args_by_id["w_new_fail"]["content"] == latest_failed
    assert tool_by_id["v2"] == active_view
    # Pruned view_file stub clamped end_line to start_line + 299 = 300 (not 1000)
    assert tool_by_id["v1"] == "[pruned view_file oversized.py lines 1-300]"


async def test_walk_and_list_block_junction_escapes_and_view_caps_large_files(
    worker_env: Path,
    tmp_path: Path,
) -> None:
    from overseer.worker import MAX_VIEW_FILE_BYTES, _find_file_sync, _list_dir_sync, _search_dir_sync
    from overseer.models import FindFileArgs, ListDirArgs, SearchDirArgs

    repo = worker_env
    outside_dir = tmp_path / "outside_junc_target"
    outside_dir.mkdir()
    (outside_dir / "leak.txt").write_text("SECRET_NEEDLE\n", encoding="utf-8")

    junc_dir = repo / "backend" / "escape_junc"
    junc_dir.mkdir()
    (junc_dir / "leak.txt").write_text("SECRET_NEEDLE\n", encoding="utf-8")

    orig_is_junction = getattr(Path, "is_junction", None)
    orig_resolve = Path.resolve

    with (
        patch.object(
            Path,
            "is_junction",
            lambda p: p.name == "escape_junc" or (orig_is_junction(p) if orig_is_junction else False),
            create=True,
        ),
        patch.object(
            Path,
            "resolve",
            lambda p, strict=False: outside_dir if p.name == "escape_junc" else orig_resolve(p, strict=strict),
        ),
    ):
        assert "SECRET_NEEDLE" not in _search_dir_sync(repo, SearchDirArgs(query="SECRET_NEEDLE", path="."))
        assert "escape_junc" not in _find_file_sync(repo, FindFileArgs(pattern="*leak*", path="."))
        assert "escape_junc/" not in _list_dir_sync(repo, ListDirArgs(path="backend"))

    huge_file = repo / "backend" / "giant.txt"
    huge_file.write_bytes(b"A" * (MAX_VIEW_FILE_BYTES + 16))
    with pytest.raises(Exception, match="exceeds"):
        _view_file_sync(repo, repo / "backend", ViewFileArgs(path="backend/giant.txt"))


async def test_base_url_https_policy_catalog_cache_key_and_free_fallback_filter(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.worker import get_openrouter_base_url

    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://evil.example/v1")
    with pytest.raises(ValueError, match="https://"):
        get_openrouter_base_url()

    monkeypatch.setenv("OVERSEER_ALLOW_INSECURE_BASE_URL", "1")
    assert get_openrouter_base_url() == "http://evil.example/v1"
    monkeypatch.delenv("OVERSEER_ALLOW_INSECURE_BASE_URL", raising=False)

    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:8080/v1")
    assert get_openrouter_base_url() == "http://localhost:8080/v1"

    hits = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return _models_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        catalog = ModelCatalog(ttl_sec=3600.0)
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy-a.example/v1")
        await catalog.ensure_loaded(client, "sk-same")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy-b.example/v1")
        await catalog.ensure_loaded(client, "sk-same")
        assert hits == 2

    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FREE_FALLBACK_MODELS", "paid/sneaky,model/fallback:free")
    chain = build_model_chain("free")
    assert chain == ["model/primary:free", "model/fallback:free"]


async def test_http_200_error_payload_cancel_during_retry_and_narrow_domain_packs(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = worker_env
    catalog = ModelCatalog()

    # 1. HTTP 200 containing error object without choices fails cleanly instead of empty turn
    def handler_200_err(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(200, json={"error": {"code": 400, "message": "invalid prompt sk-or-secret9999"}})

    reg_err = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_200_err)) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="hit 200 err")],
        )
        resp = await prepare_and_launch_run(req, reg_err, client, catalog, repo)
        m_err = await _wait_for_run(repo, resp.run_id, reg_err)
    assert m_err.status == "failed"
    assert m_err.workers["w1"].stop_reason == "openrouter_http_200_error"
    assert "sk-or-secret9999" not in (m_err.workers["w1"].escalation_reason or "")

    # 2. Retry wait wakes promptly on cross-process cancel.flag even without in-memory ev.set()
    reg_cancel = RunRegistry()
    hit_429 = asyncio.Event()

    def handler_429(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        hit_429.set()
        return httpx.Response(429, headers={"Retry-After": "10"}, text="rate limited")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_429)) as client:
        req2 = StartRunRequest(
            project_root=str(repo),
            max_429_wait_seconds=10,
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="cancel in backoff")],
        )
        resp2 = await prepare_and_launch_run(req2, reg_cancel, client, catalog, repo)
        await hit_429.wait()
        run_dir = get_project_data_dir(repo) / "runs" / resp2.run_id
        cancel_flag_path_for(run_dir).write_text("cancelled", encoding="utf-8")
        m_cancel = await asyncio.wait_for(_wait_for_run(repo, resp2.run_id, reg_cancel), timeout=1.5)
    assert m_cancel.status == "cancelled"

    # 3. Verify no hidden domain packs are injected regardless of keywords or extensions
    for wid, goal, targets in (
        ("w_cpp", "Fix memory buffer", ["backend/buf.cpp"]),
        ("w_lat", "Reduce API latency", ["backend/app.py"]),
        ("w_mat", "Multiply matrix tensors", ["backend/app.py"]),
        ("w_lua", "Roblox module", ["backend/x.luau"]),
    ):
        prompt = _build_worker_system_prompt(
            repo,
            WorkerTaskInput(worker_id=wid, folder="backend", goal=goal, target_files=targets),
            shared_contract=None,
        )
        assert "Domain pack (" not in prompt


async def test_in_flight_cancel_preserves_usage_malformed_choices_failover_and_truncated_view_pruning(
    worker_env: Path,
) -> None:
    repo = worker_env
    catalog = ModelCatalog()

    # 1. Malformed HTTP 200 {"choices": []} triggers failover instead of counting as an empty turn
    calls: list[str] = []

    def handler_bad_200(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body["model"])
        if body["model"] == "model/primary:free":
            return httpx.Response(200, json={"choices": []})
        return _assistant_tool_resp(
            [{"id": "f1", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "ok", "notes": []}'}}]
        )

    reg_a = RunRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_bad_200)) as client:
        req_a = StartRunRequest(
            project_root=str(repo),
            max_429_wait_seconds=0,
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="failover malformed 200")],
        )
        resp_a = await prepare_and_launch_run(req_a, reg_a, client, catalog, repo)
        m_a = await _wait_for_run(repo, resp_a.run_id, reg_a)
    assert m_a.status == "completed"
    assert m_a.workers["w1"].models_used == ["model/fallback:free"]

    # 2. Simultaneous cancel + HTTP response completion still persists billed usage.cost into manifest
    reg_b = RunRegistry()
    in_post = asyncio.Event()
    allow_return = asyncio.Event()
    run_id_box: list[str] = []

    async def handler_in_flight_cancel(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        in_post.set()
        await allow_return.wait()
        ev = reg_b.get_cancel_event(repo, run_id_box[0])
        assert ev is not None
        reg_b.mark_user_cancelled(repo, run_id_box[0])
        ev.set()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 50, "completion_tokens": 25, "cost": 0.12},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "w", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_in_flight_cancel)) as client:
        req_b = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="record cost on cancel")],
        )
        resp_b = await prepare_and_launch_run(req_b, reg_b, client, catalog, repo)
        run_id_box.append(resp_b.run_id)
        await in_post.wait()
        allow_return.set()
        m_b = await _wait_for_run(repo, resp_b.run_id, reg_b)
    assert m_b.status == "cancelled"
    assert m_b.spend_usd == pytest.approx(0.12)

    # 3. Byte-truncated view_file (lines 1..40) must NOT subsume older view_file (lines 100..200)
    msgs = [
        {"role": "system", "content": "sys"},
        _tool_call_msg("old_v", "view_file", {"path": "big.py", "start_line": 100, "end_line": 200}),
        {"role": "tool", "tool_call_id": "old_v", "content": "LINES 100..200"},
        _tool_call_msg("expl", "list_dir", {"path": "."}),
        {"role": "tool", "tool_call_id": "expl", "content": "DIR"},
        _tool_call_msg("cap_v", "view_file", {"path": "big.py", "start_line": 1, "end_line": 300}),
        {
            "role": "tool",
            "tool_call_id": "cap_v",
            "content": "# big.py (lines 1..40 of 500)\n[Output capped at 65536 bytes]\n1: x\n",
        },
        _tool_call_msg("fin", "finish", {"summary": "done", "notes": []}),
        {"role": "tool", "tool_call_id": "fin", "content": "ok"},
    ]
    pruned = prune_message_history(msgs, keep_recent_tools=3)
    by_id = {m["tool_call_id"]: m["content"] for m in pruned if m.get("role") == "tool"}
    assert by_id["old_v"] == "LINES 100..200"
    assert by_id["expl"].startswith("[pruned list_dir")


async def test_round2_ignored_case_dir_target_primary_delist_http_cancel_flag_and_shielded_manifest(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.guardrails import PathSecurityError
    from overseer.worker import _call_openrouter_with_failover, _WorkerCancelledError, run_worker_loop

    repo = worker_env
    (repo / "backend" / "Node_Modules").mkdir(parents=True)
    (repo / "backend" / "Node_Modules" / "dep.js").write_text("console.log('hit')\n", encoding="utf-8")
    (repo / "backend" / "sub").mkdir()

    ctx_ro = WorkerToolContext(
        project_root=repo,
        folder_abs=repo / "backend",
        backups_dir=repo / ".b",
        run_dir=repo / ".r",
        manifest_path=repo / ".r" / "m.json",
        manifest=RunManifest(
            version="0.1.0",
            run_id="run_20261122_000030_010203",
            server_pid=1,
            created_at_iso=utc_now_iso(),
            project_root=str(repo),
            status="running",
            limits=RunLimits(),
            patch_dir=str(repo / ".r"),
            workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="g", requested_model="m:free")},
        ),
        worker=WorkerManifestState(worker_id="w1", folder="backend", goal="g", requested_model="m:free"),
        run_lock=asyncio.Lock(),
    )
    list_res, _ = await execute_worker_tool("list_dir", {"path": "backend"}, ctx_ro)
    assert "Node_Modules/" not in list_res
    search_res, _ = await execute_worker_tool("search_dir", {"query": "hit", "path": "backend"}, ctx_ro)
    assert "dep.js" not in search_res

    # 1. Existing directory rejected in target_files; default-tier fallback promoted when primary delisted
    catalog = ModelCatalog()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200,
                json={"data": [{"id": "model/fallback:free", "supported_parameters": ["tools"]}]},
            )
        )
    ) as client:
        req_dir = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="dir target", target_files=["backend/sub"])],
        )
        with pytest.raises(PathSecurityError, match="not a directory"):
            await prepare_and_launch_run(req_dir, RunRegistry(), client, catalog, repo)

        await catalog.ensure_loaded(client, "sk-or-test")
        chain = catalog.filter_supported_chain(["model/primary:free", "model/fallback:free"], "https://openrouter.ai/api/v1", allow_primary_fallback=True)
        assert chain == ["model/fallback:free"]
        with pytest.raises(ValueError, match="not found"):
            catalog.filter_supported_chain(["model/primary:free", "model/fallback:free"], "https://openrouter.ai/api/v1", allow_primary_fallback=False)

    # 2. Cross-process cancel.flag aborts a hung HTTP client.post() in < 1.0s and cross-tier key fallback is blocked
    flag_path = repo / "cancel.flag"
    in_hung_post = asyncio.Event()

    async def hung_handler(_req: httpx.Request) -> httpx.Response:
        in_hung_post.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(hung_handler)) as hung_client:
        call_task = asyncio.create_task(
            _call_openrouter_with_failover(
                client=hung_client,
                api_key="sk-or-free",
                model_chain=["model/primary:free"],
                start_model_idx=0,
                messages=[{"role": "user", "content": "hi"}],
                limits=RunLimits(request_timeout_sec=30.0),
                cancel_flag_path=flag_path,
            )
        )
        await in_hung_post.wait()
        flag_path.write_text("1", encoding="utf-8")
        with pytest.raises(_WorkerCancelledError):
            await asyncio.wait_for(call_task, timeout=1.5)

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_CHEAP_API_KEY", raising=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(hung_handler)) as dummy_client:
        with pytest.raises(ValueError, match="OPENROUTER_CHEAP_API_KEY"):
            await _call_openrouter_with_failover(
                client=dummy_client,
                api_key="sk-or-free-only",
                model_chain=["model/primary:free", "deepseek/deepseek-v4.1-flash-1017"],
                start_model_idx=1,
                messages=[{"role": "user", "content": "hi"}],
                limits=RunLimits(),
            )

    # 3. Shielded manifest writes finish before coordinator overwrites on cancel_active_run
    reg_s = RunRegistry()
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000031_fedcba"
    m_path = manifest_path_for(proj_data, run_id)
    write_entered = asyncio.Event()
    release_write = asyncio.Event()

    async def slow_atomic(path: Path, data: bytes) -> None:
        if path == m_path:
            write_entered.set()
            await release_write.wait()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    m = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo),
        status="running",
        limits=RunLimits(),
        patch_dir=str(m_path.parent),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="g", requested_model="m:free")},
    )
    cancel_ev = reg_s.register_run(repo, run_id)
    with patch("overseer.worker.atomic_write_bytes", side_effect=slow_atomic):
        w_task = asyncio.create_task(
            run_worker_loop(
                task=WorkerTaskInput(worker_id="w1", folder="backend", goal="g"),
                folder_abs=repo / "backend",
                manifest=m,
                manifest_path=m_path,
                backups_dir=proj_data / "backups" / run_id,
                run_dir=m_path.parent,
                registry=reg_s,
                client=httpx.AsyncClient(),
                catalog=ModelCatalog(),
                api_key="sk-or-free",
                semaphore=reg_s.worker_semaphore,
                cancel_event=cancel_ev,
                model_chain=["m:free"],
            )
        )
        await write_entered.wait()
        cancel_ev.set()
        w_task.cancel()
        await asyncio.gather(w_task, return_exceptions=True)
        assert len(reg_s._commit_tasks[(reg_s._project_key(repo), run_id)]) == 1
        release_write.set()
        await reg_s.await_commits(repo, run_id)
    assert m_path.exists()
    reg_s.mark_run_finished(repo, run_id)


async def test_round3_toast_cancel_kill_finish_redaction_failed_write_cleanup_and_catalog_refresh(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from overseer.notify import send_windows_toast
    from overseer.worker import _handle_finish, _handle_write_new_file

    repo = worker_env
    # 1. Cancelled toast task kills & reaps subprocess cleanly
    if sys.platform == "win32":
        monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)

        class _HungProc:
            def __init__(self) -> None:
                self.killed = False
                self._done = asyncio.Event()

            async def wait(self) -> int:
                await self._done.wait()
                return -9

            def kill(self) -> None:
                self.killed = True
                self._done.set()

        fake_proc = _HungProc()
        with patch("overseer.notify.asyncio.create_subprocess_exec", return_value=fake_proc):
            t = asyncio.create_task(send_windows_toast("t", "m"))
            await asyncio.sleep(0.02)
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t
        assert fake_proc.killed is True

    # 2. Failed write_new_file removes newly created empty dirs & finish() redacts summary/notes
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000040_a1b2c3"
    m_path = manifest_path_for(proj_data, run_id)
    m = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=1,
        created_at_iso=utc_now_iso(),
        project_root=str(repo),
        status="running",
        limits=RunLimits(),
        patch_dir=str(m_path.parent),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="g", requested_model="m:free")},
    )
    ctx = WorkerToolContext(
        project_root=repo,
        folder_abs=(repo / "backend").resolve(),
        backups_dir=proj_data / "backups" / run_id,
        run_dir=m_path.parent,
        manifest_path=m_path,
        manifest=m,
        worker=m.workers["w1"],
        run_lock=asyncio.Lock(),
    )
    orig_write = __import__("overseer.worker", fromlist=["atomic_write_bytes"]).atomic_write_bytes

    async def fail_target_write(path: Path, data: bytes) -> None:
        if path.name == "leaf.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            raise OSError("disk full")
        await orig_write(path, data)

    with patch("overseer.worker.atomic_write_bytes", side_effect=fail_target_write):
        with pytest.raises(OSError, match="disk full"):
            await _handle_write_new_file(WriteNewFileArgs(path="backend/orphan_a/orphan_b/leaf.py", content="x = 1\n"), ctx)
    assert not (repo / "backend" / "orphan_a").exists()

    from overseer.models import FinishArgs

    out, ok = await _handle_finish(
        FinishArgs(summary="Used Bearer topsecret123", notes=["Check sk-or-mysecret9999"]),
        ctx,
    )
    assert ok and "completed" in out
    assert "topsecret123" not in ctx.worker.summary
    assert "sk-or-mysecret9999" not in ctx.worker.notes[0]

    # 3. ModelCatalog refresh drops delisted models rather than retaining stale entries
    catalog = ModelCatalog(ttl_sec=0.0)
    responses = iter(
        [
            {"data": [{"id": "model/old:free", "supported_parameters": ["tools"]}]},
            {"data": [{"id": "model/new:free", "supported_parameters": ["tools"]}]},
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=next(responses)))
    ) as client:
        await catalog.ensure_loaded(client, "sk-k1", force=True)
        catalog.validate_tool_support(["model/old:free"])
        await catalog.ensure_loaded(client, "sk-k1", force=True)
        with pytest.raises(ValueError, match="not found"):
            catalog.validate_tool_support(["model/old:free"])
        catalog.validate_tool_support(["model/new:free"])


async def test_http_200_invalid_json_fails_over(worker_env: Path) -> None:
    from overseer.worker import _call_openrouter_with_failover

    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_models.append(body["model"])
        if body["model"] == "model/primary:free":
            return httpx.Response(200, content=b"<html>proxy error</html>", headers={"content-type": "text/html"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, used_model, _ = await _call_openrouter_with_failover(
            client=client,
            api_key="sk-test",
            model_chain=["model/primary:free", "model/fallback:free"],
            start_model_idx=0,
            messages=[{"role": "user", "content": "hi"}],
            limits=RunLimits(max_429_wait_seconds=0),
            cancel_event=asyncio.Event(),
        )
    assert used_model == "model/fallback:free"
    assert data["choices"][0]["message"]["content"] == "ok"
    assert "model/fallback:free" in seen_models


async def test_v3_config_precedence_heavy_caps_and_provider_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.config import (
        get_cheap_default_model,
        get_cheap_fallback_models,
        get_default_model,
        get_heavy_default_model,
        get_heavy_fallback_models,
        resolve_effective_max_run_usd,
        save_config_file_sync,
    )
    from overseer.models import OverseerConfigFile, ProviderRoutingRule, TierConfig
    from overseer.openrouter import _call_openrouter_with_failover

    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    for k in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
        "OVERSEER_DEFAULT_MODEL",
        "OVERSEER_FREE_DEFAULT_MODEL",
        "OVERSEER_FALLBACK_MODELS",
        "OVERSEER_FREE_FALLBACK_MODELS",
        "OVERSEER_CHEAP_DEFAULT_MODEL",
        "OVERSEER_CHEAP_FALLBACK_MODELS",
        "OVERSEER_HEAVY_DEFAULT_MODEL",
        "OVERSEER_HEAVY_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(k, raising=False)

    (data_dir / ".env").write_text(
        "\n".join(
            [
                "OPENROUTER_FREE_API_KEY=sk-free",
                "OPENROUTER_CHEAP_API_KEY=sk-cheap",
                "OVERSEER_CHEAP_DEFAULT_MODEL=dotenv/cheap-primary",
                "OVERSEER_CHEAP_FALLBACK_MODELS=dotenv/cheap-fb",
            ]
        ),
        encoding="utf-8",
    )

    # 1. Heavy unconfigured initially; key fallback cheap -> shared when heavy key absent
    assert get_heavy_default_model() is None
    assert get_heavy_fallback_models() == []
    with pytest.raises(ValueError, match="Heavy tier is not configured"):
        build_model_chain("heavy")
    assert select_api_key_for_model("anthropic/claude-sonnet-4.6", tier="heavy") == "sk-cheap"
    assert select_api_key_for_model("some/model:free", tier="heavy") == "sk-free"

    monkeypatch.setenv("OPENROUTER_HEAVY_API_KEY", "sk-heavy")
    assert select_api_key_for_model("anthropic/claude-sonnet-4.6", tier="heavy") == "sk-heavy"

    # 2. Partial config.json update (cap only) preserves .env primary & fallbacks
    save_config_file_sync(
        OverseerConfigFile(
            tiers={
                "cheap": TierConfig(max_run_usd=0.60),
                "heavy": TierConfig(primary="cfg/heavy-a", fallbacks=["cfg/heavy-b"], max_run_usd=3.50),
            },
            provider_tiers={
                "cheap": ProviderRoutingRule(sort="price", allow_fallbacks=True),
                "heavy": ProviderRoutingRule(sort="latency", order=["Anthropic", "Google"]),
            },
            provider_models={
                "cfg/heavy-a": ProviderRoutingRule(only=["Anthropic"], allow_fallbacks=False),
            },
        )
    )
    assert get_cheap_default_model() == "dotenv/cheap-primary"
    assert get_cheap_fallback_models() == ["dotenv/cheap-fb"]
    assert build_model_chain("heavy") == ["cfg/heavy-a", "cfg/heavy-b"]

    # 3. Precedence: os.environ > config.json > .env > defaults
    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "env/free-top:free")
    assert get_default_model() == "env/free-top:free"
    monkeypatch.setenv("OVERSEER_CHEAP_DEFAULT_MODEL", "env/cheap-top")
    assert get_cheap_default_model() == "env/cheap-top"
    assert get_heavy_default_model() == "cfg/heavy-a"
    assert get_heavy_fallback_models() == ["cfg/heavy-b"]

    # 4. Corrupt config.json degrades safely without crashing
    (data_dir / "config.json").write_text("{not-valid-json", encoding="utf-8")
    assert get_heavy_default_model() is None

    # Restore valid config for cap + provider routing verification
    save_config_file_sync(
        OverseerConfigFile(
            tiers={
                "cheap": TierConfig(max_run_usd=0.60),
                "heavy": TierConfig(primary="cfg/heavy-a", fallbacks=["cfg/heavy-b"], max_run_usd=3.50),
            },
            provider_tiers={
                "cheap": ProviderRoutingRule(sort="price", allow_fallbacks=True),
                "heavy": ProviderRoutingRule(sort="latency", order=["Anthropic", "Google"]),
            },
            provider_models={
                "cfg/heavy-a": ProviderRoutingRule(only=["Anthropic"], allow_fallbacks=False),
            },
        )
    )
    assert resolve_effective_max_run_usd(["free"], None) == 0.25
    assert resolve_effective_max_run_usd(["cheap"], None) == 0.60
    assert resolve_effective_max_run_usd(["cheap", "heavy"], None) == 3.50
    assert resolve_effective_max_run_usd(["heavy"], 1.25) == 1.25

    # 5. Provider routing injection (model override > tier default > omitted when unset)
    captured_bodies: list[dict[str, object]] = []

    def routing_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
        if body["model"] == "cfg/heavy-a":
            return httpx.Response(404, json={"error": {"message": "failover test"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(routing_handler)) as client:
        # Heavy chain: cfg/heavy-a uses model override, then fails over to cfg/heavy-b using tier default
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-heavy",
            model_chain=["cfg/heavy-a", "cfg/heavy-b"],
            start_model_idx=0,
            messages=[{"role": "user", "content": "hi"}],
            limits=RunLimits(max_429_wait_seconds=0),
            tier="heavy",
        )
        # Free tier has no routing rule -> "provider" key omitted
        await _call_openrouter_with_failover(
            client=client,
            api_key="sk-free",
            model_chain=["particle-ai/particle-1:free"],
            start_model_idx=0,
            messages=[{"role": "user", "content": "hi"}],
            limits=RunLimits(max_429_wait_seconds=0),
            tier="free",
        )

    assert captured_bodies[0]["provider"] == {"only": ["Anthropic"], "allow_fallbacks": False}
    assert captured_bodies[1]["provider"] == {"sort": "latency", "order": ["Anthropic", "Google"]}
    assert "provider" not in captured_bodies[2]


async def test_audit_cancel_event_aborts_inflight_http_dotenv_toast_and_coordinator_cleanup(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import notify
    from overseer.openrouter import _post_with_cancel_flag_poll, _WorkerCancelledError
    from overseer.state import active_lock_path_for

    repo = worker_env
    data_dir = get_project_data_dir(repo).parent.parent

    # 1. _post_with_cancel_flag_poll aborts immediately when cancel_event fires
    started_post = asyncio.Event()
    never_set = asyncio.Event()

    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        started_post.set()
        await never_set.wait()
        return httpx.Response(200, json={})

    cancel_ev = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_handler)) as client:
        post_coro = _post_with_cancel_flag_poll(
            client=client,
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test"},
            body={"model": "m:free", "messages": []},
            timeout=30.0,
            cancel_flag_path=None,
            cancel_event=cancel_ev,
        )
        task = asyncio.create_task(post_coro)
        await started_post.wait()
        cancel_ev.set()
        with pytest.raises(_WorkerCancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    # 2. OVERSEER_DISABLE_TOASTS=true in <data_dir>/.env suppresses auto toasts and send_windows_toast
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".env").write_text("OVERSEER_DISABLE_TOASTS=true\n", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
        mp.delenv("PYTEST_CURRENT_TEST", raising=False)
        mp.setattr("overseer.notify.sys.platform", "win32")
        assert notify.should_suppress_auto_toast() is True
        assert await notify.send_windows_toast("Title", "Body") is False

    # 3. _coordinator releases active.lock and registry state even if final write_manifest_locked raises OSError
    def ok_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.0},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_err",
                                    "type": "function",
                                    "function": {"name": "finish", "arguments": '{"summary": "ok", "notes": []}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    import overseer.worker as worker_mod

    orig_write_locked = worker_mod.write_manifest_locked
    fail_next_final = True

    async def flaky_write_locked(m_path: Path, manifest: RunManifest) -> None:
        nonlocal fail_next_final
        if manifest.completed_at_iso is not None and fail_next_final:
            fail_next_final = False
            raise OSError("disk full on final manifest write")
        await orig_write_locked(m_path, manifest)

    registry = RunRegistry()
    catalog = ModelCatalog()
    with patch("overseer.worker.write_manifest_locked", side_effect=flaky_write_locked):
        async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler)) as client:
            req = StartRunRequest(
                project_root=str(repo),
                tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="cleanup check")],
            )
            resp = await prepare_and_launch_run(req, registry, client, catalog, repo)
            coord = registry.get_coordinator(repo, resp.run_id)
            assert coord is not None
            await asyncio.gather(coord, return_exceptions=True)

    assert registry.has_active_run(repo) is None
    assert not active_lock_path_for(get_project_data_dir(repo)).exists()


async def test_free_tier_refuses_paid_model_key(
    worker_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.guardrails import get_data_dir

    repo = worker_env
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".env").write_text("OPENROUTER_CHEAP_API_KEY=sk-or-paid-only-123456\n", encoding="utf-8")
    monkeypatch.setenv("OVERSEER_FREE_DEFAULT_MODEL", "openai/gpt-4o")

    with pytest.raises(ValueError, match=":free"):
        select_api_key_for_model("openai/gpt-4o", tier="free")

    registry = RunRegistry()
    catalog = ModelCatalog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(404))) as client:
        req = StartRunRequest(
            project_root=str(repo),
            tasks=[WorkerTaskInput(worker_id="w1", folder="backend", goal="should fail", model="free")],
        )
        with pytest.raises(ValueError, match=":free"):
            await prepare_and_launch_run(req, registry, client, catalog, repo)










