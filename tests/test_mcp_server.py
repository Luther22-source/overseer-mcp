"""Stage 0 & Stage 2 integration tests for server.py MCP tools, stdout isolation, and multi-project routing."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from overseer.guardrails import get_project_data_dir
from overseer.models import StartRunRequest, WorkerTaskInput
from overseer.server import APP_URI, OverseerServerState, configure_logging, create_server
from overseer.state import load_manifest_sync, manifest_path_for


def _models_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "model/primary:free",
                    "supported_parameters": ["tools"],
                    "pricing": {"prompt": "0", "completion": "0"},
                }
            ]
        },
    )


def _extract_tool_payload(call_result: Any) -> dict[str, Any]:
    if hasattr(call_result, "structuredContent") and call_result.structuredContent:
        return dict(call_result.structuredContent)
    if isinstance(call_result, dict):
        return call_result
    for block in getattr(call_result, "content", []):
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise AssertionError(f"Unexpected call_tool result shape: {call_result!r}")


async def test_configure_logging_writes_external_and_protects_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "overseer_appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    fake_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    log_path = configure_logging()
    assert log_path == data_dir / "logs" / "overseer.log"
    assert log_path.exists()

    repo = tmp_path / "repo_a"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            # Stray print during MCP start_run handler execution must redirect to sys.stderr
            print("stray print that would corrupt JSON-RPC")
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
                                    "id": "fin",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"summary": "Checked src", "notes": ["Also update README"]}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        tools = await mcp.list_tools()
        assert {t.name for t in tools} == {
            "start_run",
            "get_report",
            "cancel_run",
            "rollback",
            "get_config",
            "update_config",
            "list_models",
            "open_overseer_monitor",
            "app_list_runs",
            "app_get_run_snapshot",
            "app_cancel_run",
            "app_get_custom_instructions",
            "app_set_custom_instructions",
            "app_get_config",
            "app_update_config",
            "app_list_models",
            "app_rollback_run",
            "app_get_worker_patch",
        }

        res = await mcp.call_tool(
            "start_run",
            {
                "project_root": str(repo),
                "tasks": [{"worker_id": "w1", "folder": "src", "goal": "check"}],
            },
        )
        payload = _extract_tool_payload(res)
        assert payload["status"] == "running"
        while state.registry.is_run_active(repo, payload["run_id"]):
            await asyncio.sleep(0.01)

        rep_res = await mcp.call_tool("get_report", {"project_root": str(repo), "run_id": "latest"})
        rep_payload = _extract_tool_payload(rep_res)
        assert rep_payload["status"] == "completed"
        assert rep_payload["workers"][0]["notes"] == ["Also update README"]
        assert len(json.dumps(rep_payload).encode("utf-8")) <= 8192

        assert fake_stdout.getvalue() == ""
        assert not any(repo.rglob(".overseer"))
        assert not any(repo.rglob("__pycache__"))
    finally:
        await state.aclose()




async def test_multi_project_switching_same_project_guard_and_e2e_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "overseer_appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    with pytest.raises(ValidationError, match="Duplicate worker_id"):
        StartRunRequest(
            project_root=str(tmp_path),
            tasks=[
                WorkerTaskInput(worker_id="dup", folder="a", goal="g1"),
                WorkerTaskInput(worker_id="dup", folder="b", goal="g2"),
            ],
        )

    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    (repo_a / "pkg").mkdir(parents=True)
    (repo_b / "pkg").mkdir(parents=True)
    target_a = repo_a / "pkg" / "mod.py"
    target_a.write_text("X = 1\n", encoding="utf-8")
    (repo_b / "pkg" / "mod.py").write_text("Y = 1\n", encoding="utf-8")

    gate_a = asyncio.Event()
    turn_a = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn_a
        if request.url.path.endswith("/models"):
            return _models_response()

        body = json.loads(request.content.decode("utf-8"))
        sys_prompt = body["messages"][0]["content"]
        if repo_a.name in sys_prompt:
            await gate_a.wait()
            turn_a += 1
            if turn_a == 1:
                tool_name, args = "view_file", {"path": "pkg/mod.py"}
            elif turn_a == 2:
                tool_name, args = "replace_block", {"path": "pkg/mod.py", "target": "X = 1", "replacement": "X = 2"}
            else:
                tool_name, args = "finish", {"summary": "Updated X", "notes": []}
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
                                    {"id": f"a_{turn_a}", "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
                                ],
                            }
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
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "b_1", "type": "function", "function": {"name": "finish", "arguments": '{"summary": "Repo B ok", "notes": []}'}}
                            ],
                        }
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        run_a = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {"project_root": str(repo_a), "tasks": [{"worker_id": "wa", "folder": "pkg", "goal": "update X"}]},
            )
        )

        # Switching freely to repo_b while repo_a is running succeeds
        run_b = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {"project_root": str(repo_b), "tasks": [{"worker_id": "wb", "folder": "pkg", "goal": "check Y"}]},
            )
        )
        assert run_b["project_root"] == str(repo_b.resolve())

        # Concurrent run on the same repo_a must be rejected cleanly
        with pytest.raises(Exception, match="already running"):
            await mcp.call_tool(
                "start_run",
                {"project_root": str(repo_a), "tasks": [{"worker_id": "wa2", "folder": "pkg", "goal": "overlap"}]},
            )

        gate_a.set()
        while state.registry.is_run_active(repo_a, run_a["run_id"]) or state.registry.is_run_active(repo_b, run_b["run_id"]):
            await asyncio.sleep(0.01)

        assert target_a.read_text(encoding="utf-8") == "X = 2\n"
        rep_a = _extract_tool_payload(await mcp.call_tool("get_report", {"project_root": str(repo_a), "run_id": run_a["run_id"]}))
        assert rep_a["workers"][0]["patch_path"] is not None
        assert Path(rep_a["workers"][0]["patch_path"]).exists()

        rb_a = _extract_tool_payload(await mcp.call_tool("rollback", {"project_root": str(repo_a), "run_id": run_a["run_id"]}))
        assert rb_a["restored_files"] == ["pkg/mod.py"]
        assert target_a.read_text(encoding="utf-8") == "X = 1\n"
        assert not any(repo_a.rglob(".overseer"))
        assert not any(repo_b.rglob(".overseer"))

        with pytest.raises(Exception, match="No runs found|run_id|not found"):
            await mcp.call_tool("get_report", {"project_root": str(repo_a), "run_id": "run_20260101_000000_000000"})
        with pytest.raises(Exception, match="No runs found|run_id|not found"):
            await mcp.call_tool("rollback", {"project_root": str(repo_a), "run_id": "run_20260101_000000_000000"})
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_default_server_lifespan_closes_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(tmp_path / "appdata"))
    mcp_default, default_state = create_server()
    lowlevel = mcp_default._lowlevel_server
    async with lowlevel.lifespan(lowlevel):
        assert default_state is not None


@pytest.mark.asyncio
async def test_cancel_run_stops_active_and_queued_workers_and_allows_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    repo = tmp_path / "repo"
    for folder in ("d1", "d2", "d3", "d4", "d5"):
        (repo / folder).mkdir(parents=True)
    target = repo / "d1" / "f.py"
    target.write_text("VAL = 1\n", encoding="utf-8")

    never_set = asyncio.Event()
    w1_turn = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal w1_turn
        if request.url.path.endswith("/models"):
            return _models_response()

        body = json.loads(request.content.decode("utf-8"))
        sys_prompt = body["messages"][0]["content"]
        if "folder jail: d1" in sys_prompt:
            w1_turn += 1
            if w1_turn == 1:
                tname, args = "view_file", {"path": "d1/f.py"}
            elif w1_turn == 2:
                tname, args = "replace_block", {"path": "d1/f.py", "target": "VAL = 1", "replacement": "VAL = 2"}
            else:
                tname, args = "finish", {"summary": "d1 done", "notes": []}
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
                                    {"id": f"c_{w1_turn}", "type": "function", "function": {"name": tname, "arguments": json.dumps(args)}}
                                ],
                            }
                        }
                    ],
                },
            )

        await never_set.wait()
        raise AssertionError("Should be cancelled before unblocking")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        start_payload = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo),
                    "tasks": [
                        {"worker_id": f"w{i}", "folder": f"d{i}", "goal": f"edit d{i}"}
                        for i in range(1, 6)
                    ],
                },
            )
        )
        run_id = start_payload["run_id"]

        for _ in range(100):
            rep_mid = _extract_tool_payload(await mcp.call_tool("get_report", {"project_root": str(repo), "run_id": run_id}))
            w1_status = next(w["status"] for w in rep_mid["workers"] if w["worker_id"] == "w1")
            if w1_status == "completed":
                break
            await asyncio.sleep(0.01)
        assert w1_status == "completed"

        cancel_payload = _extract_tool_payload(
            await mcp.call_tool("cancel_run", {"project_root": str(repo), "run_id": "latest"})
        )
        assert cancel_payload["status"] == "cancelled"
        assert cancel_payload["already_finished"] is False
        assert cancel_payload["cancelled_workers"] == ["w2", "w3", "w4", "w5"]
        assert not state.registry.is_run_active(repo, run_id)

        cancel_again = _extract_tool_payload(
            await mcp.call_tool("cancel_run", {"project_root": str(repo), "run_id": run_id})
        )
        assert cancel_again["already_finished"] is True
        assert cancel_again["status"] == "cancelled"
        assert cancel_again["cancelled_workers"] == []

        rep_final = _extract_tool_payload(await mcp.call_tool("get_report", {"project_root": str(repo), "run_id": run_id}))
        assert rep_final["status"] == "cancelled"

        rb_payload = _extract_tool_payload(await mcp.call_tool("rollback", {"project_root": str(repo), "run_id": run_id}))
        assert rb_payload["restored_files"] == ["d1/f.py"]
        assert target.read_text(encoding="utf-8") == "VAL = 1\n"

        with pytest.raises(Exception, match="No runs found|run_id|not found"):
            await mcp.call_tool("cancel_run", {"project_root": str(repo), "run_id": "run_20260101_000000_000000"})
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_server_shutdown_marks_active_run_interrupted_not_halted_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    repo = tmp_path / "repo"
    (repo / "d1").mkdir(parents=True)
    (repo / "d2").mkdir(parents=True)
    (repo / "d1" / "f.py").write_text("X = 1\n", encoding="utf-8")

    entered_post = asyncio.Event()
    never_set = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        entered_post.set()
        await never_set.wait()
        raise AssertionError("Should be cancelled by server shutdown")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    mcp, _ = create_server(state=state)
    start_payload = _extract_tool_payload(
        await mcp.call_tool(
            "start_run",
            {
                "project_root": str(repo),
                "tasks": [
                    {"worker_id": "w1", "folder": "d1", "goal": "edit d1"},
                    {"worker_id": "w2", "folder": "d2", "goal": "edit d2"},
                ],
            },
        )
    )
    run_id = start_payload["run_id"]
    await entered_post.wait()

    # Simulate MCP server shutdown while run is active
    await state.aclose()

    m_path = manifest_path_for(get_project_data_dir(repo), run_id)
    manifest = load_manifest_sync(m_path)
    assert manifest.status == "interrupted"
    assert manifest.completed_at_iso is not None
    assert all(w.status == "interrupted" for w in manifest.workers.values())
    assert all(w.stop_reason == "server_shutdown" for w in manifest.workers.values())

    # Fresh server instance reports interrupted (not halted_budget) and allows rollback
    fresh_state = OverseerServerState(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        fresh_mcp, _ = create_server(state=fresh_state)
        rep = _extract_tool_payload(await fresh_mcp.call_tool("get_report", {"project_root": str(repo), "run_id": run_id}))
        assert rep["status"] == "interrupted"
        rb = _extract_tool_payload(await fresh_mcp.call_tool("rollback", {"project_root": str(repo), "run_id": run_id}))
        assert rb["status"] == "rolled_back"
    finally:
        await fresh_state.aclose()


@pytest.mark.asyncio
async def test_server_converts_oserror_to_tool_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)

    state = OverseerServerState(client=httpx.AsyncClient())
    try:
        mcp, _ = create_server(state=state)
        with patch("overseer.server.prepare_and_launch_run", side_effect=PermissionError("start locked")):
            with pytest.raises(Exception, match="start locked"):
                await mcp.call_tool(
                    "start_run",
                    {"project_root": str(repo), "tasks": [{"worker_id": "w1", "folder": "backend", "goal": "g"}]},
                )
        with patch("overseer.server.collect_run_report", side_effect=PermissionError("disk locked")):
            with pytest.raises(Exception, match="disk locked"):
                await mcp.call_tool("get_report", {"project_root": str(repo), "run_id": "latest"})
        with patch("overseer.server.cancel_active_run", side_effect=PermissionError("cancel locked")):
            with pytest.raises(Exception, match="cancel locked"):
                await mcp.call_tool("cancel_run", {"project_root": str(repo), "run_id": "latest"})
        with (
            patch("overseer.server.validate_and_resolve_run_id", return_value="run_20260101_000000_aaaaaa"),
            patch("overseer.server.rollback_run", side_effect=PermissionError("rollback locked")),
        ):
            with pytest.raises(Exception, match="rollback locked"):
                await mcp.call_tool("rollback", {"project_root": str(repo), "run_id": "latest"})
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_shutdown_after_budget_halt_preserves_halted_budget_and_terminates_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    repo = tmp_path / "repo"
    (repo / "d1").mkdir(parents=True)
    (repo / "d2").mkdir(parents=True)

    w1_done_turn = asyncio.Event()
    w2_blocked = asyncio.Event()
    never_set = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        body = json.loads(request.content.decode("utf-8"))
        sys_prompt = body["messages"][0]["content"]
        if "folder jail: d1" in sys_prompt:
            # Exceed max_run_usd=0.25 on first response after w2 is blocked in-flight
            if not w1_done_turn.is_set():
                await w2_blocked.wait()
                w1_done_turn.set()
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
                                        {
                                            "id": "b1",
                                            "type": "function",
                                            "function": {"name": "list_dir", "arguments": '{"path": "."}'},
                                        }
                                    ],
                                }
                            }
                        ],
                    },
                )
            await never_set.wait()

        w2_blocked.set()
        await never_set.wait()
        raise AssertionError("Should be cancelled")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    mcp, _ = create_server(state=state)
    start_payload = _extract_tool_payload(
        await mcp.call_tool(
            "start_run",
            {
                "project_root": str(repo),
                "max_run_usd": 0.25,
                "tasks": [
                    {"worker_id": "w1", "folder": "d1", "goal": "exceed budget"},
                    {"worker_id": "w2", "folder": "d2", "goal": "blocked sibling"},
                ],
            },
        )
    )
    run_id = start_payload["run_id"]
    await w1_done_turn.wait()
    await w2_blocked.wait()
    # Allow w1 to process its 0.30 cost and set cancel_event before shutdown_all runs
    cancel_ev = state.registry.get_cancel_event(repo, run_id)
    assert cancel_ev is not None
    await cancel_ev.wait()

    await state.aclose()

    m_path = manifest_path_for(get_project_data_dir(repo), run_id)
    manifest = load_manifest_sync(m_path)
    assert manifest.status == "halted_budget"
    assert all(w.status == "halted_budget" for w in manifest.workers.values())
    assert all(w.stop_reason == "budget_halted" for w in manifest.workers.values())


@pytest.mark.asyncio
async def test_concurrent_cancel_active_run_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.state import cancel_active_run

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    repo = tmp_path / "repo"
    (repo / "d1").mkdir(parents=True)
    entered = asyncio.Event()
    never_set = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        entered.set()
        await never_set.wait()
        raise AssertionError("Should be cancelled")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        start_payload = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo),
                    "tasks": [{"worker_id": "w1", "folder": "d1", "goal": "wait"}],
                },
            )
        )
        run_id = start_payload["run_id"]
        await entered.wait()

        r1, r2 = await asyncio.gather(
            cancel_active_run(repo, run_id, state.registry),
            cancel_active_run(repo, run_id, state.registry),
        )
        assert sorted([r1.already_finished, r2.already_finished]) == [False, True]
        assert r1.status == "cancelled"
        assert r2.status == "cancelled"
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_cross_process_active_lock_cancel_flag_and_global_semaphore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.state import (
        RunRegistry,
        RunStateError,
        acquire_project_active_lock_sync,
        cancel_active_run,
        release_project_active_lock_sync,
    )

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    (repo_a / "d1").mkdir(parents=True)
    (repo_b / "d1").mkdir(parents=True)
    proj_a_data = get_project_data_dir(repo_a)

    # 1) Cross-process lock blocks live foreign PID and reclaims dead PID
    acquire_project_active_lock_sync(proj_a_data, "run_20261116_000001_aaaaaa", pid=424242)
    with patch("overseer.state.is_pid_alive", return_value=True):
        with pytest.raises(RunStateError, match="PID 424242"):
            acquire_project_active_lock_sync(proj_a_data, "run_20261116_000002_bbbbbb", pid=111111)
    with patch("overseer.state.is_pid_alive", return_value=False):
        acquire_project_active_lock_sync(proj_a_data, "run_20261116_000002_bbbbbb", pid=111111)
    release_project_active_lock_sync(proj_a_data, "run_20261116_000002_bbbbbb")

    # 2) Global semaphore bounds concurrency to 3 workers across repo_a + repo_b, and cross-process cancel.flag stops run
    in_flight = 0
    max_in_flight = 0
    first_turn_done = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        if request.url.path.endswith("/models"):
            return _models_response()
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.05)
            first_turn_done.set()
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
                                    {"id": "l1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}
                                ],
                            }
                        }
                    ],
                },
            )
        finally:
            in_flight -= 1

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)
        (repo_a / "d2").mkdir(parents=True)
        (repo_b / "d2").mkdir(parents=True)
        r_a = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo_a),
                    "tasks": [
                        {"worker_id": "a1", "folder": "d1", "goal": "g1"},
                        {"worker_id": "a2", "folder": "d2", "goal": "g2"},
                    ],
                },
            )
        )
        r_b = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo_b),
                    "tasks": [
                        {"worker_id": "b1", "folder": "d1", "goal": "g3"},
                        {"worker_id": "b2", "folder": "d2", "goal": "g4"},
                    ],
                },
            )
        )
        await first_turn_done.wait()
        await asyncio.sleep(0.1)
        assert max_in_flight <= 3

        # Simulate a second MCP stdio process invoking cancel_run on repo_a via cancel.flag
        foreign_registry = RunRegistry()
        with patch("overseer.state.os.getpid", return_value=777_777), patch(
            "overseer.state.is_pid_alive", return_value=True
        ):
            cancel_resp = await cancel_active_run(repo_a, r_a["run_id"], foreign_registry)
        assert cancel_resp.already_finished is False

        # Wait until repo_a completes cancellation via cancel.flag
        for _ in range(100):
            if not state.registry.is_run_active(repo_a, r_a["run_id"]):
                break
            await asyncio.sleep(0.02)
        m_a = load_manifest_sync(manifest_path_for(proj_a_data, r_a["run_id"]))
        assert m_a.status == "cancelled"

        # Stop repo_b cleanly before teardown
        await cancel_active_run(repo_b, r_b["run_id"], state.registry)
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_mcp_app_resource_metadata_monitor_entrypoint_and_app_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import re

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OVERSEER_DEFAULT_MODEL", "model/primary:free")
    monkeypatch.setenv("OVERSEER_FALLBACK_MODELS", "")

    fake_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    repo = tmp_path / "repo_app"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("X = 1\n", encoding="utf-8")

    entered_run = asyncio.Event()
    never_set = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response()
        entered_run.set()
        await never_set.wait()
        raise AssertionError("Should be cancelled by app_cancel_run")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)

        # 1. Verify MCP App resource metadata and self-contained HTML bundle
        resources = await mcp.list_resources()
        app_resources = [r for r in resources if str(r.uri) == APP_URI]
        assert len(app_resources) == 1
        app_res = app_resources[0]
        assert app_res.mime_type == "text/html;profile=mcp-app"
        res_dump = app_res.model_dump(by_alias=True)
        assert res_dump["_meta"] == {
            "ui": {
                "prefersBorder": False,
                "csp": {
                    "connectDomains": [],
                    "resourceDomains": [],
                },
            }
        }

        contents = await mcp.read_resource(APP_URI)
        assert len(contents) == 1
        html_text = contents[0].content
        assert contents[0].mime_type == "text/html;profile=mcp-app"
        html_bytes = html_text.encode("utf-8")
        assert 0 < len(html_bytes) < 10 * 1024 * 1024
        assert len(html_bytes) < 500 * 1024
        assert "innerHTML" not in html_text
        assert not re.findall(r"<(?:script|link|img|iframe)[^>]*(?:src|href)=[^>]+>", html_text, re.I)

        # 2. Verify tool metadata: open_overseer_monitor entrypoints & app_* visibility=["app"]
        tools = {t.name: t for t in await mcp.list_tools()}
        mon_meta = tools["open_overseer_monitor"].model_dump(by_alias=True)["_meta"]
        assert mon_meta["ui"]["resourceUri"] == APP_URI
        assert mon_meta["com.google.antigravity/ui"]["entrypoints"] == [
            {"type": "aux_pane", "title": "Overseer"},
            {"type": "global", "title": "Overseer"},
        ]
        for app_tool in (
            "app_list_runs",
            "app_get_run_snapshot",
            "app_cancel_run",
            "app_get_custom_instructions",
            "app_set_custom_instructions",
            "app_get_config",
            "app_update_config",
        ):
            assert tools[app_tool].model_dump(by_alias=True)["_meta"] == {"ui": {"visibility": ["app"]}}

        # 3. Empty-args {} open_overseer_monitor works cleanly even before any runs exist
        empty_mon = _extract_tool_payload(await mcp.call_tool("open_overseer_monitor", {}))
        assert empty_mon["project_root"] is None
        assert empty_mon["runs"] == []
        assert empty_mon["active_count"] == 0
        assert empty_mon["selected_run"] is None
        assert "custom_instructions" not in empty_mon
        assert "config" not in empty_mon

        app_cfg = _extract_tool_payload(await mcp.call_tool("app_get_config", {}))
        assert app_cfg["tiers"]["heavy"]["primary"] is None
        assert app_cfg["tiers"]["cheap"]["max_run_usd"] == 0.25

        # Verify app_set_custom_instructions / app_get_custom_instructions round-trip
        saved_ci = _extract_tool_payload(
            await mcp.call_tool("app_set_custom_instructions", {"instructions": "Always use strict types"})
        )
        assert saved_ci == {"custom_instructions": "Always use strict types"}
        loaded_ci = _extract_tool_payload(await mcp.call_tool("app_get_custom_instructions", {}))
        assert loaded_ci == {"custom_instructions": "Always use strict types"}
        cleared_ci = _extract_tool_payload(
            await mcp.call_tool("app_set_custom_instructions", {"instructions": ""})
        )
        assert cleared_ci == {"custom_instructions": ""}

        empty_scoped_mon = _extract_tool_payload(
            await mcp.call_tool("open_overseer_monitor", {"project_root": str(repo)})
        )
        assert empty_scoped_mon["project_root"] == str(repo.resolve())
        assert empty_scoped_mon["runs"] == []
        assert empty_scoped_mon["selected_run"] is None

        # 4. Launch an active run and exercise open_overseer_monitor, app_list_runs, app_get_run_snapshot, app_cancel_run
        started = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo),
                    "tasks": [{"worker_id": "w1", "folder": "src", "goal": "run monitor test"}],
                },
            )
        )
        run_id = started["run_id"]
        await entered_run.wait()

        mon_all = _extract_tool_payload(await mcp.call_tool("open_overseer_monitor", {}))
        assert mon_all["active_count"] == 1
        assert mon_all["selected_run"] is not None
        assert mon_all["selected_run"]["run_id"] == run_id
        assert mon_all["selected_run"]["status"] == "running"

        mon_scoped = _extract_tool_payload(
            await mcp.call_tool("open_overseer_monitor", {"project_root": str(repo)})
        )
        assert mon_scoped["project_root"] == str(repo.resolve())
        assert len(mon_scoped["runs"]) == 1
        assert mon_scoped["selected_run"]["run_id"] == run_id

        listed = _extract_tool_payload(await mcp.call_tool("app_list_runs", {}))
        assert listed["active_count"] == 1
        assert [r["run_id"] for r in listed["runs"]] == [run_id]

        snap = _extract_tool_payload(
            await mcp.call_tool("app_get_run_snapshot", {"project_root": str(repo), "run_id": "latest"})
        )
        assert snap["run_id"] == run_id
        assert snap["status"] == "running"
        assert snap["workers"][0]["worker_id"] == "w1"

        cancelled = _extract_tool_payload(
            await mcp.call_tool("app_cancel_run", {"project_root": str(repo), "run_id": run_id})
        )
        assert cancelled["status"] == "cancelled"
        assert cancelled["already_finished"] is False

        snap_after = _extract_tool_payload(
            await mcp.call_tool("app_get_run_snapshot", {"project_root": str(repo), "run_id": run_id})
        )
        assert snap_after["status"] == "cancelled"

        # Missing run raises ToolError cleanly
        with pytest.raises(Exception, match="not found|No runs found|run_id"):
            await mcp.call_tool(
                "app_get_run_snapshot",
                {"project_root": str(repo), "run_id": "run_20260101_000000_000000"},
            )

        # OVERSEER_APP_HTML override path works and missing override raises FileNotFoundError
        custom_html = tmp_path / "custom_app.html"
        custom_html.write_text("<!DOCTYPE html><html><body>Custom</body></html>", encoding="utf-8")
        monkeypatch.setenv("OVERSEER_APP_HTML", str(custom_html))
        override_contents = await mcp.read_resource(APP_URI)
        assert "Custom" in override_contents[0].content

        monkeypatch.setenv("OVERSEER_APP_HTML", str(tmp_path / "missing_app.html"))
        with pytest.raises(Exception, match="Error reading resource"):
            await mcp.read_resource(APP_URI)

        # Zero stdout pollution across all resource reads and tool calls
        assert fake_stdout.getvalue() == ""
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_start_run_bounds_secret_redaction_app_html_validation_and_toast_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer.worker import _BG_TOAST_TASKS

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)

    with pytest.raises(ValidationError):
        StartRunRequest(
            project_root=str(repo),
            max_run_usd=100.0,
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="too pricey")],
        )
    with pytest.raises(ValidationError):
        StartRunRequest(
            project_root=str(repo),
            max_steps_per_worker=500,
            tasks=[WorkerTaskInput(worker_id="w1", folder="src", goal="too many steps")],
        )

    state = OverseerServerState(client=httpx.AsyncClient())
    toast_cancelled = False

    async def slow_toast() -> None:
        nonlocal toast_cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            toast_cancelled = True
            raise

    bg_task = asyncio.create_task(slow_toast())
    _BG_TOAST_TASKS.add(bg_task)
    bg_task.add_done_callback(_BG_TOAST_TASKS.discard)
    await asyncio.sleep(0)

    try:
        mcp, _ = create_server(state=state)
        with patch(
            "overseer.server.prepare_and_launch_run",
            side_effect=httpx.ConnectError("Authorization Bearer sk-or-leak1234567890 failed"),
        ):
            with pytest.raises(Exception) as exc_info:
                await mcp.call_tool(
                    "start_run",
                    {"project_root": str(repo), "tasks": [{"worker_id": "w1", "folder": "src", "goal": "g"}]},
                )
            assert "sk-or-leak1234567890" not in str(exc_info.value)

        bad_ext = tmp_path / "not_html.txt"
        bad_ext.write_text("hi", encoding="utf-8")
        monkeypatch.setenv("OVERSEER_APP_HTML", str(bad_ext))
        with pytest.raises(Exception, match="Error reading resource"):
            await mcp.read_resource(APP_URI)
    finally:
        await state.aclose()


    assert toast_cancelled is True or bg_task.done()


@pytest.mark.asyncio
async def test_log_redaction_filter_and_deleted_project_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging
    import shutil

    from overseer.models import RunLimits, RunManifest, WorkerManifestState
    from overseer.state import save_manifest_atomic, utc_now_iso, write_latest_run_pointer

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    log_path = configure_logging()
    logging.getLogger("overseer.test").warning("leaked Authorization: Bearer secret123 and sk-or-verysecret99")
    for handler in logging.getLogger().handlers:
        handler.flush()
    log_text = log_path.read_text(encoding="utf-8")
    assert "secret123" not in log_text
    assert "sk-or-verysecret99" not in log_text
    assert "[REDACTED]" in log_text

    # Historical snapshot lookup still succeeds after original repo folder is deleted
    repo = tmp_path / "deleted_repo"
    (repo / "src").mkdir(parents=True)
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000010_abcdef"
    state = OverseerServerState(client=httpx.AsyncClient())
    try:
        m = RunManifest(
            version="0.1.0",
            run_id=run_id,
            server_pid=999_999_999,
            created_at_iso=utc_now_iso(),
            project_root=str(repo.resolve()),
            status="completed",
            limits=RunLimits(),
            patch_dir=str(proj_data / "runs" / run_id),
            workers={"w1": WorkerManifestState(worker_id="w1", folder="src", goal="g", requested_model="m:free", status="completed")},
        )
        await save_manifest_atomic(manifest_path_for(proj_data, run_id), m, state.registry.get_lock(repo, run_id))
        await write_latest_run_pointer(proj_data, run_id)
        shutil.rmtree(repo)

        mcp, _ = create_server(state=state)
        snap = _extract_tool_payload(
            await mcp.call_tool("app_get_run_snapshot", {"project_root": str(repo), "run_id": run_id})
        )
        assert snap["run_id"] == run_id
        assert snap["status"] == "completed"
    finally:
        await state.aclose()




@pytest.mark.asyncio
async def test_monitor_finds_older_run_beyond_20_and_deleted_repo_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    from overseer.models import RunLimits, RunManifest, WorkerManifestState
    from overseer.state import save_manifest_atomic, utc_now_iso, write_latest_run_pointer

    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    repo = tmp_path / "hist_repo"
    (repo / "src").mkdir(parents=True)
    proj_data = get_project_data_dir(repo)
    state = OverseerServerState(client=httpx.AsyncClient())
    try:
        older_run = "run_20261101_000001_000001"
        for i in range(1, 23):
            rid = older_run if i == 1 else f"run_20261102_0000{i:02d}_{i:06x}"
            m = RunManifest(
                version="0.1.0",
                run_id=rid,
                server_pid=999_999_999,
                created_at_iso=f"2026-11-02T00:{i:02d}:00+00:00",
                project_root=str(repo.resolve()),
                status="completed",
                limits=RunLimits(),
                patch_dir=str(proj_data / "runs" / rid),
                workers={"w1": WorkerManifestState(worker_id="w1", folder="src", goal="g", requested_model="m:free", status="completed")},
            )
            await save_manifest_atomic(manifest_path_for(proj_data, rid), m, state.registry.get_lock(repo, rid))
        await write_latest_run_pointer(proj_data, "run_20261102_000022_000016")

        mcp, _ = create_server(state=state)
        mon_old = _extract_tool_payload(
            await mcp.call_tool("open_overseer_monitor", {"run_id": older_run})
        )
        assert mon_old["selected_run"] is not None
        assert mon_old["selected_run"]["run_id"] == older_run

        shutil.rmtree(repo)
        mon_deleted = _extract_tool_payload(
            await mcp.call_tool("open_overseer_monitor", {"project_root": str(repo)})
        )
        assert mon_deleted["selected_run"] is not None
        listed_deleted = _extract_tool_payload(
            await mcp.call_tool("app_list_runs", {"project_root": str(repo), "limit": 5})
        )
        assert len(listed_deleted["runs"]) == 5
    finally:
        await state.aclose()


@pytest.mark.asyncio
async def test_round3_stdio_subprocess_framing_refcounted_stdout_and_compact_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import sys

    from overseer import __version__
    from overseer.models import RunLimits, RunManifest, WorkerManifestState
    from overseer.server import _redirect_stdout_to_stderr
    from overseer.state import save_manifest_atomic, utc_now_iso

    # 1. Refcounted stdout redirection preserves outer fake_stdout across interleaved blocks
    fake_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    async def overlapped(tag: str, delay: float) -> None:
        with _redirect_stdout_to_stderr():
            print(f"inside-{tag}")
            await asyncio.sleep(delay)

    await asyncio.gather(overlapped("a", 0.04), overlapped("b", 0.01))
    assert sys.stdout is fake_stdout
    assert fake_stdout.getvalue() == ""

    # 2. Compact open_overseer_monitor stays <= ~8.5 KB even with large worker summaries
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    repo = tmp_path / "big_summary_repo"
    (repo / "src").mkdir(parents=True)
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000050_112233"
    state = OverseerServerState(client=httpx.AsyncClient())
    try:
        m = RunManifest(
            version="0.1.0",
            run_id=run_id,
            server_pid=999_999_999,
            created_at_iso=utc_now_iso(),
            project_root=str(repo.resolve()),
            status="completed",
            limits=RunLimits(),
            patch_dir=str(proj_data / "runs" / run_id),
            workers={
                f"w{i}": WorkerManifestState(
                    worker_id=f"w{i}",
                    folder="src",
                    goal="g",
                    requested_model="m:free",
                    status="completed",
                    summary="S" * 1900,
                    notes=["N" * 250] * 10,
                )
                for i in range(6)
            },
        )
        await save_manifest_atomic(manifest_path_for(proj_data, run_id), m, state.registry.get_lock(repo, run_id))
        mcp, _ = create_server(state=state)
        mon = _extract_tool_payload(await mcp.call_tool("open_overseer_monitor", {"project_root": str(repo)}))
        assert mon["selected_run"] is not None
        assert len(json.dumps(mon["selected_run"], separators=(",", ":")).encode("utf-8")) <= 8192
    finally:
        await state.aclose()

    # 3. Live stdio subprocess framing check (python -m overseer.server)
    env = os.environ.copy()
    env["OVERSEER_DATA_DIR"] = str(tmp_path / "stdio_appdata")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "overseer.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    try:
        init_req = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0.1"},
                },
            }
        )
        proc.stdin.write((init_req + "\n").encode("utf-8"))
        await proc.stdin.drain()

        parsed_msgs: list[dict[str, Any]] = []
        while not any(m.get("id") == 1 for m in parsed_msgs):
            raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=8.0)
            assert raw_line, "MCP stdio server exited before responding to initialize"
            line_str = raw_line.decode("utf-8").strip()
            if line_str:
                parsed_msgs.append(json.loads(line_str))

        follow_reqs = (
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            + "\n"
        )
        proc.stdin.write(follow_reqs.encode("utf-8"))
        await proc.stdin.drain()

        while not any(m.get("id") == 2 for m in parsed_msgs):
            raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=8.0)
            assert raw_line, "MCP stdio server exited before responding to tools/list"
            line_str = raw_line.decode("utf-8").strip()
            if line_str:
                parsed_msgs.append(json.loads(line_str))

        proc.stdin.write_eof()
        rem_out, err_bytes = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        for trailing in rem_out.decode("utf-8").splitlines():
            if trailing.strip():
                parsed_msgs.append(json.loads(trailing))
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    assert proc.returncode == 0, err_bytes.decode("utf-8", errors="replace")
    init_resp = next(msg for msg in parsed_msgs if msg.get("id") == 1)
    assert init_resp["result"]["serverInfo"]["name"] == "overseer"
    assert init_resp["result"]["serverInfo"]["version"] == __version__
    tool_names = {t["name"] for msg in parsed_msgs if msg.get("id") == 2 for t in msg["result"]["tools"]}
    assert {"start_run", "get_report", "cancel_run", "rollback", "open_overseer_monitor"} <= tool_names


def test_configure_logging_falls_back_when_mkdir_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", _boom)
    log_path = configure_logging(data_dir=tmp_path / "ro_appdata")
    assert log_path == tmp_path / "ro_appdata" / "logs" / "overseer.log"
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_safe_rotating_file_handler_rollover_cooldown_and_max_app_html_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging
    import logging.handlers
    from overseer.server import _SafeRotatingFileHandler, _load_app_html

    log_file = tmp_path / "test.log"
    handler = _SafeRotatingFileHandler(log_file, maxBytes=10, backupCount=1, encoding="utf-8")
    try:
        assert handler.stream is not None
        handler.stream.write("01234567890123456789")
        handler.stream.flush()

        now = [100.0]
        monkeypatch.setattr("overseer.server.time.monotonic", lambda: now[0])

        def _locked_rollover(_self: logging.handlers.RotatingFileHandler) -> None:
            raise OSError("WinError 32: file locked")

        monkeypatch.setattr(logging.handlers.RotatingFileHandler, "doRollover", _locked_rollover)
        handler.doRollover()

        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        assert handler.shouldRollover(record) == 0

        now[0] += 61.0
        assert handler.shouldRollover(record) == 1
    finally:
        handler.close()

    html_file = tmp_path / "oversized.html"
    html_file.write_text("<html><body>too big</body></html>", encoding="utf-8")
    monkeypatch.setenv("OVERSEER_APP_HTML", str(html_file))
    monkeypatch.setattr("overseer.server.MAX_APP_HTML_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        _load_app_html()


async def test_send_windows_toast_strips_nul_and_catches_kill_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import notify

    captured_env: dict[str, str] = {}

    class _FakeProc:
        async def wait(self) -> int:
            raise PermissionError("process already exiting")

        def kill(self) -> None:
            raise PermissionError("access denied on kill")

    async def _fake_exec(*_args: Any, **kwargs: Any) -> _FakeProc:
        captured_env.update(kwargs["env"])
        return _FakeProc()

    async def _fake_wait_for(awaitable: Any, timeout: float) -> Any:
        _ = timeout
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError("timed out")

    monkeypatch.setattr("overseer.notify.sys.platform", "win32")
    monkeypatch.delenv("OVERSEER_DISABLE_TOASTS", raising=False)
    monkeypatch.setattr("overseer.notify.asyncio.create_subprocess_exec", _fake_exec)
    monkeypatch.setattr("overseer.notify.asyncio.wait_for", _fake_wait_for)

    ok = await notify.send_windows_toast("Ti\x00tle", "Ms\x00g", timeout_sec=0.1)
    assert ok is False
    assert captured_env["OVERSEER_TOAST_TITLE"] == "Title"
    assert captured_env["OVERSEER_TOAST_BODY"] == "Msg"


async def test_start_run_tool_schema_exposes_field_bounds() -> None:
    state = OverseerServerState(client=httpx.AsyncClient())
    try:
        mcp, _ = create_server(state=state)
        tools = {t.name: t for t in await mcp.list_tools()}
        schema = tools["start_run"].input_schema
        props = schema["properties"]
        assert props["project_root"]["minLength"] == 1
        assert props["tasks"]["minItems"] == 1
        assert props["tasks"]["maxItems"] == 20
        max_usd_schema = props["max_run_usd"]
        num_schema = (
            next(item for item in max_usd_schema["anyOf"] if item.get("type") == "number")
            if "anyOf" in max_usd_schema
            else max_usd_schema
        )
        assert num_schema["exclusiveMinimum"] == 0
        assert num_schema["maximum"] == 50.0
        assert props["max_steps_per_worker"]["minimum"] == 1
        assert props["max_steps_per_worker"]["maximum"] == 200
    finally:
        await state.aclose()


async def test_v3_get_and_update_config_validation_and_secret_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True)
    (data_dir / ".env").write_text(
        "\n".join(
            [
                "OPENROUTER_FREE_API_KEY=sk-free-secret-1234",
                "OPENROUTER_CHEAP_API_KEY=sk-cheap-secret-5678",
                "OPENROUTER_HEAVY_API_KEY=sk-heavy-secret-9012",
            ]
        ),
        encoding="utf-8",
    )
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

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")

    posted_bodies: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "particle-ai/particle-1:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        },
                        {
                            "id": "qwen/qwen3-coder:free",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0", "completion": "0"},
                        },
                        {
                            "id": "no/tools-model:free",
                            "supported_parameters": [],
                            "pricing": {"prompt": "0", "completion": "0"},
                        },
                        {
                            "id": "anthropic/claude-sonnet-4.6",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                        },
                        {
                            "id": "openai/gpt-5.2",
                            "supported_parameters": ["tools"],
                            "pricing": {"prompt": "0.000002", "completion": "0.000010"},
                        },
                    ]
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        posted_bodies.append((auth, body))
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "fin_1",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"summary": "Done", "notes": []}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = OverseerServerState(client=client)
    try:
        mcp, _ = create_server(state=state)

        initial = _extract_tool_payload(await mcp.call_tool("get_config", {}))
        assert initial["tiers"]["heavy"]["primary"] is None
        assert initial["tiers"]["heavy"]["max_run_usd"] == 2.0
        assert initial["tiers"]["cheap"]["max_run_usd"] == 0.25
        assert initial["api_keys_configured"] == {
            "free": True,
            "cheap": True,
            "heavy": True,
            "shared": False,
        }
        assert "sk-" not in json.dumps(initial)

        # Validation failures must reject before persisting config.json
        with pytest.raises(ToolError, match="must end with ':free'"):
            await mcp.call_tool("update_config", {"tiers": {"free": {"primary": "openai/gpt-5.2"}}})
        with pytest.raises(ToolError, match="does not advertise 'tools'"):
            await mcp.call_tool("update_config", {"tiers": {"free": {"primary": "no/tools-model:free"}}})
        with pytest.raises(ToolError, match="not found"):
            await mcp.call_tool("update_config", {"tiers": {"heavy": {"primary": "missing/model"}}})
        with pytest.raises(ToolError, match="free"):
            await mcp.call_tool("update_config", {"tiers": {"free": {"max_run_usd": 1.0}}})
        assert not (data_dir / "config.json").exists()

        # Valid update via Boss tool + App tool
        updated = _extract_tool_payload(
            await mcp.call_tool(
                "update_config",
                {
                    "tiers": {
                        "free": {"primary": "qwen/qwen3-coder:free"},
                        "heavy": {
                            "primary": "anthropic/claude-sonnet-4.6",
                            "fallbacks": ["openai/gpt-5.2"],
                            "max_run_usd": 4.5,
                        },
                    },
                    "provider_routing": {
                        "tiers": {"heavy": {"sort": "latency", "allow_fallbacks": True}},
                        "models": {
                            "anthropic/claude-sonnet-4.6": {
                                "only": ["Anthropic"],
                                "allow_fallbacks": False,
                            }
                        },
                    },
                },
            )
        )
        assert updated["tiers"]["free"]["primary"] == "qwen/qwen3-coder:free"
        assert updated["tiers"]["heavy"] == {
            "primary": "anthropic/claude-sonnet-4.6",
            "fallbacks": ["openai/gpt-5.2"],
            "max_run_usd": 4.5,
        }
        assert updated["provider_routing"]["models"]["anthropic/claude-sonnet-4.6"]["only"] == ["Anthropic"]
        assert (data_dir / "config.json").is_file()

        # Unknown provider routing fields must fail rather than silently clearing the existing rule
        with pytest.raises(ToolError):
            await mcp.call_tool("update_config", {"provider_routing": {"tiers": {"heavy": {"providers": ["Anthropic"]}}}})
        still_intact = _extract_tool_payload(await mcp.call_tool("get_config", {}))
        assert still_intact["provider_routing"]["tiers"]["heavy"]["sort"] == "latency"

        # App tool mirrors effective config and can clear/replace overrides
        app_cfg = _extract_tool_payload(await mcp.call_tool("app_get_config", {}))
        assert app_cfg == updated
        assert "sk-" not in json.dumps(app_cfg)

        # Launch a heavy run without explicit max_run_usd -> uses 4.5 cap, heavy key, and model provider routing
        started = _extract_tool_payload(
            await mcp.call_tool(
                "start_run",
                {
                    "project_root": str(repo),
                    "tasks": [{"worker_id": "hw", "folder": "src", "goal": "inspect", "model": "heavy"}],
                },
            )
        )
        run_id = started["run_id"]
        for _ in range(50):
            rep = _extract_tool_payload(await mcp.call_tool("get_report", {"project_root": str(repo), "run_id": run_id}))
            if rep["status"] != "running":
                break
            await asyncio.sleep(0.05)
        assert rep["status"] == "completed"
        manifest = load_manifest_sync(manifest_path_for(get_project_data_dir(repo), run_id))
        assert manifest.limits.max_run_usd == 4.5
        assert len(posted_bodies) == 1
        auth_header, req_body = posted_bodies[0]
        assert auth_header == "Bearer sk-heavy-secret-9012"
        assert req_body["model"] == "anthropic/claude-sonnet-4.6"
        assert req_body["provider"] == {"only": ["Anthropic"], "allow_fallbacks": False}

        # Clear heavy primary via app_update_config and replace model overrides
        cleared = _extract_tool_payload(
            await mcp.call_tool(
                "app_update_config",
                {
                    "tiers": {"heavy": {"primary": None}},
                    "provider_routing": {"replace_models": True, "models": {}},
                },
            )
        )
        assert cleared["tiers"]["heavy"]["primary"] is None
        assert cleared["provider_routing"]["models"] == {}

        # Corrupt config.json must fail loudly on update instead of overwriting the broken file
        (data_dir / "config.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(ToolError, match="Cannot update Overseer config.json"):
            await mcp.call_tool("update_config", {"tiers": {"free": {"primary": "particle-ai/particle-1:free"}}})
        assert (data_dir / "config.json").read_text(encoding="utf-8") == "{broken"

        # Single-key user (only OPENROUTER_FREE_API_KEY) can save a free-tier update cleanly
        (data_dir / "config.json").unlink()
        (data_dir / ".env").write_text("OPENROUTER_FREE_API_KEY=sk-free-secret-1234\n", encoding="utf-8")
        free_only = _extract_tool_payload(
            await mcp.call_tool(
                "update_config",
                {"tiers": {"free": {"primary": "qwen/qwen3-coder:free"}}},
            )
        )
        assert free_only["tiers"]["free"]["primary"] == "qwen/qwen3-coder:free"
        assert free_only["api_keys_configured"] == {
            "free": True,
            "cheap": False,
            "heavy": False,
            "shared": False,
        }
    finally:
        await state.aclose()


async def test_audit_concurrent_update_config_and_dotenv_app_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True)
    custom_html = tmp_path / "env_app.html"
    custom_html.write_text("<!DOCTYPE html><html><body>EnvApp</body></html>", encoding="utf-8")
    (data_dir / ".env").write_text(
        "\n".join(
            [
                "OPENROUTER_FREE_API_KEY=sk-free-secret-1234",
                "OPENROUTER_CHEAP_API_KEY=sk-cheap-secret-5678",
                f"OVERSEER_APP_HTML={custom_html}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OVERSEER_APP_HTML", raising=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            await asyncio.sleep(0.03)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "qwen/qwen3-coder:free", "supported_parameters": ["tools"], "pricing": {"prompt": "0", "completion": "0"}},
                        {"id": "openai/gpt-5.2", "supported_parameters": ["tools"], "pricing": {"prompt": "0.000002", "completion": "0.000010"}},
                    ]
                },
            )
        return httpx.Response(404)

    state = OverseerServerState(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        mcp, _ = create_server(state=state)

        # OVERSEER_APP_HTML from <data_dir>/.env is honored
        contents = await mcp.read_resource(APP_URI)
        assert "EnvApp" in contents[0].content

        # Concurrent update_config calls must serialize and preserve both updates
        await asyncio.gather(
            mcp.call_tool("update_config", {"tiers": {"free": {"primary": "qwen/qwen3-coder:free"}}}),
            mcp.call_tool("update_config", {"tiers": {"cheap": {"primary": "openai/gpt-5.2"}}}),
        )
        final_cfg = _extract_tool_payload(await mcp.call_tool("get_config", {}))
        assert final_cfg["tiers"]["free"]["primary"] == "qwen/qwen3-coder:free"
        assert final_cfg["tiers"]["cheap"]["primary"] == "openai/gpt-5.2"
    finally:
        await state.aclose()


async def test_audit2_provider_models_cap_and_data_dir_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import time
    from mcp.server.mcpserver.exceptions import ToolError
    from overseer.config import data_dir_file_lock

    data_dir = tmp_path / "appdata"
    data_dir.mkdir(parents=True)
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))

    # Stale lock file is reclaimed cleanly
    lock_path = data_dir / ".config.lock"
    lock_path.write_text("999999", encoding="utf-8")
    stale_time = time.time() - 30.0
    os.utime(lock_path, (stale_time, stale_time))
    with data_dir_file_lock("config", timeout_sec=1.0, stale_sec=5.0):
        assert lock_path.exists()
    assert not lock_path.exists()

    state = OverseerServerState(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(404)))
    )
    try:
        mcp, _ = create_server(state=state)
        too_many_models = {f"vendor/model-{i}": {"sort": "price"} for i in range(101)}
        with pytest.raises(ToolError, match="100"):
            await mcp.call_tool("update_config", {"provider_routing": {"models": too_many_models}})
    finally:
        await state.aclose()







