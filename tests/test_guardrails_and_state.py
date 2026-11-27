"""Stage 1 unit tests for guardrails.py and state.py (path jail, atomic I/O, external state, rollback)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from overseer.guardrails import (
    BlockReplaceError,
    PathSecurityError,
    _run_optional_ruff_check,
    apply_replace_block,
    atomic_write_bytes,
    atomic_write_bytes_sync,
    extract_file_outline,
    get_data_dir,
    get_project_data_dir,
    is_denied_rel_path,
    read_bytes_retry_sync,
    read_text_retry_sync,
    resolve_project_root,
    resolve_scoped_path,
    run_static_precheck,
    sha256_bytes,
    unlink_retry_sync,
    validate_non_overlapping_folders,
    validate_raw_path_syntax,
)
from overseer.models import (
    CreatedFileRecord,
    FileEditRecord,
    RunLimits,
    RunManifest,
    StaticCheckResult,
    WorkerManifestState,
    WorkerTaskInput,
)
from overseer.state import (
    RunRegistry,
    RunStateError,
    acquire_project_active_lock_sync,
    build_bounded_report,
    generate_run_id,
    get_run_monitor_snapshot,
    is_pid_alive,
    list_all_runs_snapshot,
    load_manifest_sync,
    manifest_path_for,
    reconcile_manifest_liveness,
    record_backup_once_sync,
    rollback_run,
    save_manifest_atomic,
    utc_now_iso,
    validate_and_resolve_run_id,
    write_latest_run_pointer,
    write_worker_patch_sync,
)


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data_dir = tmp_path / "overseer_appdata"
    monkeypatch.setenv("OVERSEER_DATA_DIR", str(data_dir))
    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "frontend").mkdir(parents=True)
    return repo, data_dir


def test_raw_path_syntax_rejects_dangerous_inputs() -> None:
    bad_inputs = [
        "",
        "   ",
        "backend/\x00secret.py",
        r"\\attacker\share\file.py",
        "//attacker/share/file.py",
        r"\\?\C:\Windows\System32",
        r"\\.\pipe\overseer",
        "C:relative.py",
        "backend/file.txt:zone.identifier",
        "backend/.env ",
        "backend/.env.",
        "backend/../secrets.py",
        "backend/CON",
        "backend/nul.txt",
        "backend/COM1.py",
        "backend/LPT9",
    ]
    for raw in bad_inputs:
        with pytest.raises(PathSecurityError):
            validate_raw_path_syntax(raw)

    if os.name == "nt":
        with pytest.raises(PathSecurityError):
            validate_raw_path_syntax("/etc/passwd")


def test_deny_globs_cover_files_and_dirs() -> None:
    denied = [
        ".env",
        ".env.local",
        "backend/.env",
        "config/prod.env",
        "certs/server.pem",
        "keys/private.key",
        "id_rsa",
        "ssh/id_ed25519.pub",
        ".git",
        ".git/config",
        ".vscode/settings.json",
        ".idea/workspace.xml",
        ".github/workflows/deploy.yml",
        ".overseer/manifest.json",
        "SUB/.GIT/HEAD",
        "PROD.ENV",
    ]
    for path in denied:
        assert is_denied_rel_path(path), f"Expected deny for {path}"

    allowed = [
        "backend/api.py",
        "frontend/App.tsx",
        "config/settings.json",
        "pyproject.toml",
        ".gitignore",
    ]
    for path in allowed:
        assert not is_denied_rel_path(path), f"Expected allow for {path}"


def test_resolve_project_root_and_external_isolation(
    isolated_env: tuple[Path, Path],
) -> None:
    repo, data_dir = isolated_env
    resolved = resolve_project_root(str(repo))
    assert resolved == repo.resolve()

    gemini_dir = Path.home() / ".gemini"
    gemini_dir.mkdir(exist_ok=True)
    with pytest.raises(PathSecurityError):
        resolve_project_root(str(gemini_dir))

    data_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathSecurityError):
        resolve_project_root(str(data_dir))

    with pytest.raises(PathSecurityError):
        resolve_project_root("relative/path")

    file_target = repo / "backend" / "api.py"
    file_target.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(PathSecurityError):
        resolve_project_root(str(file_target))

    proj_data = get_project_data_dir(resolved)
    assert proj_data.is_relative_to(get_data_dir())
    assert not any(repo.rglob(".overseer"))


def test_scoped_path_containment_and_symlink_escape(
    isolated_env: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    repo, _ = isolated_env
    backend = (repo / "backend").resolve()
    api_file = backend / "api.py"
    api_file.write_text("ok = True\n", encoding="utf-8")

    # Read allowed across project_root, write constrained to allowed_root (backend)
    assert resolve_scoped_path(repo, repo, "backend/api.py", require_exists=True) == api_file
    with pytest.raises(PathSecurityError):
        resolve_scoped_path(repo, backend, "frontend/app.py", require_exists=False)
    with pytest.raises(PathSecurityError):
        resolve_scoped_path(repo, backend, "backend/.env", require_exists=False)
    with pytest.raises(PathSecurityError):
        resolve_scoped_path(repo, backend, "backend/missing_dir/new.py", require_exists=False, allow_new_parents=False)

    # Nested new parent allowed when creating new files
    nested_target = resolve_scoped_path(
        repo,
        backend,
        "backend/sub/deep/new_mod.py",
        require_exists=False,
        allow_new_parents=True,
    )
    assert nested_target == (backend / "sub" / "deep" / "new_mod.py")

    # Symlink / junction escape check (real OS link when permitted, plus mocked reparse verification)
    outside_dir = tmp_path / "outside_secret"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("top-secret", encoding="utf-8")
    link_path = backend / "escape_link"
    created_symlink = False
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
        created_symlink = True
    except OSError:
        link_path.mkdir()

    if created_symlink:
        with pytest.raises(PathSecurityError):
            resolve_scoped_path(repo, backend, "backend/escape_link/secret.txt", require_exists=True)
        with pytest.raises(PathSecurityError):
            resolve_scoped_path(repo, backend, "backend/escape_link/new_sub/x.py", require_exists=False, allow_new_parents=True)
    else:
        from overseer.guardrails import _ensure_no_reparse_escape

        orig_is_symlink = Path.is_symlink
        orig_resolve = Path.resolve
        with (
            patch.object(Path, "is_symlink", lambda p: p.name == "escape_link" or orig_is_symlink(p)),
            patch.object(
                Path,
                "resolve",
                lambda p, strict=False: outside_dir if p.name == "escape_link" else orig_resolve(p, strict=strict),
            ),
        ):
            with pytest.raises(PathSecurityError, match="escape"):
                _ensure_no_reparse_escape(link_path / "secret.txt", repo.resolve())


def test_validate_non_overlapping_folders(isolated_env: tuple[Path, Path]) -> None:
    repo, _ = isolated_env
    (repo / "backend" / "routes").mkdir()

    ok_tasks = [
        WorkerTaskInput(worker_id="w1", folder="backend", goal="edit backend"),
        WorkerTaskInput(worker_id="w2", folder="frontend", goal="edit frontend"),
    ]
    folders = validate_non_overlapping_folders(repo, ok_tasks)
    assert set(folders) == {"w1", "w2"}

    # Single worker with '.' is allowed
    single_dot = [WorkerTaskInput(worker_id="w1", folder=".", goal="whole repo")]
    assert validate_non_overlapping_folders(repo, single_dot)["w1"] == repo.resolve()

    # Multi-worker '.' or nested/identical rejected
    with pytest.raises(PathSecurityError):
        validate_non_overlapping_folders(
            repo,
            [
                WorkerTaskInput(worker_id="w1", folder=".", goal="all"),
                WorkerTaskInput(worker_id="w2", folder="frontend", goal="fe"),
            ],
        )
    with pytest.raises(PathSecurityError):
        validate_non_overlapping_folders(
            repo,
            [
                WorkerTaskInput(worker_id="w1", folder="backend", goal="a"),
                WorkerTaskInput(worker_id="w2", folder="backend/routes", goal="b"),
            ],
        )


def test_apply_replace_block_preserves_crlf_and_bom() -> None:
    raw = "\ufeffline1 = 1\r\nline2 = 2\r\nline3 = 3\r\n".encode("utf-8")
    updated, added, removed = apply_replace_block(
        raw,
        target="line2 = 2\nline3 = 3",
        replacement="line2 = 20\nline3 = 30\nline4 = 40",
    )
    assert updated.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in updated
    assert added == 3
    assert removed == 2

    with pytest.raises(BlockReplaceError, match="non-empty"):
        apply_replace_block(raw, "", "x")
    with pytest.raises(BlockReplaceError, match="not valid UTF-8"):
        apply_replace_block(b"\xff\xfe\x00\x01", "a", "b")
    with pytest.raises(BlockReplaceError, match="File has 3 lines"):
        apply_replace_block(raw, "completely absent text", "x")
    with pytest.raises(BlockReplaceError, match="Closest match around line"):
        apply_replace_block(raw, "line2 = 2\nline3 = WRONG", "x")

    dup = b"x = 1\nx = 1\n"
    with pytest.raises(BlockReplaceError, match="matched 2 times"):
        apply_replace_block(dup, "x = 1", "x = 2")


async def test_atomic_write_retries_permission_error(tmp_path: Path) -> None:
    target = tmp_path / "atomic.txt"
    calls = 0

    def flaky_write(_path: Path, _data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("locked by AV")
        _path.write_bytes(_data)

    with patch("overseer.guardrails._atomic_write_sync", side_effect=flaky_write):
        await atomic_write_bytes(target, b"hello", retries=3, backoff_sec=0.001)
    assert calls == 3
    assert target.read_bytes() == b"hello"

    with patch("overseer.guardrails._atomic_write_sync", side_effect=PermissionError("stuck")):
        with pytest.raises(PermissionError):
            await atomic_write_bytes(target, b"fail", retries=1, backoff_sec=0.001)


def test_run_static_precheck_in_memory_without_pycache(isolated_env: tuple[Path, Path]) -> None:
    repo, _ = isolated_env
    (repo / "backend" / "good.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    (repo / "backend" / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    (repo / "backend" / "bad.json").write_text('{"a": ', encoding="utf-8")
    (repo / "backend" / "bad.toml").write_text("key = [\n", encoding="utf-8")
    (repo / "backend" / "good.luau").write_text(
        "--!strict\nlocal M = {}\nlocal s = if true then os.time() else 0\ntask.wait(1)\nreturn M\n",
        encoding="utf-8",
    )
    (repo / "backend" / "bad.luau").write_text("local M = {\nwait(1)\nreturn M\n", encoding="utf-8")
    (repo / "backend" / "bad_time.luau").write_text("--!strict\nlocal t = time()\nlocal function f()\nlocal x = (1 + 2\n", encoding="utf-8")
    (repo / "backend" / "good.cpp").write_text("#include <iostream>\nint main() { return 0; }\n", encoding="utf-8")
    (repo / "backend" / "bad.cpp").write_text("int main( { if (true) {\n", encoding="utf-8")
    (repo / "backend" / "good.ps1").write_text("if ($true) { Write-Host 'ok' }\n", encoding="utf-8")
    (repo / "backend" / "bad.ps1").write_text("if ($true {\n", encoding="utf-8")
    (repo / "frontend" / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")

    # Unsupported extension -> not_run
    (repo / "frontend" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    md_res = run_static_precheck(repo, ["frontend/notes.md"])
    assert md_res.status == "not_run"

    # Supported TS extension -> passed
    ts_res = run_static_precheck(repo, ["frontend/app.ts"])
    assert ts_res.status == "passed"

    ok_res = run_static_precheck(repo, ["backend/good.py", "backend/good.luau", "backend/good.cpp", "backend/good.ps1"])
    assert ok_res.status == "passed"
    assert not any(repo.rglob("__pycache__"))

    err_res = run_static_precheck(
        repo,
        [
            "backend/bad.py",
            "backend/bad.json",
            "backend/bad.toml",
            "backend/bad.luau",
            "backend/bad_time.luau",
            "backend/bad.cpp",
            "backend/bad.ps1",
        ],
    )
    assert err_res.status == "failed"
    assert len(err_res.errors) >= 10
    assert any("LuauError" in e and "time()" in e for e in err_res.errors)
    assert any("LuauError" in e and "Unclosed '('" in e for e in err_res.errors)
    assert any("LuauError" in e and "Unclosed 'function' block" in e for e in err_res.errors)
    assert any("CppSyntaxError" in e and "Unclosed '('" in e for e in err_res.errors)
    assert any("PowerShellError" in e and "Unclosed '('" in e for e in err_res.errors)
    assert not any(repo.rglob("__pycache__"))


async def test_state_liveness_bounded_report_and_rollback(
    isolated_env: tuple[Path, Path],
) -> None:
    repo, _ = isolated_env
    assert is_pid_alive(os.getpid()) is True
    assert is_pid_alive(-1) is False
    assert is_pid_alive(999_999_999) is False

    registry = RunRegistry()
    proj_data = get_project_data_dir(repo)
    run_id = generate_run_id()
    run_dir = proj_data / "runs" / run_id
    backups_dir = proj_data / "backups" / run_id
    m_path = manifest_path_for(proj_data, run_id)

    # Reject invalid run_id traversal attempts
    with pytest.raises(RunStateError):
        validate_and_resolve_run_id(proj_data, "../escape")

    # Prepare files on disk as if a worker ran
    mod_unchanged = repo / "backend" / "unchanged.py"
    mod_edited_by_user = repo / "backend" / "conflict.py"
    created_nested = repo / "backend" / "newpkg" / "created.py"
    created_nested.parent.mkdir(parents=True)

    orig_unchanged = b"val = 1\n"
    new_unchanged = b"val = 2\n"
    mod_unchanged.write_bytes(new_unchanged)
    b1 = record_backup_once_sync(backups_dir, "backend/unchanged.py", orig_unchanged)

    orig_conflict = b"c = 1\n"
    worker_conflict = b"c = 2\n"
    user_conflict = b"c = 999\n"
    mod_edited_by_user.write_bytes(user_conflict)
    b2 = record_backup_once_sync(backups_dir, "backend/conflict.py", orig_conflict)

    created_bytes = b"def helper():\n    return 42\n"
    created_nested.write_bytes(created_bytes)

    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=999_999_999,
        created_at_iso=utc_now_iso(),
        project_root=str(repo),
        status="running",
        limits=RunLimits(),
        spend_usd=0.01,
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="test",
                requested_model="m:free",
                models_used=["m:free"],
                status="running",
                steps_used=2,
                modified_files={
                    "backend/unchanged.py": FileEditRecord(
                        backup_path=str(b1),
                        initial_hash=sha256_bytes(orig_unchanged),
                        worker_last_hash=sha256_bytes(new_unchanged),
                        added_lines=1,
                        removed_lines=1,
                    ),
                    "backend/conflict.py": FileEditRecord(
                        backup_path=str(b2),
                        initial_hash=sha256_bytes(orig_conflict),
                        worker_last_hash=sha256_bytes(worker_conflict),
                        added_lines=1,
                        removed_lines=1,
                    ),
                },
                created_files={
                    "backend/newpkg/created.py": CreatedFileRecord(
                        worker_last_hash=sha256_bytes(created_bytes),
                        line_count=2,
                    )
                },
                summary="W" * 500,
                notes=["Edit frontend/app.ts to call endpoint"] * 20,
            ),
            "w2": WorkerManifestState(
                worker_id="w2",
                folder="frontend",
                goal="noop",
                requested_model="m:free",
                status="completed",
            ),
        },
    )

    lock = registry.get_lock(repo, run_id)
    await save_manifest_atomic(m_path, manifest, lock)
    await write_latest_run_pointer(proj_data, run_id)
    assert validate_and_resolve_run_id(proj_data, "latest") == run_id

    # Reconcile dead PID -> interrupted
    reconciled = await reconcile_manifest_liveness(load_manifest_sync(m_path), m_path, repo, registry)
    assert reconciled.status == "interrupted"
    assert reconciled.workers["w1"].status == "interrupted"
    assert reconciled.workers["w1"].stop_reason == "process_terminated"

    # Empty worker produces no patch file; edited worker produces unified patch
    assert write_worker_patch_sync(repo, run_dir, reconciled.workers["w2"]) is None
    bounded = build_bounded_report(repo, reconciled, run_dir, max_bytes=1200)
    serialized = json.dumps(bounded.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")
    assert len(serialized) <= 1200
    assert bounded.truncated is True
    assert Path(bounded.full_report_path).exists()

    # Rollback cannot execute while active in registry
    registry.register_run(repo, run_id)
    with pytest.raises(RunStateError, match="running"):
        await rollback_run(repo, run_id, registry)
    registry.mark_run_finished(repo, run_id)

    # Conflict-safe rollback deletes unchanged created file (+ empty parent dir), restores unchanged, skips user conflict
    rb = await rollback_run(repo, "latest", registry)
    assert rb.deleted_files == ["backend/newpkg/created.py"]
    assert not created_nested.exists()
    assert not created_nested.parent.exists()
    assert rb.restored_files == ["backend/unchanged.py"]
    assert mod_unchanged.read_bytes() == orig_unchanged
    assert rb.skipped_conflicts == ["backend/conflict.py"]
    assert mod_edited_by_user.read_bytes() == user_conflict

    # Second rollback is idempotent
    rb2 = await rollback_run(repo, run_id, registry)
    assert rb2.already_rolled_back is True
    assert rb2.restored_files == []
    assert rb2.deleted_files == []


@pytest.mark.asyncio
async def test_registry_lifecycle_report_truncation_and_missing_file_rollback(
    isolated_env: tuple[Path, Path],
) -> None:
    repo, _ = isolated_env
    proj_data = get_project_data_dir(repo)
    with pytest.raises(RunStateError, match="No runs found"):
        validate_and_resolve_run_id(proj_data, "latest")

    registry = RunRegistry()
    live_id = "run_20261115_000001_aaaaaa"
    ev = registry.register_run(repo, live_id)
    sleeper = asyncio.create_task(asyncio.sleep(10))
    registry.attach_tasks(repo, live_id, [sleeper])
    assert registry.has_active_run(repo) == live_id
    assert registry.get_cancel_event(repo, live_id) is ev
    await registry.shutdown_all()
    assert ev.is_set()
    assert sleeper.cancelled()
    assert registry.has_active_run(repo) is None

    run_older = "run_20261115_000001_aaaaaa"
    run_newer = "run_20261115_000002_bbbbbb"
    for rid in (run_older, run_newer):
        rdir = proj_data / "runs" / rid
        (rdir / "backups").mkdir(parents=True, exist_ok=True)
        m_item = RunManifest(
            version="0.1.0",
            run_id=rid,
            server_pid=os.getpid(),
            created_at_iso=utc_now_iso(),
            project_root=str(repo.resolve()),
            status="completed",
            limits=RunLimits(),
            patch_dir=str(rdir),
            workers={},
        )
        await save_manifest_atomic(manifest_path_for(proj_data, rid), m_item, registry.get_lock(repo, rid))

    assert validate_and_resolve_run_id(proj_data, "latest") == run_newer

    run_dir = proj_data / "runs" / run_newer
    backups_dir = run_dir / "backups"
    vanished_orig = b"v = 1\n"
    b_vanished = record_backup_once_sync(backups_dir, "backend/vanished.py", vanished_orig)

    created_records: dict[str, CreatedFileRecord] = {}
    for idx in range(12):
        rel = f"backend/file_{idx:02d}.py"
        data = f"VAL_{idx} = {idx}\n".encode("utf-8")
        (repo / rel).write_bytes(data)
        created_records[rel] = CreatedFileRecord(worker_last_hash=sha256_bytes(data), line_count=1)

    rich_manifest = RunManifest(
        version="0.1.0",
        run_id=run_newer,
        server_pid=os.getpid(),
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed_with_escalations",
        limits=RunLimits(),
        spend_usd=0.05,
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="many edits",
                requested_model="m:free",
                models_used=["m:free"],
                status="completed",
                steps_used=3,
                modified_files={
                    "backend/vanished.py": FileEditRecord(
                        backup_path=str(b_vanished),
                        initial_hash=sha256_bytes(vanished_orig),
                        worker_last_hash=sha256_bytes(b"v = 2\n"),
                        added_lines=1,
                        removed_lines=1,
                    )
                },
                created_files=created_records,
                static_check=StaticCheckResult(
                    status="failed",
                    errors=[f"SyntaxError line {i}: invalid syntax " + ("x" * 60) for i in range(6)],
                ),
                summary="Summary " * 80,
                notes=["Check sibling file " + ("n" * 80) for _ in range(6)],
            )
        },
    )
    await save_manifest_atomic(manifest_path_for(proj_data, run_newer), rich_manifest, registry.get_lock(repo, run_newer))

    unbounded = build_bounded_report(repo, rich_manifest, run_dir, max_bytes=100_000)
    assert unbounded.truncated is False

    mid_bounded = build_bounded_report(repo, rich_manifest, run_dir, max_bytes=1600)
    assert mid_bounded.truncated is True
    assert len(json.dumps(mid_bounded.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")) <= 1600

    tiny_bounded = build_bounded_report(repo, rich_manifest, run_dir, max_bytes=10)
    assert tiny_bounded.truncated is True
    assert tiny_bounded.workers == []

    rb = await rollback_run(repo, run_newer, registry)
    assert "backend/vanished.py" in rb.skipped_conflicts
    assert len(rb.deleted_files) == 12


def test_sync_atomic_write_and_read_retries_permission_error(tmp_path: Path) -> None:
    target = tmp_path / "sync_atomic.txt"
    write_calls = 0

    def flaky_sync_write(path: Path, data: bytes, mode: int | None = None) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls < 3:
            raise PermissionError("locked write")
        path.write_bytes(data)

    with patch("overseer.guardrails._atomic_write_sync", side_effect=flaky_sync_write):
        atomic_write_bytes_sync(target, b"sync-ok", retries=3, backoff_sec=0.001)
    assert write_calls == 3
    assert target.read_bytes() == b"sync-ok"

    with patch("overseer.guardrails._atomic_write_sync", side_effect=PermissionError("perm")):
        with pytest.raises(PermissionError):
            atomic_write_bytes_sync(target, b"fail", retries=1, backoff_sec=0.001)

    orig_read_bytes = Path.read_bytes
    read_calls = 0

    def flaky_read_bytes(self: Path) -> bytes:
        nonlocal read_calls
        if self == target:
            read_calls += 1
            if read_calls < 3:
                raise PermissionError("locked read")
        return orig_read_bytes(self)

    with patch.object(Path, "read_bytes", flaky_read_bytes):
        assert read_text_retry_sync(target, retries=3, backoff_sec=0.001) == "sync-ok"
    assert read_calls == 3

    with patch.object(Path, "read_bytes", side_effect=PermissionError("perm")):
        with pytest.raises(PermissionError):
            read_bytes_retry_sync(target, retries=1, backoff_sec=0.001)

    orig_unlink = Path.unlink
    unlink_calls = 0

    def flaky_unlink(self: Path) -> None:
        nonlocal unlink_calls
        if self == target:
            unlink_calls += 1
            if unlink_calls < 3:
                raise PermissionError("locked unlink")
        orig_unlink(self)

    with patch.object(Path, "unlink", flaky_unlink):
        unlink_retry_sync(target, retries=3, backoff_sec=0.001)
    assert unlink_calls == 3
    assert not target.exists()

    with patch.object(Path, "unlink", side_effect=PermissionError("stuck unlink")):
        with pytest.raises(PermissionError):
            unlink_retry_sync(target, retries=1, backoff_sec=0.001)


async def test_registry_prunes_finished_run_state(isolated_env: tuple[Path, Path]) -> None:
    repo, _ = isolated_env
    registry = RunRegistry()
    run_id = "run_20261116_000001_abc123"
    registry.register_run(repo, run_id)
    _ = registry.get_lock(repo, run_id)
    registry.mark_user_cancelled(repo, run_id)

    key = (registry._project_key(repo), run_id)
    assert key in registry._locks
    assert key in registry._cancel_events
    assert key in registry._user_cancelled

    registry.mark_run_finished(repo, run_id)
    assert registry.has_active_run(repo) is None
    assert key not in registry._locks
    assert key not in registry._cancel_events
    assert key not in registry._user_cancelled
    assert key not in registry._interrupted


def test_run_static_precheck_retries_permission_error(isolated_env: tuple[Path, Path]) -> None:
    repo, _ = isolated_env
    py_file = repo / "backend" / "retried.py"
    py_file.write_text("X = 1\n", encoding="utf-8")

    orig_read_bytes = Path.read_bytes
    calls = 0

    def flaky_read(self: Path) -> bytes:
        nonlocal calls
        if self == py_file:
            calls += 1
            if calls == 1:
                raise PermissionError("transient AV scan")
        return orig_read_bytes(self)

    with (
        patch.object(Path, "read_bytes", flaky_read),
        patch("overseer.guardrails.time.sleep"),
    ):
        res = run_static_precheck(repo, ["backend/retried.py"])
    assert res.status == "passed"
    assert calls == 2


async def test_concurrent_manifest_lock_and_monkeypatched_pid_liveness(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = isolated_env
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261116_000022_112233"
    run_dir = proj_data / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)

    registry = RunRegistry()
    lock = registry.get_lock(repo, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=12345,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={"w1": WorkerManifestState(worker_id="w1", folder="backend", goal="conc", requested_model="m:free", status="running")},
    )

    async def _write_step(step: int) -> None:
        manifest.workers["w1"].steps_used = step
        await save_manifest_atomic(m_path, manifest, lock)

    await asyncio.gather(*(_write_step(i) for i in range(1, 7)))
    loaded = load_manifest_sync(m_path)
    assert loaded.workers["w1"].steps_used in range(1, 7)

    monkeypatch.setattr("overseer.state.is_pid_alive", lambda pid: False)
    reconciled = await reconcile_manifest_liveness(loaded, m_path, repo, registry)
    assert reconciled.status == "interrupted"
    assert reconciled.workers["w1"].status == "interrupted"
    assert reconciled.workers["w1"].stop_reason == "process_terminated"


async def test_rollback_preserves_preexisting_empty_dir_and_keeps_post_rollback_patch(
    isolated_env: tuple[Path, Path],
) -> None:
    from overseer.guardrails import sha256_bytes as _hash
    from overseer.state import collect_run_report

    repo, _ = isolated_env
    preexisting_empty = repo / "backend" / "existing_dir"
    preexisting_empty.mkdir(parents=True, exist_ok=True)
    new_file_in_existing = preexisting_empty / "added.py"
    new_bytes = b"A = 1\n"
    new_file_in_existing.write_bytes(new_bytes)

    new_subdir = repo / "backend" / "created_sub"
    new_subdir.mkdir(parents=True, exist_ok=True)
    new_sub_file = new_subdir / "b.py"
    new_sub_file.write_bytes(new_bytes)

    proj_data = get_project_data_dir(repo)
    run_id = "run_20261116_000033_445566"
    run_dir = proj_data / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)

    registry = RunRegistry()
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=999_999_999,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="dirs",
                requested_model="m:free",
                status="completed",
                created_dirs=["backend/created_sub"],
                created_files={
                    "backend/existing_dir/added.py": CreatedFileRecord(worker_last_hash=_hash(new_bytes), line_count=1),
                    "backend/created_sub/b.py": CreatedFileRecord(worker_last_hash=_hash(new_bytes), line_count=1),
                },
            )
        },
    )
    await save_manifest_atomic(m_path, manifest, registry.get_lock(repo, run_id))
    await write_latest_run_pointer(proj_data, run_id)

    report_before = await collect_run_report(repo, run_id, registry)
    patch_file = Path(report_before.workers[0].patch_path or "")
    assert patch_file.exists() and "+A = 1" in patch_file.read_text(encoding="utf-8")

    rb = await rollback_run(repo, run_id, registry)
    assert sorted(rb.deleted_files) == ["backend/created_sub/b.py", "backend/existing_dir/added.py"]
    assert preexisting_empty.exists()
    assert not new_subdir.exists()

    report_after = await collect_run_report(repo, run_id, registry)
    assert report_after.status == "rolled_back"
    assert patch_file.exists() and "+A = 1" in patch_file.read_text(encoding="utf-8")


def test_guardrails_rejects_leading_space_drive_root_and_extra_secrets(
    isolated_env: tuple[Path, Path],
) -> None:
    from overseer.guardrails import is_denied_rel_path, resolve_project_root, validate_raw_path_syntax

    with pytest.raises(PathSecurityError):
        validate_raw_path_syntax(" backend/app.py")
    with pytest.raises(PathSecurityError):
        resolve_project_root(str(Path.home()))
    anchor = Path.cwd().anchor
    if anchor:
        with pytest.raises(PathSecurityError):
            resolve_project_root(anchor)
    for secret_path in (".aws/credentials", "certs/server.crt", ".npmrc", ".netrc", ".pypirc"):
        assert is_denied_rel_path(secret_path) is True


def test_extract_file_outline_languages_errors_and_caps() -> None:
    py_src = (
        "class Worker(Base):\n"
        "    def sync_method(self, x: int) -> str:\n"
        "        return str(x)\n\n"
        "    async def async_method(self) -> None:\n"
        "        pass\n\n"
        "async def top_fetch(url: str) -> dict[str, str]:\n"
        "    return {'url': url}\n"
    )
    py_out = extract_file_outline(py_src, "backend/mod.py")
    assert "# backend/mod.py (9 lines)" in py_out
    assert "L1-L6: class Worker(Base)" in py_out
    assert "L2-L3: def sync_method(self, x: int) -> str" in py_out
    assert "L5-L6: async def async_method(self) -> None" in py_out
    assert "L8-L9: async def top_fetch(url: str) -> dict[str, str]" in py_out

    py_err = extract_file_outline("def broken(:\n    pass\n", "backend/broken.py")
    assert "# backend/broken.py (2 lines)" in py_err
    assert "SyntaxError at L1:" in py_err

    luau_src = (
        "--!strict\n"
        "--[[ block comment\n"
        "local function ignored_in_block()\n"
        "]]\n"
        "-- local function ignored_line()\n"
        "export type RewardResult = { ok: boolean }\n"
        "local function validate(userId: number): boolean\n"
        "    return userId > 0\n"
        "end\n"
        "function M.Award(player: Player)\n"
        "end\n"
    )
    luau_out = extract_file_outline(luau_src, "src/Server.luau")
    assert "L6: export type RewardResult" in luau_out
    assert "L7: local function validate(userId: number): boolean" in luau_out
    assert "L10: function M.Award(player: Player)" in luau_out
    assert "ignored_in_block" not in luau_out
    assert "ignored_line" not in luau_out

    cpp_src = (
        "#include <string>\n"
        "/* class Ignored {}; */\n"
        "namespace demo {\n"
        "struct Record {\n"
        "    int id;\n"
        "    void innerMethod() {}\n"
        "};\n"
        "int computeScore(int a, int b)\n"
        "{\n"
        "    if (a > b) { return a; }\n"
        "    return b;\n"
        "}\n"
        "} // namespace demo\n"
    )
    cpp_out = extract_file_outline(cpp_src, "src/main.cpp")
    assert "L4: struct Record" in cpp_out
    assert "L8: int computeScore(int a, int b)" in cpp_out
    assert "Ignored" not in cpp_out
    assert "innerMethod" not in cpp_out
    assert "if (a > b)" not in cpp_out

    ps1_src = (
        "<#\n"
        "function Ignored-Comment {}\n"
        "#>\n"
        "class NetProfile {\n"
        "    [string]$Name\n"
        "}\n"
        "function Invoke-LatencyAudit {\n"
        '    $here = @"\n'
        "function Fake-InsideHereString {}\n"
        '"@\n'
        "}\n"
        "filter Select-ActiveAdapter { $_ }\n"
    )
    ps1_out = extract_file_outline(ps1_src, "scripts/audit.ps1")
    assert "L4: class NetProfile" in ps1_out
    assert "L7: function Invoke-LatencyAudit" in ps1_out
    assert "L12: filter Select-ActiveAdapter" in ps1_out
    assert "Ignored-Comment" not in ps1_out
    assert "Fake-InsideHereString" not in ps1_out

    other_out = extract_file_outline("# Title\nBody\n", "README.md")
    assert "# README.md (2 lines)" in other_out
    assert "Outline not supported for .md; use view_file or search_dir." in other_out

    many_fns = "\n".join(f"def fn_{i}():\n    return {i}\n" for i in range(12))
    sym_capped = extract_file_outline(many_fns, "backend/many.py", max_symbols=5, max_bytes=16_384)
    assert "... [truncated at 5 symbols (12 total)]" in sym_capped

    byte_capped = extract_file_outline(many_fns, "backend/many.py", max_symbols=200, max_bytes=220)
    assert len(byte_capped.encode("utf-8")) <= 220
    assert "... [truncated at 220 bytes]" in byte_capped


def test_run_optional_ruff_check_missing_findings_disabled_and_timeout(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = isolated_env
    py_file = repo / "backend" / "lint_target.py"
    py_file.write_text("def fn():\n    return missing_var\n", encoding="utf-8")

    # 1. Missing ruff binary -> no-op
    monkeypatch.delenv("OVERSEER_DISABLE_RUFF", raising=False)
    with patch("overseer.guardrails.shutil.which", return_value=None):
        assert _run_optional_ruff_check("backend/lint_target.py", py_file.read_text(encoding="utf-8")) == []
        res = run_static_precheck(repo, ["backend/lint_target.py"])
        assert res.status == "passed"

    # 2. Mocked ruff binary returning F821 undefined name finding -> captured cleanly without executing repo code
    fake_completed = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=1,
        stdout="backend/lint_target.py:2:12: F821 Undefined name `missing_var`\nFound 1 error.\n",
        stderr="",
    )
    with (
        patch("overseer.guardrails.shutil.which", return_value="C:/tools/ruff.exe"),
        patch("overseer.guardrails.subprocess.run", return_value=fake_completed) as mock_run,
    ):
        findings = _run_optional_ruff_check("backend/lint_target.py", py_file.read_text(encoding="utf-8"))
        assert findings == ["backend/lint_target.py: RuffError: 2:12: F821 Undefined name `missing_var`"]
        call_args = mock_run.call_args
        assert call_args.args[0][-1] == "-"
        assert "--no-cache" in call_args.args[0]
        assert "--stdin-filename" in call_args.args[0]
        assert call_args.kwargs["input"] == "def fn():\n    return missing_var\n"
        if sys.platform == "win32":
            assert call_args.kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        precheck_res = run_static_precheck(repo, ["backend/lint_target.py"])
        assert precheck_res.status == "failed"
        assert any("RuffError" in e and "F821" in e for e in precheck_res.errors)

    # 3. OVERSEER_DISABLE_RUFF=1 skips subprocess entirely
    monkeypatch.setenv("OVERSEER_DISABLE_RUFF", "1")
    with (
        patch("overseer.guardrails.shutil.which", return_value="C:/tools/ruff.exe"),
        patch("overseer.guardrails.subprocess.run") as mock_run_disabled,
    ):
        assert _run_optional_ruff_check("backend/lint_target.py", "x = y\n") == []
        mock_run_disabled.assert_not_called()

    # 4. TimeoutExpired or OSError is safely ignored
    monkeypatch.delenv("OVERSEER_DISABLE_RUFF", raising=False)
    with (
        patch("overseer.guardrails.shutil.which", return_value="C:/tools/ruff.exe"),
        patch("overseer.guardrails.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ruff", timeout=5.0)),
    ):
        assert _run_optional_ruff_check("backend/lint_target.py", "x = 1\n") == []
        assert run_static_precheck(repo, ["backend/lint_target.py"]).status == "passed"

    with (
        patch("overseer.guardrails.shutil.which", return_value="C:/tools/ruff.exe"),
        patch("overseer.guardrails.subprocess.run", side_effect=OSError("cannot spawn")),
    ):
        assert _run_optional_ruff_check("backend/lint_target.py", "x = 1\n") == []


async def test_monitor_snapshot_helpers_multi_project_reconcile_and_file_uris(
    isolated_env: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    repo_a, _ = isolated_env
    repo_b = tmp_path / "repo_b"
    (repo_b / "src").mkdir(parents=True)

    edited_file = repo_a / "backend" / "api.py"
    edited_file.write_bytes(b"VAL = 2\n")
    created_file = repo_a / "backend" / "added.py"
    created_file.write_bytes(b"NEW = 1\n")

    proj_a_data = get_project_data_dir(repo_a)
    proj_b_data = get_project_data_dir(repo_b)
    registry = RunRegistry()

    # Corrupt directory / manifest should be safely ignored by list_all_runs_snapshot
    corrupt_dir = proj_a_data / "runs" / "run_20261115_000001_bad000"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "manifest.json").write_text("{broken json", encoding="utf-8")

    # Run 1 in repo_a: dead-PID running -> should be reconciled to interrupted
    run_stale = "run_20261116_100000_111111"
    stale_dir = proj_a_data / "runs" / run_stale
    b1 = record_backup_once_sync(proj_a_data / "backups" / run_stale, "backend/api.py", b"VAL = 1\n")
    m_stale = RunManifest(
        version="0.1.0",
        run_id=run_stale,
        server_pid=999_999_999,
        created_at_iso="2026-11-16T10:00:00+00:00",
        project_root=str(repo_a.resolve()),
        status="running",
        limits=RunLimits(max_run_usd=0.5, max_steps_per_worker=15),
        spend_usd=0.0123,
        patch_dir=str(stale_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="update api",
                requested_model="m:free",
                models_used=["m:free"],
                status="running",
                steps_used=3,
                spend_usd=0.0123,
                modified_files={
                    "backend/api.py": FileEditRecord(
                        backup_path=str(b1),
                        initial_hash=sha256_bytes(b"VAL = 1\n"),
                        worker_last_hash=sha256_bytes(b"VAL = 2\n"),
                        added_lines=1,
                        removed_lines=1,
                    )
                },
                created_files={
                    "backend/added.py": CreatedFileRecord(
                        worker_last_hash=sha256_bytes(b"NEW = 1\n"),
                        line_count=1,
                    )
                },
                summary="Updated backend/api.py",
                notes=["Wire added.py in frontend"],
            )
        },
    )
    await save_manifest_atomic(manifest_path_for(proj_a_data, run_stale), m_stale, registry.get_lock(repo_a, run_stale))
    await write_latest_run_pointer(proj_a_data, run_stale)

    # Run 2 in repo_b: actively running in registry, newer timestamp
    run_active = "run_20261117_120000_222222"
    active_dir = proj_b_data / "runs" / run_active
    registry.register_run(repo_b, run_active)
    m_active = RunManifest(
        version="0.1.0",
        run_id=run_active,
        server_pid=os.getpid(),
        created_at_iso="2026-11-17T12:00:00+00:00",
        project_root=str(repo_b.resolve()),
        status="running",
        limits=RunLimits(),
        patch_dir=str(active_dir),
        workers={
            "wb": WorkerManifestState(
                worker_id="wb",
                folder="src",
                goal="live worker",
                requested_model="m:free",
                status="running",
                steps_used=1,
            )
        },
    )
    await save_manifest_atomic(manifest_path_for(proj_b_data, run_active), m_active, registry.get_lock(repo_b, run_active))
    await write_latest_run_pointer(proj_b_data, run_active)

    all_listing = await list_all_runs_snapshot(registry, project_root=None, limit=10)
    assert all_listing["active_count"] == 1
    assert [r["run_id"] for r in all_listing["runs"]] == [run_active, run_stale]
    assert all_listing["runs"][0]["status"] == "running"
    assert all_listing["runs"][1]["status"] == "interrupted"

    scoped_listing = await list_all_runs_snapshot(registry, project_root=repo_a, limit=10)
    assert scoped_listing["active_count"] == 0
    assert [r["run_id"] for r in scoped_listing["runs"]] == [run_stale]

    snapshot = await get_run_monitor_snapshot(repo_a, "latest", registry)
    assert snapshot["run_id"] == run_stale
    assert snapshot["status"] == "interrupted"
    assert snapshot["limits"]["max_steps_per_worker"] == 15
    assert len(snapshot["workers"]) == 1
    w1 = snapshot["workers"][0]
    assert w1["worker_id"] == "w1"
    assert w1["max_steps"] == 15
    assert w1["stop_reason"] == "process_terminated"
    assert [tf["rel_path"] for tf in w1["touched_files"]] == ["backend/added.py", "backend/api.py"]
    for tf in w1["touched_files"]:
        assert tf["abs_uri"].startswith("file:///")
    assert w1["touched_files"][0]["kind"] == "created"
    assert w1["touched_files"][0]["stat"] == "+1 (created)"
    assert w1["touched_files"][1]["kind"] == "modified"
    assert w1["touched_files"][1]["stat"] == "+1 / -1"

    registry.mark_run_finished(repo_b, run_active)


def test_expanded_deny_globs_os_and_sensitive_home_dirs(
    isolated_env: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = isolated_env
    newly_denied = [
        "infra/main.tfvars",
        "infra/prod.tfvars.json",
        ".git-credentials",
        "k8s/kubeconfig",
        "clusters/prod.kubeconfig",
        "auth/.htpasswd",
        "credentials.json",
        "config/google_credentials.yaml",
        "config/app_secrets.toml",
        ".ssh/id_rsa",
        ".gnupg/secring.kbx",
        ".kube/config",
        ".docker/config.json",
    ]
    for rel in newly_denied:
        assert is_denied_rel_path(rel), f"Expected deny for {rel}"
    assert not is_denied_rel_path("backend/user_service.py")

    fake_home = tmp_path / "fake_home"
    ssh_dir = fake_home / ".ssh"
    gcloud_dir = fake_home / ".config" / "gcloud"
    ssh_dir.mkdir(parents=True)
    gcloud_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    with pytest.raises(PathSecurityError, match="sensitive home"):
        resolve_project_root(str(ssh_dir))
    with pytest.raises(PathSecurityError, match="sensitive home"):
        resolve_project_root(str(gcloud_dir))

    sys_candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) if sys.platform == "win32" else Path("/etc")
    if sys_candidate.is_dir():
        with pytest.raises(PathSecurityError, match="OS system directory"):
            resolve_project_root(str(sys_candidate))


def test_string_aware_static_precheck_and_ruff_isolated(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import patch as _patch

    repo, _ = isolated_env
    (repo / "backend" / "tricky.luau").write_text(
        "--!strict\n"
        'local flag = "--strict"\n'
        'local url = "https://example.com -- wait(1) ("\n'
        "local long_str = [[\n"
        "  -- [[ time() (\n"
        "]]\n"
        "local M = { a = 1 }\n"
        "return M\n",
        encoding="utf-8",
    )
    (repo / "backend" / "tricky.ps1").write_text(
        '<#\n'
        'Unmatched delimiters in block comment: { ( "\n'
        '#>\n'
        '$hex = "#ff0000"\n'
        '$url = "https://example.com # ("\n'
        '$nested = "color $(Write-Output "#abc")"\n'
        "if ($true) { Write-Output $hex }\n",
        encoding="utf-8",
    )
    (repo / "backend" / "tricky.cpp").write_text(
        "#include <string>\n"
        "#ifdef _WIN32\n"
        "int helper() {\n"
        "#else\n"
        "int helper(int v) {\n"
        "#endif\n"
        '    const char* url = "https://example.com /* (";\n'
        "    char ch = '{';\n"
        '    const char* raw = R"tag(raw { ( // )tag";\n'
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    res = run_static_precheck(
        repo,
        ["backend/tricky.luau", "backend/tricky.ps1", "backend/tricky.cpp"],
    )
    assert res.status == "passed", f"Unexpected precheck errors: {res.errors}"

    monkeypatch.delenv("OVERSEER_DISABLE_RUFF", raising=False)
    ok_proc = subprocess.CompletedProcess(args=["ruff"], returncode=0, stdout="", stderr="")
    with (
        _patch("overseer.guardrails.shutil.which", return_value="C:/tools/ruff.exe"),
        _patch("overseer.guardrails.subprocess.run", return_value=ok_proc) as mock_run,
    ):
        assert _run_optional_ruff_check("backend/app.py", "X = 1\n") == []
        assert "--isolated" in mock_run.call_args.args[0]


async def test_rollback_verifies_backup_sha256_and_skips_already_restored(
    isolated_env: tuple[Path, Path],
) -> None:
    repo, _ = isolated_env
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000001_deadbe"
    run_dir = proj_data / "runs" / run_id
    backups_dir = proj_data / "backups" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    m_path = manifest_path_for(proj_data, run_id)

    orig_corrupt = b"ORIG_A = 1\n"
    worker_corrupt = b"WORKER_A = 2\n"
    file_corrupt = repo / "backend" / "corrupt.py"
    file_corrupt.write_bytes(worker_corrupt)
    b_corrupt = record_backup_once_sync(backups_dir, "backend/corrupt.py", b"CORRUPT_BYTES\n")

    orig_restored = b"RESTORED = 1\n"
    worker_restored = b"RESTORED = 2\n"
    file_restored = repo / "backend" / "already.py"
    file_restored.write_bytes(orig_restored)
    b_restored = record_backup_once_sync(backups_dir, "backend/already.py", orig_restored)

    registry = RunRegistry()
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=999_999_999,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="backup check",
                requested_model="m:free",
                status="completed",
                modified_files={
                    "backend/corrupt.py": FileEditRecord(
                        backup_path=str(b_corrupt),
                        initial_hash=sha256_bytes(orig_corrupt),
                        worker_last_hash=sha256_bytes(worker_corrupt),
                    ),
                    "backend/already.py": FileEditRecord(
                        backup_path=str(b_restored),
                        initial_hash=sha256_bytes(orig_restored),
                        worker_last_hash=sha256_bytes(worker_restored),
                    ),
                },
            )
        },
    )
    await save_manifest_atomic(m_path, manifest, registry.get_lock(repo, run_id))
    rb = await rollback_run(repo, run_id, registry)
    assert rb.skipped_conflicts == ["backend/corrupt.py"]
    assert rb.restored_files == []
    assert file_corrupt.read_bytes() == worker_corrupt


async def test_pid_reuse_lock_reclaim_snapshot_hard_cap_and_bounded_shutdown(
    isolated_env: tuple[Path, Path],
) -> None:
    from overseer.state import (
        _build_monitor_snapshot_sync,
        acquire_project_active_lock_sync,
        read_active_lock_sync,
        release_project_active_lock_sync,
    )

    repo, _ = isolated_env
    proj_data = get_project_data_dir(repo)
    acquire_project_active_lock_sync(proj_data, "run_20261122_000001_aaaaaa", pid=4242, pid_create_time=100)
    lock_data = read_active_lock_sync(proj_data)
    assert lock_data is not None and lock_data["pid_create_time"] == 100

    with (
        patch("overseer.state.is_pid_alive", return_value=True),
        patch("overseer.state.get_pid_create_time", return_value=200),
    ):
        acquire_project_active_lock_sync(proj_data, "run_20261122_000002_bbbbbb", pid=4243, pid_create_time=200)
    new_lock = read_active_lock_sync(proj_data)
    assert new_lock is not None and new_lock["run_id"] == "run_20261122_000002_bbbbbb"
    release_project_active_lock_sync(proj_data, "run_20261122_000002_bbbbbb")

    big_manifest = RunManifest(
        version="0.1.0",
        run_id="run_20261122_000003_cccccc",
        server_pid=os.getpid(),
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(proj_data / "runs" / "run_20261122_000003_cccccc"),
        workers={
            f"w{i}": WorkerManifestState(
                worker_id=f"w{i}",
                folder="backend",
                goal="big",
                requested_model="m:free",
                status="completed",
                summary="S" * 500,
                notes=["N" * 100] * 5,
            )
            for i in range(6)
        },
    )
    tight_snap = _build_monitor_snapshot_sync(repo, big_manifest, max_bytes=600)
    assert len(json.dumps(tight_snap, separators=(",", ":")).encode("utf-8")) <= 600
    assert tight_snap["workers"] == []

    registry = RunRegistry()
    run_id = "run_20261122_000004_dddddd"
    registry.register_run(repo, run_id)

    async def stubborn_worker() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(10)

    stub_task = asyncio.create_task(stubborn_worker())
    registry.attach_tasks(repo, run_id, [stub_task])
    await asyncio.sleep(0)
    await asyncio.wait_for(registry.shutdown_all(timeout_sec=0.05), timeout=1.0)
    assert registry.has_active_run(repo) is None


def test_patch_writer_skips_post_run_escape_and_dangling_junction_blocked(
    isolated_env: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    repo, _ = isolated_env
    outside_dir = tmp_path / "outside_patch_escape"
    outside_dir.mkdir()
    (outside_dir / "leak.py").write_text("SECRET = 1\n", encoding="utf-8")

    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000005_eeeeee"
    run_dir = proj_data / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    esc_dir = repo / "backend" / "esc_dir"
    esc_dir.mkdir(parents=True)
    (esc_dir / "leak.py").write_text("OK = 1\n", encoding="utf-8")

    orig_is_junction = getattr(Path, "is_junction", None)
    orig_resolve = Path.resolve
    orig_exists = Path.exists

    worker_state = WorkerManifestState(
        worker_id="w1",
        folder="backend",
        goal="patch guard",
        requested_model="m:free",
        status="completed",
        created_files={
            "backend/esc_dir/leak.py": CreatedFileRecord(
                worker_last_hash=sha256_bytes(b"OK = 1\n"),
                line_count=1,
            )
        },
    )
    with (
        patch.object(
            Path,
            "is_junction",
            lambda p: p.name == "esc_dir" or (orig_is_junction(p) if orig_is_junction else False),
            create=True,
        ),
        patch.object(
            Path,
            "resolve",
            lambda p, strict=False: outside_dir if p.name == "esc_dir" else orig_resolve(p, strict=strict),
        ),
    ):
        assert write_worker_patch_sync(repo, run_dir, worker_state) is None

    # Dangling junction (exists() == False, is_junction() == True) is caught by resolve_scoped_path
    with (
        patch.object(Path, "exists", lambda p: False if p.name == "dangling_junc" else orig_exists(p)),
        patch.object(
            Path,
            "is_junction",
            lambda p: p.name == "dangling_junc" or (orig_is_junction(p) if orig_is_junction else False),
            create=True,
        ),
        patch.object(
            Path,
            "resolve",
            lambda p, strict=False: (_ for _ in ()).throw(OSError("broken link"))
            if p.name == "dangling_junc"
            else orig_resolve(p, strict=strict),
        ),
    ):
        with pytest.raises(PathSecurityError, match="unresolvable link"):
            resolve_scoped_path(repo, repo / "backend", "backend/dangling_junc/file.py", require_exists=False)


async def test_round2_deny_globs_redaction_case_worker_id_deleted_folder_rollback_and_lru_lookup(
    isolated_env: tuple[Path, Path],
) -> None:
    import shutil

    from overseer.guardrails import redact_secrets
    from overseer.models import StartRunRequest
    from overseer.state import find_run_across_projects_sync, write_worker_patch_sync

    repo, _ = isolated_env
    for denied in ("infra/prod.tfstate", "vault.kdbx", "keys/AuthKey.p8", "gcp-service-account.json"):
        assert is_denied_rel_path(denied), denied

    scrubbed = redact_secrets("ghp_1234567890abcdefghij AKIAIOSFODNN7EXAMPLE AIzaSyDummyKey1234567890123 xoxb-1234567890-abcdef")
    for raw_tok in ("ghp_1234567890", "AKIAIOSFODNN7EXAMPLE", "AIzaSyDummyKey", "xoxb-1234567890"):
        assert raw_tok not in scrubbed

    with pytest.raises(Exception, match="case-insensitive"):
        StartRunRequest(
            project_root=str(repo),
            tasks=[
                WorkerTaskInput(worker_id="WorkerA", folder="backend", goal="one"),
                WorkerTaskInput(worker_id="workera", folder="frontend", goal="two"),
            ],
        )

    # Deleted worker folder in rollback_run & write_worker_patch_sync records skipped_conflicts without crashing
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261122_000020_ffffff"
    b_path = proj_data / "backups" / run_id / "backend" / "api.py"
    atomic_write_bytes_sync(b_path, b"OLD = 1\n")
    worker_st = WorkerManifestState(
        worker_id="w1",
        folder="backend",
        goal="deleted folder test",
        requested_model="m:free",
        status="completed",
        modified_files={
            "backend/api.py": FileEditRecord(
                backup_path=str(b_path),
                initial_hash=sha256_bytes(b"OLD = 1\n"),
                worker_last_hash=sha256_bytes(b"NEW = 2\n"),
                added_lines=1,
                removed_lines=1,
            )
        },
        created_files={
            "backend/new_mod.py": CreatedFileRecord(
                worker_last_hash=sha256_bytes(b"X = 1\n"),
                line_count=1,
            )
        },
    )
    reg = RunRegistry()
    m_path = manifest_path_for(proj_data, run_id)
    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=999_999_999,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(proj_data / "runs" / run_id),
        workers={"w1": worker_st},
    )
    await save_manifest_atomic(m_path, manifest, reg.get_lock(repo, run_id))
    shutil.rmtree(repo / "backend")

    patch_path = write_worker_patch_sync(repo, m_path.parent, worker_st)
    assert patch_path is not None
    rb = await rollback_run(repo, run_id, reg)
    assert rb.status == "rolled_back"
    assert "backend/api.py" in rb.skipped_conflicts

    found = find_run_across_projects_sync(run_id)
    assert found is not None and found[1].run_id == run_id


def test_round3_ancestor_root_block_new_globs_precheck_redaction_and_reclaim_owner_check(
    isolated_env: tuple[Path, Path],
) -> None:
    from overseer.state import active_lock_path_for

    repo, data_dir = isolated_env
    # Ancestors of home_dir, ~/.gemini, or data_dir are rejected
    for bad_ancestor in (Path.home().parent, data_dir.parent, (Path.home() / ".gemini").parent):
        if bad_ancestor.parent != bad_ancestor:
            with pytest.raises(PathSecurityError):
                resolve_project_root(str(bad_ancestor))

    for denied in (".envrc", ".githooks/pre-commit", ".husky/pre-commit", ".pgpass", "keys/id.ppk", ".gemini/config"):
        assert is_denied_rel_path(denied), denied

    bad_py = repo / "backend" / "leak.py"
    bad_py.write_text('x = ("Bearer secret9999" +\n', encoding="utf-8")
    res = run_static_precheck(repo, ["backend/leak.py", "../outside.py"])
    assert res.status == "failed"
    joined = "\n".join(res.errors)
    assert "secret9999" not in joined
    assert "PathSecurityError" in joined

    # Reclaim check skips unlink when active.lock owner changed under sleep/race
    proj_data = get_project_data_dir(repo)
    lock_file = active_lock_path_for(proj_data)
    atomic_write_bytes_sync(
        lock_file,
        json.dumps({"run_id": "run_stale", "pid": 99999991, "pid_create_time": 100}).encode("utf-8"),
    )
    orig_read = __import__("overseer.state", fromlist=["read_active_lock_sync"]).read_active_lock_sync
    calls = 0

    def swapped_read(p_dir: Path):
        nonlocal calls
        calls += 1
        if calls >= 2:
            return {"run_id": f"run_live_peer_{calls}", "pid": os.getpid(), "pid_create_time": 200}
        return orig_read(p_dir)

    with (
        patch("overseer.state.is_lock_owner_alive", return_value=False),
        patch("overseer.state.read_active_lock_sync", side_effect=swapped_read),
    ):
        with pytest.raises(RunStateError, match="Could not acquire active run lock"):
            acquire_project_active_lock_sync(proj_data, "run_new", pid=os.getpid(), pid_create_time=300)
    assert lock_file.exists()


async def test_review_fixes_items_1_2_3_4_8_9_12(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    from overseer.state import (
        _build_monitor_snapshot_sync,
        _scan_candidate_manifests_sync,
        _win32_kernel32,
        active_lock_path_for,
    )

    repo, _ = isolated_env

    # Item 1: _build_monitor_snapshot_sync resolves project-root-relative paths for subfolder workers
    sub_dir = repo / "packages" / "api"
    sub_dir.mkdir(parents=True)
    (sub_dir / "server.py").write_text("X = 1\n", encoding="utf-8")
    m_sub = RunManifest(
        version="0.1.0",
        run_id="run_20261123_100000_abcdef",
        server_pid=os.getpid(),
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(get_project_data_dir(repo) / "runs" / "run_20261123_100000_abcdef"),
        workers={
            "w_sub": WorkerManifestState(
                worker_id="w_sub",
                folder="packages/api",
                goal="edit server",
                requested_model="m:free",
                status="completed",
                created_files={"packages/api/server.py": CreatedFileRecord(worker_last_hash=sha256_bytes(b"X = 1\n"), line_count=1)},
            )
        },
    )
    snap = _build_monitor_snapshot_sync(repo, m_sub)
    uri = snap["workers"][0]["touched_files"][0]["abs_uri"]
    assert uri.endswith("/packages/api/server.py")
    assert "/packages/api/packages/api/" not in uri

    # Item 2: _win32_kernel32 sets c_void_p restype on OpenProcess
    if sys.platform == "win32":
        k32 = _win32_kernel32()
        assert k32 is not None
        assert k32.OpenProcess.restype is ctypes.c_void_p

    # Item 3: shutdown_all keeps active.lock when run task did not finish
    reg = RunRegistry()
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261123_100001_abcdef"
    acquire_project_active_lock_sync(proj_data, run_id, pid=os.getpid())
    blocker = asyncio.Event()

    async def _stuck_task() -> None:
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            await asyncio.sleep(2.0)

    t = asyncio.create_task(_stuck_task())
    reg.register_run(repo, run_id)
    reg.attach_tasks(repo, run_id, [t])
    await asyncio.sleep(0)
    await reg.shutdown_all(timeout_sec=0.01)
    assert active_lock_path_for(proj_data).exists()
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)

    # Item 4: apply_replace_block raises BlockReplaceError on newline-only target
    with pytest.raises(BlockReplaceError, match="not found"):
        apply_replace_block(b"abc", "\n", "def")

    # Item 8: outline extractors ignore comments/tokens inside strings & raw strings
    cpp_src = (
        'const char* u = "https://example.com";\n'
        'const char* r = R"(/* inside raw)";\n'
        "int after_strings() { return 1; }\n"
    )
    assert "L3: int after_strings()" in extract_file_outline(cpp_src, "src/a.cpp")

    luau_src = (
        'local s = "--[["\n'
        "local raw = [[\n"
        "function fake_in_long()\n"
        "]]\n"
        "local function real_fn()\n"
        "end\n"
    )
    luau_out = extract_file_outline(luau_src, "src/a.luau")
    assert "L5: local function real_fn()" in luau_out
    assert "fake_in_long" not in luau_out

    ps1_src = (
        '# note @"\n'
        "function Real-AfterComment {\n"
        "}\n"
    )
    assert "L2: function Real-AfterComment" in extract_file_outline(ps1_src, "src/a.ps1")

    # Item 9: _scan_candidate_manifests_sync filters invalid dirs before slicing per_project_limit
    runs_dir = proj_data / "runs"
    valid_id = "run_20261123_100002_112233"
    valid_dir = runs_dir / valid_id
    valid_dir.mkdir(parents=True, exist_ok=True)
    m_valid = m_sub.model_copy(update={"run_id": valid_id, "patch_dir": str(valid_dir)})
    atomic_write_bytes_sync(valid_dir / "manifest.json", m_valid.model_dump_json().encode("utf-8"))
    (runs_dir / "zzz_not_a_run").mkdir(parents=True, exist_ok=True)
    (runs_dir / "run_20261123_999999_aaaaaa").mkdir(parents=True, exist_ok=True)  # no manifest.json
    found_manifests = _scan_candidate_manifests_sync(project_root=repo, per_project_limit=1)
    assert [m.run_id for _, m in found_manifests] == [valid_id]

    # Item 12: run_static_precheck skips files larger than MAX_VIEW_FILE_BYTES
    big_py = repo / "backend" / "big_gen.py"
    big_py.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr("overseer.guardrails.MAX_VIEW_FILE_BYTES", 5)
    precheck = run_static_precheck(repo, ["backend/big_gen.py"])
    assert precheck.status == "passed"
    assert precheck.errors == []


def test_prune_old_runs_respects_limit_and_active_runs(
    isolated_env: tuple[Path, Path],
) -> None:
    from overseer.state import prune_old_runs_sync

    repo, _ = isolated_env
    proj_data = get_project_data_dir(repo)

    run_specs = [
        ("run_20261124_000001_000001", "completed"),
        ("run_20261124_000002_000002", "running"),  # protected even though older
        ("run_20261124_000003_000003", "completed"),
        ("run_20261124_000004_000004", "completed"),
        ("run_20261124_000005_000005", "completed"),
    ]
    for rid, status in run_specs:
        rdir = proj_data / "runs" / rid
        bdir = proj_data / "backups" / rid
        rdir.mkdir(parents=True, exist_ok=True)
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "orig.py").write_text("x = 1\n", encoding="utf-8")
        m = RunManifest(
            version="0.1.0",
            run_id=rid,
            server_pid=os.getpid(),
            created_at_iso=utc_now_iso(),
            project_root=str(repo.resolve()),
            status=status,
            limits=RunLimits(),
            patch_dir=str(rdir),
            workers={},
        )
        atomic_write_bytes_sync(rdir / "manifest.json", m.model_dump_json().encode("utf-8"))

    # max_runs=0 disables pruning
    assert prune_old_runs_sync(proj_data, max_runs=0) == []

    # max_runs=2 keeps newest 2 (000005, 000004) + running (000002), prunes 000003 and 000001
    pruned = prune_old_runs_sync(proj_data, max_runs=2)
    assert set(pruned) == {"run_20261124_000003_000003", "run_20261124_000001_000001"}
    assert (proj_data / "runs" / "run_20261124_000005_000005").is_dir()
    assert (proj_data / "runs" / "run_20261124_000004_000004").is_dir()
    assert (proj_data / "runs" / "run_20261124_000002_000002").is_dir()
    assert not (proj_data / "runs" / "run_20261124_000003_000003").exists()
    assert not (proj_data / "backups" / "run_20261124_000003_000003").exists()
    assert not (proj_data / "runs" / "run_20261124_000001_000001").exists()
    assert not (proj_data / "backups" / "run_20261124_000001_000001").exists()


async def test_audit_nested_deny_dotenv_ruff_dir_conflict_rollback_and_inode_cache(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from overseer.state import _MANIFEST_STAT_CACHE, _load_manifest_cached_sync

    repo, data_dir = isolated_env

    # 1. Nested .github/workflows and .docker/config.json are denied at any depth
    assert is_denied_rel_path("packages/web/.github/workflows/deploy.yml")
    assert is_denied_rel_path("services/api/.docker/config.json")
    assert not is_denied_rel_path("packages/web/src/workflows/deploy.py")

    # 2. OVERSEER_DISABLE_RUFF=true in <data_dir>/.env disables optional ruff check
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".env").write_text("OVERSEER_DISABLE_RUFF=true\n", encoding="utf-8")
    monkeypatch.delenv("OVERSEER_DISABLE_RUFF", raising=False)
    with patch("overseer.guardrails.subprocess.run", side_effect=AssertionError("ruff should be skipped")):
        assert _run_optional_ruff_check("backend/a.py", "x = undefined_name\n") == []

    # 3. rollback_run handles file-replaced-by-directory conflicts without crashing
    proj_data = get_project_data_dir(repo)
    run_id = "run_20261125_000001_abcdef"
    run_dir = proj_data / "runs" / run_id
    backups_dir = proj_data / "backups" / run_id
    m_path = manifest_path_for(proj_data, run_id)

    good_mod = repo / "backend" / "good_mod.py"
    good_orig = b"x = 1\n"
    good_new = b"x = 2\n"
    good_mod.write_bytes(good_new)
    b_good = record_backup_once_sync(backups_dir, "backend/good_mod.py", good_orig)

    dir_over_mod = repo / "backend" / "dir_over_mod.py"
    dir_over_mod.mkdir(parents=True)
    b_dir_mod = record_backup_once_sync(backups_dir, "backend/dir_over_mod.py", b"m = 1\n")

    dir_over_created = repo / "backend" / "dir_over_created.py"
    dir_over_created.mkdir(parents=True)

    manifest = RunManifest(
        version="0.1.0",
        run_id=run_id,
        server_pid=999_999_999,
        created_at_iso=utc_now_iso(),
        project_root=str(repo.resolve()),
        status="completed",
        limits=RunLimits(),
        patch_dir=str(run_dir),
        workers={
            "w1": WorkerManifestState(
                worker_id="w1",
                folder="backend",
                goal="dir conflict",
                requested_model="m:free",
                status="completed",
                modified_files={
                    "backend/good_mod.py": FileEditRecord(
                        backup_path=str(b_good),
                        initial_hash=sha256_bytes(good_orig),
                        worker_last_hash=sha256_bytes(good_new),
                        added_lines=1,
                        removed_lines=1,
                    ),
                    "backend/dir_over_mod.py": FileEditRecord(
                        backup_path=str(b_dir_mod),
                        initial_hash=sha256_bytes(b"m = 1\n"),
                        worker_last_hash=sha256_bytes(b"m = 2\n"),
                        added_lines=1,
                        removed_lines=1,
                    ),
                },
                created_files={
                    "backend/dir_over_created.py": CreatedFileRecord(
                        worker_last_hash=sha256_bytes(b"c = 1\n"),
                        line_count=1,
                    )
                },
            )
        },
    )
    registry = RunRegistry()
    await save_manifest_atomic(m_path, manifest, registry.get_lock(repo, run_id))
    rb = await rollback_run(repo, run_id, registry)
    assert rb.restored_files == ["backend/good_mod.py"]
    assert good_mod.read_bytes() == good_orig
    assert set(rb.skipped_conflicts) == {"backend/dir_over_created.py", "backend/dir_over_mod.py"}

    # 4. _MANIFEST_STAT_CACHE invalidates when st_ino changes at identical (mtime_ns, size)
    _MANIFEST_STAT_CACHE.clear()
    manifest_a = manifest.model_copy(deep=True)
    manifest_a.status = "completed"
    manifest_b = manifest.model_copy(deep=True)
    manifest_b.status = "rolled_back"
    atomic_write_bytes_sync(m_path, manifest_a.model_dump_json().encode("utf-8"))
    size_bytes = len(manifest_a.model_dump_json().encode("utf-8"))

    fake_ino = [101]
    orig_stat = Path.stat

    def _fake_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == m_path:
            return SimpleNamespace(st_mtime_ns=999_999, st_size=size_bytes, st_ino=fake_ino[0], st_mode=0o100644)
        return orig_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", _fake_stat):
        assert _load_manifest_cached_sync(m_path).status == "completed"
        atomic_write_bytes_sync(m_path, manifest_b.model_dump_json().encode("utf-8"))
        fake_ino[0] = 202
        assert _load_manifest_cached_sync(m_path).status == "rolled_back"


def test_cross_platform_backslash_and_macos_var_folders_policy(
    isolated_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from overseer import guardrails as guardrails_mod

    repo, _ = isolated_env
    api_file = repo / "backend" / "api.py"
    api_file.write_text("ok = True\n", encoding="utf-8")

    # Relative backslash paths resolve identically across POSIX and Windows
    assert resolve_scoped_path(repo, repo, r"backend\api.py", require_exists=True) == api_file.resolve()

    # On macOS (darwin), per-user temp repos under /private/var/folders/... are allowed,
    # while /private/var/folders itself and other /var or OS system dirs stay rejected.
    with monkeypatch.context() as mp:
        mp.setattr(guardrails_mod.sys, "platform", "darwin")
        var_folders_repo = Path("/var/folders/ab/cd1234/T/pytest-0/repo").resolve(strict=False)
        guardrails_mod._validate_project_root_policy(var_folders_repo)
        for forbidden in (
            Path("/var/folders").resolve(strict=False),
            Path("/var/db/myproj").resolve(strict=False),
            Path("/var/root/myproj").resolve(strict=False),
            Path("/etc/myproj").resolve(strict=False),
        ):
            with pytest.raises(PathSecurityError):
                guardrails_mod._validate_project_root_policy(forbidden)

    # On Linux, /var/folders/... must remain rejected as part of /var
    with monkeypatch.context() as mp:
        mp.setattr(guardrails_mod.sys, "platform", "linux")
        with pytest.raises(PathSecurityError):
            guardrails_mod._validate_project_root_policy(Path("/var/folders/ab/cd1234/T/pytest-0/repo").resolve(strict=False))

