"""Multi-project run registry, external manifest/patch persistence, bounded reports, and rollback."""

from __future__ import annotations

import asyncio
import ctypes
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from overseer.config import get_config_value, get_max_concurrent_workers
from overseer.guardrails import (
    PathSecurityError,
    _normcase_path,
    _win32_kernel32,
    atomic_write_bytes,
    atomic_write_bytes_sync,
    get_data_dir,
    get_project_data_dir,
    is_normcase_relative_to,
    is_pid_alive,
    read_bytes_retry_sync,
    read_text_retry_sync,
    resolve_scoped_path,
    sha256_bytes,
    unlink_retry_sync,
    validate_historical_project_root,
)
from overseer.models import (
    RUN_ID_PATTERN,
    CancelRunResponse,
    GetReportResponse,
    RollbackResponse,
    RunManifest,
    WorkerManifestState,
    WorkerReportItem,
)

_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
_logger = logging.getLogger(__name__)


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def get_pid_create_time(pid: int) -> int | None:
    """Return OS process start timestamp (Windows FILETIME, Linux /proc starttime, or macOS ps lstart) or None."""
    if pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            rparen = stat_text.rfind(")")
            if rparen != -1:
                fields = stat_text[rparen + 2 :].split()
                if len(fields) > 19:
                    return int(fields[19])
        except (OSError, ValueError):
            return None
        return None
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                normalized = " ".join(proc.stdout.strip().split())
                dt = datetime.strptime(normalized, "%a %b %d %H:%M:%S %Y")
                return int(dt.timestamp())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return None
    if sys.platform != "win32":
        return None
    kernel32 = _win32_kernel32()
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int

    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        creation = _FileTime()
        exit_t = _FileTime()
        kernel_t = _FileTime()
        user_t = _FileTime()
        if kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        ):
            return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return None
    finally:
        kernel32.CloseHandle(handle)


def is_lock_owner_alive(pid: int, expected_create_time: int | None = None) -> bool:
    """Return True if pid is alive and (when verifiable) matches expected_create_time."""
    if not is_pid_alive(pid):
        return False
    if expected_create_time is not None:
        actual_create_time = get_pid_create_time(pid)
        if actual_create_time is not None and actual_create_time != expected_create_time:
            return False
    return True


def utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def generate_run_id() -> str:
    """Generate a unique timestamped run_id matching RUN_ID_PATTERN."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    rand_hex = os.urandom(3).hex()
    return f"run_{stamp}_{rand_hex}"


class RunStateError(ValueError):
    """Raised on invalid run IDs, conflicting active runs, or disallowed rollbacks."""


class RunRegistry:
    """Coordinates active multi-project runs, per-run locks, and cancellation state."""

    def __init__(self) -> None:
        self._semaphore_limit = get_max_concurrent_workers()
        self.worker_semaphore = asyncio.Semaphore(self._semaphore_limit)
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._report_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._active_by_project: dict[str, str] = {}
        self._tasks_by_run: dict[tuple[str, str], list[asyncio.Task[None]]] = {}
        self._coord_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._commit_tasks: dict[tuple[str, str], set[asyncio.Task[Any]]] = {}
        self._cancel_events: dict[tuple[str, str], asyncio.Event] = {}
        self._run_paths: dict[tuple[str, str], tuple[Path, Path]] = {}
        self._user_cancelled: set[tuple[str, str]] = set()
        self._interrupted: set[tuple[str, str]] = set()

    @staticmethod
    def _project_key(project_root: Path) -> str:
        resolved = str(project_root.resolve(strict=False))
        if sys.platform == "darwin":
            return os.path.normcase(resolved).lower()
        return os.path.normcase(resolved)

    def _get_scoped_lock(
        self,
        store: dict[tuple[str, str], asyncio.Lock],
        project_root: Path,
        run_id: str,
    ) -> asyncio.Lock:
        key = (self._project_key(project_root), run_id)
        if key not in store:
            if len(store) > 128:
                active_keys = set(self._active_by_project.items())
                for old_key, old_lock in list(store.items()):
                    if old_key not in active_keys and not old_lock.locked():
                        store.pop(old_key, None)
                        if len(store) <= 128:
                            break
            store[key] = asyncio.Lock()
        return store[key]

    def get_worker_semaphore(self) -> asyncio.Semaphore:
        desired = get_max_concurrent_workers()
        if desired != self._semaphore_limit and not self._active_by_project:
            self._semaphore_limit = desired
            self.worker_semaphore = asyncio.Semaphore(desired)
        return self.worker_semaphore

    def get_lock(self, project_root: Path, run_id: str) -> asyncio.Lock:
        return self._get_scoped_lock(self._locks, project_root, run_id)

    def get_report_lock(self, project_root: Path, run_id: str) -> asyncio.Lock:
        return self._get_scoped_lock(self._report_locks, project_root, run_id)

    def attach_run_paths(
        self,
        project_root: Path,
        run_id: str,
        project_data_dir: Path,
        manifest_path: Path,
    ) -> None:
        self._run_paths[(self._project_key(project_root), run_id)] = (project_data_dir, manifest_path)

    def has_active_run(self, project_root: Path) -> str | None:
        return self._active_by_project.get(self._project_key(project_root))

    def is_run_active(self, project_root: Path, run_id: str) -> bool:
        return self._active_by_project.get(self._project_key(project_root)) == run_id

    def register_run(self, project_root: Path, run_id: str) -> asyncio.Event:
        proj_key = self._project_key(project_root)
        existing = self._active_by_project.get(proj_key)
        if existing is not None:
            raise RunStateError(
                f"Active run {existing!r} is already running for project_root {project_root}"
            )
        self._active_by_project[proj_key] = run_id
        cancel_event = asyncio.Event()
        self._cancel_events[(proj_key, run_id)] = cancel_event
        return cancel_event

    def get_cancel_event(self, project_root: Path, run_id: str) -> asyncio.Event | None:
        return self._cancel_events.get((self._project_key(project_root), run_id))

    def mark_user_cancelled(self, project_root: Path, run_id: str) -> None:
        self._user_cancelled.add((self._project_key(project_root), run_id))

    def is_user_cancelled(self, project_root: Path, run_id: str) -> bool:
        return (self._project_key(project_root), run_id) in self._user_cancelled

    def is_interrupted(self, project_root: Path, run_id: str) -> bool:
        return (self._project_key(project_root), run_id) in self._interrupted

    def attach_tasks(self, project_root: Path, run_id: str, tasks: list[asyncio.Task[None]]) -> None:
        self._tasks_by_run[(self._project_key(project_root), run_id)] = list(tasks)

    def get_tasks(self, project_root: Path, run_id: str) -> list[asyncio.Task[None]]:
        return list(self._tasks_by_run.get((self._project_key(project_root), run_id), ()))

    def attach_coordinator(self, project_root: Path, run_id: str, coord_task: asyncio.Task[None]) -> None:
        self._coord_tasks[(self._project_key(project_root), run_id)] = coord_task

    def get_coordinator(self, project_root: Path, run_id: str) -> asyncio.Task[None] | None:
        return self._coord_tasks.get((self._project_key(project_root), run_id))

    def track_commit(self, project_root: Path, run_id: str, commit_task: asyncio.Task[Any]) -> None:
        key = (self._project_key(project_root), run_id)
        bucket = self._commit_tasks.setdefault(key, set())
        bucket.add(commit_task)
        commit_task.add_done_callback(lambda t, k=key: self._commit_tasks.get(k, set()).discard(t))

    async def await_commits(self, project_root: Path, run_id: str) -> None:
        key = (self._project_key(project_root), run_id)
        while True:
            pending = [t for t in self._commit_tasks.get(key, set()) if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

    def mark_run_finished(self, project_root: Path, run_id: str) -> None:
        proj_key = self._project_key(project_root)
        key = (proj_key, run_id)
        if self._active_by_project.get(proj_key) == run_id:
            self._active_by_project.pop(proj_key, None)
        self._tasks_by_run.pop(key, None)
        self._coord_tasks.pop(key, None)
        self._commit_tasks.pop(key, None)
        self._cancel_events.pop(key, None)
        self._run_paths.pop(key, None)
        self._user_cancelled.discard(key)
        self._interrupted.discard(key)
        for store in (self._locks, self._report_locks):
            lock = store.get(key)
            if lock is None or not lock.locked():
                store.pop(key, None)

    async def shutdown_all(self, timeout_sec: float = 5.0) -> None:
        all_workers: list[asyncio.Task[None]] = []
        all_coords: list[asyncio.Task[None]] = []
        for key, event in list(self._cancel_events.items()):
            if not event.is_set():
                self._interrupted.add(key)
            event.set()
        for task_list in self._tasks_by_run.values():
            for task in task_list:
                if not task.done():
                    task.cancel()
                    all_workers.append(task)
        for coord in self._coord_tasks.values():
            if not coord.done():
                all_coords.append(coord)

        deadline = time.monotonic() + timeout_sec
        if all_workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_workers, return_exceptions=True),
                    timeout=max(0.01, deadline - time.monotonic()),
                )
            except TimeoutError:
                _logger.warning("Timed out waiting for %d worker tasks on shutdown", len(all_workers))

        pending_commits = [
            t for commit_set in list(self._commit_tasks.values()) for t in commit_set if not t.done()
        ]
        if pending_commits:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_commits, return_exceptions=True),
                    timeout=max(0.05, deadline - time.monotonic()),
                )
            except TimeoutError:
                _logger.warning("Timed out waiting for %d pending commit tasks on shutdown", len(pending_commits))
                for task in pending_commits:
                    if not task.done():
                        task.cancel()

        if all_coords:
            remaining_coord = max(0.05, deadline - time.monotonic())
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_coords, return_exceptions=True),
                    timeout=remaining_coord,
                )
            except TimeoutError:
                _logger.warning("Timed out waiting for %d coordinator tasks on shutdown", len(all_coords))
                for coord in all_coords:
                    if not coord.done():
                        coord.cancel()

        for (proj_key, run_id), (proj_data_dir, m_path) in list(self._run_paths.items()):
            if self._active_by_project.get(proj_key) != run_id:
                continue
            key = (proj_key, run_id)
            workers_done = all(t.done() for t in self._tasks_by_run.get(key, ()))
            commits_done = all(t.done() for t in self._commit_tasks.get(key, set()))
            if workers_done and commits_done:
                try:
                    lock = self._locks.setdefault(key, asyncio.Lock())
                    async with lock:
                        if m_path.is_file():
                            manifest = await asyncio.to_thread(load_manifest_sync, m_path)
                            if manifest.status == "running":
                                for worker_st in manifest.workers.values():
                                    if worker_st.status in {"queued", "running"}:
                                        worker_st.status = "interrupted"
                                        worker_st.stop_reason = worker_st.stop_reason or "server_shutdown"
                                manifest.status = "interrupted"
                                manifest.completed_at_iso = manifest.completed_at_iso or utc_now_iso()
                                await write_manifest_locked(m_path, manifest)
                except (RunStateError, OSError, ValueError) as exc:
                    _logger.debug("Best-effort shutdown manifest cleanup failed for %s: %s", run_id, exc)
                try:
                    await asyncio.to_thread(release_project_active_lock_sync, proj_data_dir, run_id)
                except OSError:
                    pass

        self._active_by_project.clear()
        self._tasks_by_run.clear()
        self._coord_tasks.clear()
        self._commit_tasks.clear()
        self._cancel_events.clear()
        self._run_paths.clear()
        self._user_cancelled.clear()
        self._interrupted.clear()
        self._locks.clear()
        self._report_locks.clear()


def validate_and_resolve_run_id(project_data_dir: Path, run_id: str) -> str:
    """Resolve 'latest' or validate an explicit run_id against RUN_ID_PATTERN."""
    if run_id == "latest":
        latest_file = project_data_dir / "latest_run.txt"
        if latest_file.exists():
            candidate = read_text_retry_sync(latest_file, encoding="utf-8").strip()
            if _RUN_ID_RE.match(candidate) and (project_data_dir / "runs" / candidate / "manifest.json").exists():
                return candidate
        runs_dir = project_data_dir / "runs"
        if runs_dir.is_dir():
            valid_dirs = sorted(
                (d.name for d in runs_dir.iterdir() if d.is_dir() and _RUN_ID_RE.match(d.name) and (d / "manifest.json").exists()),
                reverse=True,
            )
            if valid_dirs:
                return valid_dirs[0]
        raise RunStateError(f"No runs found under {project_data_dir}")

    if not _RUN_ID_RE.match(run_id):
        raise RunStateError(f"Invalid run_id format: {run_id!r}")
    return run_id


def manifest_path_for(project_data_dir: Path, run_id: str) -> Path:
    return project_data_dir / "runs" / run_id / "manifest.json"


def active_lock_path_for(project_data_dir: Path) -> Path:
    return project_data_dir / "active.lock"


def cancel_flag_path_for(run_dir: Path) -> Path:
    return run_dir / "cancel.flag"


def read_active_lock_sync(project_data_dir: Path) -> dict[str, Any] | None:
    lock_path = active_lock_path_for(project_data_dir)
    for attempt in range(3):
        if not lock_path.exists():
            return None
        try:
            raw = read_text_retry_sync(lock_path, encoding="utf-8")
            if not raw.strip() and attempt < 2:
                time.sleep(0.02)
                continue
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("run_id"), str) and isinstance(data.get("pid"), int):
                return data
            return None
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError):
            if attempt < 2:
                time.sleep(0.02)
                continue
            return None
    return None


def acquire_project_active_lock_sync(
    project_data_dir: Path,
    run_id: str,
    pid: int,
    pid_create_time: int | None = None,
) -> None:
    """Create <project_data_dir>/active.lock atomically, reclaiming stale dead-PID locks."""
    lock_path = active_lock_path_for(project_data_dir)
    reclaim_path = project_data_dir / "active.lock.reclaim"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_create_time = get_pid_create_time(pid) if pid_create_time is None else pid_create_time
    payload = json.dumps(
        {"run_id": run_id, "pid": pid, "pid_create_time": resolved_create_time}
    ).encode("utf-8")

    def _try_acquire_reclaim_mutex() -> bool:
        for _ in range(2):
            try:
                rfd = os.open(reclaim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(rfd, str(pid).encode("ascii", errors="ignore"))
                finally:
                    os.close(rfd)
                return True
            except FileExistsError:
                try:
                    if (time.time() - reclaim_path.stat().st_mtime) > 5.0:
                        reclaim_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.02)
                return False
        return False

    for _attempt in range(5):
        tmp_path = lock_path.parent / f".{lock_path.name}.{uuid4().hex[:8]}.tmp"
        try:
            tmp_path.write_bytes(payload)
            try:
                os.link(tmp_path, lock_path)
                return
            except FileExistsError:
                raise
            except (AttributeError, NotImplementedError, PermissionError, OSError):
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                return
        except FileExistsError:
            got_reclaim = _try_acquire_reclaim_mutex()
            try:
                existing = read_active_lock_sync(project_data_dir)
                stale_sig: tuple[Any, Any, Any] | None = None
                should_unlink = False
                if existing is not None:
                    ex_pid = existing["pid"]
                    ex_run = existing["run_id"]
                    raw_ct = existing.get("pid_create_time")
                    ex_create_time = raw_ct if isinstance(raw_ct, int) and not isinstance(raw_ct, bool) else None
                    stale_sig = (ex_run, ex_pid, raw_ct)
                    if is_lock_owner_alive(ex_pid, ex_create_time):
                        ex_mpath = manifest_path_for(project_data_dir, ex_run)
                        is_terminal = False
                        if ex_mpath.exists():
                            try:
                                ex_manifest = load_manifest_sync(ex_mpath)
                                is_terminal = ex_manifest.status != "running"
                            except (RunStateError, OSError, ValueError):
                                is_terminal = False
                        if not is_terminal:
                            raise RunStateError(
                                f"Active run {ex_run!r} is already running in PID {ex_pid}."
                            )
                    should_unlink = True
                else:
                    try:
                        # Avoid unlinking a lock file created within the last 1.0s while its writer is initializing
                        should_unlink = (time.time() - lock_path.stat().st_mtime) > 1.0
                    except OSError:
                        should_unlink = False

                if got_reclaim and should_unlink:
                    # Verify lock owner did not change while stalled past reclaim TTL before unlinking
                    current = read_active_lock_sync(project_data_dir)
                    same_owner = (
                        current is None
                        if stale_sig is None
                        else (
                            current is not None
                            and (current.get("run_id"), current.get("pid"), current.get("pid_create_time"))
                            == stale_sig
                        )
                    )
                    if same_owner:
                        try:
                            unlink_retry_sync(lock_path)
                        except FileNotFoundError:
                            pass
                else:
                    time.sleep(0.08)
            finally:
                if got_reclaim:
                    try:
                        reclaim_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            continue
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    raise RunStateError(f"Could not acquire active run lock at {lock_path}")


def release_project_active_lock_sync(project_data_dir: Path, run_id: str | None = None) -> None:
    lock_path = active_lock_path_for(project_data_dir)
    if not lock_path.exists():
        return
    if run_id is not None:
        existing = read_active_lock_sync(project_data_dir)
        if existing is not None and existing.get("run_id") != run_id:
            return
    try:
        unlink_retry_sync(lock_path)
    except FileNotFoundError:
        pass


async def write_manifest_locked(manifest_path: Path, manifest: RunManifest) -> None:
    """Serialize and atomically write RunManifest JSON (caller must hold the run lock)."""
    await atomic_write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))


async def save_manifest_atomic(
    manifest_path: Path,
    manifest: RunManifest,
    lock: asyncio.Lock,
) -> None:
    """Serialize and atomically write RunManifest JSON under lock."""
    async with lock:
        await write_manifest_locked(manifest_path, manifest)


async def write_latest_run_pointer(project_data_dir: Path, run_id: str) -> None:
    await atomic_write_bytes(project_data_dir / "latest_run.txt", run_id.encode("utf-8"))


def load_manifest_sync(manifest_path: Path) -> RunManifest:
    if not manifest_path.exists():
        raise RunStateError(f"Run manifest not found: {manifest_path}")
    return RunManifest.model_validate_json(read_text_retry_sync(manifest_path, encoding="utf-8"))


async def _reconcile_manifest_liveness_locked(
    manifest: RunManifest,
    manifest_path: Path,
    project_root: Path,
    registry: RunRegistry,
) -> RunManifest:
    """Transition stale 'running' workers/runs to 'interrupted' (caller must hold run lock)."""
    if manifest.status != "running":
        return manifest
    if registry.is_run_active(project_root, manifest.run_id):
        return manifest
    if manifest.server_pid == os.getpid():
        # Same process ID but not in registry -> interrupted leftover
        alive_in_registry = False
    else:
        alive_in_registry = is_lock_owner_alive(manifest.server_pid, manifest.server_pid_create_time)

    if alive_in_registry:
        return manifest

    for worker in manifest.workers.values():
        if worker.status in {"queued", "running"}:
            worker.status = "interrupted"
            worker.stop_reason = worker.stop_reason or "process_terminated"
    manifest.status = "interrupted"
    manifest.completed_at_iso = manifest.completed_at_iso or utc_now_iso()
    await atomic_write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
    proj_data_dir = manifest_path.parents[2] if len(manifest_path.parents) >= 3 else get_project_data_dir(project_root)
    await asyncio.to_thread(release_project_active_lock_sync, proj_data_dir, manifest.run_id)
    return manifest


async def reconcile_manifest_liveness(
    manifest: RunManifest,
    manifest_path: Path,
    project_root: Path,
    registry: RunRegistry,
) -> RunManifest:
    """Transition stale 'running' workers/runs to 'interrupted' if not actively running."""
    if manifest.status != "running":
        return manifest
    if registry.is_run_active(project_root, manifest.run_id):
        return manifest
    lock = registry.get_lock(project_root, manifest.run_id)
    async with lock:
        return await _reconcile_manifest_liveness_locked(manifest, manifest_path, project_root, registry)


def record_backup_once_sync(backups_dir: Path, rel_posix: str, raw_bytes: bytes) -> Path:
    """Persist pre-run file bytes once under external backups_dir/<rel_posix>."""
    backup_target = backups_dir / rel_posix
    if not backup_target.exists():
        atomic_write_bytes_sync(backup_target, raw_bytes)
    return backup_target


_MAX_PATCH_FILE_BYTES = 2 * 1024 * 1024


def _read_scoped_patch_text_sync(project_root: Path, folder_abs: Path | None, rel_path: str) -> str:
    if folder_abs is None or not folder_abs.is_dir():
        return ""
    try:
        disk_file = resolve_scoped_path(
            project_root=project_root,
            allowed_root=folder_abs,
            raw_path=rel_path,
            require_exists=False,
            allow_new_parents=True,
        )
    except (PathSecurityError, OSError):
        return ""
    if not disk_file.is_file():
        return ""
    try:
        if disk_file.stat().st_size > _MAX_PATCH_FILE_BYTES:
            return "[skipped unified diff: file exceeds 2 MiB]\n"
        return read_bytes_retry_sync(disk_file).decode("utf-8", errors="replace")
    except OSError:
        return ""


_T = TypeVar("_T")


def _resolve_worker_folder(project_root: Path, folder: str, *, strict_dir: bool = True) -> Path | None:
    try:
        folder_candidate = resolve_scoped_path(
            project_root=project_root,
            allowed_root=project_root,
            raw_path=folder,
            require_exists=False,
            allow_new_parents=True,
        )
        if strict_dir:
            return folder_candidate.resolve(strict=True) if folder_candidate.is_dir() else None
        return folder_candidate.resolve(strict=False)
    except (PathSecurityError, OSError):
        return None


def _shrink_to_fit(
    target: _T,
    max_bytes: int,
    to_jsonable: Callable[[_T], Any],
    stages: tuple[Callable[[_T], None], ...],
) -> _T:
    def _size(obj: _T) -> int:
        return len(json.dumps(to_jsonable(obj), separators=(",", ":")).encode("utf-8"))

    if _size(target) <= max_bytes:
        return target
    for shrink_fn in stages:
        shrink_fn(target)
        if _size(target) <= max_bytes:
            return target
    return target


async def _load_reconciled_manifest_locked(
    m_path: Path,
    project_root: Path,
    registry: RunRegistry,
) -> RunManifest:
    manifest = await asyncio.to_thread(load_manifest_sync, m_path)
    return await _reconcile_manifest_liveness_locked(manifest, m_path, project_root, registry)


def write_worker_patch_sync(
    project_root: Path,
    run_dir: Path,
    worker: WorkerManifestState,
) -> str | None:
    """Compute unified diff patch for a worker's edits/creations; return path or None if empty."""
    chunks: list[str] = []
    folder_abs = _resolve_worker_folder(project_root, worker.folder, strict_dir=True)
    backups_root = (
        run_dir.parent.parent / "backups"
        if run_dir.parent.name == "runs"
        else get_project_data_dir(project_root) / "backups"
    ).resolve(strict=False)

    for rel_path in sorted(worker.modified_files):
        record = worker.modified_files[rel_path]
        backup_file = Path(record.backup_path)
        if not is_normcase_relative_to(backup_file.resolve(strict=False), backups_root):
            chunks.append(
                f"--- a/{rel_path}\n+++ b/{rel_path}\n[skipped unified diff: backup path outside backups directory]\n"
            )
            continue
        if backup_file.exists():
            try:
                if backup_file.stat().st_size > _MAX_PATCH_FILE_BYTES:
                    chunks.append(f"--- a/{rel_path}\n+++ b/{rel_path}\n[skipped unified diff: backup exceeds 2 MiB]\n")
                    continue
                old_text = read_bytes_retry_sync(backup_file).decode("utf-8", errors="replace")
            except OSError:
                old_text = ""
        else:
            old_text = ""
        new_text = _read_scoped_patch_text_sync(project_root, folder_abs, rel_path)
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        if diff_lines:
            chunks.append("".join(diff_lines))

    for rel_path in sorted(worker.created_files):
        new_text = _read_scoped_patch_text_sync(project_root, folder_abs, rel_path)
        diff_lines = list(
            difflib.unified_diff(
                [],
                new_text.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{rel_path}",
            )
        )
        if diff_lines:
            chunks.append("".join(diff_lines))

    if not chunks:
        return None

    patch_path = run_dir / f"{worker.worker_id}.patch"
    atomic_write_bytes_sync(patch_path, "\n".join(chunks).encode("utf-8"))
    return str(patch_path)


def read_worker_patch_sync(
    project_root: Path,
    run_id: str,
    worker_id: str,
    max_bytes: int = 65536,
) -> str:
    """Safely read a bounded UTF-8 unified diff patch for a worker in a run."""
    if not worker_id or "/" in worker_id or "\\" in worker_id or worker_id in {".", ".."}:
        raise RunStateError(f"Invalid worker_id for patch read: {worker_id!r}")
    project_data_dir = get_project_data_dir(project_root)
    resolved_run_id = validate_and_resolve_run_id(project_data_dir, run_id)
    run_dir = (project_data_dir / "runs" / resolved_run_id).resolve(strict=False)
    patch_path = resolve_scoped_path(
        project_root=run_dir,
        allowed_root=run_dir,
        raw_path=f"{worker_id}.patch",
        require_exists=False,
        allow_new_parents=False,
    )
    if not patch_path.is_file():
        m_path = manifest_path_for(project_data_dir, resolved_run_id)
        if m_path.is_file():
            try:
                manifest = load_manifest_sync(m_path)
                if manifest.status != "rolled_back":
                    worker = manifest.workers.get(worker_id)
                    if worker is not None and (worker.modified_files or worker.created_files):
                        write_worker_patch_sync(project_root, run_dir, worker)
            except (RunStateError, OSError, UnicodeDecodeError, ValueError, PathSecurityError):
                pass
        if not patch_path.is_file():
            return ""
    budget = max(1, min(int(max_bytes), _MAX_PATCH_FILE_BYTES))
    with patch_path.open("rb") as f:
        raw = f.read(budget + 1)
    truncated = len(raw) > budget
    if truncated:
        raw = raw[:budget]
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... [truncated at {budget} bytes]"
    return text


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def build_bounded_report(
    project_root: Path,
    manifest: RunManifest,
    run_dir: Path,
    max_bytes: int = 8192,
) -> GetReportResponse:
    """Write untruncated report_full.json to disk and return a response <= max_bytes UTF-8."""
    worker_items: list[WorkerReportItem] = []
    for worker_id in sorted(manifest.workers):
        worker = manifest.workers[worker_id]
        if manifest.status == "rolled_back":
            existing_patch = run_dir / f"{worker.worker_id}.patch"
            patch_path = str(existing_patch) if existing_patch.exists() else None
        else:
            patch_path = write_worker_patch_sync(project_root, run_dir, worker)
        diffstat: dict[str, str] = {}
        for rel_path, edit_rec in sorted(worker.modified_files.items()):
            diffstat[rel_path] = f"+{edit_rec.added_lines} / -{edit_rec.removed_lines}"
        for rel_path, create_rec in sorted(worker.created_files.items()):
            diffstat[rel_path] = f"+{create_rec.line_count} (created)"

        worker_items.append(
            WorkerReportItem(
                worker_id=worker.worker_id,
                status=worker.status,
                steps_used=worker.steps_used,
                models_used=list(worker.models_used),
                spend_usd=round(worker.spend_usd, 6),
                cached_tokens=worker.cached_tokens,
                diffstat=diffstat,
                static_check=worker.static_check.model_copy(deep=True),
                summary=worker.summary,
                notes=list(worker.notes),
                escalation_reason=worker.escalation_reason,
                stop_reason=worker.stop_reason,
                patch_path=patch_path,
            )
        )

    full_report_path = run_dir / "report_full.json"
    full_response = GetReportResponse(
        run_id=manifest.run_id,
        project_root=manifest.project_root,
        status=manifest.status,
        spend_usd=round(manifest.spend_usd, 6),
        cached_tokens=manifest.cached_tokens,
        patch_dir=manifest.patch_dir,
        full_report_path=str(full_report_path),
        truncated=False,
        workers=worker_items,
    )
    full_report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes_sync(full_report_path, full_response.model_dump_json(indent=2).encode("utf-8"))

    def _step1(resp: GetReportResponse) -> None:
        for w in resp.workers:
            w.summary = _truncate_text(w.summary, 200)
            w.notes = [_truncate_text(n, 200) for n in w.notes[:5]]
            if w.escalation_reason:
                w.escalation_reason = _truncate_text(w.escalation_reason, 200)

    def _step2(resp: GetReportResponse) -> None:
        for w in resp.workers:
            if len(w.diffstat) > 10:
                kept = dict(list(w.diffstat.items())[:10])
                kept["..."] = f"+{len(w.diffstat) - 10} more files"
                w.diffstat = kept
            w.static_check.errors = [_truncate_text(e, 180) for e in w.static_check.errors[:3]]

    def _step3(resp: GetReportResponse) -> None:
        for w in resp.workers:
            w.diffstat = {}
            w.notes = []
            w.summary = ""
            w.static_check.errors = []
            if w.escalation_reason:
                w.escalation_reason = _truncate_text(w.escalation_reason, 80)

    def _step4(resp: GetReportResponse) -> None:
        resp.workers = []

    if len(json.dumps(full_response.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")) <= max_bytes:
        return full_response
    bounded = full_response.model_copy(deep=True)
    bounded.truncated = True
    return _shrink_to_fit(
        bounded,
        max_bytes,
        lambda r: r.model_dump(mode="json"),
        (_step1, _step2, _step3, _step4),
    )


def _prune_empty_dirs(start_dir: Path, stop_root: Path) -> None:
    cur = start_dir
    norm_stop = _normcase_path(stop_root.resolve())
    while True:
        norm_cur = _normcase_path(cur.resolve(strict=False))
        if norm_cur == norm_stop or not is_normcase_relative_to(cur, stop_root):
            break
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


async def collect_run_report(
    project_root: Path,
    run_id: str,
    registry: RunRegistry,
    max_bytes: int = 8192,
) -> GetReportResponse:
    """Load run state under per-run lock, reconcile liveness, and build a bounded report."""
    project_data_dir = get_project_data_dir(project_root)
    resolved_run_id = await asyncio.to_thread(validate_and_resolve_run_id, project_data_dir, run_id)
    m_path = manifest_path_for(project_data_dir, resolved_run_id)
    lock = registry.get_lock(project_root, resolved_run_id)
    async with lock:
        manifest = await _load_reconciled_manifest_locked(m_path, project_root, registry)
        manifest_snapshot = manifest.model_copy(deep=True)
    run_dir = m_path.parent
    report_lock = registry.get_report_lock(project_root, resolved_run_id)
    async with report_lock:
        return await asyncio.to_thread(build_bounded_report, project_root, manifest_snapshot, run_dir, max_bytes)


async def rollback_run(
    project_root: Path,
    run_id: str,
    registry: RunRegistry,
) -> RollbackResponse:
    """Safely revert a non-running run without overwriting post-run manual edits."""
    project_data_dir = get_project_data_dir(project_root)
    resolved_run_id = await asyncio.to_thread(validate_and_resolve_run_id, project_data_dir, run_id)
    m_path = manifest_path_for(project_data_dir, resolved_run_id)
    lock = registry.get_lock(project_root, resolved_run_id)
    restored_files: list[str] = []
    deleted_files: list[str] = []
    skipped_conflicts: list[str] = []

    async with lock:
        manifest = await _load_reconciled_manifest_locked(m_path, project_root, registry)

        if manifest.status == "running" or registry.is_run_active(project_root, resolved_run_id):
            raise RunStateError(f"Cannot rollback run {resolved_run_id!r} while status is 'running'.")

        if manifest.status == "rolled_back":
            return RollbackResponse(run_id=resolved_run_id, already_rolled_back=True)

        for worker in manifest.workers.values():
            folder_abs = _resolve_worker_folder(project_root, worker.folder, strict_dir=True)

            if folder_abs is None:
                for rel_path in sorted(worker.modified_files):
                    skipped_conflicts.append(rel_path)
                continue

            for rel_path, created_rec in sorted(worker.created_files.items()):
                try:
                    disk_path = resolve_scoped_path(
                        project_root=project_root,
                        allowed_root=folder_abs,
                        raw_path=rel_path,
                        require_exists=False,
                        allow_new_parents=True,
                    )
                except PathSecurityError:
                    skipped_conflicts.append(rel_path)
                    continue
                if not disk_path.exists():
                    continue
                if not disk_path.is_file():
                    skipped_conflicts.append(rel_path)
                    continue
                current_hash = await asyncio.to_thread(
                    lambda p=disk_path: sha256_bytes(read_bytes_retry_sync(p))
                )
                if current_hash == created_rec.worker_last_hash:
                    await asyncio.to_thread(unlink_retry_sync, disk_path)
                    if worker.created_dirs is None:
                        await asyncio.to_thread(_prune_empty_dirs, disk_path.parent, folder_abs)
                    deleted_files.append(rel_path)
                else:
                    skipped_conflicts.append(rel_path)

            if worker.created_dirs is not None:
                for rel_dir in sorted(set(worker.created_dirs), key=lambda p: p.count("/"), reverse=True):
                    try:
                        dir_path = resolve_scoped_path(
                            project_root=project_root,
                            allowed_root=folder_abs,
                            raw_path=rel_dir,
                            require_exists=False,
                            allow_new_parents=True,
                        )
                    except PathSecurityError:
                        continue
                    if dir_path.is_dir() and dir_path != folder_abs and is_normcase_relative_to(dir_path, folder_abs):
                        try:
                            await asyncio.to_thread(dir_path.rmdir)
                        except OSError:
                            pass

            backups_root = (project_data_dir / "backups").resolve(strict=False)
            for rel_path, edit_rec in sorted(worker.modified_files.items()):
                try:
                    disk_path = resolve_scoped_path(
                        project_root=project_root,
                        allowed_root=folder_abs,
                        raw_path=rel_path,
                        require_exists=False,
                        allow_new_parents=True,
                    )
                except PathSecurityError:
                    skipped_conflicts.append(rel_path)
                    continue
                backup_path = Path(edit_rec.backup_path)
                if not is_normcase_relative_to(backup_path.resolve(strict=False), backups_root):
                    _logger.warning("Backup path outside backups directory during rollback for %s", rel_path)
                    skipped_conflicts.append(rel_path)
                    continue
                if not disk_path.is_file() or not backup_path.is_file():
                    skipped_conflicts.append(rel_path)
                    continue
                current_hash = await asyncio.to_thread(
                    lambda p=disk_path: sha256_bytes(read_bytes_retry_sync(p))
                )
                if current_hash == edit_rec.initial_hash:
                    # Already restored (e.g., retrying after a partial rollback failure).
                    continue
                if current_hash == edit_rec.worker_last_hash:
                    backup_bytes = await asyncio.to_thread(read_bytes_retry_sync, backup_path)
                    if sha256_bytes(backup_bytes) != edit_rec.initial_hash:
                        _logger.warning("Backup SHA-256 mismatch during rollback for %s", rel_path)
                        skipped_conflicts.append(rel_path)
                        continue
                    await atomic_write_bytes(disk_path, backup_bytes)
                    restored_files.append(rel_path)
                else:
                    skipped_conflicts.append(rel_path)

        manifest.status = "rolled_back"
        manifest.completed_at_iso = manifest.completed_at_iso or utc_now_iso()
        await write_manifest_locked(m_path, manifest)

    return RollbackResponse(
        run_id=resolved_run_id,
        restored_files=restored_files,
        deleted_files=deleted_files,
        skipped_conflicts=skipped_conflicts,
        already_rolled_back=False,
    )


async def cancel_active_run(
    project_root: Path,
    run_id: str,
    registry: RunRegistry,
) -> CancelRunResponse:
    """Stop queued and running workers for an active run cleanly without reverting edits."""
    project_data_dir = get_project_data_dir(project_root)
    resolved_run_id = await asyncio.to_thread(validate_and_resolve_run_id, project_data_dir, run_id)
    m_path = manifest_path_for(project_data_dir, resolved_run_id)
    lock = registry.get_lock(project_root, resolved_run_id)

    already_cancelling = False
    async with lock:
        manifest = await _load_reconciled_manifest_locked(m_path, project_root, registry)

        if not registry.is_run_active(project_root, resolved_run_id):
            if (
                manifest.status == "running"
                and manifest.server_pid != os.getpid()
                and is_lock_owner_alive(manifest.server_pid, manifest.server_pid_create_time)
            ):
                await atomic_write_bytes(
                    cancel_flag_path_for(m_path.parent),
                    utc_now_iso().encode("utf-8"),
                )
                return CancelRunResponse(
                    run_id=resolved_run_id,
                    project_root=str(project_root),
                    status=manifest.status,
                    cancelled_workers=[],
                    already_finished=False,
                )
            return CancelRunResponse(
                run_id=resolved_run_id,
                project_root=str(project_root),
                status=manifest.status,
                cancelled_workers=[],
                already_finished=True,
            )

        if manifest.status != "running":
            return CancelRunResponse(
                run_id=resolved_run_id,
                project_root=str(project_root),
                status=manifest.status,
                cancelled_workers=[],
                already_finished=True,
            )

        if registry.is_user_cancelled(project_root, resolved_run_id):
            already_cancelling = True
            coord_task = registry.get_coordinator(project_root, resolved_run_id)
            worker_tasks = registry.get_tasks(project_root, resolved_run_id)
        else:
            registry.mark_user_cancelled(project_root, resolved_run_id)
            cancel_ev = registry.get_cancel_event(project_root, resolved_run_id)
            if cancel_ev is not None:
                cancel_ev.set()

            worker_tasks = registry.get_tasks(project_root, resolved_run_id)
            for task in worker_tasks:
                if not task.done():
                    task.cancel()

            coord_task = registry.get_coordinator(project_root, resolved_run_id)

    if already_cancelling:
        if coord_task is not None:
            await asyncio.gather(coord_task, return_exceptions=True)
        elif worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        post_lock = registry.get_lock(project_root, resolved_run_id)
        async with post_lock:
            final_manifest = await asyncio.to_thread(load_manifest_sync, m_path)
        return CancelRunResponse(
            run_id=resolved_run_id,
            project_root=str(project_root),
            status=final_manifest.status,
            cancelled_workers=[],
            already_finished=True,
        )

    if coord_task is not None:
        await asyncio.gather(coord_task, return_exceptions=True)
    else:
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        mid_lock = registry.get_lock(project_root, resolved_run_id)
        async with mid_lock:
            manifest = await asyncio.to_thread(load_manifest_sync, m_path)
            for worker in manifest.workers.values():
                if worker.status in ("queued", "running"):
                    worker.status = "cancelled"
                    worker.stop_reason = "cancelled_by_user"
            manifest.status = "cancelled"
            manifest.completed_at_iso = manifest.completed_at_iso or utc_now_iso()
            await write_manifest_locked(m_path, manifest)
        await asyncio.to_thread(release_project_active_lock_sync, project_data_dir, resolved_run_id)
        registry.mark_run_finished(project_root, resolved_run_id)

    final_lock = registry.get_lock(project_root, resolved_run_id)
    async with final_lock:
        final_manifest = await asyncio.to_thread(load_manifest_sync, m_path)
    cancelled_workers = sorted(
        wid for wid, w in final_manifest.workers.items() if w.status == "cancelled"
    )
    return CancelRunResponse(
        run_id=resolved_run_id,
        project_root=str(project_root),
        status=final_manifest.status,
        cancelled_workers=cancelled_workers,
        already_finished=False,
    )


_MANIFEST_STAT_CACHE: OrderedDict[str, tuple[int, int, int, RunManifest]] = OrderedDict()
_MANIFEST_CACHE_LOCK = threading.Lock()


def _load_manifest_cached_sync(m_path: Path, *, copy: bool = True) -> RunManifest:
    st = m_path.stat()
    key = str(m_path)
    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_STAT_CACHE.get(key)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size and cached[2] == st.st_ino:
            _MANIFEST_STAT_CACHE.move_to_end(key)
            return cached[3].model_copy(deep=True) if copy else cached[3]
    manifest = load_manifest_sync(m_path)
    with _MANIFEST_CACHE_LOCK:
        if len(_MANIFEST_STAT_CACHE) >= 2048:
            _MANIFEST_STAT_CACHE.popitem(last=False)
        _MANIFEST_STAT_CACHE[key] = (st.st_mtime_ns, st.st_size, st.st_ino, manifest)
    return manifest.model_copy(deep=True) if copy else manifest


def _scan_candidate_manifests_sync(
    project_root: Path | None,
    per_project_limit: int | None = None,
    *,
    copy: bool = True,
) -> list[tuple[Path, RunManifest]]:
    """Discover and parse valid RunManifest files for one project or all stored projects."""
    if project_root is not None:
        runs_dirs = [get_project_data_dir(project_root) / "runs"]
    else:
        projects_root = get_data_dir() / "projects"
        if not projects_root.is_dir():
            return []
        try:
            proj_children = list(projects_root.iterdir())
        except OSError:
            return []
        runs_dirs = [p / "runs" for p in proj_children if p.is_dir()]

    loaded: list[tuple[Path, RunManifest]] = []
    for runs_dir in runs_dirs:
        if not runs_dir.is_dir():
            continue
        try:
            run_entries = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
        except OSError:
            continue
        seen_ids: set[str] = set()
        valid_dirs = [
            run_dir
            for run_dir in run_entries
            if run_dir.is_dir() and _RUN_ID_RE.match(run_dir.name) and (run_dir / "manifest.json").is_file()
        ]
        if per_project_limit is not None:
            valid_dirs = valid_dirs[: max(0, per_project_limit)]
        for run_dir in valid_dirs:
            m_path = run_dir / "manifest.json"
            try:
                manifest = _load_manifest_cached_sync(m_path, copy=copy)
            except (RunStateError, OSError, UnicodeDecodeError, ValueError):
                continue
            seen_ids.add(run_dir.name)
            loaded.append((m_path, manifest))

        if per_project_limit is not None:
            lock_data = read_active_lock_sync(runs_dir.parent)
            locked_run_id = lock_data.get("run_id") if lock_data else None
            if (
                isinstance(locked_run_id, str)
                and locked_run_id not in seen_ids
                and _RUN_ID_RE.match(locked_run_id)
            ):
                m_path = runs_dir / locked_run_id / "manifest.json"
                if m_path.is_file():
                    try:
                        manifest = _load_manifest_cached_sync(m_path, copy=copy)
                    except (RunStateError, OSError, UnicodeDecodeError, ValueError):
                        pass
                    else:
                        loaded.append((m_path, manifest))
    return loaded


def find_run_across_projects_sync(run_id: str) -> tuple[Path, RunManifest] | None:
    """Locate an explicit run_id across all stored project directories by direct path check."""
    if not _RUN_ID_RE.match(run_id):
        return None
    projects_root = get_data_dir() / "projects"
    if not projects_root.is_dir():
        return None
    try:
        proj_children = list(projects_root.iterdir())
    except OSError:
        return None
    for proj_dir in proj_children:
        if not proj_dir.is_dir():
            continue
        m_path = proj_dir / "runs" / run_id / "manifest.json"
        if not m_path.is_file():
            continue
        try:
            manifest = _load_manifest_cached_sync(m_path)
            return m_path, manifest
        except (RunStateError, OSError, UnicodeDecodeError, ValueError):
            continue
    return None


def _run_summary_from_manifest(manifest: RunManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "project_root": manifest.project_root,
        "status": manifest.status,
        "created_at_iso": manifest.created_at_iso,
        "completed_at_iso": manifest.completed_at_iso,
        "spend_usd": round(manifest.spend_usd, 6),
        "max_run_usd": manifest.limits.max_run_usd,
        "worker_count": len(manifest.workers),
        "completed_workers": sum(1 for w in manifest.workers.values() if w.status == "completed"),
    }


async def list_all_runs_snapshot(
    registry: RunRegistry,
    project_root: Path | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List recent runs across one project or all stored projects, reconciling stale active runs."""
    effective_limit = max(0, min(int(limit), 200))
    effective_offset = max(0, int(offset))
    per_project_cap = max(effective_offset + effective_limit, 25)
    candidates = await asyncio.to_thread(_scan_candidate_manifests_sync, project_root, per_project_cap, copy=False)

    summaries: list[dict[str, Any]] = []
    for m_path, manifest in candidates:
        try:
            target_root = (
                project_root
                if project_root is not None
                else validate_historical_project_root(manifest.project_root)
            )
        except (PathSecurityError, OSError, ValueError):
            continue
        if manifest.status == "running":
            try:
                if not registry.is_run_active(target_root, manifest.run_id):
                    lock = registry.get_lock(target_root, manifest.run_id)
                    async with lock:
                        manifest = await _load_reconciled_manifest_locked(m_path, target_root, registry)
            except (RunStateError, OSError, UnicodeDecodeError, ValueError, RuntimeError):
                continue
        summaries.append(_run_summary_from_manifest(manifest))

    summaries.sort(key=lambda item: (item["created_at_iso"], item["run_id"]), reverse=True)
    active_count = sum(1 for item in summaries if item["status"] == "running")
    total_count = len(summaries)
    page = summaries[effective_offset : effective_offset + effective_limit]
    has_more = (effective_offset + len(page)) < total_count
    return {
        "runs": page,
        "active_count": active_count,
        "total_count": total_count,
        "has_more": has_more,
    }


def _safe_file_uri(resolved_root: Path, rel_path: str) -> str:
    """Resolve a repository-relative path safely to a file:/// URI, returning '' if invalid."""
    if not resolved_root.exists():
        return ""
    try:
        target = resolve_scoped_path(
            project_root=resolved_root,
            allowed_root=resolved_root,
            raw_path=rel_path,
            require_exists=False,
            allow_new_parents=True,
        )
        return target.as_uri()
    except (PathSecurityError, OSError, ValueError):
        return ""


def _build_monitor_snapshot_sync(
    project_root: Path,
    manifest: RunManifest,
    max_bytes: int = 524_288,
) -> dict[str, Any]:
    """Build a bounded UI-ready snapshot dictionary for a single run manifest."""
    resolved_root = project_root.resolve(strict=False)
    workers_payload: list[dict[str, Any]] = []
    for worker_id in sorted(manifest.workers):
        worker = manifest.workers[worker_id]
        touched_files: list[dict[str, str]] = []
        for rel_path, edit_rec in sorted(worker.modified_files.items()):
            touched_files.append(
                {
                    "rel_path": rel_path,
                    "abs_uri": _safe_file_uri(resolved_root, rel_path),
                    "kind": "modified",
                    "stat": f"+{edit_rec.added_lines} / -{edit_rec.removed_lines}",
                }
            )
        for rel_path, create_rec in sorted(worker.created_files.items()):
            touched_files.append(
                {
                    "rel_path": rel_path,
                    "abs_uri": _safe_file_uri(resolved_root, rel_path),
                    "kind": "created",
                    "stat": f"+{create_rec.line_count} (created)",
                }
            )
        touched_files.sort(key=lambda item: item["rel_path"])

        workers_payload.append(
            {
                "worker_id": worker.worker_id,
                "folder": worker.folder,
                "status": worker.status,
                "steps_used": worker.steps_used,
                "max_steps": manifest.limits.max_steps_per_worker,
                "models_used": list(worker.models_used),
                "requested_model": worker.requested_model,
                "prompt_tokens": worker.prompt_tokens,
                "completion_tokens": worker.completion_tokens,
                "cached_tokens": worker.cached_tokens,
                "spend_usd": round(worker.spend_usd, 6),
                "static_check": {
                    "status": worker.static_check.status,
                    "errors": [_truncate_text(e, 300) for e in worker.static_check.errors[:20]],
                },
                "summary": _truncate_text(worker.summary, 2000),
                "notes": [_truncate_text(n, 300) for n in worker.notes[:20]],
                "escalation_reason": (
                    _truncate_text(worker.escalation_reason, 500) if worker.escalation_reason else None
                ),
                "stop_reason": worker.stop_reason,
                "touched_files": touched_files[:200],
            }
        )

    snapshot: dict[str, Any] = {
        "run_id": manifest.run_id,
        "project_root": manifest.project_root,
        "status": manifest.status,
        "created_at_iso": manifest.created_at_iso,
        "completed_at_iso": manifest.completed_at_iso,
        "spend_usd": round(manifest.spend_usd, 6),
        "cached_tokens": manifest.cached_tokens,
        "limits": manifest.limits.model_dump(mode="json"),
        "workers": workers_payload,
    }

    def _snap_step1(payload: dict[str, Any]) -> None:
        for w in payload["workers"]:
            w["summary"] = _truncate_text(w["summary"], 200)
            w["notes"] = [_truncate_text(n, 150) for n in w["notes"][:5]]
            w["static_check"]["errors"] = [_truncate_text(e, 150) for e in w["static_check"]["errors"][:3]]
            w["touched_files"] = w["touched_files"][:30]

    def _snap_step2(payload: dict[str, Any]) -> None:
        for w in payload["workers"]:
            w["summary"] = ""
            w["notes"] = []
            w["static_check"]["errors"] = []
            w["touched_files"] = w["touched_files"][:10]

    def _snap_step3(payload: dict[str, Any]) -> None:
        for w in payload["workers"]:
            w["touched_files"] = []
            w["escalation_reason"] = _truncate_text(w["escalation_reason"], 80) if w["escalation_reason"] else None

    def _snap_step4(payload: dict[str, Any]) -> None:
        payload["workers"] = []

    return _shrink_to_fit(
        snapshot,
        max_bytes,
        lambda p: p,
        (_snap_step1, _snap_step2, _snap_step3, _snap_step4),
    )


async def get_run_monitor_snapshot(
    project_root: Path,
    run_id: str,
    registry: RunRegistry,
    max_bytes: int = 524_288,
) -> dict[str, Any]:
    """Load and reconcile a run manifest (supporting 'latest') into a rich UI monitor snapshot."""
    project_data_dir = get_project_data_dir(project_root)
    resolved_run_id = await asyncio.to_thread(validate_and_resolve_run_id, project_data_dir, run_id)
    m_path = manifest_path_for(project_data_dir, resolved_run_id)
    lock = registry.get_lock(project_root, resolved_run_id)

    async with lock:
        manifest = await _load_reconciled_manifest_locked(m_path, project_root, registry)

    return await asyncio.to_thread(_build_monitor_snapshot_sync, project_root, manifest, max_bytes)


DEFAULT_MAX_RUNS_PER_PROJECT = 50


def prune_old_runs_sync(
    project_data_dir: Path,
    max_runs: int | None = None,
    active_run_id: str | None = None,
) -> list[str]:
    """Delete oldest finished runs and backups beyond max_runs (<= 0 disables pruning)."""
    if max_runs is None:
        raw = get_config_value("OVERSEER_MAX_RUNS_PER_PROJECT")
        if raw:
            try:
                max_runs = int(raw)
            except ValueError:
                max_runs = DEFAULT_MAX_RUNS_PER_PROJECT
        else:
            max_runs = DEFAULT_MAX_RUNS_PER_PROJECT
    if max_runs <= 0:
        return []

    runs_dir = project_data_dir / "runs"
    if not runs_dir.is_dir():
        return []
    try:
        entries = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir() and _RUN_ID_RE.match(d.name)),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return []
    if len(entries) <= max_runs:
        return []

    protected: set[str] = set()
    if active_run_id:
        protected.add(active_run_id)
    lock_data = read_active_lock_sync(project_data_dir)
    if lock_data and isinstance(lock_data.get("run_id"), str):
        protected.add(lock_data["run_id"])

    pruned: list[str] = []
    kept = 0
    backups_root = project_data_dir / "backups"
    for run_dir in entries:
        rid = run_dir.name
        if kept < max_runs or rid in protected:
            kept += 1
            continue
        m_path = run_dir / "manifest.json"
        if m_path.is_file():
            try:
                if load_manifest_sync(m_path).status == "running":
                    kept += 1
                    continue
            except (RunStateError, OSError, UnicodeDecodeError, ValueError):
                pass
        shutil.rmtree(run_dir, ignore_errors=True)
        backup_dir = backups_root / rid
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        with _MANIFEST_CACHE_LOCK:
            _MANIFEST_STAT_CACHE.pop(str(m_path), None)
        pruned.append(rid)
    return pruned
