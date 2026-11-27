"""8-tool worker execution loop, prioritized history pruning, and run coordinator."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import math
import os
import posixpath
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import ValidationError

from overseer import __version__
from overseer.config import (
    DEFAULT_CHEAP_FALLBACK_SLUGS,
    DEFAULT_CHEAP_MODEL_SLUG,
    DEFAULT_FREE_FALLBACK_SLUGS,
    DEFAULT_FREE_MODEL_SLUG,
    DEFAULT_OPENROUTER_BASE_URL,
    MODEL_SLUG_ALIASES,
    build_model_chain,
    get_cheap_default_model,
    get_cheap_fallback_models,
    get_config_value,
    get_default_model,
    get_fallback_models,
    get_heavy_default_model,
    get_heavy_fallback_models,
    get_openrouter_base_url,
    infer_tier_for_request,
    load_custom_instructions_sync,
    load_external_dotenv,
    resolve_effective_max_run_usd,
    select_api_key_for_model,
)
from overseer.guardrails import (
    IGNORED_DIRS_LOWER,
    MAX_VIEW_FILE_BYTES,
    BlockReplaceError,
    PathSecurityError,
    apply_replace_block,
    atomic_write_bytes,
    extract_file_outline,
    get_project_data_dir,
    is_denied_rel_path,
    is_normcase_relative_to,
    is_reparse_point,
    read_bytes_retry_sync,
    redact_secrets,
    resolve_scoped_path,
    run_static_precheck,
    sha256_bytes,
    to_rel_posix,
    validate_non_overlapping_folders,
)
from overseer.models import (
    CreatedFileRecord,
    FileEditRecord,
    FileOutlineArgs,
    FindFileArgs,
    FinishArgs,
    ListDirArgs,
    ReplaceBlockArgs,
    RunLimits,
    RunManifest,
    RunStatus,
    SearchDirArgs,
    StartRunRequest,
    StartRunResponse,
    TierName,
    ViewFileArgs,
    WorkerManifestState,
    WorkerTaskInput,
    WriteNewFileArgs,
)
from overseer.notify import send_completion_toast, should_suppress_auto_toast
from overseer.openrouter import (
    NON_RETRYABLE_STATUS_CODES,
    RETRYABLE_STATUS_CODES,
    WORKER_TOOL_SCHEMAS,
    ModelCatalog,
    ModelMetadata,
    _call_openrouter_with_failover,
    _post_with_cancel_flag_poll,
    _WorkerCancelledError,
    parse_retry_after_seconds,
)
from overseer.state import (
    RunRegistry,
    RunStateError,
    acquire_project_active_lock_sync,
    cancel_flag_path_for,
    generate_run_id,
    get_pid_create_time,
    manifest_path_for,
    prune_old_runs_sync,
    record_backup_once_sync,
    release_project_active_lock_sync,
    utc_now_iso,
    write_latest_run_pointer,
    write_manifest_locked,
    write_worker_patch_sync,
)

__all__ = [
    "DEFAULT_CHEAP_FALLBACK_SLUGS",
    "DEFAULT_CHEAP_MODEL_SLUG",
    "DEFAULT_FREE_FALLBACK_SLUGS",
    "DEFAULT_FREE_MODEL_SLUG",
    "DEFAULT_OPENROUTER_BASE_URL",
    "MODEL_SLUG_ALIASES",
    "NON_RETRYABLE_STATUS_CODES",
    "RETRYABLE_STATUS_CODES",
    "WORKER_TOOL_SCHEMAS",
    "ModelCatalog",
    "ModelMetadata",
    "build_model_chain",
    "get_cheap_default_model",
    "get_cheap_fallback_models",
    "get_config_value",
    "get_default_model",
    "get_fallback_models",
    "get_heavy_default_model",
    "get_heavy_fallback_models",
    "get_openrouter_base_url",
    "load_external_dotenv",
    "parse_retry_after_seconds",
    "select_api_key_for_model",
    "_post_with_cancel_flag_poll",
]
MAX_VIEW_BYTES = 65_536
MAX_VIEW_LINES = 300
MAX_LIST_ENTRIES = 200
MAX_SEARCH_MATCHES = 50
MAX_SEARCH_LINE_CHARS = 200
MAX_SEARCH_FILE_BYTES = 1_048_576
MAX_OUTLINE_FILE_BYTES = 1_048_576
MAX_FIND_PATHS = 100
MAX_CONSECUTIVE_ERRORS = 3
MAX_EMPTY_TURNS = 2
READONLY_TOOLS: frozenset[str] = frozenset(
    {"view_file", "file_outline", "list_dir", "search_dir", "find_file"}
)
_READONLY_ARG_MODELS: dict[str, type[Any]] = {
    "view_file": ViewFileArgs,
    "file_outline": FileOutlineArgs,
    "list_dir": ListDirArgs,
    "search_dir": SearchDirArgs,
    "find_file": FindFileArgs,
}
MAX_REDUNDANT_READ_TURNS = 3
POST_WRITE_NUDGE_TURN = 4
MAX_POST_WRITE_READONLY_TURNS = 8
KEEP_RECENT_TOOLS = 12
_TOOL_ERROR_PREFIXES: tuple[str, ...] = (
    "Tool error",
    "Static pre-check failed",
    "Tool argument JSON error",
    "Tool rejected",
    "Tool skipped",
)

_logger = logging.getLogger(__name__)
_BG_TOAST_TASKS: set[asyncio.Task[Any]] = set()


def _on_toast_done(task: asyncio.Task[Any]) -> None:
    _BG_TOAST_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except BaseException as err:
        _logger.debug("Toast task cleanup error: %s", redact_secrets(str(err)))
        return
    if exc is not None:
        _logger.debug("Windows completion toast failed: %s", redact_secrets(str(exc)))


async def drain_toast_tasks(timeout_sec: float = 2.0) -> None:
    """Await or cancel pending background toast tasks before server shutdown."""
    pending = [t for t in list(_BG_TOAST_TASKS) if not t.done()]
    if not pending:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_sec)
    except TimeoutError:
        for task in pending:
            if not task.done():
                task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=0.5)
        except TimeoutError:
            _logger.debug("Timed out waiting for %d cancelled toast tasks during shutdown", len(pending))


def _parse_tool_call_meta(messages: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    call_meta: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not tc_id or not name:
                continue
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (TypeError, json.JSONDecodeError):
                args = {}
            call_meta[tc_id] = (name, args)
    return call_meta


def _normalize_tool_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    if not cleaned:
        return "."
    normalized = posixpath.normpath(cleaned)
    return normalized or "."


def _slice_subsumed(old_start: int, old_end: int, new_start: int, new_end: int) -> bool:
    return new_start <= old_start and old_end <= new_end


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _message_char_len(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += len(str(msg.get("content", "")))
        for tc in msg.get("tool_calls") or ():
            fn = tc.get("function") or {}
            total += len(str(fn.get("arguments", "")))
    return total


_VIEW_HEADER_RANGE_RE = re.compile(r"\(lines (\d+)\.\.(\d+) of \d+\)")


def prune_message_history(
    messages: list[dict[str, Any]],
    keep_recent_tools: int = 6,
    max_history_chars: int = 120_000,
) -> list[dict[str, Any]]:
    """Collapse older tool outputs to keep <= keep_recent_tools unpruned and cap total char size."""
    tool_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "tool"]
    if len(tool_indices) <= keep_recent_tools and _message_char_len(messages) <= max_history_chars:
        return [dict(m) for m in messages]

    call_meta = _parse_tool_call_meta(messages)
    tool_outputs_by_id: dict[str, str] = {}
    for idx in tool_indices:
        tc_id = messages[idx].get("tool_call_id")
        if tc_id:
            tool_outputs_by_id[tc_id] = str(messages[idx].get("content", ""))

    latest_tool_idx = tool_indices[-1] if tool_indices else -1
    older_candidates = tool_indices[:-1] if tool_indices else []

    # Track view_file line slices from newest to oldest to identify fully subsumed reads
    seen_views_by_path: dict[str, list[tuple[int, int]]] = {}
    superseded_indices: set[int] = set()
    for idx in reversed(tool_indices):
        tc_id = messages[idx].get("tool_call_id", "")
        name, args = call_meta.get(tc_id, ("tool", {}))
        if name == "view_file":
            raw_path = str(args.get("path", ""))
            path = _normalize_tool_path(raw_path) if raw_path.strip() else ""
            s_line = _safe_int(args.get("start_line"), 1)
            e_line = min(_safe_int(args.get("end_line"), s_line + MAX_VIEW_LINES - 1), s_line + MAX_VIEW_LINES - 1)
            prior_ranges = seen_views_by_path.setdefault(path, [])
            if any(_slice_subsumed(s_line, e_line, ps, pe) for ps, pe in prior_ranges):
                if idx != latest_tool_idx:
                    superseded_indices.add(idx)
            content_str = str(messages[idx].get("content", ""))
            if not content_str.startswith(_TOOL_ERROR_PREFIXES):
                eff_start, eff_end = s_line, e_line
                if "[Output capped at " in content_str:
                    hdr_m = _VIEW_HEADER_RANGE_RE.search(content_str)
                    if hdr_m:
                        eff_start = int(hdr_m.group(1))
                        eff_end = int(hdr_m.group(2))
                    else:
                        eff_end = eff_start - 1
                if eff_end >= eff_start:
                    prior_ranges.append((eff_start, eff_end))

    tier1_exploratory: list[int] = []
    tier2_other: list[int] = []
    tier3_active_views: list[int] = []

    for idx in older_candidates:
        tc_id = messages[idx].get("tool_call_id", "")
        name, _ = call_meta.get(tc_id, ("tool", {}))
        content_str = str(messages[idx].get("content", ""))
        if idx in superseded_indices or name in {"list_dir", "search_dir", "find_file", "file_outline"}:
            tier1_exploratory.append(idx)
        elif name == "view_file" or (name == "finish" and content_str.startswith("Static pre-check failed:")):
            tier3_active_views.append(idx)
        else:
            tier2_other.append(idx)

    to_prune_count = max(0, len(tool_indices) - keep_recent_tools)
    prune_order = tier1_exploratory + tier2_other + tier3_active_views
    to_prune = set(prune_order[:to_prune_count])
    last_assistant_idx = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)

    def _classify_write_compaction(
        msg_idx: int, tc_id: str, name: str | None, raw_args: str, include_older_failed: bool
    ) -> str | None:
        if len(raw_args) <= 2048 or name not in {"write_new_file", "replace_block"}:
            return None
        if not tc_id or tc_id not in tool_outputs_by_id:
            return None
        paired_out = tool_outputs_by_id[tc_id]
        if not paired_out.startswith(_TOOL_ERROR_PREFIXES):
            return "succeeded"
        if include_older_failed and msg_idx < last_assistant_idx:
            return "failed"
        return None

    def _pruned_placeholder(msg: dict[str, Any]) -> str:
        tc_id = msg.get("tool_call_id", "")
        name, args = call_meta.get(tc_id, ("tool", {}))
        if name == "view_file":
            s_line = _safe_int(args.get("start_line"), 1)
            default_end = s_line + MAX_VIEW_LINES - 1
            e_line = min(_safe_int(args.get("end_line"), default_end), default_end)
            detail = f"{args.get('path', '?')} lines {s_line}-{e_line}"
        elif name in {"list_dir", "search_dir", "find_file", "file_outline"}:
            detail = str(args.get("path", args.get("query", args.get("pattern", "."))))
        else:
            detail = str(args.get("path", ""))
        return f"[pruned {name} {detail}]".strip()

    def _apply_pruning(
        prune_set: set[int],
        compact_old_writes: bool = False,
        compact_older_failed_writes: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            copied = dict(msg)
            if idx in prune_set and not str(copied.get("content", "")).startswith("[pruned "):
                copied["content"] = _pruned_placeholder(copied)
            elif compact_old_writes and copied.get("role") == "assistant" and copied.get("tool_calls"):
                compacted_tcs = []
                for tc in copied["tool_calls"]:
                    fn = dict(tc.get("function") or {})
                    fname = fn.get("name")
                    raw_args = str(fn.get("arguments", ""))
                    tc_id = str(tc.get("id", ""))
                    outcome = _classify_write_compaction(idx, tc_id, fname, raw_args, compact_older_failed_writes)
                    if outcome is not None:
                        _, parsed_args = call_meta.get(tc_id, ("tool", {}))
                        path_val = str(parsed_args.get("path") or "?")
                        if fname == "write_new_file":
                            placeholder = (
                                "[written to disk; compacted]"
                                if outcome == "succeeded"
                                else "[failed write; compacted]"
                            )
                            fn["arguments"] = json.dumps({"path": path_val, "content": placeholder})
                        else:
                            placeholder = "[compacted]" if outcome == "succeeded" else "[failed edit; compacted]"
                            fn["arguments"] = json.dumps(
                                {"path": path_val, "target": placeholder, "replacement": placeholder}
                            )
                    compacted_tcs.append({**tc, "function": fn})
                copied["tool_calls"] = compacted_tcs
            out.append(copied)
        return out

    def _prune_indices_in_place(
        pruned: list[dict[str, Any]],
        cur_chars: int,
        candidates: list[int],
    ) -> int:
        for extra_idx in candidates:
            if cur_chars <= max_history_chars:
                break
            if extra_idx in to_prune:
                continue
            to_prune.add(extra_idx)
            orig = str(pruned[extra_idx].get("content", ""))
            if not orig.startswith("[pruned "):
                placeholder = _pruned_placeholder(pruned[extra_idx])
                pruned[extra_idx]["content"] = placeholder
                cur_chars += len(placeholder) - len(orig)
        return cur_chars

    pruned_messages = _apply_pruning(to_prune)
    cur_chars = _message_char_len(pruned_messages)
    if cur_chars > max_history_chars:
        cur_chars = _prune_indices_in_place(pruned_messages, cur_chars, tier1_exploratory + tier2_other)
    if cur_chars > max_history_chars:
        pruned_messages = _apply_pruning(to_prune, compact_old_writes=True)
        cur_chars = _message_char_len(pruned_messages)
    if cur_chars > max_history_chars:
        pruned_messages = _apply_pruning(
            to_prune, compact_old_writes=True, compact_older_failed_writes=True
        )
        cur_chars = _message_char_len(pruned_messages)
    if cur_chars > max_history_chars:
        _prune_indices_in_place(pruned_messages, cur_chars, tier3_active_views)
    return pruned_messages


@dataclass(slots=True)
class WorkerToolContext:
    """Mutable per-worker runtime state passed to tool handlers."""

    project_root: Path
    folder_abs: Path
    backups_dir: Path
    run_dir: Path
    manifest_path: Path
    manifest: RunManifest
    worker: WorkerManifestState
    run_lock: asyncio.Lock
    viewed_hashes: dict[str, str] = field(default_factory=dict)
    last_read_file: tuple[str, str] | None = None
    last_readonly_sig: str | None = None
    finished_cleanly: bool = False
    registry: RunRegistry | None = None
    cancel_event: asyncio.Event | None = None


def _view_file_sync(
    project_root: Path,
    folder_abs: Path,
    args: ViewFileArgs,
) -> tuple[str, str | None, str | None]:
    if args.end_line is None:
        args.end_line = args.start_line + MAX_VIEW_LINES - 1
    if args.end_line < args.start_line:
        raise BlockReplaceError(
            f"Invalid line range: end_line ({args.end_line}) < start_line ({args.start_line})."
        )
    target = resolve_scoped_path(
        project_root=project_root,
        allowed_root=project_root,
        raw_path=args.path,
        require_exists=True,
    )
    if not target.is_file():
        raise PathSecurityError(f"Path is not a file: {args.path!r}")
    file_size = target.stat().st_size
    if file_size > MAX_VIEW_FILE_BYTES:
        raise BlockReplaceError(
            f"File exceeds {MAX_VIEW_FILE_BYTES} bytes ({file_size} bytes): {args.path!r}; file is too large for view_file."
        )

    raw_bytes = read_bytes_retry_sync(target)
    if b"\x00" in raw_bytes[:8192]:
        raise BlockReplaceError(f"Binary file cannot be viewed: {args.path!r}")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BlockReplaceError(f"Non-UTF-8 file cannot be viewed: {args.path!r}") from exc

    lines = text.splitlines()
    total_lines = len(lines)
    start_idx = max(0, args.start_line - 1)
    max_end = min(args.end_line, args.start_line + MAX_VIEW_LINES - 1, total_lines)
    if start_idx >= total_lines and (total_lines > 0 or args.start_line > 1):
        return f"(File {args.path} has {total_lines} lines; requested start_line {args.start_line} is past EOF)", None, None

    out_chunks: list[str] = []
    byte_total = 0
    last_included = args.start_line - 1
    truncated_by_bytes = False
    for line_idx in range(start_idx, max_end):
        formatted = f"{line_idx + 1}: {lines[line_idx]}\n"
        encoded_len = len(formatted.encode("utf-8"))
        if out_chunks and byte_total + encoded_len > MAX_VIEW_BYTES:
            truncated_by_bytes = True
            break
        if encoded_len > MAX_VIEW_BYTES:
            # Single ultra-long line (e.g. minified file): trim cleanly to byte cap
            clipped = formatted.encode("utf-8")[: MAX_VIEW_BYTES - 32].decode("utf-8", errors="ignore")
            out_chunks.append(clipped + "... [line truncated at 64 KB]\n")
            last_included = line_idx + 1
            truncated_by_bytes = True
            break
        out_chunks.append(formatted)
        byte_total += encoded_len
        last_included = line_idx + 1

    header = f"# {to_rel_posix(target, project_root)} (lines {args.start_line}..{last_included} of {total_lines})\n"
    if truncated_by_bytes:
        header += f"[Output capped at {MAX_VIEW_BYTES} bytes]\n"

    rel_posix = to_rel_posix(target, project_root)
    rel_hash: str | None = sha256_bytes(raw_bytes) if is_normcase_relative_to(target, folder_abs) else None

    return header + "".join(out_chunks), rel_posix if rel_hash else None, rel_hash


def _file_outline_with_hash_sync(
    project_root: Path,
    folder_abs: Path,
    args: FileOutlineArgs,
) -> tuple[str, str | None, str | None]:
    target = resolve_scoped_path(
        project_root=project_root,
        allowed_root=project_root,
        raw_path=args.path,
        require_exists=True,
    )
    if not target.is_file():
        raise PathSecurityError(f"Path is not a file: {args.path!r}")
    file_size = target.stat().st_size
    if file_size > MAX_OUTLINE_FILE_BYTES:
        raise BlockReplaceError(
            f"File exceeds {MAX_OUTLINE_FILE_BYTES} bytes ({file_size} bytes): {args.path!r}; use view_file with line ranges."
        )

    raw_bytes = read_bytes_retry_sync(target)
    if len(raw_bytes) > MAX_OUTLINE_FILE_BYTES:
        raise BlockReplaceError(
            f"File exceeds {MAX_OUTLINE_FILE_BYTES} bytes ({len(raw_bytes)} bytes): {args.path!r}; use view_file with line ranges."
        )
    if b"\x00" in raw_bytes[:8192]:
        raise BlockReplaceError(f"Binary file cannot be outlined: {args.path!r}")
    try:
        source = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BlockReplaceError(f"Non-UTF-8 file cannot be outlined: {args.path!r}") from exc

    rel_posix = to_rel_posix(target, project_root)
    rel_hash: str | None = sha256_bytes(raw_bytes) if is_normcase_relative_to(target, folder_abs) else None
    return extract_file_outline(source, rel_posix), rel_posix if rel_hash else None, rel_hash


def _list_dir_sync(project_root: Path, args: ListDirArgs) -> str:
    target = resolve_scoped_path(
        project_root=project_root,
        allowed_root=project_root,
        raw_path=args.path,
        require_exists=True,
    )
    if not target.is_dir():
        raise PathSecurityError(f"Path is not a directory: {args.path!r}")

    resolved_project = project_root.resolve(strict=True)
    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.lower() in IGNORED_DIRS_LOWER:
            continue
        if is_reparse_point(child):
            try:
                resolved_child = child.resolve(strict=True)
                if not is_normcase_relative_to(resolved_child, resolved_project):
                    continue
                resolved_rel = to_rel_posix(resolved_child, resolved_project)
                if is_denied_rel_path(resolved_rel):
                    continue
            except OSError:
                continue
        rel_posix = to_rel_posix(child, project_root)
        if is_denied_rel_path(rel_posix):
            continue
        label = f"{child.name}/" if child.is_dir() else child.name
        entries.append(label)
        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append(f"... [capped at {MAX_LIST_ENTRIES} entries]")
            break
    return "\n".join(entries) if entries else "(empty directory)"


def _walk_project_files_sync(project_root: Path, base_dir: Path) -> Iterator[Path]:
    resolved_project = project_root.resolve(strict=True)
    try:
        resolved_base = base_dir.resolve(strict=True)
    except OSError:
        return
    if not is_normcase_relative_to(resolved_base, resolved_project):
        return
    for root, dirs, files in os.walk(base_dir, followlinks=False):
        root_path = Path(root)
        filtered_dirs: list[str] = []
        for d in sorted(dirs):
            if d.lower() in IGNORED_DIRS_LOWER:
                continue
            dpath = root_path / d
            if is_reparse_point(dpath):
                continue
            try:
                rel_dir = to_rel_posix(dpath, project_root)
            except ValueError:
                continue
            if is_denied_rel_path(rel_dir):
                continue
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs
        for fname in sorted(files):
            fpath = root_path / fname
            if is_reparse_point(fpath):
                continue
            try:
                rel = to_rel_posix(fpath, project_root)
            except ValueError:
                continue
            if is_denied_rel_path(rel):
                continue
            yield fpath


def _search_dir_sync(project_root: Path, args: SearchDirArgs) -> str:
    base_dir = resolve_scoped_path(
        project_root=project_root,
        allowed_root=project_root,
        raw_path=args.path,
        require_exists=True,
    )
    if not base_dir.is_dir():
        raise PathSecurityError(f"search_dir path is not a directory: {args.path!r}")

    case_insensitive = bool(getattr(args, "case_insensitive", False))
    file_glob = getattr(args, "file_glob", None)
    glob_lower = file_glob.lower() if file_glob else None
    needle = args.query.lower() if case_insensitive else args.query

    matches: list[str] = []
    for fpath in _walk_project_files_sync(project_root, base_dir):
        rel = to_rel_posix(fpath, project_root)
        if glob_lower and not (
            fnmatch.fnmatchcase(rel.lower(), glob_lower) or fnmatch.fnmatchcase(fpath.name.lower(), glob_lower)
        ):
            continue
        try:
            if fpath.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            raw = fpath.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            haystack = line.lower() if case_insensitive else line
            if needle in haystack:
                trimmed = line[:MAX_SEARCH_LINE_CHARS]
                matches.append(f"{rel}:{idx}: {trimmed}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    matches.append(f"... [capped at {MAX_SEARCH_MATCHES} matches]")
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches found."


def _find_file_sync(project_root: Path, args: FindFileArgs) -> str:
    base_dir = resolve_scoped_path(
        project_root=project_root,
        allowed_root=project_root,
        raw_path=args.path,
        require_exists=True,
    )
    if not base_dir.is_dir():
        raise PathSecurityError(f"find_file path is not a directory: {args.path!r}")

    pat_lower = args.pattern.lower()
    hits: list[str] = []
    for fpath in _walk_project_files_sync(project_root, base_dir):
        rel = to_rel_posix(fpath, project_root)
        if fnmatch.fnmatchcase(rel.lower(), pat_lower) or fnmatch.fnmatchcase(fpath.name.lower(), pat_lower):
            hits.append(rel)
            if len(hits) >= MAX_FIND_PATHS:
                hits.append(f"... [capped at {MAX_FIND_PATHS} files]")
                break
    return "\n".join(hits) if hits else "No matching files found."


def _is_commit_cancelled(ctx: WorkerToolContext) -> bool:
    if cancel_flag_path_for(ctx.run_dir).exists():
        if ctx.cancel_event is not None:
            ctx.cancel_event.set()
        if ctx.registry is not None:
            ctx.registry.mark_user_cancelled(ctx.project_root, ctx.manifest.run_id)
        return True
    return ctx.cancel_event is not None and ctx.cancel_event.is_set()


async def _run_tracked_commit(ctx: WorkerToolContext, coro: Any) -> Any:
    commit_task = asyncio.create_task(coro)
    if ctx.registry is not None:
        ctx.registry.track_commit(ctx.project_root, ctx.manifest.run_id, commit_task)
    return await asyncio.shield(commit_task)


async def _handle_replace_block(args: ReplaceBlockArgs, ctx: WorkerToolContext) -> str:
    target = resolve_scoped_path(
        project_root=ctx.project_root,
        allowed_root=ctx.folder_abs,
        raw_path=args.path,
        require_exists=True,
    )
    if not target.is_file():
        raise PathSecurityError(f"Target path is not a file: {args.path!r}")
    rel_posix = to_rel_posix(target, ctx.project_root)

    async def _commit() -> str:
        async with ctx.run_lock:
            if _is_commit_cancelled(ctx):
                raise BlockReplaceError("Run was cancelled before write committed.")
            cur_bytes = await asyncio.to_thread(read_bytes_retry_sync, target)
            cur_hash = sha256_bytes(cur_bytes)

            if rel_posix in ctx.worker.created_files:
                rec_created = ctx.worker.created_files[rel_posix]
                expected_hash = rec_created.worker_last_hash
                if cur_hash != expected_hash:
                    raise BlockReplaceError(
                        f"Optimistic hash mismatch on {rel_posix!r}; file changed externally."
                    )
                new_bytes, added, removed = await asyncio.to_thread(
                apply_replace_block, cur_bytes, args.target, args.replacement
            )
                new_hash = sha256_bytes(new_bytes)
                prev_hash, prev_lines = rec_created.worker_last_hash, rec_created.line_count
                rec_created.worker_last_hash = new_hash
                rec_created.line_count = len(new_bytes.splitlines())
                await write_manifest_locked(ctx.manifest_path, ctx.manifest)
                try:
                    await atomic_write_bytes(target, new_bytes)
                except BaseException:
                    rec_created.worker_last_hash = prev_hash
                    rec_created.line_count = prev_lines
                    await write_manifest_locked(ctx.manifest_path, ctx.manifest)
                    raise
                return f"Updated created file {rel_posix} (+{added}/-{removed} lines)."

            if rel_posix in ctx.worker.modified_files:
                expected_hash = ctx.worker.modified_files[rel_posix].worker_last_hash
            elif rel_posix in ctx.viewed_hashes:
                expected_hash = ctx.viewed_hashes[rel_posix]
            else:
                raise BlockReplaceError(
                    f"Call view_file on {rel_posix!r} before replace_block so initial hash can be verified."
                )

            if cur_hash != expected_hash:
                raise BlockReplaceError(
                    f"Optimistic hash mismatch on {rel_posix!r}; file changed externally since last read/edit."
                )

            new_bytes, added, removed = await asyncio.to_thread(
                apply_replace_block, cur_bytes, args.target, args.replacement
            )
            backup_path = await asyncio.to_thread(record_backup_once_sync, ctx.backups_dir, rel_posix, cur_bytes)
            new_hash = sha256_bytes(new_bytes)

            prev_rec = (
                ctx.worker.modified_files[rel_posix].model_copy(deep=True)
                if rel_posix in ctx.worker.modified_files
                else None
            )
            if rel_posix in ctx.worker.modified_files:
                rec = ctx.worker.modified_files[rel_posix]
                rec.worker_last_hash = new_hash
                rec.added_lines += added
                rec.removed_lines += removed
            else:
                ctx.worker.modified_files[rel_posix] = FileEditRecord(
                    backup_path=str(backup_path),
                    initial_hash=cur_hash,
                    worker_last_hash=new_hash,
                    added_lines=added,
                    removed_lines=removed,
                )
            await write_manifest_locked(ctx.manifest_path, ctx.manifest)
            try:
                await atomic_write_bytes(target, new_bytes)
            except BaseException:
                if prev_rec is None:
                    ctx.worker.modified_files.pop(rel_posix, None)
                else:
                    ctx.worker.modified_files[rel_posix] = prev_rec
                await write_manifest_locked(ctx.manifest_path, ctx.manifest)
                raise
            return f"Replaced block in {rel_posix} (+{added}/-{removed} lines)."

    return await _run_tracked_commit(ctx, _commit())


def _pending_folder_targets(
    project_root: Path,
    folder_abs: Path,
    target_files: list[str],
    created_files: dict[str, Any] | set[str] | None = None,
    modified_files: dict[str, Any] | set[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Return (has_folder_targets, missing_new_targets, unedited_existing_targets) scoped inside folder_abs."""
    created = created_files or ()
    modified = modified_files or ()
    has_folder_targets = False
    missing_new: list[str] = []
    unedited_existing: list[str] = []
    seen: set[str] = set()
    for raw_target in target_files:
        try:
            resolved = resolve_scoped_path(
                project_root=project_root,
                allowed_root=project_root,
                raw_path=raw_target,
                require_exists=False,
                allow_new_parents=True,
            )
        except PathSecurityError:
            continue
        if not is_normcase_relative_to(resolved, folder_abs):
            continue
        rel = to_rel_posix(resolved, project_root)
        norm_key = os.path.normcase(rel)
        if norm_key in seen:
            continue
        seen.add(norm_key)
        has_folder_targets = True
        if not resolved.exists():
            missing_new.append(rel)
        elif rel not in modified and rel not in created:
            unedited_existing.append(rel)
    return has_folder_targets, missing_new, unedited_existing


def _format_validated_readonly_sig(tool_name: str, validated: Any) -> str:
    dumped = validated.model_dump()
    if "path" in dumped and isinstance(dumped["path"], str):
        dumped["path"] = _normalize_tool_path(dumped["path"])
    if (
        tool_name == "view_file"
        and isinstance(dumped.get("start_line"), int)
        and isinstance(dumped.get("end_line"), int)
    ):
        dumped["end_line"] = min(dumped["end_line"], dumped["start_line"] + MAX_VIEW_LINES - 1)
    return f"{tool_name}:{json.dumps(dumped, sort_keys=True, separators=(',', ':'))}"


def _canonical_tool_sig(tool_name: str, raw_args: Any) -> str:
    """Return a stable string key for deduplicating equivalent read-only tool invocations."""
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return f"{tool_name}:{raw_args.strip()}"
    elif isinstance(raw_args, dict):
        parsed = dict(raw_args)
    else:
        parsed = {}
    if not isinstance(parsed, dict):
        return f"{tool_name}:{json.dumps(parsed, sort_keys=True, separators=(',', ':'))}"

    model_cls = _READONLY_ARG_MODELS.get(tool_name)
    if model_cls is not None:
        try:
            return _format_validated_readonly_sig(tool_name, model_cls.model_validate(parsed))
        except ValidationError:
            if "path" in parsed and isinstance(parsed["path"], str):
                parsed["path"] = _normalize_tool_path(parsed["path"])
    return f"{tool_name}:{json.dumps(parsed, sort_keys=True, separators=(',', ':'))}"


def _is_unmodified_created_file_read(ctx: WorkerToolContext) -> bool:
    """Return True if the last read tool inspected an unmodified newly-created file (exempting failed static-check repairs)."""
    if ctx.worker.static_check.status == "failed" or ctx.last_read_file is None:
        return False
    rel_posix, cur_hash = ctx.last_read_file
    rec = ctx.worker.created_files.get(rel_posix)
    return rec is not None and rec.worker_last_hash == cur_hash


def _format_pending_target_parts(
    missing_new: list[str],
    unedited_existing: list[str],
    *,
    in_folder_label: bool = False,
) -> list[str]:
    parts: list[str] = []
    if missing_new:
        label = "Remaining new target files in your folder" if in_folder_label else "Remaining new target files"
        parts.append(f"{label}: {', '.join(missing_new)}.")
    if unedited_existing:
        parts.append(f"Existing folder target(s) not yet edited: {', '.join(unedited_existing)}.")
    return parts


def _step_budget_notice(steps_used: int, max_steps: int) -> str | None:
    """Return a step-budget warning when 2 or 1 steps remain before max_steps_per_worker."""
    if max_steps >= 4 and steps_used == max_steps - 2:
        return (
            "Step budget notice: 2 steps remain before max_steps_per_worker. "
            "Finish any pending edit and call finish(summary, notes) now."
        )
    if max_steps >= 3 and steps_used == max_steps - 1:
        return (
            "FINAL STEP: 1 step remains before max_steps_per_worker. "
            "Call finish(summary, notes) on this turn now."
        )
    return None


async def _handle_write_new_file(args: WriteNewFileArgs, ctx: WorkerToolContext) -> str:
    target = resolve_scoped_path(
        project_root=ctx.project_root,
        allowed_root=ctx.folder_abs,
        raw_path=args.path,
        require_exists=False,
        allow_new_parents=True,
    )
    rel_posix = to_rel_posix(target, ctx.project_root)

    async def _commit() -> str:
        async with ctx.run_lock:
            if _is_commit_cancelled(ctx):
                raise BlockReplaceError("Run was cancelled before write committed.")
            if (
                target.exists()
                or rel_posix in ctx.worker.created_files
                or rel_posix in ctx.worker.modified_files
            ):
                raise PathSecurityError(
                    f"write_new_file refuses to overwrite existing path: {rel_posix!r}"
                )
            prev_created_dirs = list(ctx.worker.created_dirs) if ctx.worker.created_dirs is not None else None
            if ctx.worker.created_dirs is None:
                ctx.worker.created_dirs = []
            resolved_project = ctx.project_root.resolve()
            resolved_folder = ctx.folder_abs.resolve()
            cur_dir = target.parent
            new_dirs: list[str] = []
            while cur_dir != resolved_folder and is_normcase_relative_to(cur_dir, resolved_folder):
                if not cur_dir.exists():
                    new_dirs.append(to_rel_posix(cur_dir, resolved_project))
                cur_dir = cur_dir.parent
            for rel_dir in reversed(new_dirs):
                if rel_dir not in ctx.worker.created_dirs:
                    ctx.worker.created_dirs.append(rel_dir)

            raw = args.content.encode("utf-8")
            ctx.worker.created_files[rel_posix] = CreatedFileRecord(
                worker_last_hash=sha256_bytes(raw),
                line_count=len(args.content.splitlines()),
            )
            await write_manifest_locked(ctx.manifest_path, ctx.manifest)
            try:
                await atomic_write_bytes(target, raw)
            except BaseException:
                ctx.worker.created_files.pop(rel_posix, None)
                ctx.worker.created_dirs = prev_created_dirs

                def _cleanup_orphaned_new_dirs() -> None:
                    for rel_dir in new_dirs:
                        dir_path = resolved_project / rel_dir
                        if dir_path.is_dir() and is_normcase_relative_to(dir_path, resolved_folder):
                            try:
                                dir_path.rmdir()
                            except OSError:
                                pass

                await asyncio.to_thread(_cleanup_orphaned_new_dirs)
                await write_manifest_locked(ctx.manifest_path, ctx.manifest)
                raise
            msg = f"Created {rel_posix} ({len(args.content.splitlines())} lines)."
            has_targets, missing_new, unedited_existing = _pending_folder_targets(
                ctx.project_root,
                ctx.folder_abs,
                ctx.worker.target_files,
                created_files=ctx.worker.created_files,
                modified_files=ctx.worker.modified_files,
            )
            if has_targets:
                if missing_new:
                    msg += " " + " ".join(
                        _format_pending_target_parts(missing_new, unedited_existing, in_folder_label=True)
                    )
                elif unedited_existing:
                    msg += f" All new target files created; existing folder target(s) not yet edited: {', '.join(unedited_existing)}."
                else:
                    msg += (
                        " All target files in your folder are now written; "
                        "call finish(summary, notes) now unless a specific fix is needed."
                    )
            return msg

    return await _run_tracked_commit(ctx, _commit())


async def _handle_finish(args: FinishArgs, ctx: WorkerToolContext) -> tuple[str, bool]:
    touched = list(ctx.worker.created_files) + list(ctx.worker.modified_files)
    precheck = await asyncio.to_thread(run_static_precheck, ctx.project_root, touched)

    async def _commit() -> tuple[str, bool]:
        async with ctx.run_lock:
            if _is_commit_cancelled(ctx):
                raise BlockReplaceError("Run was cancelled before finish() committed.")
            ctx.worker.static_check = precheck
            if precheck.status == "failed":
                await write_manifest_locked(ctx.manifest_path, ctx.manifest)
                return "Static pre-check failed:\n" + "\n".join(precheck.errors), False

            ctx.worker.summary = redact_secrets(args.summary.strip())
            ctx.worker.notes = [redact_secrets(n.strip()) for n in args.notes if n.strip()]
            ctx.worker.status = "completed"
            ctx.finished_cleanly = True
            await write_manifest_locked(ctx.manifest_path, ctx.manifest)
            return "Static pre-check passed; task marked completed.", True

    return await _run_tracked_commit(ctx, _commit())


async def execute_worker_tool(tool_name: str, raw_args: Any, ctx: WorkerToolContext) -> tuple[str, bool]:
    """Validate arguments and dispatch one worker tool call; returns (output_text, is_error)."""
    ctx.last_read_file = None
    ctx.last_readonly_sig = None
    if ctx.finished_cleanly:
        return "Tool rejected: worker already completed via finish().", True
    try:
        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except (TypeError, json.JSONDecodeError) as exc:
        return redact_secrets(f"Tool argument JSON error for {tool_name}: {exc}"), True

    try:
        if tool_name == "view_file":
            vargs = ViewFileArgs.model_validate(args_dict)
            text, rel_key, rel_hash = await asyncio.to_thread(_view_file_sync, ctx.project_root, ctx.folder_abs, vargs)
            if rel_key and rel_hash:
                ctx.last_read_file = (rel_key, rel_hash)
                if rel_key not in ctx.worker.modified_files and rel_key not in ctx.worker.created_files:
                    ctx.viewed_hashes[rel_key] = rel_hash
            ctx.last_readonly_sig = _format_validated_readonly_sig(tool_name, vargs)
            return text, False
        if tool_name == "file_outline":
            oargs = FileOutlineArgs.model_validate(args_dict)
            text, rel_key, rel_hash = await asyncio.to_thread(
                _file_outline_with_hash_sync, ctx.project_root, ctx.folder_abs, oargs
            )
            if rel_key and rel_hash:
                ctx.last_read_file = (rel_key, rel_hash)
            ctx.last_readonly_sig = _format_validated_readonly_sig(tool_name, oargs)
            return text, False
        if tool_name == "list_dir":
            largs = ListDirArgs.model_validate(args_dict)
            text = await asyncio.to_thread(_list_dir_sync, ctx.project_root, largs)
            ctx.last_readonly_sig = _format_validated_readonly_sig(tool_name, largs)
            return text, False
        if tool_name == "search_dir":
            sargs = SearchDirArgs.model_validate(args_dict)
            text = await asyncio.to_thread(_search_dir_sync, ctx.project_root, sargs)
            ctx.last_readonly_sig = _format_validated_readonly_sig(tool_name, sargs)
            return text, False
        if tool_name == "find_file":
            fargs = FindFileArgs.model_validate(args_dict)
            text = await asyncio.to_thread(_find_file_sync, ctx.project_root, fargs)
            ctx.last_readonly_sig = _format_validated_readonly_sig(tool_name, fargs)
            return text, False
        if tool_name == "replace_block":
            rargs = ReplaceBlockArgs.model_validate(args_dict)
            return await _handle_replace_block(rargs, ctx), False
        if tool_name == "write_new_file":
            wargs = WriteNewFileArgs.model_validate(args_dict)
            return await _handle_write_new_file(wargs, ctx), False
        if tool_name == "finish":
            fin_args = FinishArgs.model_validate(args_dict)
            msg, ok = await _handle_finish(fin_args, ctx)
            return msg, not ok
        return f"Unknown tool: {tool_name!r}", True
    except (ValidationError, PathSecurityError, BlockReplaceError, OSError) as exc:
        return redact_secrets(f"Tool error ({tool_name}): {exc}"), True


def _seed_target_metadata(project_root: Path, target_files: list[str]) -> list[str]:
    lines: list[str] = []
    for raw_target in target_files:
        resolved = resolve_scoped_path(
            project_root=project_root,
            allowed_root=project_root,
            raw_path=raw_target,
            require_exists=False,
            allow_new_parents=True,
        )
        rel = to_rel_posix(resolved, project_root)
        if resolved.is_file():
            try:
                if resolved.stat().st_size > MAX_VIEW_FILE_BYTES:
                    lines.append(f"- {rel} (large file > 10 MiB; use file_outline or view_file slices)")
                else:
                    with resolved.open("rb") as f:
                        count = sum(1 for _ in f)
                    lines.append(f"- {rel} ({count} lines)")
            except OSError:
                lines.append(f"- {rel} (unreadable)")
        else:
            lines.append(f"- {rel} (new file - does not exist yet)")
    return lines


def _load_repo_instructions_sync(project_root: Path) -> str:
    """Load optional repo-local .overseer/instructions.md (capped at 10 KB)."""
    repo_instr = project_root / ".overseer" / "instructions.md"
    if not repo_instr.is_file() or is_reparse_point(repo_instr):
        return ""
    try:
        resolved_project = project_root.resolve(strict=True)
        resolved_instr = repo_instr.resolve(strict=True)
        if not is_normcase_relative_to(resolved_instr, resolved_project):
            return ""
        with resolved_instr.open("rb") as f:
            raw = f.read(10_240)
        return raw.decode("utf-8-sig", errors="replace").strip()
    except OSError:
        return ""


def _build_worker_system_prompt(
    project_root: Path,
    task: WorkerTaskInput,
    shared_contract: str | None,
    max_steps: int = 25,
) -> str:
    parts = [
        "You are an Overseer worker executing a scoped file task.",
        (
            f"Read-only project root: {project_root.name or 'workspace'}\n"
            f"Writable folder jail: {task.folder} (only files inside this folder may be edited or created)\n"
            f"Step budget: {max_steps} turns (3 consecutive tool errors or 2 turns without a tool call halt execution)."
        ),
        (
            "Workflow:\n"
            "1. Read Target files or relevant files first with view_file (skip blind directory scans when paths are given). "
            "Use file_outline before paginating files > 300 lines. "
            f"Batch independent read-only calls together; older tool outputs are pruned after {KEEP_RECENT_TOOLS} tool calls "
            "(exploratory search/list/outline and superseded views first; active file views and failed finish diagnostics last).\n"
            "2. Plan the smallest complete change that satisfies the goal and shared contract. Reuse existing helpers and match local style.\n"
            "3. Edit existing files with replace_block (call view_file first; include only changing lines + 2-3 unique surrounding lines, matching exact whitespace/indentation). "
            "Create missing files with write_new_file. Write complete implementations—never leave TODO stubs.\n"
            "4. On a BlockReplaceError, use the printed 'Closest match' snippet or re-read those lines with view_file before retrying.\n"
            "5. If an edit is needed outside your writable folder, pass plain-text suggestions in finish(summary, notes=[...]). Work independently from Shared contract—never poll for sibling workers' files.\n"
            "6. Call finish(summary, notes) IMMEDIATELY after writing/editing your deliverables—do NOT re-read files you just wrote to double-check them. finish() checks .py/.js/.jsx/.ts/.tsx/.json/.toml/.yaml/.yml/.luau/.ps1/.cpp syntax in memory and returns any errors for you to fix. Always emit a tool call or finish() on every turn."
        ),
        "All tool path arguments must be relative to project_root.",
    ]
    custom_instructions = load_custom_instructions_sync()
    repo_instructions = _load_repo_instructions_sync(project_root)
    merged_instructions = "\n\n".join(part for part in (custom_instructions, repo_instructions) if part)
    if merged_instructions:
        parts.append(f"Custom instructions:\n{merged_instructions}")
    if shared_contract:
        parts.append(f"Shared contract:\n{shared_contract}")
    if task.target_files:
        seeded = "\n".join(_seed_target_metadata(project_root, task.target_files))
        parts.append(f"Target files:\n{seeded}")
    return "\n\n".join(parts)


_T = TypeVar("_T")


async def _commit_manifest_ctx(ctx: WorkerToolContext, mutate_fn: Callable[[], _T]) -> _T:
    async def _commit() -> _T:
        async with ctx.run_lock:
            result = mutate_fn()
            await atomic_write_bytes(
                ctx.manifest_path,
                ctx.manifest.model_dump_json(indent=2).encode("utf-8"),
            )
            return result

    return await _run_tracked_commit(ctx, _commit())


async def run_worker_loop(
    task: WorkerTaskInput,
    folder_abs: Path,
    manifest: RunManifest,
    manifest_path: Path,
    backups_dir: Path,
    run_dir: Path,
    registry: RunRegistry,
    client: httpx.AsyncClient,
    catalog: ModelCatalog,
    api_key: str,
    semaphore: asyncio.Semaphore,
    cancel_event: asyncio.Event,
    model_chain: list[str] | None = None,
    tier: TierName | None = None,
) -> None:
    """Execute one bounded worker loop with retries, failover, breakers, and budget enforcement."""
    project_root = Path(manifest.project_root)
    lock = registry.get_lock(project_root, manifest.run_id)
    worker_state = manifest.workers[task.worker_id]
    effective_tier = tier or infer_tier_for_request(task.model)
    model_chain = list(model_chain) if model_chain else build_model_chain(task.model)
    model_idx = 0
    base_url = get_openrouter_base_url()

    ctx = WorkerToolContext(
        project_root=project_root,
        folder_abs=folder_abs,
        backups_dir=backups_dir,
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        worker=worker_state,
        run_lock=lock,
        registry=registry,
        cancel_event=cancel_event,
    )

    sys_prompt = await asyncio.to_thread(
        _build_worker_system_prompt,
        project_root,
        task,
        manifest.shared_contract,
        manifest.limits.max_steps_per_worker,
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {"role": "user", "content": f"Goal: {task.goal}"},
    ]

    consecutive_errors = 0
    empty_turns = 0
    seen_readonly_calls: dict[str, int] = {}
    readonly_turns_since_write = 0
    post_write_stall_turns = 0
    consecutive_redundant_read_turns = 0
    cancel_flag_path = cancel_flag_path_for(run_dir)

    def _check_cancelled() -> bool:
        if cancel_flag_path.exists():
            registry.mark_user_cancelled(project_root, manifest.run_id)
            cancel_event.set()
        return cancel_event.is_set()

    def _apply_stopped_state() -> None:
        is_user = registry.is_user_cancelled(project_root, manifest.run_id)
        is_interrupted = registry.is_interrupted(project_root, manifest.run_id)
        if is_user:
            worker_state.status = "cancelled"
            worker_state.stop_reason = "cancelled_by_user"
        elif is_interrupted:
            worker_state.status = "interrupted"
            worker_state.stop_reason = worker_state.stop_reason or "server_shutdown"
        else:
            worker_state.status = "halted_budget"
            worker_state.stop_reason = "budget_halted"

    async def _mark_stopped() -> None:
        await _commit_manifest_ctx(ctx, _apply_stopped_state)

    async def _escalate_worker(
        stop_reason: str,
        escalation_reason: str,
        run_precheck: bool = False,
    ) -> None:
        if _check_cancelled():
            await _mark_stopped()
            return
        precheck = None
        if run_precheck:
            touched = list(ctx.worker.created_files) + list(ctx.worker.modified_files)
            if touched:
                precheck = await asyncio.to_thread(run_static_precheck, project_root, touched)

        def _commit_esc() -> None:
            if _check_cancelled():
                _apply_stopped_state()
                return
            if precheck is not None:
                worker_state.static_check = precheck
            worker_state.status = "escalated"
            worker_state.stop_reason = stop_reason
            worker_state.escalation_reason = escalation_reason

        await _commit_manifest_ctx(ctx, _commit_esc)

    async with semaphore:
        if _check_cancelled():
            await _mark_stopped()
            return

        def _commit_running() -> bool:
            if _check_cancelled():
                _apply_stopped_state()
                return False
            worker_state.status = "running"
            return True

        if not await _commit_manifest_ctx(ctx, _commit_running):
            return

        try:
            while not ctx.finished_cleanly:
                if _check_cancelled() or manifest.spend_usd >= manifest.limits.max_run_usd:
                    if manifest.spend_usd >= manifest.limits.max_run_usd:
                        cancel_event.set()
                    await _mark_stopped()
                    return

                if worker_state.steps_used >= manifest.limits.max_steps_per_worker:
                    await _escalate_worker(
                        stop_reason="max_steps_reached",
                        escalation_reason=(
                            f"Reached max_steps_per_worker ({manifest.limits.max_steps_per_worker}) without finishing."
                        ),
                        run_precheck=True,
                    )
                    return

                pruned = prune_message_history(messages, keep_recent_tools=KEEP_RECENT_TOOLS)
                try:
                    payload, used_model, model_idx = await _call_openrouter_with_failover(
                        client=client,
                        api_key=api_key,
                        model_chain=model_chain,
                        start_model_idx=model_idx,
                        messages=pruned,
                        limits=manifest.limits,
                        cancel_event=cancel_event,
                        cancel_flag_path=cancel_flag_path,
                        tier=effective_tier,
                    )
                except _WorkerCancelledError:
                    _check_cancelled()
                    await _mark_stopped()
                    return
                except RuntimeError as exc:
                    err_msg = redact_secrets(str(exc))

                    def _commit_runtime_err() -> None:
                        worker_state.status = "failed"
                        worker_state.stop_reason = err_msg.split(":", 1)[0]
                        worker_state.escalation_reason = err_msg

                    await _commit_manifest_ctx(ctx, _commit_runtime_err)
                    return

                raw_usage = payload.get("usage")
                usage = raw_usage if isinstance(raw_usage, dict) else {}
                p_tok = max(0, _safe_int(usage.get("prompt_tokens"), 0))
                c_tok = max(0, _safe_int(usage.get("completion_tokens"), 0))
                prompt_details = usage.get("prompt_tokens_details")
                raw_cached = (
                    prompt_details.get("cached_tokens")
                    if isinstance(prompt_details, dict)
                    else None
                )
                if raw_cached is None:
                    raw_cached = usage.get("cached_tokens")
                cached_tok = max(0, _safe_int(raw_cached, 0))
                raw_cost = usage.get("cost")
                parsed_cost: float | None = None
                if raw_cost is not None:
                    try:
                        val = float(raw_cost)
                        if math.isfinite(val):
                            parsed_cost = max(0.0, val)
                    except (TypeError, ValueError):
                        parsed_cost = None
                if parsed_cost is not None:
                    call_cost = parsed_cost
                else:
                    call_cost = catalog.estimate_cost(used_model, p_tok, c_tok, base_url=base_url)
                    if not used_model.endswith(":free") and call_cost == 0.0 and (p_tok + c_tok) > 0:
                        _logger.warning(
                            "unpriced_paid_model run=%s worker=%s model=%s prompt_tokens=%d completion_tokens=%d cost=0.0",
                            manifest.run_id,
                            task.worker_id,
                            used_model,
                            p_tok,
                            c_tok,
                        )

                def _commit_usage() -> bool:
                    worker_state.steps_used += 1
                    if used_model not in worker_state.models_used:
                        worker_state.models_used.append(used_model)
                    worker_state.prompt_tokens += p_tok
                    worker_state.completion_tokens += c_tok
                    worker_state.cached_tokens += cached_tok
                    worker_state.spend_usd += call_cost
                    manifest.spend_usd += call_cost
                    manifest.cached_tokens += cached_tok
                    if manifest.spend_usd >= manifest.limits.max_run_usd:
                        cancel_event.set()
                        return True
                    return False

                budget_exceeded = await _commit_manifest_ctx(ctx, _commit_usage)

                if _check_cancelled() or budget_exceeded:
                    await _mark_stopped()
                    return

                choice = (payload.get("choices") or [{}])[0]
                choice_msg = choice.get("message") or {}
                finish_reason = choice.get("finish_reason")
                tool_calls = choice_msg.get("tool_calls") or []
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice_msg.get("content") or "",
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    }
                )

                if not tool_calls:
                    empty_turns += 1
                    if empty_turns >= MAX_EMPTY_TURNS:
                        await _escalate_worker(
                            stop_reason="no_tool_calls",
                            escalation_reason="Worker emitted repeated responses without tool calls.",
                            run_precheck=True,
                        )
                        return
                    if finish_reason == "length":
                        nudge = (
                            "Your previous response hit the output token limit before emitting a tool call. "
                            "Be concise and invoke a file tool or finish(summary, notes) now."
                        )
                    else:
                        nudge = "Invoke one of your file tools or call finish(summary, notes) now."
                    budget_notice = _step_budget_notice(
                        worker_state.steps_used, manifest.limits.max_steps_per_worker
                    )
                    if budget_notice:
                        nudge = f"{nudge}\n{budget_notice}"
                    messages.append({"role": "user", "content": nudge})
                    continue

                empty_turns = 0
                _logger.info(
                    "worker_step run=%s worker=%s step=%d/%d model=%s tools=%s",
                    manifest.run_id,
                    task.worker_id,
                    worker_state.steps_used,
                    manifest.limits.max_steps_per_worker,
                    used_model,
                    [str((tc.get("function") or {}).get("name", "")) for tc in tool_calls],
                )
                turn_had_write = False
                turn_all_readonly = True
                turn_all_readonly_redundant = True
                turn_had_tool_error = False
                repeated_readonly_tool: str | None = None
                # Always emit matching tool messages for every tool_call_id, but skip execution once finish() completes
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    fn = tc.get("function") or {}
                    tname = str(fn.get("name", ""))
                    raw_targs = fn.get("arguments", "{}")
                    if ctx.finished_cleanly:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "Tool skipped: task already completed by finish().",
                            }
                        )
                        continue
                    output_text, is_error = await execute_worker_tool(tname, raw_targs, ctx)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": output_text})

                    if tname not in READONLY_TOOLS:
                        turn_all_readonly = False
                        turn_all_readonly_redundant = False
                    if is_error:
                        consecutive_errors += 1
                        turn_had_tool_error = True
                        turn_all_readonly_redundant = False
                    else:
                        consecutive_errors = 0
                        if tname in {"write_new_file", "replace_block"}:
                            turn_had_write = True
                        elif tname in READONLY_TOOLS:
                            sig = ctx.last_readonly_sig or _canonical_tool_sig(tname, raw_targs)
                            prev_count = seen_readonly_calls.get(sig, 0)
                            count = prev_count + 1
                            seen_readonly_calls[sig] = count
                            is_created_file_read = _is_unmodified_created_file_read(ctx)
                            if prev_count < 1 and not is_created_file_read:
                                turn_all_readonly_redundant = False
                            has_writes = bool(ctx.worker.created_files or ctx.worker.modified_files)
                            if (has_writes and count >= 2) or count >= 3:
                                repeated_readonly_tool = tname

                if ctx.finished_cleanly:
                    return

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    await _escalate_worker(
                        stop_reason="consecutive_tool_errors",
                        escalation_reason=(
                            f"Escalated after {MAX_CONSECUTIVE_ERRORS} consecutive tool errors."
                        ),
                        run_precheck=True,
                    )
                    return

                if turn_had_write or not turn_all_readonly:
                    seen_readonly_calls.clear()
                    readonly_turns_since_write = 0
                    post_write_stall_turns = 0
                    consecutive_redundant_read_turns = 0
                    repeated_readonly_tool = None
                else:
                    if turn_all_readonly_redundant and not turn_had_tool_error:
                        consecutive_redundant_read_turns += 1
                    else:
                        consecutive_redundant_read_turns = 0
                    if ctx.worker.created_files or ctx.worker.modified_files:
                        readonly_turns_since_write += 1
                        if turn_all_readonly_redundant or turn_had_tool_error:
                            post_write_stall_turns += 1

                has_writes = bool(ctx.worker.created_files or ctx.worker.modified_files)
                if consecutive_redundant_read_turns >= MAX_REDUNDANT_READ_TURNS:
                    if has_writes:
                        await _escalate_worker(
                            stop_reason="stalled_after_writes",
                            escalation_reason=(
                                f"Worker made {consecutive_redundant_read_turns} consecutive redundant read-only turns "
                                "after writing files without editing or calling finish()."
                            ),
                            run_precheck=True,
                        )
                    else:
                        await _escalate_worker(
                            stop_reason="read_loop_detected",
                            escalation_reason=(
                                f"Worker made {consecutive_redundant_read_turns} consecutive redundant read-only turns "
                                "without writing files or calling finish()."
                            ),
                            run_precheck=False,
                        )
                    return

                if post_write_stall_turns >= MAX_POST_WRITE_READONLY_TURNS:
                    await _escalate_worker(
                        stop_reason="stalled_after_writes",
                        escalation_reason=(
                            f"Worker made {post_write_stall_turns} consecutive non-progressing read-only turns "
                            "after writing files without editing or calling finish()."
                        ),
                        run_precheck=True,
                    )
                    return

                stagnation_nudge: str | None = None
                should_nudge_stagnation = turn_all_readonly and (
                    repeated_readonly_tool is not None
                    or consecutive_redundant_read_turns == MAX_REDUNDANT_READ_TURNS - 1
                    or readonly_turns_since_write == POST_WRITE_NUDGE_TURN
                    or post_write_stall_turns == MAX_POST_WRITE_READONLY_TURNS - 1
                )
                if should_nudge_stagnation:
                    written = sorted(set(ctx.worker.created_files) | set(ctx.worker.modified_files))
                    is_final_readonly_warning = (
                        post_write_stall_turns == MAX_POST_WRITE_READONLY_TURNS - 1
                        or consecutive_redundant_read_turns == MAX_REDUNDANT_READ_TURNS - 1
                    )
                    prefix = "FINAL READ-ONLY WARNING: " if is_final_readonly_warning else "Progress reminder: "
                    final_suffix = (
                        " One more read-only turn without an edit or finish() will halt execution."
                        if is_final_readonly_warning
                        else ""
                    )
                    repeat_clause = (
                        "stop repeating read-only calls. " if repeated_readonly_tool is not None else ""
                    )
                    if written:
                        has_folder_targets, missing_new, unedited_existing = _pending_folder_targets(
                            project_root,
                            folder_abs,
                            task.target_files,
                            created_files=ctx.worker.created_files,
                            modified_files=ctx.worker.modified_files,
                        )
                        if ctx.worker.static_check.status == "failed":
                            stagnation_nudge = (
                                f"{prefix}{repeat_clause}Already written: {', '.join(written)}. "
                                "Fix the reported static pre-check errors before calling finish(summary, notes)."
                                f"{final_suffix}"
                            )
                        elif has_folder_targets and (missing_new or unedited_existing):
                            pending_parts = _format_pending_target_parts(missing_new, unedited_existing)
                            stagnation_nudge = (
                                f"{prefix}{repeat_clause}Already written: {', '.join(written)}. "
                                f"{' '.join(pending_parts)} "
                                "Write/edit the next target file or call finish(summary, notes) if no more changes are needed."
                                f"{final_suffix}"
                            )
                        elif has_folder_targets:
                            stagnation_nudge = (
                                f"{prefix}{repeat_clause}You already wrote {', '.join(written)} and spent recent turns "
                                "only reading. Stop re-reading and call finish(summary, notes) now "
                                "(finish() runs static syntax checks automatically)."
                                f"{final_suffix}"
                            )
                        else:
                            stagnation_nudge = (
                                f"{prefix}{repeat_clause}You already wrote {', '.join(written)} and spent recent turns "
                                "only reading. Make your next edit or call finish(summary, notes) now "
                                "(finish() runs static syntax checks automatically)."
                                f"{final_suffix}"
                            )
                    elif repeated_readonly_tool is not None:
                        stagnation_nudge = (
                            f"{prefix}you repeated the exact same {repeated_readonly_tool} call; "
                            "stop repeating read-only calls. "
                            "Write/edit your target files or call finish(summary, notes) now."
                            f"{final_suffix}"
                        )
                    elif consecutive_redundant_read_turns == MAX_REDUNDANT_READ_TURNS - 1:
                        stagnation_nudge = (
                            f"{prefix}recent turns only repeated previous read-only calls; "
                            "stop repeating read-only calls. "
                            "Write/edit your target files or call finish(summary, notes) now."
                            f"{final_suffix}"
                        )

                step_budget_notice = _step_budget_notice(
                    worker_state.steps_used, manifest.limits.max_steps_per_worker
                )

                nudges = [n for n in (stagnation_nudge, step_budget_notice) if n]
                if nudges:
                    messages[-1]["content"] = f"{messages[-1]['content']}\n\n" + "\n".join(nudges)
        except Exception as exc:
            err_msg = redact_secrets(f"{type(exc).__name__}: {exc}")

            def _commit_unexpected_err() -> None:
                worker_state.status = "failed"
                worker_state.stop_reason = "unexpected_worker_error"
                worker_state.escalation_reason = err_msg

            await _commit_manifest_ctx(ctx, _commit_unexpected_err)
            return


def determine_final_run_status(
    manifest: RunManifest,
    budget_halted: bool,
    user_cancelled: bool = False,
    interrupted: bool = False,
) -> RunStatus:
    if user_cancelled:
        return "cancelled"
    if interrupted:
        return "interrupted"
    if budget_halted:
        return "halted_budget"
    statuses = {w.status for w in manifest.workers.values()}
    if statuses == {"completed"}:
        return "completed"
    if statuses <= {"failed", "interrupted"}:
        return "failed"
    if "completed" in statuses or "escalated" in statuses:
        return "completed_with_escalations"
    return "failed"


async def prepare_and_launch_run(
    request: StartRunRequest,
    registry: RunRegistry,
    client: httpx.AsyncClient,
    catalog: ModelCatalog,
    project_root: Path,
) -> StartRunResponse:
    """Validate pre-flight constraints, persist initial manifest externally, and launch background tasks."""
    folder_map = validate_non_overlapping_folders(project_root, request.tasks)
    for task in request.tasks:
        worker_folder = folder_map[task.worker_id]
        for raw_target in task.target_files:
            resolved_target = resolve_scoped_path(
                project_root=project_root,
                allowed_root=project_root,
                raw_path=raw_target,
                require_exists=False,
                allow_new_parents=True,
            )
            if resolved_target.is_dir():
                raise PathSecurityError(
                    f"target_files entry {raw_target!r} for worker {task.worker_id!r} must be a file path, not a directory."
                )
            if not resolved_target.exists() and not is_normcase_relative_to(resolved_target, worker_folder):
                raise PathSecurityError(
                    f"Non-existent target_file {raw_target!r} for worker {task.worker_id!r} must reside inside writable folder {task.folder!r}."
                )

    task_chains: dict[str, list[str]] = {}
    task_tiers: dict[str, TierName] = {}
    task_allows_fallback: dict[str, bool] = {}
    models_with_tier: list[tuple[TierName, str]] = []
    for task in request.tasks:
        tier = infer_tier_for_request(task.model)
        task_tiers[task.worker_id] = tier
        chain = build_model_chain(task.model)
        task_chains[task.worker_id] = chain
        task_allows_fallback[task.worker_id] = (
            task.model is None or task.model.strip().lower() in {"", "free", "cheap", "heavy"}
        )
        for m in chain:
            if (tier, m) not in models_with_tier:
                models_with_tier.append((tier, m))

    tier_model_keys: dict[tuple[TierName, str], str] = {}
    for tier, m in models_with_tier:
        tier_model_keys[(tier, m)] = select_api_key_for_model(m, tier=tier)

    existing_active = registry.has_active_run(project_root)
    if existing_active is not None:
        raise RunStateError(
            f"Active run {existing_active!r} is already running for project_root {project_root}"
        )

    base_url = get_openrouter_base_url()
    for unique_key in dict.fromkeys(tier_model_keys.values()):
        await catalog.ensure_loaded(client, unique_key, timeout=request.request_timeout_sec)
    for wid, chain in list(task_chains.items()):
        task_chains[wid] = catalog.filter_supported_chain(
            chain,
            base_url,
            allow_primary_fallback=task_allows_fallback[wid],
        )

    run_id = generate_run_id()
    project_data_dir = get_project_data_dir(project_root)
    semaphore = registry.get_worker_semaphore()
    cancel_event = registry.register_run(project_root, run_id)
    worker_tasks: list[asyncio.Task[None]] = []
    coord_task: asyncio.Task[None] | None = None
    uses_free_model = False
    pid_create_time = get_pid_create_time(os.getpid())

    try:
        await asyncio.to_thread(
            acquire_project_active_lock_sync,
            project_data_dir,
            run_id,
            os.getpid(),
            pid_create_time,
        )
        run_dir = project_data_dir / "runs" / run_id
        backups_dir = project_data_dir / "backups" / run_id
        m_path = manifest_path_for(project_data_dir, run_id)
        registry.attach_run_paths(project_root, run_id, project_data_dir, m_path)

        effective_max_run_usd = resolve_effective_max_run_usd(
            [t.model for t in request.tasks],
            request.max_run_usd,
        )
        limits = RunLimits(
            max_run_usd=effective_max_run_usd,
            max_steps_per_worker=request.max_steps_per_worker,
            max_429_wait_seconds=request.max_429_wait_seconds,
            request_timeout_sec=request.request_timeout_sec,
        )

        workers_state: dict[str, WorkerManifestState] = {}
        for task in request.tasks:
            chain = task_chains[task.worker_id]
            if any(m.endswith(":free") for m in chain):
                uses_free_model = True
            workers_state[task.worker_id] = WorkerManifestState(
                worker_id=task.worker_id,
                folder=to_rel_posix(folder_map[task.worker_id], project_root) or ".",
                goal=task.goal,
                target_files=list(task.target_files),
                requested_model=chain[0],
            )

        manifest = RunManifest(
            version=__version__,
            run_id=run_id,
            server_pid=os.getpid(),
            server_pid_create_time=pid_create_time,
            created_at_iso=utc_now_iso(),
            project_root=str(project_root),
            status="running",
            shared_contract=request.shared_contract,
            limits=limits,
            spend_usd=0.0,
            patch_dir=str(run_dir),
            workers=workers_state,
        )

        lock = registry.get_lock(project_root, run_id)

        async def _coordinator() -> None:
            try:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            finally:
                await registry.await_commits(project_root, run_id)
                user_cancelled = registry.is_user_cancelled(project_root, run_id)
                interrupted = registry.is_interrupted(project_root, run_id) and not user_cancelled
                budget_halted = cancel_event.is_set() and not user_cancelled and not interrupted
                try:
                    async with lock:
                        if user_cancelled:
                            for worker_st in manifest.workers.values():
                                if worker_st.status in ("queued", "running"):
                                    worker_st.status = "cancelled"
                                    worker_st.stop_reason = "cancelled_by_user"
                        elif interrupted:
                            for worker_st in manifest.workers.values():
                                if worker_st.status in ("queued", "running"):
                                    worker_st.status = "interrupted"
                                    worker_st.stop_reason = worker_st.stop_reason or "server_shutdown"
                        elif budget_halted:
                            for worker_st in manifest.workers.values():
                                if worker_st.status in ("queued", "running"):
                                    worker_st.status = "halted_budget"
                                    worker_st.stop_reason = worker_st.stop_reason or "budget_halted"
                        else:
                            for worker_st in manifest.workers.values():
                                if worker_st.status in ("queued", "running"):
                                    worker_st.status = "failed"
                                    worker_st.stop_reason = worker_st.stop_reason or "unexpected_worker_exit"
                        manifest.status = determine_final_run_status(
                            manifest,
                            budget_halted=budget_halted,
                            user_cancelled=user_cancelled,
                            interrupted=interrupted,
                        )
                        manifest.completed_at_iso = utc_now_iso()
                        await write_manifest_locked(m_path, manifest)
                        if manifest.status != "rolled_back":
                            for worker_st in manifest.workers.values():
                                if worker_st.modified_files or worker_st.created_files:
                                    try:
                                        await asyncio.to_thread(
                                            write_worker_patch_sync,
                                            project_root,
                                            run_dir,
                                            worker_st,
                                        )
                                    except (OSError, PathSecurityError, RunStateError):
                                        _logger.debug(
                                            "Best-effort patch generation failed for %s/%s",
                                            run_id,
                                            worker_st.worker_id,
                                            exc_info=True,
                                        )
                finally:
                    try:
                        await asyncio.to_thread(release_project_active_lock_sync, project_data_dir, run_id)
                    finally:
                        registry.mark_run_finished(project_root, run_id)
                try:
                    await asyncio.to_thread(prune_old_runs_sync, project_data_dir, None, run_id)
                except OSError:
                    _logger.debug("Best-effort run pruning failed for %s", run_id, exc_info=True)
                if not interrupted and not should_suppress_auto_toast():
                    completed = sum(1 for w in manifest.workers.values() if w.status == "completed")
                    total = len(manifest.workers)
                    toast_msg = (
                        f"{run_id} ({manifest.status}) — "
                        f"{completed}/{total} workers, ${manifest.spend_usd:.4f}"
                    )
                    toast_task = asyncio.create_task(
                        send_completion_toast("Overseer run finished", toast_msg),
                        name=f"overseer-toast-{run_id}",
                    )
                    _BG_TOAST_TASKS.add(toast_task)
                    toast_task.add_done_callback(_on_toast_done)

        async with lock:
            await write_manifest_locked(m_path, manifest)
            await write_latest_run_pointer(project_data_dir, run_id)
            for task in request.tasks:
                tier = task_tiers[task.worker_id]
                primary_model = task_chains[task.worker_id][0]
                t = asyncio.create_task(
                    run_worker_loop(
                        task=task,
                        folder_abs=folder_map[task.worker_id],
                        manifest=manifest,
                        manifest_path=m_path,
                        backups_dir=backups_dir,
                        run_dir=run_dir,
                        registry=registry,
                        client=client,
                        catalog=catalog,
                        api_key=tier_model_keys[(tier, primary_model)],
                        semaphore=semaphore,
                        cancel_event=cancel_event,
                        model_chain=task_chains[task.worker_id],
                        tier=tier,
                    ),
                    name=f"overseer-{run_id}-{task.worker_id}",
                )
                worker_tasks.append(t)
            coord_task = asyncio.create_task(_coordinator(), name=f"overseer-coord-{run_id}")
            registry.attach_tasks(project_root, run_id, worker_tasks)
            registry.attach_coordinator(project_root, run_id, coord_task)
    except BaseException:
        all_tasks: list[asyncio.Task[None]] = list(worker_tasks)
        if coord_task is not None:
            all_tasks.append(coord_task)
        for t in all_tasks:
            if not t.done():
                t.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        await registry.await_commits(project_root, run_id)
        await asyncio.to_thread(release_project_active_lock_sync, project_data_dir, run_id)
        registry.mark_run_finished(project_root, run_id)
        raise

    warning = (
        "Using :free OpenRouter endpoints; upstream free-tier providers may log prompts/code."
        if uses_free_model
        else None
    )
    return StartRunResponse(
        run_id=run_id,
        status="running",
        project_root=str(project_root),
        manifest_path=str(m_path),
        worker_count=len(request.tasks),
        warning=warning,
    )
