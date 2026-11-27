"""Windows path jail, external app-data storage, atomic writes, and in-memory syntax validation."""

from __future__ import annotations

import ast
import asyncio
import ctypes
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from overseer.models import StaticCheckResult, WorkerTaskInput

_T = TypeVar("_T")
MAX_VIEW_FILE_BYTES = 10 * 1024 * 1024
_STILL_ACTIVE = 259

DENY_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    ".envrc",
    "*.env",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.crt",
    "*.cer",
    "*.der",
    "*.jks",
    "*.ppk",
    "*.keystore",
    "*.tfvars",
    "*.tfvars.json",
    "*.tfstate*",
    "*.kdbx",
    "*.p8",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    ".terraform",
    ".terraform/**",
    "*.age",
    "*.gpg",
    "*.asc",
    ".vault-token",
    ".secrets",
    "*.secrets",
    "auth.json",
    "tokens.json",
    ".git-credentials",
    ".gitmodules",
    "kubeconfig",
    "*.kubeconfig",
    ".htpasswd",
    ".pgpass",
    ".my.cnf",
    "credentials.json",
    "*service-account*.json",
    "*service_account*.json",
    "*credentials*.json",
    "*credentials*.yaml",
    "*credentials*.yml",
    "*credentials*.toml",
    "*credentials*.ini",
    "*secret*.json",
    "*secret*.yaml",
    "*secret*.yml",
    "*secret*.toml",
    "*secret*.ini",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".ssh",
    ".ssh/**",
    ".gnupg",
    ".gnupg/**",
    ".kube",
    ".kube/**",
    ".docker/config.json",
    ".aws",
    ".aws/**",
    ".git",
    ".git/**",
    ".githooks",
    ".githooks/**",
    ".husky",
    ".husky/**",
    ".vscode",
    ".vscode/**",
    ".idea",
    ".idea/**",
    ".gemini",
    ".gemini/**",
    ".claude",
    ".claude/**",
    ".cursor",
    ".cursor/**",
    ".github/workflows",
    ".github/workflows/**",
    ".overseer",
    ".overseer/**",
)

_BASENAME_DENY_PATTERNS: tuple[str, ...] = tuple(
    pattern for pattern in DENY_GLOBS if "/" not in pattern
)

_PREFIX_DENY_PATHS: tuple[str, ...] = tuple(
    dict.fromkeys(
        p.removesuffix("/**")
        for p in DENY_GLOBS
        if "/" in p and "*" not in p.removesuffix("/**")
    )
)

_SECRET_REDACTION_RE = re.compile(
    r"(?i:(bearer\s+)[^\s\"']+)"
    r"|(?i:([?&](?:api[_-]?key|token|secret|key|password)=)[^&\s]+)"
    r"|(?i:(https?://)[^/\s:@]+:[^/\s@]+@)"
    r"|(?-i:sk-or-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{12,}"
    r"|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})"
)


def redact_secrets(text: str) -> str:
    """Mask Bearer tokens, URL credentials, query secrets, and common API/provider secret prefixes."""
    def _replace_secret(m: re.Match[str]) -> str:
        if m.group(1):
            return f"{m.group(1)}[REDACTED]"
        if m.group(2):
            return f"{m.group(2)}[REDACTED]"
        if m.group(3):
            return f"{m.group(3)}[REDACTED]@"
        return "[REDACTED_KEY]"

    return _SECRET_REDACTION_RE.sub(_replace_secret, text)

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "__pycache__",
        ".overseer",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        "target",
        "vendor",
    }
)
IGNORED_DIRS_LOWER: frozenset[str] = frozenset(d.lower() for d in IGNORED_DIRS)

RESERVED_DEVICE_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


class PathSecurityError(ValueError):
    """Raised when a path violates containment, Windows syntax rules, or deny-list policies."""


class BlockReplaceError(ValueError):
    """Raised when a block replacement cannot match cleanly or decode as UTF-8."""


def _win32_kernel32() -> Any:
    """Configure argtypes/restype on kernel32 process-handle functions and return kernel32."""
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


def is_pid_alive(pid: int) -> bool:
    """Return True if an OS process with pid is alive (Windows OpenProcess or POSIX kill 0)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = _win32_kernel32()
        process_query_limited_info = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_info, False, pid)
        if not handle:
            err = ctypes.GetLastError()
            return err == 5  # ERROR_ACCESS_DENIED indicates the process still exists
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == _STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def get_data_dir() -> Path:
    """Resolve the external runtime storage directory outside target repositories."""
    override = os.environ.get("OVERSEER_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "overseer"
        return Path.home() / "AppData" / "Local" / "overseer"
    return Path.home() / ".local" / "share" / "overseer"


def get_project_data_dir(project_root: Path) -> Path:
    """Return a collision-free per-project external directory under get_data_dir()."""
    norm = str(_normcase_path(project_root.resolve()))
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
    raw_name = project_root.name or "root"
    safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")[:32] or "project"
    return get_data_dir() / "projects" / f"{safe_slug}_{digest}"


def _normcase_path(path: Path) -> Path:
    raw = str(path)
    if sys.platform == "darwin":
        return Path(os.path.normcase(raw).lower())
    return Path(os.path.normcase(raw))


def is_normcase_relative_to(child: Path, parent: Path) -> bool:
    """Case-insensitive containment check suitable for Windows NTFS and macOS APFS/HFS+ paths."""
    return _normcase_path(child).is_relative_to(_normcase_path(parent))


def validate_raw_path_syntax(raw_path: str) -> None:
    """Reject dangerous Windows/POSIX path forms before performing any filesystem syscall."""
    if not raw_path or not raw_path.strip() or raw_path[0].isspace():
        raise PathSecurityError("Path must be a non-empty string without leading whitespace.")
    if "\x00" in raw_path:
        raise PathSecurityError("Null bytes are not permitted in paths.")
    if raw_path.startswith("\\\\") or raw_path.startswith("//"):
        raise PathSecurityError(f"UNC, network, or Windows device paths are not permitted: {raw_path!r}")

    # Reject root-without-drive paths like '/etc/passwd' or '\\Windows' on Windows
    if len(raw_path) >= 1 and raw_path[0] in ("/", "\\"):
        if sys.platform == "win32" or raw_path.startswith("\\"):
            raise PathSecurityError(f"Drive-less rooted paths are not permitted: {raw_path!r}")

    has_drive = len(raw_path) >= 2 and raw_path[0].isalpha() and raw_path[1] == ":"
    if has_drive:
        if len(raw_path) == 2 or raw_path[2] not in ("/", "\\"):
            raise PathSecurityError(f"Drive-relative paths are not permitted: {raw_path!r}")
        rest = raw_path[2:]
    else:
        rest = raw_path

    if ":" in rest:
        raise PathSecurityError(f"Alternate Data Streams (':') are not permitted: {raw_path!r}")

    for segment in re.split(r"[\\/]+", rest):
        if not segment:
            continue
        if segment == "..":
            raise PathSecurityError(f"Parent directory traversal ('..') is not permitted: {raw_path!r}")
        if segment != "." and (segment.endswith(" ") or segment.endswith(".")):
            raise PathSecurityError(
                f"Trailing spaces or dots are not permitted in path component: {segment!r}"
            )
        stem = segment.split(".", 1)[0].upper()
        if stem in RESERVED_DEVICE_NAMES:
            raise PathSecurityError(f"Reserved Windows device name not permitted: {segment!r}")


def is_denied_rel_path(rel_posix: str) -> bool:
    """Check whether a repository-relative POSIX path matches DENY_GLOBS (case-insensitive)."""
    cleaned = rel_posix.strip("/").replace("\\", "/").lower()
    if not cleaned or cleaned == ".":
        return False

    for prefix in _PREFIX_DENY_PATHS:
        if cleaned == prefix or cleaned.startswith(prefix + "/") or f"/{prefix}/" in f"/{cleaned}/":
            return True

    parts = [part for part in cleaned.split("/") if part and part != "."]
    for part in parts:
        for pattern in _BASENAME_DENY_PATTERNS:
            if fnmatch.fnmatchcase(part, pattern):
                return True

    for pattern in DENY_GLOBS:
        if fnmatch.fnmatchcase(cleaned, pattern.lower()):
            return True

    return False


def is_reparse_point(path: Path) -> bool:
    """Return True if path is a symlink or Windows directory junction."""
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _ensure_no_reparse_escape(target: Path, allowed_root: Path) -> None:
    """Walk existing path components under allowed_root to ensure none link outside allowed_root."""
    norm_allowed = _normcase_path(allowed_root)
    try:
        rel_parts = _normcase_path(target).relative_to(norm_allowed).parts
    except ValueError as exc:
        raise PathSecurityError(f"Path {target} escapes boundary {allowed_root}") from exc

    cur = allowed_root
    for part in rel_parts:
        cur = cur / part
        if cur.exists() or is_reparse_point(cur):
            if is_reparse_point(cur):
                try:
                    deref = cur.resolve(strict=True)
                except OSError as exc:
                    raise PathSecurityError(f"Unresolvable symlink/junction at {cur}") from exc
                if not is_normcase_relative_to(deref, allowed_root):
                    raise PathSecurityError(f"Symlink/junction escape detected at {cur} -> {deref}")


def _validate_project_root_policy(resolved: Path) -> None:
    if resolved.parent == resolved:
        raise PathSecurityError(f"project_root cannot be a filesystem root directory: {resolved}")

    home_dir = Path.home().resolve(strict=False)
    if _normcase_path(resolved) == _normcase_path(home_dir):
        raise PathSecurityError(f"project_root cannot be the user home directory: {resolved}")
    if is_normcase_relative_to(home_dir, resolved):
        raise PathSecurityError(f"project_root cannot be an ancestor of the user home directory: {resolved}")

    gemini_dir = (home_dir / ".gemini").resolve(strict=False)
    data_dir = get_data_dir().resolve(strict=False)
    if is_normcase_relative_to(resolved, gemini_dir) or is_normcase_relative_to(gemini_dir, resolved):
        raise PathSecurityError(f"project_root cannot point inside or contain ~/.gemini/: {resolved}")
    if is_normcase_relative_to(resolved, data_dir) or is_normcase_relative_to(data_dir, resolved):
        raise PathSecurityError(f"project_root cannot point inside or contain Overseer data dir: {resolved}")

    for rel_sensitive in (".ssh", ".aws", ".gnupg", ".kube", ".docker", Path(".config") / "gcloud"):
        sensitive_dir = (home_dir / rel_sensitive).resolve(strict=False)
        if is_normcase_relative_to(resolved, sensitive_dir) or is_normcase_relative_to(sensitive_dir, resolved):
            raise PathSecurityError(f"project_root cannot point inside or contain sensitive home directory: {resolved}")

    if sys.platform == "win32":
        raw_os_dirs = [
            os.environ.get("SystemRoot", r"C:\Windows"),
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("ProgramData", r"C:\ProgramData"),
        ]
    else:
        raw_os_dirs = [
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/var",
            "/proc",
            "/sys",
            "/dev",
            "/System",
            "/Library",
        ]
    for raw_sys in raw_os_dirs:
        if not raw_sys:
            continue
        sys_dir = Path(raw_sys).resolve(strict=False)
        if is_normcase_relative_to(resolved, sys_dir):
            raise PathSecurityError(f"project_root cannot point inside OS system directory: {resolved}")


def validate_historical_project_root(raw_project_root: str) -> Path:
    """Validate project_root syntax and policy without requiring a deleted repo directory to exist."""
    validate_raw_path_syntax(raw_project_root)
    candidate = Path(raw_project_root)
    if not candidate.is_absolute():
        raise PathSecurityError(f"project_root must be an absolute path: {raw_project_root!r}")
    resolved = candidate.resolve(strict=False)
    _validate_project_root_policy(resolved)
    return resolved


def resolve_project_root(raw_project_root: str) -> Path:
    """Validate and resolve the target repository root directory."""
    validate_raw_path_syntax(raw_project_root)
    candidate = Path(raw_project_root)
    if not candidate.is_absolute():
        raise PathSecurityError(f"project_root must be an absolute path: {raw_project_root!r}")

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PathSecurityError(f"project_root does not exist: {raw_project_root!r}") from exc

    if not resolved.is_dir():
        raise PathSecurityError(f"project_root is not a directory: {resolved}")

    _validate_project_root_policy(resolved)
    return resolved


def resolve_scoped_path(
    project_root: Path,
    allowed_root: Path,
    raw_path: str,
    require_exists: bool,
    allow_new_parents: bool = False,
) -> Path:
    """Validate raw_path, dereference existing ancestors, check containment, and apply DENY_GLOBS."""
    validate_raw_path_syntax(raw_path)

    resolved_project = project_root.resolve(strict=True)
    resolved_allowed = allowed_root.resolve(strict=True)

    candidate_input = Path(raw_path)
    if candidate_input.is_absolute():
        candidate = candidate_input
    else:
        candidate = resolved_project / candidate_input

    # Find deepest existing ancestor and resolve with strict=True so links/junctions are dereferenced
    probe = candidate
    missing_parts: list[str] = []
    while not probe.exists() and not is_reparse_point(probe):
        parent = probe.parent
        if parent == probe:
            raise PathSecurityError(f"No existing root ancestor found for path: {raw_path!r}")
        missing_parts.append(probe.name)
        probe = parent

    if missing_parts and require_exists:
        raise PathSecurityError(f"Target path does not exist: {raw_path!r}")
    if len(missing_parts) > 1 and not allow_new_parents:
        raise PathSecurityError(f"Parent directory does not exist for new path: {raw_path!r}")

    try:
        resolved_ancestor = probe.resolve(strict=True)
    except OSError as exc:
        raise PathSecurityError(f"Target path contains an unresolvable link: {raw_path!r}") from exc
    resolved_target = resolved_ancestor
    for part in reversed(missing_parts):
        resolved_target = resolved_target / part

    if not is_normcase_relative_to(resolved_target, resolved_project):
        raise PathSecurityError(f"Path {raw_path!r} resolves outside project_root ({resolved_project})")
    if not is_normcase_relative_to(resolved_target, resolved_allowed):
        raise PathSecurityError(f"Path {raw_path!r} resolves outside allowed folder ({resolved_allowed})")

    if is_normcase_relative_to(candidate, resolved_project):
        _ensure_no_reparse_escape(candidate, resolved_project)
    if is_normcase_relative_to(candidate, resolved_allowed):
        _ensure_no_reparse_escape(candidate, resolved_allowed)

    rel_posix = to_rel_posix(resolved_target, resolved_project)
    if is_denied_rel_path(rel_posix):
        raise PathSecurityError(f"Access denied by secret/metadata policy: {rel_posix!r}")

    return resolved_target


def to_rel_posix(path: Path, root: Path) -> str:
    """Return a normalized POSIX relative string for path within root."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return _normcase_path(path).relative_to(_normcase_path(root)).as_posix()


def validate_non_overlapping_folders(project_root: Path, tasks: list[WorkerTaskInput]) -> dict[str, Path]:
    """Resolve task folders inside project_root and reject identical or nested scopes."""
    resolved_project = project_root.resolve(strict=True)
    resolved_folders: dict[str, Path] = {}

    for task in tasks:
        folder_path = resolve_scoped_path(
            project_root=resolved_project,
            allowed_root=resolved_project,
            raw_path=task.folder,
            require_exists=True,
            allow_new_parents=False,
        )
        if not folder_path.is_dir():
            raise PathSecurityError(f"Worker {task.worker_id} folder is not a directory: {task.folder!r}")
        if len(tasks) > 1 and _normcase_path(folder_path) == _normcase_path(resolved_project):
            raise PathSecurityError("Multi-worker runs cannot scope a worker folder to project_root '.'")

        for other_id, other_path in resolved_folders.items():
            if is_normcase_relative_to(folder_path, other_path) or is_normcase_relative_to(
                other_path, folder_path
            ):
                raise PathSecurityError(
                    f"Overlapping or nested folder scopes between workers {other_id!r} and {task.worker_id!r}"
                )

        resolved_folders[task.worker_id] = folder_path

    return resolved_folders


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 digest formatted as 'sha256:<64-hex>'."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _atomic_write_sync(target_path: Path, data: bytes, mode: int | None = None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    if mode is None and sys.platform != "win32":
        try:
            existing_mode = stat.S_IMODE(target_path.stat().st_mode)
        except OSError:
            existing_mode = None
    tmp_path = target_path.parent / f".{target_path.name}.{uuid4().hex[:8]}.tmp"
    try:
        if mode is not None and os.name != "nt":
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, mode)
                view = memoryview(data)
                written = 0
                while written < len(view):
                    written += os.write(fd, view[written:])
            finally:
                os.close(fd)
        else:
            tmp_path.write_bytes(data)
            if existing_mode is not None:
                try:
                    os.chmod(tmp_path, existing_mode)
                except OSError:
                    pass
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


async def atomic_write_bytes(
    target_path: Path,
    data: bytes,
    retries: int = 3,
    backoff_sec: float = 0.05,
) -> None:
    """Write bytes atomically via sibling temp file and os.replace, retrying PermissionError."""
    last_exc: PermissionError | None = None
    for attempt in range(retries + 1):
        try:
            await asyncio.to_thread(_atomic_write_sync, target_path, data)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt >= retries:
                break
            await asyncio.sleep(backoff_sec * (2**attempt))
    assert last_exc is not None
    raise last_exc


def _retry_permission_error_sync(
    op: Callable[[], _T],
    retries: int = 3,
    backoff_sec: float = 0.05,
) -> _T:
    """Retry a synchronous file operation on transient Windows PermissionError."""
    last_exc: PermissionError | None = None
    for attempt in range(retries + 1):
        try:
            return op()
        except PermissionError as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff_sec * (2**attempt))
    assert last_exc is not None
    raise last_exc


def atomic_write_bytes_sync(
    target_path: Path,
    data: bytes,
    retries: int = 3,
    backoff_sec: float = 0.05,
    mode: int | None = None,
) -> None:
    """Synchronously write bytes atomically via sibling temp file and os.replace, retrying PermissionError."""
    _retry_permission_error_sync(
        lambda: _atomic_write_sync(target_path, data, mode=mode),
        retries=retries,
        backoff_sec=backoff_sec,
    )


def read_bytes_retry_sync(
    target_path: Path,
    retries: int = 3,
    backoff_sec: float = 0.05,
) -> bytes:
    """Read file bytes synchronously, retrying transient Windows PermissionError."""
    return _retry_permission_error_sync(
        target_path.read_bytes,
        retries=retries,
        backoff_sec=backoff_sec,
    )


def read_text_retry_sync(
    target_path: Path,
    encoding: str = "utf-8",
    retries: int = 3,
    backoff_sec: float = 0.05,
) -> str:
    """Read UTF-8 text synchronously, retrying transient Windows PermissionError."""
    return read_bytes_retry_sync(target_path, retries=retries, backoff_sec=backoff_sec).decode(encoding)


def unlink_retry_sync(
    target_path: Path,
    retries: int = 3,
    backoff_sec: float = 0.05,
) -> None:
    """Unlink a file synchronously, retrying transient Windows PermissionError."""
    _retry_permission_error_sync(
        target_path.unlink,
        retries=retries,
        backoff_sec=backoff_sec,
    )


def _format_candidate_context(lines: list[str], target_first_line: str) -> str:
    stripped_needle = target_first_line.strip()
    if stripped_needle:
        for idx, line in enumerate(lines):
            if stripped_needle in line:
                start = max(0, idx - 5)
                end = min(len(lines), idx + 6)
                snippet = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
                return f" Closest match around line {idx + 1}:\n{snippet}"
    return f" File has {len(lines)} lines."


def apply_replace_block(raw_bytes: bytes, target: str, replacement: str) -> tuple[bytes, int, int]:
    """Apply an exact unique block replacement preserving UTF-8 BOM and CRLF/LF line endings."""
    if not target:
        raise BlockReplaceError("target block must be non-empty.")

    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    payload = raw_bytes[3:] if has_bom else raw_bytes
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BlockReplaceError("File is not valid UTF-8 text and cannot be edited with replace_block.") from exc

    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    line_ending = "\r\n" if crlf_count > lf_count else "\n"

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_target = target.replace("\r\n", "\n").replace("\r", "\n")
    normalized_replacement = replacement.replace("\r\n", "\n").replace("\r", "\n")

    match_count = normalized_text.count(normalized_target)
    if match_count == 0:
        lines = normalized_text.splitlines()
        target_lines = normalized_target.splitlines()
        first_line = target_lines[0] if target_lines else ""
        hint = _format_candidate_context(lines, first_line)
        raise BlockReplaceError(f"target block not found in file (0 matches).{hint}")
    if match_count > 1:
        raise BlockReplaceError(
            f"target block matched {match_count} times; provide more surrounding lines so target matches once."
        )

    old_lines = normalized_target.splitlines()
    new_lines = normalized_replacement.splitlines()
    diff = list(difflib.ndiff(old_lines, new_lines))
    added_lines = sum(1 for entry in diff if entry.startswith("+ "))
    removed_lines = sum(1 for entry in diff if entry.startswith("- "))

    updated_norm = normalized_text.replace(normalized_target, normalized_replacement, 1)
    updated_text = updated_norm.replace("\n", line_ending) if line_ending == "\r\n" else updated_norm

    encoded = updated_text.encode("utf-8")
    if has_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    return encoded, added_lines, removed_lines


def _snippet_for_line(source: str, line_no: int | None) -> str:
    if not line_no or line_no < 1:
        return ""
    lines = source.splitlines()
    start = max(0, line_no - 4)
    end = min(len(lines), line_no + 3)
    return "\n" + "\n".join(f"  {i + 1}: {lines[i]}" for i in range(start, end))


def _blank_preserving_newlines(text: str) -> str:
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _match_luau_long_open(source: str, pos: int) -> tuple[int, int] | None:
    if pos >= len(source) or source[pos] != "[":
        return None
    j = pos + 1
    n = len(source)
    while j < n and source[j] == "=":
        j += 1
    if j < n and source[j] == "[":
        return j - pos - 1, j + 1
    return None


def _mask_luau_source(rel_path: str, source: str) -> tuple[str, list[str], bool]:
    """Strip Luau comments and mask string literals while preserving line numbers."""
    errs: list[str] = []
    out_chars: list[str] = []
    n = len(source)
    i = 0
    in_block_comment = False
    while i < n:
        if source.startswith("--", i):
            long_open = _match_luau_long_open(source, i + 2)
            if long_open is not None:
                eq_count, after_open = long_open
                closer = "]" + ("=" * eq_count) + "]"
                close_idx = source.find(closer, after_open)
                if close_idx == -1:
                    in_block_comment = True
                    out_chars.append(_blank_preserving_newlines(source[i:]))
                    break
                end_pos = close_idx + len(closer)
                out_chars.append(_blank_preserving_newlines(source[i:end_pos]))
                i = end_pos
                continue
            nl = source.find("\n", i + 2)
            if nl == -1:
                out_chars.append(" " * (n - i))
                break
            out_chars.append(" " * (nl - i))
            i = nl
            continue

        long_str = _match_luau_long_open(source, i)
        if long_str is not None:
            eq_count, after_open = long_str
            closer = "]" + ("=" * eq_count) + "]"
            close_idx = source.find(closer, after_open)
            if close_idx == -1:
                line_no = source[:i].count("\n") + 1
                errs.append(f"{rel_path}:{line_no}:1: LuauError: Unclosed '[[' long string.")
                out_chars.append(_blank_preserving_newlines(source[i:]))
                break
            end_pos = close_idx + len(closer)
            out_chars.append('""' + _blank_preserving_newlines(source[i + 2 : end_pos]))
            i = end_pos
            continue

        if source[i] in ('"', "'", "`"):
            quote = source[i]
            j = i + 1
            while j < n and source[j] != "\n":
                if source[j] == "\\":
                    j = min(n, j + 2)
                    continue
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            span = j - i
            out_chars.append('""' if span >= 2 else " ")
            if span > 2:
                out_chars.append(" " * (span - 2))
            i = j
            continue

        out_chars.append(source[i])
        i += 1
    return "".join(out_chars), errs, in_block_comment


def _check_luau_source(rel_path: str, source: str) -> list[str]:
    """Validate Luau optional strict header, deprecated globals, and ()/{} delimiter balance."""
    from overseer.config import get_config_value

    errs: list[str] = []
    if get_config_value("OVERSEER_LUAU_REQUIRE_STRICT").lower() in {"1", "true", "yes"}:
        raw_lines = source.splitlines()
        first_nonempty = next((ln.strip() for ln in raw_lines if ln.strip()), "")
        if (
            not first_nonempty.startswith("--!strict")
            and not first_nonempty.startswith("--!nocheck")
            and not first_nonempty.startswith("--!nonstrict")
        ):
            errs.append(f"{rel_path}:1:1: LuauError: Missing '--!strict' header on first non-empty line.")

    masked, lex_errs, in_block_comment = _mask_luau_source(rel_path, source)
    errs.extend(lex_errs)

    deprecated_re = re.compile(r"(?<![.\w:])\b(wait|spawn|delay|time)\s*\(")
    fn_kw_re = re.compile(r"(?<![.\w:])\bfunction\b")
    end_kw_re = re.compile(r"(?<![.\w:])\bend\b")
    open_parens = 0
    open_braces = 0
    fn_count = 0
    end_count = 0
    for idx, stripped_code in enumerate(masked.splitlines(), start=1):
        m = deprecated_re.search(stripped_code)
        if m:
            fn_name = m.group(1)
            alt = "os.clock()" if fn_name == "time" else f"task.{fn_name}()"
            errs.append(
                f"{rel_path}:{idx}:{m.start() + 1}: LuauError: Deprecated global '{fn_name}()' used; use '{alt}' instead."
            )
        fn_count += len(fn_kw_re.findall(stripped_code))
        end_count += len(end_kw_re.findall(stripped_code))
        for ch in stripped_code:
            if ch == "(":
                open_parens += 1
            elif ch == ")":
                open_parens -= 1
                if open_parens < 0:
                    errs.append(f"{rel_path}:{idx}:1: LuauError: Unexpected closing ')'.")
                    open_parens = 0
            elif ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
                if open_braces < 0:
                    errs.append(f"{rel_path}:{idx}:1: LuauError: Unexpected closing '}}'.")
                    open_braces = 0
    if in_block_comment:
        errs.append(f"{rel_path}:1:1: LuauError: Unclosed '--[[' block comment.")
    if open_parens > 0:
        errs.append(f"{rel_path}:1:1: LuauError: Unclosed '(' delimiter ({open_parens} remaining).")
    if open_braces > 0:
        errs.append(f"{rel_path}:1:1: LuauError: Unclosed '{{' delimiter ({open_braces} remaining).")
    if end_count < fn_count:
        errs.append(
            f"{rel_path}:1:1: LuauError: Unclosed 'function' block ({fn_count} 'function' vs {end_count} 'end')."
        )
    return errs


def _mask_ps1_source(source: str) -> tuple[str, bool, bool]:
    """Strip PowerShell comments and mask string/here-string literals while preserving line numbers."""
    out_chars: list[str] = []
    n = len(source)
    i = 0
    in_here: str | None = None
    unclosed_block_comment = False
    # Stack entries are "dq" for double-quoted string or ["subexpr", depth] for $(...) inside dq.
    stack: list[Any] = []

    def _is_here_open(pos: int) -> bool:
        if pos + 1 >= n or source[pos] != "@" or source[pos + 1] not in ('"', "'"):
            return False
        k = pos + 2
        while k < n and source[k] in (" ", "\t", "\r"):
            k += 1
        return k >= n or source[k] == "\n"

    while i < n:
        if in_here is not None:
            nl = source.find("\n", i)
            line_end = n if nl == -1 else nl
            line_text = source[i:line_end]
            if line_text.startswith(f"{in_here}@"):
                in_here = None
            out_chars.append(" " * (line_end - i))
            if nl != -1:
                out_chars.append("\n")
                i = nl + 1
            else:
                i = n
            continue

        top = stack[-1] if stack else None
        if top == "dq":
            ch = source[i]
            if ch == "`" and i + 1 < n:
                out_chars.append("\n" if source[i + 1] == "\n" else "  ")
                i += 2
                continue
            if ch == "$" and i + 1 < n and source[i + 1] == "(":
                stack.append(["subexpr", 1])
                out_chars.append("$(")
                i += 2
                continue
            if ch == '"':
                stack.pop()
                out_chars.append('"')
                i += 1
                continue
            out_chars.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if source.startswith("<#", i):
            close_idx = source.find("#>", i + 2)
            if close_idx == -1:
                unclosed_block_comment = True
                out_chars.append(_blank_preserving_newlines(source[i:]))
                break
            end_pos = close_idx + 2
            out_chars.append(_blank_preserving_newlines(source[i:end_pos]))
            i = end_pos
            continue

        if source[i] == "#":
            nl = source.find("\n", i + 1)
            if nl == -1:
                out_chars.append(" " * (n - i))
                break
            out_chars.append(" " * (nl - i))
            i = nl
            continue

        if _is_here_open(i):
            in_here = source[i + 1]
            nl = source.find("\n", i + 2)
            if nl == -1:
                out_chars.append(" " * (n - i))
                i = n
            else:
                out_chars.append(" " * (nl - i) + "\n")
                i = nl + 1
            continue

        if source[i] == "'":
            j = i + 1
            while j < n:
                if source[j] == "'":
                    if j + 1 < n and source[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out_chars.append("'" + _blank_preserving_newlines(source[i + 1 : j]))
            i = j
            continue

        if source[i] == '"':
            stack.append("dq")
            out_chars.append('"')
            i += 1
            continue

        ch = source[i]
        if isinstance(top, list) and top[0] == "subexpr":
            if ch == "(":
                top[1] += 1
            elif ch == ")":
                top[1] -= 1
                if top[1] == 0:
                    stack.pop()
        out_chars.append(ch)
        i += 1
    return "".join(out_chars), in_here is not None, unclosed_block_comment


def _check_ps1_source(rel_path: str, source: str) -> list[str]:
    """Validate basic PowerShell brace/parenthesis balance and unclosed here-strings."""
    errs: list[str] = []
    masked, unclosed_here, unclosed_block_comment = _mask_ps1_source(source)
    open_braces = 0
    open_parens = 0
    for idx, code_no_str in enumerate(masked.splitlines(), start=1):
        for ch in code_no_str:
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
                if open_braces < 0:
                    errs.append(f"{rel_path}:{idx}:1: PowerShellError: Unexpected closing '}}'.")
                    open_braces = 0
            elif ch == "(":
                open_parens += 1
            elif ch == ")":
                open_parens -= 1
                if open_parens < 0:
                    errs.append(f"{rel_path}:{idx}:1: PowerShellError: Unexpected closing ')'.")
                    open_parens = 0
    if unclosed_here:
        errs.append(f"{rel_path}:1:1: PowerShellError: Unclosed here-string (@\" or @').")
    if unclosed_block_comment:
        errs.append(f"{rel_path}:1:1: PowerShellError: Unclosed '<#' block comment.")
    if open_braces > 0:
        errs.append(f"{rel_path}:1:1: PowerShellError: Unclosed '{{' block ({open_braces} remaining).")
    if open_parens > 0:
        errs.append(f"{rel_path}:1:1: PowerShellError: Unclosed '(' delimiter ({open_parens} remaining).")
    return errs


_CPP_RAW_STR_RE = re.compile(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')


def _mask_cpp_source(rel_path: str, source: str) -> tuple[str, list[str], bool]:
    """Strip C/C++ comments and mask string/raw-string literals while preserving line numbers."""
    errs: list[str] = []
    out_chars: list[str] = []
    n = len(source)
    i = 0
    in_block_comment = False

    while i < n:
        if source.startswith("/*", i):
            close_idx = source.find("*/", i + 2)
            if close_idx == -1:
                in_block_comment = True
                out_chars.append(_blank_preserving_newlines(source[i:]))
                break
            end_pos = close_idx + 2
            out_chars.append(_blank_preserving_newlines(source[i:end_pos]))
            i = end_pos
            continue

        if source.startswith("//", i):
            j = i + 2
            while j < n:
                if source[j] == "\\":
                    if j + 1 < n and source[j + 1] == "\n":
                        out_chars.append(" " * (j - i) + " \n")
                        j += 2
                        i = j
                        continue
                    if j + 2 < n and source[j + 1 : j + 3] == "\r\n":
                        out_chars.append(" " * (j - i) + "  \n")
                        j += 3
                        i = j
                        continue
                if source[j] == "\n":
                    break
                j += 1
            out_chars.append(" " * (j - i))
            i = j
            continue

        raw_m = _CPP_RAW_STR_RE.match(source, i)
        if raw_m:
            delim = raw_m.group(1)
            closer = f'){delim}"'
            close_idx = source.find(closer, raw_m.end())
            if close_idx == -1:
                line_no = source[:i].count("\n") + 1
                errs.append(f"{rel_path}:{line_no}:1: CppSyntaxError: Unclosed raw string literal.")
                out_chars.append(_blank_preserving_newlines(source[i:]))
                break
            end_pos = close_idx + len(closer)
            out_chars.append('""' + _blank_preserving_newlines(source[i + 2 : end_pos]))
            i = end_pos
            continue

        if source[i] in ('"', "'"):
            quote = source[i]
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    if j + 1 < n and source[j + 1] == "\n":
                        j += 2
                        continue
                    if j + 2 < n and source[j + 1 : j + 3] == "\r\n":
                        j += 3
                        continue
                    j = min(n, j + 2)
                    continue
                if source[j] == "\n":
                    break
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            out_chars.append('""'[: max(1, min(2, j - i))] + _blank_preserving_newlines(source[i + min(2, j - i) : j]))
            i = j
            continue

        out_chars.append(source[i])
        i += 1
    return "".join(out_chars), errs, in_block_comment


def _check_cpp_source(rel_path: str, source: str) -> list[str]:
    """Validate basic C/C++ structural balance (braces, parentheses, and block comments)."""
    masked, errs, in_block_comment = _mask_cpp_source(rel_path, source)
    open_braces = 0
    open_parens = 0
    in_pp_continuation = False
    if_stack: list[tuple[int, int, list[tuple[int, int]]]] = []
    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        stripped = raw_line.strip()
        if in_pp_continuation or stripped.startswith("#"):
            in_pp_continuation = raw_line.rstrip().endswith("\\")
            if stripped.startswith("#"):
                directive = stripped[1:].lstrip().split(None, 1)[0] if stripped[1:].strip() else ""
                if directive in {"if", "ifdef", "ifndef"}:
                    if_stack.append((open_braces, open_parens, []))
                elif directive in {"elif", "else"} and if_stack:
                    base_b, base_p, branches = if_stack[-1]
                    branches.append((open_braces, open_parens))
                    open_braces, open_parens = base_b, base_p
                elif directive == "endif" and if_stack:
                    _, _, branches = if_stack.pop()
                    branches.append((open_braces, open_parens))
                    open_braces, open_parens = branches[0]
            continue

        for ch in raw_line:
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
                if open_braces < 0:
                    errs.append(f"{rel_path}:{idx}:1: CppSyntaxError: Unexpected closing '}}'.")
                    open_braces = 0
            elif ch == "(":
                open_parens += 1
            elif ch == ")":
                open_parens -= 1
                if open_parens < 0:
                    errs.append(f"{rel_path}:{idx}:1: CppSyntaxError: Unexpected closing ')'.")
                    open_parens = 0
    if in_block_comment:
        errs.append(f"{rel_path}:1:1: CppSyntaxError: Unclosed '/*' block comment.")
    if open_braces > 0:
        errs.append(f"{rel_path}:1:1: CppSyntaxError: Unclosed '{{' block ({open_braces} remaining).")
    if open_parens > 0:
        errs.append(f"{rel_path}:1:1: CppSyntaxError: Unclosed '(' delimiter ({open_parens} remaining).")
    return errs


def _clip_utf8_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _format_outline_output(
    rel_path: str,
    total_lines: int,
    symbols: list[str],
    empty_note: str,
    max_symbols: int,
    max_bytes: int,
) -> str:
    header = f"# {rel_path} ({total_lines} lines)"
    if not symbols:
        return _clip_utf8_bytes(f"{header}\n{empty_note}", max_bytes)

    limit = max(0, max_symbols)
    capped = symbols[:limit]
    truncated_by_count = len(symbols) > len(capped)
    count_marker = f"... [truncated at {limit} symbols ({len(symbols)} total)]"
    byte_marker = f"... [truncated at {max_bytes} bytes]"
    marker_reserve = max(len(count_marker.encode("utf-8")), len(byte_marker.encode("utf-8"))) + 1

    lines_out: list[str] = [header]
    cur_bytes = len(header.encode("utf-8"))
    truncated_by_bytes = False

    for sym in capped:
        entry_bytes = 1 + len(sym.encode("utf-8"))
        if cur_bytes + entry_bytes + marker_reserve > max_bytes:
            truncated_by_bytes = True
            break
        lines_out.append(sym)
        cur_bytes += entry_bytes

    if truncated_by_bytes:
        lines_out.append(byte_marker)
    elif truncated_by_count:
        lines_out.append(count_marker)

    return _clip_utf8_bytes("\n".join(lines_out), max_bytes)


def _format_py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = " ".join(ast.unparse(node.args).split())
    if len(params) > 120:
        arg_names = [a.arg for a in node.args.posonlyargs + node.args.args]
        if node.args.vararg:
            arg_names.append(f"*{node.args.vararg.arg}")
        arg_names.extend(a.arg for a in node.args.kwonlyargs)
        if node.args.kwarg:
            arg_names.append(f"**{node.args.kwarg.arg}")
        params = ", ".join(arg_names)
        if len(params) > 120:
            params = params[:117] + "..."
    ret = ""
    if node.returns is not None:
        ret_str = " ".join(ast.unparse(node.returns).split())
        if len(ret_str) > 60:
            ret_str = ret_str[:57] + "..."
        ret = f" -> {ret_str}"
    return f"{prefix} {node.name}({params}){ret}"


def _extract_python_outline(source: str, rel_path: str) -> tuple[list[str], str | None]:
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        note = f"SyntaxError at L{exc.lineno or 0}: {exc.msg}. Use view_file to inspect."
        return [], note

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_ln = node.end_lineno or node.lineno
            symbols.append(f"L{node.lineno}-L{end_ln}: {_format_py_signature(node)}")
        elif isinstance(node, ast.ClassDef):
            end_ln = node.end_lineno or node.lineno
            bases = ", ".join(" ".join(ast.unparse(b).split()) for b in node.bases)
            if len(bases) > 100:
                bases = bases[:97] + "..."
            base_part = f"({bases})" if bases else ""
            symbols.append(f"L{node.lineno}-L{end_ln}: class {node.name}{base_part}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_end = child.end_lineno or child.lineno
                    symbols.append(f"  L{child.lineno}-L{m_end}: {_format_py_signature(child)}")
    return symbols, None


def _extract_luau_outline(source: str) -> list[str]:
    symbols: list[str] = []
    fn_re = re.compile(r"^(?:local\s+)?function\s+[A-Za-z_][A-Za-z0-9_.:]*\s*\(")
    export_type_re = re.compile(r"^export\s+type\s+[A-Za-z_][A-Za-z0-9_]*\b")
    masked, _, _ = _mask_luau_source("<outline>", source)

    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        code_part = raw_line.strip()
        if not code_part:
            continue
        if fn_re.match(code_part) or export_type_re.match(code_part):
            decl = " ".join(code_part.split())
            if len(decl) > 140:
                decl = decl[:137] + "..."
            symbols.append(f"L{idx}: {decl}")
    return symbols


def _extract_ps1_outline(source: str) -> list[str]:
    symbols: list[str] = []
    fn_re = re.compile(r"^(?:function|filter)\s+[A-Za-z_][A-Za-z0-9_:-]*", re.IGNORECASE)
    class_re = re.compile(r"^class\s+[A-Za-z_][A-Za-z0-9_-]*", re.IGNORECASE)
    masked, _, _ = _mask_ps1_source(source)

    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        code_part = raw_line.strip()
        if not code_part:
            continue
        if fn_re.match(code_part) or class_re.match(code_part):
            decl = " ".join(code_part.rstrip("{").strip().split())
            if len(decl) > 140:
                decl = decl[:137] + "..."
            symbols.append(f"L{idx}: {decl}")
    return symbols


_CPP_SKIP_NAMES: frozenset[str] = frozenset(
    {"if", "else", "for", "while", "switch", "catch", "sizeof", "decltype", "alignof", "typeid", "static_assert", "return"}
)


def _looks_like_cpp_fn_def(
    brace_lines: list[str],
    line_idx: int,
    prev_nonempty: str,
) -> bool:
    line = brace_lines[line_idx].strip()
    if not line or line.startswith(
        ("}", "#", "typedef ", "using ", "namespace ", "enum ", "friend ", "return ", "public:", "private:", "protected:")
    ):
        return False
    paren_idx = line.find("(")
    if paren_idx <= 0:
        return False
    before_paren = line[:paren_idx].strip()
    if ";" in before_paren or ("=" in before_paren and "operator" not in before_paren):
        return False

    name_match = re.search(r"(?:operator\S+|~?[A-Za-z_][A-Za-z0-9_]*)\s*$", before_paren)
    if not name_match:
        return False
    fn_name = name_match.group(0).strip().lstrip("~")
    if fn_name in _CPP_SKIP_NAMES:
        return False

    prefix = before_paren[: name_match.start()].strip()
    has_scope_qual = "::" in prefix
    has_ret_tokens = bool(prefix)
    prev_had_ret = bool(prev_nonempty) and prev_nonempty[-1] not in ";{}:" and not prev_nonempty.startswith("#")
    if not (has_ret_tokens or has_scope_qual or prev_had_ret):
        return False

    # Scan forward from '(' until parens close, then verify '{' precedes ';'
    paren_depth = 0
    started = False
    for scan_idx in range(line_idx, min(line_idx + 10, len(brace_lines))):
        segment = brace_lines[scan_idx][paren_idx:] if scan_idx == line_idx else brace_lines[scan_idx]
        for ch in segment:
            if not started:
                if ch == "(":
                    paren_depth = 1
                    started = True
                continue
            if paren_depth > 0:
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                continue
            if ch == "{":
                return True
            if ch in (";", "}"):
                return False
    return False


def _extract_cpp_outline(source: str) -> list[str]:
    masked, _, _ = _mask_cpp_source("<outline>", source)
    brace_lines = [line.strip() for line in masked.splitlines()]

    class_re = re.compile(
        r"^(?:template\s*<[^;{]*>\s*)?(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b(?:\s+final\b)?(?:\s*:\s*[^{;(]+)?\s*(?:\{|;|$)"
    )
    symbols: list[str] = []
    scope_stack: list[str] = []
    pending_ns = False
    prev_nonempty = ""

    for idx, b_line in enumerate(brace_lines):
        non_ns_depth = sum(1 for item in scope_stack if item != "ns")
        stripped = b_line.strip()

        if non_ns_depth == 0 and stripped and not stripped.startswith("#"):
            if class_re.match(stripped):
                decl = " ".join(b_line.split("{", 1)[0].strip().split())
                if len(decl) > 140:
                    decl = decl[:137] + "..."
                symbols.append(f"L{idx + 1}: {decl}")
            elif _looks_like_cpp_fn_def(brace_lines, idx, prev_nonempty):
                sig = " ".join(b_line.split("{", 1)[0].strip().split())
                if len(sig) > 140:
                    sig = sig[:137] + "..."
                symbols.append(f"L{idx + 1}: {sig}")

        if stripped.startswith("namespace ") and ";" not in stripped:
            pending_ns = True
        for ch in stripped:
            if ch == "{":
                scope_stack.append("ns" if pending_ns else "other")
                pending_ns = False
            elif ch == "}" and scope_stack:
                scope_stack.pop()
        if stripped:
            prev_nonempty = stripped

    return symbols


_REGEX_PREFIX_KEYWORDS: frozenset[str] = frozenset(
    {
        "return",
        "throw",
        "case",
        "delete",
        "void",
        "typeof",
        "in",
        "instanceof",
        "new",
        "yield",
        "await",
        "else",
        "do",
        "of",
        "export",
        "default",
    }
)


def _mask_jsts_source(source: str) -> tuple[str, bool]:
    """Strip JS/TS comments and mask string/template/regex literals while preserving line numbers."""
    out_chars: list[str] = []
    n = len(source)
    i = 0
    in_block_comment = False
    can_start_regex = True

    while i < n:
        if source.startswith("/*", i):
            close_idx = source.find("*/", i + 2)
            if close_idx == -1:
                in_block_comment = True
                out_chars.append(_blank_preserving_newlines(source[i:]))
                break
            end_pos = close_idx + 2
            out_chars.append(_blank_preserving_newlines(source[i:end_pos]))
            i = end_pos
            continue
        if source.startswith("//", i):
            nl = source.find("\n", i + 2)
            if nl == -1:
                out_chars.append(" " * (n - i))
                break
            out_chars.append(" " * (nl - i))
            i = nl
            continue
        if source[i] in ('"', "'", "`"):
            quote = source[i]
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    if j + 1 < n and source[j + 1] == "\n":
                        j += 2
                        continue
                    j = min(n, j + 2)
                    continue
                if quote != "`" and source[j] == "\n":
                    break
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            span = j - i
            if quote == "`":
                out_chars.append('""'[: max(1, min(2, span))] + _blank_preserving_newlines(source[i + min(2, span) : j]))
            else:
                out_chars.append('""'[: max(1, min(2, span))] + (" " * max(0, span - 2)))
            i = j
            can_start_regex = False
            continue
        if source[i] == "/" and can_start_regex:
            j = i + 1
            in_char_class = False
            found_regex = False
            while j < n:
                ch = source[j]
                if ch == "\n":
                    break
                if ch == "\\":
                    if j + 1 < n and source[j + 1] == "\n":
                        break
                    j = min(n, j + 2)
                    continue
                if ch == "[":
                    in_char_class = True
                    j += 1
                    continue
                if ch == "]" and in_char_class:
                    in_char_class = False
                    j += 1
                    continue
                if ch == "/" and not in_char_class:
                    j += 1
                    while j < n and source[j].isalpha():
                        j += 1
                    out_chars.append('""' + (" " * max(0, (j - i) - 2)))
                    i = j
                    can_start_regex = False
                    found_regex = True
                    break
                j += 1
            if found_regex:
                continue
        ch = source[i]
        if ch.isspace():
            out_chars.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch in ("_", "$"):
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in ("_", "$")):
                j += 1
            ident = source[i:j]
            out_chars.append(ident)
            can_start_regex = ident in _REGEX_PREFIX_KEYWORDS
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in (".", "_")):
                j += 1
            out_chars.append(source[i:j])
            can_start_regex = False
            i = j
            continue
        if source.startswith("++", i) or source.startswith("--", i):
            out_chars.append(source[i : i + 2])
            can_start_regex = False
            i += 2
            continue
        out_chars.append(ch)
        can_start_regex = ch not in (")", "]")
        i += 1
    return "".join(out_chars), in_block_comment


def _check_jsts_source(rel_path: str, source: str) -> list[str]:
    """Validate basic JS/TS delimiter balance ({}, (), []) and unclosed block comments."""
    masked, in_block_comment = _mask_jsts_source(source)
    errs: list[str] = []
    counts = {"{": 0, "(": 0, "[": 0}
    closers = {"}": "{", ")": "(", "]": "["}
    for idx, line in enumerate(masked.splitlines(), start=1):
        for ch in line:
            if ch in counts:
                counts[ch] += 1
            elif ch in closers:
                opener = closers[ch]
                counts[opener] -= 1
                if counts[opener] < 0:
                    errs.append(f"{rel_path}:{idx}:1: SyntaxError: Unexpected closing '{ch}'.")
                    counts[opener] = 0
    if in_block_comment:
        errs.append(f"{rel_path}:1:1: SyntaxError: Unclosed '/*' block comment.")
    for opener, remaining in counts.items():
        if remaining > 0:
            errs.append(f"{rel_path}:1:1: SyntaxError: Unclosed '{opener}' delimiter ({remaining} remaining).")
    return errs


def _starts_yaml_quoted_token(line: str, idx: int) -> bool:
    """Return True if a quote at idx starts a YAML scalar/key token rather than an unquoted apostrophe."""
    j = idx - 1
    while j >= 0 and line[j] in (" ", "\t"):
        j -= 1
    return j < 0 or line[j] in (":", "-", "[", "{", ",")


def _check_yaml_source(rel_path: str, source: str) -> list[str]:
    """Validate YAML syntax (via PyYAML if installed, or lightweight tab/flow-delimiter/quote checks)."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        try:
            list(yaml.safe_load_all(source))
            return []
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            if mark is not None:
                return [f"{rel_path}:{mark.line + 1}:{mark.column + 1}: YAMLError: {exc}"]
            return [f"{rel_path}: YAMLError: {exc}"]

    errs: list[str] = []
    open_braces = 0
    open_brackets = 0
    block_indent: int | None = None
    for idx, raw_line in enumerate(source.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped in {"---", "..."}:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" \t"))
        if "\t" in raw_line[:indent]:
            errs.append(f"{rel_path}:{idx}:1: YAMLError: Tab character used for indentation.")
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        if re.search(r":\s+[|>][+-]?\d*\s*(?:#.*)?$", stripped):
            block_indent = indent
            continue
        in_sq = False
        in_dq = False
        i = 0
        while i < len(raw_line):
            ch = raw_line[i]
            if in_sq:
                if ch == "'":
                    if i + 1 < len(raw_line) and raw_line[i + 1] == "'":
                        i += 2
                        continue
                    in_sq = False
            elif in_dq:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_dq = False
            else:
                if ch == "#":
                    break
                if ch == "'":
                    if _starts_yaml_quoted_token(raw_line, i):
                        in_sq = True
                elif ch == '"':
                    if _starts_yaml_quoted_token(raw_line, i):
                        in_dq = True
                elif ch == "{":
                    open_braces += 1
                elif ch == "}":
                    open_braces -= 1
                    if open_braces < 0:
                        errs.append(f"{rel_path}:{idx}:1: YAMLError: Unexpected closing '}}'.")
                        open_braces = 0
                elif ch == "[":
                    open_brackets += 1
                elif ch == "]":
                    open_brackets -= 1
                    if open_brackets < 0:
                        errs.append(f"{rel_path}:{idx}:1: YAMLError: Unexpected closing ']'.")
                        open_brackets = 0
            i += 1
        if in_sq or in_dq:
            errs.append(f"{rel_path}:{idx}:1: YAMLError: Unclosed quoted string.")
    if open_braces > 0:
        errs.append(f"{rel_path}:1:1: YAMLError: Unclosed '{{' flow mapping ({open_braces} remaining).")
    if open_brackets > 0:
        errs.append(f"{rel_path}:1:1: YAMLError: Unclosed '[' flow sequence ({open_brackets} remaining).")
    return errs


def _extract_jsts_outline(source: str) -> list[str]:
    masked, _ = _mask_jsts_source(source)
    decl_re = re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:"
        r"(?:async\s+)?function\s+[A-Za-z_$][A-Za-z0-9_$]*"
        r"|(?:abstract\s+)?class\s+[A-Za-z_$][A-Za-z0-9_$]*"
        r"|interface\s+[A-Za-z_$][A-Za-z0-9_$]*"
        r"|type\s+[A-Za-z_$][A-Za-z0-9_$]*\s*(?:<[^>]*>)?\s*="
        r"|enum\s+[A-Za-z_$][A-Za-z0-9_$]*"
        r"|(?:const|let|var)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
        r")"
    )
    symbols: list[str] = []
    brace_depth = 0
    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        code_part = raw_line.strip()
        if brace_depth == 0 and code_part and decl_re.match(code_part):
            decl = " ".join(code_part.split("{", 1)[0].strip().split())
            if len(decl) > 140:
                decl = decl[:137] + "..."
            symbols.append(f"L{idx}: {decl}")
        for ch in code_part:
            if ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
    return symbols


def _extract_yaml_outline(source: str) -> list[str]:
    key_re = re.compile(r"""^([A-Za-z0-9_.-]+|"[^"]+"|'[^']+')\s*:""")
    symbols: list[str] = []
    for idx, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line or raw_line[0].isspace() or raw_line.startswith(("#", "---", "...", "-")):
            continue
        m = key_re.match(raw_line)
        if m:
            symbols.append(f"L{idx}: {m.group(1)}:")
    return symbols


def _extract_go_outline(source: str) -> list[str]:
    masked, _, _ = _mask_cpp_source("<outline>", source)
    fn_re = re.compile(r"^func\s+(?:\([^)]+\)\s+)?[A-Za-z_][A-Za-z0-9_]*")
    type_re = re.compile(r"^type\s+[A-Za-z_][A-Za-z0-9_]*\b")
    symbols: list[str] = []
    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        if not raw_line or raw_line[0].isspace():
            continue
        code_part = raw_line.strip()
        if fn_re.match(code_part) or type_re.match(code_part):
            decl = " ".join(code_part.split("{", 1)[0].strip().split())
            if len(decl) > 140:
                decl = decl[:137] + "..."
            symbols.append(f"L{idx}: {decl}")
    return symbols


def _extract_rust_outline(source: str) -> list[str]:
    masked, _, _ = _mask_cpp_source("<outline>", source)
    item_re = re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:"
        r"(?:async\s+|unsafe\s+|const\s+|extern\s+(?:\"[^\"]+\"\s+)?)*fn\s+[A-Za-z_][A-Za-z0-9_]*"
        r"|(?:struct|enum|trait|mod|type)\s+[A-Za-z_][A-Za-z0-9_]*"
        r"|impl(?:<[^>]*>)?\s+\S+"
        r")"
    )
    symbols: list[str] = []
    brace_depth = 0
    for idx, raw_line in enumerate(masked.splitlines(), start=1):
        code_part = raw_line.strip()
        if brace_depth == 0 and code_part and item_re.match(code_part):
            decl = " ".join(code_part.split("{", 1)[0].strip().split())
            if len(decl) > 140:
                decl = decl[:137] + "..."
            symbols.append(f"L{idx}: {decl}")
        for ch in code_part:
            if ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
    return symbols


def extract_file_outline(
    source: str,
    rel_path: str,
    max_symbols: int = 200,
    max_bytes: int = 16_384,
) -> str:
    """Extract a bounded structural symbol outline for supported code and YAML files."""
    total_lines = len(source.splitlines())
    ext = Path(rel_path).suffix.lower()

    if ext == ".py":
        symbols, err_note = _extract_python_outline(source, rel_path)
        note = err_note or "(no top-level classes or functions found)"
        return _format_outline_output(rel_path, total_lines, symbols, note, max_symbols, max_bytes)
    if ext == ".luau":
        symbols = _extract_luau_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no functions or exported types found)", max_symbols, max_bytes
        )
    if ext == ".ps1":
        symbols = _extract_ps1_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no functions or classes found)", max_symbols, max_bytes
        )
    if ext in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
        symbols = _extract_cpp_outline(source)
        return _format_outline_output(
            rel_path,
            total_lines,
            symbols,
            "(no classes, structs, or top-level functions found)",
            max_symbols,
            max_bytes,
        )
    if ext in {".ts", ".tsx", ".js", ".jsx"}:
        symbols = _extract_jsts_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no top-level declarations found)", max_symbols, max_bytes
        )
    if ext in {".yaml", ".yml"}:
        symbols = _extract_yaml_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no top-level keys found)", max_symbols, max_bytes
        )
    if ext == ".go":
        symbols = _extract_go_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no top-level funcs or types found)", max_symbols, max_bytes
        )
    if ext == ".rs":
        symbols = _extract_rust_outline(source)
        return _format_outline_output(
            rel_path, total_lines, symbols, "(no top-level items found)", max_symbols, max_bytes
        )

    ext_label = ext or "(no extension)"
    return _format_outline_output(
        rel_path,
        total_lines,
        [],
        f"Outline not supported for {ext_label}; use view_file or search_dir.",
        max_symbols,
        max_bytes,
    )


def _run_optional_ruff_check(rel_path: str, source: str) -> list[str]:
    """Run non-executing ruff stdin lint check (E9,F63,F7,F82) when ruff is available."""
    from overseer.config import get_config_value

    if get_config_value("OVERSEER_DISABLE_RUFF").lower() in {"1", "true", "yes"}:
        return []
    ruff_bin = shutil.which("ruff")
    if not ruff_bin:
        return []

    cmd = [
        ruff_bin,
        "check",
        "--isolated",
        "--stdin-filename",
        rel_path,
        "--select",
        "E9,F63,F7,F82",
        "--output-format",
        "concise",
        "--no-fix",
        "--no-cache",
        "-",
    ]
    minimal_env = {
        k: v
        for k in ("PATH", "SystemRoot", "TEMP", "TMP")
        if (v := os.environ.get(k)) is not None
    }
    kwargs: dict[str, Any] = {
        "input": source,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "capture_output": True,
        "timeout": 5.0,
        "check": False,
        "env": minimal_env,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    try:
        proc = subprocess.run(cmd, **kwargs)
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return []

    if proc.returncode == 0:
        return []

    findings: list[str] = []
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("Found ", "[*] ", "No fixes ", "help:", "= help:")):
            continue
        m = re.match(r"^.*?:(\d+:(?:\d+:)?)\s*(.+)$", line)
        if m:
            findings.append(f"{rel_path}: RuffError: {m.group(1)} {m.group(2)}")
        else:
            findings.append(f"{rel_path}: RuffError: {line}")
    return findings


def run_static_precheck(project_root: Path, rel_paths: list[str]) -> StaticCheckResult:
    """Validate supported code and config files strictly in memory without creating __pycache__."""
    supported_exts = {
        ".py",
        ".json",
        ".toml",
        ".luau",
        ".ps1",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".yaml",
        ".yml",
    }
    supported = [p for p in rel_paths if Path(p).suffix.lower() in supported_exts]
    if not supported:
        return StaticCheckResult(status="not_run", errors=[])

    errors: list[str] = []
    for rel_path in sorted(set(supported)):
        try:
            full_path = resolve_scoped_path(
                project_root=project_root,
                allowed_root=project_root,
                raw_path=rel_path,
                require_exists=False,
                allow_new_parents=True,
            )
        except PathSecurityError as exc:
            errors.append(f"{rel_path}: PathSecurityError: {exc}")
            continue
        if not full_path.exists():
            continue
        suffix = full_path.suffix.lower()
        try:
            if full_path.stat().st_size > MAX_VIEW_FILE_BYTES:
                continue
            source = read_text_retry_sync(full_path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{rel_path}: failed to read as UTF-8 ({exc})")
            continue

        try:
            if suffix == ".py":
                ast.parse(source, filename=rel_path)
                errors.extend(_run_optional_ruff_check(rel_path, source))
            elif suffix == ".json":
                json.loads(source)
            elif suffix == ".toml":
                tomllib.loads(source)
            elif suffix == ".luau":
                errors.extend(_check_luau_source(rel_path, source))
            elif suffix == ".ps1":
                errors.extend(_check_ps1_source(rel_path, source))
            elif suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                errors.extend(_check_cpp_source(rel_path, source))
            elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
                errors.extend(_check_jsts_source(rel_path, source))
            elif suffix in {".yaml", ".yml"}:
                errors.extend(_check_yaml_source(rel_path, source))
        except SyntaxError as exc:
            snippet = _snippet_for_line(source, exc.lineno)
            errors.append(f"{rel_path}:{exc.lineno or 0}:{exc.offset or 0}: SyntaxError: {exc.msg}{snippet}")
        except json.JSONDecodeError as exc:
            snippet = _snippet_for_line(source, exc.lineno)
            errors.append(f"{rel_path}:{exc.lineno}:{exc.colno}: JSONDecodeError: {exc.msg}{snippet}")
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"{rel_path}: TOMLDecodeError: {exc}")

    redacted_errors = [redact_secrets(err) for err in errors]
    return StaticCheckResult(status="failed" if redacted_errors else "passed", errors=redacted_errors)
