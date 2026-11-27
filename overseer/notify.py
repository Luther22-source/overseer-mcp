"""Non-blocking desktop completion notifications (Windows PowerShell WinRT, macOS osascript, Linux notify-send)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys

from overseer.config import get_config_value

_TRUTHY_DISABLE_VALUES: frozenset[str] = frozenset({"1", "true", "yes"})

# Dynamic title and body are read solely from environment variables inside PowerShell
# and XML-escaped with SecurityElement::Escape to eliminate script or XML injection.
_TOAST_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$title = [System.Security.SecurityElement]::Escape([string]$env:OVERSEER_TOAST_TITLE)
$body = [System.Security.SecurityElement]::Escape([string]$env:OVERSEER_TOAST_BODY)
$xml = '<toast><visual><binding template="ToastGeneric"><text>' + $title + '</text><text>' + $body + '</text></binding></visual></toast>'
$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)
$toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
$appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
""".strip()


def _is_toast_disabled() -> bool:
    return get_config_value("OVERSEER_DISABLE_TOASTS").lower() in _TRUTHY_DISABLE_VALUES


def should_suppress_auto_toast() -> bool:
    """Return True when automatic run-completion toasts should not fire."""
    return _is_toast_disabled() or "PYTEST_CURRENT_TEST" in os.environ


async def _run_notify_subprocess(
    cmd: list[str],
    env: dict[str, str],
    timeout_sec: float,
    creationflags: int = 0,
) -> bool:
    try:
        kwargs: dict[str, object] = {
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.DEVNULL,
            "stderr": asyncio.subprocess.DEVNULL,
            "env": env,
        }
        if creationflags:
            kwargs["creationflags"] = creationflags
        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)  # type: ignore[arg-type]
    except (FileNotFoundError, OSError):
        return False

    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
        return rc == 0
    except TimeoutError:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            await proc.wait()
        except OSError:
            pass
        return False
    except asyncio.CancelledError:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            await proc.wait()
        except BaseException:
            pass
        raise
    except OSError:
        return False


def _clean_text(value: str) -> str:
    return str(value).replace("\x00", "")


async def send_completion_toast(title: str, body: str, timeout_sec: float = 5.0) -> bool:
    """Display a cross-platform desktop notification without blocking or shell interpolation."""
    if _is_toast_disabled():
        return False

    clean_title = _clean_text(title)
    clean_body = _clean_text(body)

    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
        ps_exe = os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        env: dict[str, str] = {
            "SystemRoot": system_root,
            "OVERSEER_TOAST_TITLE": clean_title,
            "OVERSEER_TOAST_BODY": clean_body,
        }
        for key in ("PATH", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return await _run_notify_subprocess(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS_SCRIPT],
            env=env,
            timeout_sec=timeout_sec,
            creationflags=creationflags,
        )

    if sys.platform == "darwin":
        osascript = shutil.which("osascript")
        if not osascript:
            return False
        env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
        return await _run_notify_subprocess(
            [
                osascript,
                "-e",
                "on run argv",
                "-e",
                "display notification (item 2 of argv) with title (item 1 of argv)",
                "-e",
                "end run",
                "--",
                clean_title,
                clean_body,
            ],
            env=env,
            timeout_sec=timeout_sec,
        )

    if sys.platform.startswith("linux"):
        notify_send = shutil.which("notify-send")
        if not notify_send:
            return False
        env = {
            k: os.environ[k]
            for k in ("PATH", "DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR")
            if k in os.environ
        }
        return await _run_notify_subprocess(
            [notify_send, "--", clean_title, clean_body],
            env=env,
            timeout_sec=timeout_sec,
        )

    return False


async def send_windows_toast(title: str, message: str, timeout_sec: float = 5.0) -> bool:
    """Backward-compatible alias for send_completion_toast."""
    return await send_completion_toast(title, message, timeout_sec=timeout_sec)
