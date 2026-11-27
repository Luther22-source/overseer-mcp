"""stdio MCP server entrypoint exposing Boss tools, MCP App resource, and app-only monitor tools."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import logging.handlers
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation
from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import ClientCapabilities, ElicitationCapability, ToolAnnotations
from pydantic import BaseModel, Field

from overseer import __version__
from overseer.config import (
    MODEL_SLUG_ALIASES,
    VALID_TIERS,
    data_dir_file_lock,
    get_config_value,
    get_effective_config_snapshot,
    get_openrouter_base_url,
    load_config_file_sync,
    load_custom_instructions_sync,
    save_config_file_sync,
    save_custom_instructions_sync,
    select_api_key_for_model,
)
from overseer.guardrails import (
    get_data_dir,
    get_project_data_dir,
    redact_secrets,
    resolve_project_root,
    validate_historical_project_root,
    validate_raw_path_syntax,
)
from overseer.models import (
    ListModelsArgs,
    Max429WaitSecondsArg,
    MaxRunUsdArg,
    MaxStepsPerWorkerArg,
    OverseerConfigFile,
    ProjectRootArg,
    ProviderRoutingRule,
    RequestTimeoutSecArg,
    SharedContractArg,
    StartRunRequest,
    TasksArg,
    TierConfig,
    TierName,
)
from overseer.openrouter import ModelCatalog, fetch_openrouter_key_quotas
from overseer.state import (
    RunRegistry,
    cancel_active_run,
    collect_run_report,
    find_run_across_projects_sync,
    get_run_monitor_snapshot,
    list_all_runs_snapshot,
    read_worker_patch_sync,
    rollback_run,
    validate_and_resolve_run_id,
)
from overseer.worker import drain_toast_tasks, prepare_and_launch_run

APP_URI = "ui://overseer/app.html"
MAX_APP_HTML_BYTES = 10 * 1024 * 1024

_STDOUT_LOCK = threading.Lock()
_STDOUT_DEPTH = 0
_ORIG_STDOUT: Any = None


@contextlib.contextmanager
def _redirect_stdout_to_stderr() -> Iterator[None]:
    """Refcounted sys.stdout -> sys.stderr redirect safe across overlapping async tool calls."""
    global _STDOUT_DEPTH, _ORIG_STDOUT
    with _STDOUT_LOCK:
        if _STDOUT_DEPTH == 0:
            _ORIG_STDOUT = sys.stdout
            sys.stdout = sys.stderr
        _STDOUT_DEPTH += 1
    try:
        yield
    finally:
        with _STDOUT_LOCK:
            _STDOUT_DEPTH -= 1
            if _STDOUT_DEPTH == 0 and _ORIG_STDOUT is not None:
                sys.stdout = _ORIG_STDOUT
                _ORIG_STDOUT = None


def _load_app_html() -> str:
    """Load the self-contained Overseer Monitor HTML bundle from OVERSEER_APP_HTML or dist/app.html."""
    override = get_config_value("OVERSEER_APP_HTML")
    if override:
        validate_raw_path_syntax(override)
        html_path = Path(override).expanduser()
        if html_path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError(f"OVERSEER_APP_HTML must point to an .html/.htm file: {override!r}")
    else:
        html_path = Path(__file__).resolve().parent / "ui" / "dist" / "app.html"
    if not html_path.is_file():
        raise FileNotFoundError(f"Overseer MCP App HTML bundle not found: {html_path}")
    file_size = html_path.stat().st_size
    if file_size > MAX_APP_HTML_BYTES:
        raise ValueError(f"Overseer MCP App HTML bundle exceeds {MAX_APP_HTML_BYTES} bytes: {file_size}")
    return html_path.read_text(encoding="utf-8")


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that degrades cleanly with cooldown if another Windows process holds the log file."""

    _ROLLOVER_COOLDOWN_SEC = 60.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_rollover_fail = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if (
            self._last_rollover_fail > 0.0
            and (time.monotonic() - self._last_rollover_fail) < self._ROLLOVER_COOLDOWN_SEC
        ):
            return 0
        return super().shouldRollover(record)

    def doRollover(self) -> None:
        try:
            super().doRollover()
            self._last_rollover_fail = 0.0
        except OSError:
            self._last_rollover_fail = time.monotonic()
            if self.stream is None and not self.delay:
                try:
                    self.stream = self._open()
                except OSError:
                    self.stream = None


class _RedactingFormatter(logging.Formatter):
    """Formatter that masks secret token patterns across log messages and tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def configure_logging(data_dir: Path | None = None) -> Path:
    """Configure root logging to sys.stderr and external <data_dir>/logs/overseer.log."""
    target_dir = data_dir or get_data_dir()
    log_dir = target_dir / "logs"
    log_file = log_dir / "overseer.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    root.addHandler(stderr_handler)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = _SafeRotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Keep stderr logging active if log_dir cannot be created or another Overseer process locks the log file
        pass
    return log_file


@dataclass
class OverseerServerState:
    """Shared dependencies held for the lifetime of the MCP server."""

    registry: RunRegistry = field(default_factory=RunRegistry)
    catalog: ModelCatalog = field(default_factory=ModelCatalog)
    client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=60.0))
    config_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def aclose(self) -> None:
        await self.registry.shutdown_all()
        await drain_toast_tasks()
        await self.client.aclose()


async def _apply_config_update(
    server_state: OverseerServerState,
    tiers: dict[str, Any] | None = None,
    provider_routing: dict[str, Any] | None = None,
    max_concurrent_workers: int | None = None,
) -> dict[str, Any]:
    """Validate and persist partial tier/limit/provider updates into <data_dir>/config.json."""

    def _apply_patch_to_cfg(cfg: OverseerConfigFile) -> list[tuple[TierName, str]]:
        models_to_validate: list[tuple[TierName, str]] = []

        if max_concurrent_workers is not None:
            if isinstance(max_concurrent_workers, bool) or not isinstance(max_concurrent_workers, int):
                raise ValueError("max_concurrent_workers must be an integer between 1 and 10.")
            cfg.max_concurrent_workers = OverseerConfigFile(
                max_concurrent_workers=max_concurrent_workers
            ).max_concurrent_workers

        if tiers is not None:
            if not isinstance(tiers, dict):
                raise ValueError("tiers must be an object mapping tier names ('free', 'cheap', 'heavy').")
            for raw_tier, raw_patch in tiers.items():
                if raw_tier not in VALID_TIERS:
                    raise ValueError(f"Unknown tier {raw_tier!r}; expected one of {', '.join(VALID_TIERS)}.")
                tier: TierName = raw_tier
                if raw_patch is None:
                    cfg.tiers.pop(tier, None)
                    continue
                if not isinstance(raw_patch, dict):
                    raise ValueError(f"tiers[{tier!r}] must be an object or null.")
                unknown_keys = set(raw_patch) - {"primary", "fallbacks", "max_run_usd"}
                if unknown_keys:
                    raise ValueError(f"Unknown keys in tiers[{tier!r}]: {', '.join(sorted(unknown_keys))}.")

                updated = (cfg.tiers.get(tier) or TierConfig()).model_copy(deep=True)
                if "primary" in raw_patch:
                    raw_primary = raw_patch["primary"]
                    if raw_primary is None or (isinstance(raw_primary, str) and not raw_primary.strip()):
                        if tier in ("free", "cheap"):
                            raise ValueError(f"Tier {tier!r} requires a non-empty primary model slug.")
                        updated.primary = None
                    elif not isinstance(raw_primary, str):
                        raise ValueError(f"tiers[{tier!r}].primary must be a string.")
                    else:
                        norm_primary = MODEL_SLUG_ALIASES.get(raw_primary.strip(), raw_primary.strip())
                        if tier == "free" and not norm_primary.endswith(":free"):
                            raise ValueError(f"Free tier primary model {norm_primary!r} must end with ':free'.")
                        updated.primary = norm_primary
                        models_to_validate.append((tier, norm_primary))

                if "fallbacks" in raw_patch:
                    raw_fbs = raw_patch["fallbacks"]
                    if raw_fbs is None:
                        updated.fallbacks = None
                    else:
                        temp = TierConfig(fallbacks=raw_fbs)
                        assert temp.fallbacks is not None
                        norm_fbs: list[str] = []
                        seen_fbs: set[str] = set()
                        for slug in temp.fallbacks:
                            norm_slug = MODEL_SLUG_ALIASES.get(slug, slug)
                            if tier == "free" and not norm_slug.endswith(":free"):
                                raise ValueError(f"Free tier fallback model {norm_slug!r} must end with ':free'.")
                            if norm_slug != updated.primary and norm_slug not in seen_fbs:
                                seen_fbs.add(norm_slug)
                                norm_fbs.append(norm_slug)
                        updated.fallbacks = norm_fbs
                        for slug in norm_fbs:
                            models_to_validate.append((tier, slug))

                if "max_run_usd" in raw_patch:
                    raw_cap = raw_patch["max_run_usd"]
                    if tier == "free" and raw_cap is not None:
                        raise ValueError("The 'free' tier does not accept a max_run_usd spend limit.")
                    if raw_cap is None:
                        updated.max_run_usd = None
                    else:
                        updated.max_run_usd = TierConfig(max_run_usd=raw_cap).max_run_usd

                cfg.tiers[tier] = TierConfig.model_validate(updated.model_dump())

        if provider_routing is not None:
            if not isinstance(provider_routing, dict):
                raise ValueError("provider_routing must be an object.")
            unknown_pr = set(provider_routing) - {"tiers", "models", "replace_models"}
            if unknown_pr:
                raise ValueError(f"Unknown keys in provider_routing: {', '.join(sorted(unknown_pr))}.")

            if "tiers" in provider_routing and provider_routing["tiers"] is not None:
                pr_tiers = provider_routing["tiers"]
                if not isinstance(pr_tiers, dict):
                    raise ValueError("provider_routing.tiers must be an object.")
                for raw_tier, raw_rule in pr_tiers.items():
                    if raw_tier not in VALID_TIERS:
                        raise ValueError(
                            f"Unknown tier {raw_tier!r} in provider_routing.tiers; expected one of {', '.join(VALID_TIERS)}."
                        )
                    if raw_rule is None:
                        cfg.provider_tiers.pop(raw_tier, None)
                        continue
                    rule = ProviderRoutingRule.model_validate(raw_rule)
                    if rule.to_openrouter_dict() is None:
                        cfg.provider_tiers.pop(raw_tier, None)
                    else:
                        cfg.provider_tiers[raw_tier] = rule

            if provider_routing.get("replace_models"):
                cfg.provider_models.clear()
            if "models" in provider_routing and provider_routing["models"] is not None:
                pr_models = provider_routing["models"]
                if not isinstance(pr_models, dict):
                    raise ValueError("provider_routing.models must be an object.")
                for raw_slug, raw_rule in pr_models.items():
                    if not isinstance(raw_slug, str) or not raw_slug.strip():
                        raise ValueError("provider_routing.models keys must be non-empty model slugs.")
                    trimmed = raw_slug.strip()
                    norm_slug = MODEL_SLUG_ALIASES.get(trimmed, trimmed)
                    if raw_rule is None:
                        cfg.provider_models.pop(norm_slug, None)
                        cfg.provider_models.pop(trimmed, None)
                        continue
                    rule = ProviderRoutingRule.model_validate(raw_rule)
                    if rule.to_openrouter_dict() is None:
                        cfg.provider_models.pop(norm_slug, None)
                        cfg.provider_models.pop(trimmed, None)
                    else:
                        cfg.provider_models[norm_slug] = rule

        if len(cfg.provider_models) > 100:
            raise ValueError("provider_routing.models cannot exceed 100 entries.")

        return models_to_validate

    async with server_state.config_lock:
        preview_cfg = await asyncio.to_thread(load_config_file_sync, strict=True)
        models_to_validate = _apply_patch_to_cfg(preview_cfg)

        if models_to_validate:
            deduped = list(dict.fromkeys(models_to_validate))
            keys = [select_api_key_for_model(slug, tier=tier) for tier, slug in deduped]
            base_url = get_openrouter_base_url()
            for unique_key in dict.fromkeys(keys):
                await server_state.catalog.ensure_loaded(server_state.client, unique_key)
            for _, slug in deduped:
                server_state.catalog.filter_supported_chain(
                    [slug],
                    base_url=base_url,
                    allow_primary_fallback=False,
                )

        def _commit_locked() -> dict[str, Any]:
            with data_dir_file_lock("config"):
                latest_cfg = load_config_file_sync(strict=True)
                _apply_patch_to_cfg(latest_cfg)
                save_config_file_sync(latest_cfg)
                return get_effective_config_snapshot()

        return await asyncio.to_thread(_commit_locked)


async def _list_models_impl(
    server_state: OverseerServerState,
    free_only: bool = False,
    tools_only: bool = True,
    search: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    parsed = ListModelsArgs(
        free_only=free_only,
        tools_only=tools_only,
        search=search,
        limit=limit,
    )
    probes = (
        (("openrouter/free:free", "free"),)
        if parsed.free_only
        else (
            ("openai/gpt-4o-mini", "cheap"),
            ("openai/gpt-4o", "heavy"),
            ("openrouter/free:free", "free"),
        )
    )
    keys: list[str] = []
    for probe_model, probe_tier in probes:
        try:
            key = select_api_key_for_model(probe_model, tier=probe_tier)  # type: ignore[arg-type]
            if key not in keys:
                keys.append(key)
        except ValueError:
            continue
    if not keys:
        raise ValueError("An OpenRouter API key is required to list models.")

    base_url = get_openrouter_base_url()
    for key in keys:
        await server_state.catalog.ensure_loaded(server_state.client, key)
    models = server_state.catalog.list_models(
        free_only=parsed.free_only,
        tools_only=parsed.tools_only,
        search=parsed.search,
        limit=parsed.limit,
        base_url=base_url,
    )
    return {"models": models, "count": len(models)}


class _RollbackConfirmSchema(BaseModel):
    """Form-mode elicitation schema for confirming destructive rollback."""

    confirm: bool = Field(
        default=True,
        description="Confirm reverting created and modified files for this run.",
    )


class _MissingApiKeySchema(BaseModel):
    """Form-mode elicitation schema for supplying a missing OpenRouter API key."""

    api_key: str = Field(
        ...,
        min_length=1,
        description="OpenRouter API key (kept in process memory only; never logged).",
    )
    env_var: Literal[
        "OPENROUTER_API_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_CHEAP_API_KEY",
        "OPENROUTER_HEAVY_API_KEY",
    ] = Field(
        default="OPENROUTER_API_KEY",
        description="Target OpenRouter environment variable for this session.",
    )


def _supports_elicitation(ctx: Context | None) -> bool:
    """Return True only when an active MCP session advertises client elicitation support."""
    if ctx is None:
        return False
    try:
        req_ctx = ctx.request_context
    except ValueError:
        return False
    session = getattr(req_ctx, "session", None)
    if session is None:
        return False
    check = getattr(session, "check_client_capability", None)
    if callable(check):
        try:
            return bool(check(ClientCapabilities(elicitation=ElicitationCapability())))
        except Exception:
            return False
    caps = getattr(session, "client_capabilities", None)
    return getattr(caps, "elicitation", None) is not None


def _is_missing_api_key_error(exc: ValueError) -> bool:
    """Detect missing OpenRouter API key errors suitable for interactive elicitation."""
    msg = str(exc)
    return ("An OpenRouter API key is required" in msg) or (
        "OPENROUTER_" in msg and "is required" in msg
    )


def _default_env_var_for_key_error(msg: str) -> Literal[
    "OPENROUTER_API_KEY",
    "OPENROUTER_FREE_API_KEY",
    "OPENROUTER_CHEAP_API_KEY",
    "OPENROUTER_HEAVY_API_KEY",
]:
    if "OPENROUTER_FREE_API_KEY" in msg:
        return "OPENROUTER_FREE_API_KEY"
    if "OPENROUTER_HEAVY_API_KEY" in msg:
        return "OPENROUTER_HEAVY_API_KEY"
    if "OPENROUTER_CHEAP_API_KEY" in msg:
        return "OPENROUTER_CHEAP_API_KEY"
    return "OPENROUTER_API_KEY"


async def _maybe_elicit_missing_api_key(ctx: Context | None, exc: ValueError) -> bool:
    """Prompt the user via MCP elicitation for a missing API key; store in os.environ only."""
    if not _is_missing_api_key_error(exc) or not _supports_elicitation(ctx):
        return False
    assert ctx is not None
    default_var = _default_env_var_for_key_error(str(exc))
    prompt = (
        f"{redact_secrets(str(exc))} "
        f"Enter an OpenRouter API key for {default_var} (stored in memory for this session only, never logged)."
    )
    try:
        result = await ctx.elicit(message=prompt, schema=_MissingApiKeySchema)
    except (MCPError, ValueError, RuntimeError, AttributeError, NotImplementedError):
        return False
    if isinstance(result, AcceptedElicitation):
        raw_key = result.data.api_key.strip()
        if not raw_key or raw_key.startswith("PASTE_"):
            return False
        os.environ[result.data.env_var] = raw_key
        return True
    return False


def create_server(state: OverseerServerState | None = None) -> tuple[FastMCP, OverseerServerState]:
    """Build the FastMCP/MCPServer instance with Boss tools, MCP App resource, and app-only tools."""
    server_state = state or OverseerServerState()

    @contextlib.asynccontextmanager
    async def _lifespan(_server: Any) -> AsyncIterator[OverseerServerState]:
        try:
            yield server_state
        finally:
            await server_state.aclose()

    mcp = FastMCP(
        name="overseer",
        instructions=(
            "IDE-as-Boss OpenRouter worker pool. Call start_run with project_root and scoped tasks; "
            "wait for the user to ask for results before calling get_report; use cancel_run to stop an "
            "active run, rollback to revert, and get_config / update_config to inspect or change tier "
            "models (free, cheap, heavy), per-run USD limits, or OpenRouter provider routing."
        ),
        lifespan=_lifespan if state is None else None,
    )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def start_run(
        project_root: ProjectRootArg,
        tasks: TasksArg,
        shared_contract: SharedContractArg = None,
        max_run_usd: MaxRunUsdArg | None = None,
        max_steps_per_worker: MaxStepsPerWorkerArg = 25,
        max_429_wait_seconds: Max429WaitSecondsArg = 30,
        request_timeout_sec: RequestTimeoutSecArg = 60.0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Validate inputs and launch scoped OpenRouter workers in the background.

        Do NOT call get_report in a polling loop after calling start_run. Return the
        acknowledgment to the user and wait until the user asks to collect results.
        """
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = resolve_project_root(project_root)
                req = StartRunRequest(
                    project_root=str(resolved_root),
                    tasks=tasks,
                    shared_contract=shared_contract,
                    max_run_usd=max_run_usd,
                    max_steps_per_worker=max_steps_per_worker,
                    max_429_wait_seconds=max_429_wait_seconds,
                    request_timeout_sec=request_timeout_sec,
                )
                try:
                    resp = await prepare_and_launch_run(
                        request=req,
                        registry=server_state.registry,
                        client=server_state.client,
                        catalog=server_state.catalog,
                        project_root=resolved_root,
                    )
                except ValueError as exc:
                    if not await _maybe_elicit_missing_api_key(ctx, exc):
                        raise
                    resp = await prepare_and_launch_run(
                        request=req,
                        registry=server_state.registry,
                        client=server_state.client,
                        catalog=server_state.catalog,
                        project_root=resolved_root,
                    )
                return resp.model_dump(mode="json")
            except (ValueError, OSError, httpx.HTTPError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_report(project_root: str, run_id: str = "latest") -> dict[str, Any]:
        """Load run state for project_root, reconcile interrupted workers, and return a <= 8 KB summary."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = resolve_project_root(project_root)
                report = await collect_run_report(
                    project_root=resolved_root,
                    run_id=run_id,
                    registry=server_state.registry,
                )
                return report.model_dump(mode="json")
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    async def cancel_run(project_root: str, run_id: str = "latest") -> dict[str, Any]:
        """Stop queued and in-flight workers for an active run cleanly without reverting edits."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = resolve_project_root(project_root)
                resp = await cancel_active_run(
                    project_root=resolved_root,
                    run_id=run_id,
                    registry=server_state.registry,
                )
                return resp.model_dump(mode="json")
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    async def rollback(
        project_root: str,
        run_id: str = "latest",
        confirm: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Conflict-safely revert created and modified files for a finished or interrupted run."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = resolve_project_root(project_root)
                resolved_run_id = await asyncio.to_thread(
                    validate_and_resolve_run_id,
                    get_project_data_dir(resolved_root),
                    run_id,
                )
                if not confirm and _supports_elicitation(ctx):
                    assert ctx is not None
                    try:
                        elicit_res = await ctx.elicit(
                            message=(
                                f"Revert created and modified files for run {resolved_run_id!r} "
                                f"in {resolved_root}?"
                            ),
                            schema=_RollbackConfirmSchema,
                        )
                    except (
                        MCPError,
                        ValueError,
                        RuntimeError,
                        AttributeError,
                        NotImplementedError,
                    ):
                        elicit_res = None
                    if elicit_res is not None:
                        if isinstance(elicit_res, AcceptedElicitation):
                            if not elicit_res.data.confirm:
                                return {
                                    "run_id": resolved_run_id,
                                    "project_root": str(resolved_root),
                                    "status": "declined",
                                    "reverted": False,
                                    "message": "Rollback declined by user.",
                                }
                        elif isinstance(elicit_res, DeclinedElicitation):
                            return {
                                "run_id": resolved_run_id,
                                "project_root": str(resolved_root),
                                "status": "declined",
                                "reverted": False,
                                "message": "Rollback declined by user.",
                            }
                        else:
                            return {
                                "run_id": resolved_run_id,
                                "project_root": str(resolved_root),
                                "status": "cancelled",
                                "reverted": False,
                                "message": "Rollback cancelled by user.",
                            }
                resp = await rollback_run(
                    project_root=resolved_root,
                    run_id=resolved_run_id,
                    registry=server_state.registry,
                )
                return resp.model_dump(mode="json")
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_config() -> dict[str, Any]:
        """Return effective tier models (free/cheap/heavy), per-run USD caps, provider routing, key status, and quota preflight."""
        with _redirect_stdout_to_stderr():
            try:
                snapshot = await asyncio.to_thread(get_effective_config_snapshot)
                try:
                    snapshot["key_quotas"] = await fetch_openrouter_key_quotas(client=server_state.client)
                except Exception as exc:
                    snapshot["key_quotas"] = {
                        "status": "unavailable",
                        "error": redact_secrets(str(exc)),
                        "tiers": {},
                    }
                return snapshot
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    async def update_config(
        tiers: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
        max_concurrent_workers: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Validate models against OpenRouter /models and update tier models, spend caps, concurrency, or routing."""
        with _redirect_stdout_to_stderr():
            try:
                try:
                    return await _apply_config_update(
                        server_state,
                        tiers=tiers,
                        provider_routing=provider_routing,
                        max_concurrent_workers=max_concurrent_workers,
                    )
                except ValueError as exc:
                    if not await _maybe_elicit_missing_api_key(ctx, exc):
                        raise
                    return await _apply_config_update(
                        server_state,
                        tiers=tiers,
                        provider_routing=provider_routing,
                        max_concurrent_workers=max_concurrent_workers,
                    )
            except (ValueError, OSError, httpx.HTTPError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def list_models(
        free_only: bool = False,
        tools_only: bool = True,
        search: str | None = None,
        limit: int = 50,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """List available OpenRouter models with tool-support and pricing metadata."""
        with _redirect_stdout_to_stderr():
            try:
                try:
                    return await _list_models_impl(
                        server_state,
                        free_only=free_only,
                        tools_only=tools_only,
                        search=search,
                        limit=limit,
                    )
                except ValueError as exc:
                    if not await _maybe_elicit_missing_api_key(ctx, exc):
                        raise
                    return await _list_models_impl(
                        server_state,
                        free_only=free_only,
                        tools_only=tools_only,
                        search=search,
                        limit=limit,
                    )
            except (ValueError, OSError, httpx.HTTPError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.resource(
        APP_URI,
        name="Overseer Monitor",
        mime_type="text/html;profile=mcp-app",
        meta={
            "ui": {
                "prefersBorder": False,
                "csp": {
                    "connectDomains": [],
                    "resourceDomains": [],
                },
            }
        },
    )
    async def overseer_monitor_app() -> str:
        """Serve the self-contained Overseer Monitor MCP App HTML document."""
        with _redirect_stdout_to_stderr():
            return await asyncio.to_thread(_load_app_html)

    @mcp.tool(
        title="Overseer Monitor",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        ),
        meta={
            "ui": {"resourceUri": APP_URI},
            "com.google.antigravity/ui": {
                "entrypoints": [
                    {"type": "aux_pane", "title": "Overseer"},
                    {"type": "global", "title": "Overseer"},
                ]
            },
        },
    )
    async def open_overseer_monitor(
        project_root: str | None = None,
        run_id: str = "latest",
    ) -> dict[str, Any]:
        """Open the interactive Overseer Monitor UI. Do NOT use to collect results; call get_report instead."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = (
                    validate_historical_project_root(project_root) if project_root is not None else None
                )
                runs_data = await list_all_runs_snapshot(
                    registry=server_state.registry,
                    project_root=resolved_root,
                    limit=20,
                )
                runs = runs_data["runs"]
                selected_run: dict[str, Any] | None = None

                if resolved_root is not None:
                    if run_id != "latest":
                        selected_run = await get_run_monitor_snapshot(
                            project_root=resolved_root,
                            run_id=run_id,
                            registry=server_state.registry,
                            max_bytes=8192,
                        )
                    elif runs:
                        selected_run = await get_run_monitor_snapshot(
                            project_root=resolved_root,
                            run_id="latest",
                            registry=server_state.registry,
                            max_bytes=8192,
                        )
                elif run_id != "latest":
                    match = next((r for r in runs if r["run_id"] == run_id), None)
                    if match is not None:
                        hist_root = validate_historical_project_root(match["project_root"])
                        hist_run_id = match["run_id"]
                    else:
                        found = await asyncio.to_thread(find_run_across_projects_sync, run_id)
                        if found is None:
                            raise ValueError(f"Run {run_id!r} not found across stored projects.")
                        _, found_manifest = found
                        hist_root = validate_historical_project_root(found_manifest.project_root)
                        hist_run_id = found_manifest.run_id
                    selected_run = await get_run_monitor_snapshot(
                        project_root=hist_root,
                        run_id=hist_run_id,
                        registry=server_state.registry,
                        max_bytes=8192,
                    )
                elif runs:
                    chosen = next((r for r in runs if r["status"] == "running"), runs[0])
                    try:
                        selected_run = await get_run_monitor_snapshot(
                            project_root=validate_historical_project_root(chosen["project_root"]),
                            run_id=chosen["run_id"],
                            registry=server_state.registry,
                            max_bytes=8192,
                        )
                    except (ValueError, OSError):
                        selected_run = None

                if selected_run is not None:
                    summary = (
                        f"{len(runs)} run(s) ({runs_data['active_count']} active). "
                        f"Selected {selected_run.get('run_id', '?')} [{selected_run.get('status', '?')}] "
                        f"(${float(selected_run.get('spend_usd', 0.0)):.4f})."
                    )
                else:
                    summary = f"{len(runs)} run(s) ({runs_data['active_count']} active). No run selected."

                return {
                    "summary": summary,
                    "project_root": str(resolved_root) if resolved_root is not None else None,
                    "runs": runs,
                    "active_count": runs_data["active_count"],
                    "selected_run": selected_run,
                }
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_list_runs(
        project_root: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """App-only tool listing stored Overseer runs (all projects or scoped to project_root)."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = (
                    validate_historical_project_root(project_root) if project_root is not None else None
                )
                return await list_all_runs_snapshot(
                    registry=server_state.registry,
                    project_root=resolved_root,
                    limit=limit,
                    offset=offset,
                )
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_get_run_snapshot(project_root: str, run_id: str = "latest") -> dict[str, Any]:
        """App-only tool returning rich worker status and touched file URIs for a run."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = validate_historical_project_root(project_root)
                return await get_run_monitor_snapshot(
                    project_root=resolved_root,
                    run_id=run_id,
                    registry=server_state.registry,
                )
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_cancel_run(project_root: str, run_id: str = "latest") -> dict[str, Any]:
        """App-only tool cancelling an active run cleanly without reverting edits."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = validate_historical_project_root(project_root)
                resp = await cancel_active_run(
                    project_root=resolved_root,
                    run_id=run_id,
                    registry=server_state.registry,
                )
                return resp.model_dump(mode="json")
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_get_custom_instructions() -> dict[str, Any]:
        """App-only tool returning saved custom instructions injected into worker prompts."""
        with _redirect_stdout_to_stderr():
            try:
                return {"custom_instructions": await asyncio.to_thread(load_custom_instructions_sync)}
            except OSError as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_set_custom_instructions(instructions: str = "") -> dict[str, Any]:
        """App-only tool saving or clearing custom instructions injected into worker prompts."""
        with _redirect_stdout_to_stderr():
            try:
                saved = await asyncio.to_thread(save_custom_instructions_sync, instructions)
                return {"custom_instructions": saved}
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_get_config() -> dict[str, Any]:
        """App-only tool returning effective Overseer tier models, limits, provider routing, and key status."""
        with _redirect_stdout_to_stderr():
            try:
                return await asyncio.to_thread(get_effective_config_snapshot)
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_update_config(
        tiers: dict[str, Any] | None = None,
        provider_routing: dict[str, Any] | None = None,
        max_concurrent_workers: int | None = None,
    ) -> dict[str, Any]:
        """App-only tool validating models against OpenRouter /models and persisting config.json updates."""
        with _redirect_stdout_to_stderr():
            try:
                return await _apply_config_update(
                    server_state,
                    tiers=tiers,
                    provider_routing=provider_routing,
                    max_concurrent_workers=max_concurrent_workers,
                )
            except (ValueError, OSError, httpx.HTTPError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_list_models(
        free_only: bool = False,
        tools_only: bool = True,
        search: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """App-only tool listing OpenRouter models with tool-support and pricing metadata."""
        with _redirect_stdout_to_stderr():
            try:
                return await _list_models_impl(
                    server_state,
                    free_only=free_only,
                    tools_only=tools_only,
                    search=search,
                    limit=limit,
                )
            except (ValueError, OSError, httpx.HTTPError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_rollback_run(
        project_root: str,
        run_id: str = "latest",
        confirm: bool = True,
    ) -> dict[str, Any]:
        """App-only tool conflict-safely reverting created and modified files for a finished or interrupted run."""
        with _redirect_stdout_to_stderr():
            try:
                if not confirm:
                    raise ValueError("Rollback requires confirm=True.")
                resolved_root = validate_historical_project_root(project_root)
                resp = await rollback_run(
                    project_root=resolved_root,
                    run_id=run_id,
                    registry=server_state.registry,
                )
                return resp.model_dump(mode="json")
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    @mcp.tool(meta={"ui": {"visibility": ["app"]}})
    async def app_get_worker_patch(
        project_root: str,
        run_id: str,
        worker_id: str,
    ) -> dict[str, Any]:
        """App-only tool returning bounded unified diff patch text for one worker."""
        with _redirect_stdout_to_stderr():
            try:
                resolved_root = validate_historical_project_root(project_root)
                patch = await asyncio.to_thread(
                    read_worker_patch_sync,
                    resolved_root,
                    run_id,
                    worker_id,
                )
                return {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "patch": patch,
                }
            except (ValueError, OSError) as exc:
                raise ToolError(redact_secrets(str(exc))) from exc

    return mcp, server_state


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - OS stdio entrypoint
    """Dispatch CLI subcommands (`init`, `doctor`) or serve Overseer MCP over stdio."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  overseer-mcp          Start the Overseer stdio MCP server\n"
            "  overseer-mcp init     Configure OpenRouter API keys and print MCP client snippets\n"
            "  overseer-mcp doctor   Verify Python, keys, config, UI bundle, and OpenRouter models"
        )
        raise SystemExit(0)
    if args and args[0] in {"init", "doctor"}:
        from overseer.cli import run_doctor, run_init

        rc = run_init(args[1:]) if args[0] == "init" else run_doctor(args[1:])
        raise SystemExit(rc)

    configure_logging()
    logging.getLogger(__name__).info("Starting overseer MCP server v%s", __version__)
    mcp, _ = create_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
