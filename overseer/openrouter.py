"""OpenRouter model catalog, retry/failover HTTP client, and worker tool JSON schemas."""

from __future__ import annotations

import asyncio
import email.utils
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from overseer.config import (
    DEFAULT_OPENROUTER_BASE_URL,
    get_configured_tier_keys,
    get_openrouter_base_url,
    get_provider_routing_for_model,
    select_api_key_for_model,
)
from overseer.guardrails import redact_secrets, sha256_bytes
from overseer.models import (
    FileOutlineArgs,
    FindFileArgs,
    FinishArgs,
    ListDirArgs,
    ModelSummary,
    ReplaceBlockArgs,
    RunLimits,
    SearchDirArgs,
    TierName,
    ViewFileArgs,
    WriteNewFileArgs,
)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
CATALOG_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})
NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403})

WORKER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "Read up to 300 lines (and 64 KB) of a UTF-8 file inside project_root.",
            "parameters": ViewFileArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_outline",
            "description": "Extract a compact symbol outline (classes, functions, line spans) of a UTF-8 file inside project_root.",
            "parameters": FileOutlineArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List up to 200 non-ignored entries in a directory inside project_root.",
            "parameters": ListDirArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_dir",
            "description": "Search literal text across non-binary files in project_root (max 50 matches).",
            "parameters": SearchDirArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "Find up to 100 files matching a glob pattern inside project_root.",
            "parameters": FindFileArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_block",
            "description": "Replace an exact unique substring inside an editable file in your assigned folder.",
            "parameters": ReplaceBlockArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_new_file",
            "description": "Create a new UTF-8 file inside your assigned folder (fails if file exists).",
            "parameters": WriteNewFileArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Validate touched files in memory and mark the task completed.",
            "parameters": FinishArgs.model_json_schema(),
        },
    },
]


@dataclass(slots=True)
class ModelMetadata:
    """Cached capability and per-token USD pricing for one OpenRouter model."""

    model_id: str
    supports_tools: bool
    prompt_price: float
    completion_price: float
    name: str = ""
    context_length: int | None = None


class ModelCatalog:
    """TTL-cached OpenRouter /models metadata and pricing lookup."""

    def __init__(self, ttl_sec: float = 3600.0) -> None:
        self.ttl_sec = ttl_sec
        self._models: dict[str, ModelMetadata] = {}
        self._models_by_base: dict[str, dict[str, ModelMetadata]] = {}
        self._models_by_key: dict[tuple[str, str], dict[str, ModelMetadata]] = {}
        self._loaded_by_key: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    async def ensure_loaded(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        force: bool = False,
        timeout: float = 60.0,
    ) -> None:
        base_url = get_openrouter_base_url()
        cache_key = (base_url, sha256_bytes(api_key.encode("utf-8")))
        async with self._lock:
            now = time.monotonic()
            last_loaded = self._loaded_by_key.get(cache_key)
            if (
                not force
                and last_loaded is not None
                and (now - last_loaded) < self.ttl_sec
            ):
                self._models = self._models_by_base.setdefault(base_url, {})
                return

            url = f"{base_url}/models"
            response: httpx.Response | None = None
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "X-OpenRouter-Title": "Overseer MCP",
                        },
                        timeout=timeout,
                    )
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt < 2:
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    raise
                if response.status_code in CATALOG_TRANSIENT_STATUS_CODES and attempt < 2:
                    retry_after = parse_rate_limit_wait_seconds(response.headers)
                    delay = min(retry_after, 2.0) if retry_after is not None else (0.25 * (2**attempt))
                    await asyncio.sleep(max(0.05, delay))
                    continue
                break
            assert response is not None
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("data") or []

            parsed: dict[str, ModelMetadata] = {}
            for item in entries:
                mid = item.get("id")
                if not isinstance(mid, str) or not mid:
                    continue
                supported = item.get("supported_parameters") or []
                supports_tools = "tools" in supported
                pricing = item.get("pricing") or {}
                try:
                    prompt_price = float(pricing.get("prompt") or 0.0)
                except (TypeError, ValueError):
                    prompt_price = 0.0
                try:
                    completion_price = float(pricing.get("completion") or 0.0)
                except (TypeError, ValueError):
                    completion_price = 0.0
                raw_ctx = item.get("context_length")
                ctx_len = int(raw_ctx) if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool) else None
                raw_name = item.get("name")
                model_name = raw_name.strip() if isinstance(raw_name, str) else ""
                parsed[mid] = ModelMetadata(
                    model_id=mid,
                    supports_tools=supports_tools,
                    prompt_price=max(0.0, prompt_price),
                    completion_price=max(0.0, completion_price),
                    name=model_name,
                    context_length=ctx_len,
                )

            self._models_by_key[cache_key] = parsed
            merged: dict[str, ModelMetadata] = {}
            for (cached_base, _), entries_for_key in self._models_by_key.items():
                if cached_base == base_url:
                    merged.update(entries_for_key)
            self._models_by_base[base_url] = merged
            self._models = merged
            self._loaded_by_key[cache_key] = time.monotonic()

    def suggest_free_tool_models(
        self,
        limit: int = 3,
        base_url: str | None = None,
    ) -> list[str]:
        """Return up to `limit` active :free model IDs that advertise tool support."""
        models_map = self._models_by_base.get(base_url, self._models) if base_url else self._models
        candidates = [
            meta
            for meta in models_map.values()
            if meta.model_id.endswith(":free") and meta.supports_tools
        ]
        candidates.sort(key=lambda m: (-(m.context_length or 0), m.model_id))
        return [m.model_id for m in candidates[: max(0, limit)]]

    def list_models(
        self,
        free_only: bool = False,
        tools_only: bool = True,
        search: str | None = None,
        limit: int = 50,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return compact sorted model summaries filtered by free/tools/search."""
        models_map = self._models_by_base.get(base_url, self._models) if base_url else self._models
        needle = (search or "").strip().lower()
        clamped_limit = max(1, min(200, limit))
        matched: list[tuple[tuple[bool, float, str], ModelSummary]] = []
        for meta in models_map.values():
            is_free = meta.model_id.endswith(":free") or (
                meta.prompt_price == 0.0 and meta.completion_price == 0.0
            )
            if free_only and not is_free:
                continue
            if tools_only and not meta.supports_tools:
                continue
            if needle and needle not in meta.model_id.lower() and needle not in meta.name.lower():
                continue
            summary = ModelSummary(
                id=meta.model_id,
                name=meta.name,
                context_length=meta.context_length,
                prompt_price_per_m=round(meta.prompt_price * 1_000_000, 6),
                completion_price_per_m=round(meta.completion_price * 1_000_000, 6),
                is_free=is_free,
                supports_tools=meta.supports_tools,
            )
            sort_key = (not is_free, meta.prompt_price + meta.completion_price, meta.model_id)
            matched.append((sort_key, summary))
        matched.sort(key=lambda item: item[0])
        return [item[1].model_dump() for item in matched[:clamped_limit]]

    def validate_tool_support(self, model_ids: list[str], base_url: str | None = None) -> None:
        models_map = self._models_by_base.get(base_url, self._models) if base_url else self._models
        for mid in model_ids:
            meta = models_map.get(mid)
            if meta is None:
                hint = ""
                if mid.endswith(":free"):
                    suggestions = self.suggest_free_tool_models(limit=3, base_url=base_url)
                    if suggestions:
                        hint = f" Suggested active :free tool models: {', '.join(suggestions)}."
                raise ValueError(f"Model {mid!r} was not found in OpenRouter /models catalog.{hint}")
            if not meta.supports_tools:
                raise ValueError(f"Model {mid!r} does not advertise 'tools' capability on OpenRouter.")

    def filter_supported_chain(
        self,
        chain: list[str],
        base_url: str,
        *,
        allow_primary_fallback: bool = False,
    ) -> list[str]:
        """Validate primary model (or promote fallback for default/tier chains) and drop unsupported fallbacks."""
        if not chain:
            return chain
        models_map = self._models_by_base.get(base_url, self._models)
        # OpenAI-compatible custom proxies may omit OpenRouter's supported_parameters metadata
        if base_url != DEFAULT_OPENROUTER_BASE_URL and not any(m.supports_tools for m in models_map.values()):
            return list(chain)
        if not allow_primary_fallback:
            self.validate_tool_support([chain[0]], base_url=base_url)
            kept = [chain[0]]
            for fb in chain[1:]:
                meta = models_map.get(fb)
                if meta is not None and meta.supports_tools:
                    kept.append(fb)
            return kept
        kept = [mid for mid in chain if (meta := models_map.get(mid)) is not None and meta.supports_tools]
        if not kept:
            self.validate_tool_support([chain[0]], base_url=base_url)
        return kept

    def estimate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        base_url: str | None = None,
    ) -> float:
        models_map = self._models_by_base.get(base_url, self._models) if base_url else self._models
        meta = models_map.get(model_id)
        if meta is None:
            return 0.0
        return (prompt_tokens * meta.prompt_price) + (completion_tokens * meta.completion_price)


def parse_retry_after_seconds(header_value: str | None) -> float | None:
    if not header_value:
        return None
    stripped = header_value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(stripped)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (dt - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def parse_rate_limit_wait_seconds(headers: httpx.Headers | dict[str, str]) -> float | None:
    """Parse Retry-After or fallback X-RateLimit-Reset (epoch ms) into seconds to wait."""
    retry_after = parse_retry_after_seconds(headers.get("Retry-After"))
    if retry_after is not None:
        return retry_after
    reset_raw = headers.get("X-RateLimit-Reset")
    if not reset_raw:
        return None
    try:
        reset_ms = float(str(reset_raw).strip())
        return max(0.0, (reset_ms - (time.time() * 1000.0)) / 1000.0)
    except (TypeError, ValueError):
        return None


_PROMPT_CACHE_MODEL_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "qwen/",
    "google/gemini-",
)


_PROMPT_CACHE_UNSUPPORTED_MAX = 256
_prompt_cache_unsupported: dict[tuple[str, str], None] = {}
_KEY_QUOTA_CACHE_TTL_SEC = 30.0
_KEY_QUOTA_CACHE_MAX = 64
_key_quota_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def clear_key_quota_cache() -> None:
    """Clear the in-memory OpenRouter key quota TTL cache."""
    _key_quota_cache.clear()


def _mark_prompt_cache_unsupported(base_url: str, model_id: str) -> None:
    key = (base_url, model_id)
    if key in _prompt_cache_unsupported:
        return
    if len(_prompt_cache_unsupported) >= _PROMPT_CACHE_UNSUPPORTED_MAX:
        oldest = next(iter(_prompt_cache_unsupported))
        _prompt_cache_unsupported.pop(oldest, None)
    _prompt_cache_unsupported[key] = None


def _supports_prompt_caching(model_id: str, base_url: str) -> bool:
    if base_url != DEFAULT_OPENROUTER_BASE_URL:
        return False
    if (base_url, model_id) in _prompt_cache_unsupported:
        return False
    return model_id.strip().lower().startswith(_PROMPT_CACHE_MODEL_PREFIXES)


def _apply_prompt_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cached_messages = list(messages)
    if cached_messages and cached_messages[0].get("role") == "system":
        sys_content = cached_messages[0].get("content")
        if isinstance(sys_content, str) and sys_content:
            cached_messages[0] = {
                **cached_messages[0],
                "content": [
                    {
                        "type": "text",
                        "text": sys_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
    cached_tools = list(tools)
    if cached_tools:
        cached_tools[-1] = {
            **cached_tools[-1],
            "cache_control": {"type": "ephemeral"},
        }
    return cached_messages, cached_tools


def _parse_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


async def fetch_openrouter_key_quotas(
    timeout: float = 5.0,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Query GET /api/v1/key once per distinct configured tier key; never raises or leaks keys."""
    tier_keys = get_configured_tier_keys()
    if not any(tier_keys.values()):
        return {
            "status": "unconfigured",
            "tiers": {
                tier: {
                    "configured": False,
                    "status": "unconfigured",
                    "limit": None,
                    "limit_remaining": None,
                    "usage": None,
                    "is_free_tier": None,
                    "rate_limit": None,
                    "error": None,
                }
                for tier in ("free", "cheap", "heavy")
            },
        }

    try:
        base_url = get_openrouter_base_url()
    except ValueError as exc:
        err_msg = redact_secrets(str(exc))
        return {
            "status": "unavailable",
            "tiers": {
                tier: {
                    "configured": bool(tier_keys.get(tier)),
                    "status": "error" if tier_keys.get(tier) else "unconfigured",
                    "limit": None,
                    "limit_remaining": None,
                    "usage": None,
                    "is_free_tier": None,
                    "rate_limit": None,
                    "error": err_msg if tier_keys.get(tier) else None,
                }
                for tier in ("free", "cheap", "heavy")
            },
        }

    unique_keys: list[str] = []
    for key in tier_keys.values():
        if key and key not in unique_keys:
            unique_keys.append(key)

    async def _fetch_one(client: httpx.AsyncClient, api_key: str) -> dict[str, Any]:
        try:
            resp = await client.get(
                f"{base_url}/key",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-OpenRouter-Title": "Overseer MCP",
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                return {
                    "configured": True,
                    "status": "error",
                    "limit": None,
                    "limit_remaining": None,
                    "usage": None,
                    "is_free_tier": None,
                    "rate_limit": None,
                    "error": redact_secrets(f"HTTP {resp.status_code}"),
                }
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                return {
                    "configured": True,
                    "status": "error",
                    "limit": None,
                    "limit_remaining": None,
                    "usage": None,
                    "is_free_tier": None,
                    "rate_limit": None,
                    "error": "Unexpected /key response format.",
                }
            raw_rl = data.get("rate_limit")
            rate_limit: dict[str, Any] | None = None
            if isinstance(raw_rl, dict):
                rate_limit = {}
                if raw_rl.get("requests") is not None and not isinstance(raw_rl.get("requests"), bool):
                    try:
                        rate_limit["requests"] = int(raw_rl["requests"])
                    except (TypeError, ValueError):
                        pass
                if isinstance(raw_rl.get("interval"), str) and raw_rl["interval"].strip():
                    rate_limit["interval"] = raw_rl["interval"].strip()
                if not rate_limit:
                    rate_limit = None
            is_free = data.get("is_free_tier")
            return {
                "configured": True,
                "status": "ok",
                "limit": _parse_optional_float(data.get("limit")),
                "limit_remaining": _parse_optional_float(data.get("limit_remaining")),
                "usage": _parse_optional_float(data.get("usage")),
                "is_free_tier": is_free if isinstance(is_free, bool) else None,
                "rate_limit": rate_limit,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "configured": True,
                "status": "error",
                "limit": None,
                "limit_remaining": None,
                "usage": None,
                "is_free_tier": None,
                "rate_limit": None,
                "error": redact_secrets(str(exc) or type(exc).__name__),
            }

    now = time.monotonic()
    by_key: dict[str, dict[str, Any]] = {}
    uncached_keys: list[str] = []
    for key in unique_keys:
        cache_key = (base_url, sha256_bytes(key.encode("utf-8")))
        cached = _key_quota_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < _KEY_QUOTA_CACHE_TTL_SEC:
            by_key[key] = dict(cached[1])
        else:
            uncached_keys.append(key)

    if uncached_keys:
        if client is not None:
            results = await asyncio.gather(*(_fetch_one(client, k) for k in uncached_keys))
        else:
            async with httpx.AsyncClient() as owned_client:
                results = await asyncio.gather(*(_fetch_one(owned_client, k) for k in uncached_keys))
        fetched_at = time.monotonic()
        for key, res in zip(uncached_keys, results, strict=True):
            by_key[key] = res
            if res.get("status") == "ok":
                cache_key = (base_url, sha256_bytes(key.encode("utf-8")))
                if cache_key not in _key_quota_cache and len(_key_quota_cache) >= _KEY_QUOTA_CACHE_MAX:
                    oldest = next(iter(_key_quota_cache))
                    _key_quota_cache.pop(oldest, None)
                _key_quota_cache[cache_key] = (fetched_at, dict(res))

    tiers_out: dict[str, dict[str, Any]] = {}
    ok_count = 0
    configured_count = 0
    for tier in ("free", "cheap", "heavy"):
        key = tier_keys.get(tier, "")
        if not key:
            tiers_out[tier] = {
                "configured": False,
                "status": "unconfigured",
                "limit": None,
                "limit_remaining": None,
                "usage": None,
                "is_free_tier": None,
                "rate_limit": None,
                "error": None,
            }
        else:
            configured_count += 1
            entry = dict(by_key[key])
            if entry["status"] == "ok":
                ok_count += 1
            tiers_out[tier] = entry

    if ok_count == configured_count:
        overall = "ok"
    elif ok_count > 0:
        overall = "partial"
    else:
        overall = "unavailable"
    return {"status": overall, "tiers": tiers_out}


def fetch_openrouter_key_quotas_sync(timeout: float = 5.0) -> dict[str, Any]:
    """Synchronous wrapper around fetch_openrouter_key_quotas for CLI diagnostics."""
    return asyncio.run(fetch_openrouter_key_quotas(timeout=timeout))


class _WorkerCancelledError(Exception):
    """Internal signal raised when cancellation occurs during an HTTP retry/failover cycle."""


async def _post_with_cancel_flag_poll(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
    cancel_flag_path: Path | None,
    cancel_event: asyncio.Event | None = None,
) -> httpx.Response:
    if cancel_flag_path is None and cancel_event is None:
        return await client.post(url, headers=headers, json=body, timeout=timeout)
    if cancel_flag_path is not None and cancel_flag_path.exists():
        if cancel_event is not None:
            cancel_event.set()
        raise _WorkerCancelledError("Run cancelled via cancel.flag before HTTP request")
    if cancel_event is not None and cancel_event.is_set():
        raise _WorkerCancelledError("Run cancelled before HTTP request")

    post_task = asyncio.create_task(client.post(url, headers=headers, json=body, timeout=timeout))
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    try:
        wait_set: set[asyncio.Task[Any]] = {post_task}
        if cancel_task is not None:
            wait_set.add(cancel_task)
        poll_timeout = 0.25 if cancel_flag_path is not None else None
        while not post_task.done():
            done, _ = await asyncio.wait(
                wait_set,
                timeout=poll_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if post_task in done:
                return post_task.result()
            if cancel_task is not None and cancel_task in done:
                raise _WorkerCancelledError("Run cancelled during HTTP request")
            if cancel_flag_path is not None and cancel_flag_path.exists():
                if cancel_event is not None:
                    cancel_event.set()
                raise _WorkerCancelledError("Run cancelled via cancel.flag during HTTP request")
        return post_task.result()
    finally:
        cleanup: list[asyncio.Task[Any]] = []
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
            cleanup.append(cancel_task)
        if not post_task.done():
            post_task.cancel()
            cleanup.append(post_task)
        if cleanup:
            await asyncio.gather(*cleanup, return_exceptions=True)


async def _call_openrouter_with_failover(
    client: httpx.AsyncClient,
    api_key: str,
    model_chain: list[str],
    start_model_idx: int,
    messages: list[dict[str, Any]],
    limits: RunLimits,
    cancel_event: asyncio.Event | None = None,
    cancel_flag_path: Path | None = None,
    tier: TierName | None = None,
) -> tuple[dict[str, Any], str, int]:
    """Call OpenRouter chat completions with bounded retries and fallback without advancing steps_used."""
    url = f"{get_openrouter_base_url()}/chat/completions"

    def _is_cancelled() -> bool:
        if cancel_flag_path is not None and cancel_flag_path.exists():
            if cancel_event is not None:
                cancel_event.set()
            return True
        return cancel_event is not None and cancel_event.is_set()

    async def _sleep_or_cancel(delay: float) -> None:
        if _is_cancelled():
            raise _WorkerCancelledError("Run cancelled before retry wait")
        remaining = max(0.0, delay)
        if cancel_event is None and cancel_flag_path is None:
            await asyncio.sleep(remaining)
            return
        poll_slice = 0.25 if cancel_flag_path is not None else max(remaining, 0.0)
        deadline = time.monotonic() + remaining
        while True:
            if _is_cancelled():
                raise _WorkerCancelledError("Run cancelled during retry wait")
            left = deadline - time.monotonic()
            if left <= 0.0:
                return
            wait_sec = min(left, poll_slice) if cancel_flag_path is not None else left
            if cancel_event is None:
                await asyncio.sleep(wait_sec)
            else:
                try:
                    await asyncio.wait_for(cancel_event.wait(), timeout=wait_sec)
                    raise _WorkerCancelledError("Run cancelled during retry wait")
                except TimeoutError:
                    pass
            if _is_cancelled():
                raise _WorkerCancelledError("Run cancelled via cancel.flag")

    idx = start_model_idx
    while idx < len(model_chain):
        if _is_cancelled():
            raise _WorkerCancelledError("Run cancelled before model call")
        model_id = model_chain[idx]
        if tier == "free" and not model_id.strip().endswith(":free"):
            raise ValueError(
                f"Free tier model {model_id!r} must end with ':free' (refusing to use paid model)."
            )
        try:
            model_api_key = select_api_key_for_model(model_id, tier=tier)
        except ValueError:
            same_tier = model_id.endswith(":free") == model_chain[0].endswith(":free")
            if not api_key or not same_tier:
                raise
            model_api_key = api_key
        base_url = get_openrouter_base_url()
        use_prompt_cache = _supports_prompt_caching(model_id, base_url)
        provider_pref = get_provider_routing_for_model(model_id, tier=tier)
        headers = {
            "Authorization": f"Bearer {model_api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Overseer MCP",
        }
        for attempt in range(3):
            if _is_cancelled():
                raise _WorkerCancelledError("Run cancelled before HTTP attempt")
            if use_prompt_cache:
                req_messages, req_tools = _apply_prompt_cache_control(messages, WORKER_TOOL_SCHEMAS)
            else:
                req_messages, req_tools = messages, WORKER_TOOL_SCHEMAS
            body: dict[str, Any] = {
                "model": model_id,
                "messages": req_messages,
                "tools": req_tools,
                "usage": {"include": True},
            }
            if provider_pref:
                body["provider"] = provider_pref
            try:
                resp = await _post_with_cancel_flag_poll(
                    client, url, headers, body, limits.request_timeout_sec, cancel_flag_path, cancel_event
                )
                if resp.status_code == 400 and use_prompt_cache:
                    use_prompt_cache = False
                    _mark_prompt_cache_unsupported(base_url, model_id)
                    body["messages"] = messages
                    body["tools"] = WORKER_TOOL_SCHEMAS
                    resp = await _post_with_cancel_flag_poll(
                        client, url, headers, body, limits.request_timeout_sec, cancel_flag_path, cancel_event
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if _is_cancelled():
                    raise _WorkerCancelledError("Run cancelled during HTTP request") from exc
                if attempt < 2:
                    await _sleep_or_cancel(
                        min(0.25 * (2**attempt), max(0.05, float(limits.max_429_wait_seconds)))
                    )
                    continue
                if idx + 1 < len(model_chain):
                    break
                raise RuntimeError(redact_secrets(f"openrouter_network_error: {exc}")) from exc

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    if attempt < 2:
                        await _sleep_or_cancel(
                            min(0.25 * (2**attempt), max(0.05, float(limits.max_429_wait_seconds)))
                        )
                        continue
                    if idx + 1 < len(model_chain):
                        break
                    raise RuntimeError(redact_secrets(f"openrouter_invalid_json: {exc}")) from exc
                if not isinstance(data, dict):
                    if attempt < 2:
                        await _sleep_or_cancel(
                            min(0.25 * (2**attempt), max(0.05, float(limits.max_429_wait_seconds)))
                        )
                        continue
                    if idx + 1 < len(model_chain):
                        break
                    raise RuntimeError("openrouter_malformed_response: Expected JSON object.")
                if data.get("error") is not None and not data.get("choices"):
                    err_obj = data["error"]
                    err_code: int | None = None
                    if isinstance(err_obj, dict):
                        raw_code = err_obj.get("code")
                        if isinstance(raw_code, int) and not isinstance(raw_code, bool):
                            err_code = raw_code
                        err_msg = str(err_obj.get("message") or err_obj)
                    else:
                        err_msg = str(err_obj)
                    if err_code == 404 or (
                        err_code == 402 and any(m.endswith(":free") for m in model_chain[idx + 1 :])
                    ):
                        if idx + 1 < len(model_chain):
                            break
                    if err_code in RETRYABLE_STATUS_CODES:
                        if err_code == 429 and limits.max_429_wait_seconds <= 0:
                            break
                        if attempt < 2:
                            await _sleep_or_cancel(
                                min(0.25 * (2**attempt), max(0.05, float(limits.max_429_wait_seconds)))
                            )
                            continue
                        if idx + 1 < len(model_chain):
                            break
                    raise RuntimeError(redact_secrets(f"openrouter_http_200_error: {err_msg[:300]}"))
                choices = data.get("choices")
                first_choice = choices[0] if isinstance(choices, list) and choices else None
                if not isinstance(first_choice, dict) or not isinstance(first_choice.get("message"), dict):
                    if attempt < 2:
                        await _sleep_or_cancel(
                            min(0.25 * (2**attempt), max(0.05, float(limits.max_429_wait_seconds)))
                        )
                        continue
                    if idx + 1 < len(model_chain):
                        break
                    raise RuntimeError("openrouter_malformed_response: Missing choices[0].message.")
                return data, model_id, idx

            if resp.status_code == 404 or (
                resp.status_code == 402 and any(m.endswith(":free") for m in model_chain[idx + 1 :])
            ):
                if idx + 1 < len(model_chain):
                    break
                raise RuntimeError(redact_secrets(f"openrouter_http_{resp.status_code}: {resp.text[:300]}"))

            if resp.status_code == 402 or resp.status_code in NON_RETRYABLE_STATUS_CODES:
                raise RuntimeError(redact_secrets(f"openrouter_http_{resp.status_code}: {resp.text[:300]}"))

            if resp.status_code in RETRYABLE_STATUS_CODES:
                if resp.status_code == 429 and limits.max_429_wait_seconds <= 0:
                    break  # Fail over immediately to next model in chain
                retry_after = parse_rate_limit_wait_seconds(resp.headers)
                if retry_after is not None and retry_after > limits.max_429_wait_seconds:
                    break  # Fail over immediately to next model in chain
                if attempt < 2:
                    delay = retry_after if retry_after is not None else (0.25 * (2**attempt))
                    await _sleep_or_cancel(min(delay, max(0.05, float(limits.max_429_wait_seconds))))
                    continue
                break

            raise RuntimeError(redact_secrets(f"openrouter_http_{resp.status_code}: {resp.text[:300]}"))
        idx += 1

    raise RuntimeError("all_models_exhausted: All primary and fallback OpenRouter models failed after retries.")
