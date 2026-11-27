"""External .env/config.json settings, model tier chains, API keys, provider routing, and instructions."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import urllib.parse
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from overseer.guardrails import (
    atomic_write_bytes_sync,
    get_data_dir,
    is_pid_alive,
    read_text_retry_sync,
    unlink_retry_sync,
)
from overseer.models import OverseerConfigFile, TierName

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_FREE_MODEL_SLUG = "particle-ai/particle-1:free"
DEFAULT_FREE_FALLBACK_SLUGS: tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
)
DEFAULT_CHEAP_MODEL_SLUG = "deepseek/deepseek-v4.1-flash-1017"
DEFAULT_CHEAP_FALLBACK_SLUGS: tuple[str, ...] = ("z-ai/glm-5.3-flash",)
DEFAULT_CHEAP_MAX_RUN_USD = 0.25
DEFAULT_HEAVY_MAX_RUN_USD = 2.00
DEFAULT_MAX_CONCURRENT_WORKERS = 3
VALID_TIERS: tuple[TierName, ...] = ("free", "cheap", "heavy")
MODEL_SLUG_ALIASES: dict[str, str] = {
    "particle-ai/particle-1-20261027:free": "particle-ai/particle-1:free",
    "nvidia/nemotron-3-ultra-550b-a55b-20260604:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it-20260402:free": "google/gemma-4-31b-it:free",
    "deepseek/deepseek-v4.1-flash-20261017": "deepseek/deepseek-v4.1-flash-1017",
    "z-ai/glm-5.3-flash-20260826": "z-ai/glm-5.3-flash",
}
MAX_CUSTOM_INSTRUCTIONS_CHARS = 20_000

_logger = logging.getLogger(__name__)
_DOTENV_CACHE: tuple[str, int, int, int, dict[str, str]] | None = None
_CONFIG_CACHE: tuple[str, int, int, int, OverseerConfigFile] | None = None


def load_external_dotenv() -> dict[str, str]:
    """Load optional KEY=VALUE pairs from <data_dir>/.env outside any code repo (cached by stat)."""
    global _DOTENV_CACHE
    env_path = get_data_dir() / ".env"
    try:
        st = env_path.stat()
    except OSError:
        _DOTENV_CACHE = None
        return {}
    if not env_path.is_file():
        _DOTENV_CACHE = None
        return {}

    path_key = str(env_path)
    if (
        _DOTENV_CACHE is not None
        and _DOTENV_CACHE[0] == path_key
        and _DOTENV_CACHE[1] == st.st_mtime_ns
        and _DOTENV_CACHE[2] == st.st_size
        and _DOTENV_CACHE[3] == st.st_ino
    ):
        return dict(_DOTENV_CACHE[4])

    try:
        raw_text = read_text_retry_sync(env_path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}

    parsed: dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        if key:
            parsed[key] = value
    _DOTENV_CACHE = (path_key, st.st_mtime_ns, st.st_size, st.st_ino, parsed)
    return dict(parsed)


def get_env_only_value(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key].strip()
    return None


def get_dotenv_value(key: str) -> str | None:
    dotenv = load_external_dotenv()
    if key in dotenv:
        return dotenv[key].strip()
    return None


def get_config_value(key: str, default: str = "") -> str:
    if key in os.environ:
        return os.environ[key].strip()
    dotenv = load_external_dotenv()
    if key in dotenv:
        return dotenv[key].strip()
    return default


def _parse_lock_pid(content: str) -> int | None:
    raw_pid = content.strip().split(":", 1)[0].strip()
    if not raw_pid:
        return None
    try:
        return int(raw_pid)
    except ValueError:
        return None


@contextlib.contextmanager
def data_dir_file_lock(
    lock_name: str,
    *,
    timeout_sec: float = 5.0,
    stale_sec: float = 10.0,
) -> Iterator[None]:
    """Acquire an exclusive cross-process lock file under <data_dir> with PID-checked stale reclaim."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / f".{lock_name}.lock"
    deadline = time.monotonic() + timeout_sec
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    token_bytes = token.encode("ascii", errors="ignore")

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, token_bytes)
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                st = lock_path.stat()
                if (time.time() - st.st_mtime) > stale_sec:
                    try:
                        existing = lock_path.read_text(encoding="ascii", errors="ignore")
                    except OSError:
                        existing = ""
                    holder_pid = _parse_lock_pid(existing)
                    if holder_pid is None or not is_pid_alive(holder_pid):
                        lock_path.unlink(missing_ok=True)
                        continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Overseer lock {lock_path}")
            time.sleep(0.02)

    try:
        yield
    finally:
        try:
            current = lock_path.read_text(encoding="ascii", errors="ignore").strip()
            if current == token:
                unlink_retry_sync(lock_path)
        except (FileNotFoundError, OSError):
            pass


def config_file_path() -> Path:
    """Return external path for structured non-secret Overseer settings (<data_dir>/config.json)."""
    return get_data_dir() / "config.json"


def load_config_file_sync(*, strict: bool = False) -> OverseerConfigFile:
    """Load structured config from <data_dir>/config.json (stat-cached)."""
    global _CONFIG_CACHE
    path = config_file_path()
    try:
        st = path.stat()
    except FileNotFoundError:
        _CONFIG_CACHE = None
        return OverseerConfigFile()
    except OSError as exc:
        _CONFIG_CACHE = None
        if strict:
            raise ValueError(
                f"Cannot update Overseer config.json at {path} (fix or remove the file first): {exc}"
            ) from exc
        _logger.warning("Ignoring unreadable Overseer config.json at %s: %s", path, exc)
        return OverseerConfigFile()
    if not path.is_file():
        _CONFIG_CACHE = None
        if strict:
            raise ValueError(
                f"Cannot update Overseer config.json at {path} (fix or remove the file first): path is not a file."
            )
        return OverseerConfigFile()

    path_key = str(path)
    if (
        _CONFIG_CACHE is not None
        and _CONFIG_CACHE[0] == path_key
        and _CONFIG_CACHE[1] == st.st_mtime_ns
        and _CONFIG_CACHE[2] == st.st_size
        and _CONFIG_CACHE[3] == st.st_ino
    ):
        return _CONFIG_CACHE[4].model_copy(deep=True)

    try:
        raw_text = read_text_retry_sync(path, encoding="utf-8-sig")
        raw_obj = json.loads(raw_text)
        cfg = OverseerConfigFile.model_validate(raw_obj)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        if strict:
            raise ValueError(
                f"Cannot update Overseer config.json at {path} (fix or remove the file first): {exc}"
            ) from exc
        _logger.warning("Ignoring invalid Overseer config.json at %s: %s", path, exc)
        return OverseerConfigFile()

    _CONFIG_CACHE = (path_key, st.st_mtime_ns, st.st_size, st.st_ino, cfg)
    return cfg.model_copy(deep=True)


def save_config_file_sync(cfg: OverseerConfigFile) -> OverseerConfigFile:
    """Validate and atomically save <data_dir>/config.json, refreshing the stat cache."""
    global _CONFIG_CACHE
    validated = OverseerConfigFile.model_validate(cfg.model_dump())
    path = config_file_path()
    payload = validated.model_dump_json(indent=2) + "\n"
    atomic_write_bytes_sync(path, payload.encode("utf-8"))
    try:
        st = path.stat()
        _CONFIG_CACHE = (str(path), st.st_mtime_ns, st.st_size, st.st_ino, validated)
    except OSError:
        _CONFIG_CACHE = None
    return validated.model_copy(deep=True)


def get_openrouter_base_url() -> str:
    val = (
        get_config_value("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        or DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")
    parsed = urllib.parse.urlparse(val)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid OPENROUTER_BASE_URL: {val!r}")
    scheme = parsed.scheme.lower()
    if scheme == "https":
        return val
    host = (parsed.hostname or "").lower()
    allow_insecure = get_config_value("OVERSEER_ALLOW_INSECURE_BASE_URL").lower() in {"1", "true", "yes"}
    if scheme == "http" and (host in {"localhost", "127.0.0.1", "::1"} or allow_insecure):
        return val
    raise ValueError(
        f"OPENROUTER_BASE_URL must use https:// (or http://localhost unless OVERSEER_ALLOW_INSECURE_BASE_URL=1): {val!r}"
    )


def _filter_tier_fallbacks(tier: TierName, items: list[str], source_label: str) -> list[str]:
    if tier != "free":
        return items
    kept: list[str] = []
    for slug in items:
        if slug.endswith(":free"):
            kept.append(slug)
        else:
            _logger.warning("Ignoring non-:free fallback slug in %s: %s", source_label, slug)
    return kept


def get_default_model() -> str:
    env_val = get_env_only_value("OVERSEER_FREE_DEFAULT_MODEL") or get_env_only_value("OVERSEER_DEFAULT_MODEL")
    if env_val:
        return env_val
    tier_cfg = load_config_file_sync().tiers.get("free")
    if tier_cfg is not None and tier_cfg.primary:
        return tier_cfg.primary
    dotenv_val = get_dotenv_value("OVERSEER_FREE_DEFAULT_MODEL") or get_dotenv_value("OVERSEER_DEFAULT_MODEL")
    return dotenv_val or DEFAULT_FREE_MODEL_SLUG


def _resolve_fallback_models(tier: TierName, tier_key: str, defaults: tuple[str, ...]) -> list[str]:
    # Precedence: os.environ > config.json > .env > hardcoded defaults
    if tier_key in os.environ:
        items = [item.strip() for item in os.environ[tier_key].split(",") if item.strip()]
        return _filter_tier_fallbacks(tier, items, tier_key)
    if tier in ("free", "cheap") and "OVERSEER_FALLBACK_MODELS" in os.environ:
        items = [item.strip() for item in os.environ["OVERSEER_FALLBACK_MODELS"].split(",") if item.strip()]
        if not items:
            return []
        return [m for m in items if m.endswith(":free")] if tier == "free" else [m for m in items if not m.endswith(":free")]

    tier_cfg = load_config_file_sync().tiers.get(tier)
    if tier_cfg is not None and tier_cfg.fallbacks is not None:
        return _filter_tier_fallbacks(tier, list(tier_cfg.fallbacks), f"config.json[{tier}].fallbacks")

    dotenv = load_external_dotenv()
    if tier_key in dotenv:
        items = [item.strip() for item in dotenv[tier_key].split(",") if item.strip()]
        return _filter_tier_fallbacks(tier, items, tier_key)
    if tier in ("free", "cheap") and "OVERSEER_FALLBACK_MODELS" in dotenv:
        items = [item.strip() for item in dotenv["OVERSEER_FALLBACK_MODELS"].split(",") if item.strip()]
        if not items:
            return []
        return [m for m in items if m.endswith(":free")] if tier == "free" else [m for m in items if not m.endswith(":free")]

    return list(defaults)


def get_fallback_models() -> list[str]:
    return _resolve_fallback_models("free", "OVERSEER_FREE_FALLBACK_MODELS", DEFAULT_FREE_FALLBACK_SLUGS)


def get_cheap_default_model() -> str:
    env_val = get_env_only_value("OVERSEER_CHEAP_DEFAULT_MODEL")
    if env_val:
        return env_val
    tier_cfg = load_config_file_sync().tiers.get("cheap")
    if tier_cfg is not None and tier_cfg.primary:
        return tier_cfg.primary
    return get_dotenv_value("OVERSEER_CHEAP_DEFAULT_MODEL") or DEFAULT_CHEAP_MODEL_SLUG


def get_cheap_fallback_models() -> list[str]:
    return _resolve_fallback_models("cheap", "OVERSEER_CHEAP_FALLBACK_MODELS", DEFAULT_CHEAP_FALLBACK_SLUGS)


def get_heavy_default_model() -> str | None:
    env_val = get_env_only_value("OVERSEER_HEAVY_DEFAULT_MODEL")
    if env_val:
        return env_val
    tier_cfg = load_config_file_sync().tiers.get("heavy")
    if tier_cfg is not None and tier_cfg.primary:
        return tier_cfg.primary
    dotenv_val = get_dotenv_value("OVERSEER_HEAVY_DEFAULT_MODEL")
    return dotenv_val or None


def get_heavy_fallback_models() -> list[str]:
    return _resolve_fallback_models("heavy", "OVERSEER_HEAVY_FALLBACK_MODELS", ())


def infer_tier_for_request(requested_model: str | None) -> TierName:
    """Resolve which Overseer tier ('free', 'cheap', 'heavy') governs a worker model request."""
    req = (requested_model or "").strip()
    low = req.lower()
    if not req or low == "free":
        return "free"
    if low == "cheap":
        return "cheap"
    if low == "heavy":
        return "heavy"
    normalized = MODEL_SLUG_ALIASES.get(req, req)
    if normalized.endswith(":free"):
        return "free"
    heavy_primary = get_heavy_default_model()
    heavy_slugs = {
        MODEL_SLUG_ALIASES.get(m, m)
        for m in ([heavy_primary] if heavy_primary else []) + get_heavy_fallback_models()
    }
    if normalized in heavy_slugs:
        return "heavy"
    return "cheap"


def build_model_chain(requested_model: str | None) -> list[str]:
    req = (requested_model or "").strip()
    if not req or req.lower() == "free":
        raw_primary = get_default_model()
        fallbacks = get_fallback_models()
    elif req.lower() == "cheap":
        raw_primary = get_cheap_default_model()
        fallbacks = get_cheap_fallback_models()
    elif req.lower() == "heavy":
        heavy_primary = get_heavy_default_model()
        if not heavy_primary:
            raise ValueError(
                "Heavy tier is not configured. Set a heavy primary model via update_config or the Overseer Monitor UI."
            )
        raw_primary = heavy_primary
        fallbacks = get_heavy_fallback_models()
    else:
        raw_primary = req
        tier = infer_tier_for_request(req)
        if tier == "free":
            fallbacks = get_fallback_models()
        elif tier == "heavy":
            fallbacks = get_heavy_fallback_models()
        else:
            fallbacks = get_cheap_fallback_models()

    primary = MODEL_SLUG_ALIASES.get(raw_primary, raw_primary)
    chain = [primary]
    for raw_fb in fallbacks:
        fb = MODEL_SLUG_ALIASES.get(raw_fb, raw_fb)
        if fb not in chain:
            chain.append(fb)
    return chain


def _usable_key(var_name: str) -> str:
    val = get_config_value(var_name)
    return val if (val and not val.startswith("PASTE_")) else ""


def get_configured_tier_keys() -> dict[TierName, str]:
    """Return effective configured API key per tier (empty string when unconfigured)."""
    raw_shared = _usable_key("OPENROUTER_API_KEY")
    cheap_key = _usable_key("OPENROUTER_CHEAP_API_KEY") or raw_shared
    return {
        "free": _usable_key("OPENROUTER_FREE_API_KEY") or raw_shared,
        "cheap": cheap_key,
        "heavy": _usable_key("OPENROUTER_HEAVY_API_KEY") or cheap_key,
    }


def select_api_key_for_model(model_id: str, tier: TierName | None = None) -> str:
    """Select tier-isolated API key (:free -> FREE, heavy -> HEAVY->CHEAP, cheap -> CHEAP)."""
    is_free = model_id.strip().endswith(":free")
    if tier == "free" and not is_free:
        raise ValueError(
            f"Free tier model {model_id!r} must end with ':free' (refusing to use paid API key)."
        )
    raw_shared = _usable_key("OPENROUTER_API_KEY")
    env_path = get_data_dir() / ".env"

    if is_free:
        key = _usable_key("OPENROUTER_FREE_API_KEY") or raw_shared
        if not key:
            raise ValueError(
                f"OPENROUTER_FREE_API_KEY (or OPENROUTER_API_KEY) is required for free models; set it in {env_path}."
            )
        return key

    if tier == "heavy":
        key = (
            _usable_key("OPENROUTER_HEAVY_API_KEY")
            or _usable_key("OPENROUTER_CHEAP_API_KEY")
            or raw_shared
        )
        if not key:
            raise ValueError(
                "OPENROUTER_HEAVY_API_KEY (or OPENROUTER_CHEAP_API_KEY / OPENROUTER_API_KEY) "
                f"is required for heavy models; set it in {env_path}."
            )
        return key

    key = _usable_key("OPENROUTER_CHEAP_API_KEY") or raw_shared
    if not key:
        raise ValueError(
            f"OPENROUTER_CHEAP_API_KEY (or OPENROUTER_API_KEY) is required for cheap/paid models; set it in {env_path}."
        )
    return key


def get_tier_max_run_usd(tier: TierName) -> float:
    """Return configured per-run USD cap for a tier (defaults: $0.25 cheap/free, $2.00 heavy)."""
    if tier == "free":
        return DEFAULT_CHEAP_MAX_RUN_USD
    tier_cfg = load_config_file_sync().tiers.get(tier)
    if tier_cfg is not None and tier_cfg.max_run_usd is not None:
        return float(tier_cfg.max_run_usd)
    return DEFAULT_HEAVY_MAX_RUN_USD if tier == "heavy" else DEFAULT_CHEAP_MAX_RUN_USD


def resolve_effective_max_run_usd(
    requested_models: list[str | None],
    explicit_max_run_usd: float | None,
) -> float:
    """Use explicit start_run max_run_usd if provided, else the highest non-free tier cap used."""
    if explicit_max_run_usd is not None:
        return float(explicit_max_run_usd)
    if not requested_models:
        return DEFAULT_CHEAP_MAX_RUN_USD
    non_free_caps = [
        get_tier_max_run_usd(tier)
        for m in requested_models
        if (tier := infer_tier_for_request(m)) != "free"
    ]
    if not non_free_caps:
        return DEFAULT_CHEAP_MAX_RUN_USD
    return max(non_free_caps)


def _clamp_workers(value: int) -> int:
    return max(1, min(10, value))


def get_max_concurrent_workers() -> int:
    """Return max concurrent workers (os.environ > config.json > .env > default 3), clamped 1..10."""
    env_val = get_env_only_value("OVERSEER_MAX_CONCURRENT_WORKERS")
    if env_val:
        try:
            return _clamp_workers(int(env_val))
        except ValueError:
            _logger.warning("Ignoring invalid OVERSEER_MAX_CONCURRENT_WORKERS=%r", env_val)

    cfg = load_config_file_sync()
    if cfg.max_concurrent_workers is not None:
        return _clamp_workers(int(cfg.max_concurrent_workers))

    dotenv_val = get_dotenv_value("OVERSEER_MAX_CONCURRENT_WORKERS")
    if dotenv_val:
        try:
            return _clamp_workers(int(dotenv_val))
        except ValueError:
            _logger.warning("Ignoring invalid .env OVERSEER_MAX_CONCURRENT_WORKERS=%r", dotenv_val)

    return DEFAULT_MAX_CONCURRENT_WORKERS


def get_provider_routing_for_model(model_id: str, tier: TierName | None = None) -> dict[str, Any] | None:
    """Return OpenRouter provider routing dict (exact model override wins over tier default)."""
    cfg = load_config_file_sync()
    raw_id = model_id.strip()
    norm_id = MODEL_SLUG_ALIASES.get(raw_id, raw_id)
    rule = cfg.provider_models.get(norm_id) or cfg.provider_models.get(raw_id)
    if rule is not None:
        rendered = rule.to_openrouter_dict()
        if rendered is not None:
            return rendered
    if tier is not None:
        tier_rule = cfg.provider_tiers.get(tier)
        if tier_rule is not None:
            return tier_rule.to_openrouter_dict()
    return None


def _collect_env_overrides() -> dict[str, str]:
    """Return mapping of config fields currently overridden by process environment variables."""
    overrides: dict[str, str] = {}
    if get_env_only_value("OVERSEER_FREE_DEFAULT_MODEL"):
        overrides["tiers.free.primary"] = "OVERSEER_FREE_DEFAULT_MODEL"
    elif get_env_only_value("OVERSEER_DEFAULT_MODEL"):
        overrides["tiers.free.primary"] = "OVERSEER_DEFAULT_MODEL"

    if "OVERSEER_FREE_FALLBACK_MODELS" in os.environ:
        overrides["tiers.free.fallbacks"] = "OVERSEER_FREE_FALLBACK_MODELS"
    elif "OVERSEER_FALLBACK_MODELS" in os.environ:
        overrides["tiers.free.fallbacks"] = "OVERSEER_FALLBACK_MODELS"

    if get_env_only_value("OVERSEER_CHEAP_DEFAULT_MODEL"):
        overrides["tiers.cheap.primary"] = "OVERSEER_CHEAP_DEFAULT_MODEL"
    if "OVERSEER_CHEAP_FALLBACK_MODELS" in os.environ:
        overrides["tiers.cheap.fallbacks"] = "OVERSEER_CHEAP_FALLBACK_MODELS"
    elif "OVERSEER_FALLBACK_MODELS" in os.environ:
        overrides["tiers.cheap.fallbacks"] = "OVERSEER_FALLBACK_MODELS"

    if get_env_only_value("OVERSEER_HEAVY_DEFAULT_MODEL"):
        overrides["tiers.heavy.primary"] = "OVERSEER_HEAVY_DEFAULT_MODEL"
    if "OVERSEER_HEAVY_FALLBACK_MODELS" in os.environ:
        overrides["tiers.heavy.fallbacks"] = "OVERSEER_HEAVY_FALLBACK_MODELS"

    if get_env_only_value("OVERSEER_MAX_CONCURRENT_WORKERS"):
        overrides["max_concurrent_workers"] = "OVERSEER_MAX_CONCURRENT_WORKERS"
    if get_env_only_value("OPENROUTER_BASE_URL"):
        overrides["openrouter_base_url"] = "OPENROUTER_BASE_URL"
    if get_env_only_value("OVERSEER_DISABLE_TOASTS"):
        overrides["disable_toasts"] = "OVERSEER_DISABLE_TOASTS"
    return overrides


def get_effective_config_snapshot() -> dict[str, Any]:
    """Return effective non-secret config state (tiers, limits, provider routing, key presence)."""
    cfg = load_config_file_sync()
    provider_tiers: dict[str, Any] = {}
    for tier in VALID_TIERS:
        rule = cfg.provider_tiers.get(tier)
        if rule is not None and rule.to_openrouter_dict() is not None:
            provider_tiers[tier] = rule.model_dump()
    provider_models: dict[str, Any] = {}
    for slug, rule in cfg.provider_models.items():
        if rule.to_openrouter_dict() is not None:
            provider_models[slug] = rule.model_dump()

    return {
        "config_path": str(config_file_path()),
        "max_concurrent_workers": get_max_concurrent_workers(),
        "env_overrides": _collect_env_overrides(),
        "tiers": {
            "free": {
                "primary": get_default_model(),
                "fallbacks": get_fallback_models(),
                "max_run_usd": None,
            },
            "cheap": {
                "primary": get_cheap_default_model(),
                "fallbacks": get_cheap_fallback_models(),
                "max_run_usd": get_tier_max_run_usd("cheap"),
            },
            "heavy": {
                "primary": get_heavy_default_model(),
                "fallbacks": get_heavy_fallback_models(),
                "max_run_usd": get_tier_max_run_usd("heavy"),
            },
        },
        "provider_routing": {
            "tiers": provider_tiers,
            "models": provider_models,
        },
        "api_keys_configured": {
            "free": bool(_usable_key("OPENROUTER_FREE_API_KEY")),
            "cheap": bool(_usable_key("OPENROUTER_CHEAP_API_KEY")),
            "heavy": bool(_usable_key("OPENROUTER_HEAVY_API_KEY")),
            "shared": bool(_usable_key("OPENROUTER_API_KEY")),
        },
    }


def custom_instructions_path() -> Path:
    """Return external path for optional user-saved custom instructions."""
    return get_data_dir() / "custom_instructions.txt"


def load_custom_instructions_sync() -> str:
    """Load saved user custom instructions from <data_dir>/custom_instructions.txt, or '' if absent."""
    path = custom_instructions_path()
    if not path.is_file():
        return ""
    try:
        return read_text_retry_sync(path, encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def save_custom_instructions_sync(instructions: str) -> str:
    """Persist or clear optional user custom instructions atomically under <data_dir>."""
    if not isinstance(instructions, str):
        raise ValueError("instructions must be a string.")
    cleaned = instructions.replace("\x00", "").replace("\r\n", "\n").strip()
    if len(cleaned) > MAX_CUSTOM_INSTRUCTIONS_CHARS:
        raise ValueError(
            f"Custom instructions exceed {MAX_CUSTOM_INSTRUCTIONS_CHARS} characters ({len(cleaned)})."
        )
    path = custom_instructions_path()
    with data_dir_file_lock("custom_instructions"):
        if not cleaned:
            if path.exists():
                try:
                    unlink_retry_sync(path)
                except FileNotFoundError:
                    pass
            return ""
        atomic_write_bytes_sync(path, cleaned.encode("utf-8"))
    return cleaned
