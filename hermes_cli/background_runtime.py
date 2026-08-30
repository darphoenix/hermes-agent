"""Routing and foreground admission for an optional local sidecar runtime."""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any, Optional
from urllib.parse import urlparse


DEFAULT_USE_FOR = {
    "background_review": True,
    "cli_background": True,
    "gateway_background": True,
    "gateway_hygiene": True,
    "auxiliary": False,
}


class BackgroundRuntimeError(RuntimeError):
    """Raised when an enabled sidecar route is unusable."""


class BackgroundRuntimeDeferred(BackgroundRuntimeError):
    """Raised when optional sidecar work must wait for the foreground turn."""


_FOREGROUND_ACTIVITY_LOCK = threading.Lock()
_FOREGROUND_ACTIVITY: dict[str, dict[str, Any]] = {}

_DEFAULT_FOREGROUND_DEFER_TASKS = frozenset(
    {
        "background_review",
        "cli_background",
        "gateway_background",
        "gateway_hygiene",
        "auxiliary:curator",
        "auxiliary:title_generation",
    }
)
_DEFAULT_FOREGROUND_ALLOW_TASKS = frozenset(
    {
        "auxiliary:approval",
        "auxiliary:compression",
        "auxiliary:mcp",
        "auxiliary:session_search",
        "auxiliary:skills_hub",
        "auxiliary:vision",
        "auxiliary:web_extract",
        "conscience",
        "delegation",
    }
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def begin_foreground_activity(
    *, session_id: str = "", platform: str = "", source: str = "main_agent"
) -> str:
    token = f"fg_{uuid.uuid4().hex[:12]}"
    with _FOREGROUND_ACTIVITY_LOCK:
        _FOREGROUND_ACTIVITY[token] = {
            "session_id": session_id,
            "platform": platform,
            "source": source,
            "started_at": time.monotonic(),
        }
    return token


def end_foreground_activity(token: Optional[str]) -> None:
    if token:
        with _FOREGROUND_ACTIVITY_LOCK:
            _FOREGROUND_ACTIVITY.pop(str(token), None)


def foreground_activity_snapshot() -> dict[str, dict[str, Any]]:
    with _FOREGROUND_ACTIVITY_LOCK:
        return {key: dict(value) for key, value in _FOREGROUND_ACTIVITY.items()}


def is_foreground_active() -> bool:
    return bool(foreground_activity_snapshot())


def _coerce_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if config is not None:
        return config if isinstance(config, Mapping) else {}
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly()
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def get_background_runtime_config(
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    root = _coerce_config(config)
    for key in ("background_runtime", "sidecar_runtime"):
        value = root.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalise_name_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {part.strip() for part in raw.split(",") if part.strip()}
    if isinstance(raw, Mapping):
        return {
            str(key).strip()
            for key, value in raw.items()
            if str(key).strip() and _as_bool(value)
        }
    if isinstance(raw, (list, tuple, set)):
        return {str(value).strip() for value in raw if str(value).strip()}
    return set()


def _normalise_use_for(raw: Any) -> dict[str, bool]:
    result = dict(DEFAULT_USE_FOR)
    if isinstance(raw, Mapping):
        result.update(
            {
                str(key).strip(): _as_bool(value)
                for key, value in raw.items()
                if str(key).strip()
            }
        )
    elif raw is not None:
        result.update({name: True for name in _normalise_name_set(raw)})
    return result


def _task_enabled(task: str, cfg: Mapping[str, Any]) -> bool:
    use_for = _normalise_use_for(cfg.get("use_for"))
    if task.startswith("auxiliary:"):
        return _as_bool(use_for.get(task), _as_bool(use_for.get("auxiliary")))
    return _as_bool(use_for.get(task))


def _task_deferred_while_foreground(task: str, cfg: Mapping[str, Any]) -> bool:
    gate_cfg = cfg.get("foreground_gate")
    enabled = _as_bool(cfg.get("pause_while_foreground"), True)
    if isinstance(gate_cfg, Mapping):
        enabled = _as_bool(gate_cfg.get("enabled"), enabled)
    if not enabled or not is_foreground_active():
        return False

    allow = set(_DEFAULT_FOREGROUND_ALLOW_TASKS)
    defer = set(_DEFAULT_FOREGROUND_DEFER_TASKS)
    allow.update(_normalise_name_set(cfg.get("allow_while_foreground")))
    defer.update(_normalise_name_set(cfg.get("defer_while_foreground")))
    if isinstance(gate_cfg, Mapping):
        allow.update(_normalise_name_set(gate_cfg.get("allow")))
        defer.update(_normalise_name_set(gate_cfg.get("defer")))
    if task in allow or (task.startswith("auxiliary:") and "auxiliary" in allow):
        return False
    if task in defer or (task.startswith("auxiliary:") and "auxiliary" in defer):
        return True
    return True


def _task_bool_override(raw: Any, task: str) -> Optional[bool]:
    if isinstance(raw, Mapping):
        if task in raw:
            return _as_bool(raw.get(task))
        if task.startswith("auxiliary:") and "auxiliary" in raw:
            return _as_bool(raw.get("auxiliary"))
        return None
    if raw is None:
        return None
    enabled = _normalise_name_set(raw)
    return task in enabled or (task.startswith("auxiliary:") and "auxiliary" in enabled)


def background_runtime_enabled_for(
    task: str, config: Optional[Mapping[str, Any]] = None
) -> bool:
    cfg = get_background_runtime_config(config)
    return _as_bool(cfg.get("enabled")) and _task_enabled(task, cfg)


def _env_or_cfg(env_name: str, cfg: Mapping[str, Any], key: str) -> Optional[str]:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    value = cfg.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_local_base_url(base_url: str) -> bool:
    try:
        return (urlparse(base_url).hostname or "").lower() in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
    except Exception:
        return False


def resolve_background_runtime(
    task: str,
    *,
    parent_model: Optional[str] = None,
    parent_runtime: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    force_stateless: bool = False,
) -> Optional[tuple[str, dict[str, Any]]]:
    cfg = get_background_runtime_config(config)
    if not _as_bool(cfg.get("enabled")) or not _task_enabled(task, cfg):
        return None
    if _task_deferred_while_foreground(task, cfg):
        active = foreground_activity_snapshot()
        oldest = min(
            (float(item.get("started_at") or 0.0) for item in active.values()),
            default=time.monotonic(),
        )
        raise BackgroundRuntimeDeferred(
            f"background runtime task {task!r} deferred while foreground turn is "
            f"active (active={len(active)}, oldest_seconds="
            f"{max(0.0, time.monotonic() - oldest):.1f})"
        )

    parent = dict(parent_runtime or {})
    model = (
        _env_or_cfg("HERMES_BACKGROUND_MODEL", cfg, "model")
        or parent_model
        or str(parent.get("model") or "").strip()
    )
    if not model:
        raise BackgroundRuntimeError(
            f"background_runtime is enabled for {task!r}, but no model is configured"
        )

    base_url = _env_or_cfg("HERMES_BACKGROUND_BASE_URL", cfg, "base_url")
    api_key = _env_or_cfg("HERMES_BACKGROUND_API_KEY", cfg, "api_key") or str(
        parent.get("api_key") or ""
    ).strip()
    provider = _env_or_cfg("HERMES_BACKGROUND_PROVIDER", cfg, "provider") or (
        "custom" if base_url else str(parent.get("provider") or "").strip()
    )
    api_mode = _env_or_cfg("HERMES_BACKGROUND_API_MODE", cfg, "api_mode") or str(
        parent.get("api_mode") or ""
    ).strip()
    if base_url and not api_key and _is_local_base_url(base_url):
        api_key = "no-key-required"
    if provider == "custom" and not base_url:
        raise BackgroundRuntimeError(
            f"background_runtime is enabled for {task!r}, but no base_url is configured"
        )
    if base_url and not api_key:
        raise BackgroundRuntimeError(
            f"background_runtime is enabled for {task!r}, but no api_key is configured"
        )

    task_override = _task_bool_override(cfg.get("responses_stateful_for"), task)
    if force_stateless:
        responses_stateful = False
    elif task_override is not None:
        responses_stateful = task_override
    elif "responses_stateful" in cfg:
        responses_stateful = _as_bool(cfg.get("responses_stateful"))
    else:
        responses_stateful = _as_bool(parent.get("responses_stateful"))

    runtime: dict[str, Any] = {
        "api_key": api_key or None,
        "base_url": base_url or None,
        "provider": provider or None,
        "api_mode": api_mode or None,
        "responses_stateful": responses_stateful,
        "command": cfg.get("command") if isinstance(cfg.get("command"), str) else None,
        "args": list(cfg.get("args") or []),
        "credential_pool": None if base_url else parent.get("credential_pool"),
    }
    try:
        max_tokens = int(cfg.get("max_tokens", parent.get("max_tokens")) or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    if max_tokens > 0:
        runtime["max_tokens"] = max_tokens
    return model, runtime
