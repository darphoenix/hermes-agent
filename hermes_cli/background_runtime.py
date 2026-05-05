"""Runtime routing for optional sidecar/background model calls.

The foreground runtime should stay focused on interactive turns.  This helper
keeps sidecar selection in one place so background review, gateway helpers, and
CLI background tasks do not each reinvent the same config parsing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_USE_FOR: Dict[str, bool] = {
    "background_review": True,
    "cli_background": True,
    "gateway_background": True,
    "gateway_hygiene": True,
    "auxiliary": False,
}


class BackgroundRuntimeError(RuntimeError):
    """Raised when a sidecar runtime is enabled but unusable."""


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


def _coerce_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if config is not None:
        return config if isinstance(config, Mapping) else {}
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def get_background_runtime_config(
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the configured sidecar/background runtime block."""

    root = _coerce_config(config)
    for key in ("background_runtime", "sidecar_runtime"):
        value = root.get(key) if isinstance(root, Mapping) else None
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalise_use_for(raw: Any) -> Dict[str, bool]:
    result = dict(DEFAULT_USE_FOR)
    if raw is None:
        return result
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(key, str) and key.strip():
                result[key.strip()] = _as_bool(value, False)
        return result
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
        return {**result, **{name: True for name in names}}
    if isinstance(raw, (list, tuple, set)):
        return {**result, **{str(name).strip(): True for name in raw if str(name).strip()}}
    return result


def _task_enabled(task: str, cfg: Mapping[str, Any]) -> bool:
    use_for = _normalise_use_for(cfg.get("use_for"))
    if task.startswith("auxiliary:"):
        return _as_bool(use_for.get(task), _as_bool(use_for.get("auxiliary"), False))
    return _as_bool(use_for.get(task), False)


def _task_bool_override(raw: Any, task: str) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        if task in raw:
            return _as_bool(raw.get(task), False)
        if task.startswith("auxiliary:") and "auxiliary" in raw:
            return _as_bool(raw.get("auxiliary"), False)
        return None
    if isinstance(raw, str):
        enabled = {part.strip() for part in raw.split(",") if part.strip()}
        return task in enabled or (task.startswith("auxiliary:") and "auxiliary" in enabled)
    if isinstance(raw, (list, tuple, set)):
        enabled = {str(name).strip() for name in raw if str(name).strip()}
        return task in enabled or (task.startswith("auxiliary:") and "auxiliary" in enabled)
    return None


def background_runtime_enabled_for(
    task: str,
    config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether *task* should be routed to the sidecar runtime."""

    cfg = get_background_runtime_config(config)
    if not _as_bool(cfg.get("enabled"), False):
        return False
    return _task_enabled(task, cfg)


def _env_or_cfg(env_name: str, cfg: Mapping[str, Any], key: str) -> Optional[str]:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip()
    value = cfg.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _is_local_base_url(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def _normalise_runtime_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, str):
        return _as_bool(value, default)
    return _as_bool(value, default)


def resolve_background_runtime(
    task: str,
    *,
    parent_model: Optional[str] = None,
    parent_runtime: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    force_stateless: bool = False,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Resolve sidecar runtime for *task*.

    Returns ``None`` when the sidecar is disabled for the task.  If the sidecar
    is enabled but missing required fields, raises ``BackgroundRuntimeError`` so
    callers can fail the background task instead of silently using the main
    runtime.
    """

    cfg = get_background_runtime_config(config)
    if not _as_bool(cfg.get("enabled"), False):
        return None
    if not _task_enabled(task, cfg):
        return None

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
    api_key = (
        _env_or_cfg("HERMES_BACKGROUND_API_KEY", cfg, "api_key")
        or str(parent.get("api_key") or "").strip()
    )
    provider = (
        _env_or_cfg("HERMES_BACKGROUND_PROVIDER", cfg, "provider")
        or ("custom" if base_url else str(parent.get("provider") or "").strip())
        or None
    )
    api_mode = (
        _env_or_cfg("HERMES_BACKGROUND_API_MODE", cfg, "api_mode")
        or str(parent.get("api_mode") or "").strip()
        or None
    )

    if base_url and not api_key and _is_local_base_url(base_url):
        api_key = "no-key-required"

    if base_url and not provider:
        provider = "custom"
    if provider == "custom" and not base_url:
        raise BackgroundRuntimeError(
            f"background_runtime is enabled for {task!r}, but no base_url is configured"
        )
    if base_url and not api_key:
        raise BackgroundRuntimeError(
            f"background_runtime is enabled for {task!r}, but no api_key is configured"
        )

    if force_stateless:
        responses_stateful = False
    elif (task_override := _task_bool_override(cfg.get("responses_stateful_for"), task)) is not None:
        responses_stateful = task_override
    elif "responses_stateful" in cfg:
        responses_stateful = _normalise_runtime_bool(cfg.get("responses_stateful"), False)
    else:
        responses_stateful = _normalise_runtime_bool(parent.get("responses_stateful"), False)

    runtime: Dict[str, Any] = {
        "api_key": api_key or None,
        "base_url": base_url or None,
        "provider": provider,
        "api_mode": api_mode,
        "responses_stateful": responses_stateful,
        "command": cfg.get("command") if isinstance(cfg.get("command"), str) else None,
        "args": list(cfg.get("args") or []),
        "credential_pool": None if base_url else parent.get("credential_pool"),
    }
    max_tokens = cfg.get("max_tokens", parent.get("max_tokens"))
    if max_tokens is not None:
        try:
            max_tokens_int = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens_int = None
        if max_tokens_int and max_tokens_int > 0:
            runtime["max_tokens"] = max_tokens_int
    return model, runtime
