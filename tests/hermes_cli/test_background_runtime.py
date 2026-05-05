from __future__ import annotations

import pytest

from hermes_cli.background_runtime import (
    BackgroundRuntimeError,
    background_runtime_enabled_for,
    resolve_background_runtime,
)


def _config(**overrides):
    cfg = {
        "background_runtime": {
            "enabled": True,
            "provider": "custom",
            "model": "/models/qwen",
            "base_url": "http://127.0.0.1:1237/v1",
            "api_key": "side-key",
            "api_mode": "codex_responses",
            "responses_stateful": False,
            "use_for": {
                "background_review": True,
                "gateway_background": True,
                "auxiliary": False,
            },
        }
    }
    cfg["background_runtime"].update(overrides)
    return cfg


def test_background_runtime_routes_enabled_task_to_sidecar():
    model, runtime = resolve_background_runtime(
        "gateway_background",
        parent_model="/models/main",
        parent_runtime={
            "provider": "custom",
            "base_url": "http://127.0.0.1:1236/v1",
            "api_key": "main-key",
            "api_mode": "codex_responses",
            "responses_stateful": True,
        },
        config=_config(),
    )

    assert model == "/models/qwen"
    assert runtime["base_url"] == "http://127.0.0.1:1237/v1"
    assert runtime["api_key"] == "side-key"
    assert runtime["responses_stateful"] is False


def test_background_runtime_force_stateless_disables_stateful_resume():
    model, runtime = resolve_background_runtime(
        "background_review",
        parent_model="/models/main",
        parent_runtime={"responses_stateful": True},
        config=_config(responses_stateful=True),
        force_stateless=True,
    )

    assert model == "/models/qwen"
    assert runtime["responses_stateful"] is False


def test_background_runtime_allows_task_specific_stateful_resume():
    cfg = _config(
        responses_stateful=False,
        responses_stateful_for={"background_review": True},
    )

    model, runtime = resolve_background_runtime(
        "background_review",
        parent_model="/models/main",
        parent_runtime={"responses_stateful": False},
        config=cfg,
    )
    _gateway_model, gateway_runtime = resolve_background_runtime(
        "gateway_background",
        parent_model="/models/main",
        parent_runtime={"responses_stateful": False},
        config=cfg,
    )

    assert model == "/models/qwen"
    assert runtime["responses_stateful"] is True
    assert gateway_runtime["responses_stateful"] is False


def test_background_runtime_keeps_auxiliary_disabled_by_default():
    assert not background_runtime_enabled_for("auxiliary:compression", _config())
    assert (
        resolve_background_runtime(
            "auxiliary:compression",
            parent_model="/models/main",
            config=_config(),
        )
        is None
    )


def test_background_runtime_routes_exact_auxiliary_task_when_enabled():
    cfg = _config(
        use_for={
            "background_review": True,
            "gateway_background": True,
            "auxiliary": False,
            "auxiliary:compression": True,
        }
    )

    assert background_runtime_enabled_for("auxiliary:compression", cfg)
    assert not background_runtime_enabled_for("auxiliary:title_generation", cfg)

    model, runtime = resolve_background_runtime(
        "auxiliary:compression",
        parent_model="/models/main",
        parent_runtime={"responses_stateful": True},
        config=cfg,
    )

    assert model == "/models/qwen"
    assert runtime["base_url"] == "http://127.0.0.1:1237/v1"
    assert runtime["responses_stateful"] is False


def test_background_runtime_fails_closed_when_enabled_but_incomplete():
    with pytest.raises(BackgroundRuntimeError):
        resolve_background_runtime(
            "gateway_background",
            parent_model="/models/main",
            config=_config(base_url=""),
        )
