"""Regression tests for background review agent cleanup."""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.responses_stateful = False
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def test_background_review_shuts_down_memory_provider_before_close(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert [name for name, _payload in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]


def test_background_review_installs_auto_deny_approval_callback(monkeypatch):
    """Regression guard for #15216.

    The background review thread must install a non-interactive approval
    callback. If it doesn't, any dangerous-command guard the review agent
    trips falls back to input() on a daemon thread, which deadlocks against
    the parent's prompt_toolkit TUI.
    """
    import tools.terminal_tool as tt

    observed: dict = {"during_run": "<unread>", "after_finally": "<unread>"}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            # Capture what the callback looks like mid-run. It must be
            # a callable (the auto-deny) -- not None.
            observed["during_run"] = tt._get_approval_callback()

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    # Start from a clean slot.
    tt.set_approval_callback(None)
    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    observed["after_finally"] = tt._get_approval_callback()

    assert callable(observed["during_run"]), (
        "Background review did not install an approval callback on its "
        "worker thread; dangerous-command prompts will deadlock against "
        "the parent TUI (#15216)."
    )
    # The installed callback must deny (it's a safety gate, not a prompt).
    assert observed["during_run"]("rm -rf /", "test") == "deny"

    assert observed["after_finally"] is None, (
        "Background review leaked its approval callback into the worker "
        "thread's TLS slot; a recycled thread-id could reuse it."
    )


def test_background_review_uses_configured_sidecar_runtime(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    import hermes_cli.background_runtime as background_runtime

    monkeypatch.setattr(
        background_runtime,
        "get_background_runtime_config",
        lambda config=None: {
            "enabled": True,
            "provider": "custom",
            "model": "sidecar-model",
            "base_url": "http://127.0.0.1:1237/v1",
            "api_key": "side-key",
            "api_mode": "codex_responses",
            "responses_stateful": False,
            "responses_stateful_for": {"background_review": True},
            "use_for": {"background_review": True},
        },
    )
    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.provider = "custom"
    agent.base_url = "http://127.0.0.1:1236/v1"
    agent.api_key = "main-key"
    agent.api_mode = "codex_responses"
    agent.responses_stateful = True

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    init_kwargs = events[0][1]
    assert init_kwargs["model"] == "sidecar-model"
    assert init_kwargs["base_url"] == "http://127.0.0.1:1237/v1"
    assert init_kwargs["api_key"] == "side-key"
    assert init_kwargs["responses_stateful"] is True


def test_background_review_scrubs_foreground_responses_state(monkeypatch):
    events = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self._session_messages = []

        def run_conversation(self, **kwargs):
            events.append(("run_conversation", kwargs))

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    import hermes_cli.background_runtime as background_runtime

    monkeypatch.setattr(
        background_runtime,
        "get_background_runtime_config",
        lambda config=None: {
            "enabled": True,
            "provider": "custom",
            "model": "sidecar-model",
            "base_url": "http://127.0.0.1:1237/v1",
            "api_key": "side-key",
            "api_mode": "codex_responses",
            "responses_stateful_for": {"background_review": True},
            "use_for": {"background_review": True},
        },
    )
    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = _bare_agent()
    agent.provider = "custom"
    agent.base_url = "http://127.0.0.1:1236/v1"
    agent.api_key = "main-key"
    agent.api_mode = "codex_responses"
    agent.responses_stateful = True
    inherited = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "hi",
            "responses_response_id": "resp_main",
            "codex_message_items": [{"id": "msg_main", "type": "message"}],
            "codex_reasoning_items": [{"id": "rs_main", "encrypted_content": "..."}],
        },
    ]

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=inherited,
        review_memory=True,
    )

    run_kwargs = [payload for name, payload in events if name == "run_conversation"][0]
    assistant_history = run_kwargs["conversation_history"][1]
    assert "responses_response_id" not in assistant_history
    assert "codex_message_items" not in assistant_history
    assert "codex_reasoning_items" not in assistant_history
    assert inherited[1]["responses_response_id"] == "resp_main"
