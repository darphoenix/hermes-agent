from unittest.mock import patch

from run_agent import AIAgent


def _agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._delegate_depth = 0
    agent._trace_turn_id = "turn_parent"
    agent._trace_task_id = "task_parent"
    agent._current_task_id = "task_parent"
    agent._foreground_cache_lease_id = "turn_parent"
    agent._foreground_cache_lease_emitted = False
    agent.session_id = "session_parent"
    agent.base_url = "http://127.0.0.1:1236/v1"
    agent.api_mode = "codex_responses"
    return agent


def test_local_main_requests_share_one_turn_scoped_cache_lease():
    agent = _agent()

    first = agent._trace_apply_headers(
        {"extra_headers": {"X-Custom": "kept"}},
        request_id="request_one",
        actor="main",
    )
    second = agent._trace_apply_headers(
        {}, request_id="request_two", actor="main"
    )

    assert first["extra_headers"]["X-Hermes-Cache-Lease"] == "turn_parent"
    assert second["extra_headers"]["X-Hermes-Cache-Lease"] == "turn_parent"
    assert first["extra_headers"]["X-Custom"] == "kept"
    assert agent._foreground_cache_lease_emitted is True


def test_non_main_and_nonlocal_requests_do_not_claim_cache_lease():
    agent = _agent()
    child = agent._trace_apply_headers(
        {},
        request_id="request_child",
        actor="delegate",
    )
    assert "X-Hermes-Cache-Lease" not in child["extra_headers"]
    assert agent._foreground_cache_lease_emitted is False

    agent.base_url = "https://api.example.com/v1"
    remote = agent._trace_apply_headers(
        {}, request_id="request_remote", actor="main"
    )
    assert "X-Hermes-Cache-Lease" not in remote["extra_headers"]
    assert agent._foreground_cache_lease_emitted is False

    agent.base_url = "http://127.0.0.1:1236/v1"
    agent.api_mode = "chat_completions"
    chat = agent._trace_apply_headers(
        {}, request_id="request_chat", actor="main"
    )
    assert "X-Hermes-Cache-Lease" not in chat["extra_headers"]
    assert agent._foreground_cache_lease_emitted is False


def test_turn_finalizer_release_uses_the_same_lease():
    agent = _agent()
    agent._foreground_cache_lease_emitted = True

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def __init__(self):
            self.request = None

        def open(self, request, timeout):
            self.request = request
            assert timeout == 0.5
            return Response()

    opener = Opener()
    with patch("urllib.request.build_opener", return_value=opener):
        assert agent._release_foreground_cache_lease() is True

    assert opener.request.full_url == (
        "http://127.0.0.1:1236/v1/cache/lease/release"
    )
    assert opener.request.headers["X-hermes-cache-lease"] == "turn_parent"
    assert opener.request.headers["X-hermes-actor"] == "main"


def test_unemitted_lease_does_not_probe_the_endpoint():
    agent = _agent()
    with patch("urllib.request.build_opener") as build_opener:
        assert agent._release_foreground_cache_lease() is False
    build_opener.assert_not_called()
