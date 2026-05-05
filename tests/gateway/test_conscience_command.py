from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner._running_agents = {}
    return runner


@pytest.mark.asyncio
async def test_conscience_command_reports_latest_review(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-conscience",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)

    conscience_dir = tmp_path / "conscience" / session_entry.session_id
    conscience_dir.mkdir(parents=True)
    (conscience_dir / "last-review.json").write_text(
        '{"review_type":"midtask","verdict":{"should_intervene":true},"open_criteria":[{"source_text":"verify the fix"}]}'
    )
    (conscience_dir / "last-review-payload.json").write_text('{"review_type":"midtask","draft_answer":""}')
    (conscience_dir / "critique-tickets.json").write_text(
        '[{"verdict":"repair","reason":"course_correction_needed","next_best_action":"Inspect the failure.","evidence":["looping"]}]'
    )
    (conscience_dir / "stop-audit.json").write_text('{}')
    (conscience_dir / "task-contract.json").write_text('{"explicit_asks":[{"criterion_id":"criterion_001","source_text":"verify the fix"}]}')
    (conscience_dir / "completion-ledger.json").write_text('{"criterion_001":{"status":"open"}}')

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    result = await runner._handle_conscience_command(_make_event("/conscience"))

    assert "Conscience status" in result
    assert "review type: midtask" in result
    assert "course_correction_needed" in result
    assert "verify the fix" in result
    assert '"review_type": "midtask"' in result
