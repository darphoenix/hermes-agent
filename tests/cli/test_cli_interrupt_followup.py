"""Regression tests for CLI busy-mode interrupt handoff."""

from __future__ import annotations

import queue


def _make_cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    return cli


def test_interrupt_payload_is_pending_even_without_agent_result():
    from cli import HermesCLI

    assert HermesCLI._turn_was_interrupted(None, "follow-up") is True
    assert HermesCLI._pending_interrupt_message(None, "follow-up") == "follow-up"


def test_queue_interrupt_followup_drains_later_messages():
    cli = _make_cli()
    cli._interrupt_queue.put("second")
    cli._interrupt_queue.put("third")

    assert cli._queue_interrupt_followup("first") is True

    assert cli._pending_input.get_nowait() == "first\nsecond\nthird"
    assert cli._interrupt_queue.empty()


def test_queue_interrupt_followup_preserves_image_payloads():
    cli = _make_cli()
    cli._interrupt_queue.put(("second", ["b.png"]))

    assert cli._queue_interrupt_followup(("first", ["a.png"])) is True

    text, images = cli._pending_input.get_nowait()
    assert text == "first\nsecond"
    assert images == ["a.png", "b.png"]
