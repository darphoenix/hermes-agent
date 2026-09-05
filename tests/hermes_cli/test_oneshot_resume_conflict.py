"""One-shot mode must never silently ignore session continuation flags."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.parametrize(
    "session_args",
    [
        ["--resume", "session_123"],
        ["-r", "session_123"],
        ["--continue"],
        ["-c", "named-session"],
    ],
)
def test_fast_launch_rejects_oneshot_session_continuation(
    monkeypatch, capsys, session_args
):
    import hermes_cli.main as main_mod

    monkeypatch.delenv("HERMES_DISABLE_FAST_CHAT_LAUNCH", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "continue work", *session_args])
    monkeypatch.setattr(
        main_mod,
        "_prepare_agent_startup",
        lambda _args: pytest.fail("validation must run before agent startup"),
    )
    monkeypatch.setattr(
        main_mod,
        "_run_and_exit_oneshot",
        lambda *_args, **_kwargs: pytest.fail("one-shot must not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod._try_fast_chat_launch()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "always starts a new session" in error
    assert "hermes chat -Q --resume SESSION_ID -q PROMPT" in error


def test_resumed_chat_query_remains_valid():
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.main import _validate_oneshot_session_flags

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args(
        ["chat", "-Q", "--resume", "session_123", "-q", "continue work"]
    )

    _validate_oneshot_session_flags(args, parser)
    assert args.oneshot is None
    assert args.resume == "session_123"
    assert args.query == "continue work"


def test_termux_fast_launch_rejects_oneshot_resume(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "continue work", "-r", "session_123"])
    monkeypatch.setattr(main_mod, "_is_termux_startup_environment", lambda: True)
    monkeypatch.setattr(main_mod, "_wants_tui_early", lambda _argv: False)
    monkeypatch.setattr(main_mod, "_is_termux_fast_version_argv", lambda _argv: False)
    monkeypatch.setattr(
        main_mod,
        "_prepare_agent_startup",
        lambda _args: pytest.fail("validation must run before agent startup"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod._try_termux_fast_cli_launch()

    assert exc_info.value.code == 2
    assert "always starts a new session" in capsys.readouterr().err


def test_full_dispatch_rejects_oneshot_resume(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "continue work", "-r", "session_123"])
    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_fast_chat_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_set_process_title", lambda: None)
    monkeypatch.setattr(main_mod, "_advertise_agent_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sweep_stale_bytecode_if_checkout_changed", lambda: None)
    monkeypatch.setattr(main_mod, "_recover_from_interrupted_install", lambda: None)
    monkeypatch.setattr(main_mod, "_warn_pending_fleet_restart_on_startup", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "_prepare_agent_startup",
        lambda _args: pytest.fail("validation must run before agent startup"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2
    assert "always starts a new session" in capsys.readouterr().err
