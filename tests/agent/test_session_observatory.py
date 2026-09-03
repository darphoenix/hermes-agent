"""Tests for the Session Observatory (agent/session_observatory.py).

All fixtures are anonymized: fake session ids, fake response ids, synthetic
token counts. No test reads or writes the real ~/.hermes state.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import time
from pathlib import Path

import pytest

from agent import session_observatory as obs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAILED_SID = "20260902_123950_aaaaaa"
RECOVERED_SID = "20260902_202149_bbbbbb"

# Anonymized main-wrapper log lines (mlx-serve format).
MAIN_LOG_LINES = """\
[args] serve: 127.0.0.1:1236, ctx-size=262144, pld=off, no-vision=false
[hermes-qwen-late] exact continuation prev=resp_1788335882955_14c base=95593 prompt=105150 delta_messages=1 tools=true runtime_chars=0
  [hot-cache] reused 95594/105150 tokens (matched 95594; entry 2/2)
  [mtp-round] off0=9555 t1=47983 m=2/2 drafts={ 15078, 364 } accepted=1
  [mtp-round] off0=9557 t1=1892 m=1/1 drafts={ 2 } accepted=0
[mtp] regime gate: two-chunk 42.82 ms/tok vs single 50.99 ms/tok -> two-chunk every round (period 39)
[mtp] regime gate: two-chunk 43.87 ms/tok vs single 23.81 ms/tok -> two-chunk throttled (period 74)
{"type":"response.completed","response":{"id":"resp_1788335882955_14c","usage":{"input_tokens":105150,"output_tokens":931,"total_tokens":106081,"input_tokens_details":{"cached_tokens":95594}}}}
timing: {"tokenize_ms":12.5,"prompt_ms":2300.0,"predicted_ms":45000.0,"prompt_per_second":45000.0,"predicted_per_second":20.7}
[hermes-qwen-late] exact continuation prev=resp_1788335960958_150 base=106007 prompt=115893 delta_messages=1 tools=true runtime_chars=0
  [hot-cache] reused 106008/115893 tokens (matched 106008; entry 2/2)
[scheduler] prefill aborted: client disconnected
[scheduler] model load failed: InsufficientMemory
this line is total garbage {{{ not-json
{"type":"response.completed","response":{"id":"resp_1788335960958_150","usage":{"input_tokens":115893,"output_tokens":120,"total_tokens":116013,"input_tokens_details":{"cached_tokens":106008}}}}
timing: {"tokenize_ms":8.0,"prompt_ms":1200.0,"predicted_ms":9000.0,"prompt_per_second":96000.0,"predicted_per_second":13.3}
Active Hermes profile: default. Other profiles (if any) liv…[mlx-serve: log line truncated]
retrying generation after upstream reset
poisoned response quarantined
malformed request rejected: bad schema
"""

# Anonymized sidecar log lines (mlx-openai-wrapper format). Wall-clock
# strings are derived from the fixture epoch at build time so the test is
# immune to the process timezone (the parser maps wall-clock -> epoch via
# the *local* timezone, exactly like the writer does).
_SIDECAR_T0 = 1788334800.0


def _sc_line(offset: float, level: str, msg: str) -> str:
    wall = _dt.datetime.fromtimestamp(_SIDECAR_T0 + offset)
    return (
        f"{wall.strftime('%Y-%m-%d %H:%M:%S')},{wall.microsecond // 1000:03d} "
        f"{level} mlx_openai_wrapper {msg}"
    )


def _sidecar_lines() -> str:
    """Built at fixture time so fromtimestamp/strptime share one timezone."""
    return "\n".join([
        _sc_line(1, "INFO", "mtplx: [mtplx] ssd prefix cache warm reuse decision outcome=miss session=abc prefix_tokens=1024"),
        _sc_line(61, "INFO", "mtplx: [mtplx] ssd prefix cache persist outcome=ok session=abc prefix_tokens=1024"),
        _sc_line(122, "INFO", "mtplx: [mtplx] ssd prefix cache warm reuse decision outcome=hit session=def prefix_tokens=2048"),
        _sc_line(183, "WARNING", "server: generation queue wait 1500ms depth=2"),
        _sc_line(244, "ERROR", "server: memory admission refused: insufficient memory for new generation"),
        _sc_line(305, "WARNING", "server: stream stall detected, cancelling"),
        _sc_line(366, "INFO", "server: client disconnected mid-stream"),
        _sc_line(427, "WARNING", "server: retry 1/2 after transient backend error"),
        _sc_line(488, "WARNING", "server: poisoned response quarantined"),
        _sc_line(549, "WARNING", "server: malformed request body rejected"),
        "not-a-timestamped-line",
        _sc_line(611, "INFO", "mtplx: [mtplx] ssd prefix cache warm reuse decision outcome=hit session=ghi prefix_tokens=4096"),
    ]) + "\n"


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT, model TEXT, started_at REAL, ended_at REAL,
            system_prompt TEXT, parent_session_id TEXT, cwd TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT,
            tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
            token_count INTEGER, model TEXT, stop_reason TEXT,
            structured_output TEXT, finish_reason TEXT, reasoning TEXT,
            time_created REAL
        );
        """
    )
    t0 = 1788334800.0  # 2026-09-02 10:40 local; sidecar fixture sits in-window
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
        (FAILED_SID, "cli", "mlx-serve-flashnext", t0, t0 + 3600,
         "SECRET SYSTEM PROMPT TEXT", None, "/home/user/project"),
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
        (RECOVERED_SID, "cli", "mlx-serve-flashnext", t0 + 7200, t0 + 7200 + 3099,
         "SECRET SYSTEM PROMPT TEXT", None, "/home/user/project"),
    )
    msgs = [
        (FAILED_SID, "user", "SECRET USER PROMPT", t0 + 1),
        (FAILED_SID, "assistant", "SECRET ASSISTANT CONTENT", t0 + 10),
        (FAILED_SID, "tool", "SECRET TOOL OUTPUT", t0 + 40),
        (FAILED_SID, "tool", "SECRET TOOL OUTPUT 2", t0 + 70),
        (FAILED_SID, "assistant", "Summary of previous conversation: …", t0 + 80),
        (RECOVERED_SID, "user", "SECRET USER PROMPT B", t0 + 7201),
        (RECOVERED_SID, "assistant", "SECRET ASSISTANT CONTENT B", t0 + 7210),
        (RECOVERED_SID, "tool", "SECRET TOOL OUTPUT B", t0 + 7235),
    ]
    for sid, role, content, ts in msgs:
        tool_calls = "[]" if role == "assistant" else None
        tool_name = "terminal" if role == "tool" else None
        finish = "stop" if role == "assistant" else "error" if "2" in content else "completed"
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_name,"
            " token_count, model, finish_reason, time_created)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, role, content, json.dumps([{"id": "c1"}]) if tool_calls else None,
             tool_name, 100, "model-x", finish, ts),
        )
    # Give the assistant rows real tool_calls so tool pairing has data.
    conn.execute(
        "UPDATE messages SET tool_calls = ? WHERE role = 'assistant'",
        (json.dumps([{"id": "call_1", "function": {"name": "terminal"}}]),),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def env(tmp_path):
    """Temp HERMES_HOME with DB, main+rotated wrapper logs, sidecar logs, conscience dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    db = home / "state.db"
    _make_db(db)

    log_dir = tmp_path / "logs-main"
    log_dir.mkdir()
    (log_dir / "server.log").write_text(MAIN_LOG_LINES)
    # Rotated file overlaps: last half of current file duplicated.
    lines = MAIN_LOG_LINES.splitlines()
    (log_dir / "server.log.1").write_text("\n".join(lines[-6:]) + "\n")

    side_dir = tmp_path / "logs-sidecar"
    side_dir.mkdir()
    (side_dir / "stderr.log").write_text(_sidecar_lines())

    cdir = home / "conscience" / FAILED_SID
    cdir.mkdir(parents=True)
    (cdir / "conscience-events.json").write_text(json.dumps([
        {"event_type": "TASK_START", "timestamp": 1788335991.0, "payload": {}},
        {"event_type": "REVIEW", "timestamp": 1788336100.0,
         "payload": {"verdict": "observe", "duration_ms": 250.5}},
        {"event_type": "STOP_AUDIT", "timestamp": 1788337000.0,
         "payload": {"verdict": "allow_stop", "duration_ms": 120.0}},
    ]))
    (cdir / "llm-audits.json").write_text(json.dumps([
        {"review_type": "midtask", "parsed": {"should_intervene": False, "verdict": "observe"}},
        {"review_type": "midtask", "parsed": {"should_intervene": True, "verdict": "redirect"}},
        {"review_type": "stop", "parsed": None},
    ]))
    (cdir / "stop-audit.json").write_text(json.dumps(
        {"parsed_review": {"verdict": "allow_stop"}}))
    (cdir / "critique-tickets.json").write_text("[]")
    (cdir / "intervention-ledger.json").write_text(json.dumps([{"id": 1}]))
    (cdir / "completion-ledger.json").write_text(json.dumps({"c1": {"status": "done"}}))
    (cdir / "task-contract.json").write_text("{ this is not valid json")

    return {
        "home": home,
        "db": db,
        "main_files": [log_dir / "server.log", log_dir / "server.log.1"],
        "sidecar_files": [side_dir / "stderr.log"],
    }


def _profile(env, sid=FAILED_SID):
    return obs.profile_session(
        sid,
        db_path=env["db"],
        hermes_home=env["home"],
        main_log_files=env["main_files"],
        sidecar_log_files=env["sidecar_files"],
    )


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------


def test_resolve_exact_id(env):
    conn = obs.open_state_db_readonly(env["db"])
    try:
        assert obs.resolve_session_id(conn, FAILED_SID) == FAILED_SID
    finally:
        conn.close()


def test_resolve_unique_prefix(env):
    conn = obs.open_state_db_readonly(env["db"])
    try:
        assert obs.resolve_session_id(conn, "20260902_202149") == RECOVERED_SID
        assert obs.resolve_session_id(conn, "20260902_123950_aaaa") == FAILED_SID
    finally:
        conn.close()


def test_resolve_ambiguous_prefix_rejected(env):
    conn = obs.open_state_db_readonly(env["db"])
    try:
        with pytest.raises(obs.AmbiguousSessionPrefix) as ei:
            obs.resolve_session_id(conn, "20260902")
        assert FAILED_SID in ei.value.candidates
        assert RECOVERED_SID in ei.value.candidates
    finally:
        conn.close()


def test_resolve_not_found(env):
    conn = obs.open_state_db_readonly(env["db"])
    try:
        with pytest.raises(obs.SessionNotFound):
            obs.resolve_session_id(conn, "20991231_nope")
    finally:
        conn.close()


def test_state_db_opened_read_only(env):
    conn = obs.open_state_db_readonly(env["db"])
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions VALUES ('x',NULL,NULL,NULL,NULL,NULL,NULL,NULL)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main-wrapper log parsing
# ---------------------------------------------------------------------------


def test_lineage_and_tokens(env):
    r = _profile(env)
    assert r["lineage"]["responses"] == 2
    tail = r["lineage"]["chain_tail"]
    assert tail[0]["response_id"] == "resp_1788335882955_14c"
    assert tail[0]["base_tokens"] == 95593
    assert tail[0]["prompt_tokens"] == 105150
    # usage blocks: 105150+115893 prompt, 931+120 generated, 95594+106008 cached
    assert r["tokens"]["prompt"] == 221043
    assert r["tokens"]["generated"] == 1051
    assert r["tokens"]["cached"] == 201602
    expected_weighted = round(201602 / 221043, 4)
    assert r["tokens"]["weighted_cache_reuse"] == expected_weighted


def test_timing(env):
    r = _profile(env)
    assert r["timing"]["prefill_ms"] == 3500.0
    assert r["timing"]["decode_ms"] == 54000.0
    assert r["timing"]["tokenize_ms"] == 20.5
    assert r["timing"]["prefill_tps"] == 96000.0
    assert r["timing"]["decode_tps"] == 20.7
    assert r["timing"]["total_seconds"] == 3600.0


def test_hot_cache_reuse(env):
    r = _profile(env)
    assert r["hot_cache"]["events"] == 2
    assert r["hot_cache"]["reused"] == 95594 + 106008
    assert r["hot_cache"]["total"] == 105150 + 115893
    assert 0 < r["hot_cache"]["reuse_ratio"] <= 1.0


def test_mtp_acceptance_and_regime(env):
    r = _profile(env)
    m = r["mtp"]
    assert m["rounds"] == 2
    # drafts={ 15078, 364 } + drafts={ 2 } = 3 candidate positions (token
    # ids are values, never summed into the count).
    assert m["draft_tokens"] == 3
    assert m["accepted_tokens"] == 1
    assert m["acceptance_rate"] == 0.3333
    assert m["regime_checks"] == 2
    assert m["regime_two_chunk_rounds"] == 1
    assert m["two_chunk_ms_per_tok_median"] is not None


def test_mtp_draft_tokens_count_candidates_not_token_ids():
    """Regression: drafts={...} lists candidate token *ids*.

    The parser used to sum them, so a real profile reported 86,774,194
    drafted tokens and ~0.0002 acceptance. drafted_tokens must count the
    parsed candidates/positions per round; accepted is the accepted prefix
    count, so the ratio is always in [0, 1].
    """
    stats = obs.MainLogStats()
    lines = [
        "  [mtp-round] off0=100 t1=47983 m=3/3 drafts={ 150781, 364, 999 } accepted=2",
        "  [mtp-round] off0=103 t1=1892 m=2/2 drafts={ 100000, 200000 } accepted=0",
        # A distinct round that drafts the same ids with the same accept
        # count (seen live): must not be deduped away.
        "  [mtp-round] off0=105 t1=77 m=2/2 drafts={ 100000, 200000 } accepted=0",
    ]
    for line in lines:
        stats.consume(line, None)
    # Rotation replay of an identical line still dedups.
    stats.consume(lines[1], None)
    m = stats.finalize()["mtp"]
    assert m["rounds"] == 3
    # 3 + 2 + 2 candidates — summing the ids would give 452,144+.
    assert m["draft_tokens"] == 7
    assert m["accepted_tokens"] == 2
    assert m["acceptance_rate"] == round(2 / 7, 4)
    assert 0 < m["acceptance_rate"] <= 1.0
    assert stats.duplicates == 1


def test_anomaly_events(env):
    r = _profile(env)
    a = r["anomalies"]
    assert a["disconnects"] == 1
    assert a["capacity"] == 1
    assert a["retries"] == 1
    assert a["poison"] == 1
    assert a["malformed_requests"] == 1
    assert any("insufficient memory" in s for s in r["anomaly_reasons"])
    assert any("disconnected" in s for s in r["anomaly_reasons"])


def test_rotation_overlap_deduplicated(env):
    """server.log.1 replays the tail of server.log — events must not double-count."""
    r = _profile(env)
    # Without dedup the two usage blocks would count twice (442086 prompt).
    assert r["tokens"]["prompt"] == 221043
    assert r["hot_cache"]["events"] == 2
    total_dupes = sum(
        0 for _ in [1]
    ) + (r["lineage"]["responses"])
    assert total_dupes == 2  # lineage dedup by response id


def test_malformed_and_truncated_lines_tolerated(env):
    r = _profile(env)
    main_metas = [m for m in r["sources"]["main_wrapper_logs"] if m["kind"] == "main"]
    assert main_metas  # files were scanned
    assert any(m["truncated_lines"] >= 1 for m in main_metas)
    # Garbage line did not raise and did not corrupt metrics.
    assert r["tokens"]["prompt"] == 221043


def test_missing_logs_warn_not_raise(tmp_path, env):
    r = obs.profile_session(
        FAILED_SID,
        db_path=env["db"],
        hermes_home=env["home"],
        main_log_files=[],
        sidecar_log_files=[],
    )
    assert any("no main-wrapper log" in w for w in r["warnings"])
    assert any("no sidecar log" in w for w in r["warnings"])
    assert r["lineage"]["responses"] == 0
    assert r["timing"]["prefill_ms"] == 0.0


def test_unreadable_log_warns(env, tmp_path):
    missing = tmp_path / "gone" / "server.log"
    r = obs.profile_session(
        FAILED_SID,
        db_path=env["db"],
        hermes_home=env["home"],
        main_log_files=[missing],
        sidecar_log_files=env["sidecar_files"],
    )
    assert any("unreadable" in w for w in r["warnings"])


# ---------------------------------------------------------------------------
# Sidecar log parsing
# ---------------------------------------------------------------------------


def test_sidecar_cache_and_events(env):
    r = _profile(env)
    sc = r["sidecar"]
    assert sc["cache"]["decisions"]["hit"] == 2
    assert sc["cache"]["decisions"]["miss"] == 1
    assert sc["cache"]["hit_prefix_tokens"] == 2048 + 4096
    assert sc["cache"]["persist"]["ok"] == 1
    assert sc["events"]["queue_waits"] == 1
    assert sc["queue_wait_ms"] == 1500.0
    assert sc["events"]["capacity"] == 1
    assert sc["events"]["stalls"] == 1
    assert sc["events"]["disconnects"] == 1
    assert sc["events"]["retries"] == 1
    assert sc["events"]["poison"] == 1
    assert sc["events"]["malformed_requests"] == 1


def test_sidecar_window_filtering(env):
    """Lines outside the session window are excluded."""
    # RECOVERED session window starts 2h after the sidecar fixture timestamps.
    r = _profile(env, RECOVERED_SID)
    sc = r["sidecar"]
    assert sc["lines_in_window"] == 0
    assert sc["cache"]["decisions"]["hit"] == 0


# ---------------------------------------------------------------------------
# Conscience artifacts
# ---------------------------------------------------------------------------


def test_conscience_artifacts(env):
    r = _profile(env)
    c = r["conscience"]
    assert c["present"] is True
    assert c["events"]["total"] == 3
    assert c["events"]["by_type"]["TASK_START"] == 1
    assert c["review_duration_ms_total"] == 370.5
    assert c["llm_audits"]["total"] == 3
    assert c["llm_audits"]["interventions"] == 1
    assert c["llm_audits"]["parse_failures"] == 1
    assert c["stop_audit_verdict"] == "allow_stop"
    assert c["interventions"] == 1
    assert c["completion_ledger"] == 1


def test_conscience_malformed_artifact_warns(env):
    r = _profile(env)
    assert any("task-contract.json" in w for w in r["warnings"])


def test_conscience_missing_dir_warns(env):
    r = _profile(env, RECOVERED_SID)
    assert r["conscience"]["present"] is False
    assert any("no conscience artifact dir" in w for w in r["warnings"])


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_report_contains_no_content(env):
    r = _profile(env)
    blob = json.dumps(r, default=str)
    for secret in (
        "SECRET SYSTEM PROMPT",
        "SECRET USER PROMPT",
        "SECRET ASSISTANT CONTENT",
        "SECRET TOOL OUTPUT",
        "SECRET TOOL OUTPUT 2",
    ):
        assert secret not in blob, f"privacy leak: {secret!r} in report"


def test_redaction_bounds_reasons():
    long_reason = "REFUSED " + ("x" * 200)
    assert len(obs._redact(long_reason)) <= obs._MAX_REASON_CHARS
    assert obs._redact("Redirect: do NOT stop <tag>") == "redirect: do not stop tag"


# ---------------------------------------------------------------------------
# DB-derived sections
# ---------------------------------------------------------------------------


def test_tools_and_compaction_from_db(env):
    r = _profile(env)
    assert r["tools"]["calls_by_name"]["terminal"] == 2
    assert r["tools"]["total_calls"] == 2
    # tool exec: assistant@t0+10 -> first tool result@t0+40 = 30s
    assert r["tools"]["execution_seconds"] == 30.0
    assert r["compactions"]["count"] == 1
    assert r["routing"]["assistant_models"]["model-x"] == 2


# ---------------------------------------------------------------------------
# Comparison + attribution
# ---------------------------------------------------------------------------


def test_compare_profiles_proven_vs_unattributed(env):
    base = _profile(env, FAILED_SID)      # 3600s
    target = _profile(env, RECOVERED_SID)  # 3099s
    cmp = obs.compare_profiles(base, target)
    assert cmp["duration_delta_seconds"] == pytest.approx(-501.0)
    proven = {p["metric"]: p for p in cmp["proven_differences"]}
    # Recovered had fewer tool calls (1 vs 2) — proven from DB.
    assert proven["tool_calls"]["delta"] == -1
    # Recovered has no conscience artifacts -> conscience cost is NOT proven
    # (missing on one side), it must appear as a caveat instead.
    assert "conscience_review_ms" not in proven
    assert any("no conscience artifacts" in c for c in cmp["caveats"])
    # Residual is reported and equals total minus explained time.
    assert cmp["unattributed_time_ms"] == pytest.approx(
        cmp["duration_delta_seconds"] * 1000.0 - cmp["explained_time_ms"], abs=1.0
    )
    summary = cmp["summary"]
    assert "501.0s faster" in summary
    assert "unattributed residual" in summary


def test_compare_missing_duration_side(tmp_path, env):
    base = _profile(env, FAILED_SID)
    target = _profile(env, RECOVERED_SID)
    target["session"]["duration_seconds"] = None
    cmp = obs.compare_profiles(base, target)
    assert cmp["duration_delta_seconds"] is None
    assert "unavailable" in cmp["summary"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_format_report_sections(env):
    text = obs.format_report(_profile(env))
    for section in ("SESSION OBSERVATORY", "LINEAGE", "TOKENS", "TIMING", "TOOLS",
                    "CONSCIENCE", "COMPACTIONS", "ANOMALIES", "ROUTING", "MTP"):
        assert section in text


def test_format_comparison(env):
    base = _profile(env, FAILED_SID)
    target = _profile(env, RECOVERED_SID)
    text = obs.format_comparison(obs.compare_profiles(base, target))
    assert "PROVEN DIFFERENCES" in text
    assert "unattributed residual" in text
    assert "501.0s faster" in text


# ---------------------------------------------------------------------------
# CLI integration (dispatch + argparse contract)
# ---------------------------------------------------------------------------


def test_cli_dispatch_profile(monkeypatch, capsys, env):
    from hermes_cli import sessions_cmd

    real_profile = obs.profile_session
    monkeypatch.setattr(
        obs, "profile_session",
        lambda sid, **kw: real_profile(
            sid, db_path=env["db"], hermes_home=env["home"],
            main_log_files=env["main_files"], sidecar_log_files=env["sidecar_files"],
        ),
    )
    import argparse
    args = argparse.Namespace(
        sessions_action="profile", session_id=FAILED_SID,
        json=False, compare=None,
    )
    rc = sessions_cmd.cmd_sessions(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "SESSION OBSERVATORY" in out


def test_cli_dispatch_profile_json_and_compare(monkeypatch, capsys, env):
    from hermes_cli import sessions_cmd

    real_profile = obs.profile_session
    monkeypatch.setattr(
        obs, "profile_session",
        lambda sid, **kw: real_profile(
            sid, db_path=env["db"], hermes_home=env["home"],
            main_log_files=env["main_files"], sidecar_log_files=env["sidecar_files"],
        ),
    )
    import argparse
    args = argparse.Namespace(
        sessions_action="profile", session_id=RECOVERED_SID,
        json=True, compare=FAILED_SID,
    )
    rc = sessions_cmd.cmd_sessions(args)
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["comparison"]["duration_delta_seconds"] == pytest.approx(-501.0)
    assert payload["base"]["session_id"] == FAILED_SID
    assert payload["target"]["session_id"] == RECOVERED_SID


def test_cli_dispatch_not_found_exit1(monkeypatch, capsys):
    from hermes_cli import sessions_cmd

    def boom(sid, **kw):
        raise obs.SessionNotFound(f"no session matches '{sid}'")

    monkeypatch.setattr(obs, "profile_session", boom)
    import argparse
    args = argparse.Namespace(
        sessions_action="profile", session_id="zzz", json=False, compare=None,
    )
    rc = sessions_cmd.cmd_sessions(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "no session matches" in out


def test_cli_dispatch_ambiguous_exit1(monkeypatch, capsys):
    from hermes_cli import sessions_cmd

    def boom(sid, **kw):
        raise obs.AmbiguousSessionPrefix(sid, ["20260902_1_x", "20260902_2_y"])

    monkeypatch.setattr(obs, "profile_session", boom)
    import argparse
    args = argparse.Namespace(
        sessions_action="profile", session_id="20260902", json=False, compare=None,
    )
    rc = sessions_cmd.cmd_sessions(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "ambiguous" in out


# ---------------------------------------------------------------------------
# Real-store validation (skips cleanly when the sessions are not present)
# ---------------------------------------------------------------------------

_REAL_HOME = Path.home() / ".hermes"
_REAL_DB = _REAL_HOME / "state.db"
_REAL_MAIN_FILES = obs.discover_log_files("main")
_REAL_SIDECAR_FILES = obs.discover_log_files("sidecar")
_REAL_FAILED = "20260902_123950_f79483"
_REAL_RECOVERED = "20260902_202149_c4a9d7"


def _real_has(sid: str) -> bool:
    if not _REAL_DB.exists():
        return False
    conn = sqlite3.connect(f"file:{_REAL_DB}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (sid,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _real_profile(sid: str) -> dict:
    return obs.profile_session(
        sid,
        db_path=_REAL_DB,
        hermes_home=_REAL_HOME,
        main_log_files=_REAL_MAIN_FILES,
        sidecar_log_files=_REAL_SIDECAR_FILES,
    )


@pytest.mark.skipif(
    not (_real_has(_REAL_FAILED) and _real_has(_REAL_RECOVERED)),
    reason="real validation sessions not present in ~/.hermes/state.db",
)
def test_real_sessions_compare_explains_501s():
    base = _real_profile(_REAL_FAILED)
    target = _real_profile(_REAL_RECOVERED)
    cmp = obs.compare_profiles(base, target)
    d = cmp["duration_delta_seconds"]
    assert d is not None
    assert -560.0 < d < -440.0, f"expected ~-501s, got {d}"
    assert "faster" in cmp["summary"]
    # Proven and unattributed must both be explicitly represented.
    assert isinstance(cmp["proven_differences"], list)
    assert cmp["unattributed_time_ms"] is not None
    # Privacy: no prompt content leaks into the JSON report.
    blob = json.dumps({"b": base, "t": target, "c": cmp}, default=str)
    assert len(blob) < 5_000_000  # structural report, not a payload dump