"""Tests for the Session Replay Capsule (agent/replay_capsule.py).

Uses a fixture state DB, synthetic mlx-serve-format wrapper logs, and a
threaded fake wrapper that speaks /v1/responses with previous_response_id
chaining — no real model, no network beyond loopback.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent import replay_capsule as rc

BASE_TS = 1_788_000_000.0  # fixed anchor so response-id ms stamps land in-window


def rid(i: int, tag: str = "aa") -> str:
    return f"resp_{int((BASE_TS + i * 10) * 1000)}_{tag}{i}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state_db(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT,
            system_prompt_hash TEXT, billing_base_url TEXT,
            started_at REAL NOT NULL, ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
            tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
            finish_reason TEXT, responses_response_id TEXT,
            timestamp REAL NOT NULL, api_content TEXT,
            _compressed_summary INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def add_session(db, sid="sess1", *, started=BASE_TS, ended=BASE_TS + 60,
                model="test-model", url="http://127.0.0.1:9999/v1"):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO sessions (id, source, model, system_prompt_hash, "
        "billing_base_url, started_at, ended_at) VALUES (?, 'cli', ?, 'hash1', ?, ?, ?)",
        (sid, model, url, started, ended),
    )
    conn.commit()
    conn.close()


def add_message(db, sid, role, content="", **kw):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_call_id, "
        "tool_calls, tool_name, finish_reason, responses_response_id, "
        "timestamp, api_content, _compressed_summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid, role, content,
            kw.get("tool_call_id"), json.dumps(kw["tool_calls"]) if kw.get("tool_calls") else None,
            kw.get("tool_name"), kw.get("finish_reason"), kw.get("response_id"),
            kw.get("timestamp", BASE_TS + 5), kw.get("api_content"),
            1 if kw.get("compressed") else 0,
        ),
    )
    conn.commit()
    conn.close()


def write_session_log(path: Path) -> Path:
    """Synthetic main-wrapper log covering the original session window."""
    lines = [
        # turn 0 generation: usage + timing attributed to rid(0)
        f'{rid(0)} "usage": {{"input_tokens": 80, "output_tokens": 5, '
        f'"total_tokens": 85, "input_tokens_details": {{"cached_tokens": 0}}}} '
        f'"prompt_ms": 12.5, "predicted_ms": 30.0',
        "[mtp-round] off0=1 t1=2 drafts={11 22 33} accepted=2",
        # turn 1: exact continuation from turn 0's response, then usage
        f"[hermes-qwen-late] exact continuation prev={rid(0)} base=85 "
        f"prompt=110 delta_messages=1 tools=true runtime_chars=0",
        f'{rid(1)} "usage": {{"input_tokens": 110, "output_tokens": 8, '
        f'"total_tokens": 118, "input_tokens_details": {{"cached_tokens": 85}}}} '
        f'"prompt_ms": 9.0, "predicted_ms": 40.0',
        "[mtp-round] off0=5 t1=6 drafts={44 55} accepted=1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def session_log(tmp_path):
    return write_session_log(tmp_path / "server.log")


@pytest.fixture()
def capsule_dir(tmp_path):
    d = tmp_path / "capsules"
    d.mkdir()
    return d


def make_two_turn_session(db, sid="sess1", with_lineage=True):
    add_session(db, sid)
    add_message(db, sid, "user", "Run pwd.", timestamp=BASE_TS + 1)
    add_message(
        db, sid, "assistant", "",
        tool_calls=[{"id": "call_1", "call_id": "call_1", "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command": "pwd"}'}}],
        finish_reason="tool_calls",
        response_id=rid(0) if with_lineage else None,
        timestamp=BASE_TS + 2,
    )
    add_message(db, sid, "tool", "/home/u", tool_call_id="call_1",
                tool_name="terminal", timestamp=BASE_TS + 3)
    add_message(db, sid, "assistant", "You are in /home/u.",
                finish_reason="stop",
                response_id=rid(1) if with_lineage else None,
                timestamp=BASE_TS + 4)


# ---------------------------------------------------------------------------
# Fake mlx-serve-compatible wrapper
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.wrapper.handle_post(self, body)

    def do_GET(self):
        self.server.wrapper.handle_get(self)

    def log_message(self, *args):
        pass


class FakeWrapper:
    """Minimal /v1/responses server with previous_response_id chaining.

    ``script``: per-turn canned outputs, each {text?, tool_calls?}.
    ``fail_unknown_parent``: 404 on unknown previous_response_id (like a
    wrapper whose response store rotated the entry out).
    Logs mlx-serve-format continuation/usage lines to ``log_path``.
    """

    def __init__(self, script, log_path: Path, *, fail_unknown_parent=True,
                 instructions="You are Hermes.", seed_store=(), diverge=False):
        self.script = list(script)
        self.log_path = log_path
        self.fail_unknown_parent = fail_unknown_parent
        self.instructions = instructions
        self.diverge = diverge
        self.store = {}
        for sid in seed_store:
            self.store[sid] = {"id": sid, "instructions": instructions}
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self._n = 0

    def _log(self, line: str):
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def handle_post(self, handler, body):
        self.posts.append(body)
        pid = body.get("previous_response_id")
        if pid and pid not in self.store:
            if self.fail_unknown_parent:
                payload = json.dumps(
                    {"error": {"message": f"previous_response_id '{pid}' not found"}}
                ).encode()
                handler.send_response(404)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)
                return
        elif pid:
            self._log(
                f"[hermes-qwen-late] exact continuation prev={pid} base=100 "
                f"prompt=150 delta_messages=1 tools=true runtime_chars=0"
            )
        n = self._n
        self._n += 1
        # content-based turn selection so every replay mode (chain delta vs
        # full transcript) gets the same canned output for the same turn
        step_idx = 1 if any(
            i.get("type") == "function_call_output" for i in body.get("input", [])
        ) else 0
        step = self.script[step_idx if len(self.script) > 1 else 0]
        new_id = f"resp_{int((BASE_TS + 9000 + n) * 1000)}_new{n}"
        output = []
        text = step.get("text", "")
        if self.diverge:
            text = (text + " but different") if text else "totally different"
        if text:
            output.append({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": text}]})
        for tc in step.get("tool_calls") or []:
            output.append({"type": "function_call", "call_id": tc["call_id"],
                           "name": tc["name"], "arguments": tc["arguments"]})
        usage = {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130,
                 "input_tokens_details": {"cached_tokens": 100},
                 "output_tokens_details": {"reasoning_tokens": 0}}
        obj = {"id": new_id, "object": "response", "status": "completed",
               "output": output, "usage": usage}
        self.store[new_id] = obj
        self._log(
            f'{new_id} "usage": {{"input_tokens": 120, "output_tokens": 10, '
            f'"total_tokens": 130, "input_tokens_details": {{"cached_tokens": 100}}}} '
            f'"prompt_ms": 10.0, "predicted_ms": 20.0'
        )
        self._log("[mtp-round] off0=1 t1=2 drafts={7 8 9} accepted=2")
        step_idx = 1 if any(
            i.get("type") == "function_call_output" for i in body.get("input", [])
        ) else 0
        step = self.script[step_idx if len(self.script) > 1 else 0]
        payload = json.dumps(obj).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def handle_get(self, handler):
        rid_ = handler.path.rsplit("/", 1)[-1]
        self.gets.append(rid_)
        stored = self.store.get(rid_)
        if stored is None:
            payload = json.dumps({"error": "not found"}).encode()
            handler.send_response(404)
        else:
            payload = json.dumps({
                "id": rid_, "instructions": self.instructions,
                "status": "completed", "output": [],
            }).encode()
        handler.send_response(200 if stored is not None else 404)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


@pytest.fixture()
def fake_wrapper(tmp_path):
    servers = []

    def spawn(**kw):
        log_path = kw.pop("log_path", tmp_path / "replay-server.log")
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        srv.wrapper = FakeWrapper(log_path=log_path, **kw)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        servers.append(srv)
        port = srv.server_address[1]
        return srv, f"http://127.0.0.1:{port}/v1", log_path

    yield spawn
    for s in servers:
        s.shutdown()
        s.server_close()


# ---------------------------------------------------------------------------
# Capsule creation
# ---------------------------------------------------------------------------


def test_create_capsule_turns_and_lineage(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    manifest = rc.create_capsule(
        "sess1", db_path=state_db, capsules_dir=capsule_dir,
        main_log_files=[session_log],
    )
    assert manifest["turns"] == 2
    assert manifest["lineage"]["coverage"] == 1.0
    cap = rc.load_capsule("sess1", capsules_dir=capsule_dir)
    t0, t1 = cap["trajectory"]["turns"]
    assert t0["parent_response_id"] is None
    assert t0["delta"][0]["content"] == "Run pwd."
    # recorded tool result is captured in turn 1's delta — replay supplies it
    assert t1["parent_response_id"] == rid(0)
    assert t1["delta"] == [{"role": "tool", "content": "/home/u",
                            "tool_call_id": "call_1", "tool_name": "terminal"}]
    assert t1["recorded"]["content"] == "You are in /home/u."
    assert t0["recorded"]["tool_calls"][0]["function"]["name"] == "terminal"


def test_create_capsule_baseline_attribution(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    cap = rc.load_capsule("sess1", capsules_dir=capsule_dir)
    t0, t1 = cap["trajectory"]["turns"]
    assert t0["baseline"]["prompt_tokens"] == 80
    assert t0["baseline"]["prefill_ms"] == 12.5
    assert t0["baseline"]["mtp_drafted"] == 3
    assert t0["baseline"]["mtp_accepted"] == 2
    assert t1["baseline"]["continuation"] == "exact"
    assert t1["baseline"]["continuation_base_tokens"] == 85
    assert t1["baseline"]["cached_tokens"] == 85
    assert manifest_cover(manifest_of(rc, "sess1", capsule_dir), 2)


def manifest_of(rc_mod, name, capsule_dir):
    return rc_mod.load_capsule(name, capsules_dir=capsule_dir)["manifest"]


def manifest_cover(manifest, turns):
    return manifest["baseline_turns_covered"] == turns


def test_create_capsule_missing_lineage_tolerated(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db, with_lineage=False)
    manifest = rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                                 main_log_files=[session_log])
    assert manifest["lineage"]["coverage"] == 0.0
    assert any("stateful chaining impossible" in w for w in manifest["warnings"])
    assert any("only run in 'full'" in w for w in manifest["warnings"])


def test_create_capsule_compaction_rebase(state_db, session_log, capsule_dir):
    add_session(state_db)
    add_message(state_db, "sess1", "user", "old ask", timestamp=BASE_TS + 1)
    add_message(state_db, "sess1", "assistant", "old answer",
                finish_reason="stop", response_id=rid(0), timestamp=BASE_TS + 2)
    add_message(state_db, "sess1", "user", "Summary of previous conversation: ...",
                compressed=True, timestamp=BASE_TS + 3)
    add_message(state_db, "sess1", "assistant", "new answer",
                finish_reason="stop", response_id=rid(1), timestamp=BASE_TS + 4)
    manifest = rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                                 main_log_files=[session_log])
    assert manifest["lineage"]["compaction_rebases"] == 1
    cap = rc.load_capsule("sess1", capsules_dir=capsule_dir)
    rebase_turn = cap["trajectory"]["turns"][1]
    assert rebase_turn["rebase"] is True
    assert rebase_turn["parent_response_id"] is None
    assert any("compaction rebase" in w for w in manifest["warnings"])


def test_create_requires_force(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    with pytest.raises(rc.CapsuleExists):
        rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                          main_log_files=[session_log])
    m = rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                          main_log_files=[session_log], force=True)
    assert m["turns"] == 2


def test_create_tolerates_rotated_logs(state_db, capsule_dir, tmp_path):
    make_two_turn_session(state_db)
    missing = tmp_path / "server.log.5"  # unreadable/absent file
    manifest = rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                                 main_log_files=[missing])
    assert manifest["baseline_turns_covered"] == 0
    assert any("no baseline telemetry" in w or "unreadable" in w
               for w in manifest["warnings"])


def test_load_capsule_tamper_warning(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    traj = capsule_dir / "sess1" / rc.TRAJECTORY_NAME
    traj.write_text(traj.read_text().replace("Run pwd.", "Run ls."), encoding="utf-8")
    cap = rc.load_capsule("sess1", capsules_dir=capsule_dir)
    assert any("does not match the manifest hash" in w for w in cap["manifest"]["warnings"])


def test_hydrate_instructions_from_wrapper(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    _srv, url, _log = fake_wrapper(script=[{"text": "x"}], seed_store=[rid(0)])
    manifest = rc.create_capsule(
        "sess1", db_path=state_db, capsules_dir=capsule_dir,
        main_log_files=[session_log], hydrate_endpoint=url,
    )
    assert manifest["instructions_captured"] is True
    cap = rc.load_capsule("sess1", capsules_dir=capsule_dir)
    assert cap["trajectory"]["instructions"] == "You are Hermes."


def test_hydration_failure_tolerated(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    manifest = rc.create_capsule(
        "sess1", db_path=state_db, capsules_dir=capsule_dir,
        main_log_files=[session_log],
        hydrate_endpoint="http://127.0.0.1:1/v1",  # nothing listening
    )
    assert manifest["instructions_captured"] is False
    assert any("hydration failed" in w for w in manifest["warnings"])


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _golden_script():
    return [
        {"tool_calls": [{"call_id": "call_1", "name": "terminal",
                         "arguments": '{"command": "pwd"}'}]},
        {"text": "You are in /home/u."},
    ]


def test_replay_golden_exact(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir, endpoint=url, chain="golden",
        main_log_files=[log],
    )
    assert run["turns_completed"] == 2
    assert run["chain_breaks"] == 0
    t0, t1 = run["turns"]
    # turn 0: fresh root, delta-only input, no parent
    assert t0["requested_parent"] is None
    # turn 1: chained from the ORIGINAL recorded parent
    assert t1["requested_parent"] == rid(0)
    assert t1["telemetry"]["continuation"] == "exact"
    assert t1["telemetry"]["mtp_accepted"] == 2
    # recorded tool result was supplied, not re-executed
    tool_item = json.loads(json.dumps(_srv.wrapper.posts[1]["input"]))
    assert tool_item[0]["type"] == "function_call_output"
    assert tool_item[0]["output"] == "/home/u"

    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    assert cmp["verdict"] == "exact"
    assert cmp["behavior"]["identical"] == 2
    assert cmp["chain_continuity"]["exact_continuations"] == 1


def test_replay_chain_break_falls_back_to_full(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    # wrapper's response store rotated the original parent out
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[])
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir, endpoint=url, chain="golden",
        main_log_files=[log],
    )
    assert run["chain_breaks"] == 1
    t1 = run["turns"][1]
    assert t1["degraded_full"] is True
    assert t1["mode"] == "full"
    assert "not found" in t1["chain_break_detail"]
    # full fallback resent the whole transcript
    resent = _srv.wrapper.posts[-1]
    assert resent.get("previous_response_id") in (None,)
    assert len(resent["input"]) >= 3

    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    # behaviour matches but continuity was NOT preserved -> never 'exact'
    assert cmp["verdict"] != "exact"
    assert cmp["chain_continuity"]["chain_breaks"] == 1


def test_replay_replay_mode_chains_new_ids(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[])
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir, endpoint=url, chain="replay",
        main_log_files=[log],
    )
    assert run["chain_breaks"] == 0  # chains its own fresh ids
    assert run["turns"][1]["requested_parent"] == run["turns"][0]["response_id"]


def test_replay_full_mode_never_exact(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[])
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir, endpoint=url, chain="full",
        main_log_files=[log],
    )
    assert all(e["mode"] == "full" for e in run["turns"])
    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    assert cmp["verdict"] == "equivalent-unverified"
    assert any("not be claimed" in r for r in cmp["reasons"])


def test_replay_diverged_detected(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)],
                                  diverge=True)
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir, endpoint=url, chain="golden",
        main_log_files=[log],
    )
    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    assert cmp["verdict"] == "diverged"
    assert cmp["behavior"]["diverged"] >= 1


def test_replay_refuses_non_loopback(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    with pytest.raises(rc.NonLoopbackEndpoint):
        rc.replay_capsule("sess1", capsules_dir=capsule_dir,
                          endpoint="https://api.example.com/v1")
    # ... but accepts it with explicit consent
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    run = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                            chain="golden", main_log_files=[log], allow_remote=True)
    assert run["turns_completed"] == 2


def test_replay_endpoint_defaults_to_session_url(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    _srv, url, _log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    # rewrite the session's recorded URL to the fake wrapper's port
    conn = sqlite3.connect(state_db)
    conn.execute("UPDATE sessions SET billing_base_url = ?", (url,))
    conn.commit()
    conn.close()
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    run = rc.replay_capsule("sess1", capsules_dir=capsule_dir, chain="golden",
                            main_log_files=[_srv.wrapper.log_path])
    assert run["endpoint"] == url


def test_replay_connection_error_recorded_not_raised(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    run = rc.replay_capsule(
        "sess1", capsules_dir=capsule_dir,
        endpoint="http://127.0.0.1:1/v1",  # nothing listening
        chain="full", main_log_files=[], timeout=2.0,
    )
    assert run["turns_failed"] == 2
    assert run["turns_completed"] == 0
    assert any("URLError" in e["error"] or "ConnectionRefused" in e["error"]
               or "refused" in e["error"].lower() for e in run["turns"])
    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    assert cmp["verdict"] == "failed"


def test_replay_max_turns(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    run = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                            chain="golden", max_turns=1, main_log_files=[log])
    assert run["turns_requested"] == 1
    assert len(_srv.wrapper.posts) == 1


# ---------------------------------------------------------------------------
# Runs, listing, A/B
# ---------------------------------------------------------------------------


def test_runs_persisted_and_listed(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    run = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                            chain="golden", main_log_files=[log])
    run_file = Path(run["_path"])
    assert run_file.is_file()
    assert json.loads(run_file.read_text())["run_id"] == run["run_id"]
    listed = rc.list_capsules(capsules_dir=capsule_dir)
    assert listed[0]["name"] == "sess1"
    assert run["run_id"] in listed[0]["_runs"]


def test_compare_runs_ab(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    a = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                          chain="golden", main_log_files=[log])
    b = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                          chain="full", main_log_files=[log])
    cmp = rc.compare_runs("sess1", a["run_id"], b["run_id"], capsules_dir=capsule_dir)
    assert cmp["turns_compared"] == 2
    assert cmp["a_chain_mode"] == "golden"
    assert cmp["b_chain_mode"] == "full"
    assert all(r["content_similarity"] == 1.0 for r in cmp["turns"])


def test_compare_missing_run_raises(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    with pytest.raises(rc.CapsuleError):
        rc.compare_run("sess1", capsules_dir=capsule_dir)


def test_resolve_capsule_by_path_or_name(state_db, session_log, capsule_dir):
    make_two_turn_session(state_db)
    rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                      main_log_files=[session_log])
    by_name = rc.resolve_capsule_dir("sess1", capsule_dir)
    by_path = rc.resolve_capsule_dir(str(capsule_dir / "sess1"))
    assert by_name == by_path
    with pytest.raises(rc.CapsuleNotFound):
        rc.resolve_capsule_dir("nope", capsule_dir)


# ---------------------------------------------------------------------------
# Rendering smoke (terminal output, no markdown)
# ---------------------------------------------------------------------------


def test_renderers_produce_plain_text(state_db, session_log, capsule_dir, fake_wrapper):
    make_two_turn_session(state_db)
    manifest = rc.create_capsule("sess1", db_path=state_db, capsules_dir=capsule_dir,
                                 main_log_files=[session_log])
    assert "CAPSULE — sess1" in rc.format_manifest(manifest)
    _srv, url, log = fake_wrapper(script=_golden_script(), seed_store=[rid(0)])
    run = rc.replay_capsule("sess1", capsules_dir=capsule_dir, endpoint=url,
                            chain="golden", main_log_files=[log])
    out = rc.format_run(run)
    assert "REPLAY RUN" in out and "**" not in out
    cmp = rc.compare_run("sess1", capsules_dir=capsule_dir, run_id=run["run_id"])
    cout = rc.format_comparison(cmp)
    assert "VERDICT:" in cout and "MTP acceptance" in cout