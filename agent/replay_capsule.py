"""Session Replay Capsule — portable replay of a completed session's
model-facing trajectory against a local inference wrapper.

A *capsule* is a local directory (default ``~/.hermes/capsules/<name>/``)
containing everything needed to re-drive a finished session's API calls
against a wrapper without re-executing any tools:

* ``manifest.json``  — provenance, lineage coverage, completeness verdict.
* ``trajectory.json`` — one entry per model call: the recorded request
  parent (``previous_response_id``), the recorded input delta (including
  the recorded tool results, which are replayed verbatim), the full
  transcript projection (for stateless fallback), the recorded model
  output, and best-effort baseline telemetry attributed from the
  main-wrapper logs (tokens, prefill/decode ms, exact-continuation vs
  full-rebuild decisions, MTP acceptance).

Replay POSTs each turn to ``/v1/responses`` on a chosen loopback endpoint
and records latency, usage, finish reason, generated text/tool calls, and
post-run wrapper-log telemetry (continuation decisions, MTP rounds) keyed
to the freshly generated response ids. Tools are NEVER re-executed — the
recorded tool results are supplied as input — so replay has no external
side effects beyond the model call itself.

Chain modes:
* ``golden`` — replay each turn from the ORIGINAL recorded parent id.
  Requires the original responses to still live in the wrapper's response
  store. Maximum cache reuse; isolates per-turn regressions.
* ``replay`` — chain from the freshly generated response ids. Divergence
  compounds; exercises the wrapper's chaining of new branches.
* ``full``   — no chaining: send the whole transcript each turn. Tests
  stateless prompt-cache continuity only.
``auto`` picks golden when lineage is complete, else full.

Honesty contract: a comparison reports ``exact`` ONLY when every turn's
behaviour matches the recording AND baseline evidence exists for that
turn AND the wrapper confirmed an exact continuation. Missing lineage,
rotated logs, or a chain break downgrade the verdict explicitly — the
capsule never silently claims an exact replay.

Model-facing instructions and tool definitions are not persisted in the
state DB, but the wrapper's response store keeps them per response. They
are NOT stable across calls within one conversation: available tools and
transient internal policy can change per model call (e.g. conscience-
driven tool narrowing), so a single global evidence set would silently
replay most calls with the wrong request fields. The capsule therefore
hydrates evidence PER TURN: each turn is fetched from its OWN recorded
response id at creation, with provenance recorded per turn
(``turns[].evidence``). Values are content-addressed into
``evidence_blobs`` so byte-identical instructions across turns share one
copy. A manual ``--instructions-file`` still wins (applied to every turn,
source ``file``). A turn whose response has rotated out keeps an
explicit gap — its evidence is NEVER backfilled from a neighbouring
turn's response, because "close" evidence is exactly the fabrication
this module exists to prevent. Such turns replay without instructions
and/or tool definitions, compare reports the gap per turn, and the
verdict is capped below ``exact`` on the strength of evidence that no
longer exists. Legacy schema-version-1 capsules (one global evidence set
for all turns) still replay with that global evidence, unchanged.

Remaining fidelity gap (recorded as warnings, never silently ignored):
* Baseline telemetry is attributed from wrapper logs by response-id
  anchor; rotated logs or truncated lines yield partial baselines.

Stdlib only (urllib), mirroring the Session Observatory conventions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import session_observatory as obs

CAPSULE_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 1

MANIFEST_NAME = "manifest.json"
TRAJECTORY_NAME = "trajectory.json"
RUNS_DIR_NAME = "runs"
RUN_NAME = "run.json"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CapsuleError(Exception):
    """Base error for replay capsules."""


class CapsuleNotFound(CapsuleError):
    """No capsule directory matches the given name/path."""


class CapsuleExists(CapsuleError):
    """A capsule with this name already exists (use --force)."""


class NonLoopbackEndpoint(CapsuleError):
    """Replay endpoint is not loopback and --allow-remote was not given."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_capsules_dir(hermes_home: str | os.PathLike[str] | None = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home) / "capsules"
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home) / "capsules"
    return Path.home() / ".hermes" / "capsules"


def resolve_capsule_dir(
    name_or_path: str, capsules_dir: Path | None = None
) -> Path:
    """Resolve a capsule by explicit path or by name under the capsules dir."""
    p = Path(name_or_path).expanduser()
    if p.is_dir() and (p / MANIFEST_NAME).exists():
        return p
    base = Path(capsules_dir) if capsules_dir else get_capsules_dir()
    cand = base / name_or_path
    if cand.is_dir() and (cand / MANIFEST_NAME).exists():
        return cand
    raise CapsuleNotFound(f"no capsule '{name_or_path}' (looked in {base})")


def list_capsules(capsules_dir: Path | None = None) -> list[dict[str, Any]]:
    base = Path(capsules_dir) if capsules_dir else get_capsules_dir()
    out: list[dict[str, Any]] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        mf = d / MANIFEST_NAME
        if not mf.is_file():
            continue
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {"name": d.name, "manifest_error": True}
        runs = []
        runs_dir = d / RUNS_DIR_NAME
        if runs_dir.is_dir():
            runs = sorted(r.name for r in runs_dir.iterdir() if r.is_dir())
        manifest.setdefault("name", d.name)
        manifest["_runs"] = runs
        manifest["_path"] = str(d)
        out.append(manifest)
    return out


# ---------------------------------------------------------------------------
# Message decoding (mirrors hermes_state._rows_to_conversation, minimal)
# ---------------------------------------------------------------------------

_MESSAGE_COLUMNS_BASE = (
    "id, role, content, tool_call_id, tool_calls, tool_name, finish_reason, "
    "responses_response_id, timestamp, api_content, _compressed_summary"
)
# Optional columns present only in newer stores; absent ones are skipped so
# capsules can be built from older DBs (schema-adaptive, like the Observatory).
_MESSAGE_COLUMNS_OPTIONAL = ("codex_reasoning_items", "codex_message_items")


def _message_select(conn) -> str:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    wanted = [c for c in _MESSAGE_COLUMNS_BASE.split(", ") if c in cols]
    wanted += [c for c in _MESSAGE_COLUMNS_OPTIONAL if c in cols]
    return ", ".join(wanted)


def _decode_content(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return str(raw.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError):
        return str(raw)


def _col(row, key: str, default: Any = None) -> Any:
    """Fetch a column that may be absent in older store schemas."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _row_to_model_message(row) -> dict[str, Any]:
    """Rebuild the model-facing chat message for one stored row.

    Prefers ``api_content`` (the byte-fidelity sidecar) over ``content`` so
    replayed prompts stay byte-stable with what the wrapper originally saw.
    """
    api_content = _col(row, "api_content")
    content = api_content or _decode_content(_col(row, "content"))
    if isinstance(content, bytes):
        content = content.decode("utf-8", "replace")
    msg: dict[str, Any] = {"role": row["role"], "content": content}
    if row["tool_call_id"]:
        msg["tool_call_id"] = row["tool_call_id"]
    if row["tool_name"]:
        msg["tool_name"] = row["tool_name"]
    if row["tool_calls"]:
        try:
            msg["tool_calls"] = json.loads(row["tool_calls"])
        except (json.JSONDecodeError, TypeError):
            msg["tool_calls"] = []
    return msg


def _is_compaction_summary(row, msg: dict[str, Any]) -> bool:
    if _col(row, "_compressed_summary"):
        return True
    content = str(msg.get("content") or "")
    return bool(obs.RE_COMPACTION_PREFIX.match(content))


# ---------------------------------------------------------------------------
# Chat messages -> Responses input items (minimal, deterministic)
# ---------------------------------------------------------------------------


def chat_messages_to_input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored chat messages to Responses API input items.

    Deliberately minimal (no reasoning replay — the capsule stores the
    reasoning item payloads on the recorded turn so a golden chain, which
    already contains them server-side, never needs to resend them).
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            items.append({
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function") or {}
                    items.append({
                        "type": "function_call",
                        "call_id": tc.get("call_id") or tc.get("id") or "",
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
            if content:
                items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or "",
                "output": content,
            })
    return items


def responses_output_to_content(response: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Extract (text, tool_calls) from a Responses API response object."""
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        elif itype == "reasoning":
            pass  # reasoning never counts as visible content
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "call_id": item.get("call_id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
    return "".join(texts), tool_calls


def _finish_from_response(response: dict[str, Any], tool_calls: list[dict]) -> str:
    status = response.get("status")
    if status == "incomplete":
        return "length"
    return "tool_calls" if tool_calls else "stop"


# ---------------------------------------------------------------------------
# Baseline telemetry attribution from main-wrapper logs
# ---------------------------------------------------------------------------


def _attribute_baseline(
    turns: list[dict[str, Any]],
    window: tuple[float, float] | None,
    log_files: list[Path],
    warnings: list[str],
) -> None:
    """Best-effort per-turn baseline from wrapper logs (mutates turns).

    Attributes by response-id anchor, exactly like the Observatory: usage
    and timing lines carry the *generated* response id; continuation /
    rebuild lines carry the *parent* id; MTP rounds inherit the last anchor
    seen in the file. Rotated/overlapping logs are fingerprint-deduped.
    """
    by_response: dict[str, dict[str, Any]] = {}
    by_parent: dict[str, dict[str, Any]] = {}
    for t in turns:
        rid = (t.get("recorded") or {}).get("response_id")
        if rid:
            by_response[rid] = t
        pid = t.get("parent_response_id")
        if pid:
            by_parent.setdefault(pid, t)

    for t in turns:
        t["baseline"] = None

    seen_fp: set[str] = set()
    lo, hi = (window if window else (0.0, time.time() + 3600))
    lo, hi = lo - 300, hi + 300

    def _bump(turn: dict[str, Any]) -> dict[str, Any]:
        if turn["baseline"] is None:
            turn["baseline"] = {
                "prompt_tokens": 0, "generated_tokens": 0, "cached_tokens": 0,
                "prefill_ms": 0.0, "decode_ms": 0.0, "tokenize_ms": 0.0,
                "continuation": None, "rebuild_reason": None,
                "mtp_rounds": 0, "mtp_drafted": 0, "mtp_accepted": 0,
                "hot_cache_reused": 0, "hot_cache_total": 0,
            }
        return turn["baseline"]

    for path in log_files:
        anchor: float | None = None
        anchor_id: str | None = None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    new_anchor = obs._resp_anchor(line, anchor)
                    attributed = new_anchor if new_anchor is not None else anchor
                    if window and (attributed is None or not (lo <= attributed <= hi)):
                        if new_anchor is not None:
                            anchor = new_anchor
                            m = obs.RE_RESP_ID.search(line)
                            anchor_id = m.group(0) if m else anchor_id
                        continue

                    m = obs.RE_CONTINUATION.search(line)
                    if m:
                        turn = by_parent.get(m.group("prev"))
                        if turn is not None:
                            b = _bump(turn)
                            b["continuation"] = "exact"
                            b["continuation_base_tokens"] = int(m.group("base"))
                            b["continuation_prompt_tokens"] = int(m.group("prompt"))
                        continue

                    if "full rebuild" in line:
                        m2 = re.search(r"full rebuild prev=(\S+) reason=(\S+)", line)
                        if m2:
                            turn = by_parent.get(m2.group(1))
                            if turn is not None:
                                b = _bump(turn)
                                if b["continuation"] != "exact":
                                    b["continuation"] = "rebuild"
                                    b["rebuild_reason"] = obs._redact(m2.group(2))
                        continue

                    m = obs.RE_USAGE.search(line)
                    if m:
                        rid = obs.RE_RESP_ID.search(line)
                        turn = by_response.get(rid.group(0)) if rid else None
                        if turn is not None:
                            b = _bump(turn)
                            b["prompt_tokens"] += int(m.group("input"))
                            b["generated_tokens"] += int(m.group("output"))
                            b["cached_tokens"] += int(m.group("cached"))
                        # fall through to timing on the same line

                    timing_pairs: dict[str, float] = {}
                    for tm in obs.RE_TIMING.finditer(line):
                        timing_pairs[tm.group("key")] = float(tm.group("val"))
                    if timing_pairs:
                        rid = obs.RE_RESP_ID.search(line)
                        turn = by_response.get(rid.group(0)) if rid else None
                        if turn is not None:
                            b = _bump(turn)
                            if "prompt_ms" in timing_pairs:
                                b["prefill_ms"] += timing_pairs["prompt_ms"]
                            if "predicted_ms" in timing_pairs:
                                b["decode_ms"] += timing_pairs["predicted_ms"]
                            if "tokenize_ms" in timing_pairs:
                                b["tokenize_ms"] += timing_pairs["tokenize_ms"]

                    m = obs.RE_MTP_ROUND.search(line)
                    if m and anchor_id is not None:
                        turn = by_response.get(anchor_id)
                        if turn is not None:
                            b = _bump(turn)
                            b["mtp_rounds"] += 1
                            b["mtp_drafted"] += len(re.findall(r"\d+", m.group("drafts")))
                            b["mtp_accepted"] += int(m.group("accepted"))

                    if new_anchor is not None:
                        anchor = new_anchor
                        m = obs.RE_RESP_ID.search(line)
                        anchor_id = m.group(0) if m else anchor_id
        except OSError as exc:
            warnings.append(
                f"baseline log unreadable: {path} ({exc.__class__.__name__})"
            )
    _ = seen_fp  # fingerprints reserved for future overlap dedup


# ---------------------------------------------------------------------------
# Per-turn model-facing evidence
# ---------------------------------------------------------------------------


def _blob_key(value: Any) -> str:
    """Content address for an evidence value (instructions str / tools list)."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _blob_value(blobs: dict[str, Any], ref: str | None) -> Any:
    if not ref:
        return None
    blob = blobs.get(ref)
    return blob.get("value") if isinstance(blob, dict) else None


def _resolve_turn_evidence(
    trajectory: dict[str, Any], turn: dict[str, Any]
) -> tuple[str | None, list[dict] | None, dict[str, Any]]:
    """Resolve (instructions, tools, meta) for replaying ONE recorded call.

    Schema v2 capsules carry per-turn evidence: each turn's instructions
    and tool definitions are exactly what that model call was sent (or
    explicitly absent if its response rotated out before capture). There
    is no cross-turn fallback — using turn 0's tool list for turn 3 would
    be a fabrication, and tool availability genuinely changes between
    calls (conscience-driven narrowing).

    Legacy schema v1 capsules have one global evidence set and no
    per-turn ``evidence`` key; they keep replaying with it (meta mode
    ``global``) so existing capsules stay usable.
    """
    ev = turn.get("evidence")
    if ev is None:  # legacy v1 capsule
        instr = trajectory.get("instructions") or None
        tools = trajectory.get("tools") or None
        return instr, tools, {
            "mode": "global",
            "instructions": "global" if instr else "none",
            "tools": "global" if tools else "none",
            "instructions_source": trajectory.get("instructions_source"),
            "tools_source": trajectory.get("tools_source"),
        }
    blobs = trajectory.get("evidence_blobs") or {}
    instr = _blob_value(blobs, ev.get("instructions_ref"))
    tools = _blob_value(blobs, ev.get("tools_ref"))
    # When missing, distinguish a capture GAP ("none") from faithful absence
    # the store itself proves ("absent-by-evidence": the wrapper answered
    # for this exact call and its record carries no such field, i.e. the
    # original request omitted it — replaying without it is correct).
    instr_state = (
        "per-turn" if instr
        else "absent-by-evidence" if ev.get("instructions_evidence") == "absent-in-store"
        else "none"
    )
    tools_state = (
        "per-turn" if tools
        else "absent-by-evidence" if ev.get("tools_evidence") == "absent-in-store"
        else "none"
    )
    return instr, tools, {
        "mode": "per-turn",
        "instructions": instr_state,
        "tools": tools_state,
        "instructions_source": ev.get("instructions_source"),
        "tools_source": ev.get("tools_source"),
        "evidence_response_id": ev.get("evidence_response_id"),
    }


def _fmt_idx(idxs: list[int]) -> str:
    head = ", ".join(str(i) for i in idxs[:5])
    return head + (f" (+{len(idxs) - 5} more)" if len(idxs) > 5 else "")


# ---------------------------------------------------------------------------
# Capsule creation
# ---------------------------------------------------------------------------


def create_capsule(
    session_ident: str,
    *,
    hermes_home: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    capsules_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
    main_log_files: list[Path] | None = None,
    instructions_file: str | os.PathLike[str] | None = None,
    hydrate_endpoint: str | None = None,
    hydrate_api_key: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze one completed session's model-facing trajectory into a capsule."""
    db = Path(db_path) if db_path else obs.get_state_db_path(hermes_home)
    conn = obs.open_state_db_readonly(db)
    try:
        session_id = obs.resolve_session_id(conn, session_ident)
        srow = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        rows = conn.execute(
            f"SELECT {_message_select(conn)} FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    warnings: list[str] = []
    if not rows:
        warnings.append("session has no stored messages — capsule is empty")

    # ---- build turns -----------------------------------------------------
    turns: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    last_anchor_id: str | None = None
    rebase = False
    lineage_total = 0
    lineage_present = 0

    for row in rows:
        msg = _row_to_model_message(row)
        if msg["role"] == "system":
            continue  # instructions handled separately
        if msg["role"] == "user" and _is_compaction_summary(row, msg):
            rebase = True
            pending.append(msg)
            continue
        if msg["role"] == "assistant":
            lineage_total += 1
            rid = row["responses_response_id"] or None
            if rid:
                lineage_present += 1
            recorded = {
                "response_id": rid,
                "finish_reason": row["finish_reason"] or None,
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or None,
                "timestamp": row["timestamp"],
            }
            turns.append({
                "index": len(turns),
                "parent_response_id": None if (rebase or last_anchor_id is None) else last_anchor_id,
                "delta": pending,
                "full": transcript + pending,
                "rebase": rebase,
                "recorded": recorded,
            })
            # After this turn, the chain anchor is this response (if known).
            # An unknown id is a lineage gap: the next turn cannot chain.
            last_anchor_id = rid
            rebase = False
            pending = []
            transcript = transcript + turns[-1]["delta"] + [
                {k: v for k, v in msg.items() if not k.startswith("_")}
            ]
            if not rid:
                warnings.append(
                    f"turn {turns[-1]['index']}: assistant row has no "
                    "responses_response_id — stateful chaining impossible "
                    "from this point (older/stateless session?)"
                )
            if turns[-1]["rebase"]:
                warnings.append(
                    f"turn {turns[-1]['index']}: compaction rebase — replay "
                    "resends the full stored transcript, not the exact "
                    "compacted projection the original request carried"
                )
        elif msg["role"] in ("tool", "user"):
            # Regular user turns join the pending delta; compaction summaries
            # were already handled above (they force a chain rebase).
            pending.append(msg)

    if pending:
        warnings.append(
            f"{len(pending)} trailing message(s) after the last assistant "
            "response were never answered — not captured as a turn"
        )

    if lineage_total and lineage_present == 0:
        warnings.append(
            "no stateful response lineage in this session at all — replay "
            "can only run in 'full' (stateless) mode"
        )

    # ---- per-turn model-facing evidence -----------------------------------
    # The wrapper's response store keeps the exact instructions and tool
    # definitions each model call was sent. They are NOT byte-stable across
    # calls (tool availability and transient policy change per call — e.g.
    # conscience-driven tool narrowing), so hydrate EACH turn from its OWN
    # recorded response id. Values are content-addressed into
    # ``evidence_blobs``; turns whose response has rotated out keep an
    # explicit gap and are never backfilled from a neighbour's evidence.
    file_instructions: str | None = None
    if instructions_file:
        try:
            file_instructions = Path(instructions_file).read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(f"instructions file unreadable: {exc}")

    evidence_blobs: dict[str, dict[str, Any]] = {}

    def _put_blob(kind: str, value: Any) -> str:
        key = _blob_key(value)
        evidence_blobs.setdefault(key, {"kind": kind, "value": value})
        return key

    # Default the hydration source to the endpoint the session actually
    # used (loopback only; an explicit --hydrate-endpoint is the user's own
    # call). A manual --instructions-file still wins for instructions.
    hydrate_base = (hydrate_endpoint or "").strip().rstrip("/")
    if not hydrate_base:
        recorded = str(
            (srow["billing_base_url"] if "billing_base_url" in srow.keys() else "")
            or ""
        ).strip().rstrip("/")
        if recorded and _endpoint_is_loopback(recorded):
            hydrate_base = recorded
    if hydrate_base and not hydrate_base.endswith("/v1"):
        hydrate_base += "/v1"
    hydrate_key = (
        hydrate_api_key or os.environ.get("HERMES_REPLAY_API_KEY") or None
    )

    endpoint_dead = False
    last_error: Exception | None = None
    for t in turns:
        own_rid = t["recorded"]["response_id"]
        ev: dict[str, Any] = {
            "instructions_ref": None, "instructions_source": None,
            "tools_ref": None, "tools_source": None,
            "instructions_evidence": "gap", "tools_evidence": "gap",
            "evidence_response_id": None, "fetch": None, "status": None,
        }
        if file_instructions is not None:
            ev["instructions_ref"] = _put_blob("instructions", file_instructions)
            ev["instructions_source"] = "file"
            ev["instructions_evidence"] = "captured"
        if not hydrate_base:
            ev["fetch"] = "no-source"
        elif not own_rid:
            ev["fetch"] = "no-response-id"
        elif endpoint_dead:
            ev["fetch"] = "endpoint-unreachable"
        else:
            try:
                body = _http_get_json(
                    f"{hydrate_base}/responses/{own_rid}", hydrate_key, timeout=5.0
                )
                ev["evidence_response_id"] = own_rid
                if (
                    ev["instructions_ref"] is None
                    and isinstance(body.get("instructions"), str)
                    and body["instructions"]
                ):
                    ev["instructions_ref"] = _put_blob(
                        "instructions", body["instructions"]
                    )
                    ev["instructions_source"] = f"wrapper-hydrated:{own_rid}"
                if isinstance(body.get("tools"), list) and body["tools"]:
                    ev["tools_ref"] = _put_blob("tools", body["tools"])
                    ev["tools_source"] = f"wrapper-hydrated:{own_rid}"
                ev["fetch"] = "ok"
                # The store ANSWERED for this exact call: a field it doesn't
                # carry is positive evidence the original request omitted it
                # (faithful absence), not a capture gap.
                if ev["instructions_evidence"] != "captured":
                    ev["instructions_evidence"] = (
                        "absent-in-store"
                        if "instructions" not in body or not body["instructions"]
                        else "captured"
                    )
                ev["tools_evidence"] = (
                    "captured" if ev["tools_ref"] else "absent-in-store"
                )
            except urllib.error.HTTPError as exc:
                # 404 = the store rotated this response out; other codes =
                # the store answered but failed. Both leave an explicit gap.
                ev["fetch"] = "rotated" if exc.code == 404 else "store-error"
            except (CapsuleError, OSError, urllib.error.URLError) as exc:
                last_error = exc
                ev["fetch"] = "endpoint-unreachable"
                endpoint_dead = True  # don't re-probe a dead endpoint per turn
        ev["status"] = (
            "complete" if (ev["instructions_ref"] and ev["tools_ref"])
            else "partial" if (ev["instructions_ref"] or ev["tools_ref"])
            else "missing"
        )
        t["evidence"] = ev

    n_instr_turns = sum(1 for t in turns if t["evidence"]["instructions_ref"])
    n_tools_turns = sum(1 for t in turns if t["evidence"]["tools_ref"])
    if turns:
        if (
            hydrate_base
            and all(t["evidence"]["fetch"] == "endpoint-unreachable" for t in turns)
            and last_error is not None
        ):
            warnings.append(
                f"instruction/tool hydration failed ({last_error.__class__.__name__}) — "
                "no turn has model-facing evidence; full-rebuild replays will lack "
                "the system prompt and tool definitions"
            )
        else:
            rotated = [t["index"] for t in turns if t["evidence"]["fetch"] == "rotated"]
            if rotated:
                warnings.append(
                    f"turn(s) {_fmt_idx(rotated)}: response rotated out of the wrapper "
                    "store — that call's instructions/tool definitions are unavailable "
                    "and are NOT backfilled from another turn's evidence"
                )
            store_err = [t["index"] for t in turns if t["evidence"]["fetch"] == "store-error"]
            if store_err:
                warnings.append(
                    f"turn(s) {_fmt_idx(store_err)}: wrapper store errored on fetch — "
                    "per-turn evidence unavailable for those calls"
                )
            noid = [
                t["index"] for t in turns
                if t["evidence"]["fetch"] == "no-response-id"
            ]
            if noid and hydrate_base:
                warnings.append(
                    f"turn(s) {_fmt_idx(noid)}: no recorded response id — per-turn "
                    "evidence impossible for those calls"
                )
        tc_gap = [
            t["index"] for t in turns
            if t["recorded"]["tool_calls"]
            and t["evidence"]["tools_evidence"] == "gap"
        ]
        if tc_gap:
            warnings.append(
                f"turn(s) {_fmt_idx(tc_gap)} recorded tool calls but their tool "
                "definitions were not captured — replays of those turns may diverge "
                "without them"
            )

    # ---- baseline telemetry ---------------------------------------------
    started = srow["started_at"] if "started_at" in srow.keys() else None
    ended = srow["ended_at"] if "ended_at" in srow.keys() else None
    window: tuple[float, float] | None = None
    if started:
        window = (float(started), float(ended) if ended else time.time())
    else:
        warnings.append("session row has no started_at — baseline attribution skipped")

    log_files = main_log_files
    if log_files is None:
        log_files = obs.discover_log_files("main") if window else []
    if window and log_files:
        _attribute_baseline(turns, window, log_files, warnings)
    elif window:
        warnings.append(
            "no main-wrapper log files found — no baseline telemetry; "
            "comparisons will report 'unverified', never 'exact'"
        )

    baseline_coverage = sum(1 for t in turns if t.get("baseline"))
    if turns and baseline_coverage < len(turns):
        warnings.append(
            f"baseline telemetry covers {baseline_coverage}/{len(turns)} turns "
            "(rotated logs, truncation, or non-stateful turns)"
        )

    # ---- write capsule ----------------------------------------------------
    cap_name = name or session_id
    base = Path(capsules_dir) if capsules_dir else get_capsules_dir(hermes_home)
    cap_dir = base / cap_name
    if (cap_dir / MANIFEST_NAME).exists() and not force:
        raise CapsuleExists(f"capsule '{cap_name}' exists at {cap_dir} — use --force")

    trajectory = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "session_id": session_id,
        # Legacy compat view: global fields are only meaningful for a
        # manual --instructions-file; per-turn evidence lives in
        # turns[].evidence + evidence_blobs.
        "instructions": file_instructions,
        "instructions_source": (
            "file" if file_instructions is not None
            else "per-turn" if n_instr_turns
            else "unavailable"
        ),
        "tools": None,
        "tools_source": "per-turn" if n_tools_turns else "unavailable",
        "evidence_blobs": evidence_blobs,
        "turns": turns,
    }
    traj_bytes = json.dumps(trajectory, ensure_ascii=False).encode("utf-8")
    manifest = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "name": cap_name,
        "created_at": time.time(),
        "created_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "source": srow["source"],
        "model": srow["model"],
        "billing_base_url": srow["billing_base_url"] if "billing_base_url" in srow.keys() else None,
        "system_prompt_hash": srow["system_prompt_hash"] if "system_prompt_hash" in srow.keys() else None,
        "session_started_at": srow["started_at"],
        "session_ended_at": srow["ended_at"],
        "turns": len(turns),
        "lineage": {
            "assistant_turns": lineage_total,
            "with_response_id": lineage_present,
            "coverage": round(lineage_present / lineage_total, 4) if lineage_total else None,
            "compaction_rebases": sum(1 for t in turns if t["rebase"]),
        },
        "instructions_captured": bool(n_instr_turns),
        "instructions_source": trajectory["instructions_source"],
        "tools_captured": bool(n_tools_turns),
        "tools_source": trajectory["tools_source"],
        "evidence_coverage": {
            "turns_total": len(turns),
            "instructions_turns": n_instr_turns,
            "tools_turns": n_tools_turns,
            "complete_turns": sum(
                1 for t in turns if t["evidence"]["status"] == "complete"
            ),
        },
        "baseline_turns_covered": baseline_coverage,
        "trajectory_sha256": hashlib.sha256(traj_bytes).hexdigest(),
        "privacy": {
            "content_stored_locally": True,
            "note": (
                "trajectory contains prompts, tool arguments and recorded tool "
                "output verbatim — treat the capsule dir like your session DB"
            ),
        },
        "warnings": warnings,
    }

    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / RUNS_DIR_NAME).mkdir(exist_ok=True)
    (cap_dir / TRAJECTORY_NAME).write_bytes(traj_bytes)
    (cap_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["_path"] = str(cap_dir)
    return manifest


def load_capsule(name_or_path: str, capsules_dir: Path | None = None) -> dict[str, Any]:
    cap_dir = resolve_capsule_dir(name_or_path, capsules_dir)
    try:
        manifest = json.loads((cap_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        trajectory = json.loads((cap_dir / TRAJECTORY_NAME).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CapsuleError(f"capsule at {cap_dir} is corrupt: {exc}") from exc
    stored_hash = manifest.get("trajectory_sha256")
    if stored_hash:
        actual = hashlib.sha256((cap_dir / TRAJECTORY_NAME).read_bytes()).hexdigest()
        if actual != stored_hash:
            manifest.setdefault("warnings", []).append(
                "trajectory.json does not match the manifest hash — it was "
                "modified after creation"
            )
    return {"dir": cap_dir, "manifest": manifest, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def _endpoint_is_loopback(endpoint: str) -> bool:
    m = re.match(r"https?://([^:/]+)", endpoint)
    host = m.group(1).lower() if m else ""
    return host in {"127.0.0.1", "localhost", "::1", "[::1]"}


def _require_loopback(endpoint: str, allow_remote: bool) -> None:
    if _endpoint_is_loopback(endpoint):
        return
    if not allow_remote:
        m = re.match(r"https?://([^:/]+)", endpoint)
        host = m.group(1).lower() if m else ""
        raise NonLoopbackEndpoint(
            f"endpoint host '{host}' is not loopback — capsule content would "
            "leave this machine; pass --allow-remote to accept"
        )


def _http_get_json(url: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_post_json(
    url: str, body: dict[str, Any], api_key: str | None, timeout: float
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error_text": raw[:2000]}
        return exc.code, parsed


def _is_chain_break_error(code: int, payload: dict[str, Any]) -> bool:
    if code != 404 and not (code == 400):
        return False
    text = json.dumps(payload).lower()
    if "previous_response_id" in text or "previous response" in text:
        return any(
            marker in text
            for marker in ("not found", "unknown", "404", "poison")
        )
    return False


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_capsule(
    name_or_path: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    chain: str = "auto",
    max_turns: int | None = None,
    timeout: float = 300.0,
    allow_remote: bool = False,
    main_log_files: list[Path] | None = None,
    capsules_dir: Path | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Re-drive a capsule's turns against a wrapper. Tools are never re-run."""
    cap = load_capsule(name_or_path, capsules_dir)
    manifest, trajectory = cap["manifest"], cap["trajectory"]
    turns = trajectory["turns"]
    if not turns:
        raise CapsuleError("capsule has no turns to replay")

    endpoint = (endpoint or manifest.get("billing_base_url") or "").rstrip("/")
    if not endpoint:
        raise CapsuleError(
            "no replay endpoint given and the session recorded none — "
            "pass --endpoint http://127.0.0.1:PORT/v1"
        )
    if not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"
    _require_loopback(endpoint, allow_remote)

    if chain == "auto":
        cov = (manifest.get("lineage") or {}).get("coverage") or 0
        chain = "golden" if cov >= 1.0 else "full"

    api_key = api_key or os.environ.get("HERMES_REPLAY_API_KEY") or None

    run_started = time.time()
    replay_turns: list[dict[str, Any]] = []
    generated_ids: list[str | None] = []
    selected = turns[: max_turns or len(turns)]

    for t in selected:
        recorded = t["recorded"]
        if chain == "golden":
            parent = t["parent_response_id"]
            use_full = parent is None and t["index"] != 0 and not t["rebase"]
            if t["rebase"] or t["index"] == 0:
                parent = None
                use_full = t["index"] != 0  # rebase turn sends compacted projection
        elif chain == "replay":
            parent = generated_ids[-1] if generated_ids else None
            use_full = parent is None and t["index"] != 0
            if t["index"] == 0 or t["rebase"]:
                parent = None
                use_full = t["index"] != 0
        else:  # full
            parent = None
            use_full = True

        input_msgs = t["full"] if use_full else t["delta"]
        # Per-turn model-facing evidence: exactly what THIS recorded call was
        # sent (schema v2), or the capsule's global set (legacy v1 compat).
        # No cross-turn fallback — tool availability genuinely changes
        # between calls, so a neighbour's evidence would be a fabrication.
        turn_instructions, turn_tools, ev_meta = _resolve_turn_evidence(
            trajectory, t
        )
        body: dict[str, Any] = {
            "model": manifest.get("model"),
            "stream": False,
            "input": chat_messages_to_input_items(input_msgs),
        }
        if turn_instructions:
            body["instructions"] = turn_instructions
        if turn_tools:
            # Golden/replay chains already carry the evidence server-side
            # (resending what the original request sent is what the original
            # did); full-rebuild fallbacks NEED it — without tool
            # declarations the model cannot reproduce a recorded tool call.
            body["tools"] = turn_tools
        if parent:
            body["previous_response_id"] = parent
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        entry: dict[str, Any] = {
            "index": t["index"],
            "requested_parent": parent,
            "mode": "full" if use_full else "chain",
            "degraded_full": False,
            "chain_break": False,
            "evidence": ev_meta,
        }
        t0 = time.monotonic()
        try:
            code, resp = _http_post_json(
                f"{endpoint}/responses", body, api_key, timeout
            )
            if code >= 400 and parent and _is_chain_break_error(code, resp):
                entry["chain_break"] = True
                entry["chain_break_detail"] = json.dumps(resp)[:300]
                # Honest fallback: resend the whole projection, unchained.
                body_f = dict(body)
                body_f["previous_response_id"] = None
                body_f["input"] = chat_messages_to_input_items(t["full"])
                entry["degraded_full"] = True
                entry["mode"] = "full"
                code, resp = _http_post_json(
                    f"{endpoint}/responses", body_f, api_key, timeout
                )
            latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
            entry["http_status"] = code
            if code >= 400:
                entry["error"] = json.dumps(resp)[:500]
                generated_ids.append(None)
            else:
                text, tool_calls = responses_output_to_content(resp)
                usage = resp.get("usage") or {}
                entry.update({
                    "response_id": resp.get("id"),
                    "finish_reason": _finish_from_response(resp, tool_calls),
                    "status": resp.get("status"),
                    "content": text,
                    "tool_calls": tool_calls or None,
                    "latency_ms": latency_ms,
                    "usage": {
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "cached_tokens": (
                            (usage.get("input_tokens_details") or {}).get("cached_tokens")
                        ),
                        "reasoning_tokens": (
                            (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
                        ),
                    },
                })
                generated_ids.append(resp.get("id"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            entry["http_status"] = None
            entry["error"] = f"{exc.__class__.__name__}: {exc}"[:500]
            entry["latency_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
            generated_ids.append(None)
        replay_turns.append(entry)

    run_finished = time.time()

    # ---- post-run telemetry: attribute wrapper log lines to replay ids ----
    warnings: list[str] = []
    log_files = main_log_files if main_log_files is not None else obs.discover_log_files("main")
    if log_files:
        _attribute_replay_telemetry(
            replay_turns, turns, log_files,
            (run_started, run_finished), warnings,
        )
    else:
        warnings.append(
            "no main-wrapper logs found — continuation/MTP telemetry for "
            "this run is unavailable (usage numbers are response-body only)"
        )

    run = {
        "schema_version": RUN_SCHEMA_VERSION,
        # second-resolution + entropy: two runs in the same second must not
        # clobber each other's directory
        "run_id": (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + os.urandom(2).hex()
        ),
        "capsule": manifest.get("name"),
        "session_id": manifest.get("session_id"),
        "endpoint": endpoint,
        "chain_mode": chain,
        "turns_requested": len(selected),
        "turns_completed": sum(1 for e in replay_turns if "response_id" in e),
        "turns_failed": sum(1 for e in replay_turns if "error" in e),
        "chain_breaks": sum(1 for e in replay_turns if e.get("chain_break")),
        "degraded_full_turns": sum(1 for e in replay_turns if e.get("degraded_full")),
        "started_at": run_started,
        "finished_at": run_finished,
        "wall_seconds": round(run_finished - run_started, 2),
        "turns": replay_turns,
        "warnings": warnings,
    }

    runs_dir = Path(cap["dir"]) / RUNS_DIR_NAME / run["run_id"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / RUN_NAME).write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run["_path"] = str(runs_dir / RUN_NAME)
    return run


def _attribute_replay_telemetry(
    replay_turns: list[dict[str, Any]],
    capsule_turns: list[dict[str, Any]],
    log_files: list[Path],
    window: tuple[float, float],
    warnings: list[str],
) -> None:
    """Attach continuation/MTP telemetry to replay turns by response id."""
    by_generated: dict[str, dict[str, Any]] = {}
    by_parent: dict[str, dict[str, Any]] = {}
    for e in replay_turns:
        if e.get("response_id"):
            by_generated[e["response_id"]] = e
        if e.get("requested_parent"):
            by_parent.setdefault(e["requested_parent"], e)

    for e in replay_turns:
        e["telemetry"] = {
            "continuation": None, "rebuild_reason": None,
            "mtp_rounds": 0, "mtp_drafted": 0, "mtp_accepted": 0,
        }

    lo, hi = window[0] - 5, window[1] + 60
    for path in log_files:
        anchor_id: str | None = None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = obs.RE_CONTINUATION.search(line)
                    if m:
                        e = by_parent.get(m.group("prev"))
                        if e is not None:
                            e["telemetry"]["continuation"] = "exact"
                            e["telemetry"]["continuation_base_tokens"] = int(m.group("base"))
                            e["telemetry"]["continuation_prompt_tokens"] = int(m.group("prompt"))
                        continue
                    if "full rebuild" in line:
                        m2 = re.search(r"full rebuild prev=(\S+) reason=(\S+)", line)
                        if m2:
                            e = by_parent.get(m2.group(1))
                            if e is not None and e["telemetry"]["continuation"] != "exact":
                                e["telemetry"]["continuation"] = "rebuild"
                                e["telemetry"]["rebuild_reason"] = obs._redact(m2.group(2))
                        continue
                    m = obs.RE_MTP_ROUND.search(line)
                    if m and anchor_id is not None:
                        e = by_generated.get(anchor_id)
                        if e is not None:
                            e["telemetry"]["mtp_rounds"] += 1
                            e["telemetry"]["mtp_drafted"] += len(
                                re.findall(r"\d+", m.group("drafts"))
                            )
                            e["telemetry"]["mtp_accepted"] += int(m.group("accepted"))
                    rid = obs.RE_RESP_ID.search(line)
                    if rid:
                        anchor_id = rid.group(0)
        except OSError as exc:
            warnings.append(f"replay telemetry log unreadable: {path} ({exc.__class__.__name__})")


def load_run(capsule_dir: Path, run_id: str) -> dict[str, Any]:
    p = Path(capsule_dir) / RUNS_DIR_NAME / run_id / RUN_NAME
    if not p.is_file():
        raise CapsuleNotFound(f"no run '{run_id}' under {capsule_dir / RUNS_DIR_NAME}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def _tool_calls_key(tcs: list[dict] | None) -> list[tuple[str, str]]:
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        out.append((fn.get("name", ""), fn.get("arguments", "")))
    return out


def compare_run(
    name_or_path: str,
    *,
    run_id: str | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare the newest (or named) replay run against the capsule baseline."""
    cap = load_capsule(name_or_path, capsules_dir)
    cap_dir = Path(cap["dir"])
    runs_root = cap_dir / RUNS_DIR_NAME
    if run_id is None:
        available = sorted(
            d.name for d in runs_root.iterdir() if d.is_dir()
        ) if runs_root.is_dir() else []
        if not available:
            raise CapsuleError("capsule has no replay runs yet — run "
                               "'hermes sessions capsule replay' first")
        run_id = available[-1]
    run = load_run(cap_dir, run_id)
    baseline_turns = {t["index"]: t for t in cap["trajectory"]["turns"]}

    per_turn: list[dict[str, Any]] = []
    missing_evidence: list[str] = []
    n_identical = n_similar = n_diverged = n_failed = 0
    for e in run["turns"]:
        bt = baseline_turns.get(e["index"], {})
        rec = bt.get("recorded", {})
        base = bt.get("baseline")
        row: dict[str, Any] = {"index": e["index"]}
        if "error" in e:
            row["behavior"] = "failed"
            row["error"] = e["error"]
            n_failed += 1
            per_turn.append(row)
            continue
        finish_match = (rec.get("finish_reason") or "stop") == e.get("finish_reason")
        tc_match = _tool_calls_key(rec.get("tool_calls")) == _tool_calls_key(e.get("tool_calls"))
        sim = _similarity(rec.get("content") or "", e.get("content") or "")
        if finish_match and tc_match and sim >= 0.999:
            behavior = "identical"
            n_identical += 1
        elif finish_match and tc_match and sim >= 0.8:
            behavior = "similar"
            n_similar += 1
        else:
            behavior = "diverged"
            n_diverged += 1
        row.update({
            "behavior": behavior,
            "finish_reason": {"recorded": rec.get("finish_reason"), "replayed": e.get("finish_reason")},
            "tool_calls_match": tc_match,
            "content_similarity": sim,
            "replay_latency_ms": e.get("latency_ms"),
            "replay_usage": e.get("usage"),
            "replay_continuation": (e.get("telemetry") or {}).get("continuation"),
            "replay_mtp": {
                k: (e.get("telemetry") or {}).get(k)
                for k in ("mtp_rounds", "mtp_drafted", "mtp_accepted")
            },
        })
        if base:
            row["baseline"] = {
                "prefill_ms": base.get("prefill_ms"),
                "decode_ms": base.get("decode_ms"),
                "inference_ms": round((base.get("prefill_ms") or 0) + (base.get("decode_ms") or 0), 1),
                "prompt_tokens": base.get("prompt_tokens"),
                "cached_tokens": base.get("cached_tokens"),
                "continuation": base.get("continuation"),
                "mtp_acceptance": (
                    round(base["mtp_accepted"] / base["mtp_drafted"], 4)
                    if base.get("mtp_drafted") else None
                ),
            }
            if e.get("latency_ms") is not None:
                row["latency_delta_ms"] = round(
                    e["latency_ms"] - row["baseline"]["inference_ms"], 1
                )
        else:
            row["baseline"] = None
            missing_evidence.append(f"turn {e['index']}: no baseline telemetry")
        if (
            (e.get("telemetry") or {}).get("continuation") is None
            and e.get("mode") == "chain"
            and e.get("requested_parent")
        ):
            missing_evidence.append(
                f"turn {e['index']}: no wrapper-log continuation telemetry for this replay"
            )
        # Per-turn model-facing evidence gaps (schema v2 capsules only —
        # legacy v1 capsules get one global reason below, preserving their
        # existing verdict behaviour). 'absent-by-evidence' is NOT a gap:
        # the wrapper store proved that call was sent without the field.
        ev_meta = e.get("evidence") or {}
        row["evidence"] = ev_meta or None
        if ev_meta.get("mode") == "per-turn":
            if ev_meta.get("instructions") == "none":
                missing_evidence.append(
                    f"turn {e['index']}: instructions (system prompt) for this call "
                    "were not captured — replayed without them (never backfilled "
                    "from another turn's evidence)"
                )
            if ev_meta.get("tools") == "none":
                if rec.get("tool_calls"):
                    missing_evidence.append(
                        f"turn {e['index']}: tool definitions were not captured for "
                        "this call — the recorded tool call replayed without them"
                    )
                else:
                    missing_evidence.append(
                        f"turn {e['index']}: tool definitions for this call were not "
                        "captured"
                    )
        per_turn.append(row)

    # Chain continuity summary (turn-0 / rebase turns have no parent to
    # continue from — they are not evidence for or against continuity)
    chained = [
        e for e in run["turns"]
        if e.get("mode") == "chain" and e.get("requested_parent")
    ]
    exact = sum(1 for e in chained if (e.get("telemetry") or {}).get("continuation") == "exact")
    rebuilds = [e for e in chained if (e.get("telemetry") or {}).get("continuation") == "rebuild"]

    reasons: list[str] = []
    # Verdict — never silently claim exact.
    if n_failed:
        verdict = "failed"
    elif n_diverged:
        verdict = "diverged"
    elif n_similar:
        verdict = "similar"
    elif run.get("chain_mode") == "full":
        # Behaviour matches, but lineage/continuity was never exercised.
        # Checked before missing_evidence so its explanation always lands.
        verdict = "equivalent-unverified"
        reasons.append(
            "stateless full replay: cache continuity and response lineage "
            "were not exercised, so 'exact' cannot be claimed"
        )
    elif missing_evidence:
        verdict = "equivalent-unverified"
    elif chained and exact < len(chained):
        verdict = "equivalent-rebuilt"
    elif run.get("chain_breaks"):
        # Original parents were gone: continuity was NOT preserved.
        verdict = "equivalent-rebuilt"
    else:
        verdict = "exact"

    if run.get("chain_breaks"):
        reasons.append(
            f"{run['chain_breaks']} chain break(s): original parent ids were "
            "rejected (rotated response store?) — those turns fell back to "
            "full-prompt replay"
        )
    if rebuilds:
        reasons.append(
            f"{len(rebuilds)} chained turn(s) took a full-rebuild path "
            "(cache-continuity regression vs the recording)"
        )
    evidence_modes = {(e.get("evidence") or {}).get("mode") for e in run["turns"]}
    if evidence_modes == {"global"}:
        # Legacy v1 capsule (one global evidence set for all turns): keep the
        # original global reasons and their verdict semantics. Schema v2
        # capsules report evidence gaps per turn instead, via missing_evidence.
        if not cap["trajectory"].get("instructions"):
            reasons.append(
                "instructions (system prompt) were not captured — full-rebuild "
                "turns ran without them, behaviour may legitimately differ")
        if not cap["trajectory"].get("tools") and any(
            (t.get("recorded") or {}).get("tool_calls") for t in cap["trajectory"]["turns"]
        ):
            reasons.append(
                "tool definitions were not captured — turns that recorded tool "
                "calls replay without them on full-rebuild fallbacks")
    if missing_evidence:
        reasons.extend(missing_evidence[:10])

    lat = [e.get("latency_ms") for e in run["turns"] if e.get("latency_ms")]
    mtp_drafted = sum((e.get("telemetry") or {}).get("mtp_drafted", 0) for e in run["turns"])
    mtp_accepted = sum((e.get("telemetry") or {}).get("mtp_accepted", 0) for e in run["turns"])
    base_drafted = sum((bt.get("baseline") or {}).get("mtp_drafted", 0) for bt in baseline_turns.values())
    base_accepted = sum((bt.get("baseline") or {}).get("mtp_accepted", 0) for bt in baseline_turns.values())

    return {
        "capsule": cap["manifest"].get("name"),
        "run_id": run_id,
        "session_id": cap["manifest"].get("session_id"),
        "chain_mode": run.get("chain_mode"),
        "verdict": verdict,
        "reasons": reasons,
        "behavior": {
            "identical": n_identical, "similar": n_similar,
            "diverged": n_diverged, "failed": n_failed,
        },
        "chain_continuity": {
            "chained_turns": len(chained),
            "exact_continuations": exact,
            "full_rebuilds": len(rebuilds),
            "rebuild_reasons": sorted({
                (e.get("telemetry") or {}).get("rebuild_reason") or "?"
                for e in rebuilds
            }),
            "chain_breaks": run.get("chain_breaks", 0),
        },
        "latency": {
            "replay_total_ms": round(sum(lat), 1) if lat else None,
            "replay_median_ms": sorted(lat)[len(lat) // 2] if lat else None,
        },
        "mtp": {
            "baseline_acceptance": (
                round(base_accepted / base_drafted, 4) if base_drafted else None
            ),
            "replay_acceptance": (
                round(mtp_accepted / mtp_drafted, 4) if mtp_drafted else None
            ),
        },
        "turns": per_turn,
    }


def compare_runs(
    name_or_path: str,
    run_a: str,
    run_b: str,
    *,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    """A/B two replay runs of the same capsule (e.g. two wrapper builds)."""
    cap = load_capsule(name_or_path, capsules_dir)
    a = load_run(Path(cap["dir"]), run_a)
    b = load_run(Path(cap["dir"]), run_b)
    ta = {e["index"]: e for e in a["turns"]}
    tb = {e["index"]: e for e in b["turns"]}
    rows: list[dict[str, Any]] = []
    for idx in sorted(set(ta) & set(tb)):
        ea, eb = ta[idx], tb[idx]
        sim = _similarity(ea.get("content") or "", eb.get("content") or "")
        row = {
            "index": idx,
            "content_similarity": sim,
            "tool_calls_match": _tool_calls_key(ea.get("tool_calls")) == _tool_calls_key(eb.get("tool_calls")),
            "finish_match": ea.get("finish_reason") == eb.get("finish_reason"),
            "a_latency_ms": ea.get("latency_ms"),
            "b_latency_ms": eb.get("latency_ms"),
        }
        if row["a_latency_ms"] is not None and row["b_latency_ms"] is not None:
            row["latency_delta_ms"] = round(eb["latency_ms"] - ea["latency_ms"], 1)
        row["a_continuation"] = (ea.get("telemetry") or {}).get("continuation")
        row["b_continuation"] = (eb.get("telemetry") or {}).get("continuation")
        rows.append(row)
    lat_a = [r["a_latency_ms"] for r in rows if r.get("a_latency_ms")]
    lat_b = [r["b_latency_ms"] for r in rows if r.get("b_latency_ms")]
    return {
        "capsule": cap["manifest"].get("name"),
        "run_a": run_a, "run_b": run_b,
        "a_chain_mode": a.get("chain_mode"), "b_chain_mode": b.get("chain_mode"),
        "turns_compared": len(rows),
        "turns_diverged": sum(1 for r in rows if r["content_similarity"] < 0.8 or not r["tool_calls_match"]),
        "total_latency_a_ms": round(sum(lat_a), 1) if lat_a else None,
        "total_latency_b_ms": round(sum(lat_b), 1) if lat_b else None,
        "turns": rows,
    }


# ---------------------------------------------------------------------------
# Rendering (terminal-friendly, no markdown)
# ---------------------------------------------------------------------------


def _fmt(v: Any, suffix: str = "") -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v == int(v):
            return f"{int(v)}{suffix}"
        return f"{v:.2f}{suffix}"
    return "n/a"


def format_manifest(manifest: dict[str, Any]) -> str:
    lin = manifest.get("lineage", {})
    L = []
    L.append(f"CAPSULE — {manifest.get('name')}")
    L.append(f"  session {manifest.get('session_id')}  source={manifest.get('source')}")
    L.append(f"  model   {manifest.get('model')}")
    L.append(f"  created {manifest.get('created_at_iso')}  turns {manifest.get('turns')}")
    L.append(f"  lineage coverage {_fmt(lin.get('coverage'))} "
             f"({_fmt(lin.get('with_response_id'))}/{_fmt(lin.get('assistant_turns'))} assistant turns)"
             f"  compaction rebases {_fmt(lin.get('compaction_rebases'))}")
    cov = manifest.get("evidence_coverage")
    if cov:
        L.append(
            f"  model-facing evidence (per call, from the wrapper response store): "
            f"instructions {cov['instructions_turns']}/{cov['turns_total']} turns, "
            f"tool definitions {cov['tools_turns']}/{cov['turns_total']} turns, "
            f"both {cov['complete_turns']}/{cov['turns_total']}")
    else:  # legacy v1 manifest: one global evidence set
        L.append(f"  instructions captured: {manifest.get('instructions_captured')} "
                 f"({manifest.get('instructions_source')})")
        L.append(f"  tool definitions captured: {manifest.get('tools_captured')} "
                 f"({manifest.get('tools_source')})")
    L.append(f"  baseline telemetry covers "
             f"{_fmt(manifest.get('baseline_turns_covered'))}/{_fmt(manifest.get('turns'))} turns")
    L.append(f"  trajectory sha256 {manifest.get('trajectory_sha256')}")
    if manifest.get("_runs"):
        L.append(f"  runs: {', '.join(manifest['_runs'])}")
    warns = manifest.get("warnings") or []
    if warns:
        L.append(f"  WARNINGS ({len(warns)})")
        for w in warns[:10]:
            L.append(f"    ! {w}")
        if len(warns) > 10:
            L.append(f"    ! ... {len(warns) - 10} more")
    return "\n".join(L)


def _ev_tag(ev: dict[str, Any]) -> str:
    """Compact per-turn evidence tag for format_run.

    ``ev=i+t+`` both replayed with this call's own captured evidence;
    ``i-``/``t-`` evidence gap (rotated/unreachable — replayed without);
    ``i0``/``t0`` the store proved that call was sent without the field;
    ``ev=global`` legacy v1 capsule using its single global evidence set.
    """
    if not ev:
        return "ev=?"
    if ev.get("mode") == "global":
        return "ev=global"

    def _s(v: Any) -> str:
        return {"per-turn": "+", "absent-by-evidence": "0", "none": "-"}.get(v, "?")

    return f"ev=i{_s(ev.get('instructions'))}t{_s(ev.get('tools'))}"


def format_run(run: dict[str, Any]) -> str:
    L = []
    L.append(f"REPLAY RUN — capsule {run.get('capsule')}  run {run.get('run_id')}")
    L.append(f"  endpoint {run.get('endpoint')}  chain={run.get('chain_mode')}")
    L.append(f"  turns {run.get('turns_completed')}/{run.get('turns_requested')} ok"
             f"  failed {run.get('turns_failed')}  chain breaks {run.get('chain_breaks')}")
    L.append(f"  wall {_fmt(run.get('wall_seconds'), 's')}")
    for e in run.get("turns", []):
        tel = e.get("telemetry") or {}
        usage = e.get("usage") or {}
        flag = ""
        if e.get("chain_break"):
            flag = " CHAIN-BREAK->full"
        elif e.get("degraded_full"):
            flag = " full"
        L.append(
            f"  turn {e['index']:>3}  {_fmt(e.get('latency_ms'), 'ms'):>10}  "
            f"finish={e.get('finish_reason') or 'ERR':<10} "
            f"cont={tel.get('continuation') or '-':<8} "
            f"in={_fmt(usage.get('input_tokens'))} cached={_fmt(usage.get('cached_tokens'))}"
            f"  {_ev_tag(e.get('evidence') or {})}"
            f"{flag}"
        )
        if e.get("error"):
            L.append(f"        error: {e['error'][:160]}")
    warns = run.get("warnings") or []
    if warns:
        L.append("  WARNINGS")
        for w in warns[:10]:
            L.append(f"    ! {w}")
    return "\n".join(L)


def format_comparison(cmp: dict[str, Any]) -> str:
    b = cmp["behavior"]
    cc = cmp["chain_continuity"]
    L = []
    L.append(f"COMPARISON — capsule {cmp['capsule']}  run {cmp['run_id']}  vs recording")
    L.append(f"  VERDICT: {cmp['verdict']}")
    L.append(f"  behavior: identical {b['identical']}  similar {b['similar']}"
             f"  diverged {b['diverged']}  failed {b['failed']}")
    L.append(f"  cache continuity: exact continuations {cc['exact_continuations']}"
             f"/{cc['chained_turns']} chained  rebuilds {cc['full_rebuilds']}"
             f"  chain breaks {cc['chain_breaks']}")
    if cc["rebuild_reasons"]:
        L.append(f"  rebuild reasons: {', '.join(cc['rebuild_reasons'])}")
    mtp = cmp["mtp"]
    L.append(f"  MTP acceptance: baseline {_fmt(mtp.get('baseline_acceptance'))}"
             f"  replay {_fmt(mtp.get('replay_acceptance'))}")
    L.append(f"  replay latency total {_fmt(cmp['latency'].get('replay_total_ms'), 'ms')}"
             f"  median {_fmt(cmp['latency'].get('replay_median_ms'), 'ms')}")
    L.append("")
    L.append("  PER-TURN")
    for r in cmp["turns"]:
        base = r.get("baseline") or {}
        delta = r.get("latency_delta_ms")
        L.append(
            f"  turn {r['index']:>3}  {r['behavior']:<9} sim {_fmt(r.get('content_similarity'))}"
            f"  replay {_fmt(r.get('replay_latency_ms'), 'ms')}"
            f"  vs baseline-inference {_fmt(base.get('inference_ms'), 'ms')}"
            + (f"  (delta {delta:+.0f} ms)" if delta is not None else "")
        )
    if cmp["reasons"]:
        L.append("")
        L.append("  WHY NOT 'exact'")
        for r in cmp["reasons"][:12]:
            L.append(f"    ! {r}")
    return "\n".join(L)


def format_runs_comparison(cmp: dict[str, Any]) -> str:
    L = []
    L.append(f"RUN-vs-RUN — capsule {cmp['capsule']}  {cmp['run_a']} ({cmp.get('a_chain_mode')})"
             f"  vs  {cmp['run_b']} ({cmp.get('b_chain_mode')})")
    L.append(f"  turns compared {cmp['turns_compared']}  diverged {cmp['turns_diverged']}")
    L.append(f"  total latency A {_fmt(cmp.get('total_latency_a_ms'), 'ms')}"
             f"  B {_fmt(cmp.get('total_latency_b_ms'), 'ms')}")
    for r in cmp["turns"]:
        L.append(
            f"  turn {r['index']:>3}  sim {_fmt(r.get('content_similarity'))}"
            f"  A {_fmt(r.get('a_latency_ms'), 'ms')} [{r.get('a_continuation') or '-'}]"
            f"  B {_fmt(r.get('b_latency_ms'), 'ms')} [{r.get('b_continuation') or '-'}]"
            + (f"  delta {r['latency_delta_ms']:+.0f} ms" if r.get("latency_delta_ms") is not None else "")
        )
    return "\n".join(L)