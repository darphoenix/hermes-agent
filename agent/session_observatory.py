"""Session Observatory — read-only profiler for a single Hermes session.

Combines four durable, read-only sources into one structural report:

1. The Hermes state DB (SQLite, opened read-only) — session row, message
   structure, tool-call metadata, routing, compaction markers.
2. The main inference wrapper logs (current + rotated) — response lineage,
   token usage, prefill/decode timings, hot-cache reuse, MTP acceptance,
   scheduler disconnect/capacity events.
3. The sidecar (conscience model) wrapper logs (current + rotated) — SSD
   prefix-cache decisions, queue/admission/memory telemetry.
4. Durable conscience artifacts (~/.hermes/conscience/<session_id>/) —
   event ledger, LLM audits, stop audit, ledgers.

Privacy contract: the report contains NO prompts, message content, tool
arguments, raw tool output, headers, request/response bodies, or giant
payloads. Only structural metadata (counts, ids, durations, token totals)
and short redacted reason/verdict fields appear.

Tolerance contract: missing files, rotated/overlapping logs, truncated
lines, malformed JSON and older line formats never raise — they produce
warnings and are skipped. Duplicate events (log overlap across rotations)
are deduplicated.

No third-party dependencies: stdlib only.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ObservatoryError(Exception):
    """Base error for the Session Observatory."""


class SessionNotFound(ObservatoryError):
    """No session row matches the given id/prefix."""


class AmbiguousSessionPrefix(ObservatoryError):
    """A session prefix matched more than one session id."""

    def __init__(self, prefix: str, candidates: list[str]) -> None:
        self.prefix = prefix
        self.candidates = candidates[:20]
        shown = ", ".join(self.candidates)
        more = "" if len(candidates) <= 20 else f" (+{len(candidates) - 20} more)"
        super().__init__(
            f"session prefix '{prefix}' is ambiguous — {len(candidates)} matches: {shown}{more}"
        )


# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------


def get_state_db_path(hermes_home: str | os.PathLike[str] | None = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home) / "state.db"
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home) / "state.db"
    return Path.home() / ".hermes" / "state.db"


def get_conscience_dir(
    session_id: str, hermes_home: str | os.PathLike[str] | None = None
) -> Path:
    if hermes_home is not None:
        base = Path(hermes_home)
    else:
        base = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return base / "conscience" / session_id


def _candidate_log_dirs(kind: str) -> list[Path]:
    """Candidate directories for wrapper logs, most-specific first.

    ``kind`` is "main" or "sidecar". Env overrides win; then the known
    macOS log locations; then generic fallbacks.
    """
    if kind == "main":
        env = os.environ.get("HERMES_OBSERVATORY_MAIN_LOG_DIR")
        names = ["mlx-serve-flashnext", "mlx-serve*", "mlx-openai-wrapper"]
        patterns = ["server.log*"]
    else:
        env = os.environ.get("HERMES_OBSERVATORY_SIDECAR_LOG_DIR")
        names = ["mlx-openai-wrapper-sidecar", "mlx-openai-wrapper*sidecar*",
                 "mlx-openai-wrapper*"]
        patterns = ["stderr.log*", "stdout.log*", "server.log*"]
    dirs: list[Path] = []
    if env:
        dirs.append(Path(env).expanduser())
    logs_root = Path.home() / "Library" / "Logs"
    for name in names:
        dirs.extend(sorted(logs_root.glob(name)))
    state_root = Path.home() / ".local" / "state"
    for name in names:
        dirs.extend(sorted(state_root.glob(name)))
    return dirs


@dataclass
class LogFile:
    path: Path
    kind: str  # "main" | "sidecar"
    rotated: bool
    ok: bool = True
    error: str | None = None
    lines_scanned: int = 0
    truncated_lines: int = 0
    malformed_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "rotated": self.rotated,
            "ok": self.ok,
            "error": self.error,
            "lines_scanned": self.lines_scanned,
            "truncated_lines": self.truncated_lines,
            "malformed_lines": self.malformed_lines,
        }


def discover_log_files(kind: str, dirs: Iterable[Path] | None = None) -> list[Path]:
    """Return log files for ``kind``, current first, then rotated (.1, .2...)."""
    found: list[Path] = []
    seen: set[Path] = set()
    candidates = list(dirs) if dirs is not None else _candidate_log_dirs(kind)
    patterns = (
        ["server.log*"] if kind == "main"
        else ["stderr.log*", "stdout.log*", "server.log*"]
    )
    for d in candidates:
        for pat in patterns:
            for p in sorted(d.glob(pat)):
                if not p.is_file():
                    continue
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                found.append(p)

    def sort_key(p: Path) -> tuple[int, int]:
        # base name priority (server.log before stderr.log for main), then
        # rotation index: no suffix = 0 (current), .N = N (older).
        name = p.name
        base, _, suffix = name.partition(".log")
        if kind == "main":
            base_rank = 0 if base == "server" else 1
        else:
            base_rank = {"stderr": 0, "stdout": 1, "server": 2}.get(base, 3)
        try:
            rot = int(suffix.lstrip(".")) if suffix else 0
        except ValueError:
            rot = 99
        return (base_rank, rot)

    return sorted(found, key=sort_key)


# ---------------------------------------------------------------------------
# Session resolution (read-only SQLite, unique-prefix matching)
# ---------------------------------------------------------------------------


def open_state_db_readonly(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    p = Path(db_path).expanduser().resolve()
    if not p.exists():
        raise ObservatoryError(f"state DB not found: {p}")
    conn = sqlite3.connect(f"file:{p.as_uri()[len('file:'):]}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_session_id(conn: sqlite3.Connection, ident: str) -> str:
    """Resolve an exact session id or a unique prefix.

    Raises SessionNotFound / AmbiguousSessionPrefix.
    """
    ident = (ident or "").strip()
    if not ident:
        raise SessionNotFound("empty session id")
    row = conn.execute(
        "SELECT id FROM sessions WHERE id = ?", (ident,)
    ).fetchone()
    if row:
        return row["id"]
    rows = conn.execute(
        "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY id LIMIT 21",
        (_like_escape(ident) + "%",),
    ).fetchall()
    if not rows:
        raise SessionNotFound(f"no session matches '{ident}'")
    if len(rows) > 1:
        raise AmbiguousSessionPrefix(ident, [r["id"] for r in rows])
    return rows[0]["id"]


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------------------
# Log line parsing (tolerant, structural only)
# ---------------------------------------------------------------------------

RE_RESP_ID = re.compile(r"resp_(?P<ms>\d{10,17})_[0-9a-zA-Z]+")
RE_CONTINUATION = re.compile(
    r"\[hermes-qwen-late\]\s+exact continuation\s+prev=(?P<prev>\S+)\s+"
    r"base=(?P<base>\d+)\s+prompt=(?P<prompt>\d+)"
)
RE_HOT_CACHE_REUSE = re.compile(
    r"\[hot-cache\]\s+reused\s+(?P<reused>\d+)/(?P<total>\d+)\s+tokens"
    r"\s+\(matched\s+(?P<matched>\d+)"
)
RE_MTP_ROUND = re.compile(
    r"\[mtp-round\](?:\s+off0=(?P<off0>\d+))?(?:\s+t1=(?P<t1>\d+))?"
    r".*?drafts=\{(?P<drafts>[^}]*)\}\s+accepted=(?P<accepted>\d+)"
)
RE_MTP_REGIME = re.compile(
    r"\[mtp\]\s+regime gate:\s+two-chunk\s+(?P<two>\d+(?:\.\d+)?)\s*ms/tok\s+vs\s+"
    r"single\s+(?P<single>\d+(?:\.\d+)?)\s*ms/tok\s+->\s+(?P<decision>[^\(\n]+)"
)
RE_SCHED_DISCONNECT = re.compile(
    r"\[scheduler\].*(prefill aborted.*disconnected|client disconnected)", re.I
)
RE_SCHED_CAPACITY = re.compile(
    r"\[scheduler\].*(model load failed.*InsufficientMemory|insufficient memory|capacity)",
    re.I,
)
RE_USAGE = re.compile(
    r'"usage"\s*:\s*\{\s*"input_tokens"\s*:\s*(?P<input>\d+)\s*,\s*"output_tokens"\s*:\s*(?P<output>\d+)'
    r'(?:\s*,\s*"total_tokens"\s*:\s*(?P<total>\d+))?.*?"cached_tokens"\s*:\s*(?P<cached>\d+)'
)
RE_TIMING = re.compile(
    r'"(?P<key>tokenize_ms|prompt_ms|predicted_ms|prompt_per_second|predicted_per_second)"'
    r"\s*:\s*(?P<val>-?\d+(?:\.\d+)?)"
)
RE_RETRY = re.compile(r"\b(retry|retries|retrying)\b", re.I)
RE_POISON = re.compile(r"\bpoison(ed)?\b", re.I)
RE_MALFORMED_REQ = re.compile(r"\bmalformed\b", re.I)
RE_QUEUE_WAIT = re.compile(
    r"queue.*?(?P<ms>\d+(?:\.\d+)?)\s*ms|generation[- ]queue.*?(?P<s>\d+(?:\.\d+)?)\s*s",
    re.I,
)
RE_TRUNC_MARK = re.compile(r"log line truncated|\[truncated[^\]]*\]$")
RE_SIDECAR_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?)\s+"
    r"(?P<level>[A-Z]+)\s+"
)
RE_SIDECAR_CACHE_DECISION = re.compile(
    r"prefix cache warm reuse decision outcome=(?P<outcome>\w+)"
    r"(?:.*?prefix_tokens=(?P<tokens>\d+))?"
)
RE_SIDECAR_PERSIST = re.compile(r"prefix cache persist outcome=(?P<outcome>\w+)")
RE_SIDECAR_ADMISSION = re.compile(r"admission (?:check|denied|refus\w+)", re.I)
RE_SIDECAR_OOM = re.compile(r"insufficient memory|memory admission.*(?:denied|refus)", re.I)
RE_SIDECAR_STALL = re.compile(r"stream stall|stalled", re.I)
RE_SIDECAR_CANCEL = re.compile(r"stream cancel|client (?:disconnected|gone)", re.I)

# Compaction markers in stored message content (prefix check only — the
# content itself is never copied into the report).
RE_COMPACTION_PREFIX = re.compile(
    r"^\s*(\[?Summary of (previous|earlier|prior) conversation|\[Context compaction|<compaction)",
    re.I,
)

_MAX_REASON_CHARS = 48


def _redact(text: Any) -> str:
    """Short, structural-only redaction of a reason/verdict field."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9 :_.\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:_MAX_REASON_CHARS]


def _event_fingerprint(kind: str, ts: float | None, payload: dict) -> str:
    raw = f"{kind}|{ts if ts is not None else ''}|{sorted(payload.items())}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _parse_sidecar_ts(line: str) -> float | None:
    m = RE_SIDECAR_TS.match(line)
    if not m:
        return None
    ts = m.group("ts").replace("T", " ").replace(",", ".")
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return dt.timestamp()


def _resp_anchor(line: str, last_anchor: float | None) -> float | None:
    """Best-effort unix-seconds anchor for a main-wrapper log line.

    Uses any resp_<ms>_ id embedded in the line (the id encodes a unix-ms
    creation timestamp); otherwise falls back to the most recent anchor
    seen earlier in the same file (log lines for one generation are
    contiguous).
    """
    m = RE_RESP_ID.search(line)
    if m:
        ms = int(m.group("ms"))
        # Guard against absurd values from truncated ids.
        if 1_000_000_000 < ms < 100_000_000_000_000:
            return ms / 1000.0
    return last_anchor


@dataclass
class MainLogStats:
    responses: list[dict[str, Any]] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)
    tokens: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "generated": 0, "cached": 0}
    )
    timing: dict[str, float] = field(
        default_factory=lambda: {
            "prefill_ms": 0.0, "decode_ms": 0.0, "tokenize_ms": 0.0,
            "prefill_tps": 0.0, "decode_tps": 0.0,
        }
    )
    _timing_samples: int = 0
    usage_events: int = 0
    hot_cache: dict[str, int] = field(
        default_factory=lambda: {"reused": 0, "total": 0, "matched": 0, "events": 0}
    )
    mtp: dict[str, Any] = field(
        default_factory=lambda: {
            "rounds": 0, "draft_tokens": 0, "accepted_tokens": 0,
            "regime_checks": 0, "regime_two_chunk_rounds": 0,
            "two_chunk_ms_per_tok": [], "single_ms_per_tok": [],
        }
    )
    events: dict[str, int] = field(
        default_factory=lambda: {
            "disconnects": 0, "capacity": 0, "retries": 0, "poison": 0,
            "malformed_requests": 0,
        }
    )
    event_reasons: list[str] = field(default_factory=list)
    _fingerprints: set[str] = field(default_factory=set)
    duplicates: int = 0

    def _add(self, kind: str, ts: float | None, payload: dict) -> bool:
        fp = _event_fingerprint(kind, ts, payload)
        if fp in self._fingerprints:
            self.duplicates += 1
            return False
        self._fingerprints.add(fp)
        return True

    def consume(self, line: str, anchor: float | None) -> float | None:
        """Parse one main-wrapper log line. Returns the new anchor."""
        new_anchor = _resp_anchor(line, anchor)
        ts = new_anchor

        m = RE_CONTINUATION.search(line)
        if m:
            prev = m.group("prev")
            if prev not in self._seen_ids:
                self._seen_ids.add(prev)
                self.responses.append({
                    "response_id": prev,
                    "base_tokens": int(m.group("base")),
                    "prompt_tokens": int(m.group("prompt")),
                })
            # The continuation line names the *previous* response; the new
            # response id (if present on this line) extends the chain.
            for rid in RE_RESP_ID.findall(line):
                pass
            return ts

        m = RE_HOT_CACHE_REUSE.search(line)
        if m:
            payload = {
                "reused": int(m.group("reused")),
                "total": int(m.group("total")),
                "matched": int(m.group("matched")),
            }
            if self._add("hot_cache", ts, payload):
                self.hot_cache["reused"] += payload["reused"]
                self.hot_cache["total"] += payload["total"]
                self.hot_cache["matched"] += payload["matched"]
                self.hot_cache["events"] += 1
            return ts

        m = RE_MTP_ROUND.search(line)
        if m:
            # drafts={...} holds the round's *candidate token ids*, not a
            # count: drafted_tokens = number of parsed candidates/positions,
            # accepted = length of the accepted prefix (<= candidates).
            # Summing the ids reported 86.7M drafted tokens and ~0.0002
            # acceptance on a real profile.
            drafted = len(re.findall(r"\d+", m.group("drafts")))
            accepted = int(m.group("accepted"))
            # off0/t1 identify the round: all mtp-round lines inside one
            # generation share the same ts anchor, and two distinct rounds
            # can draft the same ids with the same accept count (seen live).
            # Without them the rotation dedup collapses real rounds.
            payload = {
                "drafts": drafted,
                "accepted": accepted,
                "off0": m.group("off0"),
                "t1": m.group("t1"),
            }
            if self._add("mtp_round", ts, payload):
                self.mtp["rounds"] += 1
                self.mtp["draft_tokens"] += drafted
                self.mtp["accepted_tokens"] += accepted
            return ts

        m = RE_MTP_REGIME.search(line)
        if m:
            payload = {
                "two": float(m.group("two")),
                "single": float(m.group("single")),
                "decision": _redact(m.group("decision")),
            }
            if self._add("mtp_regime", ts, payload):
                self.mtp["regime_checks"] += 1
                self.mtp["two_chunk_ms_per_tok"].append(payload["two"])
                self.mtp["single_ms_per_tok"].append(payload["single"])
                if "every round" in payload["decision"]:
                    self.mtp["regime_two_chunk_rounds"] += 1
            return ts

        if RE_SCHED_DISCONNECT.search(line):
            if self._add("disconnect", ts, {}):
                self.events["disconnects"] += 1
                self._note_reason("client disconnected during prefill")
            return ts

        if RE_SCHED_CAPACITY.search(line):
            if self._add("capacity", ts, {}):
                self.events["capacity"] += 1
                self._note_reason("model load failed: insufficient memory")
            return ts

        m = RE_USAGE.search(line)
        if m:
            payload = {
                "input": int(m.group("input")),
                "output": int(m.group("output")),
                "cached": int(m.group("cached")),
            }
            if self._add("usage", ts, payload):
                self.tokens["prompt"] += payload["input"]
                self.tokens["generated"] += payload["output"]
                self.tokens["cached"] += payload["cached"]
                self.usage_events += 1
            # usage blocks also carry timing keys; fall through to timing.

        timing_pairs: dict[str, float] = {}
        for tm in RE_TIMING.finditer(line):
            timing_pairs[tm.group("key")] = float(tm.group("val"))
        if timing_pairs and self._add("timing", ts, timing_pairs):
            for key, val in timing_pairs.items():
                if key == "prompt_ms":
                    self.timing["prefill_ms"] += val
                elif key == "predicted_ms":
                    self.timing["decode_ms"] += val
                elif key == "tokenize_ms":
                    self.timing["tokenize_ms"] += val
                elif key == "prompt_per_second" and val > 0:
                    self.timing["prefill_tps"] = max(self.timing["prefill_tps"], val)
                elif key == "predicted_per_second" and val > 0:
                    self.timing["decode_tps"] = max(self.timing["decode_tps"], val)
                self._timing_samples += 1

        # Anomaly keywords are only trusted on lines that do not carry model
        # content: SSE dumps quote prose that casually mentions "retry",
        # "poisoned" or "malformed" and would inflate the counts.
        if '"delta":' in line or '"text":' in line or '"arguments":' in line:
            return new_anchor if new_anchor is not None else anchor

        if RE_POISON.search(line):
            if self._add("poison", ts, {}):
                self.events["poison"] += 1
        elif RE_RETRY.search(line):
            if self._add("retry", ts, {}):
                self.events["retries"] += 1
        elif RE_MALFORMED_REQ.search(line):
            if self._add("malformed_request", ts, {}):
                self.events["malformed_requests"] += 1
                self._note_reason("malformed request rejected")

        return new_anchor if new_anchor is not None else anchor

    def _note_reason(self, reason: str) -> None:
        r = _redact(reason)
        if r and r not in self.event_reasons:
            self.event_reasons.append(r)

    def finalize(self) -> dict[str, Any]:
        mtp = dict(self.mtp)
        drafted = mtp["draft_tokens"]
        mtp["acceptance_rate"] = (
            round(mtp["accepted_tokens"] / drafted, 4) if drafted > 0 else None
        )
        for key in ("two_chunk_ms_per_tok", "single_ms_per_tok"):
            vals = mtp.pop(key)
            mtp[f"{key.replace('_ms_per_tok', '')}_ms_per_tok_median"] = (
                _median(vals) if vals else None
            )
        out = {
            "responses": len(self.responses),
            "lineage": self.responses[-50:],  # structural chain tail only
            "tokens": dict(self.tokens),
            "timing": {k: round(v, 2) for k, v in self.timing.items()},
            "hot_cache": dict(self.hot_cache),
            "mtp": mtp,
            "events": dict(self.events),
            "event_reasons": self.event_reasons[:20],
            "duplicate_events_deduped": self.duplicates,
        }
        hc = self.hot_cache
        out["hot_cache"]["reuse_ratio"] = (
            round(hc["reused"] / hc["total"], 4) if hc["total"] else None
        )
        return out


@dataclass
class SidecarLogStats:
    cache: dict[str, Any] = field(
        default_factory=lambda: {
            "decisions": {"hit": 0, "miss": 0, "other": 0},
            "hit_prefix_tokens": 0,
            "persist": {"ok": 0, "failed": 0},
        }
    )
    events: dict[str, int] = field(
        default_factory=lambda: {
            "queue_waits": 0, "admission_denials": 0, "capacity": 0,
            "stalls": 0, "disconnects": 0, "retries": 0, "poison": 0,
            "malformed_requests": 0,
        }
    )
    queue_wait_ms: float = 0.0
    event_reasons: list[str] = field(default_factory=list)
    _fingerprints: set[str] = field(default_factory=set)
    duplicates: int = 0
    lines_in_window: int = 0

    def _add(self, kind: str, ts: float | None, payload: dict) -> bool:
        fp = _event_fingerprint(kind, ts, payload)
        if fp in self._fingerprints:
            self.duplicates += 1
            return False
        self._fingerprints.add(fp)
        return True

    def consume_continuation(self, line: str) -> bool:
        """Multi-line record continuation (no timestamp of its own)."""
        return False

    def consume(self, line: str, ts: float | None) -> None:
        self.lines_in_window += 1

        m = RE_SIDECAR_CACHE_DECISION.search(line)
        if m:
            outcome = m.group("outcome").lower()
            tokens = int(m.group("tokens") or 0)
            if self._add("cache_decision", ts, {"o": outcome, "t": tokens}):
                if outcome == "hit":
                    self.cache["decisions"]["hit"] += 1
                    self.cache["hit_prefix_tokens"] += tokens
                elif outcome == "miss":
                    self.cache["decisions"]["miss"] += 1
                else:
                    self.cache["decisions"]["other"] += 1
            return

        m = RE_SIDECAR_PERSIST.search(line)
        if m:
            outcome = m.group("outcome").lower()
            if self._add("persist", ts, {"o": outcome}):
                if outcome == "ok":
                    self.cache["persist"]["ok"] += 1
                else:
                    self.cache["persist"]["failed"] += 1
            return

        if RE_SIDECAR_OOM.search(line):
            if self._add("capacity", ts, {}):
                self.events["capacity"] += 1
                self._note_reason("memory admission refused")
            return

        if RE_SIDECAR_ADMISSION.search(line):
            if self._add("admission", ts, {}):
                self.events["admission_denials"] += 1
            return

        if RE_SIDECAR_STALL.search(line):
            if self._add("stall", ts, {}):
                self.events["stalls"] += 1
                self._note_reason("stream stall")
            return

        if RE_SIDECAR_CANCEL.search(line):
            if self._add("sc_disconnect", ts, {}):
                self.events["disconnects"] += 1
                self._note_reason("client disconnected (sidecar)")
            return

        m = RE_QUEUE_WAIT.search(line)
        if m and "queue" in line.lower():
            ms = float(m.group("ms")) if m.group("ms") else float(m.group("s")) * 1000.0
            if self._add("queue", ts, {"ms": ms}):
                self.events["queue_waits"] += 1
                self.queue_wait_ms += ms
            return

        if RE_POISON.search(line):
            if self._add("poison", ts, {}):
                self.events["poison"] += 1
        elif RE_RETRY.search(line):
            if self._add("retry", ts, {}):
                self.events["retries"] += 1
        elif RE_MALFORMED_REQ.search(line):
            if self._add("malformed_request", ts, {}):
                self.events["malformed_requests"] += 1
                self._note_reason("malformed request (sidecar)")

    def _note_reason(self, reason: str) -> None:
        r = _redact(reason)
        if r and r not in self.event_reasons:
            self.event_reasons.append(r)

    def finalize(self) -> dict[str, Any]:
        return {
            "cache": {
                **{k: dict(v) for k, v in self.cache.items() if isinstance(v, dict)},
                "hit_prefix_tokens": self.cache["hit_prefix_tokens"],
            },
            "events": dict(self.events),
            "queue_wait_ms": round(self.queue_wait_ms, 1),
            "event_reasons": self.event_reasons[:20],
            "lines_in_window": self.lines_in_window,
            "duplicate_events_deduped": self.duplicates,
        }


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return round(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0, 3)


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------


def scan_main_logs(
    files: list[Path], window: tuple[float, float] | None, warnings: list[str]
) -> tuple[MainLogStats, list[LogFile]]:
    stats = MainLogStats()
    metas: list[LogFile] = []
    if not files:
        warnings.append("no main-wrapper log files found — lineage/timing from logs unavailable")
        return stats, metas
    lo, hi = window if window else (0.0, time.time() + 3600)
    lo, hi = lo - 300, hi + 300  # generous padding for clock skew
    for path in files:
        meta = LogFile(path=path, kind="main", rotated=path.name != "server.log")
        metas.append(meta)
        anchor: float | None = None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    meta.lines_scanned += 1
                    if RE_TRUNC_MARK.search(line):
                        meta.truncated_lines += 1
                    new_anchor = _resp_anchor(line, anchor)
                    attributed = new_anchor if new_anchor is not None else anchor
                    # Only consume lines attributable to this session's time
                    # window; lines before the first response id (or after
                    # the window ends) belong to other sessions.
                    if window and (attributed is None or not (lo <= attributed <= hi)):
                        if new_anchor is not None:
                            anchor = new_anchor
                        continue
                    stats.consume(line, anchor)
                    if new_anchor is not None:
                        anchor = new_anchor
        except OSError as exc:
            meta.ok = False
            meta.error = str(exc)[:120]
            warnings.append(f"main log unreadable: {path} ({exc.__class__.__name__})")
    return stats, metas


def scan_sidecar_logs(
    files: list[Path], window: tuple[float, float] | None, warnings: list[str]
) -> tuple[SidecarLogStats, list[LogFile]]:
    stats = SidecarLogStats()
    metas: list[LogFile] = []
    if not files:
        warnings.append("no sidecar log files found — conscience cache/queue telemetry unavailable")
        return stats, metas
    lo, hi = window if window else (0.0, time.time() + 3600)
    lo, hi = lo - 300, hi + 300
    for path in files:
        meta = LogFile(
            path=path, kind="sidecar",
            rotated=not path.name.endswith((".log",)),
        )
        metas.append(meta)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    meta.lines_scanned += 1
                    ts = _parse_sidecar_ts(line)
                    if ts is None:
                        # Continuation line of a multi-line record or an old
                        # format without timestamps: skip tolerantly.
                        continue
                    if window and not (lo <= ts <= hi):
                        continue
                    stats.consume(line, ts)
        except OSError as exc:
            meta.ok = False
            meta.error = str(exc)[:120]
            warnings.append(f"sidecar log unreadable: {path} ({exc.__class__.__name__})")
    return stats, metas


# ---------------------------------------------------------------------------
# State DB profile
# ---------------------------------------------------------------------------


def _message_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(messages)")}


def profile_state_db(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise SessionNotFound(f"session {session_id} not in state DB")
    session = dict(row)

    # Schema-adaptive column resolution: older/newer stores have renamed or
    # dropped optional columns; degrade gracefully instead of erroring.
    cols = _message_columns(conn)
    ts_col = "timestamp" if "timestamp" in cols else (
        "time_created" if "time_created" in cols else None
    )
    model_col = "model" if "model" in cols else None
    stop_col = "stop_reason" if "stop_reason" in cols else None

    msg_counts = conn.execute(
        "SELECT role, COUNT(*) AS n FROM messages WHERE session_id = ? GROUP BY role",
        (session_id,),
    ).fetchall()
    roles = {r["role"]: r["n"] for r in msg_counts}

    tool_rows = conn.execute(
        "SELECT tool_name, COUNT(*) AS n FROM messages "
        "WHERE session_id = ? AND role = 'tool' GROUP BY tool_name",
        (session_id,),
    ).fetchall()
    tool_calls = {r["tool_name"] or "unknown": r["n"] for r in tool_rows}

    # Tool execution time: pair each assistant message that requested tools
    # with the next tool result's timestamp.
    tool_seconds = 0.0
    if ts_col:
        pair_rows = conn.execute(
            f"SELECT m.id, m.{ts_col} AS req_ts, "
            f"       (SELECT MIN(t.{ts_col}) FROM messages t "
            "         WHERE t.session_id = m.session_id AND t.role = 'tool' "
            "           AND t.id > m.id) AS res_ts "
            "FROM messages m WHERE m.session_id = ? AND m.role = 'assistant' "
            "  AND m.tool_calls IS NOT NULL AND m.tool_calls != ''",
            (session_id,),
        ).fetchall()
        for r in pair_rows:
            if r["req_ts"] and r["res_ts"] and r["res_ts"] >= r["req_ts"]:
                tool_seconds += float(r["res_ts"]) - float(r["req_ts"])

    finish = conn.execute(
        "SELECT finish_reason, COUNT(*) AS n FROM messages "
        "WHERE session_id = ? AND role = 'assistant' AND finish_reason IS NOT NULL "
        "  AND finish_reason != '' GROUP BY finish_reason",
        (session_id,),
    ).fetchall()

    stop_reasons: dict[str, int] = {}
    if stop_col:
        stop_rows = conn.execute(
            f"SELECT {stop_col} AS sr, COUNT(*) AS n FROM messages "
            "WHERE session_id = ? AND role = 'assistant' "
            f"  AND {stop_col} IS NOT NULL AND {stop_col} != '' "
            f"GROUP BY {stop_col}",
            (session_id,),
        ).fetchall()
        stop_reasons = {(r["sr"] or "unknown"): r["n"] for r in stop_rows}

    assistant_models: dict[str, int] = {}
    if model_col:
        model_rows = conn.execute(
            f"SELECT {model_col} AS m, COUNT(*) AS n FROM messages "
            "WHERE session_id = ? AND role = 'assistant' "
            f"  AND {model_col} IS NOT NULL AND {model_col} != '' "
            f"GROUP BY {model_col}",
            (session_id,),
        ).fetchall()
        assistant_models = {(r["m"] or "unknown"): r["n"] for r in model_rows}

    # Compaction markers — prefix check on stored content only (substr keeps
    # the scan cheap and never copies content into the report).
    compactions = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND ("
        "  substr(content,1,90) LIKE 'Summary of previous conversation%'"
        "  OR substr(content,1,90) LIKE 'Summary of earlier conversation%'"
        "  OR substr(content,1,90) LIKE 'Summary of prior conversation%'"
        "  OR substr(content,1,90) LIKE '[Context compaction%'"
        "  OR substr(content,1,90) LIKE '<compaction%')",
        (session_id,),
    ).fetchone()["n"]

    token_sum = conn.execute(
        "SELECT COALESCE(SUM(token_count), 0) AS n FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()["n"]

    tool_request_count = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'assistant' "
        "  AND tool_calls IS NOT NULL AND tool_calls != ''",
        (session_id,),
    ).fetchone()["n"]

    started = session.get("started_at")
    ended = session.get("ended_at")
    # Crash-tolerant duration: fall back to the last message timestamp when
    # the session row never recorded an ended_at.
    if ended is None and ts_col:
        last = conn.execute(
            f"SELECT MAX({ts_col}) AS t FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()["t"]
        if last is not None:
            ended = float(last)
    duration = None
    if started and ended:
        duration = round(float(ended) - float(started), 2)

    return {
        "session": {
            "id": session["id"],
            "source": session.get("source"),
            "model": session.get("model"),
            "started_at": session.get("started_at"),
            "ended_at": session.get("ended_at"),
            "duration_seconds": duration,
            "parent_session_id": session.get("parent_session_id"),
            "cwd": session.get("cwd"),
        },
        "messages": {
            "by_role": roles,
            "assistant_with_tool_calls": tool_request_count,
            "token_count_sum": token_sum,
        },
        "tools": {
            "calls_by_name": tool_calls,
            "total_calls": sum(tool_calls.values()),
            "execution_seconds": round(tool_seconds, 2),
            "error_finish_reasons": {
                (r["finish_reason"] or "unknown"): r["n"] for r in finish
                if "error" in (r["finish_reason"] or "").lower()
            },
        },
        "routing": {
            "session_model": session.get("model"),
            "assistant_models": assistant_models,
            "finish_reasons": {(r["finish_reason"] or "unknown"): r["n"] for r in finish},
            "stop_reasons": stop_reasons,
        },
        "compactions": {"count": compactions},
    }


# ---------------------------------------------------------------------------
# Conscience artifacts
# ---------------------------------------------------------------------------

_CONSCIENCE_FILES = {
    "events": "conscience-events.json",
    "llm_audits": "llm-audits.json",
    "stop_audit": "stop-audit.json",
    "critique_tickets": "critique-tickets.json",
    "intervention_ledger": "intervention-ledger.json",
    "completion_ledger": "completion-ledger.json",
    "active_repair_contract": "active-repair-contract.json",
    "task_contract": "task-contract.json",
}


def _load_json_tolerant(path: Path, warnings: list[str]) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(
            f"conscience artifact malformed/unreadable: {path.name} ({exc.__class__.__name__})"
        )
        return None


def profile_conscience_artifacts(
    session_id: str, hermes_home: Path | None, window: tuple[float, float] | None
) -> dict[str, Any]:
    base = get_conscience_dir(session_id, hermes_home)
    out: dict[str, Any] = {"dir": str(base), "present": base.is_dir()}
    warnings: list[str] = []
    if not base.is_dir():
        warnings.append(f"no conscience artifact dir for {session_id}")
        out["warnings"] = warnings
        return out

    warnings_out = out.setdefault("warnings", warnings)

    events = _load_json_tolerant(base / _CONSCIENCE_FILES["events"], warnings_out)
    event_counts: dict[str, int] = {}
    durations_ms = 0.0
    verdicts: list[str] = []
    review_spans: list[tuple[float, float]] = []
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            et = str(ev.get("event_type") or "unknown")
            event_counts[et] = event_counts.get(et, 0) + 1
            payload = ev.get("payload") or {}
            if isinstance(payload, dict):
                d = payload.get("duration_ms") or payload.get("latency_ms")
                if isinstance(d, (int, float)):
                    durations_ms += float(d)
                v = payload.get("verdict") or payload.get("reason")
                if v is not None:
                    rv = _redact(v)
                    if rv and rv not in verdicts:
                        verdicts.append(rv)
            ts = ev.get("timestamp")
            if isinstance(ts, (int, float)) and et in ("REVIEW_START", "TASK_START"):
                review_spans.append((float(ts), float("inf")))

    audits = _load_json_tolerant(base / _CONSCIENCE_FILES["llm_audits"], warnings_out)
    audit_summary = {"total": 0, "by_type": {}, "interventions": 0, "parse_failures": 0}
    if isinstance(audits, list):
        audit_summary["total"] = len(audits)
        for a in audits:
            if not isinstance(a, dict):
                continue
            rt = str(a.get("review_type") or "unknown")
            audit_summary["by_type"][rt] = audit_summary["by_type"].get(rt, 0) + 1
            parsed = a.get("parsed")
            if isinstance(parsed, dict):
                if parsed.get("should_intervene"):
                    audit_summary["interventions"] += 1
                v = _redact(parsed.get("verdict"))
                if v and v not in verdicts:
                    verdicts.append(v)
            else:
                audit_summary["parse_failures"] += 1

    stop_audit = _load_json_tolerant(base / _CONSCIENCE_FILES["stop_audit"], warnings_out)
    stop_verdict = None
    if isinstance(stop_audit, dict):
        pr = stop_audit.get("parsed_review")
        if isinstance(pr, dict):
            stop_verdict = _redact(pr.get("verdict"))

    task_contract = _load_json_tolerant(
        base / _CONSCIENCE_FILES["task_contract"], warnings_out)
    repair_contract = _load_json_tolerant(
        base / _CONSCIENCE_FILES["active_repair_contract"], warnings_out)

    def _ledger_count(name: str) -> int | None:
        data = _load_json_tolerant(base / _CONSCIENCE_FILES[name], warnings_out)
        if isinstance(data, (list, dict)):
            return len(data)
        return None

    out.update({
        "events": {"by_type": event_counts, "total": sum(event_counts.values())},
        "review_duration_ms_total": round(durations_ms, 1),
        "verdicts": verdicts[:20],
        "llm_audits": audit_summary,
        "stop_audit_verdict": stop_verdict,
        "task_contract_present": isinstance(task_contract, dict),
        "active_repair_contract_present": bool(
            isinstance(repair_contract, dict) and repair_contract
        ),
        "critique_tickets": _ledger_count("critique_tickets"),
        "interventions": _ledger_count("intervention_ledger"),
        "completion_ledger": _ledger_count("completion_ledger"),
        "review_spans": [
            [s, None] for s, _ in review_spans
        ][:50],
    })
    return out


# ---------------------------------------------------------------------------
# Top-level profile
# ---------------------------------------------------------------------------


def profile_session(
    session_ident: str,
    *,
    db_path: str | os.PathLike[str] | None = None,
    hermes_home: str | os.PathLike[str] | None = None,
    main_log_files: list[Path] | None = None,
    sidecar_log_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Build the full read-only profile for one session."""
    warnings: list[str] = []
    db = Path(db_path) if db_path else get_state_db_path(hermes_home)
    home = Path(hermes_home) if hermes_home else None

    conn = open_state_db_readonly(db)
    try:
        session_id = resolve_session_id(conn, session_ident)
        db_profile = profile_state_db(conn, session_id)
    finally:
        conn.close()

    started = db_profile["session"].get("started_at")
    ended = db_profile["session"].get("ended_at")
    window: tuple[float, float] | None = None
    if started:
        window = (float(started), float(ended) if ended else time.time())
    else:
        warnings.append("session row has no started_at — log attribution falls back to full-file scan")

    m_files = main_log_files if main_log_files is not None else discover_log_files("main")
    s_files = sidecar_log_files if sidecar_log_files is not None else discover_log_files("sidecar")

    main_stats, main_metas = scan_main_logs(m_files, window, warnings)
    side_stats, side_metas = scan_sidecar_logs(s_files, window, warnings)
    conscience = profile_conscience_artifacts(session_id, home, window)
    warnings.extend(f"conscience: {w}" for w in conscience.pop("warnings", []))

    # Weighted cache reuse: token-weighted across the session's generations.
    tokens = dict(main_stats.tokens)
    if not tokens["prompt"]:
        # Fall back to DB-side estimate when logs had no usage blocks.
        tokens["db_token_count_sum"] = db_profile["messages"]["token_count_sum"]
    weighted = None
    if tokens["prompt"] > 0:
        weighted = round(tokens["cached"] / tokens["prompt"], 4)

    total_seconds = db_profile["session"].get("duration_seconds")
    timing = dict(main_stats.timing)
    timing["tool_seconds"] = db_profile["tools"]["execution_seconds"]
    timing["queue_ms"] = side_stats.queue_wait_ms
    timing["total_seconds"] = total_seconds

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "session_id": session_id,
        "sources": {
            "state_db": str(db),
            "main_wrapper_logs": [m.to_dict() for m in main_metas],
            "sidecar_logs": [m.to_dict() for m in side_metas],
            "conscience_dir": conscience.get("dir"),
            "conscience_present": conscience.get("present", False),
        },
        "session": db_profile["session"],
        "routing": db_profile["routing"],
        "lineage": {
            "responses": main_stats.finalize()["responses"],
            "chain_tail": main_stats.finalize()["lineage"],
        },
        "tokens": {
            **{k: tokens[k] for k in ("prompt", "generated", "cached") if k in tokens},
            "weighted_cache_reuse": weighted,
            "usage_blocks": main_stats.usage_events,
            "db_token_count_sum": db_profile["messages"]["token_count_sum"],
        },
        "timing": {**timing, "timing_samples": main_stats._timing_samples},
        "tools": db_profile["tools"],
        "hot_cache": main_stats.finalize()["hot_cache"],
        "mtp": main_stats.finalize()["mtp"],
        "conscience": conscience,
        "sidecar": side_stats.finalize(),
        "compactions": db_profile["compactions"],
        "anomalies": main_stats.finalize()["events"],
        "anomaly_reasons": main_stats.finalize()["event_reasons"],
        "messages": db_profile["messages"],
        "warnings": warnings,
    }
    return report


# ---------------------------------------------------------------------------
# Comparison + attribution
# ---------------------------------------------------------------------------

# Metric paths compared for attribution. Each entry: (label, getter, unit,
# higher_is_slower).
_COMPARE_METRICS: list[tuple[str, Any, str, bool]] = [
    ("duration_seconds", lambda r: r["session"].get("duration_seconds"), "s", True),
    ("prefill_ms", lambda r: r["timing"].get("prefill_ms") if r["timing"].get("timing_samples") else None, "ms", True),
    ("decode_ms", lambda r: r["timing"].get("decode_ms") if r["timing"].get("timing_samples") else None, "ms", True),
    ("tokenize_ms", lambda r: r["timing"].get("tokenize_ms") if r["timing"].get("timing_samples") else None, "ms", True),
    ("tool_seconds", lambda r: r["timing"].get("tool_seconds"), "s", True),
    ("queue_ms", lambda r: r["timing"].get("queue_ms"), "ms", True),
    ("conscience_review_ms", lambda r: r["conscience"].get("review_duration_ms_total"), "ms", True),
    ("cached_tokens", lambda r: r["tokens"].get("cached") if r["tokens"].get("usage_blocks") else None, "tok", False),
    ("generated_tokens", lambda r: r["tokens"].get("generated") if r["tokens"].get("usage_blocks") else None, "tok", True),
    ("prompt_tokens", lambda r: r["tokens"].get("prompt") if r["tokens"].get("usage_blocks") else None, "tok", True),
    ("tool_calls", lambda r: r["tools"].get("total_calls"), "n", True),
    ("compactions", lambda r: r["compactions"].get("count"), "n", True),
    ("disconnects", lambda r: r["anomalies"].get("disconnects"), "n", True),
    ("capacity_events", lambda r: r["anomalies"].get("capacity"), "n", True),
    ("retries", lambda r: r["anomalies"].get("retries"), "n", True),
    ("poison_events", lambda r: r["anomalies"].get("poison"), "n", True),
    ("sidecar_cache_hits", lambda r: r["sidecar"]["cache"]["decisions"]["hit"], "n", False),
    ("sidecar_cache_misses", lambda r: r["sidecar"]["cache"]["decisions"]["miss"], "n", True),
    ("mtp_acceptance_rate", lambda r: r["mtp"].get("acceptance_rate"), "ratio", False),
]


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def compare_profiles(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Compare two profiles (base = reference, target = subject).

    Differences are split into *proven* (both sides have the underlying
    measurement, so the delta is directly evidenced) and *unattributed*
    (the timing delta that no proven component explains).
    """
    base_dur = _num(base["session"].get("duration_seconds"))
    target_dur = _num(target["session"].get("duration_seconds"))
    total_delta = (
        round(target_dur - base_dur, 2)
        if base_dur is not None and target_dur is not None
        else None
    )

    proven: list[dict[str, Any]] = []
    for label, getter, unit, _slower in _COMPARE_METRICS:
        try:
            b = _num(getter(base))
            t = _num(getter(target))
        except (KeyError, TypeError):
            b = t = None
        if b is None or t is None:
            continue
        delta = round(t - b, 4)
        if delta == 0:
            continue
        proven.append({
            "metric": label,
            "base": b,
            "target": t,
            "delta": delta,
            "unit": unit,
            "evidence": "measured on both sides (state DB / wrapper logs)",
        })

    # Time attribution: which proven deltas plausibly explain the wall-clock
    # difference. Only same-unit time components count toward "explained".
    explained_ms = 0.0
    explained_labels: list[str] = []
    for p in proven:
        if p["metric"] == "duration_seconds":
            continue
        if p["unit"] == "ms":
            explained_ms += p["delta"]
            explained_labels.append(p["metric"])
        elif p["unit"] == "s":
            explained_ms += p["delta"] * 1000.0
            explained_labels.append(p["metric"])

    unattributed_ms = None
    if total_delta is not None:
        unattributed_ms = round(total_delta * 1000.0 - explained_ms, 1)

    return {
        "base_session": base.get("session_id"),
        "target_session": target.get("session_id"),
        "duration_delta_seconds": total_delta,
        "summary": _comparison_summary(total_delta, proven, unattributed_ms),
        "proven_differences": proven,
        "explained_time_ms": round(explained_ms, 1),
        "unattributed_time_ms": unattributed_ms,
        "caveats": _comparison_caveats(base, target),
    }


def _comparison_summary(
    total_delta: float | None,
    proven: list[dict[str, Any]],
    unattributed_ms: float | None,
) -> str:
    if total_delta is None:
        return "duration comparison unavailable (missing session timestamps)"
    side = "faster" if total_delta < 0 else "slower"
    lines = [f"target was {abs(total_delta):.1f}s {side} than base"]
    top = sorted(
        (p for p in proven if p["metric"] != "duration_seconds" and p["unit"] in ("ms", "s")),
        key=lambda p: abs(p["delta"] if p["unit"] == "ms" else p["delta"] * 1000),
        reverse=True,
    )[:5]
    for p in top:
        d = p["delta"] if p["unit"] == "ms" else p["delta"] * 1000
        lines.append(f"proven: {p['metric']} {p['base']} -> {p['target']} ({d:+.0f} ms)")
    counts = [p for p in proven if p["unit"] in ("n", "tok", "ratio")]
    for p in counts[:8]:
        lines.append(f"proven: {p['metric']} {p['base']} -> {p['target']} (delta {p['delta']:+g})")
    if unattributed_ms is not None:
        lines.append(f"unattributed residual: {unattributed_ms:+.0f} ms")
    return "\n".join(lines)


def _comparison_caveats(base: dict[str, Any], target: dict[str, Any]) -> list[str]:
    caveats: list[str] = []
    for side, r in (("base", base), ("target", target)):
        if not r["sources"]["main_wrapper_logs"]:
            caveats.append(f"{side}: no main-wrapper logs — log-derived metrics missing")
        if not r["sources"]["sidecar_logs"]:
            caveats.append(f"{side}: no sidecar logs — queue/cache metrics missing")
        if not r["sources"]["conscience_present"]:
            caveats.append(f"{side}: no conscience artifacts — review cost missing")
        if r["warnings"]:
            caveats.append(f"{side}: {len(r['warnings'])} tolerance warning(s) — see profile")
    return caveats


# ---------------------------------------------------------------------------
# Text rendering (terminal-friendly, no markdown)
# ---------------------------------------------------------------------------


def _fmt_ts(ts: Any) -> str:
    t = _num(ts)
    if t is None:
        return "?"
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_num(v: Any, suffix: str = "") -> str:
    n = _num(v)
    if n is None:
        return "n/a"
    if n == int(n):
        return f"{int(n)}{suffix}"
    return f"{n:.2f}{suffix}"


def format_report(report: dict[str, Any]) -> str:
    s = report["session"]
    t = report["timing"]
    tok = report["tokens"]
    tools = report["tools"]
    con = report["conscience"]
    side = report["sidecar"]
    mtp = report["mtp"]
    hc = report["hot_cache"]
    an = report["anomalies"]

    L: list[str] = []
    add = L.append
    add(f"SESSION OBSERVATORY — {report['session_id']}")
    add(f"  source={s.get('source')}  model={_short_model(s.get('model'))}")
    add(f"  started {_fmt_ts(s.get('started_at'))}  ended {_fmt_ts(s.get('ended_at'))}"
        f"  duration {_fmt_num(s.get('duration_seconds'), 's')}")
    if s.get("parent_session_id"):
        add(f"  parent {s['parent_session_id']}")

    add("")
    add("LINEAGE")
    add(f"  responses observed in wrapper log: {report['lineage']['responses']}")
    tail = report["lineage"]["chain_tail"]
    for r in tail[-3:]:
        add(f"    {r['response_id']}  base={r['base_tokens']} prompt={r['prompt_tokens']}")

    add("")
    add("TOKENS")
    if tok.get("usage_blocks"):
        add(f"  prompt {_fmt_num(tok.get('prompt'))}  generated {_fmt_num(tok.get('generated'))}"
            f"  cached {_fmt_num(tok.get('cached'))}")
        add(f"  weighted cache reuse: {_fmt_num(tok.get('weighted_cache_reuse'))}")
    else:
        add("  prompt/generated/cached: n/a (no usage blocks in window — log truncation)")
    add(f"  db token_count_sum: {_fmt_num(tok.get('db_token_count_sum'))}")
    add(f"  hot-cache reuse: {_fmt_num(hc.get('reused'))}/{_fmt_num(hc.get('total'))} tokens"
        f"  ratio {_fmt_num(hc.get('reuse_ratio'))}  events {_fmt_num(hc.get('events'))}")

    add("")
    add("TIMING")
    if t.get("timing_samples"):
        add(f"  prefill {_fmt_num(t.get('prefill_ms'), 'ms')}  decode {_fmt_num(t.get('decode_ms'), 'ms')}"
            f"  tokenize {_fmt_num(t.get('tokenize_ms'), 'ms')}")
    else:
        add("  prefill/decode/tokenize: n/a (no timing telemetry in window)")
    add(f"  tool exec {_fmt_num(t.get('tool_seconds'), 's')}  queue {_fmt_num(t.get('queue_ms'), 'ms')}"
        f"  total {_fmt_num(t.get('total_seconds'), 's')}")
    add(f"  peak rates: prefill {_fmt_num(t.get('prefill_tps'), ' tok/s')}"
        f"  decode {_fmt_num(t.get('decode_tps'), ' tok/s')}")

    add("")
    add("TOOLS")
    add(f"  calls: {tools.get('total_calls', 0)}")
    for name, n in sorted(tools.get("calls_by_name", {}).items(), key=lambda kv: -kv[1]):
        add(f"    {name:24s} {n}")
    if tools.get("error_finish_reasons"):
        add(f"  error finishes: {tools['error_finish_reasons']}")

    add("")
    add("CONSCIENCE")
    if con.get("present"):
        add(f"  events: {con.get('events', {}).get('total', 0)}"
            f"  review cost {_fmt_num(con.get('review_duration_ms_total'), 'ms')}")
        add(f"  llm audits: {con.get('llm_audits', {}).get('total', 0)}"
            f"  interventions {con.get('llm_audits', {}).get('interventions', 0)}"
            f"  parse failures {con.get('llm_audits', {}).get('parse_failures', 0)}")
        if con.get("stop_audit_verdict"):
            add(f"  stop verdict: {con['stop_audit_verdict']}")
        if con.get("verdicts"):
            add(f"  verdicts/reasons: {', '.join(con['verdicts'][:6])}")
        sc = side.get("cache", {})
        add(f"  sidecar cache: hits {sc.get('decisions', {}).get('hit', 0)}"
            f"  misses {sc.get('decisions', {}).get('miss', 0)}"
            f"  persist ok {sc.get('persist', {}).get('ok', 0)}")
    else:
        add("  no durable artifacts found")

    add("")
    add("COMPACTIONS")
    add(f"  count: {report['compactions'].get('count', 0)}")

    add("")
    add("ANOMALIES")
    add(f"  retries {an.get('retries', 0)}  poison {an.get('poison', 0)}"
        f"  disconnects {an.get('disconnects', 0)}  malformed {an.get('malformed_requests', 0)}"
        f"  capacity {an.get('capacity', 0)}")
    if report.get("anomaly_reasons"):
        add(f"  reasons: {'; '.join(report['anomaly_reasons'][:5])}")

    add("")
    add("ROUTING")
    add(f"  session model: {_short_model(report['routing'].get('session_model'))}")
    for m, n in report["routing"].get("assistant_models", {}).items():
        add(f"    {_short_model(m):48s} {n}")
    if report["routing"].get("finish_reasons"):
        add(f"  finish reasons: {report['routing']['finish_reasons']}")

    add("")
    add("MTP")
    add(f"  rounds {mtp.get('rounds', 0)}  draft tokens {mtp.get('draft_tokens', 0)}"
        f"  accepted {mtp.get('accepted_tokens', 0)}"
        f"  acceptance {_fmt_num(mtp.get('acceptance_rate'))}")
    add(f"  regime checks {mtp.get('regime_checks', 0)}"
        f"  two-chunk median {_fmt_num(mtp.get('two_chunk_ms_per_tok_median'), ' ms/tok')}"
        f"  single-chunk median {_fmt_num(mtp.get('single_ms_per_tok_median'), ' ms/tok')}")

    if report["warnings"]:
        add("")
        add(f"WARNINGS ({len(report['warnings'])})")
        for w in report["warnings"][:15]:
            add(f"  ! {w}")
        if len(report["warnings"]) > 15:
            add(f"  ! ... {len(report['warnings']) - 15} more")
    return "\n".join(L)


def format_comparison(cmp: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    add(f"COMPARISON — base {cmp['base_session']}  vs  target {cmp['target_session']}")
    d = cmp["duration_delta_seconds"]
    if d is not None:
        side = "faster" if d < 0 else "slower"
        add(f"  target was {abs(d):.1f}s {side}")
    add("")
    add("PROVEN DIFFERENCES (measured on both sides)")
    for p in cmp["proven_differences"]:
        add(f"  {p['metric']:24s} {p['base']} -> {p['target']}   delta {p['delta']:+g} {p['unit']}")
    add("")
    add(f"  explained time delta: {_fmt_num(cmp['explained_time_ms'], ' ms')}")
    add(f"  unattributed residual: {_fmt_num(cmp['unattributed_time_ms'], ' ms')}")
    if cmp["caveats"]:
        add("")
        add("CAVEATS")
        for c in cmp["caveats"]:
            add(f"  ! {c}")
    return "\n".join(L)


def _short_model(m: Any) -> str:
    if not m:
        return "?"
    s = str(m)
    # Keep only the model-name segment of long HF cache paths.
    if "models--" in s:
        seg = [p for p in s.split("/") if p.startswith("models--")]
        if seg:
            return seg[-1].replace("models--", "").replace("--", "/")[:48]
    parts = s.rstrip("/").split("/")
    return parts[-1][:48]