from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

TASK_START = "TASK_START"
PLAN_SUMMARY = "PLAN_SUMMARY"
TOOL_CALL = "TOOL_CALL"
TOOL_RESULT = "TOOL_RESULT"
TOOL_PROGRESS = "TOOL_PROGRESS"
TOOL_POLICY_CONFLICT = "TOOL_POLICY_CONFLICT"
ARTIFACT_UPDATED = "ARTIFACT_UPDATED"
DRAFT_ANSWER = "DRAFT_ANSWER"
INTENT_TO_STOP = "INTENT_TO_STOP"
IRREVERSIBLE_ACTION_INTENT = "IRREVERSIBLE_ACTION_INTENT"


@dataclass
class TaskCriterion:
    criterion_id: str
    source_text: str
    description: str = ""
    required: bool = True
    status: str = "open"
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class PromiseRecord:
    promise_id: str
    source_text: str
    status: str = "open"
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class TaskContract:
    task_id: str
    raw_user_request: str
    explicit_asks: List[TaskCriterion] = field(default_factory=list)
    explicit_constraints: List[str] = field(default_factory=list)
    implied_checks: List[str] = field(default_factory=list)
    open_assumptions: List[str] = field(default_factory=list)
    done_definition: List[str] = field(default_factory=list)


@dataclass
class ConscienceEvent:
    event_type: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionLedgerEntry:
    criterion_id: str
    status: str
    last_updated_event: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class CritiqueTicket:
    verdict: str
    reason: str
    evidence: List[str] = field(default_factory=list)
    next_best_action: str = ""
    criterion_ids: List[str] = field(default_factory=list)
    recommended_tools: Optional[List[str]] = None
    tool_policy: Optional[Dict[str, Any]] = None
    active_tool_decision: Optional[str] = None


@dataclass
class ConscienceVerdict:
    should_intervene: bool
    critique_ticket: Optional[CritiqueTicket] = None
    source: str = "llm"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConscienceState:
    contract: TaskContract
    ledger: Dict[str, CompletionLedgerEntry]
    promises: Dict[str, PromiseRecord] = field(default_factory=dict)
    events: List[ConscienceEvent] = field(default_factory=list)
    intervention_ledger: List[Dict[str, Any]] = field(default_factory=list)
    step_labels: List[str] = field(default_factory=list)
    last_progress_index: int = -1
    seen_issue_fingerprints: Dict[str, int] = field(default_factory=dict)
    repair_rounds: int = 0
    max_repair_rounds: int = 4
    llm_audits: List[Dict[str, Any]] = field(default_factory=list)
    last_midtask_intervention_index: int = -1
    stateful_previous_response_id: Optional[str] = None
    stateful_initialized: bool = False
    stateful_last_event_index: int = 0
    stateful_prompt_token_limit: int = 50_000
    stateful_last_prompt_tokens: int = 0
    stateful_last_total_tokens: int = 0
    stateful_reset_count: int = 0
    stateful_next_init_reason: Optional[str] = None
    stateful_intervention_hashes: Dict[str, str] = field(default_factory=dict)
    stateful_ticket_history: Dict[str, int] = field(default_factory=dict)


def asdict_safe(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def jsonish_payload(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        try:
            return str(payload)
        except Exception:
            return ""


def _short_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _text_head_tail(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = max(120, limit // 2)
    return (
        text[:half]
        + f"\n... [truncated {len(text) - (half * 2)} chars, sha256={_short_hash(text)}] ...\n"
        + text[-half:]
    )


def _summarize_large_text(text: str, *, field: str, limit: int = 1600) -> Dict[str, Any]:
    text = str(text or "")
    return {
        "kind": "large_text_summary",
        "field": field,
        "chars": len(text),
        "sha256": _short_hash(text),
        "preview": _text_head_tail(text, limit),
    }


def _try_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def extract_task_contract(task_id: str, user_message: str) -> TaskContract:
    normalized = _normalize_text(user_message)
    criterion = TaskCriterion(
        criterion_id="criterion_001",
        source_text=normalized or (user_message or ""),
    )
    done_definition = [criterion.source_text] if criterion.source_text else []
    return TaskContract(
        task_id=task_id,
        raw_user_request=user_message,
        explicit_asks=[criterion] if criterion.source_text else [],
        explicit_constraints=[],
        implied_checks=[],
        open_assumptions=[],
        done_definition=done_definition,
    )


class ConscienceMonitor:
    def __init__(self, task_id: str, user_message: str, mode: str = "shadow"):
        self.mode = (mode or "shadow").strip().lower()
        contract = extract_task_contract(task_id, user_message)
        ledger = {
            criterion.criterion_id: CompletionLedgerEntry(
                criterion_id=criterion.criterion_id,
                status=criterion.status,
            )
            for criterion in contract.explicit_asks
        }
        self.state = ConscienceState(contract=contract, ledger=ledger)

    def to_artifacts(self) -> Dict[str, Any]:
        return {
            "task_contract": asdict(self.state.contract),
            "completion_ledger": {k: asdict(v) for k, v in self.state.ledger.items()},
            "promises": {k: asdict(v) for k, v in self.state.promises.items()},
            "events": [asdict(event) for event in self.state.events],
            "intervention_ledger": list(self.state.intervention_ledger),
            "critique_tickets": self._critique_tickets_payload(),
            "llm_audits": list(self.state.llm_audits),
        }

    def record_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> ConscienceEvent:
        event = ConscienceEvent(event_type=event_type, timestamp=time.time(), payload=dict(payload or {}))
        self.state.events.append(event)
        self.state.step_labels.append(event_type.lower())
        self._record_event_against_intervention_ledger(event_type, payload or {})
        return event

    def _latest_open_intervention(self) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.state.intervention_ledger):
            if entry.get("status") in {"issued", "attempted", "still_failing", "ignored"}:
                return entry
        return None

    def _record_event_against_intervention_ledger(self, event_type: str, payload: Dict[str, Any]) -> None:
        entry = self._latest_open_intervention()
        if not entry:
            return
        event_index = len(self.state.events) - 1
        if event_type == TOOL_CALL and entry.get("status") == "issued":
            entry["status"] = "attempted"
            entry["followed_event_index"] = event_index
            tool_name = str(payload.get("tool_name") or "")
            if tool_name:
                entry["followed_tool"] = tool_name
            return
        if event_type == TOOL_RESULT and entry.get("status") == "attempted":
            entry["last_result_event_index"] = event_index
            if "success" in payload:
                entry["last_result_success"] = bool(payload.get("success"))
            if "exit_code" in payload:
                entry["last_exit_code"] = payload.get("exit_code")
            if "result_preview_truncated" in payload:
                entry["last_result_preview_truncated"] = bool(payload.get("result_preview_truncated"))
            if "output_preview_truncated" in payload:
                entry["last_output_preview_truncated"] = bool(payload.get("output_preview_truncated"))
            if "output_chars" in payload:
                entry["last_output_chars"] = payload.get("output_chars")
            if payload.get("error"):
                entry["last_result_error"] = _text_head_tail(str(payload.get("error")), 500)
            if payload.get("tool_error"):
                entry["last_tool_error"] = _text_head_tail(str(payload.get("tool_error")), 500)
            if payload.get("result_preview"):
                entry["last_result_preview"] = _text_head_tail(str(payload.get("result_preview")), 900)
            if payload.get("output_tail"):
                entry["last_output_tail"] = _text_head_tail(str(payload.get("output_tail")), 900)

    def _issue_fingerprint(self, ticket: CritiqueTicket, review_type: str = "") -> str:
        raw = "|".join(
            [review_type, ticket.verdict, ticket.reason, ticket.next_best_action]
            + sorted(ticket.criterion_ids)
            + ticket.evidence[:3]
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _dedupe_or_admit_ticket(self, ticket: CritiqueTicket, review_type: str) -> tuple[bool, str]:
        fingerprint = self._issue_fingerprint(ticket, review_type)
        event_index = len(self.state.events)

        if review_type == "midtask":
            self.state.seen_issue_fingerprints[fingerprint] = event_index
            return True, ""

        last_seen = self.state.seen_issue_fingerprints.get(fingerprint)
        if last_seen is not None and event_index - last_seen < 2:
            return False, "duplicate_recent_issue"
        if self.state.repair_rounds >= self.state.max_repair_rounds:
            return False, "repair_limit_exhausted"
        self.state.seen_issue_fingerprints[fingerprint] = event_index
        self.state.repair_rounds += 1
        return True, ""

    def _compact_value_for_review(self, key: str, value: Any, *, limit: int = 2200) -> Any:
        if isinstance(value, str):
            if len(value) <= limit:
                return value
            parsed = _try_json_loads(value)
            if parsed is not value:
                return self._compact_value_for_review(key, parsed, limit=limit)
            return _summarize_large_text(value, field=key, limit=min(limit, 1600))
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for subkey, subvalue in value.items():
                sub_limit = 1200 if str(subkey) in {"content", "patch", "new_string", "old_string"} else limit
                compact[str(subkey)] = self._compact_value_for_review(str(subkey), subvalue, limit=sub_limit)
            return compact
        if isinstance(value, list):
            if len(value) > 20:
                return {
                    "kind": "large_list_summary",
                    "items": len(value),
                    "head": [self._compact_value_for_review(key, item, limit=limit) for item in value[:8]],
                    "tail": [self._compact_value_for_review(key, item, limit=limit) for item in value[-4:]],
                }
            return [self._compact_value_for_review(key, item, limit=limit) for item in value]
        return value

    @staticmethod
    def _json_chars(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False))

    def _budget_tail_items(
        self,
        items: List[Dict[str, Any]],
        *,
        max_chars: int,
        max_items: Optional[int] = None,
        min_items: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        if max_items is not None:
            source = items[-max(0, int(max_items)) :]
            omitted = max(0, len(items) - len(source))
        else:
            source = list(items)
            omitted = 0

        selected_reversed: List[Dict[str, Any]] = []
        used_chars = 2  # JSON list brackets.
        for item in reversed(source):
            item_chars = self._json_chars(item) + 2
            if len(selected_reversed) < min_items or used_chars + item_chars <= max_chars:
                selected_reversed.append(item)
                used_chars += item_chars
            else:
                omitted += 1
        return list(reversed(selected_reversed)), omitted

    def _compact_event_for_review(self, event: ConscienceEvent, *, profile: str = "review") -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        restart_profile = profile == "restart"
        for key, value in (event.payload or {}).items():
            field_limit = 1600 if restart_profile else 3600
            if event.event_type in {TOOL_CALL, TOOL_RESULT} and key == "tool_args":
                field_limit = 900 if restart_profile else 1800
            elif event.event_type == TOOL_RESULT and key in {
                "result_preview",
                "output",
                "stdout",
                "stderr",
                "output_head",
                "output_tail",
            }:
                field_limit = 1600 if restart_profile else 4200
            elif event.event_type in {DRAFT_ANSWER, INTENT_TO_STOP}:
                field_limit = 2400 if restart_profile else 3600
            payload[key] = self._compact_value_for_review(key, value, limit=field_limit)
        return {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "payload": payload,
        }

    def _recent_events_payload(self, limit: int = 24, *, profile: str = "review") -> List[Dict[str, Any]]:
        return [self._compact_event_for_review(event, profile=profile) for event in self.state.events[-limit:]]

    def _compact_events_slice(self, start: int, end: int) -> List[Dict[str, Any]]:
        return [self._compact_event_for_review(event) for event in self.state.events[start:end]]

    def _action_ledger_payload(
        self,
        limit: Optional[int] = 48,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        profile: str = "review",
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        indexed_events = list(enumerate(self.state.events))
        restart_profile = profile == "restart"
        if start is not None or end is not None:
            lower = max(0, int(start or 0))
            upper = len(self.state.events) if end is None else max(lower, min(len(self.state.events), int(end)))
            indexed_events = indexed_events[lower:upper]
        elif limit is not None:
            indexed_events = indexed_events[-limit:]
        for idx, event in indexed_events:
            payload = event.payload or {}
            row: Dict[str, Any] = {"event_index": idx, "event_type": event.event_type}
            if event.event_type in {TOOL_CALL, TOOL_RESULT}:
                row["tool_name"] = payload.get("tool_name")
                tool_args = _try_json_loads(payload.get("tool_args"))
                if isinstance(tool_args, dict):
                    for key in ("path", "command", "mode", "action", "session_id"):
                        if key in tool_args:
                            row[key] = _text_head_tail(str(tool_args.get(key) or ""), 180 if restart_profile else 300)
                    for key in ("content", "patch", "new_string", "old_string"):
                        if isinstance(tool_args.get(key), str):
                            row[f"{key}_sha256"] = _short_hash(tool_args[key])
                            row[f"{key}_chars"] = len(tool_args[key])
                elif isinstance(tool_args, str):
                    row["args"] = _text_head_tail(tool_args, 240 if restart_profile else 500)
                if event.event_type == TOOL_RESULT:
                    row["success"] = payload.get("success")
                    if "exit_code" in payload:
                        row["exit_code"] = payload.get("exit_code")
                    row["duration_seconds"] = payload.get("duration_seconds")
                    if "result_preview_truncated" in payload:
                        row["result_preview_truncated"] = bool(payload.get("result_preview_truncated"))
                    if "output_preview_truncated" in payload:
                        row["output_preview_truncated"] = bool(payload.get("output_preview_truncated"))
                    if "output_chars" in payload:
                        row["output_chars"] = payload.get("output_chars")
                    if payload.get("error"):
                        row["error"] = _text_head_tail(str(payload.get("error")), 240 if restart_profile else 500)
                    if payload.get("tool_error"):
                        row["tool_error"] = _text_head_tail(str(payload.get("tool_error")), 240 if restart_profile else 500)
                    if payload.get("result_preview"):
                        row["result_preview"] = _text_head_tail(
                            str(payload.get("result_preview")),
                            360 if restart_profile else 900,
                        )
                    if payload.get("output_tail"):
                        row["output_tail"] = _text_head_tail(
                            str(payload.get("output_tail")),
                            360 if restart_profile else 900,
                        )
            elif event.event_type == ARTIFACT_UPDATED:
                row["path"] = payload.get("path")
            elif event.event_type in {DRAFT_ANSWER, INTENT_TO_STOP}:
                text = str(payload.get("text") or "")
                row["text_sha256"] = _short_hash(text)
                row["text_chars"] = len(text)
                row["text_preview"] = _text_head_tail(text, 360 if restart_profile else 900)
            rows.append({k: v for k, v in row.items() if v not in (None, "")})
        return rows

    def _artifact_ledger_payload(self, *, changed_since_event: Optional[int] = None) -> List[Dict[str, Any]]:
        artifacts: Dict[str, Dict[str, Any]] = {}
        for idx, event in enumerate(self.state.events):
            payload = event.payload or {}
            path = None
            tool_args = _try_json_loads(payload.get("tool_args"))
            if event.event_type == ARTIFACT_UPDATED:
                path = payload.get("path")
            elif isinstance(tool_args, dict):
                path = tool_args.get("path")
            if not path:
                continue
            path = str(path)
            entry = artifacts.setdefault(path, {"path": path})
            entry["last_event_index"] = idx
            if isinstance(tool_args, dict):
                for key in ("content", "patch", "new_string"):
                    value = tool_args.get(key)
                    if isinstance(value, str):
                        entry[f"last_{key}_sha256"] = _short_hash(value)
                        entry[f"last_{key}_chars"] = len(value)
                if tool_args.get("mode"):
                    entry["last_mode"] = tool_args.get("mode")
        values = list(artifacts.values())[-32:]
        if changed_since_event is not None:
            values = [
                entry
                for entry in values
                if int(entry.get("last_event_index") or -1) >= int(changed_since_event)
            ]
        return values

    def _compact_intervention_entry(self, entry: Dict[str, Any], *, profile: str = "review") -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        restart_profile = profile == "restart"
        for key, value in (entry or {}).items():
            if key in {"reason", "required_action", "next_best_action", "outcome"}:
                compact[key] = _text_head_tail(str(value or ""), 260 if restart_profile else 700)
            elif key in {"evidence", "outcome_evidence"} and isinstance(value, list):
                item_limit = 160 if restart_profile else 300
                item_count = 2 if restart_profile else 5
                compact[key] = [_text_head_tail(str(item), item_limit) for item in value[:item_count]]
            elif key in {"last_result_preview", "last_result_error"}:
                compact[key] = _text_head_tail(str(value or ""), 240 if restart_profile else 600)
            else:
                compact[key] = value
        return compact

    def _intervention_ledger_payload(self, limit: int = 16) -> List[Dict[str, Any]]:
        return [self._compact_intervention_entry(entry) for entry in self.state.intervention_ledger[-limit:]]

    def _critique_tickets_payload(self) -> List[Dict[str, Any]]:
        tickets: List[Dict[str, Any]] = []
        for entry in self.state.intervention_ledger:
            ticket = self._compact_intervention_entry(entry)
            if "next_best_action" not in ticket and ticket.get("required_action"):
                ticket["next_best_action"] = ticket.get("required_action")
            tickets.append(ticket)
        return tickets

    def _restart_intervention_ledger_payload(self, *, max_chars: int, max_items: int) -> tuple[List[Dict[str, Any]], int]:
        entries = list(self.state.intervention_ledger)
        if not entries:
            return [], 0

        open_statuses = {"issued", "attempted", "still_failing", "ignored"}
        important = [entry for entry in entries if entry.get("status") in open_statuses][-4:]
        recent = entries[-max_items:]
        by_identity: Dict[int, Dict[str, Any]] = {id(entry): entry for entry in important + recent}
        ordered = [entry for entry in entries if id(entry) in by_identity]
        compact = [self._compact_intervention_entry(entry, profile="restart") for entry in ordered]
        selected, extra_omitted = self._budget_tail_items(
            compact,
            max_chars=max_chars,
            max_items=max_items,
            min_items=min(2, len(compact)),
        )
        omitted = max(0, len(entries) - len(selected))
        return selected, max(omitted, extra_omitted)

    def _intervention_hashes(self) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for entry in self.state.intervention_ledger:
            intervention_id = str(entry.get("id") or "")
            if not intervention_id:
                continue
            hashes[intervention_id] = _short_hash(
                json.dumps(self._compact_intervention_entry(entry), ensure_ascii=False, sort_keys=True)
            )
        return hashes

    def _ticket_history_payload(self) -> List[Dict[str, Any]]:
        return [
            {"fingerprint": fingerprint, "last_seen_event_index": last_seen}
            for fingerprint, last_seen in self.state.seen_issue_fingerprints.items()
        ]

    def _ticket_history_delta(self) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for fingerprint, last_seen in self.state.seen_issue_fingerprints.items():
            if self.state.stateful_ticket_history.get(fingerprint) != last_seen:
                updates.append({"fingerprint": fingerprint, "last_seen_event_index": last_seen})
        return updates

    def _intervention_ledger_delta(self) -> List[Dict[str, Any]]:
        prior_hashes = self.state.stateful_intervention_hashes or {}
        current_hashes = self._intervention_hashes()
        updates: List[Dict[str, Any]] = []
        by_id = {
            str(entry.get("id") or ""): self._compact_intervention_entry(entry)
            for entry in self.state.intervention_ledger
            if str(entry.get("id") or "")
        }
        for intervention_id, digest in current_hashes.items():
            if prior_hashes.get(intervention_id) != digest and intervention_id in by_id:
                updates.append(by_id[intervention_id])
        return updates

    def _ledger_delta_payload(self, start: int, end: int) -> Dict[str, Any]:
        return {
            "action_append": self._action_ledger_payload(limit=None, start=start, end=end),
            "artifact_upsert": self._artifact_ledger_payload(changed_since_event=start),
            "intervention_upsert": self._intervention_ledger_delta(),
            "ticket_history_upsert": self._ticket_history_delta(),
        }

    def _trajectory_memory_payload(self) -> Dict[str, Any]:
        return {
            "action_ledger": self._action_ledger_payload(),
            "artifact_ledger": self._artifact_ledger_payload(),
            "intervention_ledger": self._intervention_ledger_payload(),
        }

    def _stateful_compact_restart_char_budget(self) -> int:
        token_limit = int(self.state.stateful_prompt_token_limit or 50_000)
        return max(30_000, min(60_000, int(token_limit * 3 * 0.35)))

    def _reset_stateful_session(self, reason: str) -> None:
        self.state.stateful_previous_response_id = None
        self.state.stateful_initialized = False
        self.state.stateful_last_event_index = 0
        self.state.stateful_intervention_hashes = {}
        self.state.stateful_ticket_history = {}
        self.state.stateful_reset_count += 1
        self.state.stateful_next_init_reason = str(reason or "compaction")

    def _mark_stateful_memory_synced(self) -> None:
        self.state.stateful_intervention_hashes = self._intervention_hashes()
        self.state.stateful_ticket_history = dict(self.state.seen_issue_fingerprints)

    def _stateful_compaction_metadata(self, reason: str) -> Dict[str, Any]:
        return {
            "reason": reason,
            "prompt_token_limit": self.state.stateful_prompt_token_limit,
            "last_prompt_tokens": self.state.stateful_last_prompt_tokens,
            "last_total_tokens": self.state.stateful_last_total_tokens,
            "reset_count": self.state.stateful_reset_count,
        }

    def _available_tools_payload(self) -> List[str]:
        for event in self.state.events:
            if event.event_type != TASK_START or not isinstance(event.payload, dict):
                continue
            raw_tools = event.payload.get("available_tools") or event.payload.get("enabled_tools") or []
            if not isinstance(raw_tools, list):
                continue
            tools = sorted({str(tool).strip() for tool in raw_tools if str(tool).strip()})
            if tools:
                return tools
        return []

    def build_review_payload(self, review_type: str, draft_answer: str = "") -> Dict[str, Any]:
        return {
            "review_type": review_type,
            "mode": self.mode,
            "task_contract": asdict(self.state.contract),
            "available_tools": self._available_tools_payload(),
            "trajectory_memory": self._trajectory_memory_payload(),
            "recent_events": self._recent_events_payload(),
            "draft_answer": draft_answer,
            "ticket_history": self._ticket_history_payload(),
            "repair_rounds_used": self.state.repair_rounds,
            "max_repair_rounds": self.state.max_repair_rounds,
            "stop_repair_rounds_used": self.state.repair_rounds,
            "max_stop_repair_rounds": self.state.max_repair_rounds,
        }

    def _stateful_compact_restart_payload(
        self,
        review_type: str,
        draft_answer: str,
        *,
        event_count: int,
        restart_reason: str,
    ) -> Dict[str, Any]:
        char_budget = self._stateful_compact_restart_char_budget()
        recent_budget = int(char_budget * 0.30)
        action_budget = int(char_budget * 0.22)
        intervention_budget = int(char_budget * 0.18)

        recent_candidates = self._recent_events_payload(limit=24, profile="restart")
        recent_events, recent_omitted = self._budget_tail_items(
            recent_candidates,
            max_chars=recent_budget,
            max_items=12,
            min_items=min(4, len(recent_candidates)),
        )

        action_candidates = self._action_ledger_payload(limit=48, profile="restart")
        action_ledger, action_omitted = self._budget_tail_items(
            action_candidates,
            max_chars=action_budget,
            max_items=24,
            min_items=min(6, len(action_candidates)),
        )

        intervention_ledger, intervention_omitted = self._restart_intervention_ledger_payload(
            max_chars=intervention_budget,
            max_items=8,
        )

        all_artifacts = self._artifact_ledger_payload()
        artifact_ledger = all_artifacts[-16:]
        trajectory_memory = {
            "action_ledger": action_ledger,
            "artifact_ledger": artifact_ledger,
            "intervention_ledger": intervention_ledger,
        }
        ticket_history = self._ticket_history_payload()[-24:]
        compact_draft = _text_head_tail(str(draft_answer or ""), 2400)
        compaction = self._stateful_compaction_metadata(restart_reason)
        compaction.update(
            {
                "payload_char_budget": char_budget,
                "recent_event_window": 12,
                "action_ledger_window": 24,
                "intervention_ledger_window": 8,
                "omitted": {
                    "recent_events": max(0, len(recent_candidates) - len(recent_events), recent_omitted),
                    "action_ledger": max(0, len(action_candidates) - len(action_ledger), action_omitted),
                    "intervention_ledger": intervention_omitted,
                    "ticket_history": max(0, len(self.state.seen_issue_fingerprints) - len(ticket_history)),
                    "artifact_ledger": max(0, len(all_artifacts) - len(artifact_ledger)),
                },
                "event_history": {
                    "total_events": event_count,
                    "recent_events_included": len(recent_events),
                    "recent_events_omitted_from_window": max(0, len(recent_candidates) - len(recent_events)),
                },
            }
        )

        payload = {
            "review_type": review_type,
            "mode": self.mode,
            "task_contract": asdict(self.state.contract),
            "available_tools": self._available_tools_payload(),
            "trajectory_memory": trajectory_memory,
            "recent_events": recent_events,
            "draft_answer": compact_draft,
            "ticket_history": ticket_history,
            "repair_rounds_used": self.state.repair_rounds,
            "max_repair_rounds": self.state.max_repair_rounds,
            "stop_repair_rounds_used": self.state.repair_rounds,
            "max_stop_repair_rounds": self.state.max_repair_rounds,
            "stateful_mode": "compact_restart",
            "event_cursor_start": 0,
            "event_cursor_end": event_count,
            "stateful_compaction": compaction,
        }
        compaction["component_chars"] = {
            "task_contract": self._json_chars(payload["task_contract"]),
            "trajectory_memory": self._json_chars(trajectory_memory),
            "recent_events": self._json_chars(recent_events),
            "draft_answer": self._json_chars(compact_draft),
            "ticket_history": self._json_chars(ticket_history),
        }
        for _ in range(2):
            compaction["payload_chars"] = self._json_chars(payload)
            compaction["estimated_payload_tokens"] = self._estimated_payload_tokens(payload)
        return payload

    def _stateful_full_payload(
        self,
        review_type: str,
        draft_answer: str,
        *,
        event_count: int,
        restart_reason: Optional[str],
    ) -> Dict[str, Any]:
        if restart_reason:
            return self._stateful_compact_restart_payload(
                review_type,
                draft_answer,
                event_count=event_count,
                restart_reason=restart_reason,
            )
        payload = self.build_review_payload(review_type, draft_answer)
        payload["stateful_mode"] = "init_full"
        payload["event_cursor_start"] = 0
        payload["event_cursor_end"] = event_count
        return payload

    def _estimated_payload_tokens(self, payload: Dict[str, Any]) -> int:
        payload_chars = len(json.dumps(payload, ensure_ascii=False))
        return max(1, (payload_chars + 2) // 3)

    def _stateful_delta_would_exceed_limit(self, payload: Dict[str, Any]) -> bool:
        limit = int(self.state.stateful_prompt_token_limit or 0)
        if not limit or not self.state.stateful_last_prompt_tokens:
            return False
        projected = self.state.stateful_last_prompt_tokens + self._estimated_payload_tokens(payload)
        return projected >= int(limit * 0.95)

    def build_stateful_review_payload(self, review_type: str, draft_answer: str = "") -> Dict[str, Any]:
        event_count = len(self.state.events)
        if (
            self.state.stateful_initialized
            and self.state.stateful_prompt_token_limit > 0
            and self.state.stateful_last_prompt_tokens >= self.state.stateful_prompt_token_limit
        ):
            self._reset_stateful_session("prompt_token_limit_exceeded")
        if not self.state.stateful_initialized:
            return self._stateful_full_payload(
                review_type,
                draft_answer,
                event_count=event_count,
                restart_reason=self.state.stateful_next_init_reason,
            )

        start = max(0, min(self.state.stateful_last_event_index, event_count))
        payload = {
            "review_type": review_type,
            "mode": self.mode,
            "stateful_mode": "delta",
            "task_id": self.state.contract.task_id,
            "new_events": self._compact_events_slice(start, event_count),
            "event_cursor_start": start,
            "event_cursor_end": event_count,
            "ledger_delta": self._ledger_delta_payload(start, event_count),
            "draft_answer": draft_answer,
            "repair_rounds_used": self.state.repair_rounds,
            "max_repair_rounds": self.state.max_repair_rounds,
            "stop_repair_rounds_used": self.state.repair_rounds,
            "max_stop_repair_rounds": self.state.max_repair_rounds,
        }
        if self._stateful_delta_would_exceed_limit(payload):
            self._reset_stateful_session("projected_prompt_token_limit_exceeded")
            return self._stateful_full_payload(
                review_type,
                draft_answer,
                event_count=event_count,
                restart_reason=self.state.stateful_next_init_reason,
            )
        return payload

    @staticmethod
    def _intervention_json_contract_prompt() -> str:
        return (
            "When intervention requires tool use, set recommended_tools to a useful advisory list of exact tool names from payload.available_tools. "
            "If the actor should answer or synthesize from evidence already gathered without another tool call, return recommended_tools as an empty array. "
            "If no exact tool guidance is useful, return recommended_tools as null. Do not invent tool names. "
            "tool_policy is separate from recommended_tools and defaults to null. Set tool_policy only when the recent failure is specifically repeated wrong tool selection and preventing that choice for the next actor action is necessary to break the loop. "
            "A strict policy has the form {\"mode\":\"allowlist\",\"tools\":[\"exact_tool_name\"]}; it applies to one actor action only and must use exact names from payload.available_tools. Do not use tool_policy merely to recommend an efficient next step. "
            "Hermes records attempted automatically when the actor takes the next tool action. Include intervention_outcomes only for a semantic status transition to resolved, ignored, or still_failing; never narrate attempted-to-attempted progress or repeat the current status. "
            "Mark resolved only when a recent tool result or equivalent evidence directly proves the required action succeeded. Starting a command is attempted, not resolved. "
            "Each outcome object must use keys id, status, outcome, evidence, confidence. "
            "For a TOOL_PROGRESS review, set active_tool_decision to cancel only when the available command, elapsed time, and progress evidence make continued execution clearly wasteful; otherwise set it to continue. For other reviews omit active_tool_decision. "
            "Return strict JSON with keys: should_intervene (bool), verdict (string), reason (string), evidence (array of strings), "
            "next_best_action (string), recommended_tools (array of strings or null), tool_policy (object or null), criterion_ids (array of strings), confidence (string), and optional active_tool_decision (cancel or continue)."
        )

    @classmethod
    def _midtask_review_json_contract_prompt(cls) -> str:
        return (
            "Return strict JSON. If no intervention is warranted, return only "
            '{"should_intervene": false, "verdict": "observe"} '
            "and omit reason, evidence, next_best_action, recommended_tools, tool_policy, criterion_ids, and confidence, unless adding a transition-only intervention_outcomes update. "
            "Do not explain why you are observing. "
            "If intervention is warranted, "
            + cls._intervention_json_contract_prompt()
        )

    @classmethod
    def _stop_review_json_contract_prompt(cls) -> str:
        return (
            "Return strict JSON. If the actor may stop, return only "
            '{"should_intervene": false, "verdict": "allow_stop"} '
            "and omit reason, evidence, next_best_action, recommended_tools, tool_policy, criterion_ids, confidence, and intervention_outcomes. "
            "If the actor must not stop, "
            + cls._intervention_json_contract_prompt()
        )

    @classmethod
    def _midtask_review_system_prompt(cls) -> str:
        return (
            "You are Hermes conscience sidecar. For review_type='midtask', you are a sparse trajectory monitor, not a completion judge and not the main actor. "
            "Judge only from the task request and recent event stream. The actor is still working, so do not require the task to be complete yet. "
            "Default to should_intervene=false. "
            "Intervene only when the recent trajectory is clearly bad and likely to waste more calls or damage task state: "
            "the actor repeats the same tool, same arguments, same browser action, same selector, same URL/page, or same error without materially new evidence; "
            "the actor ignores a previous conscience correction; "
            "the actor continues after evidence shows the current strategy is failing; "
            "the actor is about to use clearly wrong tools or no-op actions; "
            "or the actor has enough evidence for the next move but is circling instead. "
            "Do not intervene merely because the task is not finished yet, you can think of a better next step, the actor is gathering new evidence, the actor made one ordinary mistake, or you are uncertain. "
            "If unsure, observe silently: should_intervene=false. "
            "When intervening, give one concise course correction. Name the repeated or wrong behavior to stop, and the different strategy to try next. "
            "Separate diagnosis from invocation details: if the exact command or procedure is not supported by the evidence, recommend one targeted inspection step instead of inventing a command. "
            "Do not produce a final answer unless the actor is explicitly looping on final-answer generation and no further tool use is needed. "
            "Do not rely on fixed heuristics, regex rules, or hand-crafted trigger categories. Infer directly from the evidence whether intervention is warranted right now. "
            + cls._midtask_review_json_contract_prompt()
        )

    @classmethod
    def _stop_review_system_prompt(cls) -> str:
        return (
            "You are Hermes conscience sidecar. For review_type='stop', you are a strict completion auditor, not the main actor. "
            "Judge only from the provided task request, draft answer, and event stream. "
            "Your core question is: if a careful user read this draft, would they correctly conclude the request was fully completed, with nothing material still missing? "
            "Do not trust the draft answer at face value. Reason from first principles about whether the user's request is actually satisfied. "
            "Interpret the user's request pragmatically, not narrowly. Include the ordinary implications needed for the answer to be trustworthy, not just the literal shortest reading. "
            "If the draft contains caveats, admissions, or scope limits showing something was not inspected, replayed, verified, confirmed, or judged, treat those as evidence of incompleteness unless clearly irrelevant. "
            "Do not let a draft pass merely because it starts with 'yes' or includes run metadata. Ask whether the actor can actually stand behind the answer. "
            "If there are places where the request is not fully satisfied, name those missing parts explicitly and trigger repair so the actor completes the request fully before stopping. "
            "Allow harmless surplus: if the draft directly answers the user's request and has no clear factual error, contradiction, missing required action, unsafe advice, or materially misleading extra instruction, allow stop. "
            "Do not block merely because the draft is more detailed than necessary, includes caveats, mentions optional next steps, or asks a harmless follow-up. Treat that as a style issue, not a completion failure. "
            "Intervene on surplus only when it would likely confuse the user, change the answer, create wrong expectations, or send the user toward an unnecessary or risky action. "
            "Only treat additional steps as missing when they are required for correctness or for satisfying the user's explicit request; do not require optional polish, consolidation, cleanup, or broader improvements. "
            "For claims about testing, validation, success, readiness, correctness, or whether something worked, lack of full inspection of the substantive result is usually a reason to intervene. "
            "Intervene when the actor should not be allowed to stop because the request is not fully satisfied under a careful, trust-preserving reading. "
            "Do not rely on fixed heuristics, regex rules, or hand-crafted trigger categories. Infer directly from the evidence whether intervention is warranted right now. "
            + cls._stop_review_json_contract_prompt()
        )

    @classmethod
    def _review_system_prompt(cls, review_type: str) -> str:
        if review_type == "midtask":
            return cls._midtask_review_system_prompt()
        return cls._stop_review_system_prompt()

    @classmethod
    def _stateful_review_system_prompt(cls) -> str:
        return (
            "You are Hermes conscience sidecar in one persistent stateful audit thread for a single Hermes turn. "
            "The first user message contains a compact task/audit snapshot. Later user messages may contain only "
            "append-only compact deltas: new_events, ledger_delta, current draft_answer, and repair counters. "
            "Maintain an internal task contract, event stream, action ledger, artifact ledger, intervention ledger, and ticket history. "
            "Apply ledger_delta as a patch: append action_append rows, upsert artifact_upsert rows by path, upsert intervention_upsert rows by id, and upsert ticket_history_upsert rows by fingerprint. "
            "If a ledger_delta field is absent or empty, that stored ledger is unchanged. "
            "A compact_restart message replaces the stored compact snapshot; it may omit older low-detail rows under stateful_compaction.omitted, so treat omitted history as summarized rather than nonexistent. "
            "Large edit payloads and long outputs may be represented by paths, sizes, hashes, and previews; use those "
            "summaries as evidence, and do not assume omitted bulk content is unavailable to the actor. Retain and use "
            "the prior task contract, tools, compact event stream, tickets, and earlier audit context from this thread. "
            "Each user message is JSON and includes review_type. Apply only the section matching review_type. "
            "For review_type='midtask': "
            + cls._midtask_review_system_prompt()
            + " For review_type='stop': "
            + cls._stop_review_system_prompt()
        )

    @staticmethod
    def _callable_accepts_optional_kwarg(func: Any, name: str) -> bool:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return False
        if name in sig.parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

    @staticmethod
    def _response_content(response: Any) -> str:
        try:
            from agent.auxiliary_client import extract_content_or_reasoning

            return extract_content_or_reasoning(response) or ""
        except Exception:
            try:
                return response.choices[0].message.content or ""
            except Exception:
                return ""

    @staticmethod
    def _response_status(response: Any) -> str:
        return str(
            getattr(response, "conscience_response_status", None)
            or getattr(response, "status", None)
            or ""
        ).strip().lower()

    @staticmethod
    def _response_finish_reason(response: Any) -> str:
        try:
            return str(response.choices[0].finish_reason or "").strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _response_completion_tokens(response: Any) -> int:
        usage = getattr(response, "usage", None)
        for name in ("completion_tokens", "output_tokens"):
            try:
                value = int(getattr(usage, name, 0) or 0)
                if value:
                    return value
            except Exception:
                continue
        return 0

    @staticmethod
    def _response_prompt_tokens(response: Any) -> int:
        usage = getattr(response, "usage", None)
        for name in ("prompt_tokens", "input_tokens"):
            try:
                value = int(getattr(usage, name, 0) or 0)
                if value:
                    return value
            except Exception:
                continue
        return 0

    @staticmethod
    def _response_total_tokens(response: Any) -> int:
        usage = getattr(response, "usage", None)
        for name in ("total_tokens",):
            try:
                value = int(getattr(usage, name, 0) or 0)
                if value:
                    return value
            except Exception:
                continue
        prompt = ConscienceMonitor._response_prompt_tokens(response)
        completion = ConscienceMonitor._response_completion_tokens(response)
        return prompt + completion if prompt or completion else 0

    @staticmethod
    def _looks_degenerate_output(content: str) -> bool:
        text = (content or "").strip()
        if len(text) < 80:
            return False
        non_ws = [char for char in text if not char.isspace()]
        if not non_ws:
            return True
        counts: Dict[str, int] = {}
        for char in non_ws:
            counts[char] = counts.get(char, 0) + 1
        if max(counts.values()) / max(1, len(non_ws)) >= 0.80:
            return True
        words = _normalize_text(text).split()
        if len(words) >= 60:
            windows = [" ".join(words[i : i + 8]).lower() for i in range(0, len(words) - 7, 8)]
            if windows and len(set(windows)) <= max(2, len(windows) // 5):
                return True
        return False

    def _stop_audit_unreliable_reason(
        self,
        *,
        review_type: str,
        response: Any,
        content: str,
        parsed: Optional[Dict[str, Any]],
        max_tokens: int,
        stateful_used: bool,
    ) -> str:
        if review_type != "stop" or not stateful_used:
            return ""
        status = self._response_status(response)
        finish_reason = self._response_finish_reason(response)
        completion_tokens = self._response_completion_tokens(response)
        if status == "incomplete":
            return "stateful_response_incomplete"
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            return f"stateful_finish_reason_{finish_reason}"
        if completion_tokens >= max_tokens:
            return "stateful_output_hit_token_cap"
        if not isinstance(parsed, dict):
            return "stateful_invalid_json"
        if "should_intervene" not in parsed or "verdict" not in parsed:
            return "stateful_missing_required_keys"
        if self._looks_degenerate_output(content):
            return "stateful_degenerate_output"
        return ""

    def _append_intervention_ledger(self, ticket: CritiqueTicket, review_type: str) -> None:
        self.state.intervention_ledger.append(
            {
                "id": f"intervention_{len(self.state.intervention_ledger) + 1:03d}",
                "fingerprint": self._issue_fingerprint(ticket, review_type),
                "review_type": review_type,
                "verdict": ticket.verdict,
                "event_index": len(self.state.events),
                "issued_at": time.time(),
                "reason": ticket.reason,
                "required_action": ticket.next_best_action,
                "next_best_action": ticket.next_best_action,
                "recommended_tools": list(ticket.recommended_tools or []),
                "tool_policy": dict(ticket.tool_policy) if ticket.tool_policy else None,
                "evidence": list(ticket.evidence[:5]),
                "criterion_ids": list(ticket.criterion_ids),
                "status": "issued",
            }
        )

    def _find_intervention_entry(self, intervention_id: str) -> Optional[Dict[str, Any]]:
        intervention_id = str(intervention_id or "").strip()
        if not intervention_id:
            return None
        for entry in self.state.intervention_ledger:
            if str(entry.get("id") or "") == intervention_id:
                return entry
        return None

    def _apply_intervention_outcomes_from_review(self, parsed: Dict[str, Any], review_type: str) -> List[Dict[str, Any]]:
        raw_updates = parsed.get("intervention_outcomes")
        if not isinstance(raw_updates, list):
            return []
        applied: List[Dict[str, Any]] = []
        valid_statuses = {"attempted", "resolved", "ignored", "still_failing"}
        for raw_update in raw_updates:
            if not isinstance(raw_update, dict):
                continue
            intervention_id = str(raw_update.get("id") or raw_update.get("intervention_id") or "").strip()
            entry = self._find_intervention_entry(intervention_id)
            if entry is None:
                continue
            status = str(raw_update.get("status") or raw_update.get("outcome_status") or "").strip().lower()
            if status not in valid_statuses:
                continue
            previous_status = str(entry.get("status") or "issued")
            if previous_status == status:
                continue
            if previous_status == "resolved":
                continue
            evidence = raw_update.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            event_index = raw_update.get("event_index")
            try:
                outcome_event_index = int(event_index)
            except Exception:
                outcome_event_index = max(0, len(self.state.events) - 1)
            outcome = str(raw_update.get("outcome") or raw_update.get("reason") or status)
            entry["status"] = status
            entry["outcome"] = _text_head_tail(outcome, 700)
            entry["outcome_event_index"] = outcome_event_index
            entry["outcome_review_type"] = review_type
            entry["outcome_evidence"] = [_text_head_tail(str(item), 400) for item in evidence[:5]]
            if raw_update.get("confidence"):
                entry["outcome_confidence"] = str(raw_update.get("confidence"))
            entry["outcome_updated_at"] = time.time()
            applied.append(
                {
                    "id": intervention_id,
                    "status": status,
                    "outcome_event_index": outcome_event_index,
                }
            )
        return applied

    def _mark_completion_ledger_from_stop_review(self, parsed: Dict[str, Any], should_intervene: bool) -> None:
        if should_intervene:
            for criterion_id in parsed.get("criterion_ids") or []:
                entry = self.state.ledger.get(str(criterion_id))
                if entry:
                    entry.status = "open"
                    entry.last_updated_event = str(max(0, len(self.state.events) - 1))
                    evidence = [str(item) for item in (parsed.get("evidence") or [])[:5]]
                    if evidence:
                        entry.evidence_refs = evidence
            return

        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict not in {"allow_stop", "pass", "complete", "completed", "done"}:
            return
        evidence = [str(item) for item in (parsed.get("evidence") or [])[:8]]
        last_event = str(max(0, len(self.state.events) - 1))
        for entry in self.state.ledger.values():
            entry.status = "done"
            entry.last_updated_event = last_event
            if evidence:
                entry.evidence_refs = evidence

    def _llm_review(self, review_type: str, draft_answer: str, llm_callable, provider: str, model: str) -> ConscienceVerdict:
        payload = self.build_review_payload(review_type, draft_answer)
        stateful_payload = self.build_stateful_review_payload(review_type, draft_answer)
        messages = [
            {
                "role": "system",
                "content": self._review_system_prompt(review_type),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        call_kwargs = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1200,
        }
        if self._callable_accepts_optional_kwarg(llm_callable, "stateful_payload"):
            call_kwargs["stateful_payload"] = {
                "thread_id": f"conscience:{self.state.contract.task_id}",
                "previous_response_id": self.state.stateful_previous_response_id,
                "instructions": self._stateful_review_system_prompt(),
                "input_payload": stateful_payload,
            }
        response = llm_callable(**call_kwargs)
        active_stateful_payload = stateful_payload
        stateful_used = bool(getattr(response, "conscience_stateful_used", False))

        def append_audit_record(response_obj: Any, sent_payload: Dict[str, Any], *, fallback_reason: str = "") -> tuple[str, Optional[Dict[str, Any]], Optional[str], bool]:
            used = bool(getattr(response_obj, "conscience_stateful_used", False))
            fresh_fallback = bool(getattr(response_obj, "conscience_fresh_fallback", False))
            response_id = (
                getattr(response_obj, "conscience_response_id", None)
                or getattr(response_obj, "response_id", None)
                or None
            )
            previous_response_id = getattr(response_obj, "conscience_previous_response_id", None)
            if not previous_response_id and not fresh_fallback and isinstance(call_kwargs.get("stateful_payload"), dict):
                previous_response_id = call_kwargs["stateful_payload"].get("previous_response_id")
            content = self._response_content(response_obj)
            parsed = _extract_json_object(content)
            prompt_tokens = self._response_prompt_tokens(response_obj)
            completion_tokens = self._response_completion_tokens(response_obj)
            total_tokens = self._response_total_tokens(response_obj)
            record = {
                "review_type": review_type,
                "provider": provider,
                "model": model,
                "raw_content": content,
                "parsed": parsed,
                "payload": payload,
                "sent_payload": sent_payload if used else payload,
                "stateful": {
                    "requested": "stateful_payload" in call_kwargs,
                    "used": used,
                    "fresh_fallback": fresh_fallback,
                    "thread_id": f"conscience:{self.state.contract.task_id}",
                    "previous_response_id": previous_response_id,
                    "response_id": response_id,
                    "mode": sent_payload.get("stateful_mode") if isinstance(sent_payload, dict) else None,
                    "event_cursor_end": sent_payload.get("event_cursor_end") if isinstance(sent_payload, dict) else None,
                },
                "response_status": self._response_status(response_obj),
                "finish_reason": self._response_finish_reason(response_obj),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            if fallback_reason:
                record["fallback_from_stateful_reason"] = fallback_reason
            self.state.llm_audits.append(record)
            return content, parsed, response_id, used

        content, parsed, response_id, stateful_used = append_audit_record(response, active_stateful_payload)
        unreliable_reason = self._stop_audit_unreliable_reason(
            review_type=review_type,
            response=response,
            content=content,
            parsed=parsed,
            max_tokens=int(call_kwargs.get("max_tokens") or 0),
            stateful_used=stateful_used,
        )
        if unreliable_reason and self._callable_accepts_optional_kwarg(llm_callable, "stateful_payload"):
            fallback_stateful_payload = {
                "thread_id": f"conscience:{self.state.contract.task_id}:stop-fallback",
                "previous_response_id": None,
                "instructions": self._review_system_prompt(review_type),
                "input_payload": payload,
                "fresh_fallback": True,
                "store": False,
            }
            fallback_kwargs = dict(call_kwargs)
            fallback_kwargs["stateful_payload"] = fallback_stateful_payload
            fallback_response = llm_callable(**fallback_kwargs)
            content, parsed, response_id, stateful_used = append_audit_record(
                fallback_response,
                payload,
                fallback_reason=unreliable_reason,
            )
            self.state.stateful_previous_response_id = None
            self.state.stateful_initialized = False
            self.state.stateful_last_event_index = 0
        elif stateful_used and isinstance(response_id, str) and response_id.strip():
            prompt_tokens = self._response_prompt_tokens(response)
            total_tokens = self._response_total_tokens(response)
            if prompt_tokens:
                self.state.stateful_last_prompt_tokens = prompt_tokens
            if total_tokens:
                self.state.stateful_last_total_tokens = total_tokens
            self.state.stateful_previous_response_id = response_id.strip()
            self.state.stateful_initialized = True
            self.state.stateful_last_event_index = len(self.state.events)
            self.state.stateful_next_init_reason = None
            if (
                self.state.stateful_prompt_token_limit > 0
                and prompt_tokens >= self.state.stateful_prompt_token_limit
            ):
                self._reset_stateful_session("prompt_token_limit_exceeded")
            else:
                self._mark_stateful_memory_synced()
        elif self.state.stateful_previous_response_id and self._callable_accepts_optional_kwarg(llm_callable, "stateful_payload"):
            self.state.stateful_previous_response_id = None
            self.state.stateful_initialized = False
            self.state.stateful_last_event_index = 0

        if not isinstance(parsed, dict):
            return ConscienceVerdict(should_intervene=False, source="llm", metadata={"parse_error": True})

        applied_outcomes = self._apply_intervention_outcomes_from_review(parsed, review_type)
        should_intervene = bool(parsed.get("should_intervene"))
        if review_type == "stop":
            self._mark_completion_ledger_from_stop_review(parsed, should_intervene)
        if not should_intervene:
            metadata = {
                "confidence": str(parsed.get("confidence") or ""),
                "review_type": review_type,
                "verdict": str(
                    parsed.get("verdict")
                    or ("observe" if review_type == "midtask" else "pass")
                ),
            }
            if applied_outcomes:
                metadata["intervention_outcomes_applied"] = applied_outcomes
            return ConscienceVerdict(
                should_intervene=False,
                critique_ticket=None,
                source="llm",
                metadata=metadata,
            )

        raw_tool_policy = parsed.get("tool_policy")
        tool_policy = None
        if isinstance(raw_tool_policy, dict) and raw_tool_policy.get("mode") == "allowlist":
            raw_policy_tools = raw_tool_policy.get("tools")
            if isinstance(raw_policy_tools, list):
                tool_policy = {
                    "mode": "allowlist",
                    "tools": [
                        str(name).strip()
                        for name in raw_policy_tools
                        if str(name).strip()
                    ],
                }

        ticket = CritiqueTicket(
            verdict=str(parsed.get("verdict") or ("repair" if review_type == "midtask" else "block")),
            reason=str(parsed.get("reason") or f"{review_type}_review"),
            evidence=[str(x) for x in (parsed.get("evidence") or [])],
            next_best_action=str(parsed.get("next_best_action") or "Continue with the next best action."),
            criterion_ids=[str(x) for x in (parsed.get("criterion_ids") or [])],
            recommended_tools=(
                [str(x) for x in parsed.get("recommended_tools")]
                if isinstance(parsed.get("recommended_tools"), list)
                else None
            ),
            tool_policy=tool_policy,
            active_tool_decision=(
                str(parsed.get("active_tool_decision") or "").strip().lower()
                if str(parsed.get("active_tool_decision") or "").strip().lower() in {"cancel", "continue"}
                else None
            ),
        )
        suppressed_reason = ""
        suppressed_ticket = None
        admitted, suppressed_reason = self._dedupe_or_admit_ticket(ticket, review_type)
        if not admitted:
            suppressed_ticket = ticket
            should_intervene = False
            ticket = None
        if suppressed_ticket is not None:
            should_intervene = False
        if should_intervene and review_type == "midtask":
            self.state.last_midtask_intervention_index = len(self.state.events)
        if should_intervene and ticket is not None:
            self._append_intervention_ledger(ticket, review_type)
        metadata = {
            "confidence": str(parsed.get("confidence") or ""),
            "review_type": review_type,
        }
        if applied_outcomes:
            metadata["intervention_outcomes_applied"] = applied_outcomes
        if suppressed_reason:
            metadata["suppressed"] = suppressed_reason
            metadata["suppressed_ticket"] = asdict_safe(suppressed_ticket)
        return ConscienceVerdict(
            should_intervene=should_intervene,
            critique_ticket=ticket,
            source="llm",
            metadata=metadata,
        )

    def audit_midtask_progress(
        self,
        *,
        llm_callable=None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ConscienceVerdict:
        if not (llm_callable and provider and model):
            return ConscienceVerdict(should_intervene=False, source="llm", metadata={"skipped": "missing_llm"})
        return self._llm_review("midtask", "", llm_callable, provider, model)

    def audit_stop_decision(
        self,
        draft_answer: str,
        *,
        llm_callable=None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ConscienceVerdict:
        self.record_event(INTENT_TO_STOP, {"text": draft_answer})
        if not (llm_callable and provider and model):
            return ConscienceVerdict(should_intervene=False, source="llm", metadata={"skipped": "missing_llm"})
        return self._llm_review("stop", draft_answer, llm_callable, provider, model)

    def audit_irreversible_action(
        self,
        action_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        llm_callable=None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ConscienceVerdict:
        self.record_event(IRREVERSIBLE_ACTION_INTENT, {"action_type": action_type, **(payload or {})})
        if not (llm_callable and provider and model):
            return ConscienceVerdict(should_intervene=False, source="llm", metadata={"skipped": "missing_llm"})
        return self._llm_review("irreversible_action", "", llm_callable, provider, model)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None
