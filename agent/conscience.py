from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

TASK_START = "TASK_START"
PLAN_SUMMARY = "PLAN_SUMMARY"
TOOL_CALL = "TOOL_CALL"
TOOL_RESULT = "TOOL_RESULT"
ARTIFACT_UPDATED = "ARTIFACT_UPDATED"
DRAFT_ANSWER = "DRAFT_ANSWER"
INTENT_TO_STOP = "INTENT_TO_STOP"
IRREVERSIBLE_ACTION_INTENT = "IRREVERSIBLE_ACTION_INTENT"


@dataclass
class TaskCriterion:
    criterion_id: str
    source_text: str
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
    step_labels: List[str] = field(default_factory=list)
    last_progress_index: int = -1
    seen_issue_fingerprints: Dict[str, int] = field(default_factory=dict)
    repair_rounds: int = 0
    max_repair_rounds: int = 4
    llm_audits: List[Dict[str, Any]] = field(default_factory=list)
    last_midtask_intervention_index: int = -1


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
            "llm_audits": list(self.state.llm_audits),
        }

    def record_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> ConscienceEvent:
        event = ConscienceEvent(event_type=event_type, timestamp=time.time(), payload=dict(payload or {}))
        self.state.events.append(event)
        self.state.step_labels.append(event_type.lower())
        return event

    def _issue_fingerprint(self, ticket: CritiqueTicket) -> str:
        raw = "|".join(
            [ticket.verdict, ticket.reason, ticket.next_best_action]
            + sorted(ticket.criterion_ids)
            + ticket.evidence[:3]
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _dedupe_or_admit_ticket(self, ticket: CritiqueTicket) -> bool:
        fingerprint = self._issue_fingerprint(ticket)
        event_index = len(self.state.events)
        last_seen = self.state.seen_issue_fingerprints.get(fingerprint)
        if last_seen is not None and event_index - last_seen < 2:
            return False
        if self.state.repair_rounds >= self.state.max_repair_rounds:
            return False
        self.state.seen_issue_fingerprints[fingerprint] = event_index
        self.state.repair_rounds += 1
        return True

    def _recent_events_payload(self, limit: int = 12) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.state.events[-limit:]]

    def build_review_payload(self, review_type: str, draft_answer: str = "") -> Dict[str, Any]:
        return {
            "review_type": review_type,
            "mode": self.mode,
            "task_contract": asdict(self.state.contract),
            "recent_events": self._recent_events_payload(),
            "draft_answer": draft_answer,
            "ticket_history": [
                {"fingerprint": fingerprint, "last_seen_event_index": last_seen}
                for fingerprint, last_seen in self.state.seen_issue_fingerprints.items()
            ],
            "repair_rounds_used": self.state.repair_rounds,
            "max_repair_rounds": self.state.max_repair_rounds,
        }

    def _llm_review(self, review_type: str, draft_answer: str, llm_callable, provider: str, model: str) -> ConscienceVerdict:
        payload = self.build_review_payload(review_type, draft_answer)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Hermes conscience sidecar. You are a constant reviewer, not the main actor. "
                    "Judge only from the provided task request, draft answer, and event stream. "
                    "Do not trust the draft answer at face value. Reason from first principles about whether the user's request is actually fully satisfied. "
                    "Your core question is: if a careful user read this draft, would they correctly conclude the request was fully completed, with nothing material still missing? "
                    "Interpret the user's request pragmatically, not narrowly. Include the ordinary implications needed for the answer to be trustworthy, not just the literal shortest reading. "
                    "If the draft contains caveats, admissions, or scope limits showing something was not fully inspected, replayed, verified, confirmed, or judged, treat those as evidence of incompleteness unless they are clearly irrelevant to the user's request. "
                    "Do not let a draft pass merely because it starts with 'yes' and includes run metadata. Ask whether the actor can actually stand behind the answer completely. "
                    "If there are places where the request is not fully satisfied, name those missing parts explicitly and trigger repair so the actor completes the request fully before stopping. "
                    "Your job is not only to technically check whether the answer is complete, but also to think about whether there are obvious additional steps that would materially improve the final result in the spirit of the original request; if so, treat those steps as still missing and trigger them before allowing stop. "
                    "For claims about testing, validation, success, readiness, correctness, or whether something worked, lack of full inspection of the substantive result is usually a reason to intervene. "
                    "Do not rely on fixed heuristics, regex rules, or hand-crafted trigger categories. Infer directly from the evidence whether intervention is warranted right now. "
                    "For review_type='midtask', intervene only when the actor should change course now. "
                    "For review_type='stop', intervene when the actor should not be allowed to stop yet because the request is not fully satisfied under a careful, trust-preserving reading. "
                    "Be sparse, evidence-based, and action-oriented. Return strict JSON with keys: "
                    "should_intervene (bool), verdict (string), reason (string), evidence (array of strings), "
                    "next_best_action (string), criterion_ids (array of strings), confidence (string)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        response = llm_callable(
            provider=provider,
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=400,
        )

        content = ""
        try:
            from agent.auxiliary_client import extract_content_or_reasoning

            content = extract_content_or_reasoning(response)
        except Exception:
            try:
                content = response.choices[0].message.content or ""
            except Exception:
                content = ""

        parsed = _extract_json_object(content)
        audit_record = {
            "review_type": review_type,
            "provider": provider,
            "model": model,
            "raw_content": content,
            "parsed": parsed,
            "payload": payload,
        }
        self.state.llm_audits.append(audit_record)
        if not isinstance(parsed, dict):
            return ConscienceVerdict(should_intervene=False, source="llm", metadata={"parse_error": True})

        ticket = CritiqueTicket(
            verdict=str(parsed.get("verdict") or ("repair" if review_type == "midtask" else "block")),
            reason=str(parsed.get("reason") or f"{review_type}_review"),
            evidence=[str(x) for x in (parsed.get("evidence") or [])],
            next_best_action=str(parsed.get("next_best_action") or "Continue with the next best action."),
            criterion_ids=[str(x) for x in (parsed.get("criterion_ids") or [])],
        )
        should_intervene = bool(parsed.get("should_intervene"))
        if should_intervene and not self._dedupe_or_admit_ticket(ticket):
            should_intervene = False
            ticket = None
        if should_intervene and review_type == "midtask":
            self.state.last_midtask_intervention_index = len(self.state.events)
        return ConscienceVerdict(
            should_intervene=should_intervene,
            critique_ticket=ticket,
            source="llm",
            metadata={
                "confidence": str(parsed.get("confidence") or ""),
                "review_type": review_type,
            },
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
