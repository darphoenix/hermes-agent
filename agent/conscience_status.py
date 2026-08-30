from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_conscience_status_summary(conscience: dict | None = None, artifact_dir: str | None = None) -> dict[str, Any]:
    artifact_path = Path(artifact_dir).expanduser() if artifact_dir else None
    if artifact_path is None and conscience:
        raw_dir = conscience.get("artifact_dir")
        if raw_dir:
            artifact_path = Path(str(raw_dir)).expanduser()

    if artifact_path is None:
        return {
            "available": False,
            "reason": "no_conscience_artifacts",
        }

    last_review = _load_json(artifact_path / "last-review.json", {})
    last_review_payload = _load_json(artifact_path / "last-review-payload.json", {})
    critique_tickets = _load_json(artifact_path / "critique-tickets.json", [])
    intervention_ledger = _load_json(artifact_path / "intervention-ledger.json", [])
    if not critique_tickets and isinstance(intervention_ledger, list):
        critique_tickets = intervention_ledger
    stop_audit = _load_json(artifact_path / "stop-audit.json", {})
    task_contract = _load_json(artifact_path / "task-contract.json", {})
    completion_ledger = _load_json(artifact_path / "completion-ledger.json", {})

    latest_ticket = critique_tickets[-1] if critique_tickets else None
    review_type = last_review.get("review_type") or stop_audit.get("review_type")
    open_criteria = last_review.get("open_criteria") or stop_audit.get("open_criteria") or []
    unresolved = []
    if isinstance(task_contract, dict) and isinstance(completion_ledger, dict):
        for criterion in task_contract.get("explicit_asks") or []:
            criterion_id = criterion.get("criterion_id")
            entry = completion_ledger.get(criterion_id) if criterion_id else None
            if not entry or entry.get("status") != "done":
                unresolved.append(criterion)
    if unresolved:
        open_criteria = unresolved

    payload_preview = last_review_payload or {}
    payload_json = json.dumps(payload_preview, ensure_ascii=False, indent=2) if payload_preview else "{}"

    return {
        "available": True,
        "artifact_dir": str(artifact_path),
        "review_type": review_type,
        "latest_ticket": latest_ticket,
        "ticket_count": len(critique_tickets) if isinstance(critique_tickets, list) else 0,
        "intervention_ledger": intervention_ledger,
        "last_review": last_review,
        "last_review_payload": payload_preview,
        "payload_json": payload_json,
        "open_criteria": open_criteria,
        "stop_audit": stop_audit,
        "task_contract": task_contract,
        "completion_ledger": completion_ledger,
    }


def format_conscience_status_text(summary: Dict[str, Any]) -> str:
    if not summary.get("available"):
        return "Conscience status unavailable. No conscience artifacts found yet."

    latest_ticket = summary.get("latest_ticket") or {}
    last_review = summary.get("last_review") or {}
    verdict = (last_review.get("verdict") or {}) if isinstance(last_review.get("verdict"), dict) else {}
    open_criteria = summary.get("open_criteria") or []

    lines = [
        "Conscience status",
        f"- review type: {summary.get('review_type') or 'unknown'}",
        f"- artifact dir: {summary.get('artifact_dir')}",
        f"- intervened: {bool(verdict.get('should_intervene'))}",
        f"- tickets: {summary.get('ticket_count') or 0}",
        f"- blocked/intervened reason: {latest_ticket.get('reason') or verdict.get('critique_ticket', {}).get('reason') or verdict.get('metadata', {}).get('review_type') or 'none'}",
    ]

    if latest_ticket:
        lines.append(f"- latest critique verdict: {latest_ticket.get('verdict') or 'unknown'}")
        lines.append(f"- latest critique next action: {latest_ticket.get('next_best_action') or 'n/a'}")
        evidence = latest_ticket.get("evidence") or []
        if evidence:
            lines.append(f"- evidence: {'; '.join(str(x) for x in evidence[:3])}")

    if open_criteria:
        lines.append("- open criteria:")
        for criterion in open_criteria[:5]:
            lines.append(f"  - {criterion.get('source_text') or criterion}")
    else:
        lines.append("- open criteria: none")

    lines.append("- exact sidecar payload:")
    lines.append(summary.get("payload_json") or "{}")
    return "\n".join(lines)


def latest_conscience_artifact_dir() -> str | None:
    conscience_root = get_hermes_home() / "conscience"
    if not conscience_root.exists():
        return None
    candidates = [p for p in conscience_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)
