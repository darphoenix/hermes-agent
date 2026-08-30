import json
from types import SimpleNamespace

from agent.conscience import ConscienceMonitor


def _response(content: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(content)),
            )
        ]
    )


def test_midtask_observe_contract_is_quiet_without_lowering_token_ceiling():
    monitor = ConscienceMonitor("task-1", "Fix the bug", mode="enforce_observe")
    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        return _response({"should_intervene": False, "verdict": "observe"})

    verdict = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:local",
        model="test-model",
    )

    assert verdict.should_intervene is False
    assert verdict.critique_ticket is None
    assert verdict.metadata["verdict"] == "observe"
    assert calls[0]["max_tokens"] == 1200

    system_prompt = calls[0]["messages"][0]["content"]
    assert '{"should_intervene": false, "verdict": "observe"}' in system_prompt
    assert "omit reason, evidence, next_best_action" in system_prompt
    assert "Do not explain why you are observing." in system_prompt


def test_midtask_intervention_still_uses_full_ticket_contract():
    monitor = ConscienceMonitor("task-1", "Fix the bug", mode="enforce_observe")

    def fake_llm(**kwargs):
        return _response(
            {
                "should_intervene": True,
                "verdict": "repair",
                "reason": "same failing command repeated",
                "evidence": ["terminal returned the same error twice"],
                "next_best_action": "Inspect the error and try a different path.",
                "recommended_tools": ["terminal"],
                "criterion_ids": ["criterion_001"],
                "confidence": "high",
            }
        )

    verdict = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:local",
        model="test-model",
    )

    assert verdict.should_intervene is True
    assert verdict.critique_ticket is not None
    assert verdict.critique_ticket.reason == "same failing command repeated"
    assert verdict.critique_ticket.recommended_tools == ["terminal"]


def test_stop_contract_keeps_full_review_reporting():
    prompt = ConscienceMonitor._stop_review_system_prompt()

    assert "reason (string)" in prompt
    assert "evidence (array of strings)" in prompt
    assert "next_best_action (string)" in prompt
    assert "Do not explain why you are observing." not in prompt


def test_stop_contract_allows_harmless_surplus():
    prompt = ConscienceMonitor._stop_review_system_prompt()

    assert "Allow harmless surplus" in prompt
    assert "optional polish, consolidation, cleanup, or broader improvements" in prompt
    assert "Treat that as a style issue, not a completion failure" in prompt


def test_stateful_contract_separates_midtask_and_stop_sections():
    prompt = ConscienceMonitor._stateful_review_system_prompt()

    assert "Apply only the section matching review_type." in prompt
    assert "For review_type='midtask':" in prompt
    assert "For review_type='stop':" in prompt
