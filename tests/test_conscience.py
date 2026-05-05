import json
from pathlib import Path

from types import SimpleNamespace

from agent.conscience import (
    ARTIFACT_UPDATED,
    PLAN_SUMMARY,
    TOOL_CALL,
    TOOL_RESULT,
    DRAFT_ANSWER,
    ConscienceMonitor,
    extract_task_contract,
)


def test_extract_task_contract_preserves_raw_request_as_single_criterion():
    message = "Implement the plan and save it into new .md file without asking me again."
    contract = extract_task_contract("task1", message)
    assert len(contract.explicit_asks) == 1
    assert contract.explicit_asks[0].source_text == message
    assert contract.raw_user_request == message


def test_contract_keeps_multi_part_request_as_one_semantic_goal():
    message = "Find the bug and also write the fix and then run the tests."
    contract = extract_task_contract("task2", message)
    assert len(contract.explicit_asks) == 1
    assert contract.explicit_asks[0].source_text == message


def test_record_event_appends_clean_artifacts():
    monitor = ConscienceMonitor("task3", "Debug this and run tests.")
    monitor.record_event(PLAN_SUMMARY, {"text": "I will inspect the file and verify the result."})
    monitor.record_event(TOOL_RESULT, {"success": True, "tool_name": "terminal"})
    artifacts = monitor.to_artifacts()
    assert len(artifacts["events"]) == 2
    assert artifacts["events"][0]["event_type"] == PLAN_SUMMARY
    assert artifacts["events"][1]["event_type"] == TOOL_RESULT


def test_to_artifacts_serializes_cleanly(tmp_path):
    monitor = ConscienceMonitor("task4", "Write the report")
    monitor.record_event(PLAN_SUMMARY, {"text": "I will write the report."})
    artifacts = monitor.to_artifacts()
    path = tmp_path / "conscience.json"
    path.write_text(json.dumps(artifacts))
    loaded = json.loads(path.read_text())
    assert "task_contract" in loaded
    assert "completion_ledger" in loaded
    assert "events" in loaded


def test_llm_stop_audit_records_llm_result_and_blocks():
    monitor = ConscienceMonitor("task5", "Save it into new .md file")

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        assert payload["review_type"] == "stop"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "block",
                                "reason": "missing_deliverable",
                                "evidence": ["artifact missing"],
                                "next_best_action": "Write the file.",
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    verdict = monitor.audit_stop_decision(
        "Done.",
        llm_callable=fake_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    assert verdict.should_intervene is True
    assert verdict.source == "llm"
    assert verdict.critique_ticket is not None
    assert verdict.critique_ticket.reason == "missing_deliverable"
    assert monitor.state.llm_audits[-1]["review_type"] == "stop"


def test_llm_midtask_audit_uses_constant_review_payload():
    monitor = ConscienceMonitor("task6", "Fix the bug and verify it")
    monitor.record_event(TOOL_RESULT, {"success": False, "tool_name": "terminal", "error": "same failure"})
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal"})

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        assert payload["review_type"] == "midtask"
        assert payload["recent_events"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "course_correction_needed",
                                "evidence": ["tool attempts are not resolving the task"],
                                "next_best_action": "Inspect the failure and change approach.",
                                "criterion_ids": ["criterion_001"],
                                "confidence": "medium",
                            }
                        )
                    )
                )
            ]
        )

    verdict = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    assert verdict.should_intervene is True
    assert verdict.critique_ticket is not None
    assert verdict.critique_ticket.reason == "course_correction_needed"
    assert verdict.source == "llm"
    assert monitor.state.llm_audits[-1]["review_type"] == "midtask"


def test_same_llm_issue_not_repeated_without_new_evidence():
    monitor = ConscienceMonitor("task7", "Implement the fix and run tests")

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "block",
                                "reason": "same_issue",
                                "evidence": ["still incomplete"],
                                "next_best_action": "Finish it.",
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    first = monitor.audit_stop_decision(
        "done",
        llm_callable=fake_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    second = monitor.audit_stop_decision(
        "done",
        llm_callable=fake_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    assert first.should_intervene is True
    assert second.should_intervene is False
