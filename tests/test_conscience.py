import json
from pathlib import Path

from types import SimpleNamespace

from agent.conscience import (
    ARTIFACT_UPDATED,
    PLAN_SUMMARY,
    TASK_START,
    TOOL_CALL,
    TOOL_RESULT,
    DRAFT_ANSWER,
    INTENT_TO_STOP,
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
                                "recommended_tools": ["write_file"],
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
    assert verdict.critique_ticket.recommended_tools == ["write_file"]
    assert monitor.state.llm_audits[-1]["review_type"] == "stop"


def test_llm_review_payload_includes_available_tools_and_parses_recommendation():
    monitor = ConscienceMonitor("task5b", "Find the current source and verify it")
    monitor.record_event(TASK_START, {"available_tools": ["web_extract", "web_search", "web_search", ""]})

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        assert payload["available_tools"] == ["web_extract", "web_search"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "needs_current_source",
                                "evidence": ["no source checked"],
                                "next_best_action": "Search and extract the current source.",
                                "criterion_ids": ["criterion_001"],
                                "recommended_tools": ["web_search", "web_extract"],
                                "confidence": "high",
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
    assert verdict.critique_ticket.recommended_tools == ["web_search", "web_extract"]


def test_stateful_conscience_sends_full_payload_then_delta():
    monitor = ConscienceMonitor("task-stateful", "Check current setup")
    monitor.record_event(TASK_START, {"available_tools": ["terminal", "web_search"]})
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "pwd"})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_stateful_used=True,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"should_intervene": false}')
                )
            ],
        )

    first = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    monitor.record_event(TOOL_RESULT, {"tool_name": "terminal", "success": True, "result": "/tmp"})
    second = monitor.audit_stop_decision(
        "The setup is working.",
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    assert first.should_intervene is False
    assert second.should_intervene is False
    assert calls[0]["previous_response_id"] is None
    assert calls[0]["input_payload"]["stateful_mode"] == "init_full"
    assert "task_contract" in calls[0]["input_payload"]
    assert calls[1]["previous_response_id"] == "resp_1"
    assert calls[1]["input_payload"]["stateful_mode"] == "delta"
    assert "task_contract" not in calls[1]["input_payload"]
    assert [event["event_type"] for event in calls[1]["input_payload"]["new_events"]] == [
        TOOL_RESULT,
        INTENT_TO_STOP,
    ]
    latest_audit = monitor.state.llm_audits[-1]
    assert latest_audit["stateful"]["used"] is True
    assert latest_audit["stateful"]["previous_response_id"] == "resp_1"
    assert latest_audit["stateful"]["response_id"] == "resp_2"


def test_llm_midtask_audit_uses_sparse_trajectory_prompt():
    monitor = ConscienceMonitor("task6", "Fix the bug and verify it")
    monitor.record_event(TOOL_RESULT, {"success": False, "tool_name": "terminal", "error": "same failure"})
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal"})

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        system_prompt = messages[0]["content"]
        assert "sparse trajectory monitor" in system_prompt
        assert "Default to should_intervene=false" in system_prompt
        assert "strict completion auditor" not in system_prompt
        assert "fully completed" not in system_prompt
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


def test_llm_stop_audit_uses_strict_completion_prompt():
    monitor = ConscienceMonitor("task-stop", "Verify the setup")
    monitor.record_event(TOOL_RESULT, {"success": True, "tool_name": "terminal", "result": "ok"})

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        system_prompt = messages[0]["content"]
        assert "strict completion auditor" in system_prompt
        assert "fully completed" in system_prompt
        assert "sparse trajectory monitor" not in system_prompt
        assert "Default to should_intervene=false" not in system_prompt
        payload = json.loads(messages[1]["content"])
        assert payload["review_type"] == "stop"
        assert payload["draft_answer"] == "Done."
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": False,
                                "verdict": "pass",
                                "reason": "complete",
                                "evidence": ["tool result ok"],
                                "next_best_action": "",
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

    assert verdict.should_intervene is False
    assert monitor.state.llm_audits[-1]["review_type"] == "stop"


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
    assert second.metadata["suppressed"] == "duplicate_recent_issue"
    assert second.metadata["suppressed_ticket"]["reason"] == "same_issue"


def test_midtask_loop_breakout_is_not_limited_by_stop_repair_budget():
    monitor = ConscienceMonitor("task8", "Recover from the loop")
    monitor.state.repair_rounds = monitor.state.max_repair_rounds

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        assert payload["review_type"] == "midtask"
        assert payload["repair_rounds_used"] == monitor.state.max_repair_rounds
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "loop detected",
                                "evidence": ["same browser action repeated"],
                                "next_best_action": "Stop repeating the browser action and synthesize from gathered evidence.",
                                "criterion_ids": ["criterion_001"],
                                "recommended_tools": [],
                                "confidence": "high",
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
    assert verdict.metadata.get("suppressed") is None
    assert monitor.state.repair_rounds == monitor.state.max_repair_rounds
