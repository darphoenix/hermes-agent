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
    ConscienceMemoryPressureError,
    ConscienceMonitor,
    extract_task_contract,
)


def test_extract_task_contract_preserves_raw_request_as_single_criterion():
    message = "Implement the plan and save it into new .md file without asking me again."
    contract = extract_task_contract("task1", message)
    assert len(contract.explicit_asks) == 1
    assert contract.explicit_asks[0].source_text == message
    assert contract.raw_user_request == message


def test_review_prompts_recognize_native_image_delivery_as_direct_inspection():
    marker = "TASK_START.native_image_delivery.delivered_to_actor"

    assert marker in ConscienceMonitor._midtask_review_system_prompt()
    assert marker in ConscienceMonitor._stop_review_system_prompt()


def test_midtask_prompt_judges_semantic_loops_by_information_gain():
    prompt = ConscienceMonitor._midtask_review_system_prompt()

    assert "same underlying investigative objective" in prompt
    assert "information gained" in prompt
    assert "movement across the user's explicit deliverables" in prompt
    assert "information gathering alone is not deliverable progress" in prompt
    assert "minimal end-to-end slice" in prompt
    assert "tolerant parsing or explicit fallbacks" in prompt
    assert "preceding two actor/tool cycles" in prompt
    assert "decision-changing evidence" in prompt
    assert "intervene now rather than waiting for another variant" in prompt
    assert "new parser error" in prompt
    assert "actually retrieves needed evidence" in prompt
    assert "materially different source searches" in prompt
    assert "preserving that measurement as unavailable or uncertain" in prompt
    assert "unless that measurement blocks the whole task" in prompt
    assert "Do not recommend another variant of the same investigation" in prompt


def test_contract_keeps_multi_part_request_as_one_semantic_goal():
    message = "Find the bug and also write the fix and then run the tests."
    contract = extract_task_contract("task2", message)
    assert len(contract.explicit_asks) == 1
    assert contract.explicit_asks[0].source_text == message


def test_contract_flattens_multimodal_user_request_without_image_payload():
    message = [
        {"type": "text", "text": "Inspect this screenshot directly."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,very-large-payload"},
        },
    ]

    contract = extract_task_contract("task-vision", message)

    assert contract.raw_user_request == "Inspect this screenshot directly."
    assert contract.explicit_asks[0].source_text == contract.raw_user_request
    assert "very-large-payload" not in json.dumps(contract.raw_user_request)


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


def test_active_repair_contract_persists_until_all_checks_resolve():
    monitor = ConscienceMonitor("task-repair", "Verify Docker, search, and browser")
    monitor.record_event(
        TASK_START,
        {"available_tools": ["terminal", "web_search", "browser_exec"]},
    )
    responses = iter(
        [
            {
                "should_intervene": True,
                "verdict": "block_stop",
                "reason": "verification claims are unsupported",
                "evidence": ["the required checks have no tool results"],
                "next_best_action": "Run all three verification checks.",
                "recommended_tools": ["terminal", "web_search", "browser_exec"],
                "criterion_ids": ["criterion_001"],
                "confidence": "high",
                "repair_contract": {
                    "objective": "Collect evidence for every verification claim.",
                    "checks": [
                        {
                            "id": "docker",
                            "description": "Verify the Docker services.",
                            "expected_evidence": "A successful docker ps result.",
                            "recommended_tools": ["terminal"],
                            "status": "pending",
                        },
                        {
                            "id": "search",
                            "description": "Verify local search.",
                            "expected_evidence": "A clean search result.",
                            "recommended_tools": ["web_search"],
                            "status": "pending",
                        },
                        {
                            "id": "browser",
                            "description": "Verify browser control.",
                            "expected_evidence": "A browser title returned by the tool.",
                            "recommended_tools": ["browser_exec"],
                            "status": "pending",
                        },
                    ],
                },
            },
            {
                "should_intervene": False,
                "verdict": "observe",
                "repair_check_updates": [
                    {
                        "id": "docker",
                        "status": "resolved",
                        "outcome": "Docker services are running.",
                        "evidence": ["docker ps exited 0"],
                    }
                ],
            },
            {
                "should_intervene": False,
                "verdict": "observe",
                "repair_check_updates": [
                    {
                        "id": "search",
                        "status": "resolved",
                        "outcome": "Search returned clean results.",
                        "evidence": ["search success true"],
                    },
                    {
                        "id": "browser",
                        "status": "resolved",
                        "outcome": "Browser returned the page title.",
                        "evidence": ["Example Domain"],
                    },
                ],
            },
        ]
    )

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        if payload["review_type"] == "midtask":
            assert payload["active_repair_contract"]["status"] == "active"
        content = json.dumps(next(responses))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    verdict = monitor.audit_stop_decision(
        "Everything works.",
        llm_callable=fake_llm,
        provider="custom",
        model="local",
    )
    assert verdict.should_intervene is True
    assert monitor.active_repair_contract_payload() is not None

    monitor.record_event(TOOL_CALL, {"tool_name": "terminal"})
    monitor.record_event(TOOL_RESULT, {"tool_name": "terminal", "success": True})
    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom",
        model="local",
    )
    contract = monitor.active_repair_contract_payload()
    assert contract is not None
    statuses = {check["id"]: check["status"] for check in contract["checks"]}
    assert statuses == {"docker": "resolved", "search": "pending", "browser": "pending"}

    monitor.record_event(TOOL_CALL, {"tool_name": "web_search"})
    monitor.record_event(TOOL_RESULT, {"tool_name": "web_search", "success": True})
    monitor.record_event(TOOL_CALL, {"tool_name": "browser_exec"})
    monitor.record_event(TOOL_RESULT, {"tool_name": "browser_exec", "success": True})
    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom",
        model="local",
    )
    assert monitor.active_repair_contract_payload() is None
    resolved = monitor.active_repair_contract_payload(include_resolved=True)
    assert resolved is not None
    assert resolved["status"] == "resolved"
    assert all(check["status"] == "resolved" for check in resolved["checks"])


def test_active_structured_repair_contract_blocks_allow_stop_until_resolved():
    monitor = ConscienceMonitor("task-repair-gate", "Verify Docker and browser")
    responses = iter(
        [
            {
                "should_intervene": True,
                "verdict": "block_stop",
                "reason": "verification missing",
                "next_best_action": "Verify Docker and browser.",
                "recommended_tools": ["terminal", "browser_exec"],
                "repair_contract": {
                    "objective": "Verify both claims.",
                    "checks": [
                        {
                            "id": "docker",
                            "description": "Verify Docker.",
                            "recommended_tools": ["terminal"],
                        },
                        {
                            "id": "browser",
                            "description": "Verify browser control.",
                            "recommended_tools": ["browser_exec"],
                        },
                    ],
                },
            },
            {"should_intervene": False, "verdict": "allow_stop"},
            {
                "should_intervene": False,
                "verdict": "allow_stop",
                "repair_check_updates": [
                    {
                        "id": "docker",
                        "status": "resolved",
                        "outcome": "Docker verified.",
                        "evidence": ["docker ps exited 0"],
                    },
                    {
                        "id": "browser",
                        "status": "resolved",
                        "outcome": "Browser verified.",
                        "evidence": ["browser title returned"],
                    },
                ],
            },
        ]
    )

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(next(responses)))
                )
            ]
        )

    first = monitor.audit_stop_decision(
        "Done.", llm_callable=fake_llm, provider="custom", model="local"
    )
    assert first.should_intervene is True

    premature = monitor.audit_stop_decision(
        "Docker is verified.",
        llm_callable=fake_llm,
        provider="custom",
        model="local",
    )
    assert premature.should_intervene is True
    assert premature.critique_ticket is not None
    assert premature.critique_ticket.reason == "active_repair_contract_incomplete"
    assert premature.metadata["active_repair_contract_enforced"] is True
    assert set(premature.critique_ticket.recommended_tools or []) == {
        "terminal",
        "browser_exec",
    }

    complete = monitor.audit_stop_decision(
        "Both checks are verified.",
        llm_callable=fake_llm,
        provider="custom",
        model="local",
    )
    assert complete.should_intervene is False
    assert complete.metadata["repair_check_updates_applied"] == [
        {"id": "docker", "status": "resolved", "review_type": "stop"},
        {"id": "browser", "status": "resolved", "review_type": "stop"},
    ]
    assert monitor.active_repair_contract_payload() is None


def test_repeated_structured_stop_cannot_drop_an_existing_pending_check():
    monitor = ConscienceMonitor("task-repair-merge", "Verify Docker and browser")
    responses = iter(
        [
            {
                "should_intervene": True,
                "verdict": "block_stop",
                "reason": "verification missing",
                "next_best_action": "Verify Docker and browser.",
                "repair_contract": {
                    "objective": "Verify both claims.",
                    "checks": [
                        {"id": "docker", "description": "Verify Docker."},
                        {"id": "browser", "description": "Verify browser control."},
                    ],
                },
            },
            {
                "should_intervene": True,
                "verdict": "block_stop",
                "reason": "Docker is still unverified",
                "next_best_action": "Verify Docker.",
                "repair_contract": {
                    "objective": "Verify both claims.",
                    "checks": [
                        {"id": "docker", "description": "Verify Docker now."}
                    ],
                },
            },
        ]
    )

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(next(responses)))
                )
            ]
        )

    monitor.audit_stop_decision(
        "Done.", llm_callable=fake_llm, provider="custom", model="local"
    )
    monitor.audit_stop_decision(
        "Docker only.", llm_callable=fake_llm, provider="custom", model="local"
    )

    contract = monitor.active_repair_contract_payload()
    assert contract is not None
    assert [check["id"] for check in contract["checks"]] == ["docker", "browser"]
    assert all(check["status"] == "pending" for check in contract["checks"])
    assert contract["checks"][0]["description"] == "Verify Docker now."


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
                                "tool_policy": {
                                    "mode": "allowlist",
                                    "tools": ["web_search"],
                                },
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
    assert verdict.critique_ticket.tool_policy == {
        "mode": "allowlist",
        "tools": ["web_search"],
    }


def test_progress_review_parses_explicit_active_tool_cancel_decision():
    monitor = ConscienceMonitor("task-progress", "Find an efficient exact solution")
    monitor.record_event(
        TOOL_RESULT,
        {"tool_name": "write_file", "success": True, "result_preview": "wrote brute-force solver"},
    )

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "course_correction",
                                "reason": "current search is nonproductive",
                                "evidence": ["no output after the progress interval"],
                                "next_best_action": "Cancel and derive a pruned search.",
                                "recommended_tools": ["terminal", "write_file"],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                                "active_tool_decision": "cancel",
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

    assert verdict.critique_ticket is not None
    assert verdict.critique_ticket.active_tool_decision == "cancel"


def test_stateful_conscience_sends_full_payload_then_delta():
    monitor = ConscienceMonitor("task-stateful", "Check current setup")
    monitor.record_event(TASK_START, {"available_tools": ["terminal", "web_search"]})
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "pwd"})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        review_type = (stateful_payload or {}).get("input_payload", {}).get("review_type")
        content = {"should_intervene": False, "verdict": "pass" if review_type == "stop" else "observe"}
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_stateful_used=True,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(content))
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
    assert "trajectory_memory" not in calls[1]["input_payload"]
    assert "ledger_delta" in calls[1]["input_payload"]
    assert [event["event_type"] for event in calls[1]["input_payload"]["new_events"]] == [
        TOOL_RESULT,
        INTENT_TO_STOP,
    ]
    assert [row["event_type"] for row in calls[1]["input_payload"]["ledger_delta"]["action_append"]] == [
        TOOL_RESULT,
        INTENT_TO_STOP,
    ]
    latest_audit = monitor.state.llm_audits[-1]
    assert latest_audit["stateful"]["used"] is True
    assert latest_audit["stateful"]["previous_response_id"] == "resp_1"
    assert latest_audit["stateful"]["response_id"] == "resp_2"


def test_stateful_conscience_compacts_after_prompt_token_limit():
    monitor = ConscienceMonitor("task-stateful-compact", "Check current setup")
    monitor.record_event(TASK_START, {"available_tools": ["terminal"]})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        prompt_tokens = 50_001 if len(calls) == 1 else 4000
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_response_id=f"resp_{len(calls)}",
            conscience_previous_response_id=(stateful_payload or {}).get("previous_response_id"),
            conscience_stateful_used=True,
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=10,
                total_tokens=prompt_tokens + 10,
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"should_intervene": False, "verdict": "observe"})
                    ),
                )
            ],
        )

    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    assert monitor.state.stateful_previous_response_id is None
    assert monitor.state.stateful_initialized is False
    assert monitor.state.stateful_reset_count == 1

    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "pwd"})
    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    assert calls[1]["previous_response_id"] is None
    assert calls[1]["input_payload"]["stateful_mode"] == "compact_restart"
    assert calls[1]["input_payload"]["stateful_compaction"]["prompt_token_limit"] == 36_000
    assert calls[1]["input_payload"]["stateful_compaction"]["last_prompt_tokens"] == 50_001
    assert calls[1]["input_payload"]["recent_events"]
    assert "trajectory_memory" in calls[1]["input_payload"]
    assert monitor.state.stateful_previous_response_id == "resp_2"
    assert monitor.state.stateful_initialized is True


def test_stateful_conscience_compacts_before_projected_delta_exceeds_limit():
    monitor = ConscienceMonitor("task-stateful-projected-compact", "Check current setup")
    monitor.state.stateful_prompt_token_limit = 100_000
    monitor.record_event(TASK_START, {"available_tools": ["terminal"]})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        prompt_tokens = 94_000 if len(calls) == 1 else 5000
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_response_id=f"resp_{len(calls)}",
            conscience_previous_response_id=(stateful_payload or {}).get("previous_response_id"),
            conscience_stateful_used=True,
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=10,
                total_tokens=prompt_tokens + 10,
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"should_intervene": False, "verdict": "observe"})
                    ),
                )
            ],
        )

    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": True,
            "result_preview": "x" * 20_000,
        },
    )
    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    assert calls[1]["previous_response_id"] is None
    assert calls[1]["input_payload"]["stateful_mode"] == "compact_restart"
    assert calls[1]["input_payload"]["stateful_compaction"]["reason"] == "projected_prompt_token_limit_exceeded"
    assert "trajectory_memory" in calls[1]["input_payload"]


def test_stateful_conscience_compacts_and_retires_parent_on_memory_pressure():
    monitor = ConscienceMonitor("task-memory-pressure", "Check current setup")
    monitor.record_event(TASK_START, {"available_tools": ["terminal"]})
    monitor.state.stateful_initialized = True
    monitor.state.stateful_previous_response_id = "resp_old"
    monitor.state.stateful_last_prompt_tokens = 20_000
    monitor.state.stateful_last_event_index = 1
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "pwd"})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        if len(calls) == 1:
            raise ConscienceMemoryPressureError(
                "compact",
                details={"code": "conscience_compaction_required"},
            )
        return SimpleNamespace(
            response_id="resp_compact",
            conscience_response_id="resp_compact",
            conscience_previous_response_id=None,
            conscience_stateful_used=True,
            usage=SimpleNamespace(
                prompt_tokens=4_000,
                completion_tokens=10,
                total_tokens=4_010,
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"should_intervene": False, "verdict": "observe"}
                        )
                    ),
                )
            ],
        )

    verdict = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    assert verdict.should_intervene is False
    assert calls[0]["previous_response_id"] == "resp_old"
    assert calls[0]["input_payload"]["stateful_mode"] == "delta"
    assert calls[1]["previous_response_id"] is None
    assert calls[1]["retire_previous_response_id"] == "resp_old"
    assert calls[1]["input_payload"]["stateful_mode"] == "compact_restart"
    assert calls[1]["input_payload"]["stateful_compaction"]["reason"] == "wrapper_memory_admission"
    assert monitor.state.stateful_previous_response_id == "resp_compact"
    assert monitor.state.stateful_retire_response_id is None
    assert monitor.state.llm_audits[-1]["stateful"]["memory_pressure_retry"] is True


def test_stateful_conscience_compact_restart_is_budgeted():
    monitor = ConscienceMonitor("task-stateful-budgeted-compact", "Fix the solver and verify sol.csv")
    monitor.state.stateful_initialized = True
    monitor.state.stateful_last_prompt_tokens = 50_001
    monitor.record_event(TASK_START, {"available_tools": ["terminal", "read_file", "write_file"]})

    for i in range(36):
        monitor.record_event(
            TOOL_CALL,
            {
                "tool_name": "terminal",
                "tool_args": json.dumps(
                    {
                        "command": f"python solve.py --attempt {i}",
                        "content": "print('large edit')\n" * 600,
                    }
                ),
            },
        )
        monitor.record_event(
            TOOL_RESULT,
            {
                "tool_name": "terminal",
                "success": i % 3 == 0,
                "duration_seconds": 12.5,
                "result_preview": ("same parse failure with large traceback\n" * 240),
            },
        )
        monitor.state.intervention_ledger.append(
            {
                "id": f"intervention_{i:03d}",
                "status": "attempted",
                "event_index": i,
                "reason": "The actor is repeating the same failing solver strategy. " * 20,
                "evidence": ["same traceback repeated " * 20, "no new validation evidence " * 20],
                "next_best_action": "Change approach, inspect the data shape, and verify the output. " * 20,
                "recommended_tools": ["terminal", "read_file"],
                "last_result_preview": "long result preview " * 200,
            }
        )

    payload = monitor.build_stateful_review_payload("stop", "The task is done." * 200)
    compaction = payload["stateful_compaction"]
    memory = payload["trajectory_memory"]

    assert payload["stateful_mode"] == "compact_restart"
    assert len(payload["recent_events"]) <= 12
    assert len(memory["action_ledger"]) <= 8
    assert len(memory["intervention_ledger"]) <= 8
    assert compaction["omitted"]["recent_events"] > 0
    assert compaction["omitted"]["action_ledger"] > 0
    assert compaction["omitted"]["intervention_ledger"] > 0
    assert len(json.dumps(payload, ensure_ascii=False)) <= compaction["payload_char_budget"]
    assert compaction["estimated_payload_tokens"] < monitor.state.stateful_prompt_token_limit // 2


def test_stateful_conscience_compact_restart_bounds_one_multifield_tool_result():
    monitor = ConscienceMonitor("task-stateful-one-huge-event", "Inspect the failure and repair it")
    monitor.state.stateful_initialized = True
    monitor.state.stateful_last_prompt_tokens = 50_001
    monitor.record_event(TASK_START, {"available_tools": ["terminal", "read_file"]})
    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": False,
            "exit_code": 1,
            "duration_seconds": 90.0,
            "result_preview": "traceback\n" * 12_000,
            "output_head": "head\n" * 12_000,
            "output_tail": "tail\n" * 12_000,
            "content": "duplicate\n" * 12_000,
            "artifact_path": "/workdir/.hermes_tool_outputs/result.txt",
        },
    )

    payload = monitor.build_stateful_review_payload("midtask")
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["stateful_mode"] == "compact_restart"
    assert len(encoded) <= payload["stateful_compaction"]["payload_char_budget"]
    event = payload["recent_events"][-1]
    assert event["raw_event_truncated"] is True
    assert event["payload"]["exit_code"] == 1
    assert event["payload"]["artifact_path"].endswith("result.txt")


def test_stateful_conscience_delta_does_not_resend_full_trajectory_memory():
    monitor = ConscienceMonitor("task-stateful-true-delta", "Check current setup")
    monitor.record_event(TASK_START, {"available_tools": ["terminal"]})
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "pwd"})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_response_id=f"resp_{len(calls)}",
            conscience_previous_response_id=(stateful_payload or {}).get("previous_response_id"),
            conscience_stateful_used=True,
            usage=SimpleNamespace(prompt_tokens=5000 + (100 * len(calls)), completion_tokens=10, total_tokens=5010 + (100 * len(calls))),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"should_intervene": False, "verdict": "observe"})
                    ),
                )
            ],
        )

    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": True,
            "result_preview": "ok",
        },
    )
    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    delta = calls[1]["input_payload"]
    assert delta["stateful_mode"] == "delta"
    assert "trajectory_memory" not in delta
    assert "ticket_history" not in delta
    assert delta["ledger_delta"]["action_append"] == [
        {
            "event_index": 2,
            "event_type": TOOL_RESULT,
            "tool_name": "terminal",
            "success": True,
            "result_preview": "ok",
        }
    ]
    assert delta["ledger_delta"]["artifact_upsert"] == []
    assert delta["ledger_delta"]["intervention_upsert"] == []
    assert len(json.dumps(delta, ensure_ascii=False)) < len(json.dumps(calls[0]["input_payload"], ensure_ascii=False))


def test_stateful_conscience_delta_carries_new_intervention_once():
    monitor = ConscienceMonitor("task-stateful-intervention-delta", "Fix the parser")
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        if len(calls) == 1:
            content = {
                "should_intervene": True,
                "verdict": "repair",
                "reason": "parser still failing",
                "evidence": ["same parse error repeated"],
                "next_best_action": "Debug parser from observed file structure.",
                "recommended_tools": ["terminal"],
                "criterion_ids": ["criterion_001"],
                "confidence": "high",
            }
        else:
            content = {"should_intervene": False, "verdict": "observe"}
        return SimpleNamespace(
            response_id=f"resp_{len(calls)}",
            conscience_response_id=f"resp_{len(calls)}",
            conscience_previous_response_id=(stateful_payload or {}).get("previous_response_id"),
            conscience_stateful_used=True,
            usage=SimpleNamespace(prompt_tokens=4000 + (100 * len(calls)), completion_tokens=20, total_tokens=4020 + (100 * len(calls))),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(content)),
                )
            ],
        )

    verdict = monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    assert verdict.should_intervene is True

    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    intervention_delta = calls[1]["input_payload"]["ledger_delta"]["intervention_upsert"]
    assert [entry["id"] for entry in intervention_delta] == ["intervention_001"]
    assert intervention_delta[0]["status"] == "issued"

    monitor.audit_midtask_progress(
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )
    assert calls[2]["input_payload"]["ledger_delta"]["intervention_upsert"] == []


def test_review_payload_compacts_large_tool_payload_but_artifacts_keep_raw_events():
    monitor = ConscienceMonitor("task-large", "Write solve.py and validate it")
    large_content = "print('x')\n" * 2000
    tool_args = json.dumps({"path": "/workdir/solve.py", "content": large_content})
    monitor.record_event(TOOL_CALL, {"tool_name": "write_file", "tool_args": tool_args})

    artifacts = monitor.to_artifacts()
    raw_tool_args = json.loads(artifacts["events"][0]["payload"]["tool_args"])
    assert raw_tool_args["content"] == large_content

    payload = monitor.build_review_payload("midtask")
    compact_tool_args = payload["recent_events"][0]["payload"]["tool_args"]
    assert compact_tool_args["path"] == "/workdir/solve.py"
    assert compact_tool_args["content"]["kind"] == "large_text_summary"
    assert compact_tool_args["content"]["chars"] == len(large_content)
    assert compact_tool_args["content"]["sha256"]
    assert len(json.dumps(payload)) < len(tool_args)

    artifact_ledger = payload["trajectory_memory"]["artifact_ledger"]
    assert artifact_ledger[-1]["path"] == "/workdir/solve.py"
    assert artifact_ledger[-1]["last_content_chars"] == len(large_content)
    assert artifact_ledger[-1]["last_content_sha256"] == compact_tool_args["content"]["sha256"]


def test_stateful_stop_audit_retries_fresh_when_stateful_output_is_unreliable():
    monitor = ConscienceMonitor("task-stop-fallback", "Create sol.csv with the required header")
    monitor.state.stateful_previous_response_id = "resp_prev"
    monitor.state.stateful_initialized = True
    monitor.state.stateful_last_event_index = 1
    monitor.record_event(TOOL_RESULT, {"tool_name": "terminal", "success": True, "result_preview": "wrote sol.csv"})
    calls = []

    def fake_llm(*, provider, model, messages, temperature, max_tokens, stateful_payload=None):
        calls.append(stateful_payload)
        if len(calls) == 1:
            return SimpleNamespace(
                response_id="resp_bad",
                conscience_response_id="resp_bad",
                conscience_previous_response_id="resp_prev",
                conscience_stateful_used=True,
                conscience_response_status="incomplete",
                usage=SimpleNamespace(completion_tokens=max_tokens, prompt_tokens=99, total_tokens=99 + max_tokens),
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="!" * max_tokens),
                    )
                ],
            )
        return SimpleNamespace(
            response_id="resp_fresh",
            conscience_response_id="resp_fresh",
            conscience_previous_response_id=None,
            conscience_stateful_used=False,
            conscience_fresh_fallback=True,
            conscience_response_status="",
            usage=SimpleNamespace(completion_tokens=40, prompt_tokens=3000, total_tokens=3040),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "block",
                                "reason": "missing_csv_header",
                                "evidence": ["sol.csv was written without the expected header"],
                                "next_best_action": "Add the required header and revalidate sol.csv.",
                                "recommended_tools": ["terminal"],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    ),
                )
            ],
        )

    verdict = monitor.audit_stop_decision(
        "Done.",
        llm_callable=fake_llm,
        provider="custom:conscience-local",
        model="local-model",
    )

    assert verdict.should_intervene is True
    assert verdict.critique_ticket is not None
    assert verdict.critique_ticket.reason == "missing_csv_header"
    assert len(calls) == 2
    assert calls[0]["previous_response_id"] == "resp_prev"
    assert calls[1]["fresh_fallback"] is True
    assert calls[1]["previous_response_id"] is None
    assert monitor.state.stateful_previous_response_id is None
    assert monitor.state.stateful_initialized is False
    assert monitor.state.stateful_last_event_index == 0
    assert monitor.state.llm_audits[-1]["fallback_from_stateful_reason"] == "stateful_response_incomplete"
    assert monitor.state.llm_audits[-1]["stateful"]["fresh_fallback"] is True
    assert monitor.state.llm_audits[-1]["stateful"]["previous_response_id"] is None
    assert monitor.state.intervention_ledger[-1]["reason"] == "missing_csv_header"


def test_intervention_ledger_marks_attempted_and_records_tool_result():
    monitor = ConscienceMonitor("task-ledger", "Fix the parser")

    def fake_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "parser still failing",
                                "evidence": ["same parse error repeated"],
                                "next_best_action": "Debug parser from observed file structure.",
                                "recommended_tools": ["read_file", "write_file", "terminal"],
                                "criterion_ids": ["criterion_001"],
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
    entry = monitor.state.intervention_ledger[-1]
    assert entry["id"] == "intervention_001"
    assert entry["status"] == "issued"
    assert entry["required_action"] == "Debug parser from observed file structure."

    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "python solve.py"})
    assert entry["status"] == "attempted"
    assert entry["followed_tool"] == "terminal"
    assert entry["followed_event_index"] == len(monitor.state.events) - 1

    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": False,
            "exit_code": 1,
            "result_preview_truncated": True,
            "output_preview_truncated": True,
            "output_chars": 12_345,
            "output_tail": "ValueError: parser failed on book block",
            "error": "parser failed on book block",
            "result_preview": "ValueError: parser failed on book block",
        },
    )
    assert entry["last_result_event_index"] == len(monitor.state.events) - 1
    assert entry["last_result_success"] is False
    assert entry["last_exit_code"] == 1
    assert entry["last_output_preview_truncated"] is True
    assert entry["last_output_chars"] == 12_345
    assert "parser failed" in entry["last_result_error"]
    assert "parser failed" in entry["last_output_tail"]
    action_row = monitor._action_ledger_payload(limit=1)[0]
    assert action_row["exit_code"] == 1
    assert action_row["output_preview_truncated"] is True
    assert action_row["output_chars"] == 12_345
    assert "parser failed" in action_row["output_tail"]


def test_llm_review_updates_intervention_outcome_without_new_intervention():
    monitor = ConscienceMonitor("task-ledger-outcome", "Fix the parser")

    def initial_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "parser still failing",
                                "evidence": ["same parse error repeated"],
                                "next_best_action": "Debug parser from observed file structure.",
                                "recommended_tools": ["terminal"],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    monitor.audit_midtask_progress(
        llm_callable=initial_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "python solve.py"})
    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": True,
            "result_preview": "parser now parses 103 books",
        },
    )

    def outcome_llm(*, provider, model, messages, temperature, max_tokens):
        payload = json.loads(messages[1]["content"])
        prior = payload["trajectory_memory"]["intervention_ledger"][-1]
        assert prior["id"] == "intervention_001"
        assert prior["status"] == "attempted"
        assert "parser now parses 103 books" in prior["last_result_preview"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": False,
                                "verdict": "observe",
                                "intervention_outcomes": [
                                    {
                                        "id": "intervention_001",
                                        "status": "resolved",
                                        "outcome": "Parser now parses 103 books.",
                                        "evidence": ["terminal output says parser now parses 103 books"],
                                        "confidence": "high",
                                    }
                                ],
                            }
                        )
                    )
                )
            ]
        )

    verdict = monitor.audit_midtask_progress(
        llm_callable=outcome_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )

    assert verdict.should_intervene is False
    entry = monitor.state.intervention_ledger[-1]
    assert entry["status"] == "resolved"
    assert entry["outcome"] == "Parser now parses 103 books."
    assert entry["outcome_review_type"] == "midtask"
    assert entry["outcome_confidence"] == "high"
    assert "terminal output" in entry["outcome_evidence"][0]
    assert verdict.metadata["intervention_outcomes_applied"] == [
        {"id": "intervention_001", "status": "resolved", "outcome_event_index": len(monitor.state.events) - 1}
    ]


def test_intervention_outcome_ignores_same_status_narration():
    monitor = ConscienceMonitor("task-transition-only", "Fix the parser")
    monitor.state.intervention_ledger.append(
        {
            "id": "intervention_001",
            "status": "attempted",
            "outcome": "initial attempt",
            "outcome_updated_at": 10.0,
        }
    )

    applied = monitor._apply_intervention_outcomes_from_review(
        {
            "intervention_outcomes": [
                {
                    "id": "intervention_001",
                    "status": "attempted",
                    "outcome": "another exploratory command ran",
                    "evidence": ["command started"],
                }
            ]
        },
        "midtask",
    )

    assert applied == []
    assert monitor.state.intervention_ledger[0]["outcome"] == "initial attempt"
    assert monitor.state.intervention_ledger[0]["outcome_updated_at"] == 10.0


def test_artifacts_preserve_ticket_history_outcomes_and_done_ledger():
    monitor = ConscienceMonitor("task-ticket-history", "Create sol.csv and verify it")

    def intervention_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "solver is looping",
                                "evidence": ["same timeout repeated"],
                                "next_best_action": "Switch to SQL-backed search and verify sol.csv.",
                                "recommended_tools": ["terminal", "write_file"],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    monitor.audit_midtask_progress(
        llm_callable=intervention_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )
    monitor.record_event(TOOL_CALL, {"tool_name": "terminal", "tool_args": "python solve.py"})
    monitor.record_event(
        TOOL_RESULT,
        {
            "tool_name": "terminal",
            "success": True,
            "result_preview": "wrote /workdir/sol.csv and verified all constraints",
        },
    )

    def outcome_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": False,
                                "verdict": "observe",
                                "intervention_outcomes": [
                                    {
                                        "id": "intervention_001",
                                        "status": "resolved",
                                        "outcome": "Actor wrote sol.csv and verified all constraints.",
                                        "evidence": ["terminal result says verified all constraints"],
                                        "confidence": "high",
                                    }
                                ],
                            }
                        )
                    )
                )
            ]
        )

    monitor.audit_midtask_progress(
        llm_callable=outcome_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )

    def stop_llm(*, provider, model, messages, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": False,
                                "verdict": "allow_stop",
                                "reason": "task complete",
                                "evidence": ["sol.csv exists", "all constraints verified"],
                                "next_best_action": "Allow stop.",
                                "recommended_tools": [],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    monitor.audit_stop_decision(
        "Done.",
        llm_callable=stop_llm,
        provider="openai-codex",
        model="gpt-5.4",
    )

    artifacts = monitor.to_artifacts()
    assert artifacts["completion_ledger"]["criterion_001"]["status"] == "done"
    assert artifacts["completion_ledger"]["criterion_001"]["evidence_refs"] == [
        "sol.csv exists",
        "all constraints verified",
    ]
    assert len(artifacts["critique_tickets"]) == 1
    ticket = artifacts["critique_tickets"][0]
    assert ticket["id"] == "intervention_001"
    assert ticket["verdict"] == "repair"
    assert ticket["status"] == "resolved"
    assert ticket["outcome"] == "Actor wrote sol.csv and verified all constraints."
    assert ticket["recommended_tools"] == ["terminal", "write_file"]


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
