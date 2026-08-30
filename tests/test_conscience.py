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
    assert calls[1]["input_payload"]["stateful_compaction"]["prompt_token_limit"] == 50_000
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
    assert len(memory["action_ledger"]) <= 24
    assert len(memory["intervention_ledger"]) <= 8
    assert compaction["omitted"]["recent_events"] > 0
    assert compaction["omitted"]["action_ledger"] > 0
    assert compaction["omitted"]["intervention_ledger"] > 0
    assert compaction["payload_chars"] <= compaction["payload_char_budget"] + 5_000
    assert compaction["estimated_payload_tokens"] < monitor.state.stateful_prompt_token_limit // 2


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
