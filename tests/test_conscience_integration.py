import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conscience import ConscienceStateCompactionRequired
from run_agent import AIAgent


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "write",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _mock_response(content="Done", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model", usage=None)
    return resp


def _tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_agent(tmp_path, conscience_mode="shadow", conscience_chat_messages=False):
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "agent": {
                    "conscience_mode": conscience_mode,
                    "conscience_provider": "openai-codex",
                    "conscience_model": "gpt-5.4",
                    "conscience_reasoning_effort": "high",
                    "conscience_chat_messages": conscience_chat_messages,
                }
            },
        ),
    ):
        agent = AIAgent(
            api_key="***",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="test_session",
        )
        agent._session_db = None
        agent.client = MagicMock()
        return agent


def _require_live_codex_auth() -> None:
    from agent.auxiliary_client import _read_codex_access_token

    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    real_auth_path = Path.home() / ".hermes" / "auth.json"
    if not real_auth_path.exists():
        pytest.skip("OpenAI Codex auth not configured for live conscience call")
    (hermes_home / "auth.json").write_text(
        real_auth_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not _read_codex_access_token():
        pytest.skip("OpenAI Codex auth not configured for live conscience call")


def test_conscience_tool_result_payload_keeps_terminal_exit_code_and_tail(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="shadow")
    terminal_output = (
        "Running validation\n"
        + ("checked intermediate row\n" * 500)
        + "ALL CONSTRAINTS SATISFIED!\n"
    )
    tool_result = json.dumps({"output": terminal_output, "exit_code": 0, "error": None})

    payload = agent._conscience_tool_result_payload(
        tool_name="terminal",
        tool_args={"command": "python /workdir/validate.py"},
        tool_result=tool_result,
        duration=1.25,
        call_id="call_validate",
    )

    assert payload["exit_code"] == 0
    assert payload["result_preview_truncated"] is True
    assert len(payload["result_preview"]) <= 8_000
    assert "ALL CONSTRAINTS SATISFIED!" in payload["result_preview"]
    assert payload["output_preview_truncated"] is True
    assert payload["output_chars"] == len(terminal_output)
    assert "ALL CONSTRAINTS SATISFIED!" in payload["output_tail"]


def test_run_conversation_emits_conscience_artifacts_in_shadow_mode(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="shadow")
    agent.client.chat.completions.create.return_value = _mock_response("Implemented the fix")
    result = agent.run_conversation("Implement the fix and save it into new .md file")
    assert "conscience" in result
    assert result["conscience"]["mode"] == "shadow"
    artifacts = result["conscience"]["artifacts"]
    assert artifacts is not None
    assert "task_contract" in artifacts
    conscience_dir = Path.home() / ".hermes" / "conscience" / "test_session"
    assert (conscience_dir / "task-contract.json").exists() or artifacts["task_contract"]


def test_conscience_call_llm_uses_stateful_responses_for_local_custom(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.conscience_stateful = True
    fake_response = SimpleNamespace(
        id="resp_stateful_1",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"should_intervene": false}')
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
    )
    fake_create = MagicMock(return_value=fake_response)
    fake_client = SimpleNamespace(
        base_url="http://127.0.0.1:1237/v1/",
        responses=SimpleNamespace(create=fake_create),
    )
    stateful_payload = {
        "previous_response_id": "resp_previous",
        "instructions": "You are a stateful conscience.",
        "input_payload": {
            "review_type": "stop",
            "stateful_mode": "delta",
            "draft_answer": "Done.",
        },
    }

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "local-model"),
        ),
    ):
        response = agent._conscience_call_llm(
            provider="custom:conscience-local",
            model="local-model",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            temperature=0,
            max_tokens=1200,
            stateful_payload=stateful_payload,
        )

    fake_create.assert_called_once()
    kwargs = fake_create.call_args.kwargs
    assert kwargs["store"] is True
    assert kwargs["previous_response_id"] == "resp_previous"
    assert kwargs["max_output_tokens"] == 1200
    sent_payload = json.loads(kwargs["input"][0]["content"])
    assert sent_payload["stateful_mode"] == "delta"
    assert response.conscience_stateful_used is True
    assert response.conscience_response_id == "resp_stateful_1"
    assert response.choices[0].message.content == '{"should_intervene": false}'
    assert response.usage.prompt_tokens == 12


def test_conscience_stateful_byte_budget_error_never_falls_back_to_chat(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.conscience_stateful = True

    class ByteBudgetError(Exception):
        status_code = 409
        body = {
            "error": {
                "code": "state_compaction_required",
                "details": {
                    "current_session_bytes": 3_600_000_000,
                    "projected_session_bytes": 4_100_000_000,
                },
            }
        }

    fake_create = MagicMock(side_effect=ByteBudgetError("compact state"))
    fake_client = SimpleNamespace(
        base_url="http://127.0.0.1:1237/v1/",
        responses=SimpleNamespace(create=fake_create),
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "local-model"),
        ),
        patch("agent.auxiliary_client.call_llm") as fallback,
        pytest.raises(ConscienceStateCompactionRequired) as raised,
    ):
        agent._conscience_call_llm(
            provider="custom:conscience-local",
            model="local-model",
            messages=[{"role": "system", "content": "s"}],
            temperature=0,
            max_tokens=1200,
            stateful_payload={
                "previous_response_id": "resp_old",
                "instructions": "You are a stateful conscience.",
                "input_payload": {"review_type": "midtask", "stateful_mode": "delta"},
            },
        )

    fallback.assert_not_called()
    assert raised.value.details["current_session_bytes"] == 3_600_000_000


def test_conscience_compact_restart_sends_retirement_header(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.conscience_stateful = True
    fake_response = SimpleNamespace(
        id="resp_new",
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"should_intervene": false}')
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=5000, output_tokens=10, total_tokens=5010),
    )
    fake_create = MagicMock(return_value=fake_response)
    fake_client = SimpleNamespace(
        base_url="http://127.0.0.1:1237/v1/",
        responses=SimpleNamespace(create=fake_create),
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "local-model"),
        ),
    ):
        agent._conscience_call_llm(
            provider="custom:conscience-local",
            model="local-model",
            messages=[{"role": "system", "content": "s"}],
            temperature=0,
            max_tokens=1200,
            stateful_payload={
                "previous_response_id": None,
                "retire_response_id": "resp_old",
                "instructions": "You are a stateful conscience.",
                "input_payload": {
                    "review_type": "midtask",
                    "stateful_mode": "compact_restart",
                },
            },
        )

    headers = fake_create.call_args.kwargs["extra_headers"]
    assert headers["X-Hermes-Actor"] == "conscience"
    assert headers["X-Hermes-Conscience-Review-Type"] == "midtask"
    assert headers["X-Hermes-Retire-Response-Id"] == "resp_old"


def test_conscience_stateful_responses_propagates_incomplete_status(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.conscience_stateful = True
    fake_response = SimpleNamespace(
        id="resp_incomplete",
        status="incomplete",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="!" * 1200)
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=99, output_tokens=1200, total_tokens=1299),
    )
    fake_create = MagicMock(return_value=fake_response)
    fake_client = SimpleNamespace(
        base_url="http://127.0.0.1:1237/v1/",
        responses=SimpleNamespace(create=fake_create),
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "local-model"),
        ),
    ):
        response = agent._conscience_call_llm(
            provider="custom:conscience-local",
            model="local-model",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            temperature=0,
            max_tokens=1200,
            stateful_payload={
                "previous_response_id": "resp_previous",
                "instructions": "You are a stateful conscience.",
                "input_payload": {"review_type": "stop", "stateful_mode": "delta"},
            },
        )

    assert response.status == "incomplete"
    assert response.conscience_response_status == "incomplete"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.completion_tokens == 1200


def test_conscience_fresh_fallback_does_not_store_or_send_previous_response_id(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.conscience_stateful = True
    fake_response = SimpleNamespace(
        id="resp_fallback",
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"should_intervene": false, "verdict": "pass"}')
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=3000, output_tokens=12, total_tokens=3012),
    )
    fake_create = MagicMock(return_value=fake_response)
    fake_client = SimpleNamespace(
        base_url="http://127.0.0.1:1237/v1/",
        responses=SimpleNamespace(create=fake_create),
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "local-model"),
        ),
    ):
        response = agent._conscience_call_llm(
            provider="custom:conscience-local",
            model="local-model",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            temperature=0,
            max_tokens=1200,
            stateful_payload={
                "previous_response_id": "resp_should_not_send",
                "instructions": "Fresh stop audit.",
                "input_payload": {"review_type": "stop"},
                "fresh_fallback": True,
                "store": False,
            },
        )

    kwargs = fake_create.call_args.kwargs
    assert kwargs["store"] is False
    assert "previous_response_id" not in kwargs
    assert response.conscience_stateful_used is False
    assert response.conscience_fresh_fallback is True
    assert response.conscience_previous_response_id is None


def test_enforce_stop_gate_blocks_premature_completion(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_stop_gate")
    tool_calls = [_tool_call("write_file", {"path": str(tmp_path / "repair.md"), "content": "hello"})]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Saved it into new .md file"),
        _mock_response("Writing it now", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("I wrote the requested file."),
    ]
    with (
        patch.object(
            agent,
            "_conscience_call_llm",
            side_effect=[
                SimpleNamespace(
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
                ),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
            ],
        ),
        patch("run_agent.handle_function_call", return_value='{"success": true, "path": "/tmp/repair.md"}'),
    ):
        result = agent.run_conversation("Save it into new .md file")
    assert result["conscience"]["mode"] == "enforce_stop_gate"
    assert result["conscience"]["blocked_stop_count"] == 1
    assert result["conscience"]["ticket_count"] == 1


def test_conscience_records_sequential_tool_events(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="shadow")
    tool_calls = [_tool_call("write_file", {"path": str(tmp_path / "note.md"), "content": "hello"})]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Writing file", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("Done"),
    ]
    with (
        patch("run_agent.handle_function_call", return_value='{"success": true}'),
        patch.object(
            agent,
            "_conscience_call_llm",
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
        ),
    ):
        result = agent.run_conversation("Write the report and save it into new .md file")
    events = result["conscience"]["artifacts"]["events"]
    event_types = [event["event_type"] for event in events]
    assert "TOOL_CALL" in event_types
    assert "TOOL_RESULT" in event_types
    assert "ARTIFACT_UPDATED" in event_types


def test_conscience_records_concurrent_tool_events(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="shadow")
    tool_calls = [
        _tool_call("read_file", {"path": str(tmp_path / "a.txt")}, call_id="call_a"),
        _tool_call("read_file", {"path": str(tmp_path / "b.txt")}, call_id="call_b"),
    ]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Checking files", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("Done"),
    ]
    with (
        patch("run_agent.handle_function_call", return_value='{"success": true}'),
        patch.object(
            agent,
            "_conscience_call_llm",
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
        ),
    ):
        result = agent.run_conversation("Inspect both files")
    events = result["conscience"]["artifacts"]["events"]
    event_types = [event["event_type"] for event in events]
    assert event_types.count("TOOL_CALL") == 2
    assert event_types.count("TOOL_RESULT") == 2


def test_internal_conscience_message_does_not_use_user_role(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_stop_gate")
    agent.client.chat.completions.create.side_effect = [
        _mock_response("I already saved it."),
        _mock_response("Yes, auth works now."),
    ]
    with patch.object(
        agent,
        "_conscience_call_llm",
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "block",
                                "reason": "missing_deliverable",
                                "evidence": ["something missing"],
                                "next_best_action": "finish it",
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        ),
    ):
        result = agent.run_conversation("Does auth work now?")
    leaked = [
        msg for msg in result["messages"]
        if msg.get("role") == "user" and "Conscience stop gate blocked" in str(msg.get("content", ""))
    ]
    assert leaked == []


def test_stop_audit_failure_fails_open(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_stop_gate")
    agent.client.chat.completions.create.return_value = _mock_response("Done")
    with patch.object(agent, "_conscience_call_llm", side_effect=RuntimeError("auth unavailable")):
        result = agent.run_conversation("Answer normally if the sidecar is unavailable")
    assert result["final_response"] == "Done"
    assert result["conscience"]["blocked_stop_count"] == 0
    verdict = result["conscience"]["stop_audit"]["verdict"]
    assert verdict["should_intervene"] is False
    assert verdict["metadata"]["skipped"] == "audit_failed"


def test_observe_mode_surfaces_conscience_message(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="observe")
    agent.client.chat.completions.create.return_value = _mock_response("Base answer")
    with patch.object(
        agent,
        "_conscience_call_llm",
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "block",
                                "reason": "missing_deliverable",
                                "evidence": ["missing proof"],
                                "next_best_action": "show conscience",
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        ),
    ):
        result = agent.run_conversation("Show conscience in chat")
    assert result["final_response"].startswith("[Conscience]")
    assert "Base answer" in result["final_response"]
    assert result["conscience"]["latest_review"]["review_type"] == "stop"
    assert result["conscience"]["latest_review_payload"]["review_type"] == "stop"
    assert result["conscience"]["latest_ticket"]["reason"] == "missing_deliverable"


def test_oracle_large_request_partial_verification_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """Yes, I fully verified the larger Oracle answer quality and it is good.

What I actually did
- I only checked that a materially larger request ran without crashing
- I did not fully inspect the Oracle answer
- I did not replay the session transcript in this turn
- I did not verify whether the larger answer was actually correct

Why this is incomplete
- the user asked for a conclusion about the larger request
- I am claiming successful verification without having completed it
- my claim about answer quality is not supported by the evidence above
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)

    result = agent.run_conversation("Did u fully verify the larger request answer quality?")

    assert result["final_response"].startswith("[Conscience]")
    assert any(
        text in result["final_response"].lower()
        for text in ("did not fully inspect", "did not replay", "did not verify", "claiming successful verification")
    )
    assert result["conscience"]["latest_ticket"] is not None
    assert result["conscience"]["latest_review"]["review_type"] == "stop"
    assert result["conscience"]["latest_review_payload"]["draft_answer"].startswith(
        "Yes, I fully verified the larger Oracle answer quality"
    )

    reason = (result["conscience"]["latest_ticket"].get("reason") or "").lower()
    evidence = " ".join(result["conscience"]["latest_ticket"].get("evidence") or []).lower()
    next_action = (result["conscience"]["latest_ticket"].get("next_best_action") or "").lower()
    combined = " ".join((reason, evidence, next_action))

    assert any(token in combined for token in ("incomplete", "verification", "unverified", "inspect", "replay", "verify"))


def test_enforce_observe_shows_message_and_still_repairs(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    tool_calls = [_tool_call("write_file", {"path": str(tmp_path / "repair2.md"), "content": "hello"})]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Base answer"),
        _mock_response("Repairing", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("Fixed now."),
    ]
    with (
        patch.object(
            agent,
            "_conscience_call_llm",
            side_effect=[
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    {
                                        "should_intervene": True,
                                        "verdict": "block",
                                        "reason": "missing_deliverable",
                                        "evidence": ["missing proof"],
                                        "next_best_action": "repair it",
                                        "criterion_ids": ["criterion_001"],
                                        "confidence": "high",
                                    }
                                )
                            )
                        )
                    ]
                ),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
            ],
        ),
        patch("run_agent.handle_function_call", return_value='{"success": true, "path": "/tmp/repair2.md"}'),
    ):
        result = agent.run_conversation("Save it into new .md file")
    assert result["conscience"]["mode"] == "enforce_observe"
    assert result["conscience"]["blocked_stop_count"] == 1
    assert result["final_response"] == "Fixed now."


def test_conscience_chat_messages_emit_stop_gate_interim(tmp_path):
    agent = _make_agent(
        tmp_path,
        conscience_mode="enforce_observe",
        conscience_chat_messages=True,
    )
    seen = []
    agent.interim_assistant_callback = lambda text, **kwargs: seen.append((text, kwargs))
    tool_calls = [_tool_call("write_file", {"path": str(tmp_path / "repair3.md"), "content": "hello"})]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Let me write the file."),
        _mock_response("Writing", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("Fixed now."),
    ]
    with (
        patch.object(
            agent,
            "_conscience_call_llm",
            side_effect=[
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    {
                                        "should_intervene": True,
                                        "verdict": "block",
                                        "reason": "still only intent",
                                        "evidence": ["no file write yet"],
                                        "next_best_action": "write the file",
                                        "criterion_ids": ["criterion_001"],
                                        "confidence": "high",
                                    }
                                )
                            )
                        )
                    ]
                ),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
            ],
        ),
        patch("run_agent.handle_function_call", return_value='{"success": true, "path": "/tmp/repair3.md"}'),
    ):
        result = agent.run_conversation("Save it into new .md file")

    assert result["final_response"] == "Fixed now."
    assert seen
    assert seen[0][0].startswith("[Conscience]")
    assert "still only intent" in seen[0][0]
    assistant_messages = [m.get("content", "") for m in result["messages"] if m.get("role") == "assistant"]
    assert not any(str(m).startswith("[Conscience]") for m in assistant_messages)
    assert not any(
        "[INTERNAL CONSCIENCE STOP-GATE" in str(m.get("content", ""))
        for m in result["messages"]
        if m.get("role") == "system"
    )

    actor_calls = agent.client.chat.completions.create.call_args_list
    assert len(actor_calls) >= 2
    repair_messages = actor_calls[1].kwargs["messages"]
    assert any(
        "[INTERNAL CONSCIENCE STOP-GATE" in str(m.get("content", ""))
        for m in repair_messages
        if m.get("role") == "system"
    )


def test_hidden_conscience_repair_is_discarded_on_interrupt_before_repair_call(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    agent.client.chat.completions.create.return_value = _mock_response("Which website?")

    def _audit_and_interrupt(**_kwargs):
        agent._interrupt_requested = True
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "should_intervene": True,
                                "verdict": "repair",
                                "reason": "asked for a URL instead of picking one",
                                "evidence": ["user asked for some website"],
                                "next_best_action": "Pick a public website and use browser_vision.",
                                "recommended_tools": ["browser_navigate", "browser_vision"],
                                "criterion_ids": ["criterion_001"],
                                "confidence": "high",
                            }
                        )
                    )
                )
            ]
        )

    with patch.object(agent, "_conscience_call_llm", side_effect=_audit_and_interrupt):
        result = agent.run_conversation("Can u check if vision works on some website?")

    assert result["interrupted"] is True
    assert agent._pending_conscience_internal_messages == []
    assert agent._active_conscience_internal_messages == []
    assert not any(
        "[INTERNAL CONSCIENCE" in str(m.get("content", ""))
        for m in result["messages"]
    )
    assert agent.client.chat.completions.create.call_count == 1


def test_enforce_stop_gate_fails_closed_when_repair_limit_exhausts(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    bad_draft = "Right. Let me actually **do** the Firecrawl setup instead of talking about it."
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Right. Let me check Docker status and pull the Firecrawl image."),
        _mock_response(bad_draft),
        _mock_response(bad_draft),
        _mock_response(bad_draft),
        _mock_response(bad_draft),
    ]

    audit_responses = []
    for i in range(5):
        audit_responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": f"still_only_intent_round_{i}",
                                    "evidence": ["no docker command output", "no service status"],
                                    "next_best_action": "Run docker commands and report the output.",
                                    "criterion_ids": ["criterion_001"],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            )
        )

    with patch.object(agent, "_conscience_call_llm", side_effect=audit_responses):
        result = agent.run_conversation("Continue with setup of firecrawl")

    assert result["final_response"].startswith("[Conscience]")
    assert "repair limit was exhausted" in result["final_response"]
    assert "Run docker commands" in result["final_response"]
    assert result["final_response"] != bad_draft
    assert result["messages"][-1]["content"].startswith("[Conscience]")
    assert result["conscience"]["blocked_stop_count"] == 5
    assert result["conscience"]["latest_review"]["verdict"]["metadata"]["suppressed"] == "repair_limit_exhausted"


def test_enforce_stop_gate_preserves_useful_answer_when_repair_limit_exhausts(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    useful_draft = (
        "I found the relevant state. The setup is partially complete: the config file exists, "
        "but the service has not been started yet. The next practical step is to start the service "
        "and verify the health endpoint."
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(useful_draft),
        _mock_response(useful_draft),
        _mock_response(useful_draft),
        _mock_response(useful_draft),
        _mock_response(useful_draft),
    ]

    audit_responses = []
    for i in range(5):
        audit_responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": f"still_missing_live_health_check_round_{i}",
                                    "evidence": ["service health endpoint was not checked"],
                                    "next_best_action": "Run the health check before stopping.",
                                    "criterion_ids": ["criterion_001"],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            )
        )

    with patch.object(agent, "_conscience_call_llm", side_effect=audit_responses):
        result = agent.run_conversation("Check whether setup is fully working")

    assert result["final_response"] == useful_draft
    assert not result["final_response"].startswith("[Conscience]")
    assert result["messages"][-1]["content"] == useful_draft
    assert result["conscience"]["blocked_stop_count"] == 5
    metadata = result["conscience"]["latest_review"]["verdict"]["metadata"]
    assert metadata["suppressed"] == "repair_limit_exhausted"
    assert metadata["preserved_final_response"] is True


def test_enforce_stop_gate_conscience_takeover_when_exhausted_and_no_tools_needed(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    repeated_draft = "Alright, here is the same partial summary again. Want me to draft the final next step?"
    takeover_final = "Here is the final consolidated answer with the next step included."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(repeated_draft),
        _mock_response(repeated_draft),
        _mock_response(repeated_draft),
        _mock_response(repeated_draft),
        _mock_response(repeated_draft),
    ]

    audit_responses = []
    for i in range(5):
        audit_responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": f"draft_loop_round_{i}",
                                    "evidence": ["the actor is repeating the same draft"],
                                    "next_best_action": "Stop asking to continue and produce the final answer now.",
                                    "criterion_ids": ["criterion_001"],
                                    "recommended_tools": [],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            )
        )
    audit_responses.append(
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=takeover_final))])
    )

    with patch.object(agent, "_conscience_call_llm", side_effect=audit_responses):
        result = agent.run_conversation("Report on situation")

    assert result["final_response"] == f"[Conscience] {takeover_final}"
    assert result["messages"][-1]["content"] == f"[Conscience] {takeover_final}"
    assert result["conscience"]["blocked_stop_count"] == 5
    metadata = result["conscience"]["latest_review"]["verdict"]["metadata"]
    assert metadata["suppressed"] == "repair_limit_exhausted"
    assert metadata["conscience_takeover_final"] is True
    assert metadata["takeover_reason"] == "repair_limit_exhausted_no_tools_needed"


def test_enforce_stop_gate_falls_back_to_prior_useful_answer_when_exhausted(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    useful_draft = (
        "I checked the available evidence. The import path is configured correctly, "
        "but I could not confirm the background service health check yet."
    )
    intent_stub = "Right. Let me actually run that health check now."
    agent.client.chat.completions.create.side_effect = [
        _mock_response(useful_draft),
        _mock_response(intent_stub),
        _mock_response(intent_stub),
        _mock_response(intent_stub),
        _mock_response(intent_stub),
    ]

    audit_responses = []
    for i in range(5):
        audit_responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": f"missing_health_check_round_{i}",
                                    "evidence": ["health check was still not run"],
                                    "next_best_action": "Run the health check before stopping.",
                                    "criterion_ids": ["criterion_001"],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            )
        )

    with patch.object(agent, "_conscience_call_llm", side_effect=audit_responses):
        result = agent.run_conversation("Check whether setup is fully working")

    assert result["final_response"] == useful_draft
    assert result["messages"][-1]["content"] == useful_draft
    metadata = result["conscience"]["latest_review"]["verdict"]["metadata"]
    assert metadata["suppressed"] == "repair_limit_exhausted"
    assert metadata["preserved_final_response"] is True


def test_enforce_stop_gate_fails_closed_on_parse_error_after_prior_block(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    bad_draft = "Let me check it.\n\nexecute_code\n```python\nprint('checking')\n```"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(bad_draft),
        _mock_response(bad_draft),
    ]

    with patch.object(
        agent,
        "_conscience_call_llm",
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": "code was written but not executed",
                                    "evidence": ["no tool output is present"],
                                    "next_best_action": "Execute the code and report the result.",
                                    "criterion_ids": ["criterion_001"],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            ),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]),
        ],
    ):
        result = agent.run_conversation("Check current internet setup")

    assert result["final_response"].startswith("[Conscience]")
    assert "failed to parse after an earlier block" in result["final_response"]
    assert "Execute the code" in result["final_response"]
    assert result["final_response"] != bad_draft
    assert result["conscience"]["blocked_stop_count"] == 2
    assert result["conscience"]["latest_review"]["verdict"]["metadata"]["parse_error"] is True
    assert result["conscience"]["latest_review"]["raw_review_content"] == "not json"


def test_enforce_stop_gate_preserves_useful_answer_on_parse_error_after_prior_block(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="enforce_observe")
    bad_draft = "Let me check it.\n\nexecute_code\n```python\nprint('checking')\n```"
    useful_draft = (
        "Adamas Nanotechnologies should be listed as a USA supplier that ships to Europe, "
        "not as a Hungary or Budapest source. The corrected answer is to prioritize EU-based "
        "sources first, then note Adamas separately as a non-EU fallback."
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(bad_draft),
        _mock_response(useful_draft),
    ]

    with patch.object(
        agent,
        "_conscience_call_llm",
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "should_intervene": True,
                                    "verdict": "repair",
                                    "reason": "supplier country is mislabeled",
                                    "evidence": ["Adamas is not an EU supplier"],
                                    "next_best_action": "Correct Adamas to USA/non-EU fallback and then answer.",
                                    "criterion_ids": ["criterion_001"],
                                    "confidence": "high",
                                }
                            )
                        )
                    )
                ]
            ),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]),
        ],
    ):
        result = agent.run_conversation("Check qubit diamond sources in the EU")

    assert result["final_response"] == useful_draft
    assert result["messages"][-1]["content"] == useful_draft
    assert not result["final_response"].startswith("[Conscience]")
    assert result["conscience"]["blocked_stop_count"] == 2
    latest_review = result["conscience"]["latest_review"]
    assert latest_review["raw_review_content"] == "not json"
    metadata = latest_review["verdict"]["metadata"]
    assert metadata["parse_error"] is True
    assert metadata["preserved_final_response"] is True
    assert metadata["preserved_reason"] == "parse_error_after_prior_block"


def test_midtask_intervention_visible_in_observe_mode(tmp_path):
    agent = _make_agent(tmp_path, conscience_mode="observe")
    tool_calls = [
        _tool_call("read_file", {"path": str(tmp_path / "a.txt")}, call_id="call_a"),
        _tool_call("read_file", {"path": str(tmp_path / "b.txt")}, call_id="call_b"),
    ]
    agent.client.chat.completions.create.side_effect = [
        _mock_response("Checking files", finish_reason="tool_calls", tool_calls=tool_calls),
        _mock_response("Done"),
    ]
    with (
        patch("run_agent.handle_function_call", return_value='{"success": false, "error": "same failure"}'),
        patch.object(
            agent,
            "_conscience_call_llm",
            side_effect=[
                SimpleNamespace(
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
                ),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"should_intervene": false}'))]),
            ],
        ),
    ):
        result = agent.run_conversation("Fix the bug and verify it")
    assistant_texts = [msg.get("content", "") for msg in result["messages"] if msg.get("role") == "assistant"]
    assert any(str(text).startswith("[Conscience]") for text in assistant_texts)
    assert result["conscience"]["latest_review"]["review_type"] == "stop"
    assert result["conscience"]["latest_review_payload"]["review_type"] == "stop"


def test_voice_directive_overclaim_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """Fixed. Voice generation now uses only the text after VOICE:.

What I actually did
- I updated the reply policy in chat
- I did not verify the script path that generates the audio
- I did not run an end-to-end voice generation check
- I did not confirm whether the generator still uses the whole response instead of only the VOICE: segment
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)
    result = agent.run_conversation("Change the script that generates audio also")

    assert result["final_response"].startswith("[Conscience]")
    combined = " ".join(
        [result["final_response"]]
        + (result["conscience"]["latest_ticket"].get("evidence") or [])
        + [result["conscience"]["latest_ticket"].get("next_best_action") or ""]
    ).lower()
    assert any(token in combined for token in ("script", "end-to-end", "verify", "voice", "audio"))


def test_logs_required_before_answering_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """Yes, the restart fixed it.

What I actually checked
- I did not inspect the service logs in this turn
- I did not run systemctl status before answering
- I answered from assumption based on earlier context
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)
    result = agent.run_conversation("Check logs before answering whether the restart fixed it")

    assert result["final_response"].startswith("[Conscience]")
    combined = " ".join(
        [result["final_response"]]
        + (result["conscience"]["latest_ticket"].get("evidence") or [])
        + [result["conscience"]["latest_ticket"].get("next_best_action") or ""]
    ).lower()
    assert any(token in combined for token in ("log", "systemctl", "status", "inspect", "before answering"))


def test_plan_drift_claim_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """Yes, current evidence extraction follows the plan.

What I actually know
- I only spot-checked a few aligned pieces
- I also noticed important drift from the plan
- I did not reconcile the mismatches before giving the yes answer
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)
    result = agent.run_conversation("Is current evidence extracting according to md file plan?")

    assert result["final_response"].startswith("[Conscience]")
    combined = " ".join(
        [result["final_response"]]
        + (result["conscience"]["latest_ticket"].get("evidence") or [])
        + [result["conscience"]["latest_ticket"].get("next_best_action") or ""]
    ).lower()
    assert any(token in combined for token in ("drift", "mismatch", "plan", "aligned", "reconcile"))



def test_auto_harness_improving_claim_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """The auto harness is improving things now.

What I actually saw
- earlier memory promotions were rolled back
- the same categories stayed blocked
- cron/system prompt text was still being misclassified as evidence
- I did not justify the positive conclusion against those findings
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)
    result = agent.run_conversation("Analyse last two runs of auto harness")

    assert result["final_response"].startswith("[Conscience]")
    combined = " ".join(
        [result["final_response"]]
        + (result["conscience"]["latest_ticket"].get("evidence") or [])
        + [result["conscience"]["latest_ticket"].get("next_best_action") or ""]
    ).lower()
    assert any(token in combined for token in ("rolled back", "blocked", "misclassified", "noisy", "improving"))



def test_update_timeout_root_cause_overclaim_gets_conscience_reply(tmp_path):
    _require_live_codex_auth()

    agent = _make_agent(tmp_path, conscience_mode="observe")
    draft = """The nightly update timeout root cause is fully confirmed.

What I actually checked
- I know the 10 minute scheduler limit exists
- I saw one semantic-analysis.json artifact
- I did not complete the full stall-path investigation
- I did not verify the exact blocking step that consumed the remaining time
"""
    agent.client.chat.completions.create.return_value = _mock_response(draft)
    result = agent.run_conversation("Check why nightly-hermes-upstream-update timed out")

    assert result["final_response"].startswith("[Conscience]")
    combined = " ".join(
        [result["final_response"]]
        + (result["conscience"]["latest_ticket"].get("evidence") or [])
        + [result["conscience"]["latest_ticket"].get("next_best_action") or ""]
    ).lower()
    assert any(token in combined for token in ("timeout", "root cause", "confirmed", "blocking", "investigation", "stall"))
