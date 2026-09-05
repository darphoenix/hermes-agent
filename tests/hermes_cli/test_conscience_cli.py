from pathlib import Path
from unittest.mock import patch

from run_agent import AIAgent
from agent.conscience_status import build_conscience_status_summary, format_conscience_status_text


def test_conscience_command_is_registered_for_gateway():
    from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, GATEWAY_KNOWN_COMMANDS, resolve_command

    command = resolve_command("conscience")

    assert command is not None
    assert command.name == "conscience"
    assert "conscience" in GATEWAY_KNOWN_COMMANDS
    assert "conscience" in ACTIVE_SESSION_BYPASS_COMMANDS


def test_agent_loads_conscience_mode_from_config():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "agent": {
                    "conscience_mode": "enforce_stop_gate",
                    "conscience_provider": "openai-codex",
                    "conscience_model": "gpt-5.4",
                    "conscience_reasoning_effort": "xhigh",
                }
            },
        ),
    ):
        agent = AIAgent(
            api_key="***",
            base_url="https://chatgpt.com/backend-api/codex",
            provider="openai-codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.conscience_mode == "enforce_stop_gate"
        assert agent.conscience_provider == "openai-codex"
        assert agent.conscience_model == "gpt-5.4"
        assert agent.conscience_reasoning_config == {"enabled": True, "effort": "xhigh"}


def test_conscience_status_summary_formats_latest_review(tmp_path):
    conscience_dir = tmp_path / "conscience"
    conscience_dir.mkdir()
    (conscience_dir / "last-review.json").write_text(
        '{"review_type":"stop","verdict":{"should_intervene":true},"open_criteria":[{"source_text":"answer the question"}]}'
    )
    (conscience_dir / "last-review-payload.json").write_text('{"review_type":"stop","draft_answer":"draft"}')
    (conscience_dir / "critique-tickets.json").write_text(
        '[{"verdict":"block","reason":"missing_deliverable","next_best_action":"finish it","evidence":["missing answer"]}]'
    )
    (conscience_dir / "stop-audit.json").write_text('{}')
    (conscience_dir / "task-contract.json").write_text('{"explicit_asks":[{"criterion_id":"criterion_001","source_text":"answer the question"}]}')
    (conscience_dir / "completion-ledger.json").write_text('{"criterion_001":{"status":"open"}}')
    (conscience_dir / "active-repair-contract.json").write_text(
        '{"id":"repair_001","status":"active","checks":['
        '{"id":"check_docker","description":"verify Docker","status":"resolved"},'
        '{"id":"check_browser","description":"verify in browser","status":"pending"}'
        ']}'
    )

    summary = build_conscience_status_summary(artifact_dir=str(conscience_dir))
    text = format_conscience_status_text(summary)

    assert summary["review_type"] == "stop"
    assert summary["latest_ticket"]["reason"] == "missing_deliverable"
    assert "answer the question" in text
    assert '"review_type": "stop"' in text
    assert "missing_deliverable" in text
    assert summary["active_repair_contract"]["id"] == "repair_001"
    assert "[resolved] verify Docker" in text
    assert "[pending] verify in browser" in text
