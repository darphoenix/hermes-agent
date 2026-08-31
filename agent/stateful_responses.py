"""Stateful Responses chaining for compatible custom/local providers.

The upstream Responses transport remains stateless by default. Local wrappers
can opt in explicitly and keep exact server-side KV continuity by sending only
the messages after the last durable response anchor with ``previous_response_id``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def message_content_plain_text(content: Any) -> str:
    """Flatten message content for Responses instruction fields."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    nested = part.get("content")
                    if isinstance(nested, str):
                        parts.append(nested)
        return "".join(parts)
    return str(content)


def split_runtime_instructions(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], list[str]]:
    """Remove transient system/developer directives from durable input.

    The first system message remains the stable Responses instruction prefix.
    Later system/developer messages are one-call runtime policy, such as a
    conscience stop gate, and must not become replayed transcript content.
    """
    if not messages:
        return messages, []

    base_system_index = (
        0
        if isinstance(messages[0], dict) and messages[0].get("role") == "system"
        else None
    )
    cleaned: list[Dict[str, Any]] = []
    directives: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "developer"} and index != base_system_index:
            content = message_content_plain_text(message.get("content", "")).strip()
            if content:
                directives.append(content)
            continue
        cleaned.append(message)
    return cleaned, directives


def base_instructions(messages: List[Dict[str, Any]], fallback: str) -> str:
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = message_content_plain_text(messages[0].get("content", "")).strip()
        if content:
            return content
    return fallback


def runtime_directive_text(directives: list[str]) -> str:
    return "\n\n".join(
        directive.strip() for directive in directives if directive and directive.strip()
    ).strip()


def instructions_with_runtime_directives(
    messages: List[Dict[str, Any]],
    directives: list[str],
    fallback: str,
) -> str:
    base = base_instructions(messages, fallback)
    directive_block = runtime_directive_text(directives)
    if not directive_block:
        return base
    return (
        f"{base}\n\n"
        "[Internal runtime directive for this actor call only. "
        "Do not expose or quote it.]\n"
        f"{directive_block}"
    ).strip()


def initialize(agent: Any, enabled: bool = False) -> None:
    agent.responses_stateful = bool(enabled)
    agent._responses_previous_response_id = None
    agent._responses_blocked_response_id = None
    agent._responses_blocked_response_ids = set()
    agent._responses_force_fresh_until_success = False
    agent._responses_transient_repair_previous_response_id = None


def is_enabled(agent: Any) -> bool:
    return (
        getattr(agent, "api_mode", None) == "codex_responses"
        and bool(getattr(agent, "responses_stateful", False))
        and getattr(agent, "provider", None) != "openai-codex"
    )


def clear_chain(
    agent: Any,
    *,
    reason: str = "",
    blocked_response_id: Optional[str] = None,
    force_fresh_until_success: bool = False,
) -> None:
    blocked = (
        blocked_response_id.strip()
        if isinstance(blocked_response_id, str) and blocked_response_id.strip()
        else None
    )
    current = getattr(agent, "_responses_previous_response_id", None)
    transient = getattr(
        agent, "_responses_transient_repair_previous_response_id", None
    )
    if blocked is None and isinstance(current, str) and current.strip():
        blocked = current.strip()

    if blocked:
        agent._responses_blocked_response_id = blocked
        blocked_ids = getattr(agent, "_responses_blocked_response_ids", None)
        if not isinstance(blocked_ids, set):
            blocked_ids = set()
            agent._responses_blocked_response_ids = blocked_ids
        blocked_ids.add(blocked)
        if force_fresh_until_success:
            agent._responses_force_fresh_until_success = True

    agent._responses_previous_response_id = None
    agent._responses_transient_repair_previous_response_id = None
    if reason and (blocked or current or transient):
        logger.info(
            "Cleared stateful Responses chain (%s)%s",
            reason,
            f": {blocked}" if blocked else "",
        )


def strip_compacted_response_anchors(messages: List[Dict[str, Any]]) -> int:
    """Remove server-branch identities from a rewritten local transcript.

    A compaction summary is a new prompt projection, not a suffix of any
    response that helped produce the pre-compaction transcript. Persisting an
    old ``responses_response_id`` lets a later process silently resurrect that
    obsolete server-side branch and defeats the compaction.
    """
    removed = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        response_id = message.pop("responses_response_id", None)
        if isinstance(response_id, str) and response_id.strip():
            removed += 1
    return removed


def commit_compaction_rebase(
    agent: Any,
    messages: List[Dict[str, Any]],
    *,
    anchors_removed: int = 0,
) -> int:
    """Start a new Responses chain after local transcript compaction.

    The next request carries the complete compacted prompt without a parent.
    Once that request succeeds, ``remember_response_id`` installs its response
    as the new durable anchor and ordinary delta-only continuation resumes.
    """
    if not is_enabled(agent):
        return anchors_removed

    anchors_removed += strip_compacted_response_anchors(messages)
    previous = getattr(agent, "_responses_previous_response_id", None)
    transient = getattr(
        agent, "_responses_transient_repair_previous_response_id", None
    )
    agent._responses_previous_response_id = None
    agent._responses_transient_repair_previous_response_id = None
    agent._responses_force_fresh_until_success = True
    logger.info(
        "Stateful Responses compaction rebase: session=%s "
        "old_parent=%s transient_parent=%s removed_anchors=%d "
        "next_request=fresh_root",
        getattr(agent, "session_id", None) or "-",
        previous or "none",
        transient or "none",
        anchors_removed,
    )
    return anchors_removed


def remember_response_id(agent: Any, response_id: Any) -> None:
    if not isinstance(response_id, str) or not response_id.strip():
        return
    agent._responses_previous_response_id = response_id.strip()
    agent._responses_transient_repair_previous_response_id = None
    agent._responses_blocked_response_id = None
    agent._responses_force_fresh_until_success = False


def remember_transient_parent(
    agent: Any,
    response_id: Any,
    *,
    reason: str = "",
) -> None:
    if not is_enabled(agent):
        return
    if not isinstance(response_id, str) or not response_id.strip():
        logger.debug(
            "Could not attach transient Responses repair parent (%s): missing response_id",
            reason or "unknown",
        )
        return
    normalized = response_id.strip()
    agent._responses_transient_repair_previous_response_id = normalized
    logger.info(
        "Using transient Responses repair parent for next actor call (%s): %s",
        reason or "repair",
        normalized,
    )


def clear_transient_parent(agent: Any, *, reason: str = "") -> None:
    current = getattr(
        agent, "_responses_transient_repair_previous_response_id", None
    )
    agent._responses_transient_repair_previous_response_id = None
    if current:
        logger.info(
            "Cleared transient Responses repair parent (%s): %s",
            reason or "clear",
            current,
        )


def _get_previous_response_id(
    agent: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    blocked = getattr(agent, "_responses_blocked_response_id", None)
    blocked_ids = getattr(agent, "_responses_blocked_response_ids", None)
    if not isinstance(blocked_ids, set):
        blocked_ids = set()
        agent._responses_blocked_response_ids = blocked_ids

    current = getattr(agent, "_responses_previous_response_id", None)
    current = current.strip() if isinstance(current, str) and current.strip() else None
    if current and current != blocked and current not in blocked_ids:
        return current

    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        candidate = message.get("responses_response_id")
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        if candidate and candidate != blocked and candidate not in blocked_ids:
            agent._responses_previous_response_id = candidate
            return candidate
    return None


def _find_anchor_index(
    messages: List[Dict[str, Any]], response_id: str
) -> Optional[int]:
    target = response_id.strip()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        candidate = message.get("responses_response_id")
        if isinstance(candidate, str) and candidate.strip() == target:
            return index
    return None


def build_delta_messages(
    agent: Any,
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Return the request-local message delta and its durable parent id."""
    if bool(getattr(agent, "_responses_force_fresh_until_success", False)):
        logger.info(
            "Building forced-fresh Responses request after stateful branch reset"
        )
        return messages, None

    transient = getattr(
        agent, "_responses_transient_repair_previous_response_id", None
    )
    if isinstance(transient, str) and transient.strip():
        previous_response_id = transient.strip()
        logger.info(
            "Building Responses transient-repair request from parent: %s",
            previous_response_id,
        )
        return [], previous_response_id

    previous_response_id = _get_previous_response_id(agent, messages)
    if not previous_response_id:
        logger.info(
            "Stateful Responses request: session=%s history=%d anchors=0 parent=none delta=%d",
            getattr(agent, "session_id", None) or "-",
            len(messages),
            len(messages),
        )
        return messages, None

    anchor_index = _find_anchor_index(messages, previous_response_id)
    if anchor_index is None:
        clear_chain(
            agent,
            reason="missing_local_anchor",
            blocked_response_id=previous_response_id,
        )
        logger.info(
            "Stateful Responses request: session=%s history=%d anchors=%d parent=missing delta=%d",
            getattr(agent, "session_id", None) or "-",
            len(messages),
            sum(
                1
                for message in messages
                if isinstance(message, dict)
                and isinstance(message.get("responses_response_id"), str)
                and message["responses_response_id"].strip()
            ),
            len(messages),
        )
        return messages, None

    delta_messages = messages[anchor_index + 1 :]
    logger.info(
        "Stateful Responses request: session=%s history=%d anchors=%d parent=%s delta=%d",
        getattr(agent, "session_id", None) or "-",
        len(messages),
        sum(
            1
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("responses_response_id"), str)
            and message["responses_response_id"].strip()
        ),
        previous_response_id,
        len(delta_messages),
    )
    return delta_messages, previous_response_id


def active_previous_response_id(
    agent: Any, api_kwargs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    if isinstance(api_kwargs, dict):
        candidate = api_kwargs.get("previous_response_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    transient = getattr(
        agent, "_responses_transient_repair_previous_response_id", None
    )
    if isinstance(transient, str) and transient.strip():
        return transient.strip()
    current = getattr(agent, "_responses_previous_response_id", None)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def should_reset_after_error(agent: Any, exc: Exception) -> bool:
    """Reset only when the endpoint explicitly rejects the stored parent."""
    if not is_enabled(agent):
        return False
    message = str(exc or "").lower()
    if not message:
        return False
    poison_markers = (
        "cache_poisoned",
        "stored_response_cache_poisoned",
        "poisoned previous_response_id",
        "poisoned previous response",
    )
    if any(marker in message for marker in poison_markers):
        return True
    if "previous_response_id" in message and any(
        marker in message for marker in ("not found", "unknown", "404")
    ):
        return True
    return "previous response" in message and "not found" in message
