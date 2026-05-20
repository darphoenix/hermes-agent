"""MTPLX local provider profile.

MTPLX exposes a real SessionBank cache through Chat Completions. Supplying a
stable x-mtplx-session-id lets the server reuse prompt prefixes across tool
turns, which is the closest apples-to-apples comparison to Hermes' stateful
Responses flow without adding a full Responses endpoint to MTPLX.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class MTPLXProfile(ProviderProfile):
    """Local MTPLX server profile."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        session_id: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        if reasoning_config and isinstance(reasoning_config, dict):
            effort = str(reasoning_config.get("effort") or "").strip().lower()
            enabled = reasoning_config.get("enabled", True)
            if effort == "none" or enabled is False:
                extra_body["think"] = False

        top_level = {
            "extra_headers": {
                "x-mtplx-session-id": str(session_id or "hermes-mtplx"),
            }
        }
        return extra_body, top_level


register_provider(
    MTPLXProfile(
        name="mtplx",
        aliases=("local-mtplx",),
        api_mode="chat_completions",
        display_name="MTPLX",
        description="Local MTPLX OpenAI-compatible server with SessionBank cache.",
    )
)
