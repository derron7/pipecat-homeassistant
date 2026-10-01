"""Pipeline profile: ElevenLabs Live (speech-to-speech).

Mirrors the built-in Gemini Live / OpenAI Realtime profiles. When the
user selects this profile in the UI, the add-on constructs a Pipecat
pipeline whose sole realtime step is ElevenLabsLiveService. Home
Assistant MCP tools are exposed as ElevenLabs client tools so the
agent can control devices directly.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.elevenlabs_live import ElevenLabsLiveService, ElevenLabsLiveSettings

logger = logging.getLogger("pipecat.pipelines.elevenlabs_live")


PIPELINE = {
    "id": "elevenlabs_live",
    "name": "ElevenLabs Live",
    "runtime": "speech_to_speech",  # same category as gemini_live / openai_realtime
    "provider": "elevenlabs_live",
    "integration": "elevenlabs_agent",
    "summary": "Full-duplex voice bot — STT, LLM und TTS komplett über ElevenLabs.",
    "default_model": "elevenlabs_agent",
    "supports_mcp_tools": True,
    "supports_flows": False,  # ElevenLabs orchestriert den Flow selbst
}


async def build_pipeline(context, config: dict[str, Any]):
    """Factory the add-on runtime calls when this profile is active.

    'context' is the add-on session context and already exposes the
    active MCP client so we can wire HA tools straight into ElevenLabs.
    """
    integration = config.get("integration", {}) or {}

    settings = ElevenLabsLiveSettings(
        api_key=integration.get("api_key"),
        agent_id=integration.get("agent_id"),
        base_url=integration.get("base_url") or "wss://api.elevenlabs.io",
        first_message=integration.get("first_message") or None,
        dynamic_variables=_parse_json(integration.get("dynamic_variables"), {}),
        enable_client_tools=bool(integration.get("enable_client_tools", True)),
    )

    tool_handler = None
    if settings.enable_client_tools and hasattr(context, "mcp"):
        tool_handler = _build_mcp_tool_handler(context)

    service = ElevenLabsLiveService(settings=settings, tool_handler=tool_handler)
    return service


def _parse_json(value: Any, default):
    if not value:
        return default
    if isinstance(value, dict):
        return value
    try:
        import json as _json
        return _json.loads(value)
    except Exception:
        return default


def _build_mcp_tool_handler(context):
    async def handler(tool_name: str, params: dict[str, Any]) -> Any:
        try:
            result = await context.mcp.call_tool(tool_name, params)
            return {"ok": True, "result": result}
        except Exception as exc:
            logger.exception("MCP tool %s failed", tool_name)
            return {"ok": False, "error": str(exc)}

    return handler


def register(registry):
    """Called by the add-on's pipeline loader at startup."""
    registry.register(PIPELINE, factory=build_pipeline)
