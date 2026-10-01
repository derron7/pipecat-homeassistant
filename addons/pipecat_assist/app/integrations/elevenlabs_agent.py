"""Integration descriptor for ElevenLabs Conversational AI (Live).

The add-on's integration registry consumes this dictionary so the web UI
shows an 'ElevenLabs Agent (Live)' card under Integrations and the
pipeline builder can reference the stored credentials by id.
"""

from __future__ import annotations

INTEGRATION = {
    "id": "elevenlabs_agent",
    "name": "ElevenLabs Agent (Live)",
    "category": "realtime",  # speech-to-speech, like Gemini Live / OpenAI Realtime
    "provider": "elevenlabs_live",
    "icon": "elevenlabs",
    "summary": "Full-duplex voice agent — STT, LLM and TTS run on ElevenLabs.",
    "docs_url": "https://elevenlabs.io/docs/eleven-agents/overview",
    "fields": [
        {
            "name": "api_key",
            "type": "password",
            "label": "API Key",
            "required": True,
            "help": "Dein ElevenLabs API Key (Settings > API Keys).",
        },
        {
            "name": "agent_id",
            "type": "string",
            "label": "Agent ID",
            "required": True,
            "help": "Die ID eines im ElevenLabs Dashboard angelegten Agenten.",
        },
        {
            "name": "base_url",
            "type": "string",
            "label": "WebSocket Base URL",
            "default": "wss://api.elevenlabs.io",
            "help": "Nur ändern, wenn du einen eigenen Proxy vor ElevenLabs betreibst.",
        },
        {
            "name": "first_message",
            "type": "string",
            "label": "First Message (optional)",
            "required": False,
            "help": "Optional: Begrüßung, die der Agent beim Verbinden spricht.",
        },
        {
            "name": "dynamic_variables",
            "type": "json",
            "label": "Dynamic Variables (JSON)",
            "required": False,
            "default": "{}",
            "help": "JSON-Objekt mit Platzhaltern, die der Agent im Prompt sieht.",
        },
        {
            "name": "enable_client_tools",
            "type": "boolean",
            "label": "Enable MCP client tools",
            "default": True,
            "help": "Home Assistant MCP Tools als ElevenLabs Client-Tools freigeben.",
        },
    ],
}


def register(registry):
    """Called by the add-on's integration loader at startup."""
    registry.register(INTEGRATION)
