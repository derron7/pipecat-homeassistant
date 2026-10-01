"""ElevenLabs Conversational AI (Agents) speech-to-speech service for Pipecat.

Bridges the ElevenLabs Agents WebSocket API into a Pipecat pipeline as a
speech-to-speech step, mirroring how Gemini Live / OpenAI Realtime are used
by Pipecat Assist:

* consumes ``InputAudioRawFrame`` from the transport, resamples to the
  16 kHz PCM16 mono format ElevenLabs expects and streams it as
  ``user_audio_chunk`` messages,
* decodes ``audio`` events back into ``TTSAudioRawFrame`` so the transport
  plays the agent voice,
* maps ``user_transcript`` / ``agent_response`` to Pipecat transcription and
  LLM text frames so the UI transcript works,
* maps ``interruption`` to ``broadcast_interruption()`` for full-duplex
  barge-in,
* answers ``ping`` and executes ``client_tool_call`` through an optional
  async tool handler (wired to the Home Assistant MCP bridge by main.py).

WebSocket protocol reference:
https://elevenlabs.io/docs/agents-platform/libraries/web-sockets
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx
import websockets
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

ELEVENLABS_INPUT_SAMPLE_RATE = 16_000

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class ElevenLabsLiveSettings:
    """Connection settings for one ElevenLabs Agents session."""

    api_key: str = ""
    agent_id: str = ""
    base_url: str = "wss://api.elevenlabs.io"
    # Sent as conversation_initiation_client_data. Only has an effect when
    # the matching overrides are enabled in the agent's Security settings.
    conversation_config_override: dict[str, Any] = field(default_factory=dict)
    dynamic_variables: dict[str, Any] = field(default_factory=dict)


class ElevenLabsLiveService(FrameProcessor):
    """Full-duplex speech-to-speech bridge to an ElevenLabs Agent."""

    def __init__(
        self,
        *,
        settings: ElevenLabsLiveSettings,
        tool_handler: Optional[ToolHandler] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._tool_handler = tool_handler

        self._ws: Any = None
        self._receive_task: Optional[asyncio.Task] = None
        self._resampler = create_stream_resampler()
        self._output_sample_rate = ELEVENLABS_INPUT_SAMPLE_RATE
        self._conversation_id: str | None = None
        self._bot_speaking = False
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._connect()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            self._bot_speaking = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMRunFrame):
            # Greetings are configured as the agent's "first message" in the
            # ElevenLabs dashboard; a plain LLMRunFrame has no equivalent.
            pass
        else:
            await self.push_frame(frame, direction)

    # -------------------------------------------------------------- connection

    def _http_base(self) -> str:
        base = (self._settings.base_url or "wss://api.elevenlabs.io").rstrip("/")
        return base.replace("wss://", "https://").replace("ws://", "http://")

    def _ws_base(self) -> str:
        base = (self._settings.base_url or "wss://api.elevenlabs.io").rstrip("/")
        return base.replace("https://", "wss://").replace("http://", "ws://")

    async def _signed_url(self) -> str | None:
        """Fetch a signed WebSocket URL for private agents."""
        if not self._settings.api_key:
            return None
        url = f"{self._http_base()}/v1/convai/conversation/get-signed-url"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    url,
                    params={"agent_id": self._settings.agent_id},
                    headers={"xi-api-key": self._settings.api_key},
                )
                response.raise_for_status()
                return response.json().get("signed_url")
        except Exception as err:
            logger.warning(
                "ElevenLabs signed URL request failed ({}); falling back to public agent URL",
                err,
            )
            return None

    async def _connect(self) -> None:
        if self._ws:
            return
        if not self._settings.agent_id:
            await self.push_error(ErrorFrame("ElevenLabs Agent ID is missing", fatal=True))
            return

        url = await self._signed_url()
        if not url:
            url = f"{self._ws_base()}/v1/convai/conversation?agent_id={self._settings.agent_id}"

        try:
            self._ws = await websockets.connect(url, ping_interval=20, max_size=16 * 1024 * 1024)
        except Exception as err:
            await self.push_error(ErrorFrame(f"ElevenLabs connection failed: {err}", fatal=True))
            return

        init: dict[str, Any] = {"type": "conversation_initiation_client_data"}
        if self._settings.conversation_config_override:
            init["conversation_config_override"] = self._settings.conversation_config_override
        if self._settings.dynamic_variables:
            init["dynamic_variables"] = self._settings.dynamic_variables
        await self._ws.send(json.dumps(init))

        self._receive_task = self.create_task(self._receive_loop())
        logger.info("ElevenLabs Agent {} connecting", self._settings.agent_id)

    async def _disconnect(self) -> None:
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected.clear()
        self._bot_speaking = False

    # ------------------------------------------------------------------- audio

    async def _send_user_audio(self, frame: InputAudioRawFrame) -> None:
        if not self._ws or not self._connected.is_set():
            return
        audio = frame.audio
        if frame.sample_rate != ELEVENLABS_INPUT_SAMPLE_RATE:
            audio = await self._resampler.resample(
                audio, frame.sample_rate, ELEVENLABS_INPUT_SAMPLE_RATE
            )
        if not audio:
            return
        try:
            await self._ws.send(
                json.dumps({"user_audio_chunk": base64.b64encode(audio).decode("ascii")})
            )
        except websockets.ConnectionClosed:
            logger.warning("ElevenLabs connection closed while sending audio")

    # ------------------------------------------------------------------ events

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                await self._handle_message(message)
        except websockets.ConnectionClosed as err:
            logger.info("ElevenLabs connection closed: {}", err)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            await self.push_error(ErrorFrame(f"ElevenLabs receive error: {err}"))

    async def _handle_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")

        if kind == "conversation_initiation_metadata":
            event = message.get("conversation_initiation_metadata_event", {}) or {}
            self._conversation_id = event.get("conversation_id")
            self._output_sample_rate = _parse_pcm_rate(
                event.get("agent_output_audio_format"), ELEVENLABS_INPUT_SAMPLE_RATE
            )
            user_format = event.get("user_input_audio_format") or "pcm_16000"
            if not str(user_format).startswith("pcm_16000"):
                logger.warning(
                    "ElevenLabs agent expects user input format {} — set the agent "
                    "input format to PCM 16000 for best results",
                    user_format,
                )
            self._connected.set()
            logger.info(
                "ElevenLabs conversation {} ready (output format {})",
                self._conversation_id,
                event.get("agent_output_audio_format"),
            )
            return

        if kind == "audio":
            event = message.get("audio_event", {}) or {}
            encoded = event.get("audio_base_64")
            if not encoded:
                return
            if not self._bot_speaking:
                self._bot_speaking = True
                await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=base64.b64decode(encoded),
                    sample_rate=self._output_sample_rate,
                    num_channels=1,
                )
            )
            return

        if kind == "agent_response":
            event = message.get("agent_response_event", {}) or {}
            text = (event.get("agent_response") or "").strip()
            if text:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text))
                await self.push_frame(LLMFullResponseEndFrame())
            return

        if kind == "user_transcript":
            event = message.get("user_transcription_event", {}) or {}
            text = (event.get("user_transcript") or "").strip()
            if text:
                await self.push_frame(
                    TranscriptionFrame(text, "", time_now_iso8601())
                )
            if self._bot_speaking:
                self._bot_speaking = False
                await self.push_frame(TTSStoppedFrame())
            return

        if kind == "interruption":
            logger.debug("ElevenLabs interruption (barge-in) received")
            self._bot_speaking = False
            await self.push_frame(TTSStoppedFrame())
            await self.broadcast_interruption()
            return

        if kind == "ping":
            event = message.get("ping_event", {}) or {}
            event_id = event.get("event_id")
            if self._ws and event_id is not None:
                await self._ws.send(json.dumps({"type": "pong", "event_id": event_id}))
            return

        if kind == "client_tool_call":
            await self._handle_tool_call(message.get("client_tool_call", {}) or {})
            return

        if kind in {"vad_score", "internal_vad_score", "internal_turn_probability",
                    "agent_response_correction", "mcp_tool_call", "agent_tool_response"}:
            return

        logger.debug("Unhandled ElevenLabs event type: {}", kind)

    # -------------------------------------------------------------- tool calls

    async def _handle_tool_call(self, event: dict[str, Any]) -> None:
        tool_name = event.get("tool_name") or ""
        tool_call_id = event.get("tool_call_id") or ""
        parameters = event.get("parameters") or {}

        logger.info("ElevenLabs client tool call: {}({})", tool_name, parameters)

        is_error = False
        if self._tool_handler is None:
            result: Any = f"Tool {tool_name} is not available in this installation."
            is_error = True
        else:
            try:
                result = await self._tool_handler(tool_name, parameters)
            except Exception as err:
                logger.exception("ElevenLabs tool {} failed", tool_name)
                result = f"Tool {tool_name} failed: {err}"
                is_error = True

        if self._ws and tool_call_id:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "client_tool_result",
                        "tool_call_id": tool_call_id,
                        "result": result if isinstance(result, str) else json.dumps(result),
                        "is_error": is_error,
                    }
                )
            )


def _parse_pcm_rate(audio_format: Any, default: int) -> int:
    """Parse formats like ``pcm_16000`` / ``pcm_24000`` into a sample rate."""
    try:
        text = str(audio_format or "")
        if text.startswith("pcm_"):
            return int(text.split("_", 1)[1])
    except (ValueError, IndexError):
        pass
    if audio_format:
        logger.warning(
            "ElevenLabs agent output format {} is not raw PCM — configure the "
            "agent TTS output format as PCM so Pipecat can play it",
            audio_format,
        )
    return default
