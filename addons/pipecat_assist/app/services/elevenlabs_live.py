"""ElevenLabs Conversational AI — Pipecat speech-to-speech bridge.

This module exposes ElevenLabs' Conversational AI WebSocket API as a Pipecat
frame processor. It behaves as a speech-to-speech service: it consumes
AudioRawFrame coming from the microphone, forwards them to ElevenLabs over
a single persistent WebSocket and pushes the returned audio back into the
pipeline as AudioRawFrame. Transcripts, agent responses and interruptions
are forwarded as the matching Pipecat frames so the Home Assistant UI can
render captions and so the rest of the pipeline (MCP tools, memory, …)
sees the same events it gets from Gemini Live or OpenAI Realtime.

WebSocket reference:
    https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("pipecat.elevenlabs_live")


ToolCallHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class ElevenLabsLiveSettings:
    """Configuration for the ElevenLabs Conversational AI endpoint."""

    api_key: str | None = None
    agent_id: str | None = None
    signed_url: str | None = None
    base_url: str = "wss://api.elevenlabs.io"
    # ElevenLabs always wants 16 kHz mono PCM16 on the wire.
    sample_rate: int = 16_000
    first_message: str | None = None
    dynamic_variables: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    enable_client_tools: bool = True


class ElevenLabsLiveService(FrameProcessor):
    """Bidirectional speech-to-speech adapter for ElevenLabs Agents."""

    def __init__(
        self,
        *,
        settings: ElevenLabsLiveSettings,
        tool_handler: ToolCallHandler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._tool_handler = tool_handler

        self._ws: ClientConnection | None = None
        self._recv_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
        self._closing = asyncio.Event()
        self._conversation_id: str | None = None
        self._resampler = None
        self._in_sample_rate: int = 16_000
        self._agent_speaking: bool = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self, frame: StartFrame) -> None:
        self._in_sample_rate = frame.audio_in_sample_rate or 16_000
        self._resampler = create_default_resampler(
            in_rate=self._in_sample_rate,
            out_rate=self._settings.sample_rate,
        )
        await self._connect()

    async def stop(self, frame: EndFrame) -> None:
        await self._close()

    async def cancel(self, frame: CancelFrame) -> None:
        await self._close()

    # ----------------------------------------------------------------- connection

    def _build_url(self) -> str:
        if self._settings.signed_url:
            return self._settings.signed_url
        if not self._settings.agent_id:
            raise ValueError("ElevenLabsLiveSettings requires agent_id or signed_url")
        base = self._settings.base_url.rstrip("/")
        return f"{base}/v1/convai/conversation?agent_id={self._settings.agent_id}"

    async def _connect(self) -> None:
        url = self._build_url()
        headers: dict[str, str] = {}
        # The signed_url already carries auth; plain agent_id connections need
        # the API key as a header.
        if self._settings.api_key and not self._settings.signed_url:
            headers["xi-api-key"] = self._settings.api_key

        try:
            self._ws = await websockets.connect(
                url, additional_headers=headers, ping_interval=20
            )
        except Exception as exc:
            logger.error("ElevenLabs WS connect failed: %s", exc)
            await self.push_error(ErrorFrame(f"ElevenLabs connect failed: {exc}"))
            raise

        await self._send_init()
        self._closing.clear()
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def _send_init(self) -> None:
        msg: dict[str, Any] = {"type": "conversation_initiation_client_data"}
        payload: dict[str, Any] = {}
        if self._settings.config_overrides:
            payload["conversation_config_override"] = self._settings.config_overrides
        if self._settings.dynamic_variables:
            payload["dynamic_variables"] = self._settings.dynamic_variables
        if self._settings.first_message:
            payload.setdefault("custom_llm_extra_body", {})["first_message"] = (
                self._settings.first_message
            )
        if payload:
            msg.update(payload)
        assert self._ws is not None
        await self._ws.send(json.dumps(msg))

    async def _send_loop(self) -> None:
        try:
            while not self._closing.is_set():
                chunk = await self._audio_queue.get()
                if chunk is None or self._ws is None:
                    return
                await self._ws.send(
                    json.dumps({"user_audio_chunk": base64.b64encode(chunk).decode("ascii")})
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("ElevenLabs send loop crashed")
            await self.push_error(ErrorFrame(f"ElevenLabs send error: {exc}"))

    async def _recv_loop(self) -> None:
        try:
            assert self._ws is not None
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(data)
        except websockets.ConnectionClosed as exc:
            logger.warning("ElevenLabs WS closed: %s", exc)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("ElevenLabs recv loop failed")
            await self.push_error(ErrorFrame(f"ElevenLabs recv error: {exc}"))

    async def _dispatch(self, data: dict[str, Any]) -> None:
        t = data.get("type")

        if t == "conversation_initiation_metadata":
            meta = data.get("conversation_initiation_metadata_event", {})
            self._conversation_id = meta.get("conversation_id")
            logger.info("ElevenLabs conversation %s ready", self._conversation_id)
            return

        if t == "user_transcript":
            ev = data.get("user_transcription_event", {})
            text = ev.get("transcript", "")
            if text:
                await self.push_frame(
                    TranscriptionFrame(text=text, user_id="user", timestamp=""),
                    FrameDirection.UPSTREAM,
                )
            return

        if t == "agent_response":
            ev = data.get("agent_response_event", {})
            text = ev.get("agent_response", "")
            if text:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=text))
                await self.push_frame(LLMFullResponseEndFrame())
            return

        if t == "audio":
            ev = data.get("audio_event", {})
            b64 = ev.get("audio_base_64")
            if not b64:
                return
            pcm = base64.b64decode(b64)
            if not self._agent_speaking:
                self._agent_speaking = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(
                AudioRawFrame(
                    audio=pcm,
                    sample_rate=self._settings.sample_rate,
                    num_channels=1,
                )
            )
            return

        if t == "interruption":
            # User barged in — tell the upstream pipeline to flush.
            if self._agent_speaking:
                self._agent_speaking = False
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(InterruptFrame(), FrameDirection.UPSTREAM)
            return

        if t == "ping":
            event_id = data.get("ping_event", {}).get("event_id")
            if self._ws is not None and event_id is not None:
                await self._ws.send(json.dumps({"type": "pong", "event_id": event_id}))
            return

        if t == "client_tool_call" and self._settings.enable_client_tools:
            await self._handle_tool_call(data.get("client_tool_call_event", {}) or {})
            return

        if t in {"internal_vad_score", "internal_turn_prob", "agent_response_correction"}:
            return

        logger.debug("Unhandled ElevenLabs event: %s", t)

    async def _handle_tool_call(self, ev: dict[str, Any]) -> None:
        name = ev.get("tool_name") or ""
        params = ev.get("parameters") or {}
        call_id = ev.get("tool_call_id") or ""
        result: Any = {"ok": False, "error": "no tool handler registered"}
        if self._tool_handler is not None:
            try:
                result = await self._tool_handler(name, params)
            except Exception as exc:
                logger.exception("ElevenLabs tool %s raised", name)
                result = {"ok": False, "error": str(exc)}
        if self._ws is not None and call_id:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "client_tool_result",
                        "tool_call_id": call_id,
                        "result": result,
                        "is_error": isinstance(result, dict) and result.get("ok") is False,
                    }
                )
            )

    async def _close(self) -> None:
        self._closing.set()
        await self._audio_queue.put(None)
        for t in (self._send_task, self._recv_task):
            if t is not None:
                t.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    # ------------------------------------------------------------------- pipeline

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            await self.stop(frame)
        elif isinstance(frame, CancelFrame):
            await self.push_frame(frame, direction)
            await self.cancel(frame)
        elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._ingest_audio(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # ElevenLabs uses its own VAD; we only forward the hint.
            if self._ws is not None:
                try:
                    await self._ws.send(json.dumps({"type": "user_activity"}))
                except Exception:
                    pass
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptFrame):
            if self._ws is not None:
                try:
                    await self._ws.send(json.dumps({"type": "user_activity"}))
                except Exception:
                    pass
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _ingest_audio(self, frame: AudioRawFrame) -> None:
        if self._resampler is None or frame.audio is None:
            return
        pcm = await self._resampler.resample(frame.audio, frame.sample_rate)
        # 20 ms blocks @ 16 kHz PCM16 mono = 640 bytes.
        chunk_size = 640
        for i in range(0, len(pcm), chunk_size):
            await self._audio_queue.put(pcm[i : i + chunk_size])
