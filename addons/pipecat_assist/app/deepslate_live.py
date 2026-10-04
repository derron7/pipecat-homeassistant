"""Deepslate Realtime speech-to-speech service for Pipecat Assist ("Deepslate Live").

Bridges the Deepslate Realtime WebSocket API into a Pipecat pipeline as a
speech-to-speech step. It mirrors ``elevenlabs_live.py`` (same frame contract
towards the transport, same ``tool_handler`` hook for the Home Assistant MCP
bridge) and re-uses the protocol implementation of the official Deepslate SDK
(``deepslate-core``, the same engine that powers
``deepslate.pipecat.DeepslateRealtimeLLMService``):

* consumes ``InputAudioRawFrame`` from the transport, downmixes/resamples to
  24 kHz PCM16 mono and streams it to Deepslate (server-side VAD, so no local
  VAD step is needed). 24 kHz matches Gemini Live / OpenAI Realtime and the
  ESPHome Voice PE output contract (``va_pipecat.OUTPUT_SAMPLE_RATE = 24000``).
  Deepslate's ``InitializeSessionRequest`` uses the same rate for input *and*
  output, so sending 16 kHz made the hosted TTS (native 24 kHz) play a fifth
  too low on Voice PE.
* turns ``ModelAudioChunk`` events into ``TTSAudioRawFrame`` so the transport
  (WebRTC, Lovelace card or the ESPHome ``va_pipecat`` satellite on Home
  Assistant Voice PE) plays the agent voice,
* maps user transcriptions / model text fragments to ``TranscriptionFrame`` /
  ``LLMTextFrame`` so the add-on UI transcript works,
* maps ``PlaybackClearBuffer`` (the user started to speak) to
  ``broadcast_interruption()`` for full-duplex barge-in,
* registers the Home Assistant MCP tools with the model and executes
  ``ToolCallRequest`` through the optional async ``tool_handler``,
* keeps the session alive with a tiny silent audio packet when the transport
  stops sending audio between two wake-word turns.

WebSocket protocol reference: https://docs.deepslate.eu/websocket
SDK reference:                https://github.com/deepslate-labs/deepslate-sdks
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

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
    LLMMessagesAppendFrame,
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

from deepslate.core import (
    DeepslateOptions,
    DeepslateSession,
    DeepslateSessionListener,
    ElevenLabsTtsConfig,
    HostedTtsConfig,
    TriggerMode,
    VadConfig,
    build_user_agent,
)

# Deepslate hosted TTS is native 24 kHz. The session proto uses one rate for
# both input_audio_line and output_audio_line, so we send 24 kHz too.
# Home Assistant Voice PE (va_pipecat) plays at 24 kHz; Gemini Live and
# OpenAI Realtime in this add-on do the same. 16 kHz here makes 24 kHz TTS
# play 1.5× too slow / too low on the satellite.
DEEPSLATE_SAMPLE_RATE = 24_000
DEEPSLATE_CHANNELS = 1
DEEPSLATE_DEFAULT_BASE_URL = "https://app.deepslate.eu"
DEEPSLATE_TOOL_TIMEOUT_SECONDS = 60.0
# Deepslate closes idle sessions after ~30 s without packets. Silent audio
# counts as activity, so we send a 20 ms silence packet after 8 s of quiet.
KEEPALIVE_IDLE_SECONDS = 8.0
KEEPALIVE_CHECK_SECONDS = 4.0
_SILENCE_20MS = b"\x00" * (DEEPSLATE_SAMPLE_RATE // 50 * 2)

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class DeepslateLiveSettings:
    """Connection settings for one Deepslate Realtime session."""

    api_key: str = ""
    vendor_id: str = ""
    organization_id: str = ""
    base_url: str = DEEPSLATE_DEFAULT_BASE_URL
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.3
    # "hosted" = Deepslate hosted voice, "elevenlabs" = ElevenLabs TTS via Deepslate.
    tts_provider: str = "hosted"
    voice_id: str = ""
    # Optional ElevenLabs model id (only used when tts_provider == "elevenlabs").
    tts_model_id: str = ""
    elevenlabs_api_key: str = ""
    # Server-side VAD tuning (defaults match deepslate-core's VadConfig).
    vad_confidence_threshold: float = 0.4
    vad_min_volume: float = 0.0
    vad_start_duration_ms: int = 150
    vad_stop_duration_ms: int = 390
    vad_backbuffer_duration_ms: int = 1000

    @classmethod
    def with_env_overrides(cls, **values: Any) -> "DeepslateLiveSettings":
        """Create settings; ``DEEPSLATE_*`` env vars tune temperature and VAD."""
        settings = cls(**values)
        settings.temperature = _env_float("DEEPSLATE_TEMPERATURE", settings.temperature)
        settings.vad_confidence_threshold = _env_float(
            "DEEPSLATE_VAD_CONFIDENCE", settings.vad_confidence_threshold
        )
        settings.vad_min_volume = _env_float("DEEPSLATE_VAD_MIN_VOLUME", settings.vad_min_volume)
        settings.vad_start_duration_ms = _env_int(
            "DEEPSLATE_VAD_START_MS", settings.vad_start_duration_ms
        )
        settings.vad_stop_duration_ms = _env_int(
            "DEEPSLATE_VAD_STOP_MS", settings.vad_stop_duration_ms
        )
        return settings


class _Listener(DeepslateSessionListener):
    """Forwards Deepslate session events to the Pipecat service.

    Composition instead of multiple inheritance keeps Deepslate's callback
    names from colliding with Pipecat's ``FrameProcessor`` API.
    """

    def __init__(self, service: "DeepslateLiveService") -> None:
        self._service = service

    async def on_session_initialized(self) -> None:
        self._service._on_session_ready()

    async def on_response_begin(self, turn_id: int = 0) -> None:
        await self._service._on_response_begin()

    async def on_response_end(self, turn_id: int = 0) -> None:
        await self._service._on_response_end()

    async def on_text_fragment(self, text: str, turn_id: Optional[int] = None) -> None:
        await self._service._on_text_fragment(text)

    async def on_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        transcript: Optional[str],
        turn_id: Optional[int] = None,
    ) -> None:
        await self._service._on_audio_chunk(pcm_bytes, sample_rate, channels)

    async def on_user_transcription(
        self, text: str, language: Optional[str], turn_id: int
    ) -> None:
        await self._service._on_user_transcription(text)

    async def on_playback_buffer_clear(self) -> None:
        await self._service._on_playback_buffer_clear()

    async def on_tool_call(
        self,
        call_id: str,
        name: str,
        params: dict,
        turn_id: Optional[int] = None,
    ) -> None:
        self._service._on_tool_call(call_id, name, params)

    async def on_error(self, category: str, message: str, trace_id: Optional[str]) -> None:
        await self._service._on_error(category, message, trace_id)

    async def on_fatal_error(self, e: Exception) -> None:
        await self._service._on_fatal_error(e)


class DeepslateLiveService(FrameProcessor):
    """Full-duplex speech-to-speech bridge to Deepslate Realtime."""

    def __init__(
        self,
        *,
        settings: DeepslateLiveSettings,
        tool_handler: Optional[ToolHandler] = None,
        tools_schema: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._tool_handler = tool_handler
        self._tools = _normalise_tools(tools_schema)

        self._session: Optional[DeepslateSession] = None
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._tool_tasks: set[asyncio.Task] = set()
        self._bot_speaking = False
        self._audio_pending = False
        self._logged_output_rate = False
        self._ready = asyncio.Event()
        self._failed = False
        self._last_audio_sent = time.monotonic()

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
            # Deepslate answers on its own after server-side end-of-speech
            # detection; a bare LLMRunFrame has no equivalent.
            pass
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._handle_messages_append(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._disconnect()

    # -------------------------------------------------------------- connection

    def _build_tts_config(self) -> tuple[Any, str]:
        """Return ``(tts_config, error)``; exactly one of them is empty."""
        settings = self._settings
        provider = (settings.tts_provider or "hosted").strip().lower().replace("-", "_")
        if provider in {"elevenlabs", "eleven_labs"}:
            if not settings.elevenlabs_api_key:
                return None, (
                    "Deepslate Live is set to ElevenLabs TTS but no ElevenLabs API key "
                    "was found. Configure the ElevenLabs integration or use TTS provider "
                    "'hosted'."
                )
            if not settings.voice_id:
                return None, "Deepslate Live needs an ElevenLabs voice ID (Voice ID field)"
            return (
                ElevenLabsTtsConfig(
                    api_key=settings.elevenlabs_api_key,
                    voice_id=settings.voice_id,
                    model_id=settings.tts_model_id or None,
                ),
                "",
            )
        if not settings.voice_id:
            return None, (
                "Deepslate Live needs a hosted voice ID (Voice ID field). Create or pick a "
                "voice in the Deepslate dashboard."
            )
        return HostedTtsConfig(voice_id=settings.voice_id), ""

    async def _connect(self) -> None:
        if self._session is not None:
            return

        settings = self._settings
        missing = [
            label
            for label, value in (
                ("API key", settings.api_key),
                ("vendor ID", settings.vendor_id),
                ("organization ID", settings.organization_id),
            )
            if not (value or "").strip()
        ]
        if missing:
            await self._fatal(f"Deepslate {', '.join(missing)} missing")
            return

        tts_config, tts_error = self._build_tts_config()
        if tts_error:
            await self._fatal(tts_error)
            return

        try:
            options = DeepslateOptions(
                vendor_id=settings.vendor_id.strip(),
                organization_id=settings.organization_id.strip(),
                api_key=settings.api_key.strip(),
                base_url=(settings.base_url or DEEPSLATE_DEFAULT_BASE_URL).strip().rstrip("/"),
                system_prompt=settings.system_prompt,
                temperature=settings.temperature,
            )
            vad_config = VadConfig(
                confidence_threshold=settings.vad_confidence_threshold,
                min_volume=settings.vad_min_volume,
                start_duration_ms=settings.vad_start_duration_ms,
                stop_duration_ms=settings.vad_stop_duration_ms,
                backbuffer_duration_ms=settings.vad_backbuffer_duration_ms,
            )
            try:
                user_agent = build_user_agent("pipecat-assist", "pipecat-ai")
            except Exception:  # noqa: BLE001 - cosmetic only
                user_agent = None

            self._session = DeepslateSession.create(
                options,
                vad_config=vad_config,
                tts_config=tts_config,
                user_agent=user_agent,
                listener=_Listener(self),
            )
            self._session.start()
            if self._tools:
                await self._session.update_tools(self._tools)
        except Exception as err:  # noqa: BLE001
            logger.exception("Deepslate session setup failed")
            self._session = None
            await self._fatal(f"Deepslate connection failed: {err}")
            return

        self._last_audio_sent = time.monotonic()
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="DeepslateLive.keepalive"
        )
        logger.info(
            "Deepslate Live connecting (vendor {}, organization {}, {} tools, TTS {})",
            settings.vendor_id,
            settings.organization_id,
            len(self._tools),
            (settings.tts_provider or "hosted").lower(),
        )

    async def _disconnect(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        for task in list(self._tool_tasks):
            task.cancel()
        self._tool_tasks.clear()

        session, self._session = self._session, None
        if session is not None:
            try:
                await session.close()
            except Exception as err:  # noqa: BLE001
                logger.debug("Deepslate close error: {}", err)
        self._ready.clear()
        self._bot_speaking = False
        self._audio_pending = False
        self._logged_output_rate = False

    async def _fatal(self, message: str) -> None:
        self._failed = True
        logger.error(message)
        await self.push_error(ErrorFrame(message, fatal=True))

    # ------------------------------------------------------------------- audio

    async def _send_user_audio(self, frame: InputAudioRawFrame) -> None:
        session = self._session
        if session is None or self._failed:
            return

        audio = frame.audio
        channels = frame.num_channels or 1
        if channels > 1:
            audio = _downmix_to_mono(audio, channels)
        if frame.sample_rate != DEEPSLATE_SAMPLE_RATE:
            audio = await self._input_resampler.resample(
                audio, frame.sample_rate, DEEPSLATE_SAMPLE_RATE
            )
        if not audio:
            return

        try:
            await session.send_audio(audio, DEEPSLATE_SAMPLE_RATE, DEEPSLATE_CHANNELS)
            self._last_audio_sent = time.monotonic()
        except Exception as err:  # noqa: BLE001
            logger.debug("Deepslate send_audio failed: {}", err)

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_CHECK_SECONDS)
                session = self._session
                if session is None or self._failed:
                    return
                if time.monotonic() - self._last_audio_sent < KEEPALIVE_IDLE_SECONDS:
                    continue
                try:
                    await session.send_audio(
                        _SILENCE_20MS, DEEPSLATE_SAMPLE_RATE, DEEPSLATE_CHANNELS
                    )
                    self._last_audio_sent = time.monotonic()
                except Exception as err:  # noqa: BLE001
                    logger.debug("Deepslate keepalive failed: {}", err)
        except asyncio.CancelledError:
            raise

    # ----------------------------------------------------------- text / context

    async def _handle_messages_append(self, frame: LLMMessagesAppendFrame) -> None:
        """Forward appended context (e.g. a configured greeting) as text input."""
        session = self._session
        if session is None:
            return
        texts: list[str] = []
        for message in getattr(frame, "messages", None) or []:
            content = message.get("content") if isinstance(message, dict) else getattr(
                message, "content", None
            )
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
        text = "\n".join(item.strip() for item in texts if item and item.strip()).strip()
        if not text:
            return
        trigger = TriggerMode.IMMEDIATE if getattr(frame, "run_llm", False) else TriggerMode.NO_TRIGGER
        try:
            await session.send_text(text, trigger)
        except Exception as err:  # noqa: BLE001
            logger.debug("Deepslate send_text failed: {}", err)

    # ------------------------------------------------------- listener callbacks

    def _on_session_ready(self) -> None:
        self._ready.set()
        logger.info("Deepslate Live session ready")

    async def _on_response_begin(self) -> None:
        await self.push_frame(LLMFullResponseStartFrame())

    async def _on_response_end(self) -> None:
        await self.push_frame(LLMFullResponseEndFrame())
        if self._bot_speaking:
            self._bot_speaking = False
            await self.push_frame(TTSStoppedFrame())

    async def _on_text_fragment(self, text: str) -> None:
        if text:
            await self.push_frame(LLMTextFrame(text))

    async def _on_audio_chunk(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        if not pcm:
            return
        channels = channels or 1
        if channels > 1:
            pcm = _downmix_to_mono(pcm, channels)
            channels = 1
        # The SDK reports the *session* rate (whatever we sent on init), not
        # a per-chunk header. Hosted TTS is 24 kHz; if a chunk still arrives
        # tagged otherwise, resample so Voice PE (24 kHz) plays at the right pitch.
        reported = sample_rate or DEEPSLATE_SAMPLE_RATE
        if not self._logged_output_rate:
            self._logged_output_rate = True
            logger.info(
                "Deepslate Live first audio chunk: {} bytes, reported {} Hz {} ch → playing at {} Hz",
                len(pcm),
                reported,
                channels,
                DEEPSLATE_SAMPLE_RATE,
            )
        if reported != DEEPSLATE_SAMPLE_RATE:
            pcm = await self._output_resampler.resample(pcm, reported, DEEPSLATE_SAMPLE_RATE)
            if not pcm:
                return
        if not self._bot_speaking:
            self._bot_speaking = True
            await self.push_frame(TTSStartedFrame())
        self._audio_pending = True
        await self.push_frame(
            TTSAudioRawFrame(
                audio=pcm,
                sample_rate=DEEPSLATE_SAMPLE_RATE,
                num_channels=1,
            )
        )

    async def _on_user_transcription(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601()))

    async def _on_playback_buffer_clear(self) -> None:
        """The user started speaking: drop everything that was not played yet."""
        if self._bot_speaking:
            self._bot_speaking = False
            await self.push_frame(TTSStoppedFrame())
        # Deepslate sends this event on every speech start, so only interrupt
        # when we actually pushed audio since the last interruption.
        if self._audio_pending:
            self._audio_pending = False
            logger.debug("Deepslate interruption (barge-in) received")
            await self.broadcast_interruption()

    async def _on_error(self, category: str, message: str, trace_id: Optional[str]) -> None:
        suffix = f" (trace_id={trace_id})" if trace_id else ""
        text = f"Deepslate {category}: {message}{suffix}"
        logger.warning(text)
        await self.push_error(ErrorFrame(text))

    async def _on_fatal_error(self, err: Exception) -> None:
        await self._fatal(f"Deepslate connection lost: {err}")

    # -------------------------------------------------------------- tool calls

    def _on_tool_call(self, call_id: str, name: str, params: dict) -> None:
        task = asyncio.create_task(
            self._run_tool(call_id, name, params), name=f"DeepslateLive.tool.{name}"
        )
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool(self, call_id: str, name: str, params: dict) -> None:
        arguments = _normalize_numbers(params or {})
        logger.info("Deepslate tool call: {}({})", name, arguments)

        if self._tool_handler is None:
            result = f"Tool {name} is not available in this installation."
        else:
            try:
                value = await asyncio.wait_for(
                    self._tool_handler(name, arguments),
                    timeout=DEEPSLATE_TOOL_TIMEOUT_SECONDS,
                )
                result = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False, default=str
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                logger.warning("Deepslate tool {} timed out", name)
                result = f"Tool {name} timed out."
            except Exception as err:  # noqa: BLE001
                logger.exception("Deepslate tool {} failed", name)
                result = f"Tool {name} failed: {err}"

        # Every ToolCallRequest must be answered, even when execution failed.
        session = self._session
        if session is None:
            return
        try:
            await session.send_tool_response(call_id, result)
        except Exception as err:  # noqa: BLE001
            logger.warning("Deepslate tool response for {} failed: {}", name, err)


# ---------------------------------------------------------------------- helpers


def _normalise_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert a Pipecat ``ToolsSchema`` (or OpenAI style list) to Deepslate tools."""
    if not tools:
        return []

    standard = getattr(tools, "standard_tools", None)
    if standard is not None:
        converted: list[dict[str, Any]] = []
        for schema in standard:
            try:
                function = schema.to_default_dict()
            except Exception:  # noqa: BLE001
                function = {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": schema.properties or {},
                        "required": schema.required or [],
                    },
                }
            converted.append({"type": "function", "function": function})
        return converted

    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, dict)]
    return []


def _normalize_numbers(value: Any) -> Any:
    """Protobuf ``Struct`` stores all numbers as doubles: turn 50.0 back into 50.

    Home Assistant MCP tools declare e.g. ``brightness`` as ``integer`` and
    reject ``50.0``.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def _downmix_to_mono(audio: bytes, channels: int) -> bytes:
    """Average interleaved PCM16 channels into mono."""
    import numpy as np

    usable = len(audio) - (len(audio) % (2 * channels))
    if usable <= 0:
        return b""
    samples = np.frombuffer(audio[:usable], dtype=np.int16).reshape(-1, channels)
    return samples.mean(axis=1).astype(np.int16).tobytes()
