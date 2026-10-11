#!/usr/bin/with-contenv bashio
set -e

RUNNER_HOST="0.0.0.0"
RUNNER_PORT="$(bashio::config 'runner_port')"
LOG_LEVEL="$(bashio::config 'log_level')"

# Deepslate server-side VAD tuning (read by app/deepslate_live.py; an unset or
# invalid value falls back to the built-in default there).
DEEPSLATE_VAD_CONFIDENCE="$(bashio::config 'deepslate_vad_confidence')"
DEEPSLATE_VAD_MIN_VOLUME="$(bashio::config 'deepslate_vad_min_volume')"
DEEPSLATE_VAD_START_MS="$(bashio::config 'deepslate_vad_start_ms')"
DEEPSLATE_VAD_STOP_MS="$(bashio::config 'deepslate_vad_stop_ms')"

export RUNNER_HOST
export RUNNER_PORT
export LOG_LEVEL
export DEEPSLATE_VAD_CONFIDENCE
export DEEPSLATE_VAD_MIN_VOLUME
export DEEPSLATE_VAD_START_MS
export DEEPSLATE_VAD_STOP_MS

ARGS=(--host "$RUNNER_HOST" --port "$RUNNER_PORT" -t webrtc)

exec python3 -m app.main "${ARGS[@]}"
