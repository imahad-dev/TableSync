"""
TableSync — Speechmatics Voice Transcription Frontend
======================================================
Wires spoken-instruction transcription in front of the planner.

Accepts either:
  1. A .wav file path   (offline / test mode)
  2. A live microphone   (real-time operator mode)

Outputs: a single cleaned text string ready for the Gemini multimodal
planner that produces PlannerOutput.

Dependencies:
  pip install speechmatics-python sounddevice

Environment:
  SPEECHMATICS_API_KEY  — your Speechmatics SaaS API key
  SPEECHMATICS_RT_URL   — (optional) override the default RT endpoint
                           defaults to wss://eu2.rt.speechmatics.com/v2
"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import speechmatics
    from speechmatics.models import (
        ConnectionSettings,
        TranscriptionConfig,
        AudioSettings,
    )
except ImportError:
    raise ImportError(
        "speechmatics-python is required: pip install speechmatics-python"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEECHMATICS_RT_URL = os.environ.get(
    "SPEECHMATICS_RT_URL",
    "wss://eu2.rt.speechmatics.com/v2",
)

# Domain-specific vocabulary boost for bimanual robotics workspace
CUSTOM_DICTIONARY: list[dict] = [
    {"content": "TableSync",   "sounds_like": ["table sync", "table sink"]},
    {"content": "bimanual",    "sounds_like": ["by manual", "bi manual"]},
    {"content": "handoff",     "sounds_like": ["hand off", "hand-off"]},
    {"content": "SO-101",      "sounds_like": ["S O one oh one", "so one oh one"]},
    {"content": "plate"},
    {"content": "spoon"},
    {"content": "gripper"},
]

SAMPLE_RATE = 16_000  # 16 kHz PCM, matching Speechmatics RT recommendations
CHUNK_DURATION_S = 0.1  # 100ms audio chunks for low-latency streaming


@dataclass
class TranscriptionResult:
    """What the voice frontend hands to the planner."""
    final_text: str
    confidence: float  # average token confidence, 0.0–1.0
    partial_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core transcriber
# ---------------------------------------------------------------------------

class VoiceTranscriber:
    """Speechmatics Real-Time ASR frontend for TableSync.

    Usage (file):
        transcriber = VoiceTranscriber(api_key="...")
        result = await transcriber.transcribe_file("command.wav")
        print(result.final_text)  # → "pick up the plate and hand the spoon"

    Usage (microphone, press Ctrl+C to stop):
        result = await transcriber.transcribe_microphone()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        rt_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Speechmatics API key required. Set SPEECHMATICS_API_KEY or "
                "pass api_key= to VoiceTranscriber."
            )
        self.language = language
        self.rt_url = rt_url or SPEECHMATICS_RT_URL
        self._partials: list[str] = []
        self._finals: list[str] = []
        self._confidences: list[float] = []

    def _reset(self) -> None:
        self._partials.clear()
        self._finals.clear()
        self._confidences.clear()

    def _build_config(self) -> TranscriptionConfig:
        return TranscriptionConfig(
            language=self.language,
            enable_partials=True,
            additional_vocab=CUSTOM_DICTIONARY,
            punctuation_overrides={
                "permitted_marks": [".", ",", "?", "!"],
            },
        )

    def _build_connection(self) -> ConnectionSettings:
        return ConnectionSettings(
            url=self.rt_url,
            auth_token=self.api_key,
        )

    def _build_audio_settings(self, sample_rate: int = SAMPLE_RATE) -> AudioSettings:
        return AudioSettings(
            sample_rate=sample_rate,
            encoding="pcm_s16le",
        )

    # -- Callbacks ----------------------------------------------------------

    def _on_partial(self, msg: dict) -> None:
        text = msg.get("metadata", {}).get("transcript", "")
        if not text:
            text = " ".join(
                w.get("content", "") for w in msg.get("results", [])
            )
        if text.strip():
            self._partials.append(text.strip())

    def _on_final(self, msg: dict) -> None:
        text = msg.get("metadata", {}).get("transcript", "")
        if not text:
            text = " ".join(
                w.get("content", "") for w in msg.get("results", [])
            )
        if text.strip():
            self._finals.append(text.strip())
            for w in msg.get("results", []):
                conf = w.get("confidence", 0.0)
                if conf > 0:
                    self._confidences.append(conf)

    # -- File transcription -------------------------------------------------

    async def transcribe_file(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe a .wav file and return clean text for the planner."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._reset()

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            audio_data = wf.readframes(wf.getnframes())

        if n_channels != 1:
            raise ValueError(
                f"Expected mono audio, got {n_channels} channels. "
                "Convert with: ffmpeg -i input.wav -ac 1 -ar 16000 output.wav"
            )

        ws = speechmatics.client.WebsocketClient(self._build_connection())
        ws.add_event_handler(
            speechmatics.models.ServerMessageType.AddPartialTranscript,
            self._on_partial,
        )
        ws.add_event_handler(
            speechmatics.models.ServerMessageType.AddTranscript,
            self._on_final,
        )

        audio_settings = self._build_audio_settings(sample_rate)

        # Stream the file in chunks
        chunk_size = int(sample_rate * CHUNK_DURATION_S) * sampwidth
        stream = io.BytesIO(audio_data)

        async def audio_generator():
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        await ws.run(
            audio_generator(),
            self._build_config(),
            audio_settings,
        )

        return self._build_result()

    # -- Microphone transcription -------------------------------------------

    async def transcribe_microphone(
        self,
        duration_s: Optional[float] = None,
    ) -> TranscriptionResult:
        """Stream from the default microphone until Ctrl+C or duration_s elapsed.

        Requires: pip install sounddevice
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone input: "
                "pip install sounddevice"
            )

        self._reset()

        ws = speechmatics.client.WebsocketClient(self._build_connection())
        ws.add_event_handler(
            speechmatics.models.ServerMessageType.AddPartialTranscript,
            self._on_partial,
        )
        ws.add_event_handler(
            speechmatics.models.ServerMessageType.AddTranscript,
            self._on_final,
        )

        audio_settings = self._build_audio_settings(SAMPLE_RATE)
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION_S)
        stop_event = asyncio.Event()

        if duration_s:
            asyncio.get_event_loop().call_later(duration_s, stop_event.set)

        async def mic_generator():
            loop = asyncio.get_event_loop()
            q: asyncio.Queue[bytes] = asyncio.Queue()

            def _callback(indata, frames, time_info, status):
                loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
                callback=_callback,
            ):
                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                        yield chunk
                    except asyncio.TimeoutError:
                        continue

        try:
            await ws.run(
                mic_generator(),
                self._build_config(),
                audio_settings,
            )
        except KeyboardInterrupt:
            pass

        return self._build_result()

    # -- Result assembly ----------------------------------------------------

    def _build_result(self) -> TranscriptionResult:
        final_text = " ".join(self._finals).strip()
        avg_confidence = (
            sum(self._confidences) / len(self._confidences)
            if self._confidences
            else 0.0
        )
        return TranscriptionResult(
            final_text=final_text,
            confidence=round(avg_confidence, 4),
            partial_texts=list(self._partials),
        )


# ---------------------------------------------------------------------------
# Convenience: transcribe → planner text (the wiring point)
# ---------------------------------------------------------------------------

async def transcribe_spoken_command(
    audio_path: Optional[str | Path] = None,
    api_key: Optional[str] = None,
    duration_s: Optional[float] = None,
) -> str:
    """One-call entry point: spoken audio → clean text for the planner.

    This is the function that sits between the operator's voice and
    PlannerOutput construction. It returns the raw_instruction string
    that feeds into PlannerOutput.raw_instruction (and the Gemini
    multimodal prompt).

    Args:
        audio_path: Path to a .wav file, or None for live microphone.
        api_key: Speechmatics API key (or set SPEECHMATICS_API_KEY env var).
        duration_s: For microphone mode, auto-stop after this many seconds.

    Returns:
        The final transcribed text string, ready for the planner.
    """
    transcriber = VoiceTranscriber(api_key=api_key)

    if audio_path is not None:
        result = await transcriber.transcribe_file(audio_path)
    else:
        result = await transcriber.transcribe_microphone(duration_s=duration_s)

    if not result.final_text:
        raise RuntimeError("Speechmatics returned empty transcription — no speech detected.")

    print(f"[VoiceTranscriber] Transcribed ({result.confidence:.0%} avg confidence):")
    print(f"  \"{result.final_text}\"")

    return result.final_text


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Transcribing file: {path}")
        text = asyncio.run(transcribe_spoken_command(audio_path=path))
    else:
        print("No file provided — streaming from microphone (Ctrl+C to stop)...")
        text = asyncio.run(transcribe_spoken_command(duration_s=10.0))

    print(f"\n→ Planner input: \"{text}\"")
