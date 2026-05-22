"""Mic capture with Silero VAD — listens until the user stops talking.

`listen_once()` opens the default microphone, waits for the user to begin
speaking, keeps buffering until ~700 ms of trailing silence, then sends the
captured WAV to Gemini for transcription. Returns the transcribed text or
None if nothing usable was heard inside the timeout.

Stream layout:
    16 kHz mono int16 PCM, 512-sample chunks (32 ms) — Silero's fixed window.
"""
from __future__ import annotations

import io
import sys
import time
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

from gemini_stt import GeminiSTT

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512                # required by Silero at 16kHz (~32ms)
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE
SPEECH_THRESHOLD = 0.5             # Silero's recommended default
SILENCE_END_MS = 400               # stop after this much trailing silence (was 700)
MIN_SPEECH_MS = 250                # discard utterances shorter than this (was 400)
DEFAULT_TIMEOUT_S = 30             # bail out if no speech is heard


def _to_wav_bytes(pcm_int16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


class VoiceCapture:
    def __init__(
        self,
        stt: Optional[GeminiSTT] = None,
        on_amplitude: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._stt = stt or GeminiSTT()
        self._model = load_silero_vad()
        self._on_amplitude = on_amplitude

    def listen_once(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
        """Block until the user finishes speaking; return transcribed text.

        Returns None if no speech was detected within `timeout_s`, or if the
        captured speech was too short to be meaningful.
        """
        self._model.reset_states()
        buffered: list[np.ndarray] = []
        speech_started = False
        silence_ms = 0
        speech_ms = 0
        deadline = time.monotonic() + timeout_s

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while time.monotonic() < deadline:
                data, _ = stream.read(CHUNK_SAMPLES)
                int16 = np.frombuffer(bytes(data), dtype=np.int16)
                if int16.size != CHUNK_SAMPLES:
                    continue
                f32 = int16.astype(np.float32) / 32768.0
                if self._on_amplitude is not None:
                    try:
                        peak = float(np.max(np.abs(f32)))
                        self._on_amplitude(peak)
                    except Exception:
                        pass
                with torch.no_grad():
                    prob = self._model(torch.from_numpy(f32), SAMPLE_RATE).item()
                is_speech = prob >= SPEECH_THRESHOLD

                if is_speech:
                    if not speech_started:
                        speech_started = True
                    buffered.append(int16)
                    speech_ms += CHUNK_MS
                    silence_ms = 0
                elif speech_started:
                    buffered.append(int16)            # keep a little trailing silence
                    silence_ms += CHUNK_MS
                    if silence_ms >= SILENCE_END_MS:
                        break

        if not speech_started or speech_ms < MIN_SPEECH_MS:
            return None

        pcm = np.concatenate(buffered)
        wav_bytes = _to_wav_bytes(pcm, SAMPLE_RATE)
        text = self._stt.transcribe(wav_bytes)
        return text or None


if __name__ == "__main__":
    print("VoiceCapture smoke test — speak after the prompt, Ctrl+C to quit.")
    vc = VoiceCapture()
    try:
        while True:
            print("\nlistening...")
            t = vc.listen_once()
            print(f"  → {t!r}" if t else "  (nothing heard)")
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
