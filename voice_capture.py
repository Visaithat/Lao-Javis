"""Mic capture with Silero VAD — listens until the user stops talking.

`listen_once()` opens the default microphone, waits for the user to begin
speaking, and streams raw PCM chunks to Gemini Live for real-time
transcription. Partial transcripts arrive via `on_partial`; the final
transcript is returned once Silero detects the user finished talking.

If the Live session fails (network, model unavailable, etc.) we transparently
fall back to the legacy one-shot `transcribe(wav_bytes)` path using audio
that was buffered in parallel.

Stream layout:
    16 kHz mono int16 PCM, 512-sample chunks (32 ms) — Silero's fixed window.
"""
from __future__ import annotations

import asyncio
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
SILENCE_END_MS = 400               # stop after this much trailing silence
MIN_SPEECH_MS = 250                # discard utterances shorter than this
DEFAULT_TIMEOUT_S = 30             # bail out if no speech is heard
LIVE_FINAL_TIMEOUT_S = 5.0         # max wait for Gemini Live to emit transcript after audio_stream_end


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
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._stt = stt or GeminiSTT()
        self._model = load_silero_vad()
        self._on_amplitude = on_amplitude
        self._on_partial = on_partial

    async def listen_once(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[str]:
        """Block until the user finishes speaking; return transcribed text.

        Returns None if no speech was detected within `timeout_s`, or if the
        captured speech was too short to be meaningful.
        """
        print("[voice_capture] listening...", flush=True)
        loop = asyncio.get_running_loop()
        chunk_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        fallback_buf: list[np.ndarray] = []
        result: dict[str, object] = {"speech_started": False, "speech_ms": 0}

        def _capture_blocking() -> None:
            """Sync VAD capture loop running in a worker thread."""
            self._model.reset_states()
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
                    pcm_bytes = bytes(data)
                    int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                    if int16.size != CHUNK_SAMPLES:
                        continue
                    f32 = int16.astype(np.float32) / 32768.0
                    if self._on_amplitude is not None:
                        try:
                            self._on_amplitude(float(np.max(np.abs(f32))))
                        except Exception:
                            pass
                    with torch.no_grad():
                        prob = self._model(torch.from_numpy(f32), SAMPLE_RATE).item()
                    is_speech = prob >= SPEECH_THRESHOLD

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                        fallback_buf.append(int16)
                        loop.call_soon_threadsafe(chunk_q.put_nowait, pcm_bytes)
                        speech_ms += CHUNK_MS
                        silence_ms = 0
                    elif speech_started:
                        fallback_buf.append(int16)            # keep a little trailing silence
                        loop.call_soon_threadsafe(chunk_q.put_nowait, pcm_bytes)
                        silence_ms += CHUNK_MS
                        if silence_ms >= SILENCE_END_MS:
                            break

            result["speech_started"] = speech_started
            result["speech_ms"] = speech_ms
            loop.call_soon_threadsafe(chunk_q.put_nowait, None)

        async def _chunk_iter():
            while True:
                chunk = await chunk_q.get()
                if chunk is None:
                    return
                yield chunk

        cap_task = asyncio.create_task(asyncio.to_thread(_capture_blocking))
        stream_task = asyncio.create_task(
            self._stt.stream(_chunk_iter(), on_partial=self._on_partial)
        )

        try:
            await cap_task
        except Exception as exc:
            print(f"[voice_capture] capture thread error: {exc}", file=sys.stderr)

        print(
            f"[voice_capture] capture done speech={result['speech_started']} "
            f"ms={result['speech_ms']}",
            flush=True,
        )

        speech_started = bool(result["speech_started"])
        speech_ms = int(result["speech_ms"])  # type: ignore[arg-type]

        if not speech_started or speech_ms < MIN_SPEECH_MS:
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass
            return None

        text = ""
        used_fallback = False
        try:
            text = await asyncio.wait_for(stream_task, timeout=LIVE_FINAL_TIMEOUT_S)
        except (asyncio.TimeoutError, Exception) as exc:
            used_fallback = True
            print(
                f"[voice_capture] Live did not return in {LIVE_FINAL_TIMEOUT_S}s "
                f"({type(exc).__name__}), falling back to one-shot",
                file=sys.stderr,
            )
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass

        if used_fallback and fallback_buf:
            pcm = np.concatenate(fallback_buf)
            wav_bytes = _to_wav_bytes(pcm, SAMPLE_RATE)
            text = await asyncio.to_thread(self._stt.transcribe, wav_bytes)

        text = text.strip()
        return text or None


if __name__ == "__main__":
    print("VoiceCapture smoke test — speak after the prompt, Ctrl+C to quit.")

    def _print_partial(s: str) -> None:
        # carriage-return overwrite so the line grows in place
        print(f"\r  partial: {s}", end="", flush=True)

    async def _amain() -> None:
        vc = VoiceCapture(on_partial=_print_partial)
        while True:
            print("\nlistening...")
            t = await vc.listen_once()
            print()  # newline after the partial line
            print(f"  → {t!r}" if t else "  (nothing heard)")

    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
