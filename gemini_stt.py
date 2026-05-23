"""Speech-to-text via Gemini API.

Two paths:
- `transcribe(wav_bytes)` — sync one-shot via `gemini-2.5-flash`. Kept as the
  fallback used by VoiceCapture when the Live session fails.
- `stream(audio_chunks, on_partial)` — async, opens a Gemini Live session
  (`gemini-live-2.5-flash-preview`), forwards raw 16 kHz mono PCM chunks as
  the user speaks, surfaces partial transcripts via `on_partial(running_text)`
  and returns the final transcript when the audio stream ends.

Set GEMINI_API_KEY in the environment before constructing.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import AsyncIterator, Callable, Optional

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
    from pathlib import Path as _Path
    for _c in (_Path(__file__).parent / ".env", _Path(__file__).parent / ".env.example"):
        if _c.exists():
            _load_dotenv(_c, override=False)
except ImportError:
    pass

MODEL = "gemini-2.5-flash"
LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"
PROMPT = (
    "Transcribe the user's speech verbatim. "
    "Return ONLY the transcription text, with no commentary, no quotes, "
    "no language label. Detect language automatically (Thai, Lao, or English)."
)
# The native-audio Live model requires response_modalities=["AUDIO"] (it
# rejects TEXT-only with "Cannot extract voices from a non-audio request").
# We only care about input_transcription, so this instruction tells the
# model to keep its spoken reply empty — minimizes output audio cost.
_LIVE_SILENCE = (
    "You are a silent transcription microphone. Never speak, greet, "
    "answer, or react. Produce no audio output. Stay completely quiet."
)


class GeminiSTT:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)

    def transcribe(self, wav_bytes: bytes) -> str:
        try:
            response = self._client.models.generate_content(
                model=MODEL,
                contents=[
                    PROMPT,
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                ],
            )
            text = (response.text or "").strip()
            return text.strip("\"'`")
        except Exception as exc:
            print(f"[GeminiSTT] transcribe failed: {exc}", file=sys.stderr)
            return ""

    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Stream raw int16 PCM @ 16 kHz to Gemini Live; return final transcript.

        `audio_chunks` yields raw little-endian int16 mono PCM bytes (any chunk
        size is fine; we just forward them). End the iterator to signal
        end-of-utterance — we then flush via `audio_stream_end=True` and wait
        for the final transcription event.
        """
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=_LIVE_SILENCE,
        )
        parts: list[str] = []

        async with self._client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            async def _pump() -> None:
                try:
                    async for chunk in audio_chunks:
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                        )
                finally:
                    try:
                        await session.send_realtime_input(audio_stream_end=True)
                    except Exception:
                        pass

            pump_task = asyncio.create_task(_pump())
            got_transcript = False
            try:
                async for msg in session.receive():
                    sc = msg.server_content
                    if sc is None:
                        continue
                    tr = sc.input_transcription
                    if tr and tr.text:
                        got_transcript = True
                        parts.append(tr.text)
                        if on_partial:
                            try:
                                on_partial("".join(parts))
                            except Exception:
                                pass
                    # Bail out once transcription is done and the model
                    # starts its (unused) spoken reply — saves output-audio
                    # tokens we'd otherwise throw away.
                    if got_transcript and sc.model_turn is not None:
                        break
                    if sc.turn_complete:
                        break
            except asyncio.CancelledError:
                # Caller bailed (no-speech short-circuit or post-stream-end
                # timeout) — propagate after the finally below closes pump.
                raise
            finally:
                if not pump_task.done():
                    pump_task.cancel()
                    try:
                        await pump_task
                    except (asyncio.CancelledError, Exception):
                        pass

        return "".join(parts).strip().strip("\"'`")


if __name__ == "__main__":
    import wave

    if len(sys.argv) < 2:
        print("Usage: python gemini_stt.py <path-to.wav>")
        raise SystemExit(1)
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    with wave.open(sys.argv[1], "rb") as w:
        print(f"  {w.getnchannels()}ch  {w.getframerate()}Hz  {w.getnframes()} frames")
    print("transcript:", GeminiSTT().transcribe(data))
