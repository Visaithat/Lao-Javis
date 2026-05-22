"""Speech-to-text via Gemini API (`gemini-2.5-flash`).

Takes raw WAV bytes (mono, 16-bit PCM, 16kHz) and returns the transcribed
text. The Gemini multimodal model auto-detects language — works for Thai,
Lao, and English without a hint.

Set GEMINI_API_KEY in the environment before constructing.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

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
PROMPT = (
    "Transcribe the user's speech verbatim. "
    "Return ONLY the transcription text, with no commentary, no quotes, "
    "no language label. Detect language automatically (Thai, Lao, or English)."
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
