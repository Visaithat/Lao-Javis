"""Javis — voice-controlled PC assistant. Orchestrates every module.

Flow:
    1. Idle: ClapDetector listens in the background.
    2. Double-clap → enter conversation mode, say a short greeting.
    3. Listen via VAD → transcribe via Gemini → check for a quit phrase.
       If not quitting, send the text to the Claude agent and speak the reply.
    4. Loop on step 3 indefinitely; mic timeouts just restart listening so
       the assistant is always ready for the next utterance.
    5. Quit phrase → speak goodbye and exit the program.

Run:
    python main.py
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_HERE = Path(__file__).parent
for _candidate in (_HERE / ".env", _HERE / ".env.example"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)

from agent import Agent  # noqa: E402  (must come after dotenv load)
from clap_detector import ClapDetector  # noqa: E402
from fast_path import try_match as fast_path_match  # noqa: E402
from text_to_speech import StreamingTTS, TextToSpeech  # noqa: E402
from ui import JavisOverlay  # noqa: E402
from voice_capture import VoiceCapture  # noqa: E402

QUIT_PHRASES = [
    "ปิดโปรแกรม", "ปิดระบบ", "หยุดทำงาน", "พอแล้ว", "ปิดตัว",
    "ປິດໂປຣແກຣມ", "ປິດລະບົບ", "ປິດເຄື່ອງ",
    "stop", "exit", "quit", "shut down", "shutdown", "goodbye", "bye",
]

GREETING = "ครับ"
GOODBYE = "บายๆ"
ACK = "ครับ"  # said immediately while Claude is thinking, so user knows we heard

LISTEN_TIMEOUT_S = 30       # how long each listen attempt waits for speech
# Stay in conversation mode indefinitely; only voice quit phrase or kill
# closes it. User asked for "always ready for next speech" — no auto-idle.


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip(".!?。。").strip()


def is_quit(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    for phrase in QUIT_PHRASES:
        if phrase in norm:
            return True
    return False


async def _conversation(
    agent: Agent,
    voice: VoiceCapture,
    tts: TextToSpeech,
    streaming_tts: StreamingTTS,
    ui: JavisOverlay,
) -> bool:
    """Run one wake → conversation cycle. Returns True if the user asked to quit."""
    ui.set_state("speaking")
    await asyncio.to_thread(tts.speak, GREETING)
    import time as _time
    while True:
        ui.set_state("listening")
        ui.set_partial_transcript("")
        t0 = _time.monotonic()
        text: Optional[str] = await voice.listen_once(LISTEN_TIMEOUT_S)
        ui.set_partial_transcript("")
        if not text:
            # No speech this round — just keep listening. UI stays "listening".
            continue
        print(f"[user +{_time.monotonic() - t0:.1f}s] {text}")
        if is_quit(text):
            ui.set_state("speaking")
            await asyncio.to_thread(tts.speak, GOODBYE)
            return True

        # --- fast path: regex intent (no LLM) ---
        fp = fast_path_match(text)
        if fp is not None:
            reply, action = fp
            ui.set_state("speaking")
            await asyncio.to_thread(action)
            streaming_tts.enqueue(reply)
            await asyncio.to_thread(streaming_tts.flush_and_wait)
            print(f"[fast-path +{_time.monotonic() - t0:.1f}s] {reply}")
            continue

        # --- slow path: Claude with acknowledge-first ---
        ui.set_state("thinking")
        streaming_tts.enqueue(ACK)

        def _on_chunk(s: str) -> None:
            elapsed = _time.monotonic() - t0
            print(f"[agent +{elapsed:4.1f}s] {s}")
            ui.set_state("speaking")
            streaming_tts.enqueue(s)

        try:
            await asyncio.to_thread(agent.send_streaming, text, _on_chunk)
        except Exception as exc:
            print(f"[agent] error: {exc}")
            ui.set_state("speaking")
            await asyncio.to_thread(tts.speak, "ขอโทษ มีบั๊กนิดนึง ลองอีกทีนะ")
            continue
        # wait for all queued sentences to finish playing before next turn
        await asyncio.to_thread(streaming_tts.flush_and_wait)
        print(f"[main] turn done in {_time.monotonic() - t0:.1f}s")


async def run(ui: JavisOverlay) -> None:
    tts = TextToSpeech()
    streaming_tts = StreamingTTS()
    voice = VoiceCapture(
        on_amplitude=ui.feed_amplitude,
        on_partial=ui.set_partial_transcript,
    )
    agent = Agent()

    wake_event = threading.Event()
    clap = ClapDetector(on_double=wake_event.set)
    clap.start()

    await asyncio.to_thread(tts.speak, "พร้อมแล้ว ตบมือสองครั้งเพื่อปลุก")
    print("Javis ready — double-clap to wake. Speak 'ปิดโปรแกรม' / 'stop' to quit.")
    try:
        while True:
            await asyncio.to_thread(wake_event.wait)
            wake_event.clear()
            print("[main] wake")
            clap.pause()
            ui.show()
            try:
                quitting = await _conversation(agent, voice, tts, streaming_tts, ui)
            finally:
                ui.hide()
                clap.resume()
            if quitting:
                break
    finally:
        streaming_tts.stop()
        clap.stop()
        ui.quit()


def main() -> None:
    """Run tkinter on main thread; async backend in worker."""
    ui = JavisOverlay()

    def _backend() -> None:
        try:
            asyncio.run(run(ui))
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            print(f"[backend] crashed: {exc}")
        finally:
            ui.quit()

    threading.Thread(target=_backend, daemon=True, name="javis-backend").start()
    ui.mainloop()


if __name__ == "__main__":
    main()
