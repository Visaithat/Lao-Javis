"""Text-to-speech via Microsoft Edge online voices (`edge-tts`).

Auto-picks a voice per response based on Unicode range: Lao → Lao voice,
Thai → Thai voice, anything else → English. `speak()` blocks until playback
finishes; while it is speaking the `is_speaking` flag is set so the caller
can pause its mic/clap listeners and avoid feedback.
"""
from __future__ import annotations

import asyncio
import os
import queue
import sys
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import edge_tts

VOICE_THAI = "th-TH-PremwadeeNeural"
VOICE_LAO = "lo-LA-KeomanyNeural"
VOICE_EN = "en-US-AriaNeural"

LAO_RANGE = (0x0E80, 0x0EFF)
THAI_RANGE = (0x0E00, 0x0E7F)


def _pick_voice(text: str) -> str:
    has_thai = False
    has_lao = False
    for ch in text:
        cp = ord(ch)
        if LAO_RANGE[0] <= cp <= LAO_RANGE[1]:
            has_lao = True
            break
        if THAI_RANGE[0] <= cp <= THAI_RANGE[1]:
            has_thai = True
    if has_lao:
        return VOICE_LAO
    if has_thai:
        return VOICE_THAI
    return VOICE_EN


def _play_blocking(path: str) -> None:
    """Play an MP3 file synchronously on Windows.

    Tries `playsound` first (pure-Python, no PortAudio); falls back to
    `winsound.PlaySound` after converting MP3 → WAV is overkill, so on
    failure we just rely on the OS default `start /wait` shell call.
    """
    try:
        from playsound3 import playsound  # type: ignore
        playsound(path, block=True)
        return
    except Exception as exc:
        print(f"[TTS] playsound3 failed ({exc}); falling back", file=sys.stderr)
    try:
        import pygame  # type: ignore
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
        pygame.mixer.quit()
        return
    except Exception as exc:
        print(f"[TTS] pygame fallback failed: {exc}", file=sys.stderr)


class TextToSpeech:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_speaking = threading.Event()

    def speak(self, text: str, voice: Optional[str] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        chosen = voice or _pick_voice(text)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            with self._lock:
                self.is_speaking.set()
                try:
                    asyncio.run(self._synthesize(text, chosen, tmp.name))
                    _play_blocking(tmp.name)
                finally:
                    self.is_speaking.clear()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @staticmethod
    async def _synthesize(text: str, voice: str, out_path: str) -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(out_path)


class _SynthResult:
    """Holds a synthesized MP3 path + the original text (for logging)."""
    __slots__ = ("text", "path")

    def __init__(self, text: str, path: str) -> None:
        self.text = text
        self.path = path


class StreamingTTS:
    """Pipeline TTS: synthesize next sentence while current one is playing.

    Two worker threads:
      • synth worker: pulls sentence from `_text_q`, runs edge-tts → MP3 path,
        pushes onto `_audio_q`. Multiple sentences synth in parallel.
      • play worker: pulls MP3 path from `_audio_q`, plays it through to end,
        then deletes the temp file.

    Caller pattern:
        st = StreamingTTS()
        st.enqueue("first sentence")
        st.enqueue("second sentence")
        ...
        st.flush_and_wait()  # block until everything spoken
    """

    _DONE = object()

    def __init__(self) -> None:
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()
        self._stopping = threading.Event()
        self.is_speaking = threading.Event()
        self._synth_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-synth")
        self._play_thread = threading.Thread(target=self._play_loop, name="tts-play", daemon=True)
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="tts-dispatch", daemon=True)
        self._play_thread.start()
        self._dispatch_thread.start()

    def enqueue(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._text_q.put(text)

    def flush_and_wait(self) -> None:
        """Signal end-of-utterance and block until everything finishes playing."""
        self._text_q.put(self._DONE)
        self._dispatch_thread.join()
        self._audio_q.put(self._DONE)
        self._play_thread.join()
        # restart workers for next utterance
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="tts-dispatch", daemon=True)
        self._play_thread = threading.Thread(target=self._play_loop, name="tts-play", daemon=True)
        self._dispatch_thread.start()
        self._play_thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._text_q.put(self._DONE)
        self._audio_q.put(self._DONE)
        self._synth_pool.shutdown(wait=False)

    # ---- internals ----

    def _dispatch_loop(self) -> None:
        """Pull text, schedule synth, push completed audio onto play queue in order."""
        pending: list[tuple[str, Future]] = []
        while True:
            item = self._text_q.get()
            if item is self._DONE:
                # drain remaining synths
                for text, fut in pending:
                    try:
                        path = fut.result(timeout=30)
                        self._audio_q.put(_SynthResult(text, path))
                    except Exception as exc:
                        print(f"[TTS] synth failed for {text[:40]!r}: {exc}", file=sys.stderr)
                return
            text: str = item
            fut = self._synth_pool.submit(_synthesize_to_tempfile, text)
            pending.append((text, fut))
            # flush any that are already done, in order
            while pending and pending[0][1].done():
                t, f = pending.pop(0)
                try:
                    path = f.result()
                    self._audio_q.put(_SynthResult(t, path))
                except Exception as exc:
                    print(f"[TTS] synth failed for {t[:40]!r}: {exc}", file=sys.stderr)

    def _play_loop(self) -> None:
        while True:
            item = self._audio_q.get()
            if item is self._DONE:
                self.is_speaking.clear()
                return
            self.is_speaking.set()
            try:
                _play_blocking(item.path)
            except Exception as exc:
                print(f"[TTS] play failed: {exc}", file=sys.stderr)
            finally:
                try:
                    os.unlink(item.path)
                except OSError:
                    pass


def _synthesize_to_tempfile(text: str) -> str:
    voice = _pick_voice(text)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    asyncio.run(_synth_async(text, voice, tmp.name))
    return tmp.name


async def _synth_async(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text=text, voice=voice).save(path)


if __name__ == "__main__":
    import time
    if "--stream" in sys.argv:
        st = StreamingTTS()
        for s in [
            "สวัสดีครับ ผมชื่อ Javis",
            "วันนี้อากาศดีมากเลย",
            "Hello, this is the streaming test.",
        ]:
            print(f"[{time.strftime('%H:%M:%S')}] enqueue: {s}")
            st.enqueue(s)
            time.sleep(0.1)
        st.flush_and_wait()
        print("done")
    else:
        tts = TextToSpeech()
        for s in [
            "Hello, this is the English voice.",
            "สวัสดีครับ นี่คือเสียงภาษาไทย",
            "ສະບາຍດີ ນີ້ແມ່ນສຽງພາສາລາວ",
        ]:
            print(f"speaking: {s}")
            tts.speak(s)
