"""Detect a double-clap on the default microphone and fire a callback.

Single claps are intentionally ignored — double-clap is the only wake gesture
for Javis. Detection is amplitude-based: any sample whose absolute level
exceeds CLAP_THRESHOLD is treated as a clap candidate, then debounced via
COOLDOWN and grouped within DOUBLE_WINDOW seconds.

Run directly for a smoke test:
    python clap_detector.py
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
BLOCK_SIZE = 512          # ~32ms per block at 16kHz
CLAP_THRESHOLD = 0.20     # absolute amplitude (float32, mic-dependent)
COOLDOWN = 0.15           # min seconds between two distinct claps
DOUBLE_WINDOW = 0.90      # max seconds between the two claps of a "double"


class ClapDetector:
    def __init__(self, on_double: Callable[[], None]) -> None:
        self._on_double = on_double
        self._last_clap_at: float = 0.0
        self._stream: Optional[sd.InputStream] = None
        self._paused = threading.Event()
        self._running = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if self._paused.is_set() or not self._running.is_set():
            return
        peak = float(np.max(np.abs(indata)))
        if peak < CLAP_THRESHOLD:
            return
        now = time.monotonic()
        since = now - self._last_clap_at
        if since < COOLDOWN:
            return
        if since <= DOUBLE_WINDOW:
            self._last_clap_at = 0.0
            try:
                self._on_double()
            except Exception as exc:
                print(f"[ClapDetector] on_double raised: {exc}", file=sys.stderr)
            return
        self._last_clap_at = now

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._last_clap_at = 0.0
        self._paused.clear()


if __name__ == "__main__":
    print("ClapDetector smoke test — double-clap to trigger, Ctrl+C to quit.")
    print(f"Threshold={CLAP_THRESHOLD}, window={DOUBLE_WINDOW}s")

    hits = {"n": 0}

    def on_double() -> None:
        hits["n"] += 1
        print(f"  double-clap #{hits['n']} at {time.strftime('%H:%M:%S')}")

    det = ClapDetector(on_double=on_double)
    det.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        det.stop()
