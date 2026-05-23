"""JavisOverlay — frameless desktop popup powered by pywebview.

Wraps Microsoft Edge WebView2 (bundled with Windows 10+) so we can use HTML +
CSS + Canvas for a smooth visualizer, while still controlling show/hide/state
from Python like a native app.

Loads `web/index.html` from the project directory and pushes state and mic
amplitude into the page via `evaluate_js`. The same API as the old tkinter
version, so main.py needs no changes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import webview

WIN_W = 380
WIN_H = 220
TOP_MARGIN = 40

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_URL = "file:///" + os.path.join(_HERE, "web", "index.html").replace("\\", "/")


class JavisOverlay:
    def __init__(self) -> None:
        # Center horizontally on the primary monitor — pywebview will accept
        # an `x=` param. We resolve screen width via win32 if available, else
        # let pywebview default-place it.
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sw = user32.GetSystemMetrics(0)
        except Exception:
            sw = 1920
        x = max(0, (sw - WIN_W) // 2)

        self._win = webview.create_window(
            "Javis",
            url=_INDEX_URL,
            width=WIN_W,
            height=WIN_H,
            x=x,
            y=TOP_MARGIN,
            frameless=True,
            on_top=True,
            transparent=True,
            resizable=False,
            easy_drag=True,
            hidden=True,
        )
        self._ready = threading.Event()
        self._destroyed = False

    # ---- thread-safe API ---------------------------------------------------

    def show(self) -> None:
        if not self._wait_ready():
            return
        try:
            self._win.show()
        except Exception as exc:
            print(f"[ui] show failed: {exc}")

    def hide(self) -> None:
        if not self._wait_ready():
            return
        try:
            self._win.hide()
        except Exception as exc:
            print(f"[ui] hide failed: {exc}")

    def set_state(self, state: str) -> None:
        if not self._wait_ready():
            return
        try:
            self._win.evaluate_js(f"window.setState({json.dumps(state)})")
        except Exception:
            pass  # window may have closed; ignore

    def feed_amplitude(self, level: float) -> None:
        if not self._wait_ready(timeout=0.0):
            return  # silently drop early frames before window mounts
        try:
            self._win.evaluate_js(f"window.feedAmplitude({float(level):.3f})")
        except Exception:
            pass

    def set_partial_transcript(self, text: str) -> None:
        if not self._wait_ready(timeout=0.0):
            return
        try:
            self._win.evaluate_js(f"window.setPartial({json.dumps(text)})")
        except Exception:
            pass

    def quit(self) -> None:
        self._destroyed = True
        try:
            self._win.destroy()
        except Exception:
            pass

    def mainloop(self) -> None:
        """Block main thread running the pywebview event loop."""
        def _on_loaded() -> None:
            # Give JS a moment to register window.setState/feedAmplitude
            time.sleep(0.05)
            self._ready.set()
        self._win.events.loaded += _on_loaded
        webview.start(gui="edgechromium", debug=False)
        self._destroyed = True

    # ---- internals --------------------------------------------------------

    def _wait_ready(self, timeout: float = 5.0) -> bool:
        if self._destroyed:
            return False
        if self._ready.is_set():
            return True
        return self._ready.wait(timeout)


# ---- standalone demo --------------------------------------------------------

if __name__ == "__main__":
    """Open the window and cycle through states / fake amplitudes."""
    import random

    ui = JavisOverlay()

    def _driver() -> None:
        ui.show()
        seq = [
            ("listening", 6.0),
            ("thinking", 3.0),
            ("speaking", 4.0),
            ("listening", 4.0),
            ("idle", 2.0),
        ]
        while True:
            for state, dur in seq:
                ui.set_state(state)
                t_end = time.monotonic() + dur
                while time.monotonic() < t_end:
                    if state == "listening":
                        ui.feed_amplitude(0.2 + 0.7 * random.random() * random.random())
                    time.sleep(0.04)

    threading.Thread(target=_driver, daemon=True).start()
    ui.mainloop()
