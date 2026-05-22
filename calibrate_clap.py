"""Print live peak amplitude from the mic so you can pick CLAP_THRESHOLD.

Watch the numbers while sitting quietly (baseline noise) and while clapping.
A good threshold sits well above ambient noise but at-or-below the lowest
clap peak you reliably produce.

Usage:
    python calibrate_clap.py     # runs ~30s, Ctrl+C to stop early
"""
from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
BLOCK = 512                 # ~32ms
PRINT_FLOOR = 0.02          # only print blocks above this so output isn't spammy
DURATION_S = 30


def main() -> None:
    print(f"Listening for {DURATION_S}s — clap several times. Ctrl+C to stop.\n")
    print("Output format:  HH:MM:SS.fff  peak=<value>  [bar]")
    print("Suggested CLAP_THRESHOLD: a bit below your typical clap peak.\n")
    peaks: list[float] = []

    def cb(indata, frames, t, status):
        peak = float(np.max(np.abs(indata)))
        if peak < PRINT_FLOOR:
            return
        peaks.append(peak)
        bar = "#" * min(int(peak * 50), 50)
        ts = time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"
        print(f"  {ts}  peak={peak:.3f}  {bar}")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK,
            callback=cb,
        ):
            time.sleep(DURATION_S)
    except KeyboardInterrupt:
        pass

    if not peaks:
        print("\nNothing detected above floor. Mic may be muted or wrong device selected.")
        return

    arr = np.array(peaks)
    print("\n--- summary ---")
    print(f"  samples above floor: {len(arr)}")
    print(f"  min peak:    {arr.min():.3f}")
    print(f"  median peak: {np.median(arr):.3f}")
    print(f"  90th pct:    {np.percentile(arr, 90):.3f}")
    print(f"  max peak:    {arr.max():.3f}")
    print()
    print("  Pick CLAP_THRESHOLD ~70% of the lowest clap you want to count.")
    print("  e.g. if your softest clap is around 0.15, set CLAP_THRESHOLD = 0.10.")


if __name__ == "__main__":
    main()
