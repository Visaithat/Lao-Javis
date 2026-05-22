"""Intent-detection fast path — skip Claude for common voice commands.

The voice agent's slowest path is the Claude subprocess (~10-20s per turn).
For frequent commands like "play X on YouTube" or "open Notepad" we don't
need an LLM at all — a regex + os.startfile / subprocess.Popen does it in
1-2 seconds.

`try_match(text)` returns a `(spoken_reply, action_callable)` tuple if the
input matches a known intent, else `None`. The caller runs the action and
speaks the reply via TTS. If `try_match` returns None, the caller falls
through to the full Claude path.
"""
from __future__ import annotations

import os
import re
import subprocess
import urllib.parse
from typing import Callable, Optional, Tuple

Action = Callable[[], None]
Match = Tuple[str, Action]


# --- normalization -----------------------------------------------------------

# Trailing polite/filler particles to strip before pattern matching. Order
# matters — longer suffixes first so they consume the right amount.
_TRAILING_FILLERS = [
    "ให้หน่อยครับ", "ให้หน่อยค่ะ", "ให้หน่อยที", "ให้หน่อย",
    "หน่อยครับ", "หน่อยค่ะ", "หน่อยที", "หน่อย",
    "ให้ผมด้วย", "ให้ผมที", "ให้ผม",
    "ครับผม", "ครับ", "ค่ะ", "นะคะ", "นะครับ", "นะ", "ที", "ด้วย",
    "ให้ฉัน", "ให้ฉันหน่อย", "ฉันที",
    "please", "for me", "pls",
]

# Leading polite/request words to strip before pattern matching. Important
# because STT often inserts "ช่วย" before the action verb.
_LEADING_FILLERS = [
    "ช่วยหน่อย", "ช่วยที", "ช่วย",
    "ขอให้", "ขอ",
    "ผมอยาก", "ฉันอยาก", "อยาก",
    "could you please", "could you", "can you please", "can you",
    "please", "pls", "hey",
]

_PUNCT_RE = re.compile(r"[.!?。.!?,;:]+$")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = text.strip()
    t = _PUNCT_RE.sub("", t)
    t = _WS_RE.sub(" ", t)
    # Strip leading polite/request fillers ("ช่วย เปิด ..." → "เปิด ...")
    changed = True
    while changed:
        changed = False
        lower = t.lower()
        for f in _LEADING_FILLERS:
            if lower.startswith(f.lower()):
                # consume the prefix and any following whitespace
                t = t[len(f):].lstrip()
                changed = True
                break
    # Strip trailing fillers repeatedly (so "ให้หน่อยครับ" works)
    changed = True
    while changed:
        changed = False
        for f in _TRAILING_FILLERS:
            if t.lower().endswith(f.lower()):
                t = t[: -len(f)].strip()
                t = _PUNCT_RE.sub("", t)
                changed = True
                break
    return t


# --- actions ----------------------------------------------------------------

def _open_url(url: str) -> Action:
    def _act() -> None:
        # os.startfile opens with Windows default browser (user's real Chrome)
        os.startfile(url)
    return _act


def _open_app(app: str) -> Action:
    def _act() -> None:
        # `start "" appname` resolves the app via App Paths / PATH and detaches
        subprocess.Popen(["cmd", "/c", "start", "", app], shell=False,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    return _act


def _youtube_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


# --- patterns ---------------------------------------------------------------

# (Music play) — extract "<artist + song>" after the verb
_MUSIC_RE = re.compile(
    r"^(?:เล่น|เปิด|ฟัง|play|listen\s+to)\s*(?:เพลง|song)?\s*(?P<q>.+)$",
    re.IGNORECASE,
)

# (Open known Windows app) — map spoken word → exe name
_APP_ALIASES: dict[str, str] = {
    # Thai/English keys → app to launch via `start`
    "notepad": "notepad",
    "โน้ตแพด": "notepad",
    "โน๊ตแพด": "notepad",
    "calc": "calc",
    "calculator": "calc",
    "เครื่องคิดเลข": "calc",
    "chrome": "chrome",
    "edge": "msedge",
    "explorer": "explorer",
    "ไฟล์": "explorer",
    "paint": "mspaint",
    "spotify": "spotify",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "terminal": "wt",
    "powershell": "powershell",
    "cmd": "cmd",
}
_OPEN_APP_RE = re.compile(
    rf"^(?:เปิด|open|launch|start)\s+(?P<app>{'|'.join(re.escape(k) for k in _APP_ALIASES.keys())})$",
    re.IGNORECASE,
)

# (Open known website) — map spoken site → URL
_SITE_ALIASES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "ยูทูบ": "https://www.youtube.com",
    "google": "https://www.google.com",
    "กูเกิล": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "จีเมล": "https://mail.google.com",
    "facebook": "https://www.facebook.com",
    "เฟซบุ๊ก": "https://www.facebook.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "spotify web": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "claude": "https://claude.ai",
    "chatgpt": "https://chatgpt.com",
    "wikipedia": "https://en.wikipedia.org",
}
_OPEN_SITE_RE = re.compile(
    rf"^(?:เปิด|open|go\s+to|ไป)\s+(?P<site>{'|'.join(re.escape(k) for k in _SITE_ALIASES.keys())})$",
    re.IGNORECASE,
)

# (Bare URL or domain)
_URL_RE = re.compile(
    r"^(?:เปิด|open|go\s+to|ไป)\s+(?P<url>(?:https?://)?[\w\-]+(?:\.[\w\-]+)+(?:/\S*)?)$",
    re.IGNORECASE,
)


# --- main entry -------------------------------------------------------------

def try_match(text: str) -> Optional[Match]:
    """Match `text` to a known intent. Returns (spoken_reply, action) or None."""
    norm = _normalize(text)
    if not norm:
        return None

    # 1. open known app
    m = _OPEN_APP_RE.match(norm)
    if m:
        key = m.group("app").lower()
        app = _APP_ALIASES[key]
        return (f"เปิด {key} ให้แล้วครับ", _open_app(app))

    # 2. open known site
    m = _OPEN_SITE_RE.match(norm)
    if m:
        key = m.group("site").lower()
        url = _SITE_ALIASES[key]
        return (f"เปิด {key} ให้แล้วครับ", _open_url(url))

    # 3. bare URL / domain
    m = _URL_RE.match(norm)
    if m:
        url = m.group("url")
        if not url.startswith("http"):
            url = "https://" + url
        return ("เปิดให้แล้วครับ", _open_url(url))

    # 4. music play (must come AFTER site/app so "เปิด youtube" isn't treated as song)
    m = _MUSIC_RE.match(norm)
    if m:
        query = m.group("q").strip()
        # If query is empty or just punctuation, skip
        if not query:
            return None
        url = _youtube_search_url(query)
        # Generic confirmation in Thai
        return (f"เปิดเพลงให้แล้วครับ", _open_url(url))

    return None


if __name__ == "__main__":
    samples = [
        "เปิดเพลง Sorry ของ Justin Bieber ให้หน่อย",
        "ช่วย เปิด เพลง Sorry ของ Justin Bieber ให้ หน่อย ครับ",  # leading "ช่วย"
        "ขอเปิด Notepad หน่อย",
        "เล่นเพลง Despacito",
        "play Bohemian Rhapsody",
        "เปิด notepad",
        "open chrome please",
        "please open spotify",
        "เปิด youtube",
        "เปิด google.com",
        "ปิดโปรแกรม",                 # should NOT match
        "ค้นหาข่าวล่าสุดวันนี้",        # should NOT match
        "อ่านข่าวจาก bbc.com ให้หน่อย", # should NOT match (verb is "อ่าน" not "เปิด")
    ]
    for s in samples:
        r = try_match(s)
        if r:
            print(f"MATCH  {s!r}\n       reply={r[0]!r}")
        else:
            print(f"miss   {s!r}")
