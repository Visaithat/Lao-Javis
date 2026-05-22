"""Conversational agent built on the `claude` CLI (Claude Code subscription).

Spawns `claude --print --output-format json` per turn via synchronous
subprocess.run — avoids a Python-3.13 + Windows bug in claude-agent-sdk's
async transport that throws `[WinError 5] Access is denied`. Conversation
memory is preserved across turns via `--resume <session_id>`, where the
session id comes back in each response JSON.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import tempfile
from typing import Callable, Optional

CLAUDE_CMD = "claude"
_HERE = os.path.dirname(os.path.abspath(__file__))
MCP_CONFIG_PATH = os.path.join(_HERE, "mcp_config.json")

# Strip markdown that sounds bad when read by TTS — bold/italic/code/headings.
_MD_PATTERNS = [
    (re.compile(r"```[\s\S]*?```"), " "),          # fenced code blocks
    (re.compile(r"`([^`]+)`"), r"\1"),             # inline code → bare text
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),       # **bold**
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),  # *italic*
    (re.compile(r"__([^_]+)__"), r"\1"),           # __bold__
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""), # headings
    (re.compile(r"\s+"), " "),                     # collapse whitespace
]


def _strip_markdown(text: str) -> str:
    for pat, repl in _MD_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()

SYSTEM_PROMPT = """You are Javis. Your name is Javis. NEVER call yourself
"Claude" or "Claude Code" — your identity is Javis, period.

You are a voice-controlled assistant running on the user's Windows PC.
Replies are read aloud, so keep them short, friendly, and conversational.
Match the user's language: Thai (ภาษาไทย), Lao (ພາສາລາວ), or English. Never
include emoji, markdown bullets, asterisks, code blocks, or URLs in your
spoken reply — those sound terrible when spoken aloud.

For Thai refer to yourself as "ผม" and end statements with "ครับ". For Lao
use "ຂ້ອຍ" and natural Lao endings. For English be natural.

LENGTH BUDGET — STRICT: each reply is at most ONE short sentence (under
~80 characters). After running a tool, just say what you did in one clause.
Do not add preambles, do not explain how, do not offer "anything else?",
do not list options. Brevity matters because every extra word adds spoken
audio latency.

Bias toward ACTION, not asking. If the user says "play X" or "open X", just
DO it and briefly say what you did. Do not ask "want me to do it?" — they
already told you. Only ask back when something is genuinely ambiguous
(e.g. an artist name with no song).

To play a YouTube video: construct the search URL
`https://www.youtube.com/results?search_query=<query>` and call PowerShell
`Start-Process "<url>"`. The first auto-play result usually starts playing.
If the user wants a specific known video, use the direct watch URL instead.

Capabilities:

For Windows desktop actions — use the PowerShell tool (NOT Bash; Bash routes
through Git Bash and `start` does not actually launch GUI apps):
- Open a Windows app: PowerShell `Start-Process notepad` or `Start-Process calc`
- Open a Windows setting: PowerShell `Start-Process ms-settings:`
- Adjust volume: PowerShell `(New-Object -ComObject WScript.Shell).SendKeys([char]175)` (up, 174 down, 173 mute)

For web/browser tasks — choose the right tool:

(A) "Open this and let it run" type (play a song/video, just navigate to a
    site for the user to look at) — USE PowerShell `Start-Process "url"`.
    This opens the user's REAL Chrome with their logins, and the page keeps
    running after this turn ends. Playwright would close the page.
    Examples: "play Sorry by Justin Bieber" → search YouTube URL and Start-Process.
              "เปิด Spotify" → Start-Process "https://open.spotify.com"
              "เปิดข่าวบีบีซี" → Start-Process the URL.

(B) "Read/click/extract from a page" type — use Playwright MCP. This opens a
    separate Chromium we can drive. Browser closes at end of this turn so
    finish what you need within one turn (navigate + snapshot + click + read).
    Examples: "find me the top headline on bbc.com"
              "go to YouTube search for X and tell me the top video title"
              "fill in this form for me"

Playwright tools you can call directly (no ToolSearch needed):
- mcp__playwright__browser_navigate({"url": "..."}) — open a URL
- mcp__playwright__browser_snapshot() — see element refs on the current page
- mcp__playwright__browser_click({"ref": "..."}) — click an element from snapshot
- mcp__playwright__browser_type({"ref": "...", "text": "..."}) — type into a field
- mcp__playwright__browser_press_key({"key": "Enter"}) — press a key
- mcp__playwright__browser_wait_for({"text": "..."}) — wait for content
- mcp__playwright__browser_navigate_back() — go back
- mcp__playwright__browser_tabs() — list/switch tabs
- mcp__playwright__browser_close() — close browser

Typical flow:
1. browser_navigate to the URL.
2. If you need to interact: browser_snapshot to see refs, then click/type.
3. Tell the user briefly what you did. Do not read URLs aloud.

Examples of when to use Playwright:
- "open youtube and play X"        → navigate to youtube search results
- "search google for X"            → navigate to google search URL
- "go to wikipedia and read about" → navigate + snapshot + summarize
- "open gmail"                     → navigate to gmail.com

Use WebSearch for quick factual lookups where you do NOT need to interact
with a page (e.g. "what's the capital of France").

Critical rules:
- NEVER claim you did something before the tool returns success. Call the tool
  FIRST, then describe the result based on what actually happened.
- If a tool returns an error, say honestly that it failed and try a different
  approach — do not pretend it worked.
- If the request is ambiguous (e.g. just an artist name with no song), ask one
  short follow-up question instead of guessing.
- After running a tool, say in one short sentence what you just did.
- Never read out the raw command you ran. The user only wants the result.
"""

ALLOWED_TOOLS = (
    "PowerShell,Bash,WebSearch,WebFetch,Read,Write,Glob,Grep,ToolSearch,"
    "mcp__playwright__browser_navigate,"
    "mcp__playwright__browser_navigate_back,"
    "mcp__playwright__browser_click,"
    "mcp__playwright__browser_type,"
    "mcp__playwright__browser_hover,"
    "mcp__playwright__browser_select_option,"
    "mcp__playwright__browser_snapshot,"
    "mcp__playwright__browser_take_screenshot,"
    "mcp__playwright__browser_press_key,"
    "mcp__playwright__browser_wait_for,"
    "mcp__playwright__browser_close,"
    "mcp__playwright__browser_tabs"
)
TIMEOUT_S = 300  # browser actions can be slow (npx cold start + chromium launch + multi-step)
MODEL = "claude-haiku-4-5-20251001"  # fast + cheap; voice-assistant doesn't need Opus


def _write_system_prompt_file() -> str:
    """Write SYSTEM_PROMPT to a temp file and schedule cleanup at exit.

    Windows' CreateProcessW rejects long --system-prompt args (≥ ~1500 chars
    with Thai/Lao/special chars) with WinError 5. Passing a file path keeps
    the command line short.
    """
    fd, path = tempfile.mkstemp(prefix="javis_sysprompt_", suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(SYSTEM_PROMPT)
    except Exception:
        os.close(fd)
        raise
    atexit.register(lambda p=path: _safe_unlink(p))
    return path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class AgentError(RuntimeError):
    """Raised when the claude CLI returns an error or invalid output."""


class Agent:
    """Sync wrapper around the `claude` CLI for multi-turn conversation."""

    def __init__(self) -> None:
        self._session_id: Optional[str] = None
        self._sys_prompt_path: str = _write_system_prompt_file()

    def send(self, user_text: str) -> str:
        cmd = self._build_cmd("json")
        cmd.extend(["-p", user_text])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"claude CLI timed out after {TIMEOUT_S}s") from exc

        if proc.returncode != 0:
            raise AgentError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )

        stdout = proc.stdout.strip()
        if not stdout:
            raise AgentError(f"claude CLI returned empty stdout. stderr: {proc.stderr[:500]}")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AgentError(f"failed to parse JSON: {exc}; stdout head: {stdout[:300]}") from exc

        if payload.get("is_error"):
            raise AgentError(f"claude reported error: {payload.get('result') or payload}")

        new_session = payload.get("session_id")
        if new_session:
            self._session_id = new_session

        return _strip_markdown(payload.get("result") or "")

    def reset(self) -> None:
        """Forget the current session — next send() starts a fresh conversation."""
        self._session_id = None

    def _build_cmd(self, output_format: str, extra_args: list[str] | None = None) -> list[str]:
        cmd = [
            CLAUDE_CMD,
            "--print",
            "--output-format", output_format,
            "--model", MODEL,
            "--permission-mode", "bypassPermissions",
            "--allowed-tools", ALLOWED_TOOLS,
            "--mcp-config", MCP_CONFIG_PATH,
            "--chrome",
            "--system-prompt-file", self._sys_prompt_path,
        ]
        if extra_args:
            cmd.extend(extra_args)
        if self._session_id is not None:
            cmd.extend(["--resume", self._session_id])
        return cmd

    def send_streaming(self, user_text: str, on_sentence: Callable[[str], None]) -> str:
        """Stream Claude's response as it comes in, firing `on_sentence` per chunk.

        Uses --output-format stream-json so we see each assistant text block as
        it's emitted (often multiple per turn when the model reasons + calls
        tools + summarizes). Each block is split into sentences and dispatched
        immediately, so TTS can start playing the first sentence while the
        model is still generating later ones.

        Returns the full concatenated assistant text.
        """
        cmd = self._build_cmd("stream-json", extra_args=["--verbose"])
        cmd.extend(["-p", user_text])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise AgentError(f"claude CLI not found: {exc}") from exc

        full_parts: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                if mtype == "assistant":
                    blocks = msg.get("message", {}).get("content", [])
                    for block in blocks:
                        if block.get("type") == "text":
                            text = block.get("text") or ""
                            if not text:
                                continue
                            full_parts.append(text)
                            for sentence in _split_sentences(_strip_markdown(text)):
                                if sentence:
                                    try:
                                        on_sentence(sentence)
                                    except Exception as exc:
                                        print(f"[agent] on_sentence raised: {exc}")
                elif mtype == "result":
                    sid = msg.get("session_id")
                    if sid:
                        self._session_id = sid
                    if msg.get("is_error"):
                        raise AgentError(f"claude reported error: {msg.get('result') or msg}")
        finally:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        if proc.returncode not in (0, None):
            err = (proc.stderr.read() if proc.stderr else "")[:500]
            raise AgentError(f"claude CLI exited {proc.returncode}: {err}")

        return _strip_markdown("".join(full_parts))


_SENTENCE_RE = re.compile(r"[^.!?。\n]+(?:[.!?。]+|\n+|$)")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks for TTS.

    Splits on .!?。 or newlines. For Thai/Lao (rare punctuation), force-chunks
    long runs at the nearest space so TTS doesn't wait for end of paragraph.
    """
    if not text:
        return []
    parts: list[str] = []
    for m in _SENTENCE_RE.finditer(text):
        chunk = m.group(0).strip()
        if not chunk:
            continue
        # Force-split very long chunks (Thai/Lao often have no punctuation)
        while len(chunk) > 120:
            cut = chunk.rfind(" ", 60, 130)
            if cut <= 0:
                cut = 100
            parts.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            parts.append(chunk)
    return parts


if __name__ == "__main__":
    agent = Agent()
    for prompt in [
        "สวัสดี ทดสอบหน่อยว่าได้ยินไหม",
        "ช่วยเปิด Notepad ให้หน่อย",
        "ขอบใจมาก เพิ่งเปิดอะไรให้เรา?",  # multi-turn memory check
    ]:
        print(f"\n> {prompt}")
        try:
            reply = agent.send(prompt)
            print(f"< {reply}")
            print(f"  [session: {agent._session_id}]")
        except AgentError as e:
            print(f"! {e}")
            break
