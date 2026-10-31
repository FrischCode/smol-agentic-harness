"""extensions.speak — the demo third-party extension: a `speak` tool and a step watcher.

This file exists so the extension door has an occupant to stand on, and it travels the whole path a
contributor's file would. The operator names it on the command line (sah run --extension
extensions/speak.py) or persists that act in the machine's .env (SAH_EXTENSION; the flag
wins), the core imports it and reads its EXTENSION, and its tool joins the run's table through the
same boot gate (sah.ext.assemble) the built-ins pass: a tool that never declared how it touches the
world is refused here, at boot, before a model is ever offered it.

What it contributes:

- a `speak` tool. It POSTs the words to the /v1/audio/speech endpoint of an OpenAI-compatible TTS
  server, writes the audio to a wav file, plays it through the player the operator named, and
  returns the file's path and what happened to the audio. The endpoint is the operator's act,
  fixed at boot from the environment (the table below): the model chooses the words, never the
  place they go, and never the machine's audio — the player and the file are the tool's own
  confines, so the declaration is `self`, the class the self-confined tools sit in.
- a step watcher. It sees each step as it is written (the on contract: notification only) and, unless
  the announcing switch is off, announces the moment the model asks to speak — on the console's
  transient line, the line the record does not hold, because the fact is already in the record (the
  call step carries the words). It cannot change what happened, which is the whole of what a watcher
  may do.

Configuration, from the environment (a .env file feeds these the way it feeds SAH_HOST; a value
the operator exported or the process environment holds wins over the file):

    SAH_SPEAK_HOST     the TTS endpoint's host (default 127.0.0.1)
    SAH_SPEAK_PORT     its port (default 8000)
    SAH_SPEAK_MODEL    the model name the endpoint wants in the request (default tts)
    SAH_SPEAK_VOICE    the voice to ask for (default: `default`)
    SAH_SPEAK_OUT      where the wav files land (default: the run's working directory — for a
                       spawned child that is its own run directory, which it owns)
    SAH_SPEAK_TIMEOUT  seconds to wait for the endpoint (default 300: synthesis runs about
                       real time, and a long utterance must not be read as a hang); it is the
                       playback's ceiling too
    SAH_SPEAK_PLAY     1/0: play the audio the tool writes (default 1 — a tool named speak that
                       only wrote files would be a punchline half-delivered). Silence it for a
                       box where runs would otherwise talk over each other.
    SAH_SPEAK_ANNOUNCE 1/0: announce the ask on the console's transient line the moment the model
                       calls `speak` (default 1 — the watcher is the door's occupant, and a door
                       shown only in a test is a door a reader has to trust on faith). Silence it for
                       a box whose run flow already carries the ask in its own tool box, where the
                       announcement is a second voice for one fact.
    SAH_SPEAK_PLAYER   the player binary on PATH (default ffplay)

A file the operator never named never loads (the core does no discovery), a child never inherits
the parent's extensions (the spawner's argv carries the flag no further), and a resume does not
restore them (grants come from the operator, and this is one): the file is named again to load it
again.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
import wave
from typing import Annotated

from sah.contract import Param


def _post(url: str, payload: dict, timeout: float) -> bytes:
    """The one socket this file opens: the TTS request.

    A seam on purpose. The hermetic battery stubs it and proves the tool's contract against
    recorded bytes; the wire to a real endpoint is what the tree's opt-in live checks prove
    instead (the suite is kept out of the shipped tree on purpose, and a reader of a fresh clone
    finds the promise stated, not the path dangled). A function rather than an inline call so the
    seam is one name, the way a clock's `_now` is the one place time is held steady."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.read()


def _play(path: str, player: str, timeout: float) -> int:
    """The one process this file starts: the player, on the file the tool just wrote.

    Synchronous on purpose, the way the write is a fact the answer can state: the answer says
    `played` only when the operator has actually had it, and a player the run detaches and
    outlives is an orphaned announcement — the run's own orphan rules exist precisely because
    detached processes outlive the facts that should account for them. Returns the player's exit
    code, so a player that dies mid-utterance reads as a fact (and the file stays to open), not
    as a crash. Raises the caller's own exceptions (no player on PATH, the ceiling running out),
    and the tool turns those into words too. The flags say what a headless run needs: no window,
    no video, quiet logs, and the process ends when the audio does."""
    proc = subprocess.run([player, "-nodisp", "-autoexit", "-loglevel", "error", path],
                          timeout=timeout, check=False)
    return proc.returncode


def _wav_detail(path: str) -> str:
    """How long the audio is, in seconds, read from the wav header the endpoint wrote.

    Or an empty string when the bytes are not a wav this can parse: a duration guessed for a file
    that is not a wav is a lie the operator will hear as missing time, and an honest answer is
    shorter than a false one."""
    try:
        with wave.open(path, "rb") as handle:
            return f", {handle.getnframes() / handle.getframerate():.1f} s"
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        return ""


class Speak:
    """The speak capability: one tool bound to the endpoint the operator named at boot.

    The endpoint and the output directory travel here as construction facts, the way the built-in
    extensions carry theirs (a web.Net, a spawn.Spawner): the tool is a plain function, and
    everything it needs is a closure over what the operator already chose. A tool that needed the
    log, the policy or the loop would be smuggling a path around the veto point, and this one
    needs none of them."""

    def __init__(self, host: str, port: int, model: str, voice: str,
                 out_dir: str, timeout: float = 300.0, play: bool = True,
                 player: str = "ffplay", announce: bool = True):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1/audio/speech"
        self.model = model
        self.voice = voice
        self.out_dir = out_dir
        self.timeout = timeout
        self.play = play
        self.player = player
        self.announce = announce
        self._counter = 0
        self.speak = self._make_tool()
        self.tools = (self.speak,)

    def _free_path(self) -> str:
        """The next unused speak-NNNN.wav in the out directory.

        A run that speaks twice must not overwrite its first utterance, and a resume that speaks
        into a folder a dead run left files in must not do the same: the name is chosen against
        what is on disk, not against what this process has counted."""
        n = self._counter + 1
        while os.path.exists(os.path.join(self.out_dir, f"speak-{n:04d}.wav")):
            n += 1
        self._counter = n
        return os.path.join(self.out_dir, f"speak-{n:04d}.wav")

    def _make_tool(self):
        host, port, model, voice = self.host, self.port, self.model, self.voice
        url, out_dir, timeout = self.url, self.out_dir, self.timeout
        play, player = self.play, self.player

        def speak(text: Annotated[str, Param(description="The words to speak, in the language you "
                                                         "want them spoken in.")]) -> str:
            """The words to the endpoint, the audio to a file, the file to the speakers, the path back.

            A tool returns text and the log keeps a place, not the bytes (the same split the
            core's Media makes for pictures): a megabyte of base64 in the log is a cost every later
            reader pays for a fact — where the file is — that costs nothing to record. The answer
            says what it did, where the file is, and whether the operator heard it, because a
            model that believes an announcement it never got to play would plan around a voice
            that is not there."""
            words = text.strip()
            if not words:
                return "speak: nothing to say (the text is empty)"
            payload = {"model": model, "input": words, "voice": voice, "response_format": "wav"}
            try:
                blob = _post(url, payload, timeout)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:200].decode("utf-8", "replace")
                return f"speak: the endpoint at {host}:{port} refused (http {exc.code}): {detail}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return f"speak: the endpoint at {host}:{port} is not answering ({exc})"
            path = self._free_path()
            try:
                with open(path, "wb") as handle:
                    handle.write(blob)
            except OSError as exc:
                return f"speak: the endpoint answered but the file could not be written ({exc})"
            if not play:
                return (f"speak: wrote {os.path.basename(path)} in {out_dir} "
                        f"({len(blob) // 1024} KB{_wav_detail(path)}) — not played (playback "
                        "disabled in this run's config); the operator can open the file.")
            try:
                code = _play(path, player, timeout)
            except FileNotFoundError:
                return (f"speak: wrote {os.path.basename(path)} in {out_dir} "
                        f"({len(blob) // 1024} KB{_wav_detail(path)}) — playback unavailable "
                        f"(no {player!r} on PATH); the operator can open the file.")
            except (subprocess.TimeoutExpired, OSError) as exc:
                return (f"speak: wrote {os.path.basename(path)} in {out_dir} "
                        f"({len(blob) // 1024} KB{_wav_detail(path)}) — playback unavailable "
                        f"({exc.__class__.__name__}: {exc}); the operator can open the file.")
            if code:
                return (f"speak: wrote {os.path.basename(path)} in {out_dir} "
                        f"({len(blob) // 1024} KB{_wav_detail(path)}) — playback failed (the "
                        f"player ended with {code}); the file is there to open.")
            return (f"speak: wrote {os.path.basename(path)} in {out_dir} "
                    f"({len(blob) // 1024} KB{_wav_detail(path)}); played.")

        speak.description = (
            "Speak text aloud. Sends the words to the TTS endpoint the operator named at boot "
            f"(http://{host}:{port}, voice {voice!r}), writes the returned audio as a wav file, "
            "and plays it through the player the operator configured; the answer is the file's "
            "path and what happened to the audio (played, or why not). Reach for it when the "
            "operator should hear something — an announcement, a result — not for your own "
            "reading: a run cannot hear the file it writes.")

        # `self`, beside the web tools' own guard: the tool confines itself. The endpoint is
        # fixed at boot by the operator's act (the environment, the sibling of SAH_HOST), so the
        # model chooses the words, never the destination; the file lands in the directory the
        # operator named (default: the run's working directory), and the player is a binary the
        # operator named, on the file the tool itself just wrote there — the model's only input
        # is still the words, which travel to the endpoint, so nothing the model can direct has
        # grown, and nothing new enters the signature for the policy to judge. There is no
        # address to declare.
        speak.access, speak.address = "self", None
        return speak

    def on(self, step: dict) -> None:
        """Announce the moment the model asks to speak, on the console's transient line.

        The way the core's Questions watcher announces a live question: the operator sees the act
        where it happens, and the run that will spend a few seconds of synthesis is named before
        it starts, not only after the file lands. The line is the console's to render and the
        record's not to hold — the fact it points at (the call step, which carries the words) is
        already in the record, so the line is a word for the operator who is watching now, not a
        fact a later reader needs. It names its speaker in its own words, because the line's face
        has no subject the core could fill in. Notification only — the watcher sees the step as it is written,
        and it cannot change it, refuse it, or answer it. Switched off
        (`SAH_SPEAK_ANNOUNCE=0`), the watcher stays subscribed and says nothing: the silence is the
        operator's act, not the door's absence."""
        if not self.announce:
            return
        if step.get("kind") != "call" or step.get("name") != "speak":
            return
        words = str((step.get("args") or {}).get("text", ""))
        if len(words) > 70:
            words = words[:67] + "…"
        self.status.say(f"speak: the run asks to say — {words}")


def from_env() -> Speak:
    """The extension built from the environment, the way a run's operator configures it.

    Read at construction (import time, per run), so a value the operator exported for this run
    wins over whatever the file below it carries, and a run started with no opinion at all runs
    on the defaults spelled out here and in the module docstring."""
    play = os.environ.get("SAH_SPEAK_PLAY", "1").strip().lower()
    announce = os.environ.get("SAH_SPEAK_ANNOUNCE", "1").strip().lower()
    return Speak(host=os.environ.get("SAH_SPEAK_HOST", "127.0.0.1"),
                 port=int(os.environ.get("SAH_SPEAK_PORT", "8000")),
                 model=os.environ.get("SAH_SPEAK_MODEL", "tts"),
                 voice=os.environ.get("SAH_SPEAK_VOICE", "default"),
                 out_dir=os.environ.get("SAH_SPEAK_OUT", os.getcwd()),
                 timeout=float(os.environ.get("SAH_SPEAK_TIMEOUT", "300")),
                 play=play not in ("", "0", "false", "no", "off"),
                 player=os.environ.get("SAH_SPEAK_PLAYER", "ffplay"),
                 announce=announce not in ("", "0", "false", "no", "off"))


# The one name the door reads. An instance, not a class: the core refuses a class (an unbound
# on() takes the first step for its self), and an instance is what the operator means — this
# configured capability, ready to join a run's table through the gate.
EXTENSION = from_env()
