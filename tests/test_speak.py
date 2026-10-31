"""The demo extension (extensions/speak.py), driven through the same door a contributor's
file passes and with its one socket stubbed. The hermetic battery proves the contract the demo
travels — the declaration, the schema, the file the answer names, the watcher's announcement — and
the wire to a real TTS endpoint is what the tree's opt-in live checks prove instead (the suite is
kept out of the shipped tree on purpose, so this file names the promise, not a path)."""
import io
import pathlib
import sys
import urllib.error
import wave

import pytest

from sah import render
from sah.cli import _load_extension
from sah.contract import declaration, spec_for
from sah.ext import Extension, assemble
from sah.record import Log

DEMO = pathlib.Path(__file__).resolve().parent.parent / "extensions" / "speak.py"


def _load_demo(monkeypatch, out_dir):
    """Load the demo through the door with its out directory pointed at the test's folder, and
    return the loaded module beside the extension instance. The player seam is stubbed with a
    recorder the tests read back: the battery never starts a player, and a test that wants a
    dead or absent one overrides the stub with its own monkeypatch."""
    monkeypatch.setenv("SAH_SPEAK_OUT", str(out_dir))
    _path, (speak,) = _load_extension(str(DEMO))
    module = sys.modules[type(speak).__module__]
    module.play_calls = []
    monkeypatch.setattr(module, "_play",
                        lambda path, player, timeout: module.play_calls.append((path, player)) or 0)
    return module, speak


def _wav_bytes(seconds=1.0, rate=24000) -> bytes:
    """A real, parseable wav in memory, built the way the endpoint's answer is shaped."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _wired_watch(monkeypatch, out_dir):
    """The demo through the door with a console wired the way a run wires it: a terminal on a
    captured stream and the console's transient line handed in at assembly, so the watcher's
    announcement lands where a run's lands — the console's face, not the operator's stream."""
    module, speak = _load_demo(monkeypatch, out_dir)
    out = io.StringIO()
    terminal = render.Terminal(out, io.StringIO(), "normal", color=False)
    log = Log(out_dir / "log.jsonl")
    assemble([speak], log, status=render.Status(terminal))
    return module, speak, log, out


def test_the_demo_joins_the_table_through_the_gate(tmp_path, monkeypatch):
    """The whole promise in one test: a file named by the operator passes the same boot gate the
    built-ins pass — the schema is derived from the signature, the declaration is read, and the
    tool lands in the table under its own name."""
    module, speak = _load_demo(monkeypatch, tmp_path)
    tools = assemble([speak], Log(tmp_path / "log.jsonl"))
    assert set(tools) == {"speak"}
    assert declaration(tools["speak"]) == ("self", None)
    spec = spec_for(tools["speak"])
    assert spec.name == "speak"
    assert spec.description.startswith("Speak text aloud")
    text = spec.parameters["properties"]["text"]
    assert text["type"] == "string" and text["description"], "the parameter's meaning rides in the hint"


def test_the_demo_tool_speaks_and_writes_and_plays(tmp_path, monkeypatch):
    module, speak = _load_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_post", lambda url, payload, timeout: _wav_bytes(2.0))
    answer = speak.speak("The words the test wants spoken.")
    assert answer.startswith("speak: wrote speak-0001.wav"), answer
    assert "2.0 s" in answer, "the duration is read from the header, not guessed"
    assert "; played" in answer, "a tool named speak that does not speak is a punchline half-delivered"
    path = tmp_path / "speak-0001.wav"
    assert path.is_file()
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() / handle.getframerate() == pytest.approx(2.0)
    # The player was called with the file it wrote and the operator's player name, in that order.
    assert module.play_calls == [(str(path), "ffplay")]
    # A second utterance keeps the first: the name is chosen against what is on disk.
    assert speak.speak("Again.").startswith("speak: wrote speak-0002.wav")
    assert (tmp_path / "speak-0002.wav").is_file()
    assert len(module.play_calls) == 2


def test_the_demo_tool_keeps_what_a_dead_run_left_behind(tmp_path, monkeypatch):
    module, speak = _load_demo(monkeypatch, tmp_path)
    (tmp_path / "speak-0001.wav").write_bytes(_wav_bytes(0.5))
    monkeypatch.setattr(module, "_post", lambda url, payload, timeout: _wav_bytes(0.5))
    answer = speak.speak("Into a folder that already holds a file.")
    assert answer.startswith("speak: wrote speak-0002.wav"), "a resume must not overwrite the past"


def test_the_demo_tool_reports_a_refused_endpoint(tmp_path, monkeypatch):
    """The endpoint's own words come back to the model, so a 400 reads as an answer rather than a
    crash: the run can say what the server said and stop trying."""

    class _Refusal(urllib.error.HTTPError):
        """A synthetic 400 whose read() behaves like urllib's real one: the server's body."""

        def read(self, *args):
            return b'{"error": "input is empty"}'

    def refusing(url, payload, timeout):
        raise _Refusal(url, 400, "Bad Request", {}, None)

    module, speak = _load_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_post", refusing)
    answer = speak.speak("hello")
    assert "refused (http 400)" in answer and "input is empty" in answer
    assert list(tmp_path.iterdir()) == [], "a refused request writes no file"


def test_the_demo_tool_reports_a_silent_endpoint(tmp_path, monkeypatch):
    def silent(url, payload, timeout):
        raise urllib.error.URLError("Connection refused")

    module, speak = _load_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_post", silent)
    answer = speak.speak("hello")
    assert "is not answering" in answer and "8000" in answer


def test_the_demo_tool_says_nothing_when_given_nothing(tmp_path, monkeypatch):
    module, speak = _load_demo(monkeypatch, tmp_path)
    answer = speak.speak("   ")
    assert "nothing to say" in answer


def test_the_demo_tool_reports_a_failed_playback(tmp_path, monkeypatch):
    """A player that dies mid-utterance is a fact the answer states (the file stays to open), not a
    crash: the model reads the player's exit code and stops treating the announcement as made."""
    module, speak = _load_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_post", lambda url, payload, timeout: _wav_bytes(1.0))
    monkeypatch.setattr(module, "_play", lambda path, player, timeout: 3)
    answer = speak.speak("An announcement the player drops.")
    assert "playback failed (the player ended with 3)" in answer
    assert (tmp_path / "speak-0001.wav").is_file(), "the file stands regardless of the player"


def test_the_demo_tool_reports_a_missing_player(tmp_path, monkeypatch):
    """No player on PATH is named in the answer, the file still lands, and the tool never crashes:
    a run on a box without ffplay degrades to the write-only truth it was before playback, and says
    so."""
    def absent(path, player, timeout):
        raise FileNotFoundError("[WinError 2] The system cannot find the file specified")

    module, speak = _load_demo(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_post", lambda url, payload, timeout: _wav_bytes(1.0))
    monkeypatch.setattr(module, "_play", absent)
    answer = speak.speak("A box without speakers.")
    assert "playback unavailable" in answer and "ffplay" in answer
    assert (tmp_path / "speak-0001.wav").is_file()


def test_the_demo_tool_honours_the_playback_off_switch(tmp_path, monkeypatch):
    """SAH_SPEAK_PLAY=0 is the mute for a box whose runs would talk over each other: the file is
    written, the player is never started, and the answer says the audio was not played rather than
    claiming an announcement nobody heard."""
    monkeypatch.setenv("SAH_SPEAK_PLAY", "0")
    monkeypatch.setenv("SAH_SPEAK_OUT", str(tmp_path))
    _path, (speak,) = _load_extension(str(DEMO))
    module = sys.modules[type(speak).__module__]
    module.play_calls = []
    monkeypatch.setattr(module, "_post", lambda url, payload, timeout: _wav_bytes(1.0))
    answer = speak.speak("A muted box.")
    assert "not played (playback disabled in this run's config)" in answer
    assert module.play_calls == [], "the off switch keeps the player from starting at all"
    assert [f.name for f in tmp_path.iterdir() if f.name.endswith(".wav")] == ["speak-0001.wav"], \
        "the file still lands"


def test_the_demo_watcher_announces_the_speak_ask_and_only_it(tmp_path, monkeypatch, capfd):
    """The watcher sees the step as it is written and names the act where it happens, on the
    console's transient line — the operator's stream stays silent, because the announcement is the
    console's face and not a second voice on it; every other step passes through without a word."""
    _module, _speak, log, out = _wired_watch(monkeypatch, tmp_path)
    message = log.append("message", "model", role="assistant", text="I will say this.")
    log.append("call", "model", refs=message["id"], name="speak",
               args={"text": "hello operator, from the run"})
    log.append("call", "model", refs=message["id"], name="read", args={"path": "/x"})
    text = out.getvalue()
    assert text.count("the run asks to say") == 1, "only the speak ask is announced"
    assert "… speak: the run asks to say" in text, \
        "the line self-names its speaker: the naming rides in the words, not the console's face"
    assert "hello operator, from the run" in text
    assert "read" not in text.split("asks to say")[1]
    assert capfd.readouterr().err == "", "the operator's stream is not the transient line's home"


def test_the_demo_watcher_truncates_a_long_ask(tmp_path, monkeypatch):
    """A long ask is truncated before it is said: the caption is a line, and a line that runs to
    three hundred words is not a line, it is a paragraph with a marker in front of it."""
    _module, _speak, log, out = _wired_watch(monkeypatch, tmp_path)
    message = log.append("message", "model", role="assistant", text="A long announcement.")
    log.append("call", "model", refs=message["id"], name="speak", args={"text": "x" * 300})
    line = [l for l in out.getvalue().splitlines() if "asks to say" in l][0]
    assert len(line) < 120, "the caption stays a line"


def test_the_demo_watcher_is_wired_but_quiet_when_announcing_is_off(tmp_path, monkeypatch):
    """The switch is the operator's act, not the door's absence: announcing off keeps a real watcher —
    subscribed by the gate because its `on` is not the contract's default, still seeing the step it
    was subscribed for — that simply says nothing. That is what a box whose run flow already carries
    the ask wants, and the door keeps its occupant."""
    monkeypatch.setenv("SAH_SPEAK_ANNOUNCE", "0")
    module, speak = _load_demo(monkeypatch, tmp_path)
    out = io.StringIO()
    terminal = render.Terminal(out, io.StringIO(), "normal", color=False)
    log = Log(tmp_path / "log.jsonl")
    assemble([speak], log, status=render.Status(terminal))
    assert type(speak).on is not Extension.on, "the gate subscribes it: the silence is not an absence"
    message = log.append("message", "model", role="assistant", text="I will say this.")
    log.append("call", "model", refs=message["id"], name="speak", args={"text": "hello operator"})
    assert out.getvalue() == "", "subscribed and silent"


def test_the_demo_reads_its_configuration_from_the_environment(tmp_path, monkeypatch):
    """The operator's act is the environment, read at construction: a value the run's process holds
    wins over the file's defaults, and the model-facing description names the endpoint it will
    actually hit, so the schema is not a promise about a different server."""
    monkeypatch.setenv("SAH_SPEAK_HOST", "tts.example")
    monkeypatch.setenv("SAH_SPEAK_PORT", "9100")
    monkeypatch.setenv("SAH_SPEAK_VOICE", "f03")
    monkeypatch.setenv("SAH_SPEAK_MODEL", "tts-1")
    monkeypatch.setenv("SAH_SPEAK_PLAYER", "pmedia")
    monkeypatch.setenv("SAH_SPEAK_PLAY", "off")
    _path, (speak,) = _load_extension(str(DEMO))
    assert speak.url == "http://tts.example:9100/v1/audio/speech"
    assert speak.voice == "f03" and speak.model == "tts-1"
    assert speak.player == "pmedia" and speak.play is False
    assert "tts.example:9100" in speak.speak.description and "f03" in speak.speak.description
