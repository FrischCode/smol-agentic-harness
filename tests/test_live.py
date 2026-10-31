"""sah.live — the trace a run writes when it is asked to.

The file is a trace of arrival, not a second record: these tests hold the line between the two —
the fragments land as they arrive, one line each, and the record's closed vocabulary stays
untouched by them.
"""

import json

import pytest

from sah import live, model, record


STAMP = "2026-11-04T01:00:00+00:00"  # the moment the clock is held to, so a line is a line


def test_a_prose_fragment_lands_as_a_line_without_a_call(tmp_path):
    """A fragment with no `call_id` is the model's prose: the line carries the text, the moment it
    arrived, and nothing else — the record's `message` step settles the turn the fragments form."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta("hello "))
    lines = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one fragment, one line"
    step = json.loads(lines[0])
    assert step["kind"] == "delta"
    assert step["text"] == "hello "
    assert "call" not in step, "prose owes no call: the routing the display uses is the file's"
    assert "ts" in step, "the moment it arrived is what makes the file a trace, not a feed"


def test_a_tool_fragment_wears_the_call_it_streams_from(tmp_path):
    """A fragment stamped with a call id is that call's live output: the line names the call, so a
    reader routes the fragment to the tool row it belongs to the way the display does."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta("line one\n", call_id="call-1"))
    step = json.loads((tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert step["call"] == "call-1"
    assert step["text"] == "line one\n"


def test_a_reasoning_fragment_lands_as_the_line_its_kind_owes(tmp_path):
    """The bus also carries the reasoning the turn thinks: it stands as the line its kind is owed
    — the kind that says the fragment is the turn's own thought, the moment it arrived, and the
    word it carried — and no call beside it, because the reasoning a turn thinks is not the output
    of a call, and a line that carries a call is a line the output of the call it names stands in.
    The record's `message` step settles the thought the fragments form, the way it settles the
    prose."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.ReasoningDelta("the field names the wire it came in on", field="reasoning"))
    lines = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one fragment, one line"
    step = json.loads(lines[0])
    assert step["kind"] == "reasoning", "the line says the fragment is the turn's own thought"
    assert step["text"] == "the field names the wire it came in on"
    assert "call" not in step, "the thought owes no call: it is not the output of a call"
    assert "ts" in step, "the moment it arrived is what makes the file a trace, not a feed"


def test_the_whole_steps_pass_over(tmp_path):
    """The bus also carries the turn's whole steps: the file keeps the fragments and passes the
    rest, because the record's steps settle every fact a fragment carries."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.Usage(prompt_tokens=1, completion_tokens=1))
    assert not (tmp_path / "live.jsonl").exists(), "no fragment, no file"


def test_a_run_that_thinks_but_speaks_nothing_leaves_the_file_it_owes(tmp_path):
    """A reasoning fragment is a fragment: it creates the file the way a prose fragment does, so a
    run that thinks and speaks nothing leaves the file with the thought — and a run that thinks
    nothing and speaks nothing leaves the file the writer always left, which is no file."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.ReasoningDelta("a thought, and no word"))
    lines = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["reasoning"]
    still = live.Live(str(tmp_path / "no_thought.jsonl"))
    still.on_event(model.Usage(prompt_tokens=1, completion_tokens=1))
    still.on_event(model.Stopped("stop"))
    assert not (tmp_path / "no_thought.jsonl").exists(), "no fragment, no file"


def test_an_empty_fragment_has_nothing_to_arrive(tmp_path):
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta(""))
    assert not (tmp_path / "live.jsonl").exists()


def test_an_empty_reasoning_fragment_writes_nothing(tmp_path):
    """The writer's own discipline stands the reasoning the way it stands the prose: a fragment
    that is empty has nothing to arrive."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.ReasoningDelta(""))
    assert not (tmp_path / "live.jsonl").exists()


def test_each_fragment_lands_as_its_own_whole_line(tmp_path):
    """The handle ends with the line: each fragment is its own open, write, and stand-down, so the
    file holds whole JSON lines after every fragment — a reader that tails sees arrivals, never
    the half-line a held handle could leave."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta("one "))
    assert json.loads((tmp_path / "live.jsonl").read_text(encoding="utf-8").strip())["text"] == \
        "one "
    trace.on_event(model.TextDelta("two"))
    lines = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["one ", "two"]


def test_the_prose_the_writer_owed_stands_byte_for_byte_the_way_it_stood(tmp_path, monkeypatch):
    """A run that thinks nothing leaves the file the writer always left, line for line: the prose
    the writer owed stands byte for byte the way it stood — the clock held to one moment, the line
    is the writer's own bytes, and a reader that reads the prose the writer owed reads the same.
    The reasoning the writer now owes stands the same way, as its own line's bytes."""
    monkeypatch.setattr(record, "now", lambda: STAMP)
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta("caf\u00e9"))
    trace.on_event(model.ReasoningDelta("a thought beside it"))
    # `splitlines`, the way this file's other readers read: the platform's text mode owns the line's
    # ending, and the pin is on the line's bytes — the writer's own serialization, key order and all.
    lines = (tmp_path / "live.jsonl").read_bytes().decode("utf-8").splitlines()
    assert lines == [
        json.dumps({"kind": "delta", "text": "caf\u00e9", "ts": STAMP}, ensure_ascii=False,
                   sort_keys=True),
        json.dumps({"kind": "reasoning", "text": "a thought beside it", "ts": STAMP},
                   ensure_ascii=False, sort_keys=True),
    ], "the prose the writer owed stands byte for byte the way it stood, and the thought beside it"


def test_a_disk_that_refuses_raises_and_the_bus_is_the_guard(tmp_path):
    """The writer raises when the disk refuses, and does not wrap it away: the bus isolates a
    subscriber that raises, so the run the trace is tracing never stops, and a disk the run's own
    log needs is not hidden in the trace's silence."""
    trace = live.Live(str(tmp_path / "no" / "such" / "dir" / "live.jsonl"))
    with pytest.raises(OSError):
        trace.on_event(model.TextDelta("x"))


def test_a_line_is_one_json_line_the_way_the_record_writes_its_lines(tmp_path):
    """Same serialization discipline as the record's lines — keys sorted, non-ASCII kept literal —
    so a reader of one file reads the other the same way."""
    trace = live.Live(str(tmp_path / "live.jsonl"))
    trace.on_event(model.TextDelta("caf\u00e9", call_id="call-1"))
    line = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert "caf\u00e9" in line, "non-ASCII stays literal, not escaped"
    assert line.index('"call"') < line.index('"kind"') < line.index('"text"') < line.index('"ts"'), \
        "the keys stand sorted, the record's own order"
    assert json.loads(line)["text"] == "caf\u00e9"
