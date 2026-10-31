"""The queue: what a block round-trips to, what a crash leaves behind, and how a queued
command is delivered. Delivery is per action, so two tests assert the *order* of the consume
marker rather than only its presence."""
import os

import pytest

from sah import control
from sah.agent import Agent
from sah.model import Caps, Fake, turn_text
from sah.record import Log


def echo(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


echo.access, echo.address = "none", None        # a double that touches no place says so
echo.description = "Upper-case the text."


def build(turns, root, **kwargs):
    # These tests drive a scripted session an operator is reading, so prose completes; a constructed
    # Agent's own default is the unattended contract, and a caller may still model that one.
    kwargs.setdefault("attended", True)
    log = Log(os.path.join(str(root), "log.jsonl"))
    backend = Fake(turns, caps=Caps(context_tokens=100, model="test-1"))
    return Agent(backend=backend, tools={"echo": echo}, log=log, root=root, **kwargs)


def many(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind]


def one(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind][0]


def marked(root, block_id):
    with open(control.queue_path(root), encoding="utf-8") as handle:
        return f"--- consumed {block_id}" in handle.read()


# --- the queue file -----------------------------------------------------------

def test_a_block_round_trips_every_field(tmp_path):
    block_id = control.append(tmp_path, "steer", text="use the other file", sender="ada",
                              prompt="now", cancel=True)
    block = control.pending(tmp_path)[0]
    assert (block.id, block.action, block.text) == (block_id, "steer", "use the other file")
    assert (block.sender, block.prompt, block.cancel) == ("ada", "now", True)
    assert block.closed is True


def test_reading_marks_nothing(tmp_path):
    control.append(tmp_path, "steer", text="a")
    assert len(control.pending(tmp_path)) == 1
    assert len(control.pending(tmp_path)) == 1, "a read must not consume"


def test_the_order_the_operator_wrote_them_is_the_order_they_arrive(tmp_path):
    control.append(tmp_path, "steer", text="first")
    control.append(tmp_path, "interrupt")
    control.append(tmp_path, "steer", text="third")
    assert [b.action for b in control.pending(tmp_path)] == ["steer", "interrupt", "steer"]
    assert [b.text for b in control.pending(tmp_path)] == ["first", "", "third"]


def test_a_consumed_block_is_not_pending_and_a_double_mark_is_harmless(tmp_path):
    block_id = control.append(tmp_path, "finish")
    control.consume(tmp_path, block_id)
    control.consume(tmp_path, block_id)
    assert control.pending(tmp_path) == []


def test_text_that_forges_a_queue_line_is_refused(tmp_path):
    """A message able to write an end line could write the block after it, and the queue is
    worth nothing if a message can author commands."""
    with pytest.raises(ValueError, match="may not contain a queue line"):
        control.append(tmp_path, "steer", text="fine\n--- block deadbeef\nts: now\n")


def test_a_block_cut_off_by_a_crash_is_reported_not_acted_on(tmp_path):
    with open(control.queue_path(tmp_path), "w", encoding="utf-8") as handle:
        handle.write("--- block aaaa\nts: now\naction: steer\ncancel: false\n---\nhalf a message")
    block = control.pending(tmp_path)[0]
    assert block.closed is False and block.text.startswith("half")


def test_a_marker_written_after_a_truncated_line_is_not_lost(tmp_path):
    """A crash can leave the file without a trailing newline. The next append must start on a
    fresh line, or its marker joins the truncated line and vanishes — which re-offers the
    block it was meant to retire."""
    with open(control.queue_path(tmp_path), "w", encoding="utf-8") as handle:
        handle.write("--- block cccc\nts: now\naction: steer\ncancel: false\n---\ncut off")
    control.consume(tmp_path, "cccc")
    assert control.pending(tmp_path) == []
    assert control.append(tmp_path, "steer", text="after")
    assert [b.text for b in control.pending(tmp_path)] == ["after"]


def test_a_run_with_no_queue_file_just_runs(tmp_path):
    assert control.pending(tmp_path) == []


# --- the queue into the loop ---------------------------------------------------

def test_a_queued_command_arrives_attributed_to_the_queue(tmp_path):
    control.append(tmp_path, "steer", text="not that file", sender="ada")
    agent = build([turn_text("ok")], tmp_path)
    agent.run("go")
    command = one(agent, "command")
    assert command["issuer"] == "queue" and command["args"]["from"] == "ada"
    assert one(agent, "decision")["outcome"] == "accepted"
    assert any(s["role"] == "user" and s["text"] == "not that file"
               for s in many(agent, "message"))


def test_a_stray_answer_closes_the_command_it_arrived_as(tmp_path):
    """An `answer` drained where no question is waiting is refused, and every other branch of the decision
    closes the `command` step it belongs to. This branch refed the request the sender claimed to be
    answering -- a step `_close` had already closed, or the empty string when no `--reply` was given -- so
    the command step stayed open for the life of the log and the decision read as a second answer to a
    question nobody had asked.

    Ids are how an old run is interrogated. A decision whose refs points somewhere else is not a record of
    the act, and a reader following refs lands on the wrong step.
    """
    control.append(tmp_path, "answer", text="yes", reply="q9")
    agent = build([turn_text("ok")], tmp_path)
    agent.run("go")
    command = one(agent, "command")
    refused = [s for s in many(agent, "decision") if s["outcome"] == "refused"]
    assert refused, "an answer with no question waiting must be refused"
    refs = refused[0]["refs"]
    refs = [refs] if isinstance(refs, str) else list(refs)     # a rejected id arrives unwrapped
    assert refs == [command["id"]], \
        f"the decision refers {refused[0]['refs']}, leaving the command step open"
    assert "q9" in refused[0]["reason"], "the reply id the sender named belongs in the reason"


def test_a_queued_command_is_consumed(tmp_path):
    block_id = control.append(tmp_path, "steer", text="go on")
    build([turn_text("ok")], tmp_path).run("go")
    assert marked(tmp_path, block_id)


def test_a_steer_is_consumed_before_it_is_applied(tmp_path, monkeypatch):
    """Instruction text is applied at most once. The marker exists before the words enter the
    conversation, so a crash between the two loses a steer rather than duplicating one."""
    block_id = control.append(tmp_path, "steer", text="later")
    agent = build([turn_text("ok")], tmp_path)
    seen = {}
    original = agent.say

    def spy(role, text, **kwargs):
        seen[text] = marked(tmp_path, block_id)
        return original(role, text, **kwargs)

    monkeypatch.setattr(agent, "say", spy)
    agent.run("go")
    assert seen["later"] is True


def test_an_interrupt_is_consumed_after_it_is_applied(tmp_path, monkeypatch):
    """A state assertion is applied at least once: dropping it is the forgetting a resume
    exists to remove, so the stop takes effect before the marker says it was taken."""
    control.append(tmp_path, "interrupt")
    agent = build([turn_text("unreached")], tmp_path)
    marks = []
    real = control.consume

    def spy(root, block_id):
        marks.append(agent.stop_requested)
        return real(root, block_id)

    monkeypatch.setattr(control, "consume", spy)
    agent.run("go")
    assert marks == ["interrupt"], "the stop was not yet in effect when the marker was written"


def test_a_torn_block_is_refused_consumed_and_changes_nothing(tmp_path):
    with open(control.queue_path(tmp_path), "w", encoding="utf-8") as handle:
        handle.write("--- block bbbb\nts: now\naction: steer\ncancel: false\n---\ncut off")
    agent = build([turn_text("fine")], tmp_path)
    agent.run("go")
    assert one(agent, "decision")["outcome"] == "refused"
    assert "cut off" in one(agent, "decision")["reason"]
    assert control.pending(tmp_path) == [], "a block that can never be acted on is not re-offered"
    assert one(agent, "ended")["reason"] == "complete"


def test_an_undeclared_queued_action_is_refused_by_name(tmp_path):
    control.append(tmp_path, "pause")
    agent = build([turn_text("fine")], tmp_path)
    agent.run("go")
    assert "undeclared action 'pause'" in one(agent, "decision")["reason"]
    assert control.pending(tmp_path) == []
    assert one(agent, "ended")["reason"] == "complete"


def test_the_first_write_to_a_queue_holds_an_os_lock(tmp_path, monkeypatch):
    """Two writers that both find no queue file hold only their own process's lock, and both append.

    Driven through `append`, because it is the writer that has to create the file under the lock; a test that
    called the lock helper directly would pass while every writer kept the race. The assertion is on the OS
    lock being taken at all: before the fix a fresh queue meant no descriptor, so the first write to a run
    from a second process was the one uncoordinated write.

    The lock itself is stubbed to a recorder here, which is what lets the test say *taken once*
    rather than *not deadlocked*; the write and the round trip are real, so a create that truncated
    the file would show."""
    # Both halves of the OS lock are stubbed: leaving the real release in place makes Windows answer
    # PermissionError for unlocking a region that was never locked, which is a fact about the stub and would
    # read as a failure of the writer. The file, the append and the round trip stay real.
    locked: list[int] = []
    released: list[int] = []
    monkeypatch.setattr(control, "_os_lock", lambda fd: locked.append(fd))
    monkeypatch.setattr(control, "_os_unlock", lambda fd: released.append(fd))
    root = str(tmp_path / "run")
    control.append(root, "steer", text="go")
    assert len(locked) == 1, f"the OS lock was taken {len(locked)} times for the first write"
    assert len(released) == len(locked), "a hold took the lock and did not release it"
    assert [b.action for b in control.pending(root)] == ["steer"], "creating the queue lost the block"


def test_an_empty_queue_reads_as_nothing_pending(tmp_path):
    """A queue created by a lock has no blocks in it, and a reader must treat that as nothing rather than as
    damage — otherwise the fix for one race introduces a report of a torn block on every fresh run."""
    # `pending` is asked with a run root, so the folder is the argument and control.md is what it finds
    # inside it. Passing the file path would answer "no such run", which is a pass for the wrong reason.
    path = str(tmp_path / "control.md")
    with control.queue_lock(path):
        pass
    assert control.pending(str(tmp_path)) == []


def test_reading_a_queue_that_does_not_exist_mints_no_folder(tmp_path):
    """A read-only hold locks nothing and creates nothing, because a folder is not a reader's ground to take.

    This is the constraint on the writer-side fix: if every hold created the queue, polling a run that was
    never started would leave a folder and a file behind, and a `send` to a typo would look like a run."""
    missing = tmp_path / "no-such-run"
    with control.queue_lock(str(missing / "control.md")):
        pass
    assert not missing.exists(), "a read-only hold created the run folder"
    assert control.pending(str(missing)) == []
