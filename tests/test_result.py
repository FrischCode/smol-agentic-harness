"""result: the answer a run left and how it ended — a projection that cannot disagree with the
log.

The projection selects what the record says (the last assistant message that says something,
the last `ended` step); these tests build real logs with the real writer and read them back
through the real reader, so a projection that drifts from the record fails here rather than in
a parent reading a child's answer. The CLI tests cover the contract a scripted caller relies
on: the answer on stdout, the run's verdict as the exit code."""
import io
import json

from sah import cli, result
from sah.record import Log, lineage, read as read_log


def _log(tmp_path, run_id="r1"):
    log = Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id=run_id, root=str(tmp_path))
    return log


def _read(tmp_path):
    return read_log(tmp_path / "log.jsonl", tolerate_damage=True)


def _finished_run(tmp_path, reason="complete", exit_code=0, answer="42"):
    """The common shape: a run that answered and ended."""
    log = _log(tmp_path)
    log.append("message", "user", role="user", text="what is 6*7?")
    step = log.append("message", "model", role="assistant", text=answer)
    log.append("ended", "core", reason=reason, exit_code=exit_code)
    return step["id"]


def test_the_projection_selects_the_answer_and_the_ending(tmp_path):
    """The whole contract: what the run left, where it left it, and how it ended — all read
    from the record, nothing re-derived."""
    answer_id = _finished_run(tmp_path)
    projection = result.result(_read(tmp_path)[0])
    assert projection == {"run_id": "r1", "ended": True, "reason": "complete",
                          "exit_code": 0, "text": "42", "answer_id": answer_id,
                          "tail_notices": []}


def test_the_answer_is_the_last_assistant_message_that_says_something(tmp_path):
    """A run whose last turn was a tool call leaves no words there, so the answer is the last
    thing it actually said — even though the run did not deliver. The record's reason is what
    tells the parent that; an empty answer must not be dressed up as a delivered one."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="let me compute that")
    log.append("message", "model", role="assistant", text="")     # the last turn: a call, no words
    log.append("ended", "core", reason="max_turns", exit_code=1)
    projection = result.result(_read(tmp_path)[0])
    assert projection["reason"] == "max_turns" and projection["exit_code"] == 1
    assert projection["text"] == "let me compute that"


def test_the_answer_is_the_last_word_not_the_first(tmp_path):
    """A run that speaks twice answers with the later word: the earlier one is a step along the
    way, and a parent handed it would mistake an intermediate for the delivery. The direction is
    visible only in a log that holds two non-empty answers, so the log has two."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="first I will read the file")
    log.append("message", "model", role="assistant", text="the answer is 144")
    log.append("ended", "core", reason="complete", exit_code=0)
    projection = result.result(_read(tmp_path)[0])
    assert projection["text"] == "the answer is 144"


def test_a_stale_answer_is_read_beside_the_words_that_happened_after_it(tmp_path):
    """The shape a budgeted run leaves: words early, a last turn of nothing but the closing
    call, and the record's own notices (the budget's word, the completion's declaration) after
    the last words. A reader handed only the answer sees a mid-task fragment and a 'complete'
    verdict and cannot tell them apart; the projection carries the tail, and the CLI prints it
    beside the answer, in order."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="Now the test file, with the server inlined:")
    log.append("message", "model", role="assistant", text="")
    budget_word = ("budget ran out (45 of 45 turns); the model is told to wrap up; "
                   "3 grace turns left to deliver")
    log.append("notice", "core", subject="nudge", text=budget_word)
    completion_word = "completion declared: verify_objective called; the run ends"
    log.append("notice", "core", subject="verify", text=completion_word)
    log.append("ended", "core", reason="complete", exit_code=0)
    projection = result.result(_read(tmp_path)[0])
    assert projection["text"] == "Now the test file, with the server inlined:"
    assert projection["tail_notices"] == [budget_word, completion_word]
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path)], out, err)
    assert code == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == "r1: complete (exit 0)"
    assert lines[1] == "Now the test file, with the server inlined:"
    assert lines[2] == budget_word
    assert lines[3] == completion_word


def test_a_delivered_answer_is_not_narrated_by_its_own_notices(tmp_path):
    """The notice a normal unattended run leaves after its answer (the completion's
    declaration) does not make the answer stale: the last turn said the words, and a projection
    that appended its notices anyway would narrate a delivery that was not stale. The tail is
    empty and the output is exactly what it has always been."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="Both tools answered: 5 and mcp e2e.")
    log.append("notice", "core", subject="verify",
               text="completion declared: verify_objective called; the run ends")
    log.append("ended", "core", reason="complete", exit_code=0)
    projection = result.result(_read(tmp_path)[0])
    assert projection["tail_notices"] == []
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path)], out, err)
    assert code == 0
    assert out.getvalue().splitlines() == ["r1: complete (exit 0)",
                                           "Both tools answered: 5 and mcp e2e."]


def test_a_run_with_no_words_at_all_has_no_answer(tmp_path):
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="")
    log.append("ended", "core", reason="failed", exit_code=1)
    projection = result.result(_read(tmp_path)[0])
    assert projection["text"] is None and projection["answer_id"] is None


def test_a_run_still_running_has_no_ending(tmp_path):
    """No `ended` step yet: the projection says so rather than guessing — a run that is going
    is a different fact from a run that ended badly, and a parent chaining on the exit code
    must be able to tell them apart."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="still working")
    projection = result.result(_read(tmp_path)[0])
    assert projection["ended"] is False and projection["reason"] is None
    assert projection["exit_code"] is None and projection["text"] == "still working"


def test_a_torn_tail_is_a_damaged_read_not_a_crash(tmp_path):
    """A run killed mid-write of its closing step leaves a line that is not JSON: the lenient
    reader drops it and says so, and a projection over the surviving steps must not claim the
    ending that the torn line carried."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="half a word")
    log.append("ended", "core", reason="complete", exit_code=0)
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-40], encoding="utf-8")   # kill mid-line
    steps, damage = _read(tmp_path)
    assert damage, "the reader must name the torn line"
    projection = result.result(steps)
    assert projection["ended"] is False and projection["reason"] is None


def test_a_line_whole_but_newline_torn_is_read_not_reported(tmp_path):
    """A kill after the last byte of a step, before its newline, leaves a line that is complete
    JSON: it parses, the step it holds stands, and nothing is reported — a reader that named
    this as damage would be naming a newline, not a lost fact."""
    log = _log(tmp_path)
    log.append("ended", "core", reason="complete", exit_code=0)
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-1], encoding="utf-8")   # the newline alone is torn
    steps, damage = _read(tmp_path)
    assert damage == "" and steps[-1]["kind"] == "ended"


def test_the_run_id_falls_back_to_the_folder_name(tmp_path):
    """A log with no `started` step (foreign, or cut) still gets an identity: the caller's
    folder name, which is the run id the operator pointed at."""
    log = Log(tmp_path / "log.jsonl")
    log.append("ended", "core", reason="complete", exit_code=0)
    assert result.result(_read(tmp_path)[0])["run_id"] == ""
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path), "--json"], out, err)
    assert code == 0
    assert json.loads(out.getvalue())["run_id"] == tmp_path.name


def test_the_cli_prints_the_answer_and_exits_with_the_run_s_code(tmp_path):
    """The shell contract: the answer on stdout, the run's verdict as the exit code — a
    `sah result <dir> && next` chain gates on the run the way `sah run` does."""
    _finished_run(tmp_path, answer="The result is 42.")
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path)], out, err)
    assert code == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == "r1: complete (exit 0)", lines
    assert "The result is 42." in out.getvalue()


def test_the_cli_returns_the_run_s_failure_code(tmp_path):
    _finished_run(tmp_path, reason="max_turns", exit_code=1, answer="")
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path)], out, err)
    assert code == 1
    assert out.getvalue().splitlines()[0] == "r1: max_turns (exit 1)"
    # no answer, so nothing after the status line — an empty answer is not printed as one
    assert len(out.getvalue().splitlines()) == 1


def test_the_cli_reports_a_still_running_run_with_its_own_code(tmp_path):
    """Neither success (0) nor failure (1): the record has no ending yet, so a chain must not
    read either way. Its own code is 3."""
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="still working")
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path)], out, err)
    assert code == 3
    assert out.getvalue().splitlines() == ["r1: still running (no answer yet)"]


def test_the_cli_json_is_one_parseable_line(tmp_path):
    """The machine form: one line, every fact the projection carries, damage included when
    there is some — a parent parsing this never has to open the log itself."""
    answer_id = _finished_run(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path), "--json"], out, err)
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload == {"run_id": "r1", "ended": True, "reason": "complete",
                       "exit_code": 0, "text": "42", "answer_id": answer_id,
                       "tail_notices": []}


def test_the_cli_json_carries_damage_beside_the_answer(tmp_path):
    log = _log(tmp_path)
    log.append("message", "model", role="assistant", text="half a word")
    log.append("ended", "core", reason="complete", exit_code=0)
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-40], encoding="utf-8")   # kill mid-line
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path), "--json"], out, err)
    assert code == 3, "the torn ending means no ending: the run reports as still going"
    payload = json.loads(out.getvalue())
    assert payload["ended"] is False and payload["damage"], payload


def test_a_missing_log_is_a_usage_error(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["result", "--run-dir", str(tmp_path / "never")], out, err)
    assert code == 2
    assert "no log" in err.getvalue()


def test_the_answer_of_a_cut_run_comes_from_the_lineage_not_the_detached_tail(tmp_path):
    """The projection is only as good as the steps it is given: the raw file still holds the
    tail a cut detached — its answer and its ending — and a reader of the raw file reports the
    detached run as if it were the run in force. Over the lineage the conversation consumers
    are given, the cut is back at the step the operator named. This is the one-liner every
    conversation consumer applies before projecting, pinned here so a new raw reader shows up
    as a test failure."""
    log = _log(tmp_path)
    log.append("message", "model", role="user", text="the ask")
    log.append("message", "model", role="assistant", text="old answer")
    log.append("ended", "core", reason="complete", exit_code=0)
    target = next(s for s in log.steps()
                  if s["kind"] == "message" and s.get("role") == "user")["id"]
    log.append("cut", "core", to=target)
    log.append("message", "model", role="user", text="steer again")
    steps, _ = read_log(tmp_path / "log.jsonl", tolerate_damage=True)
    raw = result.result(steps)
    cut = result.result(lineage(steps))
    assert raw["text"] == "old answer" and raw["ended"] is True, \
        "the raw file still reports the detached tail as the run"
    assert cut["text"] is None and cut["ended"] is False, \
        "the conversation is back at the step the cut named — no answer yet, no ending yet"
