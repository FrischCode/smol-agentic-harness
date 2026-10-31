"""Recall reads raw past run logs, confined to a runs root. The checks: the index reports what the
log already knows (model, verdict, objective, calls) newest-first; since/grep filter it; run_log
renders and pages so a long run is never a flood; a run id is one folder name, never a path; and it
joins the run as a read-only extension only when a root was named."""
import io
import json

import pytest

from sah import cli
from sah.tools import recall
from sah.model import Caps, Fake
from sah import record
from sah.record import Log


def _make_run(root, run_id, *, model="m", objective="do the thing", reason="complete",
              exit_code=0, calls=0):
    log = Log(str(root / run_id / "log.jsonl"))
    log.append("started", "core", run_id=run_id, root=str(root / run_id), model=model)
    log.append("message", "user", role="user", text=objective)
    for i in range(calls):
        log.append("call", "model", refs="u", name="read", args={"path": f"/x/{i}"})
        log.append("result", "core", refs="c", ok=True, content="ok")
    log.append("ended", "core", reason=reason, exit_code=exit_code)
    return log


def test_index_lists_runs_newest_first_with_log_fields(tmp_path):
    _make_run(tmp_path, "runa", model="alpha", objective="build the widget")
    _make_run(tmp_path, "runb", model="beta", objective="fix the flake", reason="stalled",
              exit_code=3, calls=2)
    text = recall.Runs(str(tmp_path)).index()
    assert "2 run(s)" in text
    assert "runb" in text and "runa" in text
    assert "beta" in text and "stalled (exit 3)" in text        # verdict + model come from the log
    assert "build the widget" in text                            # the objective is shown
    assert "2 call(s)" in text
    assert text.index("runb") < text.index("runa")               # same-day, ordered newest-first


def test_index_since_filters_and_reports_a_bad_value(tmp_path):
    _make_run(tmp_path, "runx", objective="x")
    runs = recall.Runs(str(tmp_path))
    assert "runx" in runs.index(since="7d")                       # today is within the last 7 days
    assert "no runs matched" in runs.index(since="2999-01-01")    # a future cut-off excludes it
    assert "could not read" in runs.index(since="banana")         # a junk value is said, not ignored


def test_index_grep_filters_by_content(tmp_path):
    _make_run(tmp_path, "runa", objective="migrate the database")
    _make_run(tmp_path, "runb", objective="write a poem")
    text = recall.Runs(str(tmp_path)).index(grep="database")
    assert "runa" in text and "runb" not in text


def test_run_log_renders_and_pages(tmp_path):
    _make_run(tmp_path, "runbig", objective="the big one", calls=40)
    runs = recall.Runs(str(tmp_path))
    full = runs.run_log("runbig")
    assert "# run runbig" in full and "the big one" in full        # export's rendering, reused
    page = runs.run_log("runbig", count=5)
    assert "offset=" in page and "run has" in page                 # a small page says how to continue
    paged_from = runs.run_log("runbig", offset=3, count=3)
    assert "# run runbig" not in paged_from                        # offset skipped the header


def test_run_log_refuses_a_path_and_names_unknown_ids(tmp_path):
    _make_run(tmp_path, "known", objective="y")
    runs = recall.Runs(str(tmp_path))
    with pytest.raises(ValueError):
        runs.run_log("../escape")                                  # one folder name, never a path
    with pytest.raises(ValueError):
        runs.run_log("a/b")
    assert "no run" in runs.run_log("ghost") and "known" in runs.run_log("ghost")


def test_recall_is_an_extension_of_exactly_two_read_only_tools(tmp_path):
    ext = recall.extension(recall.Runs(str(tmp_path)))
    assert {f.__name__ for f in ext.tools} == {"runs", "run_log"}


def test_recall_tools_only_advertised_with_a_runs_root(tmp_path, monkeypatch):
    runs_root = tmp_path / "history"
    runs_root.mkdir()
    base = ["run", "--run-dir", str(tmp_path / "r"), "--host", "127.0.0.1", "--port", "1",
            "--model", "fake"]

    def names(argv):
        box = {}
        monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: box.setdefault("agent", agent) or 0)
        monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
        cli.build(argv, io.StringIO(), io.StringIO())
        return {spec.name for spec in box["agent"].specs}

    assert "runs" not in names(base) and "run_log" not in names(base)          # no root, no history
    granted = names(base + ["--runs-root", str(runs_root)])
    assert "runs" in granted and "run_log" in granted


def test_a_run_whose_last_line_is_torn_still_lists(tmp_path):
    """The newest run in the history is the one most likely to be mid-write or mid-crash, so a
    history query must not depend on it being intact. The reader drops the tail it cannot parse and
    the row says it did — a silent omission would make the run look undamaged."""
    _make_run(tmp_path, "runok")
    _make_run(tmp_path, "runlive", objective="still working")
    with (tmp_path / "runlive" / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"result","id":"000005-resu')            # no newline: mid-append
    text = recall.Runs(str(tmp_path)).index()
    assert "runlive" in text and "runok" in text
    assert "unreadable" in text.lower(), f"the index hid the damage: {text}"


def test_one_damaged_log_does_not_break_the_whole_index(tmp_path):
    """RCL-1's acceptance shape, pinned now that recall reads through the shared reader. A log
    damaged in the middle -- what a crash plus a later append leaves -- used to make every query
    raise, so one bad run cost the model its entire history. It is now listed with the damage named,
    and its log still renders."""
    _make_run(tmp_path, "good")
    _make_run(tmp_path, "rotten", objective="the one that broke")
    path = tmp_path / "rotten" / "log.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"kind":"message","id":"000001-mes'          # damage, and not at the end
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs = recall.Runs(str(tmp_path))
    text = runs.index()
    assert "rotten" in text and "good" in text, f"one bad run hid the history: {text}"
    assert "unreadable" in text.lower(), f"the row looked intact: {text}"
    assert "damaged" in runs.run_log("rotten").lower()


def test_a_full_run_log_page_fits_the_result_limit(tmp_path, fits_a_result):
    """`run_log` clipped its window at the RESULT limit and then appended "continue with offset=N", so
    the page arrived over the limit and the loop spilled a run the model was already paging -- copying
    a log to `out/` to hand back a copy of a file the tool can read. The notice was the part cut off."""
    log = Log(str(tmp_path / "runhuge" / "log.jsonl"))
    log.append("started", "core", run_id="runhuge", root=str(tmp_path / "runhuge"), model="m")
    log.append("result", "core", refs="c", ok=True,
               content="".join(f"output line {i}\n" for i in range(5000)))
    log.append("ended", "core", reason="complete", exit_code=0)
    page = recall.Runs(str(tmp_path)).run_log("runhuge")
    fits_a_result(page)
    assert "continue with run_log(run_id, offset=" in page, page[-200:]


def test_a_future_step_does_not_make_a_run_unrecallable(tmp_path):
    """A history query is asked of logs written at other times, possibly by other builds. Refusing an
    undeclared kind here would mean the index stops listing a run — the operator's question about last
    week answered by a parse error, with no way to ask it another way.

    The row and the rendered log both carry the unrecognised step crudely, because a history that
    silently skips what it cannot interpret reads as a run that did not do it."""
    log = _make_run(tmp_path, "runfuture", objective="old work")
    with open(log.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "telemetry", "id": "t9", "ts": "t", "issuer": "core",
                                 "tokens": 7}) + "\n")
    runs = recall.Runs(str(tmp_path))
    assert "runfuture" in runs.index(), "the run dropped out of the index"
    log = runs.run_log("runfuture")
    # Asserted as the crude rendering, not the bare word: when the reader refuses the line, the loss note
    # names the kind too, and a test looking only for the word passes on the message that reports it lost.
    assert "telemetry: tokens=7" in log, "the step this build predates was not rendered"


# --- what `grep` is allowed to match ------------------------------------------

def test_a_term_that_only_appears_inside_tool_output_does_not_match(tmp_path):
    """The parameter says the term is matched against run id, model and objective.

    It matched `str(steps[:60])` — the repr of up to sixty steps, which holds result bodies, reasoning, file
    contents and dictionary punctuation. So searching for a word answered with every run that had merely
    *read* a file containing it, which is not a history of what was asked."""
    log = Log(str(tmp_path / "reader" / "log.jsonl"))
    log.append("started", "core", run_id="reader", root=str(tmp_path / "reader"), model="m")
    log.append("message", "user", role="user", text="count the files here")
    log.append("call", "model", refs="c", name="read", args={"path": "/x/notes.md"})
    log.append("result", "core", refs="c", ok=True, content="pineapple appeared in that file")
    log.append("ended", "core", reason="complete", exit_code=0)

    other = Log(str(tmp_path / "asker" / "log.jsonl"))
    other.append("started", "core", run_id="asker", root=str(tmp_path / "asker"), model="m")
    other.append("message", "user", role="user", text="was pineapple ever mentioned")
    other.append("ended", "core", reason="complete", exit_code=0)

    text = recall.Runs(str(tmp_path)).index(grep="pineapple")
    assert "asker" in text, "the run that asked about the term did not match"
    assert "reader" not in text, \
        f"a run that only read a file holding the term matched: {text}"


def test_a_later_objective_in_a_long_session_still_matches(tmp_path):
    """A REPL run asks several things. The filter looked at the first sixty steps, so anything asked after
    that was invisible to the one tool a model uses to find its own past."""
    log = Log(str(tmp_path / "long" / "log.jsonl"))
    log.append("started", "core", run_id="long", root=str(tmp_path / "long"), model="m")
    log.append("message", "user", role="user", text="start the session")
    for i in range(80):
        log.append("call", "model", refs="c", name="read", args={"path": f"/x/{i}"})
        log.append("result", "core", refs="c", ok=True, content="ok")
    log.append("message", "user", role="user", text="now about the quay-bridge")
    log.append("ended", "core", reason="complete", exit_code=0)
    assert "long" in recall.Runs(str(tmp_path)).index(grep="quay-bridge")


def test_the_filter_still_answers_id_and_model(tmp_path):
    """The other two documented fields, kept honest as constraints rather than as fixes: a term matching the
    run id or the model name is what the schema promises, and a narrowing that drops them would trade one
    wrong answer for another."""
    _make_run(tmp_path, "r-20261026-a1b2c3", model="quill", objective="nothing special")
    runs = recall.Runs(str(tmp_path))
    assert "r-20261026-a1b2c3" in runs.index(grep="a1b2c3")
    assert "r-20261026-a1b2c3" in runs.index(grep="quill")


# --- what the index has to read -----------------------------------------------

def count_full_reads(monkeypatch):
    """Count calls to the one full reader, so a claim about work done is a number rather than a hope."""
    calls: list[str] = []
    real = recall._steps

    def spy(run_dir):
        calls.append(run_dir)
        return real(run_dir)

    monkeypatch.setattr(recall, "_steps", spy)
    return calls


def test_the_index_reads_only_the_logs_the_page_shows(tmp_path, monkeypatch):
    """The index used to read every log under the root, then keep `limit` of them.

    Measured on 60 logs of 208 KiB: 700 ms per call, of which 87% is parsing lines the caller never sees --
    and parsing is the part a streaming reader cannot remove, because a line has to be parsed to be judged.
    So the saving is not in how it reads, it is in how much it decides to read: order by what a first line
    says, then open only as many logs as the page holds."""
    for i in range(30):
        _make_run(tmp_path, f"r20261026-0000{i:02d}-abcdef", objective=f"task {i}")
    calls = count_full_reads(monkeypatch)
    text = recall.Runs(str(tmp_path)).index(limit=3)
    assert text.count("- r20261026") == 3, text
    assert len(calls) <= 3, f"the index opened {len(calls)} logs to show 3"


def test_a_date_filter_does_not_open_the_logs_it_is_about_to_drop(tmp_path, monkeypatch):
    """`since` is a promise about work as much as about output. Filtering by a date the log itself names must
    not require reading the logs that date excludes."""
    _make_run(tmp_path, "r20261026-000001-aaaaaa", objective="recent")
    # Dated by hand: `Log.append` stamps the current time, so an "old" run has to be written with the date
    # it is claiming, or the filter is correct to list it and the test measures nothing.
    old = tmp_path / "r20260101-000002-bbbbbb" / "log.jsonl"
    old.parent.mkdir(parents=True)
    bodies = [("started", {"run_id": "r20260101-000002-bbbbbb", "root": "r", "model": "m"}),
              ("message", {"role": "user", "text": "ancient"}),
              ("ended", {"reason": "complete", "exit_code": 0})]
    old.write_text("\n".join(record.dumps({"kind": kind, "id": f"{i}-{kind}",
                                           "ts": f"2026-01-01T05:00:0{i}+00:00",
                                           "issuer": "core", **fields})
                              for i, (kind, fields) in enumerate(bodies)) + "\n", encoding="utf-8")

    calls = count_full_reads(monkeypatch)
    text = recall.Runs(str(tmp_path)).index(since="7d")
    assert "r20261026" in text and "r20260101" not in text, text
    assert len(calls) <= 1, f"a 7-day filter opened {len(calls)} logs, including one it dropped"


def test_a_search_still_reads_every_candidate(tmp_path, monkeypatch):
    """The constraint on that saving: `grep` is a question about all of history, so narrowing the reads must
    not narrow what is searched.

    Several runs have to match, not one. With a single match the loop never fills the page, so a mutant
    that stopped a search early read all twelve anyway and passed the test -- measured. Four matches among
    twelve is the shape that distinguishes 'a search reads everything' from 'a search reads until the
    page is full'."""
    for i in range(12):
        _make_run(tmp_path, f"r20261026-0000{i:02d}-abcdef",
                  objective="the quay bridge" if i % 3 == 0 else f"task {i}")
    calls = count_full_reads(monkeypatch)
    # drives the read count; the result is not what's asserted
    recall.Runs(str(tmp_path)).index(grep="quay bridge", limit=3)
    assert len(calls) == 12, f"a search opened {len(calls)} of 12 logs and answered about all of them"
    # And the deep match is really in the result set, not merely read: a page of three shows the three
    # newest, so this asks with a page wide enough to hold all four.
    wide = recall.Runs(str(tmp_path)).index(grep="quay bridge", limit=12)
    assert wide.count("quay bridge") == 4, f"a search lost a match:\n{wide}"


def test_the_order_came_from_the_log_and_not_the_folder_name(tmp_path):
    """Ordering by name is what makes the early stop cheap, and it is only correct while every id was minted
    with a timestamp in it. A folder named by hand breaks that, so the day is read from the log."""
    _make_run(tmp_path, "aaa-late", objective="asked in october")
    _make_run(tmp_path, "zzz-early", objective="asked in january")
    # Rewrite the days: name order says zzz is newest, the logs say otherwise.
    for name, day in (("aaa-late", "2026-10-20"), ("zzz-early", "2026-01-05")):
        path = tmp_path / name / "log.jsonl"
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        for step in lines:
            step["ts"] = day + "T05:00:00+00:00"
        path.write_text("\n".join(json.dumps(s) for s in lines) + "\n", encoding="utf-8")

    text = recall.Runs(str(tmp_path)).index()
    assert text.index("aaa-late") < text.index("zzz-early"), text

def test_a_run_that_cannot_be_dated_is_still_listed(tmp_path):
    """A first line that will not parse must cost an extra read, not the run.

    The peek is an optimisation and an optimisation is allowed to be ignorant: it returns no date and the
    caller then opens the log like any other. Sorting it last is the honest fallback -- the caller cannot know
    how old it is -- but dropping it would make a damaged log disappear from the only list that says it
    exists."""
    _make_run(tmp_path, "runok", objective="fine")
    _make_run(tmp_path, "undated", objective="the torn one")
    torn = tmp_path / "undated" / "log.jsonl"
    torn.write_text("{torn first line" + chr(10) + torn.read_text(encoding="utf-8"), encoding="utf-8")

    runs = recall.Runs(str(tmp_path))
    text = runs.index()
    assert "undated" in text and "runok" in text, f"a damaged first line hid a run: {text}"
    assert text.index("runok") < text.index("undated"), "an undated run should sort last, not vanish"
    # And a date filter must not silently drop what it cannot date.
    assert "undated" in runs.index(since="7d"), "a since filter dropped a run it could not date"


def test_the_page_says_which_logs_it_never_opened(tmp_path):
    """The saving is only honest if the reader learns the page is a page. Twenty rows out of sixty must not
    read as the whole history, and the way to reach the rest has to be named in the same breath."""
    for i in range(30):
        _make_run(tmp_path, f"r20261026-0000{i:02d}-abcdef", objective=f"task {i}")
    text = recall.Runs(str(tmp_path)).index(limit=5)
    assert "30 run(s)" in text, text.splitlines()[0]
    head = text.splitlines()[0]
    assert "not read" in head and "25" in head, f"the header hides what it skipped: {head}"


def test_recall_reads_a_run_through_the_lineage_a_cut_left_in_force(tmp_path):
    """A cut is an operator act, not a step of the conversation: the history a reader of a
    run's history asks about is the line in force, and the detached tail — and the cut that
    detaches it — stay in the raw file, where the audit finds them."""
    _make_run(tmp_path, "cutrun")
    log = Log(str(tmp_path / "cutrun" / "log.jsonl"))
    log.append("cut", "core", to="000001-message")
    log.append("message", "model", role="user", text="steered")
    log.append("ended", "core", reason="complete", exit_code=0)
    steps, note = recall._steps(str(tmp_path / "cutrun"))
    assert note == ""
    assert [s["id"] for s in steps] == [
        "000000-started", "000001-message", "000004-message", "000005-ended"], \
        f"the detached tail, or the cut itself, came back into the history: {steps}"


def test_recall_reports_a_cut_it_cannot_project_and_shows_the_run_uncut(tmp_path):
    """A cut whose target the log no longer holds is a semantic gap in the record: a history
    query is answered by a report that names the gap, not by a refusal that breaks the index of
    every run beside it."""
    _make_run(tmp_path, "broken")
    log = Log(str(tmp_path / "broken" / "log.jsonl"))
    log.append("cut", "core", to="000999-message")
    steps, note = recall._steps(str(tmp_path / "broken"))
    assert any(s["kind"] == "cut" for s in steps), "the run is shown uncut, not refused"
    assert "cut not projectable" in note
