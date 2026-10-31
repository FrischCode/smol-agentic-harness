"""The Markdown export — a projection over the log, so the checks are that it renders real content
faithfully, keeps the audit (refusals, asks, decisions), never drops a step type it does not know,
and writes the same thing through the CLI it does as a pure function."""
import io
import json

import pytest

from sah import confine
from sah.record import Refused

from sah import cli, export
from sah.record import Log


def _log(tmp_path):
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m", grants={"write": ["/w"]})
    log.append("message", "user", role="user", text="do the thing")
    log.append("call", "model", refs="u1", name="write", args={"path": "/w/a.txt"})
    log.append("operation", "core", op="write", address="/w/a.txt", outcome="allowed", reason="")
    log.append("result", "core", refs="c1", ok=True, content="created /w/a.txt")
    log.append("message", "model", role="assistant", text="done")
    log.append("ended", "core", reason="complete", exit_code=0)
    return log.steps()


def test_export_renders_a_readable_transcript(tmp_path):
    md = export.render(_log(tmp_path))
    assert "# run r1" in md
    assert "**you:** do the thing" in md
    assert "`write`" in md and "/w/a.txt" in md
    assert "✓" in md
    assert "**agent:** done" in md
    assert "ended: complete (exit 0)" in md
    assert "grants:" in md and '"write"' in md   # the grants header (not a path substring) proves it's shown


def test_the_report_names_the_core_when_it_spoke_in_the_users_seat(tmp_path):
    """The core puts words in the user's seat (a nudge, a wrap-up, a future word), and a report that
    renders them as 'you' tells the operator they typed what the harness said. The voice follows the
    issuer, never the role or the subject — the channel owns the face — and operator input keeps the
    operator's face."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "user", role="user", text="typed by the operator")
    log.append("message", "core", role="user", text="emit that call fresh now")
    log.append("ended", "core", reason="complete", exit_code=0)
    md = export.render(log.steps())
    assert "**you:** typed by the operator" in md, md
    assert "**core:** emit that call fresh now" in md, md
    assert "**you:** emit that call fresh now" not in md, md


def test_the_report_closes_each_turn_with_the_status_footer(tmp_path):
    """The terminal's status footer (window pair, the token figures, the rates, the wall time) is a
    projection of the usage the record already carries, so the report shows the window filling as it
    fills: the last footer before a silence is what tells a long request from a hang. The shape is
    the one the terminal closes the box with, so a reader of either projection reads the same line.
    """
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "user", role="user", text="do the thing")
    log.append("message", "model", role="assistant", text="done", tokens_in=4912,
               context_max=32768, tokens_cached=0, tokens_out=38, prefill_tps=872.0,
               tps=45.0, ms=6600)
    log.append("ended", "core", reason="complete", exit_code=0)
    md = export.render(log.steps())
    assert "*4912/32768 CTX · ¤0 ›38 ↑872 ↓45/s TOK · 6.6s*" in md, md


def test_a_turn_the_server_never_annotated_gets_no_footer(tmp_path):
    """A footer is a projection of a measurement: a turn the server never annotated (no usage
    fields on the step) gets no line, and the report does not invent figures the record does not
    carry."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "model", role="assistant", text="done")
    log.append("ended", "core", reason="complete", exit_code=0)
    md = export.render(log.steps())
    assert "CTX" not in md


def test_a_call_only_turn_still_carries_its_footer(tmp_path):
    """The footer belongs to the turn, not to its prose: a turn that was only a tool call carries
    the usage the call was asked with, and the report shows it."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "model", role="assistant", text="", tokens_in=100, context_max=32768,
               tokens_out=5, ms=1200)
    log.append("ended", "core", reason="complete", exit_code=0)
    md = export.render(log.steps())
    assert "*100/32768 CTX · ›5 TOK · 1.2s*" in md, md


def test_export_keeps_the_audit_a_refusal_and_an_ask(tmp_path):
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("operation", "core", op="write", address="/x", outcome="refused", reason="not granted")
    log.append("request", "core", subject="widen", item="/x", why="write not granted")
    log.append("decision", "queue", refs="req1", outcome="accepted", reason="answered yes by operator")
    md = export.render(log.steps())
    assert "refused `write`" in md and "not granted" in md
    assert "asked (widen)" in md
    assert "decided `accepted`" in md


def test_export_never_drops_a_kind_it_does_not_know():
    """A future schema, or an older export reading a newer log: an unknown kind prints its declared
    fields rather than vanishing. A report that silently omits a step type is worse than crude."""
    md = export.render([{"kind": "telemetry", "id": "x", "ts": "t", "issuer": "core", "tokens": 42}])
    assert "telemetry" in md and "tokens=42" in md


def test_export_cli_writes_markdown_to_a_file(tmp_path):
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r9", root="/r")
    log.append("ended", "core", reason="complete", exit_code=0)
    target = tmp_path / "report.md"
    code = cli.build(["export", "--run-dir", str(tmp_path), "--out", str(target)],
                     io.StringIO(), io.StringIO())
    assert code == 0 and "# run r9" in target.read_text(encoding="utf-8")


def test_export_cli_reports_a_missing_log(tmp_path):
    err = io.StringIO()
    code = cli.build(["export", "--run-dir", str(tmp_path / "nowhere")], io.StringIO(), err)
    assert code == 2 and "no log" in err.getvalue()


def _raw(path, obj):
    """Append a line the *writer* would refuse — a step from a schema this build does not know.

    Writing it past `Log.append` is the point: an unknown kind hand-built in a test never went near the
    reader, which is how a fallback came to be unreachable from any real log and still look covered."""
    with open(path, "a", encoding="utf-8") as h:
        h.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def test_a_newer_kind_in_a_real_log_reaches_the_report_crude_rather_than_refusing(tmp_path):
    """The module promises that an older export reading a newer log prints the unknown kind crudely. The
    reader in front of it refused the line, so the whole report failed with `Refused` and the fallback
    could not be reached from any log on disk — the promise was about a path no caller could walk.

    The crude line is the good outcome: a report that shows a step it cannot interpret still tells the
    reader the run did something here."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r7", root="/r")
    log.append("ended", "core", reason="complete", exit_code=0)
    _raw(tmp_path / "log.jsonl", {"kind": "telemetry", "id": "t1", "seq": 3, "ts": "t",
                                  "issuer": "core", "tokens": 42})
    target = tmp_path / "report.md"
    code = cli.build(["export", "--run-dir", str(tmp_path), "--out", str(target)],
                     io.StringIO(), io.StringIO())
    md = target.read_text(encoding="utf-8") if target.exists() else ""
    assert code == 0, "a log holding one future step made the whole report unreadable"
    assert "telemetry" in md and "tokens=42" in md, f"the unknown kind was dropped: {md[:200]!r}"


def test_the_audit_reader_still_refuses_a_kind_the_taxonomy_does_not_name(tmp_path):
    """The control that keeps lenience a policy for projections and not a global loosening. The audit path
    answers "what did this run do" and must not render a step it cannot check against the taxonomy; a
    report can say *something unrecognised happened here*, an audit trail cannot."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r8", root="/r")
    _raw(tmp_path / "log.jsonl", {"kind": "telemetry", "id": "t1", "seq": 2, "ts": "t",
                                  "issuer": "core", "tokens": 1})
    with pytest.raises(Refused):
        log.steps()


def test_a_field_this_export_predates_is_printed_rather_than_dropped(tmp_path):
    """A newer log does not only add kinds; it adds fields to kinds that already exist. Refusing those
    would reintroduce the same dead end through a different door, and ignoring them would mean the report
    quietly stopped carrying information the log holds."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r6", root="/r")
    log.append("call", "model", refs="m1", name="read", args={"path": "/x"})
    _raw(tmp_path / "log.jsonl", {"kind": "result", "id": "z1", "seq": 3, "ts": "t", "issuer": "core",
                                  "refs": "c1", "ok": True, "content": "hello", "cache_hit": True})
    target = tmp_path / "report.md"
    code = cli.build(["export", "--run-dir", str(tmp_path), "--out", str(target)],
                     io.StringIO(), io.StringIO())
    md = target.read_text(encoding="utf-8") if target.exists() else ""
    assert code == 0, "a new field on a known kind made the report fail"
    assert "cache_hit" in md, f"the field this build does not know was dropped: {md[:260]!r}"


def test_each_result_is_labelled_with_the_call_that_produced_it(tmp_path):
    """`result` carries no `name` — the name is on the `call` it refers to — so the label fell back to
    "output" and a report of ten tools read as ten blocks titled `output:`. A reader reconstructing a run
    from Markdown has only the label to tell which call a result answers."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r5", root="/r")
    first = log.append("call", "model", refs="m1", name="write", args={"path": "/w/a"})
    log.append("result", "core", refs=first["id"], ok=True, content="created /w/a")
    second = log.append("call", "model", refs="m1", name="shell", args={"command": "ls"})
    log.append("result", "core", refs=second["id"], ok=False, content="no such file")
    md = export.render(log.steps())
    assert "output:" not in md, f"every result still answers to 'output':\n{md}"
    assert "write:" in md and "shell:" in md, md
    assert "✗" in md


def test_consecutive_step_blocks_are_separated_by_exactly_one_blank_line(tmp_path):
    """A turn seam is the reader's only cue that one turn ended and the next began; without the blank
    line, consecutive turns merge into one paragraph in any Markdown view. One blank line — a run of
    blanks renders identically in every view, so more than one would show only in the raw text."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "user", role="user", text="first")
    log.append("message", "model", role="assistant", text="second")
    log.append("message", "model", role="assistant", text="third")
    log.append("ended", "core", reason="complete", exit_code=0)
    assert export.render(log.steps()) == (
        "# run r1\n\n"
        "model: m · ended: complete (exit 0)\n\n"
        "**you:** first\n\n**agent:** second\n\n**agent:** third\n")


def test_an_empty_block_and_a_block_with_its_own_blank_take_no_extra_separator(tmp_path):
    """The separator is between blocks, never inside one, and never doubled: a step that renders
    nothing contributes neither a line nor a blank, and a block that carries an interior blank line
    (a summary) is separated from its neighbours exactly once."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m")
    log.append("message", "user", role="user", text="")
    log.append("summary", "core", refs=[], kept=1, dropped=1, text="the shape so far", outcome="ok")
    log.append("message", "model", role="assistant", text="after")
    log.append("ended", "core", reason="complete", exit_code=0)
    md = export.render(log.steps())
    assert "\n\n\n" not in md, md
    assert "### summary (ok, kept 1, dropped 1)\n\nthe shape so far\n\n**agent:** after" in md, md


def test_a_session_report_keeps_every_objective_boundary(tmp_path):
    """A REPL session writes one `started`/`ended` pair per objective, and the header showed only the
    first and the last — so a four-turn session reported one ending, and the places where one objective
    stopped and the next began were gone from the one artefact a person reads afterward."""
    log = Log(str(tmp_path / "log.jsonl"))
    for i, (reason, code) in enumerate((("complete", 0), ("interrupted", 130))):
        log.append("started", "core", run_id=f"r-{i}", root="/r", model="m")
        log.append("message", "user", role="user", text=f"objective {i}")
        log.append("ended", "core", reason=reason, exit_code=code)
    md = export.render(log.steps())
    assert md.count("---") >= 2, "the second objective's boundary is missing"
    assert "interrupted" in md and "complete" in md, "only one of the two endings is reported"
    # Each ending appears once. With more than one objective the header stops carrying a verdict, since
    # a single "ended:" line would be a claim about the session that only the last objective supports.
    assert md.count("ended:") == 2, "an ending was reported twice"
    # Where one objective stopped, the next began: a reader of the Markdown needs both sides of the seam,
    # not a rule with nothing on either side of it.
    assert "run r-1 begins" in md, "the second objective's opening is unmarked"
    # A rule line directly under a text line reads as a setext heading underline, not a thematic
    # break, so each boundary's `---` must sit under a blank line.
    for i, line in enumerate(md.splitlines()):
        if line == "---":
            assert i == 0 or md.splitlines()[i - 1] == "", \
                f"the boundary rule at line {i} would parse as a heading underline"


def test_a_rewording_of_the_disclosure_moves_the_report_line_with_it(tmp_path, monkeypatch):
    """The export recognises the exec disclosure by identity, not by a phrase copied out of it, and this
    checks the coupling rather than the rendering.

    The policy emits that sentence as exactly one constant, so today a substring match behaves the same
    and no fixture can tell the two apart by content alone. Retyping the constant is what shows it: the
    line follows the wording when the comparison is by identity, and a report silently loses its only
    notice that path confinement was off when the comparison is to words somebody copied.

    A refusal that quotes the sentence is asserted separately because the refusal branch runs first: a
    refusal must read as a refusal whatever its reason contains."""
    wording = "EXEC MODE IS ON: no path boundary binds this call"
    monkeypatch.setattr(confine, "EXEC_DISCLOSURE", wording)
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r4", root="/r")
    log.append("operation", "core", op="write", address="/w/a", outcome="allowed",
               reason=confine.EXEC_DISCLOSURE)
    log.append("operation", "core", op="write", address="/etc/passwd", outcome="refused",
               reason="refused: the grant is missing (" + wording + ")")
    md = export.render(log.steps())
    assert "  _" + wording + "_" in md, "the disclosure line did not follow the policy's wording"
    assert "! refused `write`" in md, "a refusal must read as a refusal whatever its reason quotes"


def test_a_failed_summary_shows_why_it_failed(tmp_path):
    """The verdict moved out of the outcome field into a single word, and the sentence went to `reason` — so a
    projection that renders only the outcome would print "failed" and drop the only part an operator can act
    on. This test exists because the fix was correct and still would have lost the fact."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m", grants={})
    log.append("summary", "core", refs=[], kept=2, dropped=5, text="(no summary produced)",
               outcome="failed", reason="the server refused the request: too big")
    log.append("ended", "core", reason="complete", exit_code=0)
    body = export.render(log.steps())
    assert "failed" in body, body
    assert "too big" in body, f"the reason never reached the report: {body}"


# --- the report an unattended run leaves behind ---------------------------------

def test_write_report_leaves_the_projection_with_its_source_named(tmp_path):
    """The file a caller arriving after the fact reads next to the record: the same projection the
    command renders, under a header that names its source, its step count, and the hierarchy —
    the log is the record, this is the reading of it."""
    steps = _log(tmp_path)
    assert export.write_report(str(tmp_path)) is True
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.startswith(f"> rendered from log.jsonl · {len(steps)} steps · the log is the record")
    assert "# run r1" in text
    assert "do the thing" in text
    assert "ended: complete (exit 0)" in text


def test_write_report_writes_nothing_without_an_ending(tmp_path):
    """A run still in flight has no report: a partial file would look finished to exactly the
    reader it is for, so the write waits for the `ended` step the record already owes."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m", grants={})
    log.append("message", "user", role="user", text="do the thing")
    assert export.write_report(str(tmp_path)) is False
    assert not (tmp_path / "report.md").exists()


def test_write_report_writes_nothing_without_a_log(tmp_path):
    """No log, no run, no report: the write is a no-op rather than a crash at the ending."""
    assert export.write_report(str(tmp_path)) is False
    assert not (tmp_path / "report.md").exists()


def test_write_report_replaces_the_older_report(tmp_path):
    """The ending writes the file; a later ending (a resumed run's own exit) rewrites it whole.
    The report is one projection of the whole log, never an append — the old step count must not
    survive inside the new file."""
    log = Log(str(tmp_path / "log.jsonl"))
    log.append("started", "core", run_id="r1", root="/r", model="m", grants={})
    log.append("ended", "core", reason="blocked", exit_code=1)
    assert export.write_report(str(tmp_path)) is True
    first = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "· 2 steps ·" in first and "ended: blocked (exit 1)" in first
    log.append("message", "user", role="user", text="again")
    log.append("ended", "core", reason="complete", exit_code=0)
    assert export.write_report(str(tmp_path)) is True
    second = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "· 4 steps ·" in second and "ended: complete (exit 0)" in second
    assert "· 2 steps ·" not in second
