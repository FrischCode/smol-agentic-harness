"""Changing a span of a file, and the two properties that make that safer than `write`.

`write` replaces a file, so a run that wants three lines changed has to retype the rest and loses
whatever it did not retype — including, when the answer is cut off by its own output limit, the whole
tail of the file. `edit` replaces one matched span instead, which buys two things worth holding shut by
test rather than by reading the code:

* **A refusal costs no bytes.** The span is found before anything is opened for writing, so the common
  failure — text remembered rather than copied — leaves the file exactly as it was and says why.
* **Nothing outside the span moves.** Not one byte, including the line endings of lines nobody
  mentioned. That second property is why the tool matches on a view where every ending is one character
  rather than normalising the file and writing that back.
"""
import pytest

from sah.tools import fs
from sah.confine import Confine
from sah.contract import arguments_for, declaration, spec_for


def _bytes(path):
    return open(path, "rb").read()


# --- the span, and only the span -----------------------------------------------------------------

def test_an_edit_changes_the_span_it_matched_and_nothing_else(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("import os\n\n\ndef f():\n    return os.path\n", encoding="utf-8")
    before = _bytes(f)
    out = fs.edit(str(f), "    return os.path", "    return os.path.sep")
    assert f.read_text(encoding="utf-8") == "import os\n\n\ndef f():\n    return os.path.sep\n"
    assert out.startswith(f"edited {f}") and "1 place" in out, out
    # The receipt's sizes are checked against the file rather than against the function's own arithmetic,
    # so a receipt that reports what it meant to write goes red.
    assert f"{len(before)} -> {len(_bytes(f))} characters" in out, out


def test_the_receipt_names_the_line_the_change_landed_on(tmp_path):
    """A run that cannot see its own edit needs a place to aim the next `read`, and a line number is
    only worth printing if it is the one `read` answers with."""
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    out = fs.edit(str(f), "three", "THREE")
    assert "at line 3" in out, out
    assert "THREE" in fs.read(str(f), offset=3, count=1)


def test_an_edit_that_cannot_find_its_text_writes_nothing(tmp_path):
    """The precondition is judged before the file is opened for writing, so a wrong `old_text` is a
    refusal the run can repair rather than a change half applied. A destructive tool that checks after
    the damage reports its damage as success."""
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    before = _bytes(f)
    out = fs.edit(str(f), "two\nthree", "x")
    assert _bytes(f) == before, "a refusal changed the file"
    assert "nothing matched" in out


def test_an_edit_refuses_text_that_matches_twice_and_says_how_many(tmp_path):
    """Uniqueness is the tool's whole protection against landing on the second copy of a repeated line,
    and *two* is the case that matters: three failures in a row would still be caught by a guard written
    `> 2`. The refusal also has to give the model the fact it needs to widen the text."""
    f = tmp_path / "a.txt"
    f.write_text("total = 1\ntotal = 2\n", encoding="utf-8")
    before = _bytes(f)
    out = fs.edit(str(f), "total = ", "x = ")
    assert _bytes(f) == before, "an ambiguous edit was applied anyway"
    assert "2 places" in out and "every=true" in out, out


def test_changing_every_occurrence_is_an_explicit_act(tmp_path):
    """A rename across a file is one call, but only because the run asked for all of them. The default
    refuses the same text, so `every` is a widening the operator performs and not a default that guesses."""
    f = tmp_path / "a.txt"
    f.write_text("total = 1\ntotal = 2\ntotal = 3\n", encoding="utf-8")
    out = fs.edit(str(f), "total = ", "x = ", every=True)
    assert f.read_text(encoding="utf-8") == "x = 1\nx = 2\nx = 3\n"
    assert "3 place(s)" in out, out


def test_a_second_edit_sees_the_file_the_first_one_wrote(tmp_path):
    """Nothing carries between calls: each `old_text` is matched against the file as it is now. The
    alternative — matching a batch against one snapshot — lets a later change land on text an earlier
    one already removed, which reads as a tool editing a file that no longer exists."""
    f = tmp_path / "a.txt"
    f.write_text("alpha\n", encoding="utf-8")
    fs.edit(str(f), "alpha", "beta")
    assert "nothing matched" in fs.edit(str(f), "alpha", "gamma")
    assert fs.edit(str(f), "beta", "gamma").startswith("edited")
    assert f.read_text(encoding="utf-8") == "gamma\n"


def test_editing_text_to_itself_touches_no_bytes(tmp_path, monkeypatch):
    """The claim "nothing was written" is checkable, so it is checked: the tool is allowed to read the
    file and forbidden to open it for writing during the call that says it did not."""
    real_open = open

    def no_writing(path, mode="r", *a, **kw):
        assert "w" not in mode, "a no-op claim was backed by a write"
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr("builtins.open", no_writing)
    f = tmp_path / "a.txt"
    f.write_text("same\n", encoding="utf-8")
    assert "nothing was written" in fs.edit(str(f), "same", "same")


# --- line endings, which the model cannot see -----------------------------------------------------

def test_an_edit_matches_a_crlf_file_from_text_that_has_no_carriage_returns(tmp_path):
    """The reason a line ending cannot be treated as an ordinary character here.

    `read` shows a file through universal newlines, so a model that copies text out of a read cannot
    possibly type the CRLF the file holds. Matching raw would refuse every edit of a CRLF file with
    "nothing matched" — and on this platform most tracked files are CRLF, so the tool would be broken
    where it is used. Normalising the whole file on the way back out would rewrite endings the run
    never mentioned. So matching sees one kind of ending, the bytes keep the file's own, and the receipt
    says which it wrote.

    The second edit is the one that carries the weight: a span *inside* one line can be found in a raw
    CRLF file too, so only a request that crosses a line ending proves the reduced view exists.
    """
    f = tmp_path / "win.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    assert "\r" not in fs.read(str(f)), "read is what teaches the model to send LF"
    out = fs.edit(str(f), "beta", "BETA\nBETA2")
    assert _bytes(f) == b"alpha\r\nBETA\r\nBETA2\r\ngamma\r\n", _bytes(f)
    assert "CRLF" in out, "the receipt did not say which endings it wrote"
    fs.edit(str(f), "alpha\r\nBETA", "alpha\nBETA")       # typed with the endings the file really has
    across = fs.edit(str(f), "BETA\nBETA2\ngamma", "delta")
    assert across.startswith("edited"), across
    assert _bytes(f) == b"alpha\r\ndelta\r\n", _bytes(f)


def test_an_edit_leaves_a_mixed_ending_file_alone_outside_the_span(tmp_path):
    """A file holding both kinds of ending keeps the endings of the lines nobody touched. Only the
    replacement is written in the file's majority ending."""
    f = tmp_path / "mixed.txt"
    f.write_bytes(b"l1\r\nl2\r\nl3\n")
    fs.edit(str(f), "l2", "X2")
    assert _bytes(f) == b"l1\r\nX2\r\nl3\n", _bytes(f)


def test_an_edit_of_an_lf_file_sends_no_news_about_endings(tmp_path):
    """The disclosure is for the case the model cannot see. On a file with LF endings there is nothing to
    explain — and a multi-line insertion into one must stay LF, because assuming CRLF is how a tool
    quietly converts the file it was only asked to edit."""
    f = tmp_path / "unix.txt"
    f.write_bytes(b"one\ntwo\n")
    out = fs.edit(str(f), "two", "three\nfour")
    assert _bytes(f) == b"one\nthree\nfour\n", _bytes(f)
    assert "CRLF" not in out, out


def test_a_file_that_is_not_utf8_is_refused_rather_than_re_written(tmp_path):
    """`read` can show an undecodable byte as `\\xNN`, but showing is not round-tripping. Writing back
    what it showed would replace bytes the tool never read, and a tool that quietly re-encodes the parts
    of a file nobody looked at destroys more than it edits."""
    f = tmp_path / "cp.txt"
    f.write_bytes(b"caf\xe9 = 1\n")
    before = _bytes(f)
    out = fs.edit(str(f), "caf", "tea")
    assert _bytes(f) == before, "the tool rewrote bytes it could not decode"
    assert "not UTF-8" in out


# --- a refusal that can be repaired ---------------------------------------------------------------

def test_a_miss_names_the_line_where_the_first_line_of_the_request_lives(tmp_path):
    """Almost every miss is text remembered rather than copied, so the refusal is worth more as a
    pointer than as a verdict: it finds the first line of what was asked for and names the `read` that
    shows it. The number has to be the one `read` answers with, or the hint costs a turn instead of
    saving one."""
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\n    def f():\n        pass\n", encoding="utf-8")
    out = fs.edit(str(f), "\n\ndef f():\n  pass", "x")
    assert "line 4" in out, out
    assert "read path=" in out and "offset=2" in out, out
    assert "def f():" in fs.read(str(f), offset=2, count=10), "the page it points at does not show it"


def test_a_miss_whose_text_is_nowhere_says_so_rather_than_pointing(tmp_path):
    """The hint is only honest when the line really is in the file. Offering to show a line that does
    not exist is a dead end written as guidance, and the run follows it once before it stops believing
    the notices."""
    f = tmp_path / "a.txt"
    f.write_text("completely\ndifferent\n", encoding="utf-8")
    out = fs.edit(str(f), "the answer is 42", "x")
    assert "not one line of old_text occurs" in out, out
    assert "offset=" not in out, out


def test_repeated_text_is_matched_left_to_right_and_never_twice(tmp_path):
    """Overlapping occurrences are the case where a naive "find the next one" walks backwards into text
    it already replaced: `aa` in `aaaa` occurs at 0, 1 and 2, and replacing all three produces a file
    that is neither the old text nor the new one. Non-overlapping, left to right, is the only reading
    that survives a rename."""
    f = tmp_path / "a.txt"
    f.write_text("aaaa\n", encoding="utf-8")
    out = fs.edit(str(f), "aa", "b", every=True)
    assert f.read_text(encoding="utf-8") == "bb\n", f.read_text(encoding="utf-8")
    assert "2 place(s)" in out, out


def test_edit_names_the_tool_that_can_do_what_was_asked(tmp_path):
    """A refusal that only says "no" sends the run to try the same call again. Each of these names the
    act that would have worked, which is the difference between a turn and a loop."""
    assert "write creates one" in fs.edit(str(tmp_path / "gone.txt"), "a", "b")
    assert "edit changes a file" in fs.edit(str(tmp_path), "a", "b")
    assert "old_text is empty" in fs.edit(str(tmp_path / "x"), "", "b")


def test_an_insertion_is_expressed_as_a_span_not_as_a_position(tmp_path):
    """There is no line-number argument to `edit`, on purpose: a number addressed by a model drifts the
    moment the file changes, while the text it copied cannot be ambiguous about where it came from. So
    adding a line is an edit whose new text holds both lines, and that is the documented way."""
    f = tmp_path / "a.txt"
    f.write_text("first\nlast\n", encoding="utf-8")
    fs.edit(str(f), "last", "middle\nlast")
    assert f.read_text(encoding="utf-8") == "first\nmiddle\nlast\n"


# --- the contract the rest of the harness relies on ----------------------------------------------

def test_edit_and_grep_declare_which_way_they_reach_the_world():
    """The veto point judges a declaration and never a name, so the declaration is the contract: `edit`
    is a write judged against the write grants, `grep` a read judged against the read grants, and both
    name `path` as the argument to resolve before deciding."""
    assert declaration(fs.edit) == ("write", "path")
    assert declaration(fs.grep) == ("read", "path")


def test_the_policy_judges_an_edit_as_a_write_to_the_place_it_names(tmp_path):
    """Declared is not judged. With read grants and no write grants the same call the tool would have
    made successfully has to be refused, and the refusal has to name the flag that would have allowed
    it — that pairing is what lets a run stop trying instead of hunting for a workaround."""
    f = tmp_path / "a.txt"
    f.write_text("one\n", encoding="utf-8")
    access, address = declaration(fs.edit)
    verdict = Confine(read_roots=[str(tmp_path)], write_roots=[])(
        "edit", {"path": str(f), "old_text": "one", "new_text": "two"},
        access=access, address=address)
    assert not verdict.allowed, "a write was permitted by a read grant"
    assert "--write" in verdict.reason, verdict.reason


def test_a_grep_is_permitted_by_a_read_grant_it_was_given(tmp_path):
    """The other half of the pair above: a search inside a granted folder is a read, and a run that may
    read but not write must still be able to find things."""
    f = tmp_path / "a.txt"
    f.write_text("needle\n", encoding="utf-8")
    access, address = declaration(fs.grep)
    verdict = Confine(read_roots=[str(tmp_path)], write_roots=[])(
        "grep", {"pattern": "needle", "path": str(tmp_path)}, access=access, address=address)
    assert verdict.allowed, verdict.reason


def test_a_number_where_text_was_asked_is_refused_before_the_tool_runs(tmp_path):
    """`edit(path=0)` is the case the boundary exists for: `open(0)` is stdin, so a call that named no
    path at all would be judged against nothing and then act on something."""
    with pytest.raises(Exception, match="must be a string"):
        arguments_for(fs.edit, {"path": 0, "old_text": "a", "new_text": "b"})


def required(fn) -> dict:
    """The minimum call the schema accepts, so the type check under test is the one that fires."""
    return {name: "x" for name in spec_for(fn).parameters["required"]}


def test_the_boolean_arguments_the_schema_promises_are_checked_too(tmp_path):
    """`every` and `literal` are the first boolean tools in the core. A schema that says boolean may not
    be handed a list, or the declared type stops being enforced at the boundary for exactly the
    arguments nobody thought about."""
    for fn, key in ((fs.edit, "every"), (fs.grep, "literal")):
        spec = spec_for(fn)
        assert spec.parameters["properties"][key]["type"] == "boolean", spec.name
        with pytest.raises(Exception, match="must be a bool"):
            arguments_for(fn, {**required(fn), key: ["yes"]})


def test_the_edit_descriptions_are_written_for_the_model_that_reads_them():
    """The tool's declared description is the prose the model is sent, so the facts a model needs to
    use it without a trial call have to be in it: that the match must be unique, that a failed match
    costs nothing, and that each call sees the file the last one left."""
    text = spec_for(fs.edit).description
    for promise in ("must match exactly one place", "costs a refusal and no bytes",
                    "as it is now", "every=true"):
        assert promise in text, f"{promise!r} is not in the prose the model is sent:\n{text}"
