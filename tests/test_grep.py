"""Searching a granted folder without a shell.

A run with no `--exec` can list a folder and read a file, but it cannot ask *which* file mentions
something: `mem_grep` asks that only of the memory tree, and asking the shell is the permission the run
does not have. So it lists, then reads, and spends a page of context per file on a question that has a
one-line answer.

The shape this tool owes is the one the rest of `sah.tools.fs` already keeps: stream rather than whole-load,
cap the *body* below the result limits so the disclosure survives the trip to the choke point, and name
a continuation the run can actually call — `grep offset=` between matches, `read ... start_char=`
inside one over-long line. And because "no matches" is a claim about a set of files, every file the
search declined to open is named in the answer.
"""
import os
import re

from sah import ingest
from sah.tools import fs
from sah.agent import Agent, Verdict
from sah.model import Caps, Fake, turn_text, turn_tool
from sah.record import Log


def _allow(name, args, **_):
    return Verdict(True)


def test_grep_reports_path_line_and_text_for_every_match(tmp_path):
    (tmp_path / "b.txt").write_text("nothing\nhere is one\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("needle at the top\n", encoding="utf-8")
    out = fs.grep("needle", str(tmp_path))
    hits = [ln for ln in out.split("\n") if ln.startswith(str(tmp_path))]
    assert hits == [f"{os.path.join(str(tmp_path), 'a.txt')}:1: needle at the top"], hits
    assert "# 1 match(es)" in out, out


def test_the_paths_it_prints_are_the_ones_read_accepts(tmp_path):
    """A hit the run cannot open is a citation to nowhere. Relative-to-the-searched-folder paths are
    shorter, but `read` resolves relative to the working directory, so the short form would be a hint
    that gets refused — and a hint the run cannot follow is worse than no hint."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("buried needle\n", encoding="utf-8")
    out = fs.grep("needle", str(tmp_path))
    shown = re.findall(r"^(.*?):\d+: ", out, re.M)[0]
    assert "needle" in fs.read(shown), "the result named a path that does not read back"


def test_grep_pages_by_match_and_the_offset_it_names_continues_it(tmp_path):
    """The property is not that a page is short but that the next page picks up where it stopped: no
    match shown twice, none skipped. Only asking for the continuation can tell those apart."""
    f = tmp_path / "many.txt"
    f.write_text("".join(f"line {i} has a needle {'x' * 90}\n" for i in range(1, 700)),
                 encoding="utf-8")
    first = fs.grep("needle", str(f))
    assert "grep offset=" in first, "a page this big has to say it is a page"
    nxt = int(re.search(r"grep offset=(\d+)", first).group(1))
    shown = [ln for ln in first.split("\n") if ln.startswith(str(f))]
    rest = fs.grep("needle", str(f), offset=nxt)
    later = [ln for ln in rest.split("\n") if ln.startswith(str(f))]
    assert later, "the continuation page carried nothing"
    assert later[0] not in shown, "the continuation re-showed the last match of the page"
    assert _line_no(later[0]) == _line_no(shown[-1]) + 1, "the page it named does not resume the file"


def _line_no(hit: str) -> int:
    """The line number a result line was printed with (`path:line: text`)."""
    return int(re.match(r"^(.*?):(\d+): ", hit).group(2))


def test_a_grep_page_fits_the_result_it_is_sent_into(fits_a_result, tmp_path):
    """The shared producer contract, applied to the tool most likely to fill a page.

    Cap the body to the *result* limit and then append a notice and the result arrives over the limit,
    is head-clipped — losing the `grep offset=` line the cap exists to provide — and is spilled to a
    file, which for a search means copying a folder's worth of matches to disk to show the model less
    of them."""
    f = tmp_path / "many.txt"
    f.write_text("".join(f"{'x' * 70} needle {i}\n" for i in range(3000)), encoding="utf-8")
    out = fits_a_result(fs.grep("needle", str(f)))
    assert re.search(r"grep offset=\d+\]$", out.rstrip()), out[-200:]


def test_a_grep_page_reaches_the_model_with_its_pointer_intact(tmp_path):
    """The two bounds live in different modules, so only a run through the loop can see the second one:
    the tool's own cap has to leave room for the notice, or the choke point clips it off on the way in."""
    f = tmp_path / "many.txt"
    f.write_text("".join(f"{'x' * 70} needle {i}\n" for i in range(3000)), encoding="utf-8")
    agent = Agent(backend=Fake([turn_tool("c1", "grep", {"pattern": "needle", "path": str(f)}),
                               turn_text("done")],
                              caps=Caps(context_tokens=100, model="t", supports_vision=False)),
                  tools={"grep": fs.grep}, log=Log(tmp_path / "r.jsonl"), policy=_allow,
                  max_turns=6, run_id="r", root=tmp_path)
    agent.run("go")
    res = next(s for s in agent.log.steps() if s["kind"] == "result")
    assert not res.get("truncated"), "the choke point re-clipped a result the tool had already capped"
    assert "spilled" not in res, res["content"][-200:]
    assert "grep offset=" in res["content"], res["content"][-200:]


def test_grep_tells_a_regular_expression_from_plain_text(tmp_path):
    """The pattern is regex by default because that is what a search means most of the time, and
    `literal=true` exists because a model searching for `a.b` or a path means the characters."""
    f = tmp_path / "a.txt"
    f.write_text("axb\na.b\n", encoding="utf-8")
    assert "axb" in fs.grep("a.b", str(f))
    literal = fs.grep("a.b", str(f), literal=True)
    assert "a.b" in literal and "axb" not in literal, literal


def test_a_pattern_that_does_not_compile_names_the_way_to_mean_it_literally(tmp_path):
    """A regex error is the model's mistake, not the tool's failure, so it comes back as an answer with
    the fix in it rather than as a traceback the run can only re-read."""
    f = tmp_path / "a.txt"
    f.write_text("a( paren\n", encoding="utf-8")
    out = fs.grep("a(", str(f))
    assert "not a regular expression" in out and "literal=true" in out, out


def test_ignore_case_is_an_argument_not_a_default(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Needle\n", encoding="utf-8")
    assert "No matches" in fs.grep("needle", str(f))
    assert "Needle" in fs.grep("needle", str(f), ignore_case=True)


def test_grep_narrows_by_file_name_and_says_when_no_name_matches(tmp_path):
    (tmp_path / "a.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("target\n", encoding="utf-8")
    assert "a.py" in fs.grep("target", str(tmp_path), glob="*.py")
    assert ".txt" not in fs.grep("target", str(tmp_path), glob="*.py")
    assert "is named" in fs.grep("target", str(tmp_path), glob="*.md")


def test_grep_shows_the_lines_around_a_match_once_even_when_matches_are_near(tmp_path):
    """Two matches a line apart must not print the line between them twice — once as trailing context
    for the first, once as leading for the second. A result that repeats itself is one the model has to
    deduplicate by hand, and a paged result cannot do that at all."""
    f = tmp_path / "a.txt"
    f.write_text("above\nX\nX\nbelow\n", encoding="utf-8")
    bodies = [ln for ln in fs.grep("X", str(f), context=1).split("\n") if ln.startswith(str(f))]
    p = os.path.join(str(tmp_path), "a.txt")
    assert bodies == [f"{p}-1- above", f"{p}:2: X", f"{p}:3: X", f"{p}-4- below"], bodies


def test_grep_holds_its_context_to_the_number_it_promised(tmp_path):
    """The lines before a match are buffered until the match arrives, so an unbounded `context` is an
    unbounded buffer. The number the schema promises and the number the tool holds must be one number."""
    f = tmp_path / "a.txt"
    f.write_text("".join(f"l{i}\n" for i in range(1, 40)) + "MATCH\n" +
                 "".join(f"m{i}\n" for i in range(40)), encoding="utf-8")
    bodies = [ln for ln in fs.grep("MATCH", str(f), context=1000).split("\n") if ln.startswith(str(f))]
    assert len(bodies) == 21, len(bodies)          # the match, 10 before, 10 after


def test_grep_does_not_open_a_git_folder_and_says_that_it_declined(tmp_path):
    """A search that quietly walked less than it claimed lets a run conclude the fact was never written
    when the truth is that it was never looked at — so the folders it would not enter are named."""
    (tmp_path / "a.txt").write_text("nothing here\n", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("target in git\n", encoding="utf-8")
    out = fs.grep("target", str(tmp_path))
    assert "config" not in out, "the object store was searched"
    assert ".git/ not searched" in out, out


def test_a_file_with_no_text_in_it_is_skipped_and_named(tmp_path):
    """A NUL byte is the marker, and the skip is declared: `2 file(s) looked at` is a different answer
    from a silent `2 file(s) searched` when one of them was a compiled artefact."""
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "blob.dat").write_bytes(b"\x00\x01needle\n")
    out = fs.grep("needle", str(tmp_path))
    assert "held no text and were skipped" in out, out
    assert "2 file(s) looked at" in out, out


def test_a_file_that_cannot_be_opened_is_counted_rather_than_swallowed(tmp_path, monkeypatch):
    """Stubbed at the one seam that opens a file, because a file the run cannot read is ordinary on a
    real folder and the answer still has to admit it did not look inside it."""
    real = fs._is_binary

    def stub(file):
        if file.endswith("locked.txt"):
            raise PermissionError("denied")
        return real(file)

    monkeypatch.setattr(fs, "_is_binary", stub)
    (tmp_path / "locked.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("other\n", encoding="utf-8")
    out = fs.grep("needle", str(tmp_path))
    assert "1 file(s) could not be opened" in out, out
    assert "No matches" in out, out


def test_an_over_long_hit_names_the_read_that_shows_the_rest(tmp_path):
    """One minified line is a fact about the file, not a licence to spend the whole result on it. The
    held-back half has to be reachable by a call the run can already make — `start_char` on `read`, at
    the line number the hit just printed — because a hint to shell out is refused by a run without
    `--exec`, and on a cmd.exe box there is no `sed` to refuse."""
    f = tmp_path / "min.js"
    f.write_text("var x=" + "y" * 4000 + ";needle=1\n", encoding="utf-8")
    out = fs.grep("needle", str(f))
    assert f"{ingest.GREP_LINE} are shown" in out, out
    assert f"read path={str(f)!r} offset=1 start_char={ingest.GREP_LINE}" in out, out
    assert len(out) < 2000, "one hit cost the whole result"
    assert "y" * 50 in fs.read(str(f), offset=1, start_char=ingest.GREP_LINE)


def test_a_match_at_the_end_of_a_file_is_reported_with_whatever_context_it_has(tmp_path):
    """A block is only complete once its trailing context has run out, and at the end of a file it runs
    out without ever reaching the count. A generator that yields only on a full block loses the last
    match of every file it searches — which reads as the file not containing it."""
    f = tmp_path / "tail.txt"
    f.write_text("one\ntwo\nthree\nthree\n", encoding="utf-8")
    bodies = [ln for ln in fs.grep("three", str(f), context=2).split("\n") if ln.startswith(str(f))]
    assert any(ln.endswith(":3") or ":3: three" in ln for ln in bodies), bodies
    assert any(":4: three" in ln for ln in bodies), "the file's last match was dropped"


def test_grep_searches_a_file_named_directly(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    assert rf"{f}:2: two" in fs.grep("two", str(f))


def test_grep_searches_a_file_that_is_not_utf8_rather_than_skipping_it(tmp_path):
    """A file in another code page is searchable text with a wrinkle, not a binary. Declining to open it
    would let the tool answer "no matches" about a file it never looked at, which is the one answer a
    search must never be able to give honestly."""
    f = tmp_path / "latin.txt"
    f.write_bytes(b"caf\xe9 = 1\nneedle = 2\n")
    assert "needle = 2" in fs.grep("needle", str(f))


def test_grep_refuses_an_empty_pattern_and_a_path_that_is_not_there(tmp_path):
    """An empty pattern matches every line, so it is a mistake rather than a search; and a misspelled
    folder must not come back as the same words a genuine no-match uses."""
    assert "pattern is empty" in fs.grep("", str(tmp_path))
    out = fs.grep("x", str(tmp_path / "gone"))
    assert "does not exist" in out and "No matches" not in out, out


def test_grep_walks_a_folder_in_a_fixed_order_so_pages_agree(tmp_path):
    """Paging is only meaningful if the second page sees the same tree in the same order. The operating
    system promises nothing about the order it hands names back, and a page built on one order and a
    continuation on another would skip matches in silence."""
    for name in ("z.txt", "a.txt", "m.txt"):
        (tmp_path / name).write_text("needle\n", encoding="utf-8")
    hits = [ln for ln in fs.grep("needle", str(tmp_path)).split("\n") if ln.startswith(str(tmp_path))]
    order = [re.match(r"^(.*?):1: needle", ln).group(1) for ln in hits]
    assert [p.rpartition(os.sep)[2] for p in order] == ["a.txt", "m.txt", "z.txt"], order


def test_the_search_description_says_what_the_search_will_not_do():
    """The tool's declared description is the prose the model is sent, so the facts that change a plan
    belong in it rather than in a manual nobody reads: that the dialect is regular expressions
    (with a way to mean plain text), and that the paths it prints can be read back as they stand."""
    from sah.contract import spec_for
    text = spec_for(fs.grep).description
    for promise in ("regular-expression", "glob", "absolute"):
        assert promise in text, f"{promise!r} is missing from the prose the model is sent:\n{text}"
