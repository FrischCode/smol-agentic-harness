"""`find`: the recursive name question, which no other tool can answer.

`ls` answers what is *in one folder*, `grep` answers what files *say*. Neither answers "which files are
called that", and a run that wants the path of a thing has to walk the tree a folder at a time — or
read every file to find one by content. These tests are about the promises that make the cheap answer
trustworthy: the wildcards mean what they say, the paths printed open in `read`, a page continues
exactly, and a folder the walk declined or could not read is named rather than silently absent.
"""
import os
import re

import pytest

from sah import ingest
from sah.tools import fs
from sah.confine import Confine
from sah.contract import arguments_for, declaration, spec_for


def tree(root):
    """A tree with the shape that distinguishes the rules: same-named files at three depths, a folder
    whose name a pattern could hit, and the two folders a walk should decline."""
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "deep").mkdir(parents=True)
    (root / ".git").mkdir(parents=True)
    (root / "__pycache__").mkdir(parents=True)
    (root / "AGENTS.md").write_text("top\n", encoding="utf-8")
    (root / "notes" / "a.md").write_text("a\n", encoding="utf-8")
    (root / "notes" / "deep" / "a.md").write_text("deep\n", encoding="utf-8")
    (root / "notes" / "deep" / "b.py").write_text("b\n", encoding="utf-8")
    (root / "notes" / "deep" / "test_deep.py").write_text("t\n", encoding="utf-8")
    (root / ".git" / "config").write_text("bare\n", encoding="utf-8")
    (root / "__pycache__" / "a.cpython-312.pyc").write_text("x\n", encoding="utf-8")


# --- the question only this tool answers -------------------------------------------------------------

def test_a_name_is_found_at_any_depth_without_naming_the_folders(tmp_path):
    tree(tmp_path)
    result = fs.find("a.md", tmp_path)
    assert result.startswith("# 2 match(es) for 'a.md' under")
    assert result.splitlines()[2] == os.path.join(str(tmp_path), "notes", "a.md")
    assert os.path.join("notes", "deep", "a.md") in result


def test_a_pattern_with_a_slash_matches_the_path_and_the_star_stops_at_a_folder(tmp_path):
    """`notes/*.md` is a promise about *one* level. `fnmatch` would let `*` cross separators and answer
    with the whole tree, which is the one glob result that sends a run reading files it never asked for.
    """
    tree(tmp_path)
    result = fs.find("notes/*.md", tmp_path)
    assert os.path.join("notes", "a.md") in result
    assert "deep" not in result


def test_a_pattern_is_anchored_at_the_end_so_the_short_form_finds_the_subtree_wherever_it_is(tmp_path):
    tree(tmp_path)
    (root := tmp_path / "everywhere").mkdir()
    (root / "notes").mkdir()
    (root / "notes" / "z.md").write_text("z\n", encoding="utf-8")
    result = fs.find("notes/*.md", tmp_path)
    assert os.path.join("everywhere", "notes", "z.md") in result
    assert os.path.join("notes", "deep", "a.md") not in result   # not a substring match


def test_double_star_crosses_a_folder_and_single_question_mark_is_one_character(tmp_path):
    tree(tmp_path)
    assert os.path.join("notes", "deep", "b.py") in fs.find("notes/**/*.py", tmp_path)
    assert os.path.join("notes", "a.md") in fs.find("notes/?.md", tmp_path)
    assert os.path.join("notes", "a.md") not in fs.find("notes/??.md", tmp_path)


def test_a_folder_matches_and_is_marked_so_a_run_can_tell_it_from_a_file(tmp_path):
    tree(tmp_path)
    result = fs.find("notes", tmp_path)
    assert os.path.join(str(tmp_path), "notes") + "/" in result
    assert not any(line.endswith("a.md/") for line in result.splitlines())


def test_names_match_with_the_case_they_have_on_every_platform(tmp_path):
    """Deliberate, and the reason it is `re` without `IGNORECASE` rather than `fnmatch`: the latter
    borrows the platform's case rule, so a pattern means one thing on this box and another on a Linux
    one. Asserted as *found or not found*, because the answer prints the file's own spelling — a check
    that the path is absent passes whether or not the matcher ignored case, since the path spells itself.
    (This box cannot show the other half: NTFS preserves case but is case-insensitive, so `a.md` and
    `A.md` cannot coexist, and a test that needed them would be green here and red on Linux.)
    """
    tree(tmp_path)
    assert os.path.join(str(tmp_path), "AGENTS.md") in fs.find("AGENTS.md", tmp_path)
    assert "Nothing under" in fs.find("agents.md", tmp_path), fs.find("agents.md", tmp_path)
    # The same rule as grep's name filter, so two tools filtering by name cannot disagree. The wording
    # is grep's own distinction: no file *has* that name, which is not the same answer as no file *says*
    # the thing searched for.
    assert "is named" in fs.grep("a", tmp_path, glob="*.MD")


# --- the paths it prints are the paths another tool accepts ----------------------------------------

def test_every_path_it_prints_is_the_one_read_opens(tmp_path):
    """Relative hits would be shorter and unreadable: `read` resolves a bare name against the process's
    own folder, not against the folder that was searched."""
    tree(tmp_path)
    for line in fs.find("**/*.md", tmp_path).splitlines()[2:]:
        if not line.startswith("#") and line.strip():
            assert os.path.isabs(line.rstrip("/"))
            assert "error" not in fs.read(line)


# --- paging ------------------------------------------------------------------------------------------

def test_a_page_continues_exactly_where_the_last_one_stopped(tmp_path):
    """A dropped entry between pages reads as 'that file does not exist', and a repeated one as a
    second file with the same name. Both are silent. Enough entries to hit the byte cap, since the body
    limits are sized to leave room for the notice under the result limits."""
    # Just over one full byte-cap page: the cap splits the page wherever the entry size lands (the temp
    # path differs between machines), and the ten-entry tail proves the continuation without the tail
    # itself hitting the cap.
    entry = len(str(tmp_path) + os.sep + "note-0000.txt") + 1
    n = ingest.BODY_BYTES // entry + 10
    for i in range(n):
        (tmp_path / f"note-{i:04d}.txt").write_text("x\n", encoding="utf-8")
    first = fs.find("*.txt", tmp_path, offset=1)
    head = [ln for ln in first.splitlines() if ln.endswith(".txt")]
    assert 0 < len(head) < n, len(head)
    assert "find offset=" in first
    rest = fs.find("*.txt", tmp_path, offset=int(re.search(r"find offset=(\d+)", first).group(1)))
    tail = [ln for ln in rest.splitlines() if ln.endswith(".txt")]
    assert tail[0] == os.path.join(str(tmp_path), "note-%04d.txt" % len(head))
    assert len(head) + len(tail) == n
    assert not set(head) & set(tail)


def test_a_whole_page_fits_the_limits_a_result_is_held_to(tmp_path):
    """The tool caps its own body at BODY_* so the `find offset=` notice survives the choke point; a
    page that reaches for MAX_* gets the notice clipped off on the way in."""
    for i in range(3000):
        (tmp_path / ("x" * 90) / f"n{i}.dat").parent.mkdir(exist_ok=True)
        (tmp_path / ("x" * 90) / f"n{i}.dat").write_text("x", encoding="utf-8")
    result = fs.find("*.dat", tmp_path)
    assert 0 < len(result.encode("utf-8")) <= ingest.MAX_BYTES
    assert result.count("\n") <= ingest.MAX_LINES


# --- what the answer says it did not look at ---------------------------------------------------------

def test_the_version_control_folder_is_not_searched_and_the_answer_says_so(tmp_path):
    tree(tmp_path)
    result = fs.find("*", tmp_path)
    assert "config" not in result
    assert ".git" in result and "__pycache__" in result


def test_a_folder_that_could_not_be_read_is_counted_rather_than_walked_past_in_silence(tmp_path, monkeypatch):
    """`os.walk` swallows a permission error by default. A run would then report that no such file
    exists, when the truth is that one folder was never opened."""
    real = os.walk

    def flaky(top, **kw):
        yield from real(top, **kw)
        kw["onerror"](PermissionError(13, "denied"))

    monkeypatch.setattr(fs.os, "walk", flaky)
    result = fs.find("*.md", tmp_path)
    assert "1 folder(s) could not be read" in result


def test_symlinks_are_not_followed_so_a_folder_linking_to_its_parent_is_not_a_tree_with_no_bottom(
        tmp_path, monkeypatch):
    seen = {}
    real = os.walk

    def watching(top, **kw):
        seen.update(kw)
        yield from real(top, **kw)

    monkeypatch.setattr(fs.os, "walk", watching)
    fs.find("*", tmp_path)
    assert seen["followlinks"] is False


def test_the_walk_stops_once_the_page_is_full(tmp_path, monkeypatch):
    """The output is the same whether or not the walk breaks out early, which makes the guard invisible
    to every other test here and easy to delete by accident. It is not cosmetic: a run that asked for one
    page should not pay to name every entry under a tree of a hundred thousand."""
    pulled = {"roots": 0}

    def wide(top, **kw):
        for _folder in range(60):
            pulled["roots"] += 1
            yield (os.path.join(top, f"d{_folder}"), [], [f"f{i}.dat" for i in range(200)])

    monkeypatch.setattr(fs.os, "walk", wide)
    result = fs.find("*.dat", tmp_path)
    assert "find offset=" in result
    assert pulled["roots"] < 60, f"walked all {pulled['roots']} roots for one page"


def test_the_line_cap_is_a_bound_of_its_own_and_not_a_consequence_of_the_byte_cap(tmp_path, monkeypatch):
    """Every entry this tool prints is an absolute path, so on a real drive the byte cap is what bites
    first and the line cap never fires — which makes it look like decoration and is only true while
    `MAX_BYTES`/`MAX_LINES` sit where they do. Loosening the byte cap for the test asks the line cap the
    question on its own: it holds, so a future change to the byte limit cannot quietly lift the line one.
    """
    for i in range(ingest.BODY_LINES + 400):
        (tmp_path / f"n{i}.dat").write_text("x", encoding="utf-8")
    monkeypatch.setattr(ingest, "BODY_BYTES", 10 ** 9)
    result = fs.find("*.dat", tmp_path)
    assert len([ln for ln in result.splitlines() if ln.endswith(".dat")]) == ingest.BODY_LINES
    assert "find offset=" in result


def test_an_answer_of_none_names_the_scope_it_searched_so_it_cannot_be_read_as_absence(tmp_path):
    tree(tmp_path)
    result = fs.find("*.rs", tmp_path)
    assert "Nothing under" in result and "entries looked at" in result
    assert "does not exist" not in result


# --- refusals ----------------------------------------------------------------------------------------

def test_an_empty_pattern_is_refused_because_it_would_match_everything(tmp_path):
    """Asserted as a prefix, not a substring: the refusal names the folder searched, and pytest names a
    test's temporary folder after the test, so a check for the word 'empty' anywhere in the answer is
    satisfied by the path of a test whose name contains it. A refusal's identity is how it starts.
    """
    tree(tmp_path)
    assert fs.find("", tmp_path).startswith("pattern is empty"), fs.find("", tmp_path)


def test_a_folder_that_is_not_there_says_so_in_words_no_other_answer_uses(tmp_path):
    assert "does not exist" in fs.find("*.py", tmp_path / "nope")
    tree(tmp_path)
    assert "not a folder" in fs.find("*.py", tmp_path / "AGENTS.md")


def test_a_page_asked_for_past_the_end_says_where_the_entries_ran_out(tmp_path):
    tree(tmp_path)
    answer = fs.find("*.md", tmp_path, offset=99)
    assert "No entries at or after" in answer and "3 match(es) before it" in answer, answer


def test_a_search_that_found_nothing_is_not_reported_as_a_page_that_ran_out(tmp_path):
    """The two empties are different facts. At the first page, 'no entries at or after offset 1' would
    read as a paging mistake and send a run retrying the call, when the truth is that nothing matched."""
    tree(tmp_path)
    assert "Nothing under" in fs.find("*.rs", tmp_path)


# --- the boundary and the promise --------------------------------------------------------------------

def test_it_declares_a_read_and_the_policy_judges_the_folder_it_names(tmp_path):
    """The declaration is not the verdict: the same call is permitted by a read grant over the folder and
    refused outside it, naming the flag that would have allowed it."""
    assert fs.find.access == "read" and fs.find.address == "path"
    access, address = declaration(fs.find)
    tree(tmp_path)
    inside = Confine(read_roots=[str(tmp_path)], write_roots=[])(
        "find", {"pattern": "*.md", "path": str(tmp_path)}, access=access, address=address)
    assert inside.allowed, inside.reason
    outside = Confine(read_roots=[str(tmp_path)], write_roots=[])(
        "find", {"pattern": "*.md", "path": str(tmp_path.parent)}, access=access, address=address)
    assert not outside.allowed and "--grant" in outside.reason, outside.reason


def test_a_quoted_page_number_is_taken_and_an_invented_one_is_refused_by_name():
    """Models quote numbers constantly, so `offset="10"` is coerced exactly (digits only) rather than
    refused for a typo of form. A value `int()` would have to guess at is refused naming the argument —
    the same rule every other page-carrying tool keeps, and the reason `find` never parses a number.
    """
    sent = arguments_for(fs.find, {"pattern": "*.md", "offset": "10"})
    assert sent["offset"] == 10 and isinstance(sent["offset"], int)
    for bad in ("5.0", " 5", "abc", "0x10"):
        with pytest.raises(Exception, match="find.offset must be an int"):
            arguments_for(fs.find, {"pattern": "*.md", "offset": bad})


def test_the_rules_a_run_cannot_guess_are_in_the_text_the_model_is_sent():
    """The declared `description` and each `Param` description are the text the model is sent — a
    docstring is maintainer prose, so the wildcard and case rules have to sit in the `pattern` field's
    description or they exist only for whoever reads the source. This test is the guard against writing
    the next one into the docstring, where it silently does nothing.
    """
    spec = spec_for(fs.find)
    pattern = spec.parameters["properties"]["pattern"]["description"]
    for promise in ("**", "folder boundary", "Case matters", "exactly"):
        assert promise in pattern, promise
    for promise in ("find offset=", ".git", "absolute paths"):
        assert promise in spec.description, promise
    assert "pattern" in spec.parameters["required"]


def test_the_description_says_what_the_run_would_otherwise_guess_wrong():
    """Kept as the paragraph-1 half of the promise: what the tool is for, in the words a run picks it by.
    """
    text = spec_for(fs.find).description
    for promise in ("recursively", "reads no file contents", "offset"):
        assert promise in text, promise
