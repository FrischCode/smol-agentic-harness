"""The file tools' paging contract, driven through the loop.

A capped read is not merely a shortened read: it carries the one line that tells the model what to do
next (`read with offset=N`). The bug these tests hold shut is that the tool capped its body to the
full result limit and *then* appended that line, so the result arrived at the choke point over the
limit, got re-clipped — losing the hint — and was spilled to a file. The README's promise that a file
is never copied or spilled, only paged, was false at exactly the size where paging matters.

These go through an `Agent` rather than calling `fs.read` directly, because the two bounds in question
live in different modules; a unit test of `fs.read` alone cannot see the second one.
"""
import re

from sah import contract, ingest
from sah.tools import fs
from sah.agent import Agent, Verdict
from sah.model import Caps, Fake, turn_text, turn_tool
from sah.record import Log


def _allow(name, args, **_):
    return Verdict(True)


def _run(tmp_path, call, tools, run="r1", vision=False):
    """One scripted tool call, then a closing prose turn; returns the `result` step. `turn_tool`
    builds a whole turn (call + stop), so the script is [that turn, the closing turn].

    Each run gets its own log file: two runs sharing `tmp_path` would share a record, and reading the
    first `result` back would silently return the earlier run's answer."""
    agent = Agent(backend=Fake([call, turn_text("done")],
                               caps=Caps(context_tokens=100, model="t", supports_vision=vision)),
                  tools=tools, log=Log(tmp_path / f"{run}.jsonl"), policy=_allow, max_turns=6,
                  run_id=run, root=tmp_path)
    agent.run("go")
    steps = agent.log.steps()
    bad = [s for s in steps if s["kind"] == "error"]
    assert not bad, f"the run recorded an error instead of a result: {bad}"
    return next(s for s in steps if s["kind"] == "result")


def test_a_read_capped_by_lines_arrives_with_its_notice_intact(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(big)}), {"read": fs.read})
    assert not res.get("truncated"), "the choke point re-clipped a result the tool had already capped"
    assert "spilled" not in res, "a file the model can page must never be spilled"
    assert res["content"].rstrip().endswith(f"read with offset={ingest.BODY_LINES + 1}]"), \
        res["content"][-200:]
    # The exact wording of that line is pinned by test_an_interrupted_read_says_more_remain_once; this
    # test only cares that the pointer survived the trip to the model at all.


def test_a_read_capped_by_bytes_keeps_the_byte_disclosure(tmp_path):
    """The byte border is where the reader most needs telling: it stopped at a size ceiling, not at
    the end of the file.

    The line width is chosen, not casual: the byte cap only binds before the line cap above ~33 bytes
    a line, and the leftover slack is bounded by one line. At 40 bytes the body lands within 40 bytes
    of the ceiling, so an appended notice genuinely pushes the result over — which is what makes this
    test fail when the reserve is deleted. With kilobyte-sized lines the slack swallows the notice and
    the test passes no matter what, which is how it was first written."""
    wide = tmp_path / "wide.txt"
    wide.write_text("".join(f"{i:039d}\n" for i in range(3000)), encoding="utf-8")   # ~120 KB
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(wide)}), {"read": fs.read})
    assert not res.get("truncated") and "spilled" not in res
    assert "byte limit" in res["content"] and "offset=" in res["content"], res["content"][-200:]


def test_the_offset_a_capped_read_names_actually_continues_the_file(tmp_path):
    """An offset hint one line off is worse than none: the model either re-reads a line it has or
    silently loses one, and neither is visible in the text it receives."""
    big = tmp_path / "big.txt"
    lines = [f"line {i}" for i in range(5000)]
    big.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(big)}), {"read": fs.read})
    shown = [ln for ln in res["content"].split("\n") if ln.startswith("line ")]
    nxt = int(re.search(r"offset=(\d+)", res["content"]).group(1))
    assert nxt == len(shown) + 1, f"the read showed {len(shown)} lines and named offset {nxt}"
    again = _run(tmp_path, turn_tool("c2", "read", {"path": str(big), "offset": nxt}),
                 {"read": fs.read}, run="r2")
    first = again["content"].split("\n")[0]
    assert first == lines[nxt - 1], f"offset {nxt} returned {first!r}, not {lines[nxt - 1]!r}"


def test_a_listing_capped_by_entries_keeps_its_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "BODY_LINES", 3)       # a folder of 2000 files is not a fast test
    for i in range(10):
        (tmp_path / f"f{i}").write_text("x", encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "ls", {"path": str(tmp_path)}), {"ls": fs.ls})
    assert not res.get("truncated") and "spilled" not in res
    # The pointer is the thing that must survive the choke point: a listing that says entries were
    # withheld but not how to reach them is a dead end dressed as a disclosure.
    assert "continue with ls offset=4" in res["content"], res["content"][-200:]


def test_a_count_larger_than_a_result_can_carry_is_capped_not_clipped(tmp_path, monkeypatch):
    """`count=500` on a file that cannot show 500 lines is the same hazard wearing a user request: the
    body came back at the count, the hint pushed it over, and the hint is what got cut. So the count is
    clamped to what fits, and the notice says what actually stopped the read."""
    monkeypatch.setattr(ingest, "BODY_LINES", 4)
    big = tmp_path / "big.txt"
    big.write_text("".join(f"line {i}\n" for i in range(500)), encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(big), "count": 500}), {"read": fs.read})
    assert not res.get("truncated") and "spilled" not in res
    tail = res["content"].rsplit("\n\n", 1)[-1]
    assert tail == f"[showing {ingest.BODY_LINES} and more remain. " \
                   f"read with offset={ingest.BODY_LINES + 1}]", tail


def test_a_count_that_genuinely_binds_says_so(tmp_path, monkeypatch):
    """The mirror of the test above: when the caller's count really is the reason the read stopped,
    only the caller knows that, so the notice has to say it. Together the two pin that the two stops
    stay distinguishable however the checks are ordered."""
    monkeypatch.setattr(ingest, "BODY_LINES", 4)
    big = tmp_path / "big.txt"
    big.write_text("".join(f"line {i}\n" for i in range(500)), encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(big), "count": 2}), {"read": fs.read})
    tail = res["content"].rsplit("\n\n", 1)[-1]
    assert tail == "[showing 2 and more remain. stopped at your count; read more with offset=3]", tail


def test_an_offset_past_the_end_of_a_folder_names_the_size_that_is_there(tmp_path):
    """A folder of 4000 entries listed past its end must not answer "is empty": that is a false fact
    about the folder, and the model would stop looking."""
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    out = fs.ls(str(tmp_path), offset=9)
    assert "is empty" not in out and "2 entries" in out, out


def test_a_file_that_is_not_utf8_shows_its_bytes_instead_of_question_marks(tmp_path):
    """A file written in Windows-1252 -- still what `open()` produces by default on this box -- read back
    as text used to answer with U+FFFD, which reads like the file is damaged. The escape says what is
    actually there, and a model that can see `0xE9` can name it (an accented e), ask for it as UTF-8, or
    decide it does not care. A character that stands in for "I could not read this" is not information.
    """
    (tmp_path / "latin.txt").write_bytes("caf\u00e9\n".encode("cp1252"))
    out = fs.read(str(tmp_path / "latin.txt"))
    assert "\ufffd" not in out, f"the byte was smoothed away: {out!r}"
    assert "\\xe9" in out, f"the byte was not shown: {out!r}"


def test_the_reserve_is_a_reserve():
    """A body limit equal to the result limit reserves nothing, and every test above could pass while
    the bug stayed shut. The pair of numbers is the whole mechanism, so it is asserted as one."""
    assert ingest.BODY_LINES < ingest.MAX_LINES
    assert ingest.BODY_BYTES < ingest.MAX_BYTES


def test_an_interrupted_read_says_more_remain_once(tmp_path):
    """`ingest.notice()` already says "and more remain", and fs's own pointer used to open with "more
    remains;" as well. Live output, verbatim: `[showing 1996 and more remain. more remains; read with
    offset=1997]`. Both halves are correct alone and every test here asserted substrings, so the seam
    was invisible -- which is why this one asserts the whole bracketed line."""
    big = tmp_path / "big.txt"
    big.write_text("".join(f"line {i}\n" for i in range(5000)), encoding="utf-8")
    res = _run(tmp_path, turn_tool("c1", "read", {"path": str(big)}), {"read": fs.read})
    tail = res["content"].rsplit("\n\n", 1)[-1]
    expected = f"[showing {ingest.BODY_LINES} and more remain. read with offset={ingest.BODY_LINES + 1}]"
    assert tail == expected, tail


# --- a hint the run can actually follow ---------------------------------------------------------
#
# These call the tools directly. The property is the tool's own contract -- what it points at next --
# not the interaction of two modules' limits, which is what the `_run` harness above is for.


def test_an_over_cap_line_is_windowed_in_place_of_a_shell_command(tmp_path, monkeypatch):
    """A line bigger than the byte cap used to be answered with `sed -n 'Np' path | head -c N`.

    That hint fails twice: a run without `--exec` is refused, and on a box whose shell is cmd.exe there is
    no `sed` -- so the tool that just *succeeded* in reading the file sent the model to a tool it could not
    use, to do something the first tool could have done itself. The window is the tool's own capability:
    page the line with `start_char` and the disclosure stays actionable on any machine, with any grants.

    The assertion is the whole line reassembled from its windows, because "it mentions start_char" would
    pass for a hint that named an offset which did not continue the text.
    """
    monkeypatch.setattr(ingest, "MAX_BYTES", 200)
    monkeypatch.setattr(ingest, "BODY_BYTES", 150)
    # Multi-byte characters on purpose: a window cut at a byte count splits them, and the replacement
    # character in the middle of a page would let every test here pass on a file that no longer
    # reassembles to itself.
    text = "".join(("\u00e9" if i % 7 == 0 else "\U0001f600" if i % 11 == 0 else str(i % 10))
                   for i in range(500))
    (tmp_path / "oneline.txt").write_text(text, encoding="utf-8")

    seen, offset = "", 0
    for _ in range(50):
        page = fs.read(str(tmp_path / "oneline.txt"), offset=1, start_char=offset)
        assert "sed" not in page and "bash" not in page, f"a shell hint came back: {page[:120]}"
        body, _sep, _notice = page.partition("\n")
        assert body, "a window came back empty"
        seen += body
        nxt = re.search(r"start_char=(\d+)", page)
        if not nxt:
            break
        offset = int(nxt.group(1))
    assert seen == text, f"windows gave {len(seen)} chars of a {len(text)}-char line"
    assert offset < len(text), "the last window claimed more to come after the line ended"


def test_a_start_char_past_the_end_of_the_line_says_so(tmp_path, monkeypatch):
    """An offset beyond the end must not read as an empty success: the model would conclude the file is
    short. It names the length it found."""
    monkeypatch.setattr(ingest, "MAX_BYTES", 200)
    monkeypatch.setattr(ingest, "BODY_BYTES", 150)
    (tmp_path / "short.txt").write_text("x" * 300, encoding="utf-8")
    out = fs.read(str(tmp_path / "short.txt"), offset=1, start_char=10_000)
    assert "300" in out and "start_char" in out, out


def test_a_huge_listing_pages_from_the_entry_the_last_one_stopped_at(tmp_path, monkeypatch):
    """A capped listing used to say `use bash: ls -la path` -- a suggestion to run a command that returns
    a *larger* listing than the tool just declined, and which a run without exec cannot run at all.

    So `ls` pages, like `read`: every entry is seen exactly once across the pages, and the offset it
    names is the one that continues the folder rather than repeating or skipping an entry.
    """
    monkeypatch.setattr(ingest, "BODY_LINES", 3)
    names = [f"n{i}.txt" for i in range(11)]
    for n in names:
        (tmp_path / n).write_text("x", encoding="utf-8")

    seen, offset, pages = [], 1, 0
    while True:
        out = fs.ls(str(tmp_path), offset=offset)
        assert "bash" not in out and "ls -la" not in out, f"a shell hint came back: {out[:120]}"
        shown = [ln.split(" (")[0].rstrip("/") for ln in out.splitlines() if ln.startswith("n")]
        seen += shown
        nxt = re.search(r"offset=(\d+)", out)
        pages += 1
        if not nxt:
            break
        assert int(nxt.group(1)) == offset + len(shown), \
            f"the page named offset={nxt.group(1)} after showing {len(shown)} of {names}"
        offset = int(nxt.group(1))
    assert sorted(seen) == sorted(names), f"across {pages} pages: {sorted(seen)}"
    assert pages > 1, "the cap never bound, so the paging was never exercised"


def test_a_last_listing_page_invents_no_continuation(tmp_path, monkeypatch):
    """A page that reached the end must not name an offset, or the model loops asking for entries that
    do not exist -- the paging claim has to be false only when there genuinely is nothing left."""
    monkeypatch.setattr(ingest, "BODY_LINES", 20)
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    out = fs.ls(str(tmp_path))
    assert "offset=" not in out, out


def test_a_page_cut_by_a_final_giant_line_still_says_more_remains(tmp_path):
    """The peek for "is there more" runs *after* the loop has already consumed the line that broke it, so
    when that line is the file's last, the peek reads EOF and the page comes back short, silent, and
    looking complete. A cut that does not say it cut is the one thing this module's own docstring
    forbids, and it lands on the most ordinary large file there is: text with no trailing newline."""
    big = "x" * (ingest.MAX_BYTES * 3)
    path = tmp_path / "onegiant.txt"
    path.write_text("header\nline two\n\n[untrusted]\n" + big, encoding="utf-8")
    got = fs.read(path=str(path), offset=1)

    assert len(got) < len(big), "the giant line was dumped into the result"
    assert "offset=" in got, f"a page was cut with no continuation named: {got[-160:]!r}"


def test_a_giant_next_line_names_the_argument_that_pages_inside_it(tmp_path):
    """`offset` moves between lines, and this file has one line left — so a pointer that offers only
    `offset` sends the run to a call that refuses again. `start_char` is the argument that windows inside
    a line, and the line's size is already known here, so the disclosure can say so at the first cut
    rather than after a wasted call."""
    big = "x" * (ingest.MAX_BYTES * 3)
    path = tmp_path / "onegiant2.txt"
    path.write_text("header\nline two\n\n[untrusted]\n" + big, encoding="utf-8")
    got = fs.read(path=str(path), offset=1)
    assert "start_char" in got, f"no way into the next line was offered: {got[-160:]!r}"


def test_a_picture_is_read_as_a_picture_and_not_as_escaped_bytes(tmp_path, picture):
    """A PNG is not text that happens to be unkind. Read as UTF-8 with `backslashreplace` it arrives as
    thousands of `\\xNN` escapes, which is a faithful rendering of nothing a model can look at. So a
    picture is named in the text (its path, kind and size) and travels as the picture itself."""
    result = _run(tmp_path, turn_tool("c1", "read", {"path": str(picture)}), {"read": fs.read},
                  vision=True)
    assert "green.png" in result["content"] and "image/png" in result["content"], result["content"]
    assert "\\x89" not in result["content"], f"bytes dumped as escapes: {result['content'][:80]!r}"
    assert result.get("images") == [str(picture)], result


def test_a_picture_is_told_by_its_bytes_and_not_by_its_name(tmp_path):
    """An extension is a claim; the bytes are the fact. Sniffing the name would let a text file called
    `lie.png` be attached as a picture, and would let a PNG saved as `.dat` be dumped as escapes."""
    text = tmp_path / "lie.png"
    text.write_text("just a note about a green square\n", encoding="utf-8")
    result = _run(tmp_path, turn_tool("c1", "read", {"path": str(text)}), {"read": fs.read},
                  vision=True)
    assert "just a note" in result["content"] and not result.get("images"), result


def test_the_path_tools_say_whose_paths_they_take():
    r"""Measured in a live run: the model listed a folder with `bash` (`ls -d /c/Users/*/Desktop`, which
    worked), then handed the same spelling to `ls` and got `C:\Users\you\Desktop is not a folder`.
    Neither tool is wrong and neither is the shell — the model speaks one dialect and uses it everywhere,
    so the description is where the other dialect has to be named. Asserted rather than trusted because
    this prose is the whole of the warning, and a reworded description drops it silently."""
    for fn in (fs.read, fs.ls):
        spec = contract.spec_for(fn)
        assert "operating system" in spec.description, spec.name
        assert "/c/" in spec.description, f"{spec.name} does not name the spelling that fails"
