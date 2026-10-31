"""The one place that bounds a result, tested. The whole point is that "how big does a result get"
has a single answer, so the cases are the two limits, which one fires, the never-a-partial-line
rule, the one-huge-line fallback, and the byte-border disclosure."""
from sah import ingest


def _text(n_lines, width=10):
    return "\n".join(f"{i} " + "x" * (width - 2) for i in range(n_lines))


def test_under_both_limits_it_is_returned_whole_untruncated():
    shown, meta = ingest.clip_head(_text(50))
    assert not meta["truncated"] and shown.count("\n") == 49


def test_the_line_limit_firest_first_and_keeps_whole_lines():
    shown, meta = ingest.clip_head(_text(5000), max_lines=100, max_bytes=10_000_000)
    assert meta["truncated"] and meta["truncated_by"] == "lines"
    assert len(shown.splitlines()) == 100          # whole lines only, never a torn line
    assert shown.splitlines()[-1].endswith("x")    # not cut mid-line


def test_the_byte_limit_fires_and_says_so_at_a_line_boundary():
    shown, meta = ingest.clip_head(_text(5000, width=100), max_lines=9999, max_bytes=4096)
    assert meta["truncated"] and meta["truncated_by"] == "bytes"
    assert meta["shown_bytes"] <= 4096
    # the preview actually fits the byte cap (not just what meta claims) and no line is torn
    assert len(shown.encode("utf-8")) <= 4096
    assert all(ln.count("x") == 98 for ln in shown.splitlines()) and shown.splitlines()
    # #4: the byte-border truncation is stated, not silent
    note = ingest.notice(meta)
    assert "byte limit" in note and "line boundary" in note


def test_a_single_line_over_the_byte_cap_shows_nothing_and_names_a_fallback():
    huge = "y" * (ingest.MAX_BYTES + 100)
    shown, meta = ingest.clip_head(huge)
    assert shown == "" and meta["truncated_by"] == "first_line"
    assert ingest.fmt_size(ingest.MAX_BYTES) in ingest.notice(meta)
    # The name of this test promised a fallback and the assertion never checked for one. `offset` is not
    # it: with one line there is nowhere to move to. `start_char` is the argument that windows inside a
    # line, so a disclosure that omits it names a refusal rather than a way on.
    note = ingest.notice(meta, extra="full output saved to /run/out/x.txt")
    assert "start_char" in note, f"no fallback was named: {note}"


def test_a_cut_at_the_first_line_does_not_report_the_whole_files_size():
    """The notice used to print `total_bytes` as though it were the first line's size. On a single-line
    source those are the same number and nothing shows; on a multi-line source the notice tells the model
    the file is 3 MB when the thing it cannot read is 64 KB. A number that is true of the wrong object is
    worse than no number, because the model sizes its next move by it."""
    huge_first = "y" * (ingest.MAX_BYTES + 100)
    text = huge_first + "\n" + ("z" * (3 * 1024 * 1024))
    shown, meta = ingest.clip_head(text)
    assert shown == "" and meta["truncated_by"] == "first_line"
    note = ingest.notice(meta)
    assert ingest.fmt_size(meta["total_bytes"]) not in note, f"the file's size was said as a line's: {note}"


def test_tail_keeps_the_end_because_the_error_is_at_the_bottom():
    text = "\n".join(f"line{i}" for i in range(5000)) + "\nlineFAIL"
    shown, meta = ingest.clip_tail(text, max_lines=20, max_bytes=10_000_000)
    assert meta["truncated"] and shown.endswith("lineFAIL")
    assert "line0\n" not in shown                  # the head fell away, the tail stayed


def test_notice_says_more_remains_when_total_is_unknown():
    # A streaming reader that stopped early does not know the total and must not invent one.
    meta = ingest.meta_fields(truncated=True, by="lines", total_lines=None, total_bytes=999,
                        shown_lines=120, shown_bytes=999)
    note = ingest.notice(meta, extra="read with offset=121")
    assert "more remain" in note and "offset=121" in note
    assert "of None" not in note


def test_a_short_result_gets_no_notice():
    shown, meta = ingest.clip_head("just a line")
    assert ingest.notice(meta, extra="offset=2") == ""


def test_sizes_are_human_readable():
    assert ingest.fmt_size(512) == "512 B"
    assert ingest.fmt_size(2048) == "2.0 KB"
    assert ingest.fmt_size(3 * 1024 * 1024) == "3.0 MB"


def test_a_tail_cut_by_one_giant_final_line_says_so_rather_than_reporting_zero():
    """`clip_head` has a branch for "the first line alone is over the cap" because that case produces an
    empty preview with a confident number attached. `clip_tail` walks from the end, so the same shape in
    the *last* line of a log — a minified payload, a base64 blob, a traceback one-liner — lands in the
    byte branch and reports `showing 0 of 3 lines; cut at 0 B by the … limit`.

    That notice is not merely unhelpful, it is false in a specific way: it says a boundary was chosen
    ("a line boundary, not mid-line") when nothing was chosen at all, and the pointer it offers cannot
    produce the missing text. The truth is that one line is too big, and the way into it is a character
    window."""
    text = "first line\nsecond line\n" + ("z" * (ingest.MAX_BYTES + 500))
    shown, meta = ingest.clip_tail(text)
    assert shown == "" and meta["truncated"] is True
    assert meta["truncated_by"] == "last_line", meta
    note = ingest.notice(meta, extra="full output: /tmp/out/1.txt")
    assert "last line" in note, note
    assert "start_char" in note, f"no way into the oversized line was offered: {note}"
    assert "not mid-line" not in note, f"a boundary was claimed where none was chosen: {note}"


def test_the_meta_builder_is_public_because_a_second_module_calls_it():
    """A name with an underscore in front is a promise that nothing outside the module depends on it, and
    `fs` depended on it: a cross-module call into a private helper means the reading of that promise is
    already false, and the rename that a caller has to earn is the useful kind of signal."""
    assert callable(ingest.meta_fields)
    assert not hasattr(ingest, "_meta"), "the private spelling is still there, so both names are live"
    fields = ingest.meta_fields(truncated=True, by="lines", total_lines=9, total_bytes=90,
                                shown_lines=3, shown_bytes=30)
    assert fields == {"truncated": True, "truncated_by": "lines", "total_lines": 9,
                      "total_bytes": 90, "shown_lines": 3, "shown_bytes": 30}


# ---- what a line is ---------------------------------------------------------------------------
#
# `split("\n")` on text that ends with a newline yields an empty final element: the remainder after the
# terminator, not a line. Counting it makes three measured claims false at once, which is why they are
# tested together — the notice a model reads, the truncation decision, and the pointer in the notice:
#
#   * 2000 lines terminated by a newline reported "showing 2000 of 2001 lines", and
#   * the same text was reported as *truncated* at the cap, so a run was told content had been dropped
#     when nothing had been, and
#   * the notice then offered `offset=2001` — a line that does not exist. `fs.read` counts lines the way
#     a file does (it iterates them), so it answers that offset with nothing. A disclosure whose
#     continuation cannot be carried out is the failure this module exists to prevent, so the count is
#     not cosmetic: it is the number a pointer is built from.

def test_a_newline_terminated_output_is_not_counted_an_extra_line():
    for text in ("a\nb\nc\n", "a\nb\nc"):
        _, meta = ingest.clip_head(text)
        assert meta["total_lines"] == 3, (repr(text[-2:]), meta)
        assert meta["truncated"] is False, repr(text)


def test_the_line_count_agrees_with_counting_the_lines():
    """The count is checked against a second, independent way of counting — iterating a text as a file
    — rather than against the formula that produced it. Asserting `clip_head(...)== my_formula(text)`
    where both halves read the same source is how a wrong definition passes two tests at once."""
    import io
    for n in (1, 2, 5, 40):
        for tail in ("", "\n", "\n\n"):
            text = "\n".join(f"line {i}" for i in range(n)) + tail
            _, meta = ingest.clip_head(text, max_lines=100, max_bytes=10 ** 6)
            counted = sum(1 for _ in io.StringIO(text))
            assert meta["total_lines"] == counted, (n, repr(tail), meta, counted)


def test_text_exactly_at_the_line_cap_is_not_reported_as_truncated():
    """At the cap with a terminating newline, the old count read one line over and so claimed a
    truncation that had not happened. A run paying a second read to recover content that was never
    omitted is the whole cost, and it is paid by the largest inputs."""
    text = "\n".join(f"line {i}" for i in range(ingest.MAX_LINES)) + "\n"
    shown, meta = ingest.clip_head(text)
    assert meta["truncated"] is False, meta
    assert shown == text, "a result that was not cut must not be rewritten"
    assert ingest.notice(meta) == "", "no notice may be attached to a result that lost nothing"


def test_empty_output_has_no_lines():
    shown, meta = ingest.clip_head("")
    assert meta["total_lines"] == 0 and meta["truncated"] is False, meta
    assert shown == ""


def test_a_cut_result_keeps_the_line_endings_it_was_given():
    """The count is fixed by dropping the empty remainder, not by switching to `splitlines()` — which is
    what the review suggested, and which cannot be used here: `splitlines()` also splits on a bare
    carriage return and *discards* the carriage return of CRLF, while the clipper rejoins with `\n`. On a
    Windows command's output that silently rewrites every line ending in the model's context and changes
    the byte arithmetic the notice reports.

    So a carriage return rides with the content of its line, and a cut through CRLF text ends on a bare
    `\r`. That is not damage: the shown text is a byte-exact prefix of the input, which is a stronger
    property than "the line endings look tidy" — it means no byte of a kept line was rewritten, and
    `shown_bytes` counts what is actually returned. Tidying that `\r` away is the rewrite this test exists
    to refuse."""
    text = "first\r\nsecond\r\nthird\r\n"
    head, meta = ingest.clip_head(text, max_lines=2, max_bytes=10 ** 6)
    assert text.startswith(head), (repr(head), "a head must be a prefix of what it was cut from")
    assert head.split("\n") == ["first\r", "second\r"], repr(head)
    assert "\r\n" in head, "the terminator between kept lines must survive the rejoin"
    assert meta["total_lines"] == 3, meta
    tail, _ = ingest.clip_tail(text, max_lines=2, max_bytes=10 ** 6)
    assert tail.split("\n") == ["second\r", "third\r"], repr(tail)


def test_a_giant_final_line_is_named_when_the_output_ends_with_a_newline():
    """Real command output ends with a newline, and that terminator is exactly what hid the last line
    from the clipper: the empty remainder sat where the final line should have been read, so an
    enormous final line fell back into the byte branch and produced the notice claiming a line boundary
    had been chosen. A fix aimed at `lines[-1]` that only works when the text has no terminator covers
    the minority of real inputs."""
    text = "first\nsecond\n" + ("z" * (ingest.MAX_BYTES + 500)) + "\n"
    shown, meta = ingest.clip_tail(text)
    assert shown == "" and meta["truncated_by"] == "last_line", meta
    assert ingest.notice(meta).count("last line") == 1, ingest.notice(meta)
