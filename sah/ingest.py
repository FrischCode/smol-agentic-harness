"""sah.ingest — the one place that bounds how much of a big thing enters the run.

A tool result reaches the model as a bounded *preview with an actionable continuation*, never as a
silent cut and never as the whole thing dumped into the window. Two independent limits, whichever is
hit first wins: a line count and a byte count. A truncated result is never a lie about its size — the
preview says what it showed and how much there was, so paging or spilling is a choice the model can
make rather than a surprise.

Two sinks, by whether the source survives:

- A **file** still exists after the read, so it is never copied or spilled — the model pages it with
  `offset`. Reading must therefore *stream* to the cap, not whole-load it: returning the first page
  of a 493 MB file must not pull 493 MB into memory. When streaming stops at the cap without reaching
  the end, the count of lines is genuinely unknown, so the notice says "more remain" rather than
  inventing a total it would have to read the whole file to know.
- A **command's output** is ephemeral — if it is not kept, it is gone. So an over-cap command result
  is written whole to a file under the run's own tree (which the run owns and can read) and the path
  is handed back alongside the preview. Nothing is lost; it is just kept where it costs the context
  nothing.

The limits live here and nowhere else, so "how big does a result get" has one answer — and that
answer has two numbers: what a *result* may carry (`MAX_LINES`/`MAX_BYTES`, enforced at the choke
point) and what a *producer* should therefore stop at (`BODY_LINES`/`BODY_BYTES`) so its own
disclosure still fits.
"""
from __future__ import annotations

MAX_LINES = 2000            # into the context before truncation
MAX_BYTES = 64 * 1024       # ... or this many bytes, whichever is hit first
GREP_LINE = 500             # cap each grep hit so one minified line can't eat the budget

# A producer that caps its own output AND adds a disclosure has to stop the *body* below the result
# limits, because the disclosure is part of the result: capped to MAX_* and then appended, the result
# arrives at the choke point over the limit, gets re-clipped (head-clipping drops the end, where the
# continuation note lives) and is spilled — so a file the model could page is copied instead.
#
# The reserve is the widest overhead any producer adds: `web.fetch` stacks three header lines (two, plus
# a redirect line when it fired), a blank and the untrusted-page banner above its body — five lines and
# roughly 300 bytes — where `fs` needs two lines. One number for all of them is the point; widen it
# here rather than special-casing a producer.
BODY_LINES = MAX_LINES - 8          # room for the widest header any producer emits
BODY_BYTES = MAX_BYTES - 1024       # ... and the bytes that come with it


def fmt_size(n: int) -> str:
    """A byte count as a human size — whole `B` under 1 KiB, then `KB` or `MB` to one decimal."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _lines(text: str) -> list[str]:
    """The lines a reader would count, split on the byte this module joins with.

    Two rules the bare `split("\n")` gets wrong, both measured: an empty text has no lines at all, and
    the empty remainder after a terminating newline is not a line. Counting that remainder reported
    "showing 2000 of 2001 lines" for a 2000-line output, called text exactly at the cap truncated, and
    built a notice pointing at `offset=2001` — a line `fs.read` answers with nothing, because a file
    iterates its lines the way a reader counts them.

    `splitlines()` is not the fix, though it drops the same remainder. It also splits on a bare carriage
    return and *discards* the carriage return of CRLF, while this module rejoins with `\n`; on a Windows
    command's output that silently rewrites every line ending the model sees and shifts the byte arithmetic
    the notice quotes. Splitting on the byte the join uses is what makes a cut lossless.
    """
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        # Safe to pop unconditionally because the empty text returned above: `split` yields the single
        # element [""] for nothing else, so an empty final element is always a terminator's remainder.
        lines.pop()             # "a\n" is one line, "a\n\n" is two: "a" and an empty one
    return lines


def meta_fields(*, truncated, by, total_lines, total_bytes, shown_lines, shown_bytes):
    """The truncation metadata a result carries: the flag, what cut it, and the total/shown sizes."""
    return {"truncated": truncated, "truncated_by": by, "total_lines": total_lines,
            "total_bytes": total_bytes, "shown_lines": shown_lines, "shown_bytes": shown_bytes}


def notice(meta: dict, extra: str = "") -> str:
    """The bracketed disclosure a truncated result carries to the model.

    The byte-border case is said out loud: cutting at a byte limit means the reader must know it
    stopped mid-file at a size ceiling, not at the end. When `total_lines` is None the source was
    never read to the end (streaming stopped early), so it says "more remain" rather than a total it
    did not count. `extra` is the caller's pointer — "use offset=N", or "full output: <path>".
    """
    if not meta or not meta.get("truncated"):
        return ""
    by = meta.get("truncated_by")
    if by == "last_line":
        body = (f"the last line alone is over the {fmt_size(MAX_BYTES)} byte limit, so nothing is "
                "shown — page inside it with start_char")
    elif by == "first_line":
        # The size of the whole text is not the size of its first line, so no number is quoted here — and
        # the continuation has to be the one that works inside a single line. `offset` moves between lines;
        # a minified page has one, so a pointer that says only "read it with offset" names an argument
        # that returns the same refusal. `start_char` is the argument that pages within a line.
        body = (f"the first line alone is over the {fmt_size(MAX_BYTES)} byte limit, so nothing is "
                "shown — page inside it with start_char")
    elif by == "bytes":
        total = (f"{meta['total_lines']} lines" if meta.get("total_lines") is not None
                 else "more remain")
        body = (f"showing {meta['shown_lines']} of {total}; cut at {fmt_size(meta['shown_bytes'])} "
                f"by the {fmt_size(MAX_BYTES)} byte limit (a line boundary, not mid-line)")
    else:  # lines, or an unknown-total stream that stopped early
        total = (f"of {meta['total_lines']} lines" if meta.get("total_lines") is not None
                 else "and more remain")
        body = f"showing {meta['shown_lines']} {total}"
    body = body + (f". {extra}" if extra else "")
    return f"[{body}]"


def clip_head(text: str, *, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES):
    """Keep the beginning — whole lines only — for reading a file's start or any front-loaded output.

    Returns (shown, meta). Never emits a partial line. If the very first line already exceeds the byte
    limit it shows nothing and reports `truncated_by="first_line"`, so the caller offers a targeted
    window of its own instead of dumping a megabyte or crashing."""
    total_bytes = len(text.encode("utf-8"))
    lines = _lines(text)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, meta_fields(truncated=False, by=None, total_lines=total_lines,
                           total_bytes=total_bytes, shown_lines=total_lines, shown_bytes=total_bytes)
    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return "", meta_fields(truncated=True, by="first_line", total_lines=total_lines,
                         total_bytes=total_bytes, shown_lines=0, shown_bytes=0)
    taken: list[str] = []
    used = 0
    by = "lines"
    for line in lines:
        if len(taken) >= max_lines:
            by = "lines"
            break
        cost = len(line.encode("utf-8")) + (1 if taken else 0)     # the joining newline
        if used + cost > max_bytes:
            by = "bytes"
            break
        taken.append(line)
        used += cost
    shown = "\n".join(taken)
    return shown, meta_fields(truncated=True, by=by, total_lines=total_lines, total_bytes=total_bytes,
                        shown_lines=len(taken), shown_bytes=used)


def clip_tail(text: str, *, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES):
    """Keep the end — for command output, where the error is at the bottom. Returns (shown, meta)."""
    total_bytes = len(text.encode("utf-8"))
    lines = _lines(text)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, meta_fields(truncated=False, by=None, total_lines=total_lines,
                           total_bytes=total_bytes, shown_lines=total_lines, shown_bytes=total_bytes)
    if lines and len(lines[-1].encode("utf-8")) > max_bytes:
        # The same failure clip_head names, arriving from the other end: a log whose last line is one
        # enormous record (a minified payload, a base64 blob, a one-line traceback) keeps nothing, and
        # letting it fall into the byte branch would report a chosen boundary — "a line boundary, not
        # mid-line" — where nothing was chosen at all.
        return "", meta_fields(truncated=True, by="last_line", total_lines=total_lines,
                               total_bytes=total_bytes, shown_lines=0, shown_bytes=0)
    kept: list[str] = []
    used = 0
    by = "lines"
    for line in reversed(lines):
        if len(kept) >= max_lines:
            by = "lines"
            break
        cost = len(line.encode("utf-8")) + (1 if kept else 0)
        if used + cost > max_bytes:
            by = "bytes"
            break
        kept.append(line)
        used += cost
    kept.reverse()
    shown = "\n".join(kept)
    first_kept = total_lines - len(kept)
    return shown, meta_fields(truncated=True, by=by, total_lines=total_lines, total_bytes=total_bytes,
                        shown_lines=len(kept), shown_bytes=used) | {"first_kept_line": first_kept}
