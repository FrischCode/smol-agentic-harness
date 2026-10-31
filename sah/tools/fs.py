"""sah.tools.fs — the tools that touch files.

Three plain functions. They take an address and do the thing; they know nothing about grants,
policies or runs, because a tool that could decide its own permission is not confined, it is
polite. The permission lives in `sah.confine`, and the loop runs a tool only after that has
said yes.

Reads and listings are *bounded at the source* and *paged*: a file is never pulled whole into memory
to return one page (returning the first lines of a 493 MB file must cost the first page, not 493 MB),
and a truncated read says so and names the next offset, so paging is a choice the model makes rather
than a surprise. `grep` is bounded the same way and pages by *match* rather than by line. The limits
live in `sah.ingest`.

Two tools change a file, and they fail differently. `write` replaces everything: whatever the run did
not retype is gone. `edit` replaces one matched span and leaves every byte outside it exactly as it
found them — which is why it looks for that span before it opens anything for writing, and why a file
whose bytes it cannot read faithfully is a file it refuses to write back.

Paths are resolved relative to the process working directory by both this module and the policy, with
one deliberate asymmetry: the policy resolves symlinks before deciding, so it asks where the file
*really* is, while the tool follows the path as given. The asymmetry only ever makes the policy
stricter, which is the safe direction.
"""
from __future__ import annotations

import fnmatch
import os
import re
from typing import Annotated

from sah import ingest
from sah.contract import Media, Param, picture_kind


def read(path: Annotated[str, Param("file to read")],
         offset: Annotated[int, Param("1-based line to start from")] = 1,
         count: Annotated[int, Param("max lines to return; 0 means up to the size cap")] = 0,
         start_char: Annotated[int, Param(
             "character to start from within that first line, to page one line too large to return")]
         = 0) -> str | Media:
    """A picture is told by its bytes rather than its name: a file named `photo.png` that holds prose
    must not be attached as a picture, and a JPEG saved as `capture.dat` must not arrive as escapes.
    The weak magics (BMP's `BM`, WebP inside RIFF) get a second look in `tools.picture_kind`.

    The file is streamed, never whole-loaded: reading one page of a huge file costs that page. A page
    that stops at the cap without reaching the end does not claim a total line count it would have to
    read the whole file to know — it says more remains.

    One line larger than the byte cap is paged too, with `start_char` counted into that line: the result
    gives a window of it and names the `start_char` that continues. That is what a minified line or a
    one-line JSON blob needs, and paging it costs no shell and no extra permission.

    Text that is not valid UTF-8 arrives as `\\xNN` escapes, so a file in another code page shows the bytes
    it holds instead of a run of `?`. A picture is not that case: it is named and attached, because
    several thousand escapes are a faithful rendering of nothing anyone can look at."""
    mime = picture_kind(path)
    if mime:
        # The tool states what the file is and stops there. Whether the bytes may be attached is the
        # endpoint's declaration, which the loop holds, and the loop says what it did — a receipt claims
        # only what the holder of the act performed.
        return Media(text=f"{path} is a picture ({mime}, {ingest.fmt_size(os.path.getsize(path))})",
                     path=path, mime=mime)
    start = max(1, int(offset))
    take = int(count) if int(count) > 0 else None       # None: no user limit, the cap decides
    # A file that is not UTF-8 is read as what it is, byte for byte visible: `\xe9` can be named, worked
    # with or ignored knowingly, while U+FFFD only says that something happened.
    with open(path, encoding="utf-8", errors="backslashreplace") as handle:
        return _read_stream(handle, start, take, path, max(0, int(start_char)))


read.description = ("Read a file. Text comes back as text, paged with offset and capped "
                    "(~2000 lines / 64 KB); a truncated read says where it stopped and the next "
                    "offset to continue from. A picture (png, jpeg, gif, webp or bmp) comes back as "
                    "the picture itself. Paths are the operating system's, not the shell's: on "
                    "Windows `/c/dir` is read as the current drive's root plus `c/dir`, so name the "
                    "drive — `C:/dir`.")


def _window(line: str, line_no: int, at: int) -> str:
    """One window of a line too large to return, with the pointer that continues it.

    A shell command is not the answer to an over-cap line. `sed -n 'Np' path | head -c N` asks a run
    without `--exec` to do something it is refused, and asks it of a cmd.exe box that has no `sed`:
    either way a tool that had just read the file successfully sends the model to a tool it cannot use,
    for something the first tool could do. And a folder hint to run `ls -la` asks for a *larger* listing
    than the one just declined. A pointer to the tool's own paging is honest whatever the grants and
    whatever the shell.

    Counted in characters and cut on character boundaries: cutting at a byte count can split a multi-byte
    character, and a `?` substituted into the middle of the text would make the pages reassemble into
    something that is not the file.
    """
    if at >= len(line):
        return f"[line {line_no} is {len(line)} characters; start_char={at} is past its end]"
    room = ingest.BODY_BYTES
    end, used = at, 0
    while end < len(line):
        cost = len(line[end].encode("utf-8", "replace"))
        if used + cost > room:
            break
        used += cost
        end += 1
    body = line[at:end]
    if end >= len(line):
        return f"{body}\n\n[the end of line {line_no}, which is {len(line)} characters]"
    return (f"{body}\n\n[line {line_no} is {len(line)} characters, over the "
            f"{ingest.fmt_size(ingest.MAX_BYTES)} byte limit; this is characters {at}-{end}, "
            f"continue with read offset={line_no} start_char={end}]")


def _read_stream(handle, start: int, take, path: str, start_char: int = 0) -> str:
    """Stream one page and disclose it, stopping at the *body* limits rather than the result limits.

    This function appends a notice, and a notice is part of the result. Cap the body to `MAX_*` and
    append, and the page arrives at the choke point over its limit, where head-clipping drops the end —
    which is the `read with offset=N` line, the one thing the cap exists to provide — and the page is
    spilled to a file instead of continued in place. A `count` above what a result can carry needs no
    special case: the line cap is tested first, so the binding limit is the one the notice names."""
    shown: list[str] = []
    used = 0
    lineno = 0
    stopped = None                       # "bytes" | "lines" | "user" once a stop is decided
    held: str | None = None              # the line that broke the loop: read, but not shown
    it = iter(handle)
    for raw in it:
        lineno += 1
        if lineno < start:
            continue
        line = raw.removesuffix("\n")
        line_bytes = len(line.encode("utf-8", "replace"))
        if not shown and line_bytes > ingest.MAX_BYTES:
            # A single line over the cap is not dumped and does not crash the read: it is windowed, and
            # the window names the position that continues it.
            return _window(line, start, start_char)
        cost = line_bytes + (1 if shown else 0)          # the newline that joins it
        if used + cost > ingest.BODY_BYTES:
            stopped, held = "bytes", line
            break
        if len(shown) >= ingest.BODY_LINES:
            stopped, held = "lines", line
            break
        if take is not None and len(shown) >= take:
            stopped, held = "user", line
            break
        shown.append(line)
        used += cost

    body = "\n".join(shown)
    if not stopped:
        return body                                     # reached end of file, nothing cut
    # There is more if a line was held back, and only otherwise does the peek decide: the loop consumes
    # the line that broke it, so peeking when *that* line is the file's last reads EOF and reports a cut
    # as a complete page. The held line is proof of content, whatever follows it.
    has_more = held is not None or next(it, None) is not None
    if not has_more:
        return body
    next_offset = start + len(shown)
    meta = ingest.meta_fields(truncated=True, by=("bytes" if stopped == "bytes" else "lines"),
                        total_lines=None, total_bytes=used, shown_lines=len(shown), shown_bytes=used)
    # `notice()` already says the source has more, so only the pointer is added here — saying it twice
    # makes one disclosure read like two authors. A count that stopped the read is named, because there
    # the notice does not say who stopped it.
    extra = ("stopped at your count; read more with offset=" if stopped == "user"
             else "read with offset=") + str(next_offset)
    if held is not None and len(held.encode("utf-8", "replace")) > ingest.MAX_BYTES:
        # The size of the next line is already in hand, so the disclosure can name the argument that
        # actually pages it. `offset` moves between lines; if the remaining line is one giant line — a
        # minified script, a one-line JSON blob — an offer to move to it is an offer to be refused again.
        extra += " start_char=0 (that line alone is over the byte limit; start_char windows inside it)"
    return f"{body}\n\n{ingest.notice(meta, extra=extra)}"


def write(path: Annotated[str, Param("file to write, created with parents if absent")],
          content: Annotated[str, Param("the full new contents")]) -> str:
    """Write a text file, creating parent folders, and say whether it was created or replaced.

    "Did I just destroy something" should not cost a second call, so the result names the act."""
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    existed = os.path.isfile(target)
    data = content.encode("utf-8")
    with open(target, "wb") as handle:
        handle.write(data)
    return f"{'replaced' if existed else 'created'} {target} ({len(data)} bytes)"


write.description = "Write a text file, creating parent folders, and say whether it was created or replaced."


def _collapse(text: str):
    """`text` with every line ending reduced to one `\\n`, and where each character came from.

    The map answers "which byte of the file is this position in the reduced text": `at[i]` is the offset
    in the original of reduced character `i`, and `at[len(view)]` is the end of the original. That is what
    lets a replacement be spliced into the file's own bytes. Normalising the whole file and writing that
    back would quietly change every line ending in a file the run meant to touch in one place, and a tool
    that rewrites what it was not asked to rewrite cannot be reasoned about.

    `None` for the map means the text held no carriage return and is its own view.
    """
    if "\r" not in text:
        return text, None
    view: list[str] = []
    at: list[int] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\r":
            view.append("\n")
            at.append(i)
            i += 2 if text.startswith("\r\n", i) else 1
        else:
            view.append(text[i])
            at.append(i)
            i += 1
    at.append(n)
    return "".join(view), at


def _at(at, i: int) -> int:
    """Map a position in the reduced view back into the file's own bytes."""
    return i if at is None else at[i]


def _spans(view: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping place `needle` occurs in `view`, in the order a reader meets them."""
    found = []
    from_ = view.find(needle)
    while from_ >= 0:
        found.append((from_, from_ + len(needle)))
        from_ = view.find(needle, from_ + len(needle))
    return found


def _missed(target: str, text: str, needle: str) -> str:
    """Why nothing matched, with the pointer that would show the text that came closest.

    Almost always the cause is that `old_text` was remembered rather than copied — the right words with
    the wrong indentation. Saying only "not found" makes the run guess at whitespace again, so the
    refusal looks for the first line of what was *asked for* and, if that line exists somewhere in the
    file, names the `read` that shows it. A disclosure that sends the run back to the file with a better
    view of it is worth more than a correct refusal.
    """
    head = next((line.strip() for line in needle.split("\n") if line.strip()), "")
    where = ([str(i) for i, line in enumerate(text.split("\n"), 1) if line.strip() == head][:3]
             if head else [])
    if where:
        first = int(where[0])
        return (f"{target}: nothing matched, though the first line of old_text is at line "
                f"{', '.join(where)} — the text around it differs from what was passed. Read that part "
                f"again and copy it exactly: read path={target!r} offset={max(1, first - 2)} count=10")
    return (f"{target}: nothing matched, and not one line of old_text occurs in the file. It has to be "
            f"copied from the file, indentation and blank lines included: read path={target!r} first")


def edit(path: Annotated[str, Param("file to change")],
         old_text: Annotated[str, Param("the exact text to replace; it must match one place in the file")],
         new_text: Annotated[str, Param("the text to put there instead")],
         every: Annotated[bool, Param("change every occurrence rather than "
                                      "insisting there is only one")] = False,
         ) -> str:
    """Insisting on exactly one match is what stops a change landing on the second copy of a repeated
    line by accident; and the file is only opened for writing once that span has been found, so a
    change that cannot be located costs a refusal and no bytes.

    A file whose lines end CRLF is *shown* without those endings, so text copied from a read arrives
    without them. Matching is done on a view where every ending is one character, and the new text is
    written with the endings the file already uses, so editing one line of a CRLF file does not turn it
    into a file with two kinds of endings. The receipt says when that happened: a claim about what was
    written is a claim about bytes.

    A file that is not valid UTF-8 is refused rather than rewritten. `read` can show such bytes as
    `\\xNN` escapes, but writing them back would replace bytes the tool never read faithfully with
    something else, and quietly re-encoding the parts of a file that were not edited destroys more than
    the edit changes."""
    target = os.path.abspath(path)
    if os.path.isdir(target):
        return f"{target} is a folder, not a file: edit changes a file, write creates one"
    if not old_text:
        return ("old_text is empty, so there is no span to change. To insert text, put an existing line "
                "in old_text and that line with the new text beside it in new_text")
    try:
        # newline="" keeps the file's own endings instead of quietly translating them on the way in.
        with open(target, encoding="utf-8", newline="") as handle:
            original = handle.read()
    except FileNotFoundError:
        return f"{target} does not exist: edit changes a file that is there, write creates one"
    except UnicodeDecodeError as exc:
        return (f"{target} is not UTF-8 — byte {exc.start} of it is not. read shows those bytes as "
                f"\\xNN escapes, but writing this file back would replace bytes the tool could not read, "
                f"so edit refuses")

    view, at = _collapse(original)
    needle, _ = _collapse(old_text)          # text the model typed usually has one kind of ending
    found = _spans(view, needle)
    if not found:
        return _missed(target, view, needle)
    if len(found) > 1 and not every:
        return (f"{target}: old_text matches {len(found)} places, and edit will not guess which was "
                f"meant. Make it longer until it is unique, or pass every=true to change all "
                f"{len(found)}")

    # The endings the file itself uses, decided by what is in it rather than by what was passed: the
    # common case is a file whose newlines are all CRLF being given text with all LF.
    crlf = (original.count("\r\n") > 0
            and original.count("\r\n") >= original.count("\n") - original.count("\r\n"))
    replacement = new_text.replace("\r\n", "\n").replace("\r", "\n")
    if crlf:
        replacement = replacement.replace("\n", "\r\n")
    pieces, last = [], 0
    for i, j in found:
        pieces.append(original[last:_at(at, i)])
        pieces.append(replacement)
        last = _at(at, j)
    pieces.append(original[last:])
    changed = "".join(pieces)
    if changed == original:
        # Nothing to write, so nothing is claimed about writing: no bytes touched, no mtime moved.
        return f"{target}: new_text is the text that is already there, so nothing was written"

    with open(target, "wb") as handle:
        handle.write(changed.encode("utf-8"))
    receipt = (f"edited {target}: changed {len(found)} place(s) at line "
               f"{view.count(chr(10), 0, found[0][0]) + 1}, {len(original)} -> {len(changed)} characters")
    if crlf and "\n" in new_text and "\r\n" not in new_text:
        receipt += " (new text written with the file's CRLF endings)"
    return receipt


edit.description = ("Change one span of a file, leaving every byte outside the matched text exactly "
                    "as it was found. `old_text` is what is there and must match exactly one place; "
                    "`new_text` is what to put in its place. Where the text occurs several times the "
                    "refusal says how many, and `every=true` is the way to mean all of them — a rename "
                    "across a file is then one call. A span that cannot be located costs a refusal and "
                    "no bytes. Nothing carries between calls: each one is matched against the file as "
                    "it is now, so a second edit sees what the first one wrote.")


def ls(path: Annotated[str, Param("folder to list")],
       offset: Annotated[int, Param("1-based entry to start from, to page a large folder")] = 1) -> str:
    """Paged rather than truncated, because a truncated listing would hide entries the model then
    assumes do not exist; the capped page names the `offset` that continues it, the same contract
    `read` and `find` carry. The drive-naming rule is there because the paths are the operating
    system's: a model that writes the shell's `/c` form is describing a view this process does not
    have."""
    target = os.path.abspath(path)
    if not os.path.isdir(target):
        return f"{target} is not a folder"
    entries = sorted(os.listdir(target))
    start = max(1, int(offset))
    window = entries[start - 1:]
    if not window:
        # "is empty" for a folder with 4000 entries would send the model away with a false fact about
        # the folder; an offset past the end is named as one, with the size that is really there.
        return (f"{target} is empty" if not entries
                else f"{target} has {len(entries)} entries; offset={start} is past the end")
    out = []
    used = 0
    for name in window:
        full = os.path.join(target, name)
        if os.path.isdir(full):
            line = name + "/"
        else:
            try:
                line = f"{name} ({os.path.getsize(full)} bytes)"
            except OSError:
                line = name
        # Both bounds, like the read: a folder of short names is stopped by the line cap, a folder of
        # long ones by the byte cap, and either way the notice still fits in the room left for it.
        cost = len(line.encode("utf-8", "replace")) + (1 if out else 0)
        if len(out) >= ingest.BODY_LINES or used + cost > ingest.BODY_BYTES:
            break
        out.append(line)
        used += cost
    shown = len(out)
    if len(window) > shown:
        out.append(f"[{len(window) - shown} more entries; continue with ls offset={start + shown}]")
    return "\n".join(out) if out else f"{target} is empty"


ls.description = ("List a folder: names, with `/` on folders and the byte size on files. A large "
                  "folder is paged, not truncated: a capped listing says how many entries remain and "
                  "names the `offset` that continues where this page stopped. Paths are the operating "
                  "system's, not the shell's: on Windows `/c/dir` is read as the current drive's root "
                  "plus `c/dir`, so name the drive — `C:/dir`.")


# A folder whose contents are not text: an object store and compiled bytecode. Neither is ever what a
# search was looking for, and walking `.git` in a real repository costs thousands of unreadable files.
# Named in the result rather than hidden — the walk reports what it declined — because "no matches" is
# a claim about a set of files.
SKIP_DIRS = (".git", "__pycache__")


def _is_binary(file: str) -> bool:
    """Whether a file's first bytes say it holds no text to search.

    A NUL byte is the honest marker: it turns up in compiled and compressed files and not in the text
    this tool is for. A file that is merely *not* UTF-8 is not treated as binary — it is searched with
    its undecodable bytes shown as `\\xNN`, exactly as `read` shows them — because a file written in
    another code page is a fact about the folder, and declining to open it would let the tool answer
    "no matches" about a file it never looked at.
    """
    with open(file, "rb") as handle:
        return b"\0" in handle.read(4096)


def _hit(display: str, line_no: int, line: str, is_match: bool) -> str:
    """One line of a search result, cut to what one hit may cost, with the pointer that shows the rest.

    Without the cut, one minified line answers a search with the whole of a bundle and the result is
    spilled instead of read. The pointer is the `read` that continues that very line — `offset` is
    already its line number, so the run needs no new permission and no shell to see what was held back.
    """
    prefix = f"{display}:{line_no}: " if is_match else f"{display}-{line_no}- "
    if len(line) <= ingest.GREP_LINE:
        return prefix + line
    return (f"{prefix}{line[:ingest.GREP_LINE]}\n[that line is {len(line)} characters, "
            f"{ingest.fmt_size(len(line.encode('utf-8', 'replace')))}; only the first "
            f"{ingest.GREP_LINE} are shown, and the rest is yours with read path={display!r} "
            f"offset={line_no} start_char={ingest.GREP_LINE}]")


def _hits(file: str, display: str, rx, context: int):
    """One file's matches, one block per matching line: its leading context, its own line, its trailing
    context. Yields the block as a list of formatted lines, in file order.

    Streaming is the shape of this function, and the *block* is its unit. The lines before a match are
    not known to matter until the match arrives, so they wait in a buffer no longer than the requested
    context; the lines after one are kept while that credit lasts. So the caller is never handed a
    context line and asked whether to show it: the match it belongs to has not been seen yet, and a page
    that began at match 40 would throw away the context of match 40 while keeping its line. Handing over
    a block leaves the caller one decision — whether this match is on the page.

    Two matches near each other still get one block each, and the lines between them are shown once: as
    leading context only if they were not already printed as the previous match's trailing.
    """
    before: list[tuple[int, str]] = []       # candidate leading context, trimmed to `context`
    block: list[str] = []                    # the match being assembled, empty when none is open
    owed = 0                                 # trailing-context lines still wanted for the open block
    with open(file, encoding="utf-8", errors="backslashreplace") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.removesuffix("\n").removesuffix("\r")
            if rx.search(line) is not None:
                if block:
                    yield block
                leading = [_hit(display, n, text, False) for n, text in before]
                block = leading + [_hit(display, line_no, line, True)]
                owed = context
                before = []
                if not owed:
                    yield block
                    block = []
            elif block and owed > 0:
                block.append(_hit(display, line_no, line, False))
                owed -= 1
                if not owed:
                    yield block
                    block = []
            elif context:
                before.append((line_no, line))
                if len(before) > context:
                    del before[0]
    if block:
        yield block


def grep(pattern: Annotated[str, Param("regular expression to look for, or plain text when literal=true")],
         path: Annotated[str, Param("file or folder to search")] = ".",
         glob: Annotated[str, Param("only search files whose name matches this, e.g. '*.py'")] = "",
         ignore_case: Annotated[bool, Param("match letters in either case")] = False,
         literal: Annotated[bool, Param("look for the pattern as plain text, "
                                        "not a regular expression")] = False,
         context: Annotated[int, Param("lines of surrounding text to show with each match (up to 10)")] = 0,
         offset: Annotated[int, Param("1-based match to report from, to page past a large result")] = 1,
         ) -> str:
    """Needing no shell is the point of the tool's existence: a search that runs the process's own
    regular-expression engine costs no command, no dialect and no exec grant, so a run that may not run
    anything can still search. The absolute paths are the contract half of the same decision: a result
    line the model cannot feed back into `read path=` unchanged is a search that has to be re-run. The
    coverage caveat — counting and naming what it could not open — is for the "no matches" case: that
    answer means something different after a folder was only half looked at, and the model has to be
    able to tell the two apart."""""
    if not pattern:
        return "pattern is empty, so every line would match. Name something to look for."
    try:
        rx = re.compile(re.escape(pattern) if literal else pattern,
                        re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return (f"pattern is not a regular expression this tool can compile: {exc}. To search for the "
                f"characters themselves (a dot, a bracket, a backslash) pass literal=true")

    top = os.path.abspath(path)
    if not os.path.exists(top):
        return f"{top} does not exist"
    if not (os.path.isfile(top) or os.path.isdir(top)):
        # A socket, a named pipe, a device: something the walk would open and never finish reading.
        return f"{top} is neither a file nor a folder"

    wanted = max(1, int(offset))
    around = max(0, min(int(context), 10))
    out: list[str] = []
    used = 0
    stopped = None                       # "bytes" | "lines" once a cap is what stopped the page
    seen = opened = shown = 0
    unread = binary = 0
    declined: set[str] = set()          # folder names the walk refused to enter, named in the result

    queue = _candidates(top, glob, declined)

    for file in queue:
        try:
            if _is_binary(file):
                binary += 1
                continue
            display = os.path.abspath(file)
            for block in _hits(file, display, rx, around):
                seen += 1
                if seen < wanted:
                    continue                  # a whole match, with its context, is not on this page
                # A block is taken whole or the page stops here. Splitting one across pages would have
                # to decide which half of a match's context each page owns, and a match printed without
                # the lines that explain it is the least useful line in the result.
                cost = sum(len(text.encode("utf-8", "replace")) + 1 for text in block) - (0 if out else 1)
                if used + cost > ingest.BODY_BYTES:
                    stopped = "bytes"
                    break
                if len(out) + len(block) > ingest.BODY_LINES:
                    stopped = "lines"
                    break
                out.extend(block)
                used += cost
                shown += 1                    # counted only once the block is really on the page
        except OSError:
            unread += 1
            continue
        opened += 1
        if stopped:
            break

    gap = ""
    notes = []
    if binary:
        notes.append(f"{binary} file(s) held no text and were skipped")
    if unread:
        notes.append(f"{unread} file(s) could not be opened")
    if declined:
        notes.append(f"{', '.join(sorted(declined))}/ not searched")
    if notes:
        gap = " (" + "; ".join(notes) + ")"

    if not out and not opened and not binary and not unread:
        # Nothing was even opened: the filter named no file at all, which is a different fact from
        # "looked and did not find".
        return ((f"No file under {top} is named {glob!r}" if glob
                 else f"Nothing to search under {top}") + gap)
    if not out:
        # "No matches" is a claim about a set of files, so the answer names how many were opened. No
        # total of the folder is claimed: the search stops when the page is full, so the rest of the
        # tree was deliberately never walked, and counting it is the work the cap exists to avoid.
        return (f"No matches for {pattern!r} in {top} — {opened + binary + unread} file(s) looked "
                f"at{gap}.")
    head = f"# {shown} match(es) for {pattern!r} in {top}"
    if stopped:
        # The page stopped inside the body, so no total is claimed either.
        body = "\n".join(out)
        meta = ingest.meta_fields(truncated=True, by=stopped, total_lines=None,
                                 total_bytes=used, shown_lines=len(out), shown_bytes=used)
        return f"{head}\n\n{body}\n\n{ingest.notice(meta, extra=f'grep offset={wanted + shown}')}{gap}"
    return f"{head}{gap}\n\n" + "\n".join(out)


grep.description = ("Search file contents and return the matching lines as `path:line: text`, with "
                    "`path-line- text` for the lines shown around a match. It walks a folder (or "
                    "searches one file): the search is Python's own regular-expression dialect, files "
                    "are matched by name with `glob`, and the paths it prints are absolute, so a "
                    "result line can be read back with `read path=` unchanged. Results are paged by "
                    "match, not cut off: a page that reaches the size cap says so and names the "
                    "`grep offset=` that continues with the next match, and a single line too long to "
                    "print is windowed with the `read` that shows the rest of it. `.git` and "
                    "`__pycache__` are not searched, files with no text in them are skipped, and "
                    "anything that could not be opened is counted and named.")


def _candidates(top: str, glob: str, declined: set):
    """The files a search may open, one at a time, folders and names in a fixed order.

    A generator rather than a list because a page that fills up must stop the walk, not sit behind it:
    on a tree with a large `node_modules` the point of a cap is that the rest is never opened. Sorted
    because a result the model pages has to be the same tree on both pages, and the operating system
    makes no promise about the order it hands names back.

    `declined` is filled by the walk with the folders it would not enter. It is passed in rather than
    returned because the walk ends when the page is full, and "the search never looked inside .git" is a
    fact the caller has to be able to state after that break, not one it can re-derive.
    """
    if os.path.isfile(top):
        yield top
        return
    for root, dirs, names in os.walk(top):
        hidden = [d for d in dirs if d in SKIP_DIRS]
        if hidden:
            declined.update(hidden)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if not glob or fnmatch.fnmatchcase(name, glob):
                yield os.path.join(root, name)


def _path_rx(pattern: str):
    """Compile a glob that contains a `/` into a regex over a `/`-joined path, relative to the search.

    The two wildcards mean different things, and the difference is the whole reason this is not
    `fnmatch`: `*` stops at a separator and `**` crosses them, so `notes/*.md` cannot reach
    `a/notes/b.md` while `notes/**/*.md` can. `fnmatch.translate` maps both to "anything at all", which
    would make a narrow pattern answer with the whole tree — the one glob result that misleads a run
    into reading files it never asked about.

    A path pattern is anchored at the *end* rather than the start, because that is what the short form
    promises: `src/**/*.py` is meant to find that subtree wherever the search begins, so a leading
    `**/` is implied unless the caller wrote one. That implication is also what lets a slash-free pattern
    mean "a name at any depth" without a second matching path: `*.py` becomes `**/*.py`, and since `*`
    cannot cross a separator the two readings cannot diverge.
    """
    if not pattern.startswith(("**/", "/")) and pattern != "**":
        pattern = "**/" + pattern
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = pattern.find("]", i + 1)
            if end < 0:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:end]
                out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
                i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def find(pattern: Annotated[str, Param(
            "glob to match names against: '*.py', '**/*.md', 'tests/test_*.py'. With no `/` it "
            "matches a name at any depth; with a `/` it matches the path under the folder searched, so "
            "'src/**/*.py' finds that subtree wherever the search starts. "
            "In a path, `*` stops at a folder boundary and `**` crosses it. Case matters, and a "
            "pattern with no wildcard names that file exactly rather than anything containing it")],
         path: Annotated[str, Param("folder to search")] = ".",
         offset: Annotated[int, Param("1-based entry to report from, to page past a large result")] = 1,
         ) -> str:
    """Find files and folders by name, recursively, and return their absolute paths one per line
    (folders marked with `/`). This reads no file contents, so it costs names rather than bytes: the way
    to ask "which files are called that" before spending a search or a read on them. `ls` answers that
    about one folder at a time, and a search matches text rather than names. A capped page names the
    `find offset=` that continues it; `.git` and `__pycache__` are not walked, symlinks are not
    followed, and a folder that could not be read is counted in the answer rather than walked past.

    The matching rules live in the `pattern`'s `Param` description, because that is the text that
    reaches the model for that field (`description` is tool-level, `Param` rides the field): a rule
    written here would be prose for the person reading the file and invisible to the run composing the
    glob. Symlinks are not
    followed for a related reason — a folder that links to its own parent is a tree with no bottom,
    and a walk that does not stop is one long command instead of an answer.
    """
    if not pattern:
        return ("pattern is empty, so every entry would match. Name what to look for — '*' is everything, "
                "and '*.py' is every file whose name ends in .py")
    top = os.path.abspath(path)
    if not os.path.exists(top):
        return f"{top} does not exist"
    if not os.path.isdir(top):
        return f"{top} is not a folder: find searches inside a folder"

    # One mechanism for both rules the description promises. A slash-free pattern becomes
    # `**/<pattern>`, which is "a name at any depth"; one with a separator matches the path. Measured
    # across the pattern shapes that matter (brackets, question mark, wrong case, depths, folders):
    # treating every pattern as a path changed no answer. A flag that no input can distinguish is not a
    # branch worth keeping.
    rx = _path_rx(pattern)

    unread = 0

    def _unread(_error):
        nonlocal unread
        unread += 1

    wanted = max(1, int(offset))
    out: list[str] = []
    used = 0
    stopped = None
    seen = shown = examined = 0
    declined: set[str] = set()

    for root, dirs, names in os.walk(top, onerror=_unread, followlinks=False):
        hidden = [d for d in dirs if d in SKIP_DIRS]
        if hidden:
            declined.update(hidden)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        here = "" if os.path.abspath(root) == top else os.path.relpath(root, top).replace(os.sep, "/")
        for name in sorted(dirs) + sorted(names):
            examined += 1
            rel = f"{here}/{name}" if here else name
            hit = rx.match(rel)
            if not hit:
                continue
            seen += 1
            if seen < wanted:
                continue
            line = os.path.join(root, name) + ("/" if name in dirs else "")
            cost = len(line.encode("utf-8", "replace")) + (1 if out else 0)
            if used + cost > ingest.BODY_BYTES:
                stopped = "bytes"
                break
            if len(out) >= ingest.BODY_LINES:
                stopped = "lines"
                break
            out.append(line)
            used += cost
            shown += 1
        if stopped:
            break

    notes = []
    if declined:
        notes.append(f"{', '.join(sorted(declined))}/ not searched")
    if unread:
        notes.append(f"{unread} folder(s) could not be read")
    gap = " (" + "; ".join(notes) + ")" if notes else ""

    if not out:
        if 0 < seen < wanted:
            return (f"No entries at or after find offset={wanted} — {seen} match(es) "
                    f"before it{gap}.")
        return (f"Nothing under {top} matches {pattern!r} — {examined} entries looked at{gap}.")
    head = f"# {shown} match(es) for {pattern!r} under {top}"
    if stopped:
        meta = ingest.meta_fields(truncated=True, by=stopped, total_lines=None,
                                 total_bytes=used, shown_lines=len(out), shown_bytes=used)
        return f"{head}\n\n" + "\n".join(out) + "\n\n" \
               f"{ingest.notice(meta, extra=f'find offset={wanted + shown}')}{gap}"
    return f"{head}{gap}\n\n" + "\n".join(out)


find.description = ("Find files and folders by name, recursively, and return their absolute paths "
                    "one per line (folders marked with `/`). This reads no file contents, so it "
                    "costs names rather than bytes: the way to ask \"which files are called that\" "
                    "before spending a search or a read on them. `ls` answers that about one folder at "
                    "a time, and a search matches text rather than names. A capped page names the "
                    "`find offset=` that continues it; `.git` and `__pycache__` are not walked, "
                    "symlinks are not followed, and a folder that could not be read is counted in the "
                    "answer rather than walked past.")


# How these touch the world, declared beside the tools instead of tabled by name somewhere else: the
# policy judges `read` and `ls` against the read grants and `write` against the write grants. A tool that
# declared nothing now fails at boot (see `sah.contract.declaration`) rather than running unconfined.
read.access, read.address = "read", "path"
write.access, write.address = "write", "path"
edit.access, edit.address = "write", "path"
grep.access, grep.address = "read", "path"
find.access, find.address = "read", "path"
ls.access, ls.address = "read", "path"


class Fs:
    """The file tools, as one extension. Always offered: a run that cannot read or write a file can
    only talk. Whether a given path is allowed is the veto point's call on each use, not a reason to
    withhold the tool."""

    tools = (read, write, edit, grep, find, ls)
