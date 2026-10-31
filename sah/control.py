"""sah.control — the inbound command queue: one file per run, append-only.

The operator — a shell, another process, a human with an editor — appends blocks here; the
loop drains them at a turn boundary. The file only grows: a block is one append and a consume
marker is one append, and nothing rewrites, truncates or rotates. That is what makes the queue
usable when the harness is in trouble, and it is why `cat control.md` is the audit trail of
what anyone asked.

The format:

    --- block <id>
    ts: <ISO time>
    action: <any word; the loop decides whether it is declared>
    cancel: <true | false>
    from: <who the writer says they are — a claim, not an authority>
    reply: <the step id this block answers, when it answers one>
    prompt: <one line, optional>
    ---
    <the text, byte for byte>
    --- end <id>

    --- consumed <id>

- A block ends only at its own end line. A block with no end line was cut off by a crash and
  is reported as such rather than acted on.
- The queue does not validate actions. The loop validates once, at the point where a command
  becomes a step, so there is one answer to "what may this run be told to do".
- `from:` is recorded as data inside the command's arguments. The issuer of a queued command
  is the queue: a file can attest which queue a command came from, never who typed it, and a
  claim written by the sender cannot become the authority on who sent it.
- Every read and write holds the queue lock: a lock per path in this process, plus an OS lock
  on the file itself. There is no lock file.
"""
from __future__ import annotations

import os
import re
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

QUEUE_NAME = "control.md"
_BLOCK_HEAD = re.compile(rb"^--- block ([0-9a-f]+)\s*$")
# An action is one lower-case word. It is written into the header as `action: <x>`, so anything else --
# above all a newline -- writes extra header lines, and a message that can write header lines can write
# the block boundaries the queue's whole safety rests on.
_ACTION = re.compile(r"[a-z][a-z0-9_-]*")
_END_HEAD = "--- end "
_CONSUMED = re.compile(r"^--- consumed ([0-9a-f]+)$", re.M)


def queue_path(run_root):
    """The queue file of a run root."""
    return os.path.join(str(run_root), QUEUE_NAME)


class Block:
    """One queued command from a block: action, body, timestamp, the sender's claim, the step it
    answers, and whether the block was closed."""
    __slots__ = ("id", "ts", "action", "cancel", "prompt", "text", "sender", "reply", "closed")

    def __init__(self, id, ts, action, cancel, prompt, text, sender, reply, closed):
        self.id, self.ts, self.action = id, ts, action
        self.cancel, self.prompt, self.text = cancel, prompt, text
        self.sender, self.reply = sender, reply
        self.closed = closed

    def __repr__(self):
        return f"<{self.action} {self.id} closed={self.closed}>"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


# --- the lock ------------------------------------------------------------------------

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

# The Windows lock is mandatory for its byte range, so it sits past any size the queue
# reaches: the queue's own reads and appends are never blocked by it.
_LOCK_OFFSET = 0x7FFFFFFF


def _thread_lock(path: str) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _os_lock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.002)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _os_unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def queue_lock(path: str, *, create: bool = False):
    """Hold the queue: this process's lock on the path, and an OS lock on the file.

    With `create`, the file is made here with `O_CREAT` rather than opened only when it already exists. Two
    writers that both find no file hold only their own process's lock, and both append — an in-process lock
    cannot reach across a process boundary, and the race is exactly the window between "does it exist" and
    "open it". Creating under the lock closes that window and gives every hold, the first one included, a
    descriptor to lock. An empty queue is a normal state that reads as nothing pending, consistent with the
    rule that the queue only ever grows. The mode is `O_RDWR` because a consumer locks the same descriptor it
    truncates through; `0o644` is requested, but Windows sets the attributes it sets and nothing reads them.

    `create` belongs to the writer. A reader that found no queue has nothing to read and mints nothing —
    otherwise polling a run folder that was never created would leave a folder behind, which is not a read's
    ground to take."""
    with _thread_lock(path):
        if create:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        else:
            fd = os.open(path, os.O_RDONLY) if os.path.isfile(path) else None
        try:
            if fd is not None:
                _os_lock(fd)
            yield
        finally:
            if fd is not None:
                _os_unlock(fd)
                os.close(fd)


# --- the queue -----------------------------------------------------------------------

def append(run_root, action: str, *, text: str = "", sender: str = "",
           prompt: str = "", cancel: bool = False, reply: str = "") -> str:
    """Append one block and return its id.

    The block text may not contain a line that opens or closes a block: a message that could
    forge an end line could forge the block after it, and the queue is worth nothing if a
    message can write commands. The action is checked for the same reason: it is a header field.
    """
    if not _ACTION.fullmatch(action):
        raise ValueError(f"a queue action must be one word of lower-case letters, digits, '-' or '_'; "
                         f"got {action[:40]!r}")
    block_id = secrets.token_hex(4)
    end = _END_HEAD + block_id
    for line in text.split("\n"):
        stripped = line.rstrip("\r")
        if _BLOCK_HEAD.match(stripped.encode()) or _CONSUMED.match(stripped) or stripped == end:
            raise ValueError(f"control text may not contain a queue line ({stripped[:20]!r}); "
                             "send the text without a line starting '--- block', "
                             "'--- consumed' or '--- end'")
    head = f"--- block {block_id}\nts: {_now()}\naction: {action}\ncancel: {str(cancel).lower()}\n"
    if reply:
        head += "reply: " + " ".join(reply.splitlines()) + "\n"
    if sender:
        head += "from: " + " ".join(sender.splitlines()) + "\n"
    if prompt:
        head += "prompt: " + " ".join(prompt.splitlines()) + "\n"
    body = head + "---\n" + text.rstrip("\r\n") + "\n" + end + "\n"

    path = queue_path(run_root)
    with queue_lock(path, create=True):
        _append_lines(path, body)
    return block_id


def _append_lines(path: str, body: str) -> None:
    """Append to the queue, keeping every record on its own line.

    A crash can leave the file without a trailing newline. Writing a marker onto such a file
    glues it to the truncated line, where nothing can find it again — and a marker that is
    silently lost re-offers the block it was meant to retire, forever."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_newline = False
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as reader:
            reader.seek(-1, os.SEEK_END)
            needs_newline = reader.read(1) != b"\n"
    with open(path, "ab") as handle:
        if needs_newline:
            handle.write(b"\n")
        handle.write(body.encode("utf-8"))


def consume(run_root, block_id: str) -> None:
    """Record that a block was taken. Marking twice is harmless; the marker is just a line."""
    path = queue_path(run_root)
    with queue_lock(path):
        _append_lines(path, f"--- consumed {block_id}\n")


def pending(run_root) -> list[Block]:
    """The blocks with no consume marker, in file order. Reading marks nothing."""
    path = queue_path(run_root)
    if not os.path.isfile(path):
        return []
    with queue_lock(path):
        with open(path, "rb") as handle:
            raw = handle.read()
    done = _CONSUMED.findall(raw.decode("utf-8", "replace"))
    taken = set(done)
    return [block for block in _parse(raw) if block.id not in taken]


def _parse(raw: bytes) -> list[Block]:
    """Blocks in file order. A block with no end line is reported with closed=False."""
    lines = raw.split(b"\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        head = _BLOCK_HEAD.fullmatch(lines[i].rstrip(b"\r"))
        i += 1
        if head is None:
            continue
        block_id = head.group(1).decode("utf-8", "replace")
        fields: dict[str, str] = {}
        while i < len(lines) and lines[i].rstrip(b"\r") != b"---" \
                and _BLOCK_HEAD.fullmatch(lines[i].rstrip(b"\r")) is None:
            key, _, value = lines[i].rstrip(b"\r").decode("utf-8", "replace").partition(":")
            fields[key.strip()] = value.strip()
            i += 1
        if i < len(lines) and lines[i].rstrip(b"\r") == b"---":
            i += 1
        stop = (_END_HEAD + block_id).encode()
        body: list[bytes] = []
        closed = False
        while i < len(lines):
            line = lines[i].rstrip(b"\r")
            if line == stop:
                closed = True
                i += 1
                break
            if _BLOCK_HEAD.fullmatch(line) is not None:
                break                       # the next block began: this one was cut off
            body.append(line)
            i += 1
        blocks.append(Block(block_id, fields.get("ts", ""), fields.get("action", ""),
                            _parse_bool(fields.get("cancel", "false")),
                            fields.get("prompt") or "",
                            b"\n".join(body).decode("utf-8", "replace"),
                            fields.get("from", ""), fields.get("reply", ""), closed))
    return blocks
