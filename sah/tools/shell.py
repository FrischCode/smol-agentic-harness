"""sah.tools.shell — the tool that runs a command, and the one tool that streams.

One function. It is kept small because a shell tool is the largest surface in a harness: it can do
anything the user can, which is exactly why it is the one tool whose permission is a separate axis
in `sah.confine` and whose every call carries a disclosure. This module does not decide whether a
command may run — the policy does, before this is ever called — it only runs one, well.

It is the one tool that yields a *stream* rather than returning a value. The reason a compile,
render or download must not go silent for a witness: a plain tool reports only when it finishes, so
a long command is indistinguishable from a hung one until it ends. `bash` therefore yields output as it
arrives, in whatever pieces the pipe delivers them (the loop streams them live into the tool's box) and
`return`s the assembled
report at the end — so what streams to a terminal and what gets recorded can differ: the console
sees progress, the log and the model get the structured report below. A tool that produced no
output, or a shell that failed to start, is simply a stream of zero chunks whose return value is the
whole result — the same protocol, the degenerate case.

What a shell tool must get right, and does:

- **A command that outruns its timeout dies with everything it spawned.** The child leads its own
  process group, so `process.kill_tree` reaches a pipeline or a script's children too. A timeout that
  leaves a `sleep 600` holding the terminal is a timeout that failed.
- **A stream nobody finishes draining kills its command.** This is the one tool that yields, so the
  consumer can stop early: an interrupt closes it, a consumer's exception arrives inside it, a generator
  nobody drains is closed when collection finds it. And the child leads its own group, which is what the
  timeout needs and also means no terminal signal reaches it. So the `finally` in `_stream` is the last
  watcher the command has.
- **The command is given no stdin.** Its input is the null device, not this process's console, so a
  command that reads input (`python` with no arguments, a `pause` in a script, a `git` credential prompt)
  meets an immediate EOF instead of taking the keystrokes the operator is typing at the REPL or blocking
  until the deadline. A tool call has no input to give.
- **The output is for a model to read.** stdout and stderr stay distinct (a traceback in a heap of
  stdout loses its shape) and a killed command says so rather than looking like it succeeded quietly.
- **What the tool holds is bounded; what streams is not.** Every piece still reaches the caller live,
  but each stream is retained only at its two ends, and the report states how many bytes fell between
  them and what to do about it.
- **The exit code is the LAST line, not the first.** Command output is the one result that is
  end-loaded for the model — the error is at the bottom — so both the tool's own cap and any clipping
  it still undergoes keep the end. If the exit code were at the top it would be the first thing
  clipped away, and it is the one fact that says whether to trust the output. So it goes last, where
  the tail keeps it.

The tool keeps the first 8 KB and the last 16 KB of each stream and drops the middle, saying so in the
report in bytes and naming the way back: run the command again redirecting to a file and read the
file. The alternative — hold every byte so the loop's ingest point can bound the result once, at the
end — protects the model's context by exhausting the harness's memory while a `yes` runs, which is
the wrong thing to protect first. The loop's ingest point stays the one place that bounds what enters
the context; this cap bounds only what the tool holds while a command runs, and `bash` still declares
via `ingest_keep` that its output is read from the end.
"""
from __future__ import annotations

import codecs
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Annotated, NamedTuple

from sah import ingest, process
from sah.contract import Param

# A named handle on the clock, so the timeout path can be exercised without sleeping for its full
# duration — the only reason this is indirect rather than a bare time.monotonic() call.
_MONO = time.monotonic

# One decoding policy, named, because there is now exactly one place that decodes and a reader must be able
# to find the rule. A byte that does not decode is shown as the byte it is (`\\x82`) rather than replaced
# with U+FFFD: the escape is nameable and routable around; the replacement character reads like text.
DECODE_ERRORS = "backslashreplace"
READ_BYTES = 65536     # a read's ceiling, not its expectation: `read1` returns whatever is already there

# The shell to run commands through, and the shape it is given. Windows has no /bin/sh, and "a shell"
# means different programs with different syntax and different parsers -- so the choice lives in one
# place, and is *named to the model*, because a tool called `bash` that runs cmd.exe invites POSIX syntax
# and turns spent discovering otherwise.
_GIT_BASH_HINT = ("This runs in Git Bash: POSIX syntax works (`ls`, `grep`, `$(...)`, single quotes), and "
                  "a Windows path is written `C:/Users/you` or `/c/Users/you`.")
_CMD_HINT = ("This runs in cmd.exe, not bash: use `dir`, `type`, `findstr`, `%VAR%` and double quotes. "
             "`ls`, `grep`, single quotes and `$(...)` are not syntax here. A bare `find`, `sort` or "
             "`xargs` may resolve to a Git-for-Windows build with different arguments, so name the "
             "System32 tool when it matters.")
_SH_HINT = "This runs in /bin/sh (POSIX)."


class Shell(NamedTuple):
    """The shell a run actually gets: its name, how it parses, where it lives, and the sentence the model
    needs in order to write syntax that works. All four are reported, because the name alone -- `bash` --
    is the thing that misleads."""

    name: str
    family: str                                        # "posix" or "cmd"
    path: str
    hint: str = ""

    def argv(self, command: str):
        """The argv for this shell, in the form *its* parser reads.

        **cmd.exe gets one string, assembled here.** Handing `Popen` a list makes it run `list2cmdline`,
        which escapes every `"` as `\\"`; cmd has no backslash escapes, so the backslashes arrive as text.
        The measured failure mode: `echo "hello world"` printed `\\"hello world\\"`, and
        `dir "C:\\Program Files" | find /c "Microsoft"` lost its quotes badly enough that `find` took `/c`
        as a path and walked the whole C: drive -- a mangled quote did not fail, it did something else,
        expensively. With `/s`, cmd strips exactly the outer pair of quotes and runs the rest verbatim, so
        the quotes the caller wrote are the quotes the command gets. `/d` disables AutoRun, which could
        otherwise inject someone else's command.

        **A POSIX shell gets the list.** Its runtime decodes that same escaping correctly (MSYS emulates
        MSVCRT argv parsing), so `list2cmdline` is right here, and a hand-built string would be mis-split
        instead. `/bin/sh -c` likewise takes the command as one argument.
        """
        if self.family == "posix":
            return [self.path, "-c", command]
        return f'"{self.path}" /d /s /c "{command}"'


def _under_system32(path: str) -> bool:
    """True for anything under the Windows system directory."""
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    return os.path.normcase(path).startswith(os.path.normcase(os.path.join(root, "system32")))


def _is_wsl_launcher(path: str) -> bool:
    """A `bash`/`sh` from the Windows system directory is the WSL launcher, not a POSIX shell over this
    machine's files: the command would run in a Linux VM where the run's paths, its venv and its file
    grants all mean something else, and it is refused even when an operator names it, because that choice
    is almost certainly not "run my build inside the VM".

    The directory alone is not the test. cmd.exe lives in the same System32 and is the right fallback, and
    an operator may name it on purpose; refusing it would make an override that looks applied and silently
    does not, which is worse than having no override at all.
    """
    name = os.path.normcase(os.path.basename(path))
    return _under_system32(path) and name in ("bash.exe", "sh.exe")


def _git_bash() -> str:
    """Git's own bash, found through git rather than through PATH. `bin\\bash.exe` is the launcher that
    sets up the coreutils PATH; `usr\\bin\\bash.exe` is the bare interpreter and is not what is wanted."""
    candidates = []
    git = shutil.which("git")
    if git:
        root = os.path.dirname(os.path.dirname(git))          # …\Git\cmd\git.exe -> …\Git
        candidates.append(os.path.join(root, "bin", "bash.exe"))
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(key)
        if base:
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
    for cand in candidates:
        if not _under_system32(cand) and os.path.isfile(cand):
            return cand
    return ""


def resolve_shell() -> Shell:
    """Decide once which shell a run gets, in this order: an operator's `SAH_SHELL`, then Git Bash, then
    cmd.exe (and on POSIX, `/bin/sh`).

    PATH is asked for `git` and never for `bash`. On this class of machine `which("bash")` answers
    `C:\\WINDOWS\\system32\\bash.exe` -- the WSL launcher -- which would run every command in a Linux VM
    where `C:\\…` paths, the venv and the run's grants all mean something different. Git Bash is located
    through git's own installation instead, which cannot answer with that impostor.
    """
    if sys.platform != "win32":
        # The documented order starts with the operator's `SAH_SHELL` on every platform. A forced
        # shell the resolver silently drops makes the operator believe they chose (the same defect
        # the Windows branch fixed), and a check that forces its shell ends up testing /bin/sh.
        chosen = os.environ.get("SAH_SHELL")
        if chosen:
            return Shell(os.path.basename(chosen), "posix", chosen,
                         f"This runs in {os.path.basename(chosen)}, the shell the operator chose; "
                         "POSIX syntax works (`ls`, `grep`, `$(...)`, single quotes).")
        return Shell("/bin/sh", "posix", "/bin/sh", _SH_HINT)

    chosen = os.environ.get("SAH_SHELL")
    if chosen and not _is_wsl_launcher(chosen):
        posix = os.path.normcase(os.path.basename(chosen)) in ("bash.exe", "sh.exe", "bash", "sh")
        hint = (f"This runs in {os.path.basename(chosen)}, the shell the operator chose; "
                + ("POSIX syntax works (`ls`, `grep`, `$(...)`, single quotes)." if posix else
                   "it is not a POSIX shell, so use its own syntax."))
        return Shell(os.path.basename(chosen), "posix" if posix else "cmd", chosen, hint)

    bash = _git_bash()
    if bash:
        return Shell("Git Bash", "posix", bash, _GIT_BASH_HINT)

    comspec = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
    return Shell("cmd.exe", "cmd", comspec, _CMD_HINT)


SHELL = resolve_shell()


def _argv(command: str):
    """The argv this run's shell needs -- `SHELL.argv`, reached through a function so the argv can be
    inspected as a pure value without starting a shell. See `Shell.argv` for why the two families need
    different forms."""
    return SHELL.argv(command)


def bash(command: Annotated[str, Param("shell command to run")],
         cwd: Annotated[str, Param("working directory to run in; NOT a sandbox — the command "
                                   "can reach anything the account can")] = "",
         timeout: Annotated[int, Param("seconds before the command and everything it spawned are "
                                       "killed")] = 120) -> str:
    """`cwd` is where it runs, not a limit on what it can touch: the run's file grants do not bind a
    shell, which is exactly why being allowed to run anything is a separate permission from being
    allowed to read or write a path."""
    return _stream(command, cwd or None, max(1, int(timeout)))


bash.description = ("Run a shell command, returning stdout, stderr and the exit code (last). Each "
                    "stream is kept at its two ends only — roughly its first 8 KB and last 16 KB — and "
                    "the report says how many bytes between them were dropped; to see them, run the "
                    "command again redirecting its output to a file (`<command> > file 2>&1`) and read "
                    "that file. A timeout kills the command's whole process tree and keeps the partial "
                    "output. The command ends with the call: a process left running in the background "
                    "(`server &`) is stopped when the command finishes, so start anything that must "
                    "outlive it with a spawned child run rather than `&`. <shell>")


# The tool is named `bash`, so the dialect it really runs has to be stated in its own contract, where a
# model reads it before writing the first command -- not in a banner only the operator sees. The slot is
# filled from the same resolution the argv is built from, so the sentence and the behaviour cannot drift.
bash.description = bash.description.replace("<shell>", SHELL.hint, 1)


# What the tool keeps of a stream while a command runs. The loop bounds a result only once the tool
# has finished, which is too late to protect the run: holding every byte a chatty build emits is how
# the harness runs out of memory instead of the command. So the two ends are kept and the middle is
# dropped, and the report says exactly what went missing and how to get it. The budgets derive from
# the loop's own limit so the report and its notices arrive underneath it rather than being clipped
# a second time, in the middle of a notice.
_HEAD_BYTES = ingest.BODY_BYTES // 8               # 8 KB of the opening
_TAIL_BYTES = ingest.BODY_BYTES // 4               # 16 KB of the ending: the error is at the end
# How long the reader threads get to finish after the command has ended -- one bound shared by both,
# because the command is over and anything that could hold a pipe open has been stopped. Generous enough
# for a pipe still draining its last buffer, short enough that a stuck reader cannot turn a 6s timeout
# into a 16s wait.
JOIN_SECONDS = 2.0


class _Ends:
    """The first and last bytes of one stream, and the size of what fell between them.

    Until the budget is passed the text is kept whole, so an ordinary command's output is untouched
    and reports no elision. Past it only the ends are held: a command that produces ten megabytes
    costs this tool its budget, not ten megabytes. A line longer than the tail budget is cut to the
    part kept, so the bound holds for one enormous line as for a million small ones.

    Every line still streams to the caller; the cap is on what is retained, never on what is shown."""

    def __init__(self, head_bytes: int = _HEAD_BYTES, tail_bytes: int = _TAIL_BYTES):
        self.head_bytes, self.tail_bytes = head_bytes, tail_bytes
        self.kept: list[str] = []                  # everything, while it still fits
        self.head = ""                             # then: the opening
        self.tail: list[str] = []                  # and a rolling window of the ending
        self._kept_len = self._tail_len = 0
        self.bytes = 0

    @property
    def cut(self) -> bool:
        """Whether the budget was exceeded and the middle was dropped (only the ends are held)."""
        return bool(self.head)

    def add(self, piece: str) -> None:
        """One piece of a stream, which is not a line: it may be half a line or three, depending on what the
        pipe held. Retention counts bytes for exactly that reason."""
        self.bytes += len(piece)
        if not self.cut:
            self.kept.append(piece)
            self._kept_len += len(piece)
            if self._kept_len > self.head_bytes + self.tail_bytes:
                whole = "".join(self.kept)
                self.head = whole[: self.head_bytes]
                self.tail = [whole[-self.tail_bytes:]]
                self._tail_len = len(self.tail[0])
                self.kept = []                     # the middle stops being held here
            return
        kept = piece[-self.tail_bytes:]        # one piece can exceed the whole tail budget
        self.tail.append(kept)
        self._tail_len += len(kept)
        while self._tail_len > self.tail_bytes and len(self.tail) > 1:
            self._tail_len -= len(self.tail.pop(0))

    def text(self, label: str) -> str:
        """The kept text, with a notice naming the gap and the way back to it."""
        if not self.cut:
            return "".join(self.kept)
        gone = self.bytes - len(self.head) - self._tail_len
        return (self.head + f"\n[{label}: {self.bytes:,} bytes in all; {gone:,} bytes between the "
                f"first {self.head_bytes // 1024} KB and the last {self.tail_bytes // 1024} KB were "
                "dropped. To see them, run the command again redirecting its output to a file "
                "(`<command> > file 2>&1`) and read that file.]\n" + "".join(self.tail))


def _stream(command: str, cwd, seconds: int):
    """Run the command and yield output lines as they arrive; `return` the assembled report.

    The subprocess is started here (lazily, on first iteration, which is when the loop begins
    draining). Two reader threads keep the OS pipes drained so a chatty child never blocks on a full
    pipe while we are between yields; each line they read goes on a queue and is yielded out in
    arrival order. A start failure is a stream of zero chunks whose return value explains itself."""
    argv = _argv(command)
    boundary = process.Boundary(group=True)   # the child is started as its group leader, below
    try:
        # stdin is DEVNULL, not inherited. A command that reads input would otherwise be attached to
        # this process's own console: `python` with no arguments, a `pause` in a script, a `git`
        # credential prompt. Each of those either takes keystrokes the operator was typing into the
        # REPL or blocks until the timeout, and a command that cannot block on input is the only kind
        # a watcher can be sure of. A tool call has no input to give; EOF is the honest answer.
        proc = subprocess.Popen(argv, cwd=cwd,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                # Read as BYTES and decoded in `reader`, never asked of text mode as lines.
                                # Text mode is the convenient shape and it rewrites the output: its default
                                # `newline=None` is universal newlines, which turns CR and CRLF into LF on the
                                # way in, so a command that wrote ` 10%\r 40%\r` is reported as having written
                                # ` 10%\n 40%\n` — two completed lines where one line was redrawn.
                                # Measured against a real child, because it reads like a detail. This is the
                                # same rule the module already holds for an undecodable byte (show it, do
                                # not smooth it): a line ending is a fact about what ran, and a model told
                                # that CRLF was LF cannot reason about line endings at all.
                                #
                                # Decoding stays UTF-8 with `backslashreplace`, chosen here rather than
                                # sniffed, because what a command writes is not ours to pick: cmd's own output
                                # follows the console code page, a Windows python follows the ANSI one, and a
                                # tool may opt into UTF-8. A byte that will not decode is therefore a fact
                                # about the command, not damage to repair.
                                **process.popen_detach_kwargs(), **boundary.popen_kwargs)
    except OSError as exc:
        # A tool that explains is one the model can act on: zero chunks, and this string is the record.
        return f"could not start the shell: {exc}"
    boundary.attach(proc)

    pieces: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def reader(pipe, label: str) -> None:
        """Push one stream's decoded pieces on the queue in arrival order, then close the pipe.

        `read1` rather than `readline`: it returns whatever the OS already holds and waits only when there
        is nothing, so output is never held back until the command happens to end a line. It is not
        CR-blind: universal newlines ends a line at CR, and a CR-only bar does stream — the line-ending
        issue above is the real one.

        One decoder per stream, kept across reads, because a chunk boundary is not a character boundary: the
        two bytes of one character can arrive in two pieces, and the second means nothing without the state
        the first left. At the end the decoder is closed, which turns a tail that never became a character
        into the bytes it is instead of losing them.
        """
        decode = codecs.getincrementaldecoder("utf-8")(DECODE_ERRORS)
        try:
            while True:
                try:
                    raw = pipe.read1(READ_BYTES)
                except (OSError, ValueError):
                    break                      # the pipe closed under us; the stream is as done as it gets
                if not raw:
                    break                      # EOF
                text = decode.decode(raw)
                if text:
                    pieces.put((label, text))
        finally:
            tail = decode.decode(b"", True)
            if tail:
                pieces.put((label, tail))
            try:
                pipe.close()
            except Exception:
                pass

    threads = [threading.Thread(target=reader, args=(proc.stdout, "out"), daemon=True),
               threading.Thread(target=reader, args=(proc.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()

    out: _Ends = _Ends()
    err: _Ends = _Ends()
    killed = False
    stopped_background = False
    try:
        deadline = _MONO() + seconds
        while True:
            try:
                label, piece = pieces.get(timeout=0.05)
            except queue.Empty:
                label = piece = None
            if piece is not None:
                (out if label == "out" else err).add(piece)
                yield piece                              # live progress to whoever is watching
            # The clock is consulted on every pass, including a pass that showed something. A check that
            # lives in the empty-queue branch is unreachable for a command that prints without a gap —
            # `yes`, a tailed log, a chatty build — which is exactly the command a deadline exists for:
            # it streamed millions of lines into memory while we waited for a quiet moment to notice.
            if _MONO() > deadline:
                killed = True
                process.kill_tree(proc, detached=True)
                break
            # Finished when nothing showed this pass AND the process exited.
            if piece is None and proc.poll() is not None:
                if not any(t.is_alive() for t in threads):
                    break
                # The command is over and its exit code is real; a reader still blocked is blocked on a
                # pipe whose write end is held by something the command left running in the background --
                # `cmd &`, `start /b`. Waiting for that EOF is the defect this branch replaces: a clean
                # `echo ready` spun to the deadline, had its exit code 0 replaced by `killed`, and then
                # killed a tree whose shell was already gone, leaving the holder alive. Stopping the
                # boundary ends what was left behind, the write ends close, and the code stands.
                boundary.stop()
                stopped_background = True

        # One bound for both readers, not one each: two `join(KILL_WAIT)` calls let a tool whose timeout
        # was 6s take 16s to return, which is a deadline that is a suggestion. Nothing should be left
        # holding a pipe by now -- the command ended and, above, so did anything it left behind.
        join_by = _MONO() + JOIN_SECONDS
        for t in threads:
            t.join(timeout=max(0.0, join_by - _MONO()))
            while not pieces.empty():                    # catch anything they read after the break
                label, piece = pieces.get()
                (out if label == "out" else err).add(piece)

        return _report(out, err, killed=killed, code=None if killed else proc.returncode,
                       seconds=seconds, stopped_background=stopped_background,
                       reached=boundary.teeth)
    finally:
        # A stream can be abandoned: the loop closes it when a run is interrupted, an exception in a
        # consumer arrives here, and a generator nobody drained is closed when garbage collection finds
        # it. The child leads its own process group so that a stray Ctrl+C cannot reach it — which also
        # means nothing else reaches it, so without this the command outlives its watcher entirely and
        # keeps holding files and the GPU. Kill it here, whatever the reason the drain stopped; the
        # timeout path above already did it, and `kill_tree` returns at once for a stopped process.
        #
        # The boundary is stopped unconditionally, unlike the direct-child kill: its whole purpose is the
        # descendant that outlives the shell, and a `poll()` guard would skip exactly that case.
        boundary.stop()
        boundary.close()
        if proc.poll() is None:
            process.kill_tree(proc, detached=True)


def _report(stdout: "_Ends", stderr: "_Ends", *, killed: bool, code, seconds: int,
            stopped_background: bool = False, reached: bool = True) -> str:
    """Assemble the report from what each stream kept, with the exit code LAST — see the module note
    for why the order is a correctness choice, not style."""
    # A trailing line ending is not content, and now that the endings are the command's own that includes
    # the CR a progress bar ends on. Everything inside is kept exactly as it was written.
    stdout = (stdout.text("stdout") or "").rstrip("\r\n")
    stderr = (stderr.text("stderr") or "").rstrip("\r\n")
    parts = []
    if stdout:
        parts.append("stdout:\n" + stdout)
    if stderr:
        parts.append("stderr:\n" + stderr)
    if not stdout and not stderr:
        parts.append("(no output)")
    if killed:
        # The claim follows the mechanism. With a job object (or, off Windows, a group this child leads)
        # everything below the shell really is gone; without one only the direct child was stopped, and
        # saying otherwise would send the model off trusting a server that is still running.
        parts.append(f"exit: killed — still running after {seconds}s; "
                     + ("everything it spawned went with it, output below is partial"
                        if reached else
                        "the command was stopped, but a process it started may still be running "
                        "(this host gave no process-tree boundary); output below is partial"))
    else:
        if stopped_background:
            # Said rather than hidden: something may have died here, and a model that believes its server
            # is still up will spend the next turn wondering why nothing answers.
            parts.append("note: a process this command left running in the background "
                         + ("was stopped with it. The tool ends with the command; start things that must "
                            "outlive it under a tool that keeps them (a spawned child run), or redirect "
                            "their output to a file" if reached else
                            "may still be running: this host gave no process-tree boundary, so only the "
                            "command itself was stopped"))
        parts.append(f"exit: {code}" + ("" if code == 0 else "  (non-zero — the command failed)"))
    return "\n\n".join(parts)


# Command output is read from the end: the error, not the echoed banner, is what matters. The loop's
# ingest point reads this to clip an over-cap result to its tail rather than its head.
bash.ingest_keep = "tail"
# the exec axis; a working directory is a start point, not a limit
bash.access, bash.address = "exec", "cwd"


class Exec:
    """The shell, as one extension. The run includes it only when `--exec` was given — a tool the
    policy would refuse on every call is not a capability but noise in the schema — and when it is
    included, every call still carries the exec axis's disclosure from the veto point."""

    tools = (bash,)
