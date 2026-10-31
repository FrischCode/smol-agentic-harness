"""The shell tool: exit code always stated, output bounded both ends, a timeout that kills the
tree, and — since it is the one streaming tool — output that reaches the caller live. Nothing here
spawns a real process; `subprocess.Popen` and `process.kill_tree` are replaced with fakes, and the
tool's clock (`shell._MONO`) with one that skips ahead so the timeout path costs no real seconds."""

import os
import subprocess
import sys
import threading
import time

import pytest

from sah import ingest, process
from sah.tools import shell


@pytest.fixture(autouse=True)
def _boundaries_are_fake(monkeypatch):
    """No test in this module may signal a process group. The boundary fake is installed for every test
    and cleared, so a `made` list never carries a previous test's boundary into the next assertion.
    """
    FakeBoundary.made = []
    monkeypatch.setattr(process, "Boundary", FakeBoundary)
    yield


class Lockstep:
    """Force two reader threads to take turns, one chunk each.

    A test about state leaking between the two streams cannot be written by hoping the threads overlap: a
    handful of in-memory reads finish in microseconds, the interpreter sees no reason to switch, and the
    first reader drains its whole stream before the second is scheduled. That is a test which passes for the
    bug and for the fix, and it did exactly that once, measured. So the turn-taking is made a property of the
    doubles rather than left to the scheduler.

    A broken barrier is swallowed on purpose: when one stream runs out, the other is on its own and must
    finish rather than wait for a partner that has gone.
    """

    def __init__(self, partners=2, limit=5.0):
        self._barrier = threading.Barrier(partners, timeout=limit)

    def step(self) -> None:
        try:
            self._barrier.wait()
        except Exception:
            pass


class FakePipe:
    """A pipe that hands back canned bytes then hits EOF.

    `hold` reproduces the shape a background grandchild creates: the command has finished, but the write
    end of this pipe is still held, so a read blocks instead of reaching EOF. `release` is what
    stopping the boundary does to it in reality -- the holder dies and the pipe closes -- so the fake
    unblocks by the same mechanism, not by the test deciding to stop waiting.
    """

    def __init__(self, text, hold=False, chunks=None, step=None):
        """Bytes in chunks, because that is what a pipe delivers.

        Canned `text` is split on newlines only so the older tests can keep naming lines; a test about where
        a piece begins says so with `chunks=`, which is the shape the tool actually sees. There is no
        `readline` here: a reader that asks for lines would fail loudly against this double instead of
        quietly passing while the tool translated the command's line endings.
        """
        self._chunks = ([c if isinstance(c, bytes) else c.encode("utf-8") for c in chunks]
                        if chunks is not None
                        else [ln.encode("utf-8") for ln in text.splitlines(keepends=True)])
        self._i = 0
        self.closed = False
        self._hold = hold
        self._released = False
        self._step = step                                  # takes a turn with a partner stream, if any

    def release(self):
        self._released = True

    def read1(self, n):
        """Whatever is available, up to n bytes -- the real method's contract, including that it waits
        for nothing when there is nothing buffered to give."""
        if self._step is not None:
            self._step.step()
        if self._i < len(self._chunks):
            chunk = self._chunks[self._i]
            self._i += 1
            return chunk
        if self._hold:
            while not self._released:
                time.sleep(0.005)
            self._hold = False
        return b""                                   # EOF


    def close(self):
        self.closed = True


class FakeFirehose:
    """A pipe with no gap in it: `n` lines, always one more ready. Pair it with a process that stays
    alive and the queue is never empty, which is the shape a deadline tucked into the empty-queue
    branch can never see: `yes`, a tailed log, a chatty build."""

    def __init__(self, n=5000, line="chunk\n"):
        self._left, self._line = n, line
        self.closed = False

    def read1(self, n):
        if self._left <= 0:
            return b""                               # EOF only once the canned flood is spent
        self._left -= 1
        return self._line.encode("utf-8")

    def close(self):
        self.closed = True


class FakeBoundary:
    """The tree boundary as the shell tool sees it, recording what it was asked to do.

    The fake releases the pipes on `stop()` because that is what a real stop does: the holder dies, its
    write end closes, and the reader reaches EOF. A fake that just counted calls would let the tool pass
    these tests while the real mechanism left the pipe open.

    Patching it in is not convenience. The real boundary signals a process group, and a test double's pid
    is not a process group -- an unpatched fake pid could be somebody else's.
    """

    made: list = []
    RELEASE = True
    TEETH = True                                   # a test may need a boundary that cannot stop the holder

    def __init__(self, *, group=False, release=True):
        self.group = group
        self.release = release and FakeBoundary.RELEASE
        self.stops = 0
        self.closes = 0
        self.attached_to: list = []
        FakeBoundary.made.append(self)

    @property
    def popen_kwargs(self):
        return {}

    @property
    def teeth(self):
        # What the report is allowed to claim. A test that wants the honest-degradation wording sets this
        # False, which is the shape of a Windows host where the job could not be made.
        return FakeBoundary.TEETH

    def attach(self, proc):
        self.attached_to.append(proc)

    def stop(self):
        self.stops += 1
        if not self.release:
            return
        for proc in self.attached_to:
            for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
                if hasattr(pipe, "release"):
                    pipe.release()

    def close(self):
        self.closes += 1


def _as_pipe(value, *, hold=False, firehose=0):
    """Canned text becomes a pipe; a pipe the test built is used as given.

    Wrapping a pipe in a pipe would silently discard the chunk shape the test asked for, and a test about
    chunk boundaries is only about chunk boundaries if the chunks are the ones it wrote."""
    if hasattr(value, "read1"):
        return value
    if firehose:
        return FakeFirehose(firehose)
    return FakePipe(value, hold=hold)


class FakeProc:
    """A child with canned output. `hang` keeps it 'alive' (poll stays None until killed) so a
    timeout can be driven; the canned stdout is still read first, so partial output is real."""

    def __init__(self, stdout="", stderr="", code=0, hang=False, firehose=0, hold=False):
        # `hold` holds *both* streams. A background holder inherits the descriptor the command was given,
        # and the tool has two of them; a fake that blocks only one leaves a per-thread wait looking
        # identical to a shared one, which is how a mutation of the join bound survived its own test.
        self.stdout = _as_pipe(stdout, hold=hold, firehose=firehose)
        self.stderr = _as_pipe(stderr, hold=hold)
        self.pid = 4242                              # a process without an id is not a process
        self._hang = hang
        self.returncode = None if hang else code
        self.killed = False

    def poll(self):
        if self._hang and not self.killed:
            return None                            # still running
        if self._hang:
            # a killed child is a stopped child: poll() reaps it with a non-zero status. Without this
            # the fake stays "alive" after kill_tree, which no real process does, and a second kill
            # looks legitimate to code that guards on poll().
            return -1
        return self.returncode


def drain(stream):
    """Consume a tool's stream the way the loop does: yield-side, return-side. Returns (recorded,
    streamed_text) — the generator's return value and the concatenation of what it yielded live."""
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            return stop.value, "".join(chunks)


def fake_clock(first=0.0, later=1e9):
    """A monotonic source that returns `first` once (to set the deadline) then jumps far ahead, so
    the timeout branch fires without waiting."""
    ticks = iter([first] + [later] * 10_000)
    return lambda: next(ticks)


class _Spy:
    def __init__(self):
        self.argv = None
        self.kwargs = None
        self.killed = []


def spy(monkeypatch, proc):
    s = _Spy()

    def fake_popen(argv, **kwargs):
        s.argv, s.kwargs = argv, kwargs
        return proc

    monkeypatch.setattr(shell.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(process, "kill_tree",
                        lambda p, *, detached: (setattr(p, "killed", True), s.killed.append(detached)))
    return s


def test_exit_zero_is_stated_and_output_reported(monkeypatch):
    s = spy(monkeypatch, FakeProc(stdout="hello\n", code=0))
    out, _ = drain(shell.bash("echo hello"))
    assert out.rstrip().endswith("exit: 0")        # exit code LAST so tail-clipping keeps it
    assert "stdout:\nhello" in out
    assert s.kwargs["stdout"] is subprocess.PIPE


def test_output_streams_live_as_well_as_being_reported(monkeypatch):
    """The whole reason bash is a generator: the console sees the lines as they arrive, while the
    recorded result stays the structured report. Display and truth, from one run."""
    spy(monkeypatch, FakeProc(stdout="one\ntwo\n", code=0))
    out, streamed = drain(shell.bash("build"))
    assert "one\n" in streamed and "two\n" in streamed          # live
    assert "stdout:\none\ntwo" in out                           # assembled report


def test_the_command_goes_as_one_string_to_the_shell(monkeypatch):
    s = spy(monkeypatch, FakeProc(code=0))
    drain(shell.bash("cat a.txt | grep x > y"))
    assert "cat a.txt | grep x > y" in s.argv, "the shell receives the command whole, not split"


def test_windows_quotes_reach_the_command_untouched():
    """cmd.exe has no backslash escapes, so a command line assembled for it must not have any inserted.

    Given a list, `Popen` runs `list2cmdline`, which escapes each `"` as `\\"` -- and cmd reads those
    backslashes as text. Measured on this machine before the fix: `echo "hello world"` printed
    `\\"hello world\\"`, `findstr /c:"c d"` split at the space and failed, and
    `dir "C:\\Program Files" | find /c "Microsoft"` lost its quotes badly enough that `find` took `/c`
    as a path and began walking the whole C: drive. The last case is the reason this is not cosmetic:
    a mangled quote did not fail, it did something else, expensively.

    With `/s`, cmd strips exactly the outer pair of quotes and runs the rest verbatim, so the quotes the
    caller wrote are the quotes the command gets."""
    # Asserted on a cmd-shaped Shell rather than on this machine's resolved shell: which shell this box
    # has is a separate fact, and on this one it is Git Bash.
    command = 'echo "a b"'
    argv = shell.Shell(name="cmd.exe", family="cmd",
                       path=r"C:\Windows\System32\cmd.exe").argv(command)
    assert isinstance(argv, str), "cmd.exe takes one string; a list is the form that gets re-escaped"
    assert argv.endswith(f'/d /s /c "{command}"'), argv
    assert '\\"' not in argv, "no backslash-quote: cmd would print it literally"


def test_the_shell_an_operator_configured_is_the_one_used(monkeypatch):
    """ComSpec is honoured when the resolver falls back to cmd, and the known path when there is none.
    On POSIX there is no ComSpec: with no operator shell the fallback is /bin/sh."""
    if sys.platform != "win32":
        monkeypatch.delenv("SAH_SHELL", raising=False)
        assert shell.resolve_shell().path == "/bin/sh"
        return
    monkeypatch.delenv("SAH_SHELL", raising=False)
    _fake_fs(monkeypatch, which={"git": None}, files=[])
    monkeypatch.setenv("ComSpec", "X:/tools/cmd.exe")
    assert shell.resolve_shell().argv("dir").startswith('"X:/tools/cmd.exe" /d /s /c ')
    monkeypatch.delenv("ComSpec")
    assert shell.resolve_shell().argv("dir").startswith('"C:\\Windows\\System32\\cmd.exe"')


# --- which shell the model is actually getting (SHL-5) -------------------------------------------

def _fake_fs(monkeypatch, *, which=None, files=(), asked=None):
    """Point the resolver at a pretend filesystem: `which` is what PATH would answer, `files` are the
    paths that exist. Everything the resolver decides, it decides from these two. `asked` records which
    program names PATH was consulted for, so a test can insist that some are never asked."""
    import shutil

    def fake_which(name):
        if asked is not None:
            asked.append(name)
        return (which or {}).get(name)

    monkeypatch.setattr(shutil, "which", fake_which)
    existing = {os.path.normcase(p) for p in files}
    monkeypatch.setattr(shell.os.path, "isfile", lambda p: os.path.normcase(p) in existing)


def test_git_bash_is_preferred_when_it_is_really_there(monkeypatch):
    """Git Bash takes real POSIX syntax, which is what a model asked for a `bash` tool is going to write,
    and its runtime parses the Windows command line with MSVCRT rules -- so unlike cmd it is not broken by
    the quoting `Popen` applies. It is found via git's own location, not via PATH, because PATH on this
    machine answers with something else entirely. On POSIX the counterpart fact: with no operator shell
    it is /bin/sh, named as such, and the resolver never consults PATH at all."""
    if sys.platform != "win32":
        monkeypatch.delenv("SAH_SHELL", raising=False)
        asked: list[str] = []
        _fake_fs(monkeypatch, which={"git": "/usr/bin/git", "bash": "/usr/bin/bash"},
                 files=[], asked=asked)
        s = shell.resolve_shell()
        assert s.path == "/bin/sh" and s.name == "/bin/sh"
        assert asked == [], "PATH is never consulted for a shell on POSIX"
        return
    monkeypatch.delenv("SAH_SHELL", raising=False)
    _fake_fs(monkeypatch, which={"git": r"X:\Git\cmd\git.exe"},
             files=[r"X:\Git\bin\bash.exe"])
    s = shell.resolve_shell()
    assert s.family == "posix" and os.path.normcase(s.path) == os.path.normcase(r"X:\Git\bin\bash.exe")
    assert s.name == "Git Bash"


def test_the_wsl_launcher_is_never_the_shell(monkeypatch):
    """The trap the spec calls out by name: `shutil.which("bash")` on Windows answers
    `C:\\WINDOWS\\system32\\bash.exe`, the WSL launcher. It would run the command inside a Linux VM where
    `C:\\...` paths, the venv and the run's file grants all mean something different -- a shell that looks
    like POSIX and is not the same machine.

    So the guard is not "find it, then reject it" -- it is that PATH is **never asked for `bash`**. That is
    what `asked` asserts: a later edit that reaches for `which("bash")` fails here rather than quietly
    putting a Linux VM between the run and its grants. On POSIX there is no launcher to dodge: the
    resolver takes /bin/sh and PATH is never consulted for anything, so `asked` stays empty no matter
    what a fake PATH would have answered.
    """
    if sys.platform != "win32":
        asked: list[str] = []
        monkeypatch.delenv("SAH_SHELL", raising=False)
        _fake_fs(monkeypatch, which={"bash": "/bin/bash"}, files=[], asked=asked)
        s = shell.resolve_shell()
        assert s.path == "/bin/sh"
        assert asked == [], "a PATH answer -- any answer -- is not a shell for the run"
        return
    asked: list[str] = []
    monkeypatch.delenv("SAH_SHELL", raising=False)
    _fake_fs(monkeypatch, which={"git": None, "bash": r"C:\WINDOWS\system32\bash.exe"},
             files=[], asked=asked)
    s = shell.resolve_shell()
    assert s.family == "cmd", "no Git Bash found, so the honest fallback"
    assert "bash" not in asked, "PATH must never be asked for bash: on Windows that answer is WSL"
    # System32 is not forbidden in itself -- cmd.exe lives there legitimately and is the right fallback.
    # What must never be used is a *bash* from there, which is why the resolver never asks for that name.
    assert os.path.normcase(s.path).endswith("cmd.exe")


def test_an_operator_shell_wins_and_is_still_not_wsl(monkeypatch):
    """`SAH_SHELL` is an explicit human choice, so it outranks detection -- but it does not outrank the
    one refusal, because the operator pointing at System32's bash almost certainly did not mean the WSL
    VM. A named shell the operator chose is reported by name, so the model is told what it has. The WSL
    refusal has no POSIX counterpart -- there is no launcher to refuse -- so on POSIX a named shell wins
    and is named in the hint the model reads."""
    if sys.platform != "win32":
        monkeypatch.setenv("SAH_SHELL", "/usr/local/bin/dash")
        s = shell.resolve_shell()
        assert s.path == "/usr/local/bin/dash" and s.family == "posix"
        assert "dash" in s.hint, "the model is told which shell it has"
        return
    _fake_fs(monkeypatch, which={"git": r"X:\Git\cmd\git.exe"}, files=[r"D:\tools\sh.exe"])
    monkeypatch.setenv("SAH_SHELL", r"D:\tools\sh.exe")
    assert os.path.normcase(shell.resolve_shell().path) == os.path.normcase(r"D:\tools\sh.exe")
    assert shell.resolve_shell().family == "posix"

    monkeypatch.setenv("SAH_SHELL", r"C:\WINDOWS\system32\bash.exe")
    assert shell.resolve_shell().family == "cmd", "an explicit path is not a licence to reach the VM"


def test_an_operator_shell_wins_on_posix_too(monkeypatch):
    """The documented order starts with `SAH_SHELL` on every platform, not only on Windows. Answering
    /bin/sh before the environment is read drops a forced shell silently: the operator believes they
    chose, and a check that forces its shell ends up testing /bin/sh."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SAH_SHELL", "/usr/bin/dash")
    s = shell.resolve_shell()
    assert s.path == "/usr/bin/dash", f"the operator asked for dash, got {s.path}"
    assert s.family == "posix"

    monkeypatch.delenv("SAH_SHELL")
    assert shell.resolve_shell().path == "/bin/sh", "with no operator choice, POSIX gets /bin/sh"


def test_an_operator_may_name_cmd_exe_although_it_lives_in_system32(monkeypatch):
    """The refusal is for the WSL impostor, which is a *bash* sitting in System32 -- not for the directory,
    where cmd.exe lives legitimately. Refusing anything under System32 was this resolver's own first bug:
    it survived the battery and was caught only by running the live check with `SAH_SHELL` pointed at
    cmd.exe, which then quietly tested Git Bash and passed. An override that is silently ignored is worse
    than no override, because the operator believes they chose.
    """
    if sys.platform != "win32":
        # There is no system directory whose shell is an impostor on POSIX: whatever the operator names
        # is what the run gets, wherever it sits.
        monkeypatch.setenv("SAH_SHELL", "/opt/tools/zsh")
        s = shell.resolve_shell()
        assert s.path == "/opt/tools/zsh" and s.family == "posix"
        return
    system = os.environ.get("SystemRoot") or r"C:\Windows"
    chosen = os.path.join(system, "System32", "cmd.exe")
    # No filesystem, and a ComSpec that is NOT the path being asked for. Both matter: with the fallback
    # free to be cmd.exe from its usual place, "did the override win" and "did the fallback happen to be
    # the same program" produce the same answer, and the first version of this test passed while the
    # override was refused outright. An assertion that cannot tell those apart is not an assertion.
    _fake_fs(monkeypatch, which={"git": None}, files=[])
    monkeypatch.setenv("ComSpec", os.path.join("X:", "elsewhere", "cmd.exe"))

    monkeypatch.setenv("SAH_SHELL", chosen)
    s = shell.resolve_shell()
    assert s.path == chosen, f"the operator asked for {chosen}, got {s.path}"
    assert s.family == "cmd", "cmd.exe in System32 is cmd.exe, not an impostor"

    monkeypatch.setenv("SAH_SHELL", os.path.join(system, "System32", "bash.exe"))
    s = shell.resolve_shell()
    assert s.path == os.path.join("X:", "elsewhere", "cmd.exe"), \
        "a System32 bash is the WSL VM, so the fallback is used instead"


def test_cmd_is_the_honest_fallback(monkeypatch):
    """No Git, no operator shell: cmd.exe, named as such in everything the model reads. On POSIX the
    honest fallback is /bin/sh, named the same way."""
    if sys.platform != "win32":
        monkeypatch.delenv("SAH_SHELL", raising=False)
        s = shell.resolve_shell()
        assert s.path == "/bin/sh" and s.name == "/bin/sh"
        assert "/bin/sh" in s.hint, "the model is told which shell it has"
        return
    monkeypatch.delenv("SAH_SHELL", raising=False)
    monkeypatch.setenv("ComSpec", r"X:\Windows\cmd.exe")
    _fake_fs(monkeypatch, which={"git": None}, files=[])
    s = shell.resolve_shell()
    assert s.family == "cmd" and os.path.normcase(s.path) == os.path.normcase(r"X:\Windows\cmd.exe")
    assert s.name == "cmd.exe"


def test_each_family_gets_the_argv_that_family_parses(monkeypatch):
    """One argv builder, two shapes, and the reason is the parser on the other end: cmd gets one string
    because `list2cmdline` escapes quotes cmd cannot read; Git Bash gets the list because its MSYS runtime
    decodes exactly that escaping correctly, and a hand-built string would then be mis-split."""
    bashish = shell.Shell(name="Git Bash", family="posix", path=r"X:\Git\bin\bash.exe")
    cmdish = shell.Shell(name="cmd.exe", family="cmd", path=r"C:\Windows\System32\cmd.exe")
    assert bashish.argv('echo "a b"') == [r"X:\Git\bin\bash.exe", "-c", 'echo "a b"']
    assert cmdish.argv('echo "a b"').endswith('/d /s /c "echo "a b""')


def test_the_model_is_told_which_shell_it_has():
    """The tool is named `bash`. A model offered that name writes `ls -la`, `grep`, `$(...)` -- and on
    cmd.exe burns turns discovering the dialect. So the resolved shell is part of the tool's contract, not
    a fact buried in a banner the model never sees.
    """
    from sah import contract
    text = contract.spec_for(shell.bash).description
    assert shell.SHELL.name in text, "the description names the shell actually in use"
    assert shell.SHELL.hint in text, "...and what that means for syntax"


def test_posix_keeps_the_argv_list():
    """`/bin/sh -c` takes the command as one argument, so nothing needs to be assembled and nothing
    may be re-quoted: the list form is the correct one and the quoting bug is not a POSIX bug."""
    assert shell.Shell("/bin/sh", "posix", "/bin/sh").argv('echo "a b"') == \
        ["/bin/sh", "-c", 'echo "a b"']


def test_the_command_line_follows_the_shell_the_model_was_told_about():
    """The argv `_stream` builds and the shell named in the tool description come from the same
    resolution, so the stated dialect cannot drift from the dialect actually used."""
    assert shell._argv('echo "a b"') == shell.SHELL.argv('echo "a b"')


def test_an_undecodable_byte_is_shown_rather_than_smoothed(monkeypatch):
    """The streams are decoded as UTF-8, and what a command writes is not ours to choose. Measured on this
    box (console code page 437, ANSI code page 1252): `echo` of an accented character leaves cmd as one
    cp437 byte, the same character from `python` leaves as one cp1252 byte, and a tool that opts into UTF-8
    (`python -X utf8`) leaves as two. No single decoding is right for those three, and a prefix that sets
    the console code page is not free -- one child running `chcp 65001` changes it for every later child of
    that console, measured, so the obvious fix would mutate the operator's terminal from inside a run.

    So what is decided is what happens to a byte that does not decode. `errors="replace"` makes it U+FFFD:
    a character that reads like text, carries no information, and cannot be told apart from a file that
    really is damaged. `errors="backslashreplace"` shows the byte, which the model can name, work around
    (`-X utf8`, or read the file it wrote), or ignore knowingly.
    """
    # Asserted on the output, not on the Popen kwargs. It used to read `s.kwargs["errors"]`, which passed
    # for as long as the decoding happened to be a text-mode pipe argument and would have failed the moment
    # the decoding moved — while telling a reader nothing about whether a bad byte is actually shown. The
    # promise is about what a model sees, so that is what is checked.
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=[b"caf\x82\n"]), code=0))
    _, report = pieces(shell.bash("python"))
    assert "caf\\x82" in report, f"an undecodable byte was not shown as itself: {report!r}"
    assert "\ufffd" not in report, f"an undecodable byte was smoothed into U+FFFD: {report!r}"


def test_the_command_gets_no_stdin(monkeypatch):
    """A command that reads stdin is told there is nothing to read, immediately.

    Inherited stdin means the command is attached to the harness's own console: `python` with no
    arguments, `pause` in a script, a credential prompt from `git`, or a `set /p` would either eat the
    operator's keystrokes meant for the REPL or sit there until the timeout. Both are worse than an
    immediate EOF, and DEVNULL is the one input that guarantees the command cannot block waiting."""
    s = spy(monkeypatch, FakeProc(code=0))
    drain(shell.bash("python"))
    assert s.kwargs.get("stdin") is subprocess.DEVNULL


def test_stderr_is_captured_to_its_own_pipe_not_merged(monkeypatch):
    s = spy(monkeypatch, FakeProc(stdout="x", stderr="y", code=0))
    drain(shell.bash("run"))
    assert s.kwargs["stdout"] is subprocess.PIPE
    assert s.kwargs["stderr"] is subprocess.PIPE               # not subprocess.STDOUT


def test_stderr_keeps_its_own_section_in_the_report(monkeypatch):
    spy(monkeypatch, FakeProc(stdout="out", stderr="traceback", code=1))
    out, _ = drain(shell.bash("run"))
    assert "stdout:\nout" in out and "stderr:\ntraceback" in out


def test_a_nonzero_exit_is_flagged_not_just_printed(monkeypatch):
    spy(monkeypatch, FakeProc(stdout="", code=2))
    out, _ = drain(shell.bash("false"))
    assert "exit: 2" in out and "non-zero" in out


def test_short_output_is_not_clipped(monkeypatch):
    spy(monkeypatch, FakeProc(stdout="a short, complete answer", code=0))
    out, _ = drain(shell.bash("small"))
    assert "a short, complete answer" in out
    assert "dropped" not in out, "an output that fit is not described as if it had been cut"


def test_no_output_says_so(monkeypatch):
    spy(monkeypatch, FakeProc(stdout="", stderr="", code=0))
    out, _ = drain(shell.bash("cd ."))
    assert "no output" in out


def test_a_timeout_kills_the_tree_and_says_the_run_was_killed(monkeypatch):
    monkeypatch.setattr(shell, "_MONO", fake_clock())
    s = spy(monkeypatch, FakeProc(stdout="partial\n", hang=True))
    out, _ = drain(shell.bash("sleep 600", timeout=5))
    assert "exit: killed" in out
    assert s.killed == [True]                                  # kill_tree, detached (own group)
    assert "everything it spawned went with it" in out
    assert "partial" in out                                    # partial output before the kill kept


def test_an_over_budget_output_keeps_its_ends_and_says_what_was_dropped(monkeypatch):
    """The tool keeps the opening and the ending of a big output and names the middle it dropped.

    This replaces the earlier invariant that the shell returned its output whole for the loop to bound:
    holding every byte of a chatty command is how a build log becomes an out-of-memory stop for the
    harness, and the operator ruled that the middle may go as long as the model is told — and told the
    way back (re-run redirecting to a file, then read that file)."""
    body = "".join(f"line {i} of the build log, padded to make the point\n" for i in range(6000))
    spy(monkeypatch, FakeProc(stdout=body, code=1))
    out, _ = drain(shell.bash("build"))
    assert "line 0 of the build log" in out, "the opening is kept"
    assert "line 5999 of the build log" in out, "the ending is kept: the error is at the end"
    assert out.count("line 3000 of the build log") == 0, "the middle is not held"
    assert "were dropped" in out and "> file" in out, "the notice says what went and how to get it"
    assert out.rstrip().splitlines()[-1].startswith("exit: 1"), "the exit code stays the last line"
    assert len(out.encode()) <= ingest.MAX_BYTES, "the report arrives under the loop's own bound"
    assert shell.bash.ingest_keep == "tail", "command output is end-loaded (the error is at the end)"


def test_a_cut_stream_stops_holding_the_middle():
    """The cap exists for memory, so it is tested as memory, not as wording.

    Once the budget is passed, what was held has to be released and what stays has to stay inside the
    two ends. A version that kept the early text in a list 'just in case' would still report bounded
    text and still exhaust the harness's RAM, so the assertion is about the object's own footprint."""
    e = shell._Ends(head_bytes=1000, tail_bytes=2000)
    for i in range(100_000):
        e.add(f"line {i} padded out to be long enough to matter\n")
    assert e.kept == [], "the text held before the cut is released, not parked"
    assert len(e.head) <= 1000, "the head is bounded"
    assert sum(len(piece) for piece in e.tail) <= 2000, "the tail window does not grow with the stream"
    assert e.bytes > 4_000_000 and "line 99999" in e.text("stdout"), "the ending is still the ending"


def test_a_flood_is_not_accumulated_by_the_tool(monkeypatch):
    """The cap exists for the harness's memory, so the claim is about size, not wording.

    Streaming is untouched — every line still reaches whoever is watching — while what the tool keeps
    stays inside the budget. Lines are counted rather than joined so the test itself stays cheap."""
    spy(monkeypatch, FakeProc(code=0, firehose=200_000))        # 1.2 MB of `chunk\n`
    stream = shell.bash("flood")
    streamed = 0
    while True:
        try:
            if next(stream):
                streamed += 1
        except StopIteration as stop:
            report = stop.value
            break
    assert streamed > 100_000, "capping retention must not cap what streams live"
    assert len(report.encode()) < ingest.MAX_BYTES, "what the tool retains is bounded by the cap"


def test_the_exit_code_is_the_last_line_so_tail_clipping_keeps_it(monkeypatch):
    spy(monkeypatch, FakeProc(stdout="a" * 50000, stderr="boom", code=1))
    out, _ = drain(shell.bash("build"))
    assert out.rstrip().splitlines()[-1].startswith("exit: 1")


def test_the_child_leads_its_own_group_so_the_tree_is_reachable(monkeypatch):
    s = spy(monkeypatch, FakeProc(code=0))
    drain(shell.bash("true"))
    assert "creationflags" in s.kwargs or "start_new_session" in s.kwargs


def test_a_shell_that_will_not_start_is_explained_not_raised(monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("no such shell")
    monkeypatch.setattr(shell.subprocess, "Popen", boom)
    out, streamed = drain(shell.bash("anything"))              # a stream of zero chunks
    assert streamed == ""
    assert "could not start the shell" in out


def test_the_killed_message_names_the_timeout_actually_applied(monkeypatch):
    monkeypatch.setattr(shell, "_MONO", fake_clock())
    spy(monkeypatch, FakeProc(hang=True))
    out, _ = drain(shell.bash("sleep 999", timeout=0))
    assert "after 1s" in out and "after 0s" not in out


# --- abandoning the stream -----------------------------------------------------

def test_abandoning_the_stream_kills_the_process_it_started(monkeypatch):
    """The one streaming tool can be left mid-stream: the loop closes it on an interrupt, a consumer can
    raise into it, and garbage collection closes a stream nobody drained. The child leads its own process
    group precisely so a stray Ctrl+C cannot reach it, which also means nothing else reaches it either —
    so an abandoned command keeps running with no watcher at all, holding files and the GPU. Killing it is
    the shell's job, not the consumer's."""
    proc = FakeProc(stdout="one\ntwo\nthree\n", hang=True)
    s = spy(monkeypatch, proc)
    stream = shell.bash("spam")
    assert next(stream) == "one\n"                 # one line seen, then the caller walks away
    stream.close()                                 # GeneratorExit, the same path close() takes
    assert proc.killed, "an abandoned command must not outlive its watcher"
    assert s.killed == [True]


def test_a_consumer_error_does_not_orphan_the_process(monkeypatch):
    """A renderer bug that escapes into the drain is the caller's bug, but the process is the shell's
    responsibility: the exception must still propagate and the child must still die."""
    proc = FakeProc(stdout="a\nb\n", hang=True)
    spy(monkeypatch, proc)
    stream = shell.bash("spam")
    next(stream)
    try:
        stream.throw(RuntimeError("the consumer blew up"))
        raise AssertionError("the caller's exception must not be swallowed")
    except RuntimeError:
        pass
    assert proc.killed


def test_a_command_that_ended_on_its_own_is_not_killed(monkeypatch):
    """The other edge: guarding the cleanup on a process that already exited, so a completed command is
    never reported as killed and kill_tree is never called for a stopped child."""
    proc = FakeProc(stdout="done\n", code=0)
    s = spy(monkeypatch, proc)
    out, _ = drain(shell.bash("echo done"))
    assert out.rstrip().endswith("exit: 0")
    assert s.killed == [], "nothing to kill after a clean exit"


def test_a_command_that_leaves_something_running_keeps_its_own_exit_code(monkeypatch):
    """`python server.py & echo ready` -- an ordinary way to start something in the background -- finished
    instantly and cleanly, and the tool reported it as killed after burning the whole timeout. The
    background holder inherited the stdout pipe, so the readers never reached EOF, and "the command is
    over" had been conflated with "every pipe is closed".

    They are different facts. The command's exit code is known the moment it exits; the pipe may be held
    by something the command deliberately left behind. Stopping that is what the boundary is for, and the
    exit code stands.
    """
    proc = FakeProc(stdout="ready\n", code=0, hold=True)
    spy(monkeypatch, proc)
    # A short timeout on purpose. If the boundary stops working, this test must cost a second, not its
    # whole budget: one of these hold-based tests was written with `timeout=1000` to keep the deadline out
    # of the way, and a mutation run then appeared to hang for fifteen minutes because a broken fix does
    # spin all the way to the deadline.
    recorded, streamed = drain(shell.bash("python server.py & echo ready", timeout=3))

    assert "killed" not in recorded, f"a clean exit was reported as killed: {recorded!r}"
    assert "exit: 0" in recorded, recorded
    assert "ready" in streamed
    assert FakeBoundary.made and FakeBoundary.made[0].stops >= 1, \
        "the thing left running was never stopped"
    assert "background" in recorded, "something died here and the model was not told"


def test_the_real_exit_code_survives_even_when_the_holder_cannot_be_stopped(monkeypatch):
    """A boundary that cannot be made (`attached` False, or a holder born in the assignment window) must
    degrade to the old waiting behaviour, not to a lie: the deadline still fires and the report says
    killed. What must not happen is reporting success for a command whose pipes nobody closed.
    """
    proc = FakeProc(stdout="ready\n", code=0, hold=True)
    spy(monkeypatch, proc)
    # the holder survives the stop, as it would without a job
    monkeypatch.setattr(FakeBoundary, "RELEASE", False)
    monkeypatch.setattr(shell, "JOIN_SECONDS", 0.05)
    monkeypatch.setattr(shell, "_MONO", fake_clock())      # the deadline is past on the first pass
    recorded, _ = drain(shell.bash("python server.py & echo ready", timeout=5))
    assert "killed" in recorded, f"an unclosable pipe was reported as a clean exit: {recorded!r}"


def test_the_readers_share_one_join_bound(monkeypatch):
    """Two readers, two `join(KILL_WAIT)` calls: a command with a 6 s timeout took 16 s to return, which
    makes the timeout a suggestion. One shared bound for both, so the tool cannot overrun its own promise
    by a multiple of the wait.

    The shape that isolates the join: both holders are unstoppables (`RELEASE = False`), so when the
    deadline fires both readers are still blocked and the only thing between the tool and returning is
    those two joins. Measured on the real clock, for two reasons that each broke an earlier version of
    this test: a frozen clock makes `join_by - now` constant, so every join gets its full budget and a
    shared bound costs exactly what a per-thread one costs (that version passed against both, measuring
    nothing); and the tool floors a 0 s timeout at 1 s, so the deadline contributes a real second either
    way -- shared ~1.6 s, per-thread ~2.2 s.
    """
    proc = FakeProc(stdout="ready\n", stderr="warn\n", code=0, hold=True)
    spy(monkeypatch, proc)
    monkeypatch.setattr(FakeBoundary, "RELEASE", False)     # nothing releases the pipes: the joins decide
    monkeypatch.setattr(shell, "JOIN_SECONDS", 0.6)
    start = time.monotonic()
    drain(shell.bash("x", timeout=1))                       # the floor: 1 s is the shortest budget there is
    took = time.monotonic() - start
    assert took < 1.9, f"the readers took {took:.1f}s to release the tool; the bound is per-thread again"


def test_abandoning_a_stream_stops_the_boundary_too(monkeypatch):
    """The direct-child kill in the `finally` is guarded by `poll()` -- correct for a child, wrong for a
    descendant, whose whole case is that the child already exited. So the boundary is stopped
    unconditionally: an abandoned command's holder must not survive the abandonment.
    """
    proc = FakeProc(stdout="one\ntwo\n", code=0, hold=True)   # the shell exits by itself
    spy(monkeypatch, proc)
    stream = shell.bash("python server.py & echo ready", timeout=30)
    next(stream)                                        # start it, then walk away
    stream.close()
    assert FakeBoundary.made and FakeBoundary.made[0].stops >= 1
    assert FakeBoundary.made[0].closes == 1, "the handle was left open, so the run cannot release its tree"


def test_an_ordinary_command_is_not_reported_as_having_left_something_behind(monkeypatch):
    """The false-positive half. A command that simply finished must not be described as killed, and must
    not gain the note about stopping a background process -- that note is the model's only clue that its
    server is gone, and a note on every command teaches it to ignore the note.

    What is *not* asserted is that the boundary was left alone. It is stopped on every path, including a
    clean one, because the `finally` cannot know whether this is the abandoned case; the promise the model
    sees is about what it is told and what code is reported, not about how many times a kernel object was
    signalled. This test failed on exactly that confusion when written, which is why it says so.
    """
    proc = FakeProc(stdout="done\n", code=0)
    spy(monkeypatch, proc)
    recorded, _ = drain(shell.bash("echo done", timeout=30))
    assert "background" not in recorded and "killed" not in recorded and "exit: 0" in recorded


def test_a_host_with_no_boundary_does_not_claim_it_killed_the_tree(monkeypatch):
    """The report's promise must follow the mechanism that backs it.

    With a job object (or, off Windows, a group this child leads) "everything it spawned went with it" is
    true. On a Windows host where the job could not be made -- already inside a non-nestable job, say --
    only the direct child was stopped, and the old sentence would then be a lie the model plans around.
    The degraded case has to say it is degraded.
    """
    proc = FakeProc(stdout="ready\n", hang=True)
    spy(monkeypatch, proc)
    monkeypatch.setattr(shell, "_MONO", fake_clock())     # the deadline is past at once
    monkeypatch.setattr(FakeBoundary, "TEETH", False)
    recorded, _ = drain(shell.bash("make", timeout=5))
    assert "may still be running" in recorded, recorded
    assert "everything it spawned went with it" not in recorded, recorded


def test_a_command_that_never_stops_printing_still_times_out(monkeypatch):
    """The deadline is checked on every pass, not only on a pass with nothing to show.

    A clock that jumps past the deadline immediately, a process that never exits, and output that never
    runs out: the run must stop at the deadline while lines are still waiting, not after the flood has
    been drained into memory. Before this, the check lived in the empty-queue branch and a gapless
    command was never timed out at all."""
    monkeypatch.setattr(shell, "_MONO", fake_clock())
    s = spy(monkeypatch, FakeProc(hang=True, firehose=5000))
    out, streamed = drain(shell.bash("yes", timeout=5))
    assert "exit: killed" in out
    assert s.killed == [True]
    got = [line for line in streamed.splitlines() if line]
    assert len(got) < 100, f"the deadline was reached only after {len(got)} lines were buffered"


# --- what a reader hands back (a bar drawn with CR, and the bytes under it) ------

def pieces(stream):
    """The yielded pieces, kept apart, plus the report. `drain` concatenates, which is right for a report
    and wrong for a test whose subject is how many pieces the bar arrived as."""
    seen = []
    while True:
        try:
            seen.append(next(stream))
        except StopIteration as stop:
            return seen, stop.value


def test_a_progress_bar_drawn_with_cr_arrives_as_it_is_written(monkeypatch):
    """A guard, not a fix, and it passes on the old code because the old code was never broken here.

    The review claimed `iter(pipe.readline, "")` cannot show a bar that redraws with CR and never sends a
    newline. Measured against a real command, that is false: the pipes are text mode with `newline=None`,
    which is universal newlines, so CR ends a line and the pieces arrived at 0.34s, 0.59s, 0.84s and 1.09s
    -- live. Kept because a future change to the reader could regress it, and because the reading that
    produced the finding was plausible enough to write down."""
    bar = FakePipe("", chunks=[" 10%\r", " 50%\r", "100%\n"])
    spy(monkeypatch, FakeProc(stdout=bar, code=0))
    seen, report = pieces(shell.bash("curl -O big.iso"))
    assert len(seen) == 3, f"a CR bar arrived as {len(seen)} piece(s): {seen}"
    assert " 50%" in seen[1], seen


def test_the_report_holds_the_line_endings_the_command_actually_wrote(monkeypatch):
    """Text mode translates, and the tool has been reporting the translation as if it were the output.

    A child that writes `a\rb\n` wrote a carriage return. Universal-newline decoding hands the reader
    `a\nb\n`, and the report then states that the command wrote two lines -- a different command's output.
    The module already refuses to smooth an undecodable byte into U+FFFD for exactly this reason; a CR is
    the same kind of fact and is currently spent anyway. It is also load-bearing: `\r\n` reaching the model
    as `\n` means line endings cannot be reasoned about from a report, and a progress bar reads as a
    transcript of four completed states rather than one line redrawn."""
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=["a\r", "b\r", "c\n"]), code=0))
    _, report = pieces(shell.bash("bar"))
    assert chr(13) in report, f"the report lost the CR the command wrote: {report!r}"
    assert "a" + chr(13) + "b" + chr(13) + "c" in report, report


def test_a_crlf_that_arrives_split_across_two_reads_stays_one_ending(monkeypatch):
    """No translating reader gets this right, and no chunked reader should get it wrong either.

    The CRLF is one fact split by a pipe boundary. Nothing may turn it into two line endings, and nothing
    may drop the LF that turned up late."""
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=["one\r", "\ntwo\n"]), code=0))
    seen, _ = pieces(shell.bash("type file.txt"))
    joined = "".join(seen)
    assert joined == "one\r\ntwo\n", f"a split CRLF was rewritten: {joined!r}"


def test_the_pieces_concatenate_to_what_the_command_wrote(monkeypatch):
    """Changing the unit of reading must not change what is reported.

    The report is assembled by concatenation, so a boundary anywhere -- mid-word, mid-line, mid-character --
    has to leave the assembled text identical to the stream. The property a chunked reader can break
    silently, and one no liveness test would notice."""
    chunks = ["down", "loadin", "g 50\r%", " of 100", "\ndone\n"]
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=chunks), code=0))
    seen, report = pieces(shell.bash("fetch"))
    assert "".join(seen) == "".join(chunks), "the live pieces are not the stream"
    assert "downloading 50\r% of 100\ndone" in report, f"the report is not the stream: {report!r}"


def test_a_character_split_across_two_reads_arrives_whole(monkeypatch):
    """A chunk boundary is not a character boundary, and the decoder has to remember that between reads.

    Bytes of one character arriving in two reads must decode to that character -- not to two replacement
    marks, and not to the backslash-escaped gap this module keeps for bytes that genuinely cannot decode.
    A split is a fact about the pipe, not about the output."""
    raw = "\u00e9\u00e9 caf\u00e9\n".encode("utf-8")           # three two-byte characters
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=[raw[:3], raw[3:]]), code=0))
    seen, _ = pieces(shell.bash("echo"))
    joined = "".join(seen)
    assert joined == "\u00e9\u00e9 caf\u00e9\n", f"a split character was mangled: {joined!r}"
    assert "\\x" not in joined, f"an ordinary character was escaped away: {joined!r}"


def test_bytes_that_cannot_decode_at_the_end_are_still_shown(monkeypatch):
    """A stream that stops mid-character must not lose the byte it did deliver.

    There is no completion to wait for, so the decoder is closed and what cannot be part of a character is
    shown as the byte it is. Dropping the incomplete tail would report a command's output as shorter than
    it was, in silence."""
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=[b"tail", b"\xc3"]), code=0))
    seen, _ = pieces(shell.bash("weird"))
    joined = "".join(seen)
    assert joined.startswith("tail"), repr(joined)
    assert "\\xc3" in joined, f"a truncated byte was dropped instead of shown: {joined!r}"


def test_a_reports_trailing_line_ending_is_not_content(monkeypatch):
    """The report ends with the exit code, so whatever the command left last is not where the report ends.

    Since the endings became the command's own, this had to be said about CR as well as LF: a bar whose
    final write is `100%\r` should not push the exit code onto the same line or leave a stray ending in
    front of it. What is inside is untouched -- see the CR test above -- this is only about the tail.
    """
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=["100%\r"]), code=0))
    _, report = pieces(shell.bash("curl"))
    assert report.index("exit:") > report.index("100%")
    assert chr(13) not in report[report.index("100%") + 4:report.index("exit:")], \
        f"a trailing CR was left in front of the verdict: {report!r}"


def test_each_stream_decodes_on_its_own(monkeypatch):
    """Two byte streams, two decoders. Out and err interleave at the pipe; they must not interleave in the
    middle of a character's state.

    A single decoder shared by both reader threads is the obvious way to write this and is wrong: the pending
    byte one stream left behind would be completed by the other, yielding a character neither command sent.

    Both streams carry a long run of split characters, not one. The first version sent a single character
    each and the shared-decoder mutant survived it — the two threads happened to take turns cleanly, which
    is the shape of a test that passes for the bug and for the fix. Forty half-characters per stream make a
    clean handoff between them essentially impossible."""
    raw = "\u00e9".encode("utf-8")                     # two bytes, deliberately fed one at a time
    halves = [b for _ in range(40) for b in (raw[:1], raw[1:])]
    turn = Lockstep()                                      # the interleave is arranged, not hoped for
    spy(monkeypatch, FakeProc(stdout=FakePipe("", chunks=halves, step=turn),
                              stderr=FakePipe("", chunks=halves, step=turn), code=0))
    seen, _ = pieces(shell.bash("both"))
    joined = "".join(seen)
    assert joined.count("\u00e9") == 80, f"the two streams completed each other's characters: {joined!r}"
    assert "\\x" not in joined, f"a decodable byte was escaped: {joined!r}"
