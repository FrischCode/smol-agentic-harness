"""sah.process — one way to start a child detached, one way to stop a process tree.

Imports nothing from this package and nothing beyond the standard library. Everything that
spawns a child needs this, so a dependency here would be a cycle waiting to happen, and a
configuration read here would be ambient state in the one place that has to work while the
harness is falling apart.

Why one stop function. Stopping a process tree is platform-dependent and easy to get wrong —
off Windows, `taskkill` raises and a fallback that signals the direct child alone leaves a
descendant alive after a kill that reported success. A capability this failure-prone belongs in
one function, not in a copy per call site, because copies of it drift.

The platform contract
---------------------

**Windows.** `taskkill /F /T` walks the tree by parent, so it reaches descendants whichever
way the child was started. `detached` is irrelevant here.

**POSIX.** There is no tree walk. Reaching descendants means signalling the child's process
group, and that is safe only if the child *leads* the group — which is what
`popen_detach_kwargs()` arranges. A child started without it shares the harness's group, and
signalling that group would signal the harness itself. `detached` is the caller saying which
one it started, and a child that does not lead a group is stopped alone. Guessing is not an
option: the call that goes wrong takes down the caller's own process group.

`kill_tree` never raises. A kill that could not finish is reported through the caller's own
state — a task's status, a tool result — not a traceback from here.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys

# How long a stopped-in-its-tracks child gets before escalation. The wait is the only clock
# here: no sleeping, so a caller whose process is already gone returns immediately.
GRACE_SECONDS = 5.0
KILL_WAIT_SECONDS = 5.0
TASKKILL_TIMEOUT_SECONDS = 10.0

# CREATE_NEW_PROCESS_GROUP: the child is in a new process group, so a Ctrl+C to the parent
# does not signal it.
_NEW_PROCESS_GROUP = 0x00000200


def popen_detach_kwargs() -> dict:
    """The `Popen` kwargs that detach a child from the parent's lifecycle.

    The caller passes these to `subprocess.Popen` and tells `kill_tree` it did. Detaching is
    a choice with two consequences: the child outlives the parent's Ctrl+C, and on POSIX it
    becomes safe to signal as a group."""
    if sys.platform == "win32":
        return {"creationflags": _NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# A child started with no console at all. NOT used -- see `popen_unattended_kwargs`. Recorded here
# because the number is the finding: given this flag, a child started into a real console reported a
# console window of its own, owned by a different process. A console, twice, not the absence one.
_DETACHED_PROCESS = 0x00000008


def popen_unattended_kwargs() -> dict:
    """The `Popen` kwargs for a child run started to be left alone: its own group, and -- at the call
    site -- no stdin. Today that is exactly `popen_detach_kwargs()`.

    **The console axis is deliberately not taken.** A spawned run is sold as outliving its parent, and on
    Windows a process attached to a console is said to die when that console's window closes, so in
    principle the child should be windowless. Two measurements have been made and neither supports
    taking the flag.

    `DETACHED_PROCESS` (0x8) does not mean "no console" here. A child given this flag while started into a
    real console reported a **nonzero console window owned by a different process** — a console, observed
    twice, contradicting its own documentation. Whatever it detaches from, it does not produce what its
    name promises.

    And the hazard it would prevent did not appear in the case that could be produced. A parent launched
    into a real console (`CREATE_NEW_CONSOLE`) and a child started the way this function starts one shared
    the same window handle and the same window-owner pid — genuinely attached — and killing the parent
    alone left the child alive.

    What stays unmeasured is narrow: an operator closing the window. Simulating it (`FreeConsole`, then
    `AttachConsole` on the parent, verified by the joined window handle matching, then `CTRL_CLOSE_EVENT`)
    killed nothing, including the process that sent it; a close that does not close proves nothing. There
    was also no separate console host to terminate — the window's owner pid was the app's own, so "kill the
    host" collapses into "kill the parent", which is the case already measured as safe. `CREATE_NO_WINDOW`
    (0x08000000) was confirmed genuinely windowless (no window, no owner) and is the spelling to reach for
    if a child is ever watched dying at a real interactive terminal.

    So the name marks where the decision belongs, and the console flag's *absence* is the contract: adding
    it is a decision that has to be made on top of having watched a child die first.
    """
    return popen_detach_kwargs()


def kill_tree(proc: subprocess.Popen, *, detached: bool = True) -> None:
    """Stop `proc` and, where that is possible and safe, its descendants.

    `detached` must say how the child was started: True if it was given
    `popen_detach_kwargs()` (its own process group, so it leads it), False if it inherited the
    harness's. On Windows the flag changes nothing.

    This reaches descendants only while the direct child is alive, because both of its levers are
    anchored on it: `taskkill /T` walks the tree by parent, and a dead child is the walk's root. A
    descendant that outlives its parent -- a `cmd &`, a `start /b` -- is invisible from here, which is why
    a caller that needs the *whole* tree stopped uses a `Boundary` as well. The early return below is
    correct for this function's contract — it stops the direct child and what is reachable from it — and is
    not a guarantee about the whole tree.

    `kill_tree` never raises. A kill that could not finish is reported through the caller's own
    state -- a task's status, a tool result -- not a traceback from here.
    """
    if proc.poll() is not None:
        return  # already stopped; nothing to signal and nothing to reap

    if os.name == "nt":
        _run_quietly(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
    elif detached:
        if not _signal_group(proc, signal.SIGTERM):
            _terminate_quietly(proc)
    else:
        _terminate_quietly(proc)

    if _wait(proc, GRACE_SECONDS):
        # The leader stopped, which is not evidence about the group. The grace wait watches the leader's
        # handle because it is the only handle this function owns, so a member that ignored the signal is
        # still running at the moment this returns. A group we signalled once is therefore killed once: a
        # group that turned out to be empty answers `ProcessLookupError`, which `_signal_group` already
        # reports as a False rather than raising. The residue is a stopped-and-reaped leader whose number was
        # reused within the grace window — not closed here, and stated rather than assumed away.
        if os.name != "nt" and detached:
            _signal_group(proc, signal.SIGKILL)
        return

    # Still alive: escalate. On Windows taskkill already ran with /F, so the only lever left
    # is the direct handle; off Windows the group gets SIGKILL when it is ours to signal, and
    # the direct child gets it when it is not.
    if os.name == "nt" or not detached:
        _kill_quietly(proc)
    elif not _signal_group(proc, signal.SIGKILL):
        _kill_quietly(proc)
    _wait(proc, KILL_WAIT_SECONDS)


def _run_quietly(argv: list[str]) -> None:
    try:
        subprocess.run(argv, capture_output=True, timeout=TASKKILL_TIMEOUT_SECONDS)
    except Exception:
        pass


def _wait(proc: subprocess.Popen, seconds: float) -> bool:
    """True when the process is gone. Also reaps it, which is why the wait is not optional
    even after a signal that cannot fail."""
    try:
        proc.wait(timeout=seconds)
        return True
    except Exception:
        return False


def _signal_group(proc: subprocess.Popen, sig: int) -> bool:
    """Signal the child's process group. The caller owns the precondition that the child
    leads it; failure is reported, not raised, so the caller can fall back to the handle."""
    try:
        os.killpg(proc.pid, sig)
        return True
    except Exception:
        return False


def _terminate_quietly(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
    except Exception:
        pass


def _kill_quietly(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass


# --- the tree boundary -----------------------------------------------------------------------
#
# Why a second mechanism next to `kill_tree`. Both of that function's levers are anchored on the direct
# child, so a descendant that outlives it survives a kill that reported success. Measured on this host:
# `python holder.py & echo ready` under Git Bash -- the command finishes instantly and cleanly, but the
# background holder inherited the stdout pipe, so the readers never reached EOF, the tool spun to its
# deadline, reported `killed` over an exit code of 0, and then killed a tree whose shell was already
# gone. The holder lived.
#
# Windows has a real boundary: a job object with KILL_ON_JOB_CLOSE. Every process the child spawns is in
# the job, whatever it does to its own parent, and closing the last handle ends them. POSIX already has
# one -- the process group -- which is what `popen_detach_kwargs()` arranges; a Boundary there is the same
# promise over a different mechanism, so callers write one rule.

_CREATE_SUSPENDED = 0x00000004
_KILL_ON_JOB_CLOSE = 0x00002000
_EXTENDED_LIMIT_INFORMATION = 9


def _job_api():
    """The kernel32 functions a job object needs, or None where they do not exist.

    Returning None rather than raising is the point: an environment without these gets the POSIX-style
    no-op boundary instead of a shell tool that cannot start. The caller says which it got, so the
    absence is never silent.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        # A HANDLE is a pointer. Left at ctypes' default return type it comes back as a C int, which on
        # a 64-bit host truncates the handle to its low 32 bits -- and every call made with it then fails
        # with ERROR_INVALID_HANDLE while the code that made it looks like it succeeded.
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        return ctypes, wintypes, kernel32
    except Exception:
        return None


def kill_on_close_info(ctypes, wintypes):
    """The `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` that carries `KILL_ON_JOB_CLOSE`.

    Built by a function rather than inline, so the flag is a returned value a reader can check, not a
    bit behind a pointer: `SetInformationJobObject` is handed `ctypes.byref(info)`, and a stub kernel32
    receives a pointer it cannot read back, so an inline struct could be checked only for its size and
    class -- which would pass happily with the flag never set. The one guarantee this exists for (a tree
    that dies because *we* died) is a single bit, and a bit nobody can
    see is a bit that can silently stop being set.
    """
    class BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64)]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimit),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
    return info


class Boundary:
    """The thing a command's whole process tree lands inside, so stopping it stops the tree.

    Owned by whoever starts the child. `attach` before relying on it, `stop` to end everything,
    `close` when finished with it. It never raises: a boundary that could not be made reports `teeth`
    False, and the caller still has `kill_tree` for the direct child.

    **Not for a child meant to outlive us.** A spawned run is sold as surviving its parent, and the
    Windows half of this guarantee runs the other way: with `KILL_ON_JOB_CLOSE`, our handle closing --
    including by our own death -- ends everything inside. That is exactly right for a shell command the
    tool just waited on, and wrong for a hand-off. `sah/spawn.py` therefore keeps `kill_tree` and takes
    no boundary.
    """

    def __init__(self, *, group: bool = False) -> None:
        """`group` is the caller's assertion that it started the child as its own process-group leader
        (`popen_detach_kwargs()`), and so that signalling the group cannot signal the harness. Same
        discipline as `kill_tree(detached=...)`, and the same reason: guessing wrong takes down the
        caller's own group. It also keeps a placeholder pid from becoming a live signal.
        """
        self._api = _job_api()
        self._handle = None
        self._pid = 0
        self._group = bool(group)
        if self._api is not None:
            ctypes, _wintypes, kernel32 = self._api
            try:
                # None, None: an unnamed job. A name would let another process reach it, and nothing
                # needs to -- the handle is the only way in, and it is never inherited.
                self._handle = kernel32.CreateJobObjectW(None, None)
                if not self._handle:
                    self._handle = None
            except Exception:
                self._handle = None
    @property
    def teeth(self) -> bool:
        """Whether `stop()` can actually reach a descendant.

        This is what a report is allowed to claim. "Everything it spawned went with it" is true with a job
        object, or off Windows where the child leads a group that can be signalled; it is false on a Windows
        host where the job could not be made, where only the direct child can be stopped. A tool that
        promises a guarantee it does not have is worse than one that admits the limit, because the model
        plans around the promise.
        """
        if self._handle is not None:
            return True
        return self._pid != 0 and self._group and os.name != "nt"

    @property
    def popen_kwargs(self) -> dict:
        """Extra `Popen` kwargs the boundary needs. On POSIX: none -- the group comes from
        `popen_detach_kwargs()`.

        **No CREATE_SUSPENDED, and the reason is the honest limit of this design.** Suspending the child
        to assign it before it can run is the textbook recipe, but `subprocess.Popen` does not expose the
        primary *thread* handle that `ResumeThread` needs, and closing that window otherwise means calling
        CreateProcess by hand. So assignment happens just after the child starts, and in the microseconds
        before it the child cannot realistically have forked a grandchild -- but cannot be *proved* not
        to have, either. A descendant born in that window would sit outside the job and survive `stop()`.
        """
        return {}

    def attach(self, proc: subprocess.Popen) -> None:
        """Put the already-started child inside the boundary. See `popen_kwargs` for the window."""
        self._pid = proc.pid
        if self._handle is None:
            return
        _ctypes, _wintypes, kernel32 = self._api
        # `Popen` has never promised a public handle: on Windows it is `_handle`, and the attribute that
        # *is* documented (`stdout` and friends) is not a process handle. Read it defensively, because a
        # future Python that renames it must degrade to "no job", not to a crash in the shell tool.
        handle = getattr(proc, "handle", None)
        if handle is None:
            handle = getattr(proc, "_handle", None)
        if handle is None:
            self._detach_handle()
            return
        try:
            if not kernel32.AssignProcessToJobObject(self._handle, int(handle)):
                # Not in the job: `stop()` falls back to the group/child, and `teeth` says so. This is a
                # normal outcome where the parent is already inside a job that cannot nest, which is what
                # a harness-in-a-container looks like from here.
                self._detach_handle()
                return
            self._set_kill_on_close()
        except Exception:
            self._detach_handle()

    def _set_kill_on_close(self) -> None:
        """Ask that everything still in the job die when the last handle closes.

        This is the part that covers the case no signal does: if this process is killed outright, the
        handles close with it and the tree goes with them, rather than being orphaned to init."""
        _ctypes, _wintypes, kernel32 = self._api
        info = kill_on_close_info(_ctypes, _wintypes)
        kernel32.SetInformationJobObject(self._handle, _EXTENDED_LIMIT_INFORMATION,
                                         _ctypes.byref(info), _ctypes.sizeof(info))

    def stop(self) -> None:
        """End everything inside. Safe to call on a tree that is already gone, and to call twice."""
        if self._handle is not None:
            _ctypes, _wintypes, kernel32 = self._api
            try:
                kernel32.TerminateJobObject(self._handle, 1)
                return
            except Exception:
                pass
        if self._pid and self._group and os.name != "nt":
            # POSIX only, and the same rule `teeth` states: on Windows the job object is the whole mechanism
            # for reaching a descendant, and once it is gone there is nothing left to signal but the direct
            # child, which is `kill_tree`'s job. `os.killpg` does not exist there, so falling through would
            # be a silent no-op that reads, in a trace, like something was done.
            _signal_pid(self._pid)

    def close(self) -> None:
        """Release the boundary. With KILL_ON_JOB_CLOSE this also ends anything still inside it."""
        self._detach_handle()

    def _detach_handle(self) -> None:
        if self._handle is None:
            return
        try:
            _ctypes, _wintypes, kernel32 = self._api
            kernel32.CloseHandle(self._handle)
        except Exception:
            pass
        self._handle = None


def _signal_pid(pid: int) -> bool:
    """Signal the group `pid` leads. Kept separate from `kill_tree` because a boundary's whole point is
    to work when the direct child is already dead, where `kill_tree` returns early."""
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except Exception:
        return False
