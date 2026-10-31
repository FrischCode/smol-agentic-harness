"""The kill contract, branch by branch. Its subject is the process boundary, so the world is
stubbed — the process handle, `subprocess.run`, `os.name`, `os.killpg`, the signal numbers —
and what is asserted is *what gets signalled*. Stubbing `signal` too is what lets the POSIX
branches be exercised on a Windows box, where `signal.SIGKILL` does not exist.

No child is spawned, so nothing here waits on a real process.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from sah import process as proc_mod

SIGTERM, SIGKILL = 15, 9


class FakeProcess:
    """A handle that records what it was signalled with. `ignores` names the signals it
    survives, which is how the grace window and the escalation are tested without a clock."""

    pid = 4242

    def __init__(self, *, alive: bool = True, ignores: tuple[str, ...] = ()):
        self.alive = alive
        self.ignores = ignores
        self.signalled: list[str] = []

    def group_signal(self, sig: int) -> None:
        self._signal(f"group:{sig}")

    def terminate(self) -> None:
        self._signal("terminate")

    def kill(self) -> None:
        self._signal("kill")

    def _signal(self, kind: str) -> None:
        self.signalled.append(kind)
        if kind not in self.ignores:
            self.alive = False

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired("child", timeout)
        return 0


class FakeOs:
    def __init__(self, name: str, *, killpg_fails: bool = False):
        self.name = name
        self.killpg_fails = killpg_fails
        self.groups: list[tuple[int, int]] = []
        self._proc: FakeProcess | None = None

    def killpg(self, pgid: int, sig: int) -> None:
        if self.killpg_fails:
            raise ProcessLookupError(3, "no such group")
        self.groups.append((pgid, sig))
        assert self._proc is not None
        self._proc.group_signal(sig)

    def attach(self, process: "FakeProcess") -> "FakeOs":
        self._proc = process
        return self


class FakeSubprocess:
    """`run` is taskkill. When it succeeds the whole tree is gone, which is the difference
    between the two Windows outcomes the module can produce."""

    Popen = subprocess.Popen

    def __init__(self, *, succeeds: bool = True):
        self.argvs: list[list[str]] = []
        self.succeeds = succeeds
        self.proc: FakeProcess | None = None

    def run(self, argv, **kwargs):
        self.argvs.append(list(argv))
        if self.succeeds and self.proc is not None:
            self.proc.alive = False


@pytest.fixture
def world(monkeypatch):
    """The stubbed world: platform, group signalling, subprocess.run, signal numbers."""
    os_stub = FakeOs("nt")
    sub_stub = FakeSubprocess()
    monkeypatch.setattr(proc_mod, "os", os_stub)
    monkeypatch.setattr(proc_mod, "subprocess", sub_stub)
    monkeypatch.setattr(proc_mod, "signal",
                        type("S", (), {"SIGTERM": SIGTERM, "SIGKILL": SIGKILL}))
    return type("W", (), {"os": os_stub, "subprocess": sub_stub})()


def run_kill(world, process, detached=True):
    """Both stubs act on the same handle, so a signal has a visible effect on the process the
    module then waits for."""
    world.os.attach(process)
    world.subprocess.proc = process
    proc_mod.kill_tree(process, detached=detached)


def test_a_process_that_already_stopped_is_left_alone(world):
    run_kill(world, FakeProcess(alive=False))
    assert world.os.groups == []
    assert world.subprocess.argvs == []


def test_windows_kills_the_tree_by_parent(world):
    process = FakeProcess()
    run_kill(world, process)
    assert world.subprocess.argvs == [["taskkill", "/F", "/T", "/PID", str(process.pid)]]
    assert process.signalled == []


def test_windows_falls_back_to_the_direct_handle_when_taskkill_leaves_it_alive(world):
    """taskkill can miss: a pid that vanished between the poll and the run, a right that was
    denied. The direct handle is the last lever, and it is used."""
    world.subprocess.succeeds = False
    process = FakeProcess()
    run_kill(world, process)
    assert process.signalled == ["kill"]


def test_posix_signals_the_group_when_the_child_leads_it(world):
    """A leader that dies on SIGTERM is signalled, and the group is then killed. The second signal is not
    redundancy: the group's emptiness is not a fact this function can observe, only the leader's handle."""
    process = FakeProcess()
    world.os.name = "posix"
    run_kill(world, process, detached=True)
    assert world.os.groups == [(process.pid, SIGTERM), (process.pid, SIGKILL)]
    assert world.subprocess.argvs == []


def test_posix_escalates_the_group_when_the_child_ignores_the_first_signal(world):
    process = FakeProcess(ignores=(f"group:{SIGTERM}",))
    world.os.name = "posix"
    run_kill(world, process, detached=True)
    assert world.os.groups == [(process.pid, SIGTERM), (process.pid, SIGKILL)]


def test_posix_never_signals_a_group_the_harness_itself_belongs_to(world):
    """A child that inherited the harness's group cannot be stopped by group signal: that
    group contains the harness. `detached=False` means stop this child and accept that a
    descendant may survive."""
    process = FakeProcess()
    world.os.name = "posix"
    run_kill(world, process, detached=False)
    assert world.os.groups == []
    assert process.signalled == ["terminate"]


def test_posix_escalates_a_non_leader_to_the_direct_kill(world):
    process = FakeProcess(ignores=("terminate",))
    world.os.name = "posix"
    run_kill(world, process, detached=False)
    assert world.os.groups == []
    assert process.signalled == ["terminate", "kill"]


def test_a_group_signal_that_does_not_land_falls_back_to_the_direct_child(world):
    process = FakeProcess(ignores=("terminate",))
    world.os.name = "posix"
    world.os.killpg_fails = True
    run_kill(world, process, detached=True)
    assert "terminate" in process.signalled
    assert "kill" in process.signalled


def test_nothing_is_raised_when_the_whole_world_refuses(monkeypatch):
    """A kill that cannot finish must not arrive as a traceback in the caller."""

    class Ruined:
        pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise OSError("no wait for you")

        def terminate(self):
            raise OSError("no terminate")

        def kill(self):
            raise OSError("no kill")

    class RuinedOs:
        name = "posix"

        def killpg(self, pgid, sig):
            raise OSError("no group")

    monkeypatch.setattr(proc_mod, "os", RuinedOs())
    monkeypatch.setattr(proc_mod, "subprocess", FakeSubprocess())
    monkeypatch.setattr(proc_mod, "signal",
                        type("S", (), {"SIGTERM": SIGTERM, "SIGKILL": SIGKILL}))
    assert proc_mod.kill_tree(Ruined(), detached=True) is None


def test_an_unattended_child_leads_its_group_and_keeps_the_console_it_was_given():
    """What "unattended" means as shipped, including the one axis it deliberately does not take.

    The child must lead its own group, so a Ctrl+C at the parent misses it. It must not be handed a
    console flag, and the two candidates are refused for different reasons, both measured by starting a
    parent into a real console (`CREATE_NEW_CONSOLE`) rather than observing from the harness's
    window-less pseudo-console:

      * `DETACHED_PROCESS` is excluded by evidence that it does the wrong thing: the child came back with
        a nonzero console window owned by a *different* process. A console, not the absence of one.
      * `CREATE_NO_WINDOW` is excluded only by absence of evidence — it did measure as genuinely windowless.
        It stays out because the hazard it answers was never produced: an attached child survived its
        parent being killed, and a simulated `CTRL_CLOSE_EVENT` killed nothing, sender included.

    So this assertion is not a shrug about an unverifiable flag. It records that one candidate is known
    wrong, the other is unneeded by every measurement available, and adding either means first watching a
    child die.
    """
    kwargs = proc_mod.popen_unattended_kwargs()
    if sys.platform == "win32":
        flags = kwargs["creationflags"]
        assert flags & 0x00000200, "its own group: the parent's Ctrl+C must miss it"
        assert not flags & 0x00000008, "DETACHED_PROCESS measured as producing a console, not none"
        assert not flags & 0x08000000, "no window was ever shown to be needed; see popen_unattended_kwargs"
    else:
        assert kwargs == {"start_new_session": True}


def test_detach_kwargs_says_which_stop_path_applies():
    """The start and the stop are one contract: whatever this returns on the running platform
    is exactly what `kill_tree(detached=True)` is allowed to assume about the child."""
    kwargs = proc_mod.popen_detach_kwargs()
    if sys.platform == "win32":
        assert kwargs == {"creationflags": 0x00000200}, "the child must escape the parent's Ctrl+C"
    else:
        assert kwargs == {"start_new_session": True}, "the child must lead its own group"


class FakeKernel32:
    """The win32 surface a `Boundary` uses, recording every call.

    `assign_ok` is the interesting knob. Assignment fails on a host whose own process already sits in a job
    that cannot nest, which is what running inside a container looks like from here; the boundary's contract
    then becomes *degrade and say so*, and that has to be reachable without a container.
    """

    def __init__(self, *, handle=1234, assign_ok=True, assign_raises=False):
        self.handle = handle
        self.assign_ok = assign_ok
        self.assign_raises = assign_raises
        self.calls: list[tuple] = []

    def CreateJobObjectW(self, sa, name):
        self.calls.append(("create", sa, name))
        return self.handle

    def AssignProcessToJobObject(self, job, proc):
        self.calls.append(("assign", job, proc))
        if self.assign_raises:
            raise OSError("the stub raises where ctypes would return a handle it cannot convert")
        return 1 if self.assign_ok else 0

    def SetInformationJobObject(self, job, info_class, info, size):
        self.calls.append(("setinfo", info_class, size))
        return 1

    def TerminateJobObject(self, job, code):
        self.calls.append(("terminate", job, code))
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("close", handle))
        return 1


class FakeChild:
    """A started child: a pid, and a process handle that is `None` when the handle is not to be found."""

    def __init__(self, pid=4242, handle=99):
        self.pid = pid
        self._handle = handle


def job_api(monkeypatch, **kwargs):
    """Put a recording kernel32 behind `Boundary`, returning (stub, factory)."""
    import ctypes
    import ctypes.wintypes

    stub = FakeKernel32(**kwargs)
    monkeypatch.setattr(proc_mod, "_job_api", lambda: (ctypes, ctypes.wintypes, stub))
    return stub


def test_the_job_carries_the_bit_that_kills_the_tree_when_we_die():
    """`KILL_ON_JOB_CLOSE` is what covers the case no signal can: this process is killed outright, its
    handles close, and the tree goes with them instead of being orphaned to init.

    The values are pinned to the numbers Windows publishes, not to the module's own constants, so editing
    a constant here has to be a deliberate act rather than a typo that still runs.
    """
    import ctypes
    import ctypes.wintypes

    info = proc_mod.kill_on_close_info(ctypes, ctypes.wintypes)
    assert info.BasicLimitInformation.LimitFlags == 0x00002000       # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert proc_mod._EXTENDED_LIMIT_INFORMATION == 9                 # JobObjectExtendedLimitInformation


def test_the_limit_really_is_handed_to_the_job(monkeypatch):
    """The struct is one thing; the call that installs it is another. The size is compared against the very
    struct the builder makes, so a call that installs a different (or no) limit reddens this."""
    import ctypes

    stub = job_api(monkeypatch)
    boundary = proc_mod.Boundary(group=True)
    boundary.attach(FakeChild())
    expected_size = ctypes.sizeof(proc_mod.kill_on_close_info(ctypes, ctypes.wintypes))
    assert ("setinfo", 9, expected_size) in stub.calls, stub.calls


def test_a_rejected_assignment_leaves_no_claim_of_teeth(monkeypatch, world):
    """Where the job cannot be made, the boundary must stop promising what it cannot do. `teeth` is what the
    shell tool consults before saying "everything it spawned went with it"; claiming it on a host that
    refused the assignment would be a receipt for work that was never done."""
    stub = job_api(monkeypatch, assign_ok=False)
    boundary = proc_mod.Boundary(group=True)
    boundary.attach(FakeChild())
    assert ("assign", stub.handle, 99) in stub.calls
    assert ("close", stub.handle) in stub.calls, "the job was created and never released"
    assert boundary.teeth is False


def test_an_unconvertible_handle_degrades_rather_than_reaching_the_model(monkeypatch, world):
    """`Popen` has never promised a public handle. Reading it must not raise into the shell tool, and must
    not leave a half-built boundary behind that reports itself as armed."""
    stub = job_api(monkeypatch)
    boundary = proc_mod.Boundary(group=True)
    boundary.attach(FakeChild(handle=None))
    assert not [c for c in stub.calls if c[0] == "assign"]
    assert boundary.teeth is False


def test_stopping_the_boundary_ends_the_job_rather_than_the_direct_child(monkeypatch, world):
    """The point of the object: the direct child may already be gone, and `kill_tree` returns early in
    exactly that case. So `stop()` talks to the job and nothing else."""
    stub = job_api(monkeypatch)
    boundary = proc_mod.Boundary(group=True)
    boundary.attach(FakeChild())
    stub.calls.clear()
    boundary.stop()
    assert stub.calls == [("terminate", stub.handle, 1)]
    assert world.os.groups == [], "no group signal on top of a job that already covers the tree"


def test_close_is_idempotent_and_a_stop_after_it_is_silent(monkeypatch, world):
    """`close` runs in a `finally`; a second close, or a stop after close, must not raise and must not
    signal a handle that was already given back."""
    stub = job_api(monkeypatch)
    boundary = proc_mod.Boundary(group=True)
    boundary.attach(FakeChild())
    boundary.close()
    stub.calls.clear()
    boundary.close()
    boundary.stop()
    assert stub.calls == []
    assert world.os.groups == []


def test_off_windows_the_boundary_is_the_group_the_child_leads(monkeypatch, world):
    """There is no job object off Windows. The group is the boundary — which is why the constructor needs
    the caller to assert it started a group leader: a group signal aimed at a group the harness belongs to
    takes the harness down."""
    world.os.name = "posix"
    monkeypatch.setattr(proc_mod, "_job_api", lambda: None)

    leader = proc_mod.Boundary(group=True)
    leader.attach(FakeChild(pid=777))
    assert leader.teeth is True
    leader.stop()
    assert world.os.groups == [(777, SIGKILL)]

    world.os.groups.clear()
    not_leader = proc_mod.Boundary(group=False)
    not_leader.attach(FakeChild(pid=778))
    assert not_leader.teeth is False
    not_leader.stop()
    assert world.os.groups == []


def test_a_detached_group_is_killed_even_after_its_leader_went_first(world):
    """A group leader that dies on SIGTERM says nothing about the rest of the group.

    The grace wait watches the leader's handle, which is the only handle there is, so a member that ignored
    the signal left the function returning as though the group had stopped. The group is therefore killed
    whenever it was signalled at all — whether it still holds anything is not a question the leader's handle
    can answer, and `ProcessLookupError` is swallowed by the same helper that reports it."""
    world.os.name = "posix"
    process = FakeProcess()
    run_kill(world, process)
    assert process.poll() is not None, "the stub did not behave as a leader that died on TERM"
    assert (process.pid, SIGKILL) in world.os.groups, \
        f"the group was never killed: {world.os.groups}"
