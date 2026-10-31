"""spawn: a child run that is addressable, not attached, and whose scope the caller cannot touch.

`subprocess_spawn` is replaced everywhere so no test launches a real run -- the contract worth
checking is the command line that would be run, the detach, and what the tool refuses to accept. A
spawn test that forked a live agent would be slow, flaky, and would test the child's reasoning
rather than the harness's boundaries.
"""

import os

import pytest

from sah import record as rec, spawn
from sah.tools.spawn import tool_for


@pytest.fixture
def spawner(tmp_path, monkeypatch):
    """A Spawner plus a captured-argv spy: monkeypatching subprocess_spawn records the argv and the
    detach without ever starting a process."""
    made = {}

    def fake(argv, root, api_key=""):
        made["argv"] = argv
        made["root"] = root
        made["api_key"] = api_key        # the key travels by environment, so the spy sees it apart
        return FakeProc()
    monkeypatch.setattr(spawn, "subprocess_spawn", fake)
    s = spawn.Spawner(runs_root=str(tmp_path / "children"), model="m1", host="h", port=9090,
                      api_key="secret", think="off", context=4096,
                      read_roots=["/data/read"], write_roots=["/data/write"],
                      allow_exec=False, memory="", max_turns=8)
    s._made = made
    return s


class FakeProc:
    pid = 12345

    def poll(self):
        return None


# --- the schema is a task, never a scope -------------------------------------

def test_the_spawn_tool_takes_only_a_task():
    """The whole boundary. The old tree's spawn schema came from a run-config vocabulary and
    advertised run_id/sessions_dir/manifest, so a model call could write a child's grants. This one
    exposes objective + model/turn overrides and nothing that names a place or a permission."""
    tools = tool_for(None, lambda: "x", lambda *_: None)
    import inspect
    params = set(inspect.signature(tools["spawn"]).parameters)
    assert params == {"objective", "model", "max_turns"}
    for forbidden in ("run_id", "sessions_dir", "manifest", "grant", "write", "path", "root"):
        assert forbidden not in params


def test_an_objective_is_passed_to_the_child_verbatim(spawner):
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("count the files in /data")
    argv = spawner._made["argv"]
    assert "--task" in argv and argv[argv.index("--task") + 1] == "count the files in /data"


def test_an_empty_objective_is_refused(spawner):
    """A child with no task does nothing and costs a whole run to do it."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    with pytest.raises(ValueError, match="objective"):
        tools["spawn"]("   ")


# --- the child inherits the parent's configuration, as visible flags ---------

def test_the_child_gets_the_parents_grants_as_explicit_flags(spawner):
    """Inheritance is the harness passing its own flags to the child's command line -- visible in
    the child's banner and process table -- not a manifest the child silently trusts."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("work")
    argv = spawner._made["argv"]
    assert "--grant" in argv and "/data/read" in argv
    assert "--write" in argv and "/data/write" in argv


def test_a_child_gets_exactly_the_parents_grants_no_more(spawner):
    """The anti-escalation guarantee, stated exactly: the set of --grant/--write paths in the
    child's command equals the parent's, with nothing extra. A test that only checks the parent's
    grants are *present* would pass even if a root grant had been slipped in -- inheritance turning
    into escalation is precisely the bug this guards."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("work")
    argv = spawner._made["argv"]
    grants = [argv[i + 1] for i, a in enumerate(argv) if a == "--grant"]
    writes = [argv[i + 1] for i, a in enumerate(argv) if a == "--write"]
    assert grants == ["/data/read"]
    assert writes == ["/data/write"]


def test_a_child_is_never_given_exec_the_parent_lacks(spawner):
    """Widening cannot be inherited. The parent has no --exec, so the child's command has none;
    there is no channel through which the tool adds a permission."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("work")
    assert "--exec" not in spawner._made["argv"]


def test_the_child_runs_a_single_task_not_an_stdin_loop(spawner):
    """An unattended child has no terminal. --task makes it run once and exit; a child waiting on
    stdin is a child that hangs forever with nobody to type into it."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("do the thing")
    assert "--task" in spawner._made["argv"]


def test_a_model_override_changes_only_the_child_model(spawner):
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    tools["spawn"]("work", model="bigger")
    argv = spawner._made["argv"]
    assert argv[argv.index("--model") + 1] == "bigger"
    # but the backend host/port are still the parent's -- an override is a model choice, not a
    # redirect to a different (possibly attacker-chosen) server
    assert argv[argv.index("--host") + 1] == "h"
    assert argv[argv.index("--port") + 1] == "9090"


# --- addressable, not attached -----------------------------------------------

def test_the_child_runs_detached_so_it_outlives_the_parent(tmp_path, monkeypatch):
    """The addressable rule: a spawned run is started in its own process group and is not killed
    when the parent exits. Assert the real subprocess_spawn asks for detach kwargs -- the fake would
    hide exactly this."""
    captured = {}

    class Popen:
        def __init__(self, argv, **kw):
            captured["argv"], captured["kw"] = argv, kw
            self.pid = 1
    monkeypatch.setattr(spawn.process, "popen_unattended_kwargs",
                        lambda: {"creationflags": 0x208})
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", Popen)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    _proc, root = s.spawn("kid1", "objective")
    if "creationflags" in captured["kw"]:
        assert captured["kw"]["creationflags"] & 0x200, "its own group: the parent's Ctrl+C misses it"
    else:
        assert captured["kw"].get("start_new_session"), "its own session, which is the POSIX form"


def test_a_resumed_run_does_not_reuse_a_child_id(tmp_path):
    """The id used to come from a counter held in the parent process, so a resumed run restarted at
    `child1`: the new child appended to the previous child's log with colliding ids, and `child_status`
    found the OLD `ended` step and reported the new child finished before it had started. The parent acted
    on a stale result. Minting therefore reads the filesystem the ids are named after.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    (tmp_path / "kids" / "r-child1").mkdir(parents=True)
    assert s.next_id("r") == "r-child2"


def test_ids_keep_counting_past_ten(tmp_path):
    """`max()` over names would answer `child9` after nine children exist, handing out an id already used.
    A resumed run that had reached twelve has to continue at thirteen.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    for n in [1, 2, 9, 10, 11, 12]:
        (tmp_path / "kids" / f"r-child{n}").mkdir(parents=True)
    assert s.next_id("r") == "r-child13"


def test_children_of_different_runs_do_not_share_a_number_line(tmp_path):
    """Two runs can be resumed in the same folder, and each counts its own children: the number is scoped
    to the parent's id, not to the directory."""
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    for name in ["r-child1", "other-child1", "other-child2"]:
        (tmp_path / "kids" / name).mkdir(parents=True)
    assert s.next_id("r") == "r-child2"
    assert s.next_id("other") == "other-child3"


def test_a_folder_that_is_not_a_numbered_child_does_not_stop_the_mint(tmp_path):
    """`runs_root` is a directory the operator can put anything in, and ids may one day carry a suffix.
    Reading the highest number must skip what has no number instead of raising inside the spawn tool -- a
    mint that crashes on `r-child-notes` means the run cannot hand off at all.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    for name in ["r-child-notes", "r-child2b", "unrelated"]:
        (tmp_path / "kids" / name).mkdir(parents=True)
    assert s.next_id("r") == "r-child1"


def test_spawning_into_an_existing_child_folder_is_refused(tmp_path):
    """Reusing a folder is how a stale read happens, so a collision is a refusal rather than a silent
    append. It can still occur -- two parents resuming the same run, or a folder left by hand -- and the
    run must find out by being told, not by reading someone else's log later.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    (tmp_path / "kids" / "r-child1").mkdir(parents=True)
    with pytest.raises(Exception) as raised:
        s.spawn("r-child1", "objective")
    assert "r-child1" in str(raised.value)


def test_the_child_is_pointed_at_the_memory_tree_its_parent_used(tmp_path, monkeypatch):
    """The child's working directory is its own run folder, so a relative `--memory notes` means
    `children/<id>/notes` -- the child silently boots an empty memory and writes its notes where nothing
    else will look, while the parent's tree sits untouched. `runs_root` is already made absolute (which is
    why `--run-dir` never showed the bug); the memory path needs the same treatment.
    """
    monkeypatch.chdir(tmp_path)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1, memory="notes")
    (tmp_path / "notes").mkdir()
    argv = s.build_command("kid1", "objective")
    got = argv[argv.index("--memory") + 1]
    assert os.path.isabs(got), f"a relative memory path means a different folder in the child: {got}"
    assert os.path.normcase(os.path.realpath(got)) == os.path.normcase(
        os.path.realpath(str(tmp_path / "notes")))


def test_the_api_key_never_reaches_the_child_command_line(tmp_path):
    """The prompt goes on the command line on purpose: an unattended child should say what it was told to
    do, in the one place anyone inspecting the process can look. A secret has the opposite requirement --
    argv is readable by every local user (`ps`, Task Manager, WMI) for the life of the process, and then
    lands in whatever collects process telemetry. The child reads the key from its environment, which
    `OpenAI` already falls back to.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      api_key="sk-secret-do-not-log")
    argv = s.build_command("kid1", "objective")
    joined = " ".join(str(a) for a in argv)
    assert "sk-secret-do-not-log" not in joined, joined
    assert "--api-key" not in argv, f"the flag itself invites the key back onto argv: {argv}"


def test_the_child_receives_the_key_in_its_environment(tmp_path, monkeypatch):
    """Withholding it from argv is only half the act: the child still has to get it, or every spawned run
    silently loses the key its parent was given."""
    captured = {}

    class Popen:
        def __init__(self, argv, **kw):
            captured["kw"] = kw
            captured["argv"] = argv
            self.pid = 1

    import subprocess
    monkeypatch.setattr(spawn.process, "popen_unattended_kwargs", lambda: {})
    monkeypatch.setattr(subprocess, "Popen", Popen)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      api_key="sk-secret-do-not-log")
    s.spawn("kid1", "objective")
    assert captured["kw"]["env"].get("SAH_API_KEY") == "sk-secret-do-not-log"
    assert "sk-secret-do-not-log" not in " ".join(str(a) for a in captured["argv"])


def test_a_child_is_told_it_may_not_spawn_in_turn(tmp_path):
    """A child inherits its parent's configuration, and until now that included `spawn`, so one confused
    model could start a line of runs each starting a line of runs -- all detached, all on the same GPU
    server, none of them bounded. The owner's ruling is that a run may hand off work once but not pass on
    the ability.

    The veto travels as a flag on the child's command line rather than as a secret, because the command
    line is where inheritance is already visible: `ps` and the child's own boot box both show the
    narrowing, so nobody has to trust a docstring about it.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    assert "--no-spawn" in s.build_command("kid1", "objective")


def test_the_spawn_veto_is_in_the_environment_as_well(tmp_path, monkeypatch):
    """A child with `--exec` can start a sah run through its shell tool, and that command line is the
    model's to write, so the flag alone is not a bound. The environment travels with the process and the
    flag does not; the cost is declared -- a child with exec can unset it, which is the cost of exec."""
    captured = {}

    class Popen:
        def __init__(self, argv, **kw):
            captured["kw"] = kw
            self.pid = 1

    import subprocess
    monkeypatch.setattr(spawn.process, "popen_unattended_kwargs", lambda: {})
    monkeypatch.setattr(subprocess, "Popen", Popen)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    s.spawn("kid1", "objective")
    assert captured["kw"]["env"].get("SAH_NO_SPAWN") == "1"


def test_the_child_is_handed_no_stdin(tmp_path, monkeypatch):
    """A spawned run has nobody at its keyboard, so stdin must not be the parent's console. Inherited,
    a child that reads input -- directly or through a tool it runs -- would take keystrokes from the
    parent's terminal, which is the defect the shell tool was given the same ruling for. EOF is the only
    honest answer for a process whose operator is a log file."""
    captured = {}

    class Popen:
        def __init__(self, argv, **kw):
            captured["kw"] = kw
            self.pid = 1
    monkeypatch.setattr(spawn.process, "popen_unattended_kwargs",
                        lambda: {"creationflags": 0x208})
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", Popen)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    s.spawn("kid1", "objective")
    assert captured["kw"].get("stdin") is subprocess.DEVNULL
    assert captured["kw"].get("creationflags") == 0x208, "the child is started unattended"


def test_the_child_log_is_the_report_not_a_terminal_buffer(spawner):
    """The return names the child's log path, because the log -- not a pipe the parent watches --
    is how anyone reads what the child did. That is the whole 'who is the reader' answer."""
    tools = tool_for(spawner, lambda: "r-child1", lambda *_: None)
    msg = tools["spawn"]("work")
    assert "log.jsonl" in msg
    assert "keeps running" in msg or "addressable" in msg


# --- the record path ---------------------------------------------------------

def test_spawning_records_a_notice(spawner):
    """A spawned run must leave a trace in the parent's log, or a run silently forks children with
    no audit of it. The record hook is passed in, so the tool does not reach into the log itself."""
    notices = []
    tools = tool_for(spawner, lambda: "r-child1",
                           lambda cid, root: notices.append(cid))
    tools["spawn"]("work")
    assert notices == ["r-child1"]


def test_each_child_gets_a_unique_id_and_folder(spawner):
    n = [0]

    def mint():
        n[0] += 1
        return f"r-child{n[0]}"
    tools = tool_for(spawner, mint, lambda *_: None)
    tools["spawn"]("a")
    first = spawner._made["root"]
    tools["spawn"]("b")
    second = spawner._made["root"]
    assert first != second                       # two children, two folders, no clobber


# --- through the loop --------------------------------------------------------

def test_a_spawn_call_through_the_loop_records_the_child(tmp_path, monkeypatch):
    """The integration: a model calls spawn, the tool runs, a notice lands in the log naming the
    child. spawn takes no address, so the policy allows it (a child inherits; nothing is widened),
    and the operation step shows the act happened."""
    from sah import model
    from sah.agent import Agent
    from sah.confine import Confine
    from sah.model import Caps, turn_text, turn_tool
    from sah.record import Log

    made = {}

    class P:
        pid = 1
        def poll(self):
            return None
    monkeypatch.setattr(spawn, "subprocess_spawn",
                        lambda argv, root, api_key="": (made.update(argv=argv), P())[1])

    log = Log(tmp_path / "log.jsonl")
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      read_roots=[str(tmp_path)])
    n = [0]

    def mint():
        n[0] += 1
        return f"parent-child{n[0]}"
    tools = tool_for(s, mint,
                           lambda cid, root: log.append("notice", "core", subject="spawn",
                                                        text=f"started {cid}"))
    agent = Agent(backend=model.Fake([
        turn_tool("c1", "spawn", {"objective": "do the subtask"}),
        turn_text("delegated")], caps=Caps(100, model="t")),
        tools=tools, log=log, root=tmp_path, policy=Confine())
    agent.run("delegate this")
    notices = [st for st in log.steps() if st["kind"] == "notice"]
    assert any(nt["subject"] == "spawn" for nt in notices)
    ops = [st for st in log.steps() if st["kind"] == "operation"]
    assert any(o["op"] == "spawn" and o["outcome"] == "allowed" for o in ops)
    assert "--task" in made["argv"] and "do the subtask" in made["argv"]


def test_the_child_is_told_where_sah_lives(tmp_path, monkeypatch):
    """A detached child launched with a foreign cwd cannot import the `sah` source package unless
    its location is named: the parent only works because its own cwd is the repo. The child must get
    PYTHONPATH pointed at the package parent or it dies on 'No module named sah' before a turn."""
    captured = {}

    class Popen:
        def __init__(self, argv, **kw):
            captured["env"] = kw.get("env", {})
            self.pid = 1
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", Popen)
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    s.spawn("k", "objective")
    import sah
    import os
    parent = os.path.dirname(os.path.dirname(os.path.abspath(sah.__file__)))
    assert parent in captured["env"]["PYTHONPATH"]


# --- the inspection tool: check on the children you own -----------------------


def _child(tmp_path, cid, steps):
    d = tmp_path / "children" / cid
    d.mkdir(parents=True)
    with open(d / "log.jsonl", "w", encoding="utf-8") as f:
        for s in steps:
            f.write(rec.dumps(s) + "\n")


def _spawner(tmp_path):
    return spawn.Spawner(runs_root=str(tmp_path / "children"), model="m", host="h", port=1)


def test_status_with_no_children_says_so(tmp_path):
    assert "no children" in _spawner(tmp_path).status()


def test_status_lists_each_child_with_its_state(tmp_path):
    _child(tmp_path, "a", [rec.make("ended", "core", 1, reason="complete", exit_code=0)])
    _child(tmp_path, "b", [rec.make("message", "model", 1, role="assistant", text="still going")])
    out = _spawner(tmp_path).status()
    assert "a: complete (exit 0)" in out
    assert "b: running" in out           # no ended step -> still running


def test_status_detail_shows_what_a_finished_child_said(tmp_path):
    _child(tmp_path, "kid", [
        rec.make("message", "model", 1, role="assistant", text="CHILD-WROTE-IT"),
        rec.make("ended", "core", 2, reason="complete", exit_code=0)])
    detail = _spawner(tmp_path).status("kid")
    assert "complete (exit 0)" in detail and "last said: CHILD-WROTE-IT" in detail


def test_status_surfaces_a_child_that_asked_and_was_hanging(tmp_path):
    """The async truth: a child with no operator is refused by default, but a parent inspecting it
    should still SEE that it asked, rather than read silence as success."""
    _child(tmp_path, "kid", [rec.make("request", "core", 1, subject="grant", item="/x")])
    assert "asked 1 question" in _spawner(tmp_path).status("kid")


def test_status_of_an_unknown_child_names_the_real_ones(tmp_path):
    _child(tmp_path, "real", [])
    out = _spawner(tmp_path).status("ghost")
    assert "no child run 'ghost'" in out and "real" in out


def test_status_refuses_a_child_id_that_is_a_path(tmp_path):
    """A child id is one folder name under the tree you own, never a path -- same intrinsic boundary
    as the memory tree. Rejecting traversal stops a status query escaping the owned tree."""
    with pytest.raises(ValueError):
        _spawner(tmp_path).status("../secret")
    with pytest.raises(ValueError):
        _spawner(tmp_path).status("a/b")


def test_child_status_is_advertised_alongside_spawn(tmp_path):
    """All three come from the spawner; a model can inspect or wait on its children only because the
    tool exists. A tool registered after the Agent is built would exist but be invisible."""
    tools = tool_for(_spawner(tmp_path), lambda: "x", lambda *a, **kw: None)
    assert set(tools) == {"spawn", "child_status", "child_wait"}
    import inspect
    assert list(inspect.signature(tools["child_status"]).parameters) == ["child_id"]
    assert list(inspect.signature(tools["child_wait"]).parameters) == ["child_id", "timeout"]


def test_status_tolerates_a_child_still_mid_write(tmp_path):
    """A running child may be caught mid-append. A status query must report on it rather than crash
    on the torn last line — and say the report is partial, because 'running' with a lost step is not
    the same answer as 'running' with an intact one."""
    d = tmp_path / "children" / "kid"
    d.mkdir(parents=True)
    good = rec.dumps(rec.make("message", "model", 1, role="assistant", text="hi"))
    (d / "log.jsonl").write_text(good + "\n" + '{"kind": "message", "id":', encoding="utf-8")  # torn
    report = _spawner(tmp_path).status("kid")
    assert "hi" in report
    assert "unreadable" in report, f"the report claimed to be whole: {report}"


# --- waiting on a child: a bounded sync block over the same log-read -----------------

def test_child_wait_returns_the_report_when_the_child_has_finished(tmp_path):
    """A finished child is seen on the first poll, so wait returns immediately with its outcome --
    no sleeping, because finishing is read from the log rather than joined as a process."""
    _child(tmp_path, "kid", [
        rec.make("message", "model", 1, role="assistant", text="CHILD-DONE"),
        rec.make("ended", "core", 2, reason="complete", exit_code=0)])
    out = _spawner(tmp_path).wait("kid", 30)
    assert "complete (exit 0)" in out and "CHILD-DONE" in out


def test_child_wait_reports_still_running_after_the_timeout(tmp_path):
    """The whole point of the bound: a child that never finishes cannot stall the caller forever. On
    timeout it hands back the current state and leaves the choice -- re-check or move on."""
    _child(tmp_path, "kid", [rec.make("message", "model", 1, role="assistant", text="still working")])
    out = _spawner(tmp_path).wait("kid", 0)          # timeout 0 -> one poll, no sleep
    assert "running" in out and "still running after 0s" in out


def test_child_wait_of_an_unknown_child_names_the_real_ones(tmp_path):
    _child(tmp_path, "real", [])
    assert "no child run 'ghost'" in _spawner(tmp_path).wait("ghost", 1) \
        and "real" in _spawner(tmp_path).wait("ghost", 1)


def test_child_wait_refuses_a_child_id_that_is_a_path(tmp_path):
    """Same intrinsic boundary as status: one folder name in the owned tree, never a path."""
    with pytest.raises(ValueError):
        _spawner(tmp_path).wait("../secret", 1)


def test_a_child_of_a_default_run_inherits_no_preference_rather_than_a_chosen_level(tmp_path):
    """The Spawner's default is a second copy of the run's default, and one that still said `off` would
    hand a child a level its parent never chose -- the exact word the default exists to avoid sending.
    argv is the only contract between the two processes, so the default is asserted there.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    argv = [str(a) for a in s.build_command("kid1", "objective")]
    assert argv[argv.index("--think") + 1] == "none", argv


def test_the_child_compacts_at_the_level_its_parent_named_for_summaries(tmp_path):
    """`--think` travels to the child and `--summary-think` must too, or a parent configured so its
    summaries run at a level its own server will take hands its child the default: every compaction
    the child attempts runs at the parent's level, on a server that spent the parent's summary calls
    refusing it. The knob is absent when the parent named no level (the child keeps its own level,
    the way the run does), and present with the named level when the parent set one — inheritance is
    a claim about the child's argv, so it is asserted there rather than on a field nobody reads.
    """
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      think="high", summary_thinking="low")
    argv = [str(a) for a in s.build_command("kid1", "objective")]
    assert argv[argv.index("--summary-think") + 1] == "low", argv
    assert argv[argv.index("--think") + 1] == "high", argv

    bare = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1, think="high")
    bare_argv = [str(a) for a in bare.build_command("kid1", "objective")]
    assert "--summary-think" not in bare_argv, \
        f"an absent level became a default the parent never chose: {bare_argv}"


def test_an_open_parent_does_not_hand_the_open_axis_to_its_child(tmp_path):
    """A child inherits the folders its parent was given and never inherits a widening — which is why the
    child's argv is assembled here rather than copied from the parent's command line.

    `--no-confine` is the widest thing a run can hold, so the ruling is written where it could be undone:
    the spawner is *told* the parent was open, and puts the flag on no child. Granting it would make one
    operator's decision at one terminal into the standing reach of every run under it, each with its own
    log that says `unrestricted: false` because its own argv never named the flag."""
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      read_roots=[str(tmp_path)], write_roots=[str(tmp_path)], unrestricted=True)
    argv = [str(a) for a in s.build_command("kid1", "objective")]
    assert "--no-confine" not in argv, f"a widening travelled to the child: {argv}"
    assert "--grant" in argv, "the folders the parent was given still travel"


def test_a_child_inherits_the_leeway_that_goes_with_the_turn_ceiling(tmp_path):
    """Grace is part of the turn ceiling's meaning — a child that inherits the ceiling inherits
    the wrap-up that goes with it, and argv is the only contract between the two processes."""
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      max_turns=8, grace_turns=5)
    argv = [str(a) for a in s.build_command("kid1", "objective")]
    assert argv[argv.index("--grace-turns") + 1] == "5", argv
    assert argv[argv.index("--max-turns") + 1] == "8", argv


def test_a_child_with_no_time_wall_gets_no_wall_flag(tmp_path):
    """0 means the operator set no wall, and forwarding a 0 would put a flag on the child's line
    that says nothing: the wall travels the way the grants do, only when one was given."""
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1)
    assert "--max-runtime" not in s.build_command("kid1", "objective")
    s = spawn.Spawner(runs_root=str(tmp_path / "kids"), model="m", host="h", port=1,
                      max_runtime=900)
    argv = [str(a) for a in s.build_command("kid1", "objective")]
    assert argv[argv.index("--max-runtime") + 1] == "900", argv
