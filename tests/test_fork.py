"""Fork: branch a conversation into a new run.

The pure-module tests for the fork's three functions: the branch-point resolution (relaxing the
cut's `ended` refusal, keeping its open-call and detached-step refusals), the prefix seed (carrying
the source's lines verbatim, minting the `fork` record, leaving the source untouched), and the grant
inheritance (the reach the source began with). The integration test drives the fork's seeded log
through a fresh Agent's replay, against a scripted backend, the way the resume tests do — the whole
fork contract must be checkable in-process, with no server.
"""
from sah import fork, model
from sah.agent import Agent
from sah.model import Caps, TextDelta, Usage
from sah.record import Log


def _tools():
    def read(path: str) -> str:
        """Read a file."""
        return "contents of " + path

    read.access, read.address = "read", "path"
    read.description = "Read a file."
    return {"read": read}


def _source(tmp_path, run_id="src", *, ended=True):
    """A source log: a `started` head carrying grants, two exchanges, and (by default) an `ended`
    tail. The grants are what a fork inherits, so they are real roots, not decoration."""
    log = Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id=run_id, root=str(tmp_path),
               grants={"read": ["/work"], "write": ["/out"], "exec": False,
                       "unrestricted": False, "project_notes": False})
    log.append("message", "user", role="user", text="first question")
    log.append("message", "model", role="assistant", text="first answer")
    log.append("message", "user", role="user", text="second question")
    log.append("message", "model", role="assistant", text="second answer")
    if ended:
        log.append("ended", "core", reason="complete", exit_code=0)
    return log


def _lines(log):
    return [line for line in log.path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- branch-point resolution --------------------------------------------------

def test_a_fork_may_branch_from_the_source_ending(tmp_path):
    """The cut's `ended` refusal is relaxed for a fork: branching from a finished conversation is
    the point, and the carried `ended` step is not model-facing, so the fork's replay starts fresh
    from the full carried conversation. A cut to the ending is refused; a fork of it is not."""
    log = _source(tmp_path)
    steps, _ = log.read()
    ended_id = log.of_kind("ended")[0]["id"]
    target, error = fork.resolve_target(steps, ended_id.split("-", 1)[0])
    assert error == "" and target == ended_id, \
        f"a fork of a finished run is refused where a cut is: {error!r}"


def test_a_fork_still_refuses_a_prefix_that_owes_a_result(tmp_path):
    """The cut's open-call refusal is kept: a carried conversation that ends mid-exchange would be
    refused by the server it is sent to. A source that ended on an unanswered tool call is not a
    branch point at the call."""
    log = Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id="src", root=str(tmp_path))
    msg = log.append("message", "model", role="assistant", text="let me look")
    call = log.append("call", "model", refs=msg["id"], name="read", call_id="c1", args={"path": "a"})
    steps, _ = log.read()
    target, error = fork.resolve_target(steps, call["id"].split("-", 1)[0])
    assert target is None and "owed a result" in error, \
        f"the open call is not refused: {error!r}"


def test_a_fork_refuses_a_step_an_earlier_cut_detached(tmp_path):
    """A step a cut detached is not a branch point: carrying it would rebuild a prefix that holds a
    tail the source no longer owns. The refusal names the detached continuation, the way a cut's does."""
    log = _source(tmp_path, ended=False)
    # a cut to the first answer detaches the second exchange
    log.append("cut", "core", to=log.of_kind("message")[1]["id"])
    steps, _ = log.read()
    # the second question is detached by the cut; forking there is refused
    second_q_id = log.of_kind("message")[2]["id"]
    target, error = fork.resolve_target(steps, second_q_id.split("-", 1)[0])
    assert target is None and "detached" in error, \
        f"a detached step is a branch point where it should not be: {error!r}"


def test_a_fork_resolves_a_step_of_the_conversation_in_force(tmp_path):
    """A step the conversation in force holds is a branch point: the target is the step's id, and the
    error is empty. The resolution walks `lineage`, so the target is a step of the conversation in
    force, not merely a line in the file."""
    log = _source(tmp_path)
    steps, _ = log.read()
    target, error = fork.resolve_target(steps, "2")       # the first answer
    assert error == "" and target == "000002-message", (target, error)


# --- the prefix seed ----------------------------------------------------------

def test_the_seed_carries_the_prefix_verbatim_and_mints_the_fork_record(tmp_path):
    """The fork's log is the source's prefix ending at the target, verbatim and with the source's
    step ids, followed by the `fork` record that names the source and the branch point. The prefix
    is the carried conversation; the `fork` record is the act that carried it. The carried lines are
    the source's own bytes — a fork that re-serialised them would be a copy, not a carry."""
    log = _source(tmp_path)
    new_log = str(tmp_path / "fork" / "log.jsonl")
    carried = fork.seed(_lines(log), "000002-message", "src", new_log)
    assert carried == 3                                     # started + two messages
    steps, _ = Log(new_log).read()
    # the carried prefix: the source's lines, same ids, in order
    assert [s["id"] for s in steps[:3]] == ["000000-started", "000001-message", "000002-message"]
    assert [s["text"] for s in steps[1:3]] == ["first question", "first answer"]
    # the `fork` record: the next seq, naming the source and the branch point
    fork_step = steps[3]
    assert fork_step["kind"] == "fork" and fork_step["id"] == "000003-fork"
    assert fork_step["source_run"] == "src" and fork_step["source_step"] == "000002-message"
    # the tail the fork did not carry: the second exchange and the ending stay out of the fork's log
    assert not any(s["id"] == "000003-message" for s in steps)
    assert log.of_kind("ended")[0]["id"] not in [s["id"] for s in steps]


def test_the_seed_leaves_the_source_log_untouched(tmp_path):
    """Forking reads a prefix, it does not perform one: the source log's bytes are unchanged after
    the seed. A fork that wrote the source would be a cut wearing a fork's name."""
    log = _source(tmp_path)
    before = log.path.read_bytes()
    fork.seed(_lines(log), "000002-message", "src", str(tmp_path / "fork" / "log.jsonl"))
    assert log.path.read_bytes() == before, "the source log was written by a fork"


def test_the_seed_carries_a_source_cut_and_its_lineage(tmp_path):
    """A source that was cut carries the cut: the fork's log holds the source's `cut` step, and the
    fork's replay applies the source's lineage over the carried prefix, so the carried cut detaches
    the same tail here as it did in the source. The prefix is carried whole, not projected."""
    log = _source(tmp_path, ended=False)
    log.append("cut", "core", to=log.of_kind("message")[1]["id"])     # cut to the first answer
    new_log = str(tmp_path / "fork" / "log.jsonl")
    # branch at the cut step itself: the prefix is everything up to and including the cut
    cut_id = log.of_kind("cut")[0]["id"]
    fork.seed(_lines(log), cut_id, "src", new_log)
    steps, _ = Log(new_log).read()
    assert any(s["kind"] == "cut" for s in steps), \
        "the source's cut is not carried: the fork's replay would not detach the source's tail"
    # the fork's replay applies the carried cut: the second exchange is out of the conversation
    fresh = Agent(backend=model.Fake([]), tools=_tools(), log=Log(new_log), root=tmp_path)
    fresh.replay()
    assert [m.content for m in fresh.messages] == ["first question", "first answer"], \
        "the carried cut did not detach the source's tail in the fork's conversation"


# --- the grant inheritance ----------------------------------------------------

def test_a_fork_inherits_the_source_read_and_write_roots(tmp_path):
    """A fork is the same run on a different path: it begins with the reach the source began with,
    read and write roots both, from the source's newest `started` grants."""
    log = _source(tmp_path)
    steps, _ = log.read()
    read_roots, write_roots = fork.inherit_grants(steps)
    assert read_roots == ["/work"] and write_roots == ["/out"]


def test_a_fork_of_a_source_without_grants_inherits_nothing(tmp_path):
    """A source whose `started` line carries no grants (an old or a no-confine run) gives a fork no
    inherited roots: the fork's own flags say its reach, and an empty inheritance is not a refusal,
    it is the honest shape of a source that recorded no reach."""
    log = Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id="src", root=str(tmp_path))
    steps, _ = log.read()
    read_roots, write_roots = fork.inherit_grants(steps)
    assert read_roots == [] and write_roots == []


def test_a_fork_inherits_from_the_source_newest_started_line(tmp_path):
    """A source resumed under a new reach records a new `started` line; the fork inherits the reach
    the source last put in force, not the reach it began with. The newest `started` line is the
    source's last run."""
    log = Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id="src", root=str(tmp_path),
               grants={"read": ["/old"], "write": [], "exec": False,
                       "unrestricted": False, "project_notes": False})
    log.append("message", "user", role="user", text="a turn")
    log.append("message", "model", role="assistant", text="a reply")
    # a resumed run under a new reach
    log.append("started", "core", run_id="src", root=str(tmp_path),
               grants={"read": ["/new"], "write": ["/out2"], "exec": False,
                       "unrestricted": False, "project_notes": False})
    steps, _ = log.read()
    read_roots, write_roots = fork.inherit_grants(steps)
    assert read_roots == ["/new"] and write_roots == ["/out2"], \
        "the fork inherited the source's first reach, not the reach its last run put in force"


# --- the integration: the fork's log replays the carried conversation ---------

def test_the_fork_log_replays_the_carried_conversation(tmp_path):
    """The whole contract in one: a fork's seeded log, replayed by a fresh Agent, carries the source's
    conversation (up to the branch point) into the fork's message list, and the fork diverges from
    there under its own objective. The carried conversation is the source's, the divergence is the
    fork's, and the fork's `started` head names the fork, not the source."""
    log = _source(tmp_path)
    new_log = str(tmp_path / "fork" / "log.jsonl")
    fork.seed(_lines(log), "000002-message", "src", new_log)
    scripted = [[TextDelta("the branch"), Usage(10, 3), model.Stopped("stop")]]
    agent = Agent(backend=model.Fake(scripted, caps=Caps(context_tokens=1000, model="t")),
                  tools=_tools(), log=Log(new_log), max_turns=30, run_id="fork",
                  root=tmp_path / "fork", queue=tmp_path / "fork" / "control.md")
    # the carried conversation comes back: the source's first exchange, verbatim
    restored = agent.replay()
    assert restored >= 2
    assert [m.content for m in agent.messages] == ["first question", "first answer"], \
        f"the carried conversation is not the source's: {[m.content for m in agent.messages]}"
    # the fork diverges under its own objective, and its `started` head names the fork
    agent.run("the branch objective")
    steps = agent.log.steps()
    started = [s for s in steps if s["kind"] == "started"]
    assert started[-1]["run_id"] == "fork", "the fork's `started` head names the source, not the fork"
    assert started[-1]["id"] != "000000-started", \
        "the fork's `started` head reuses the source's seq: the carried prefix and the fork collide"
