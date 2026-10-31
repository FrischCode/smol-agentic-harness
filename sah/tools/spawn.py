"""sah.tools.spawn — the spawn capability: the tools that start child runs and read them back.

These are the tools a top-level run offers its model (gated in `sah.cli`): `spawn` starts a child
run, and `child_status`/`child_wait` read the children's logs back. The machinery they call is
`sah.spawn` — the spawner arrives duck-typed through the closure, so this module imports nothing
from it, and the capability's only edge is the contract it declares against (`sah.contract`). The
schema here exposes a task, never a scope: what a child inherits, and what it can never be handed,
is projected into the child's command line by that machinery, not by these declarations.
"""
from __future__ import annotations

import os
from typing import Annotated

from sah.contract import Param

def tool_for(spawner, mint_id, record):
    """The `spawn` tool bound to a spawner. `mint_id` produces the child run id -- owned by the
    caller, and in the CLI that is `Spawner.next_id`, which reads the runs root rather than a counter, so a
    resumed run cannot hand out an id a live child already holds -- and `record` appends a step announcing
    the child,
    so a spawned run leaves a trace without the tool reaching into the log directly.

    The function's parameters -- objective, and at most a model/turn override -- are the whole
    surface the model sees. There is no path, no grant, no run id: those are not offered to be
    forged, because they are not in the schema at all."""

    def spawn(objective: Annotated[str, Param("what the child run should accomplish, in full — it "
                                              "sees none of your conversation")],
              model: Annotated[str, Param("model for the child; empty uses this run's model")] = "",
              max_turns: Annotated[int, Param("turn limit for the child; 0 uses this run's")] = 0) -> str:
        """The objective must stand alone, because the child sees none of this conversation: it is
        another run with its own window, and the words that reach it are the objective, not the
        exchange that produced it. Inheritance is deliberately one-directional and capped — the child
        can do what this run can, no more, and can never be given a grant its parent holds that it
        was not told about (see `Spawner.build_command`)."""
        if not objective.strip():
            raise ValueError("spawn needs an objective; a child with no task does nothing")
        run_id = mint_id()
        _proc, root = spawner.spawn(run_id, objective, model=model, max_turns=max_turns)
        record(run_id, root)
        return (f"spawned run {run_id}. Its log: {os.path.join(root, 'log.jsonl')}. It is "
                "addressable, not attached: it keeps running if this run stops. Read its log back "
                "to see what it did.")

    spawn.description = ("Start a child run that works on `objective` and returns its id immediately. "
                         "The child has its own log; read it back later. It is not attached to this run "
                         "and will not be stopped if this one ends. It inherits this run's grants — it "
                         "can do what you can, no more. Use it to pursue a second task while you "
                         "continue, or to hand off work that outlives the session.")

    def child_status(child_id: Annotated[str, Param("the child run id to inspect; empty lists all "
                                                   "your children and whether each finished")] = "") -> str:
        """State comes from the children's logs, not from this process: a child that outlived the run
        that started it (or a run that restarted) is still fully reported, because the log is the
        report and nothing here needs the live process. The unanswered-question flag is the one fact
        a bare state line would hide — a child that is waiting is not working."""
        return spawner.status(child_id)

    child_status.description = ("Report on the child runs this run started: their state, exit codes, "
                                "what each last said, and whether one is waiting on an unanswered "
                                "question. Answered from the children's logs, so it works even if you "
                                "stopped watching or this run restarted. Usually you will not need it "
                                "— a child's artifact lands where you granted and its id you already "
                                "have — this is the way to check on a child, or to ask 'are they done "
                                "and how did they do'.")

    def child_wait(child_id: Annotated[str, Param("the child run id to wait for (one name, not a path)")],
                   timeout: Annotated[int, Param(
                       "max seconds to block, 1-300 (default 60); on timeout it returns without the "
                       "child having finished")] = 60) -> str:
        """Bounded on purpose: an unbounded wait is a run that can be stalled by a child that is
        stuck, and a stall this run can feel is a fact the operator should see, not a timeout the
        operator has to find. After `timeout` the child's current state comes back, which names the
        situation — still running, or finished while the wait was deciding."""
        return spawner.wait(child_id, timeout)

    child_wait.description = ("Block until the named child finishes or `timeout` seconds pass, then "
                              "report its final state (the same detail child_status gives). Use it when "
                              "you must have a child's result before you can continue — otherwise prefer "
                              "to keep working and read its log later, since this run's turn is held "
                              "while it waits. It is bounded: after `timeout` it returns the child's "
                              "current state rather than waiting forever, so a stuck child cannot "
                              "stall you — check again, or move on.")

    # Declared `self`: a child is addressed by id inside this run's own children/ tree, never by path,
    # so the file grants have nothing to judge here. The id is recorded, so the log says which child a
    # call concerned -- the child's own log remains the report.
    spawn.access, spawn.address = "self", None
    child_status.access, child_status.address = "self", "child_id"
    child_wait.access, child_wait.address = "self", "child_id"

    return {"spawn": spawn, "child_status": child_status, "child_wait": child_wait}


class _SpawnExtension:
    """The spawn/child_status tools bound to a spawner, wrapped as an extension (duck-typed:
    `assemble` reads `.tools`). A top-level run gets it and a child does not: a child inherits the
    parent's grants, so offering those widens nothing, but offering this one would let the hand-off
    repeat without limit."""

    def __init__(self, funcs):
        self.tools = tuple(funcs)


def extension(spawner, mint_id, record):
    """The spawn capability, as an extension for `sah.ext.assemble`. Must be in the extension list
    before the Agent is built, which derives its advertised schemas once — a tool registered after
    would exist but be invisible to the model."""
    return _SpawnExtension(tool_for(spawner, mint_id, record).values())
