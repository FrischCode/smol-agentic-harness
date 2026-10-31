"""sah.fork — branch a conversation into a new run.

A fork is the same conversation on a different path: a new run whose log carries another run's
prefix, verbatim and with the source's step ids, and then diverges under a new objective. It is
the cut's sibling with the opposite direction. A cut takes a run's own log and detaches its tail
so the conversation returns to a step; a fork takes another run's log and carries its prefix
into a fresh run, so the source is untouched and the branch is a new record. Where a cut's
conversation is a projection of one file, a fork's conversation is a file that began as a copy.

The carried prefix ends at the step the operator named, and that step must be one the source's
conversation in force holds: a step an earlier cut detached is not a branch point, because
carrying it would rebuild a prefix that holds a tail the source no longer owns. The cut's other
refusal — the prefix owing a tool result — is kept: a carried conversation that ends mid-exchange
would be refused by the server it is sent to. The cut's `ended` refusal is relaxed, because
branching from a finished conversation is the point of a fork, and the source's `ended` step is
not model-facing, so the fork's replay starts fresh from the full carried conversation.

The fork inherits the reach the source began with — the read and write roots the source's
`started` line records — because a fork is the same run on a different path, and a run that began
with a reach does not arrive at its fork without it. The operator can still narrow or widen the
fork with the fork's own flags; the inheritance is the default, not a grant.
"""
from __future__ import annotations

import pathlib

from . import record


def resolve_target(steps, value) -> tuple[str | None, str]:
    """The step a fork branches from, resolved and checked against the source's conversation
    in force — the pure heart of `sah fork --step <N>`, the answer to a refusal left to the
    caller: the door prints it and exits, nothing seeded.

    It walks the same resolution as a cut (`lineage`, the step's sequence, the detached-target
    stop) and keeps the cut's open-call refusal, because a carried prefix that ends mid-exchange
    is a conversation the server refuses. It relaxes the cut's `ended` refusal: a fork may
    branch from the source's ending, and the fork's own log ends the carried prefix with that
    `ended` step, which the fork's replay skips because it is not model-facing. Returns the
    target's step id and an empty error, or None and the refusal's words."""
    try:
        seq = int(value)
    except ValueError:
        return None, (f"{value!r} is not a step number (a step's id is <number>-<kind>, as logged)")
    try:
        valid = record.lineage(steps)
    except record.Refused as exc:
        return None, str(exc)
    prefix = f"{seq:06d}-"
    target = next((s for s in valid if s["id"].startswith(prefix)), None)
    if target is None:
        if any(s["id"].startswith(prefix) for s in steps):
            return None, (f"step {seq} is in a detached continuation of this run; a fork targets "
                          "a step of the conversation in force (the log holds the lineages)")
        return None, (f"step {seq} is not in this run's conversation in force (it ends at "
                      f"{valid[-1]['id']}); the number is a step's sequence, as logged")
    owed = record.open_calls(valid[:valid.index(target) + 1])
    if owed:
        named = ", ".join(f"{c['name']} ({c['id']})" for c in owed[:3])
        return None, (f"forking at {target['id']} would leave {named} owed a result — the carried "
                      "conversation would end mid-exchange. Target a step at a completed exchange")
    return target["id"], ""


def seed(source_lines, target_id, source_run_id, new_log_path) -> int:
    """Write the fork's log: the source's prefix ending at the named step, verbatim and with
    the source's step ids, followed by the `fork` record that names the source and the branch
    point. Returns how many source lines were carried.

    The prefix is filtered by sequence, so it carries the source's lines whole — its `started`
    head, its `cut` steps, any detached tail that stands before the branch point — and the fork's
    own replay applies the source's lineages over it, so a carried `cut` detaches the same tail
    here as it did in the source. The source log is never opened for writing: forking reads a
    prefix, it does not perform one. The `fork` record is minted by a `Log` built over the seeded
    prefix, so it takes the next sequence after the carried lines."""
    target_seq = int(target_id.split("-", 1)[0])
    prefix = []
    for line in source_lines:
        line = line.strip()
        if not line:
            continue
        try:
            step = record.loads(line)
        except record.Refused:
            continue                                    # a damaged line is not part of a clean prefix
        if int(step["id"].split("-", 1)[0]) <= target_seq:
            prefix.append(line)
    path = pathlib.Path(new_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in prefix:
            handle.write(line + "\n")
    record.Log(new_log_path).append("fork", "core",
                                    source_run=source_run_id, source_step=target_id)
    return len(prefix)


def inherit_grants(source_steps) -> tuple[list, list]:
    """The read and write roots a fork inherits from the source's newest `started` grants, or
    (empty, empty) for a source whose `started` line carries none. The newest `started` line is
    the source's last run, whose reach is the one the operator last put in force — the reach a
    fork of that run ought to begin with. A fork that inherits an empty reach is not a refusal:
    the fork's own flags say it, and the run path's standing note names a reach that changed."""
    started = next((step for step in reversed(source_steps) if step["kind"] == "started"), None)
    grants = (started or {}).get("grants", {})
    return list(grants.get("read", [])), list(grants.get("write", []))
