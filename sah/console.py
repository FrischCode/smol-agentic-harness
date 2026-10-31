"""sah.console — the operator's channel on a live session: the slash commands.

The conversation is the model's; the operator's own acts on the session — switching the
endpoint, setting the reasoning level, compacting now, cutting back to an earlier prompt —
are a different channel, and this is it. A line typed at the prompt that begins with `/`
belongs to the operator, not the model: it never goes out as an objective, and its outcome —
a switch, a cut, a compaction, a refusal — is recorded as a `notice` beside the log, so the
record shows the session's control the way it shows its conversation, and a refusal is a
word too: a command that silently does nothing is a command the operator believes ran.

The commands:

    /model [ID]   the endpoint the run talks to: a profile from data/models.json by id
                  (its endpoint fields ride with it; a field it leaves silent stays as the
                  run has it), or a literal model name on the endpoint the run is on. The
                  system prompt's <situation> rebinds to the new address, and a child the run
                  spawns goes to it too
    /think [LVL]  the reasoning level the next turn runs at (and every summary call, unless
                  the run pinned one for summaries). None shows the level in force
    /compact      the compaction, now: the same act the trigger runs when the window is full,
                  with the trigger bypassed — the operator's judgement that the window is
                  filling (a long task follows) is the trigger. The arithmetic is unchanged,
                  so a room that cannot fit says so and changes nothing
    /backto [N]   cut the conversation back before the operator's Nth prompt: the prompt and
                  everything after it detach (the log keeps them, the audit reads them), and
                  the operator types the replacement — the session's edit. None lists the
                  prompts in force, numbered
    /new          start over: the conversation in force detaches whole (the log keeps it) and the
                  operator types the fresh objective — /backto's first prompt, forward-framed
    /fork [N]     branch the conversation into a new run: carry it into a fresh run and diverge
                  under the fork's objective. With a number, the branch is the boundary before
                  your Nth prompt (the same boundary /backto cuts to); without one, the
                  conversation in force is carried whole. The source is untouched; the fork is a
                  new record, spawned as a child and named in the notice
    /export       the run's own report.md, now: the Markdown projection of everything the log
                  holds, marked a snapshot (the ending's own write replaces it with the run's
                  final report)
    /live         the session's vitals on demand: the window the trigger reads and its
                  composition, the last request's server numbers, the budget the next objective
                  runs against. A query — it prints, it records nothing

A cut here is the same act the `--resume <step>` door performs: a `cut` step the log keeps,
the tail out of every conversation the lineage rebuilds. The two doors speak different
numbers on purpose: the door numbers steps — the log's own digits, a machine's vocabulary —
and this one numbers prompts — the operator's third prompt, a human's — because the act it
names is an edit, and an edit reaches for the prompt it replaces, not the step it detaches.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from sah import compact, export, local, openai, record
from sah.model import REASONING_NONE
from sah.spawn import subprocess_spawn

if TYPE_CHECKING:
    from sah.agent import Agent
    from sah.spawn import Spawner


@dataclass
class Session:
    """What the commands reach, handed in at the boot: the run itself, the child spawner whose
    endpoint a switch rebinds (None when the run has no spawn), the rebind that rebuilds the
    system prompt for a new address, and the endpoint in force — the resolved boot values,
    which /model reads the same way the invocation's flags beat the machine's .env."""
    agent: "Agent"
    spawner: "Spawner | None" = None
    rebind: Callable[[str, str, int], str] | None = None
    endpoint: dict = field(default_factory=dict)


def _notice(session: Session, text: str) -> None:
    """The command's word to the operator: a notice beside the record, rendered where the run's
    notices render. Subject `command`, issuer `operator` — the operator's own act on the
    session, distinct from the run's mechanics (a `compact` notice, a `spawn` notice) it may
    sit beside."""
    session.agent.log.append("notice", "operator", subject="command", text=text)


def dispatch(line: str, session: Session, err) -> None:
    """One slash line, routed to its command. The caller's only remaining job is done: the line
    never reaches the model. `err` is where the input-aid lines print (a bare /backto's list,
    the profile table) — the transient channel a command's *outcomes* do not use, which go to
    the notice: the list is what the operator is choosing, not what happened to the run."""
    words = line.split()
    name = words[0]
    if name == "/model":
        _model(words[1:], session, err)
    elif name == "/think":
        _think(words[1:], session)
    elif name == "/compact":
        _compact(words[1:], session)
    elif name == "/backto":
        _backto(words[1:], session, err)
    elif name == "/new":
        _new(words[1:], session, err)
    elif name == "/fork":
        _fork(words[1:], session, err)
    elif name == "/export":
        _export(words[1:], session, err)
    elif name == "/live":
        _live(words[1:], session, err)
    else:
        _notice(session, f"{name}: not a command — /model [ID], /think [LVL], /compact, /backto [N], "
                         f"/new, /fork [N], /export, /live, /quit")


def _current_level(agent) -> str:
    """The run's reasoning level in the operator's spelling: `none` for the field omitted."""
    return REASONING_NONE if agent.thinking is None else agent.thinking


# --- /model -------------------------------------------------------------------


def _model(words, session, err):
    """Switch the endpoint the run talks to: a profile from data/models.json by id, or a
    literal model name on the endpoint in force. The chain a field resolves through is the
    switch's: the profile's value when the profile names the field, the endpoint in force
    when it does not — selecting a profile is the operator's act, and an act beats the
    default it replaces."""
    current = session.endpoint
    try:
        profiles = local.load_models()
    except record.Refused as exc:
        _notice(session, f"model: {exc}")
        return
    if not words:
        if not profiles:
            _notice(session, "model: no data/models.json in the tree root; /model <NAME> sets a "
                             "literal model name on the endpoint the run is on")
            return
        print("profiles (data/models.json):", file=err, flush=True)
        for profile_id, profile in profiles.items():
            print(f"  {profile_id:16} {profile['model']} @ "
                  f"{profile.get('host', current.get('host'))}:"
                  f"{profile.get('port', current.get('port'))}", file=err, flush=True)
        return
    if len(words) > 1:
        _notice(session, "model: an id is one word — /model flash, not /model flash now")
        return
    ref = words[0]
    profile = (profiles or {}).get(ref)
    if profile is None:
        # The record says so where the switch's word says it: a literal on the endpoint in
        # force, and the notice the switch records names the provenance.
        endpoint = dict(current)
        endpoint["model"] = ref
    else:
        endpoint = dict(current)
        endpoint.update(profile)
    _switch(endpoint, ref if profile is not None else None, session)


def _switch(endpoint, profile_ref, session):
    """The switch itself, from the resolved endpoint: the new backend, the rebound system
    prompt, the spawner's child endpoint, the record's word."""
    agent = session.agent
    current = session.endpoint
    model_name = endpoint["model"]
    host = endpoint.get("host") or "127.0.0.1"
    port = int(endpoint.get("port") or 8080)
    base_url = endpoint.get("base_url", "")
    # The posture a switch of model is not: the run's streaming and usage, its preserve-
    # thinking and reasoning-field pins, its timeout — carried from the backend in force,
    # asked the way `warn_cleartext` is (a scripted backend has none of them, and the
    # adapter's own defaults stand in for a backend that is not the real one).
    old = agent.backend
    # The window: the profile's when the profile names one, the run's in-force number when
    # the endpoint did not move (the number the run runs on, still true), a probe of the new
    # endpoint when it did — a switch that keeps the old endpoint's window number would run
    # the compaction arithmetic on a number the new endpoint did not make.
    context = endpoint.get("context")
    context_source = "named by the profile" if (profile_ref and context) else None
    moved = ((host, port, base_url) != (current.get("host"), current.get("port"),
                                         current.get("base_url", "")))
    if context is None and not moved:
        # The number the run runs on, still true of the endpoint it still talks to — or the
        # default, for a session that stands on no resolved window at all.
        context, context_source = (current.get("context") or 32768, "the run's in-force window")
    backend = openai.OpenAI(model=model_name, host=host, port=port, base_url=base_url,
                            api_key=endpoint.get("api_key", ""),
                            context_tokens=int(context) if context else 32768,
                            supports_vision=bool(endpoint.get("vision",
                                                              getattr(old, "supports_vision", False))),
                            api_path=getattr(old, "api_path", "/chat/completions"),
                            preserve_thinking=getattr(old, "preserve_thinking", "none"),
                            require_reasoning_content=getattr(
                                old, "require_reasoning_content", False),
                            reasoning_out=getattr(old, "reasoning_out", None),
                            streaming=getattr(old, "streaming", True),
                            request_usage=getattr(old, "request_usage", True),
                            timeout=getattr(old, "timeout", 600.0))
    if context is None and moved:
        # A probe of the new endpoint, the boot's own resolution minus the boot box.
        probed = openai.probe_context(backend.base_url, model_name, backend.api_key,
                                      api_path=backend.api_path)
        if probed is not None:
            backend.context_tokens = probed
            context, context_source = probed, "probed from the backend"
        else:
            context = 32768
            context_source = "the default — name a context on the profile"
    # The level: the profile's when the profile names one, the run's live level otherwise —
    # the level as the run has it now, not the boot's spelling (a /think since the boot is
    # the run's current fact).
    think = endpoint.get("think") or _current_level(agent)
    system = session.rebind(model_name, host, port) if session.rebind else agent.system
    agent.set_backend(backend, system)
    if session.spawner is not None:
        # A child the run spawns goes to the endpoint the run now talks to: the switch is
        # the run's, and a child that went to the old address would answer in a voice the
        # run no longer speaks.
        session.spawner.model = model_name
        session.spawner.host = host
        session.spawner.port = port
        session.spawner.api_key = endpoint.get("api_key", "")
        session.spawner.think = think
        session.spawner.context = int(context)
    session.endpoint = {**endpoint, "think": think, "context": context}
    provenance = (f"profile {profile_ref!r} from data/models.json"
                  if profile_ref else "a literal model name on the endpoint in force")
    _notice(session, f"model: {model_name} @ {host}:{port} in force ({provenance}; was "
                     f"{current.get('model')} @ {current.get('host')}:{current.get('port')}); "
                     f"context {context} ({context_source}); think={think}")


# --- /think -------------------------------------------------------------------


def _think(words, session):
    """The reasoning level the next turn runs at. `none` omits the field (the server's own
    template decides); any other word goes out verbatim, the backend's own vocabulary —
    so the level is not checked against a table: a table here would be the harness deciding
    what a server it does not know speaks."""
    agent = session.agent
    current = _current_level(agent)
    if not words:
        _notice(session, f"think: {current}"
                 + (f" (the summary calls keep their own level: {agent.summary_thinking})"
                    if agent.summary_thinking else ""))
        return
    if len(words) > 1:
        _notice(session, "think: a level is one word — /think xhigh, not /think x high")
        return
    level = words[0]
    if level == current:
        _notice(session, f"think: {current} (in force already)")
        return
    agent.thinking = None if level == REASONING_NONE else level
    if session.spawner is not None:
        # A child runs at the level the run now runs at: the level is the run's fact, and
        # a child that reasoned at the old one would spend a budget the run no longer set.
        session.spawner.think = level
    session.endpoint["think"] = level
    _notice(session, f"think: {level} (was {current})"
             + (f" — the summary calls keep their own level ({agent.summary_thinking})"
                if agent.summary_thinking else ""))


# --- /compact -----------------------------------------------------------------


def _compact(words, session):
    """The compaction, now: the run's own `compact()` act, with the occupancy trigger
    bypassed — the operator's judgement that the window is filling is the trigger, and the
    arithmetic runs unchanged, so a room that cannot fit says so (the ContextBlocked the
    act raises) and changes nothing: the conversation stands as it was, and the record says
    why the act did not run."""
    agent = session.agent
    if words:
        _notice(session, "compact: takes no argument — /compact")
        return
    try:
        step = agent.compact()
    except compact.ContextBlocked as blocked:
        _notice(session, f"compact: {blocked.text} — nothing was changed; the run is "
                         f"unaffected")
        return
    if not step:
        _notice(session, f"compact: nothing to drop yet — the conversation is "
                         f"{len(agent.messages)} message(s) and the most recent "
                         f"{compact.KEEP_RECENT} always survive")
        return
    _notice(session, f"compact: ran — {step.get('dropped')} turn(s) summarised, slot "
                     f"{step.get('slot')} (stage {step.get('stage')}), {step.get('tokens_cost', 0)} "
                     f"token(s) the summary calls cost")


# --- /backto ------------------------------------------------------------------


def _valid(steps) -> tuple[list, str]:
    """The conversation in force (the lineage), or the refusal it carries: a cut whose
    target the log no longer holds is a corruption the projection cannot answer honestly,
    and the command's answer to it is the named refusal, not a traceback."""
    try:
        return record.lineage(steps), ""
    except record.Refused as exc:
        return [], str(exc)


def _operator_prompts(valid) -> list:
    """The operator's prompts in the conversation in force, in order: the user-role message
    steps that are not the core's own injections. A queued steer is one — the operator's
    words, from wherever the operator was — and a core injection (a wrap-up, a continue
    prompt) is not: the operator edits their own prompts, not the harness's."""
    return [s for s in valid if s["kind"] == "message" and s.get("role") == "user"
            and s.get("issuer") != "core"]


def _backto_target(steps, n) -> tuple[str, int, str]:
    """The step a /backto <n> cuts to: the conversation's last step before the operator's Nth
    prompt — the boundary the replacement prompt takes the place of.

    The boundary is a step, derived from a prompt number: the prompt's own step is refused
    as a cut target (the prefix would hold the prompt being replaced, and its exchange),
    and the step before it is the previous exchange's end — or the run's `started` step when
    the prompt is the first. The previous objective's `ended` marker sits between its last
    step and the next prompt, and it is the old timeline's bookkeeping — the marker of an
    objective the operator is about to edit is not part of the conversation the replacement
    continues — so a boundary landing on it steps back one more: the marker detaches with
    the tail it belongs to. The checks are the cut's standing checks — no ending, no call
    owed a result — because a cut that ends a conversation mid-exchange is one the server
    it is sent to would refuse, and the refusal here is cheaper than the one there."""
    valid, error = _valid(steps)
    if error:
        return "", 0, error
    prompts = _operator_prompts(valid)
    if not 1 <= n <= len(prompts):
        return "", 0, (f"prompt {n}: the conversation in force holds {len(prompts)} operator "
                       f"prompt(s) — /backto with no number lists them")
    index = valid.index(prompts[n - 1])
    boundary = index - 1
    if boundary < 0:
        return "", 0, "the prompt is the conversation's first step; there is nothing before it"
    target = valid[boundary]
    if target["kind"] == "ended":
        boundary -= 1
        if boundary < 0:
            return "", 0, "the boundary holds only the run's ending; there is no conversation " \
                          "to cut back to"
        target = valid[boundary]
    if target["kind"] == "ended":
        return "", 0, f"the boundary before prompt {n} is another ending; target a prompt later " \
                      "in the conversation"
    owed = record.open_calls(valid[:boundary + 1])
    if owed:
        named = ", ".join(f"{c['name']} ({c['id']})" for c in owed[:3])
        return "", 0, f"cutting before prompt {n} would leave {named} owed a result — the " \
                      "conversation would end mid-exchange"
    detached = len(steps) - steps.index(target) - 1
    return target["id"], detached, ""


def _list_prompts(steps, err):
    """The operator's prompts in force, numbered: what a bare /backto lists so a number can
    be chosen. Printed directly, not noticed: the list is what the operator is choosing
    (the transient channel the question announcements use), not what happened to the run —
    the act, when it happens, is the notice. A lineage the log cannot project is a line on
    the same channel: the aid's own word for why it cannot list."""
    valid, error = _valid(steps)
    if error:
        print(f"backto: the log's conversation cannot be read — {error}", file=err, flush=True)
        return
    prompts = _operator_prompts(valid)
    if not prompts:
        print("backto: no operator prompts in this conversation yet", file=err, flush=True)
        return
    print("backto: your prompts in force — /backto <N> cuts back before it; type the replacement",
          file=err, flush=True)
    for number, step in enumerate(prompts, start=1):
        text = step.get("text", "").replace("\n", " ")
        if len(text) > 60:
            text = text[:57] + "..."
        print(f"  {number:3}  {step['id']}  {text!r}", file=err, flush=True)


def _cut(session, target: str, detached: int) -> int:
    """The act both /backto and /new perform: the `cut` step the log keeps (the operator's
    decision, the core's performance — the same step the resume door writes), and the
    conversation rebuilt from the prefix. Returns the message count back in force."""
    agent = session.agent
    agent.log.append("cut", "core", to=target)
    return agent.replay()


def _backto(words, session, err):
    """Cut the conversation back before the operator's Nth prompt: the session's edit. The
    cut is recorded (the same `cut` step the resume door writes — the operator's decision,
    the core's performance), the conversation rebuilt from the prefix, and the operator is
    back at the prompt to type the replacement."""
    agent = session.agent
    steps, damage = agent.log.read()
    if damage:
        print(f"log damage: {damage}", file=err, flush=True)
    if not words:
        _list_prompts(steps, err)
        return
    if len(words) > 1 or not words[0].isdigit():
        _notice(session, f"backto: {words[0] if words else ''} is not a prompt number — "
                         "/backto <N>, or /backto with no number for the list")
        return
    target, detached, error = _backto_target(steps, int(words[0]))
    if error:
        _notice(session, f"backto: {error}")
        return
    restored = _cut(session, target, detached)
    _notice(session, f"backto: prompt {words[0]} and {detached} step(s) after it detached "
                     f"(cut to {target}); {restored} message(s) back in force — type the "
                     "replacement")


# --- /new ---------------------------------------------------------------------


def _new(words, session, err):
    """Start over: the conversation in force detaches whole and the operator types the fresh
    objective. It is /backto at its first prompt — the same cut, the same record — with the
    frame the operator meant: discard forward, not reach back for a prompt to replace. A
    conversation with no operator prompt in it has nothing to detach, and that is a word, not
    a cut."""
    agent = session.agent
    if words:
        _notice(session, "new: takes no argument — /new")
        return
    steps, damage = agent.log.read()
    if damage:
        print(f"log damage: {damage}", file=err, flush=True)
    valid, error = _valid(steps)
    if error:
        _notice(session, f"new: {error}")
        return
    if not _operator_prompts(valid):
        _notice(session, "new: the conversation holds no operator prompt yet — type the objective")
        return
    target, detached, error = _backto_target(steps, 1)
    if error:
        _notice(session, f"new: {error}")
        return
    restored = _cut(session, target, detached)
    _notice(session, f"new: the conversation detached ({detached} step(s), cut to {target}); "
                     f"{restored} message(s) back in force — type the fresh objective")


# --- /fork --------------------------------------------------------------------


def _fork(words, session, err):
    """Branch the conversation into a new run: carry the conversation in force into a fresh
    run and diverge under the fork's objective. With a number, the branch is the boundary before
    the operator's Nth prompt (the same boundary /backto cuts to); without one, the conversation
    in force is carried whole, up to its last step. The source is untouched — forking reads a
    prefix, it does not perform one — and the fork is a new record, spawned as a child and named
    in the notice so the operator can read its answer with `sah result`.

    The fork's argv is built from the spawner's configuration, the way a child's argv is built:
    the endpoint, the budget, the grants — the reach the source began with — and never the open
    axis the spawner holds back. The fork inherits the source's grants on its own, so the argv's
    grants are the standing reach, not a re-grant: a fork that arrived at its parent's reach
    without the operator typing it would be a widening the record could not name."""
    agent = session.agent
    spawner = session.spawner
    if not spawner:
        _notice(session, "fork: this run cannot spawn; the fork cannot be launched from the "
                         "session — run `sah fork` from a shell")
        return
    if not words:
        _notice(session, "fork: /fork [N] \"objective\" — branch at your Nth prompt (like "
                         "/backto) and run the objective in a fork; no number carries the "
                         "conversation whole")
        return
    # [N] "objective": a leading number is the branch point (a prompt number, the way /backto
    # reads one); otherwise the whole line is the objective and the branch is the current point.
    if words[0].isdigit():
        n, objective = int(words[0]), " ".join(words[1:])
    else:
        n, objective = None, " ".join(words)
    objective = objective.strip()
    if not objective:
        _notice(session, "fork: the fork needs an objective — /fork [N] \"objective\"")
        return
    steps, damage = agent.log.read()
    if damage:
        print(f"log damage: {damage}", file=err, flush=True)
    if n is not None:
        # the same boundary /backto cuts to: the step before the operator's Nth prompt, checked
        # by the cut's standing checks (no ending, no call owed a result)
        target, _detached, error = _backto_target(steps, n)
        if error:
            _notice(session, f"fork: {error}")
            return
    else:
        # no number: carry the conversation in force whole, up to its last step
        valid, error = _valid(steps)
        if error:
            _notice(session, f"fork: {error}")
            return
        if not valid:
            _notice(session, "fork: the conversation is empty; there is nothing to fork")
            return
        target = valid[-1]["id"]
    branch_seq = target.split("-", 1)[0]
    # the argv: a sah fork, built from the spawner's configuration the way a child's argv is
    source = str(agent.log.path)
    fork_id = spawner.next_id(agent.run_id)
    new_dir = spawner.child_dir(fork_id)
    argv = [sys.executable, "-m", "sah", "fork", source, "--step", branch_seq,
            "--run-dir", new_dir,
            "--model", spawner.model, "--host", spawner.host, "--port", str(spawner.port),
            "--context", str(spawner.context), "--think", spawner.think,
            "--max-turns", str(spawner.max_turns),
            "--grace-turns", str(spawner.grace_turns),
            "--view", "off", "--no-spawn"]
    for grant in spawner.read_roots:
        argv += ["--grant", grant]
    for grant in spawner.write_roots:
        argv += ["--write", grant]
    if spawner.allow_exec:
        argv += ["--exec"]
    argv += ["--task", objective]
    subprocess_spawn(argv, new_dir, api_key=spawner.api_key)
    _notice(session, f"fork: branched at {target} into {fork_id} (objective: {objective!r}); "
                     "the source is untouched — its answer is `sah result --run-dir "
                     f"{os.path.basename(new_dir)}`")


# --- /export ------------------------------------------------------------------


def _export(words, session, err):
    """The run's own report.md, now: the Markdown projection of everything the log holds, written
    to the run's own folder and marked a snapshot in its header (the ending's own write replaces
    it with the run's final report). No path argument: the run writes where the run writes, and a
    command that could write anywhere would be a hole in the confinement story the run's grants
    are the edge of."""
    agent = session.agent
    if words:
        _notice(session, "export: takes no argument — /export writes the run's own report.md")
        return
    steps, damage = agent.log.read()
    if damage:
        print(f"log damage: {damage}", file=err, flush=True)
    if not steps:
        _notice(session, "export: the log holds no steps yet — run an objective first")
        return
    if not export.write_report(agent.root, err, snapshot=True):
        _notice(session, "export: the report could not be written — nothing was projected")
        return
    _notice(session, f"export: report.md written ({len(steps)} steps, snapshot — the log is the record)")


# --- /live --------------------------------------------------------------------


def _live(words, session, err):
    """The session's vitals on demand: the window the trigger reads (the arithmetic, not a guess),
    its composition, the last request's server numbers, the budget the next objective runs
    against. A query — it prints on the transient line and records nothing: the run's own notices
    record the acts, and a reading of the window is not an act."""
    agent = session.agent
    if words:
        _notice(session, "live: takes no argument — /live")
        return
    window = agent.backend.capabilities().context_tokens
    effective = agent.context.effective(agent.messages)
    if window <= 0:
        # A backend with no window stated has no answer to "how full is it", and the trigger
        # cannot fire on one either: the reading says so instead of dividing by nothing.
        print("live: no window in force (the backend declared none) — the trigger cannot fire; "
              "name a context on the endpoint", file=err, flush=True)
    else:
        share = compact.occupancy(effective, window)
        print(f"live: ~{effective} of {window} window tokens ({share:.1%}) — the trigger's "
              f"arithmetic (it fires at {compact.TRIGGER:.0%})", file=err, flush=True)
        calibrated = (f"; calibrated ×{agent.context.ratio:.2f} from the server's own counts"
                      if agent.context.ratio else "; estimates uncorrected — the server has "
                      "reported no counts")
        print(f"     {agent.context.decomposition(agent.messages)}{calibrated}", file=err, flush=True)
    usage = agent.last_usage
    if usage:
        last = (f"last request: {usage.get('tokens_in', 0)} tokens in, "
                f"{usage.get('tokens_out', 0)} out (server-reported)")
    else:
        last = "no request yet — the window reading is the estimate, bare"
    print(f"     {last} · next objective: {agent.max_turns} turns, {agent.grace_turns} grace",
          file=err, flush=True)
