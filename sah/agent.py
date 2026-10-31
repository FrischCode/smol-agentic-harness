"""sah.agent — the loop. Commands in, events out, one append path.

Three things happen here and nothing else does: ask the backend for a turn, run the tools it
asked for through the one veto point, and append what happened. Everything a caller can impose
on a running loop arrives as a command, is validated, is recorded, and only then takes effect —
so the log shows who asked and whether it was accepted, and no path changes the run that does
not appear in it.

The append path is the invariant. The `Log` never leaves this module: a tool returns a value or
yields a stream of chunks but writes nothing, a plugin subscribes to steps, and neither can append
a fact. A refusal, an undeclared
action and a failed backend call are all recorded too — recording that something was attempted
is not the same as granting it, which is the distinction the whole design leans on.

`interrupt` here takes effect at the turn boundary. Killing an in-flight tool call is a
different mechanism (the queue plus `process.kill_tree`) and is deliberately not faked here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Generator, Iterator
from dataclasses import dataclass

from sah import compact, control, ingest, model, prompts
from sah.guard import RepetitionGuard
from sah.record import Log, lineage
from sah.contract import Media, Verdict, arguments_for, declaration, picture_kind, spec_for

# The actions the loop understands. Anything else is recorded and refused by name, because a
# silently ignored command is worse than a refused one. `answer` is the reply to a `request` the
# core put out; it is matched where the question was asked, not at the turn boundary.
ACTIONS = ("steer", "interrupt", "finish", "answer")

# How often a wait for an answer looks at the queue. A poll, not a wake-up: the queue is a file and
# nothing in this tree owns a thread.
POLL_SECONDS = 0.1

# What counts as a yes. Deliberately short: an answer that has to be interpreted was not an answer.
YES_WORDS = frozenset({"yes", "y", "allow", "grant", "accept", "approved", "true", "ok"})

# A no-call turn is told what came back and given the exits that leave it, up to three times.
# The exits differ by attendance: an attended run may re-emit the call or say done in words, and
# an unattended run may re-emit the call or make the declaration, because no one is there to read
# prose as completion. A run still short of both after three tellings ends failed (attended) or
# blocked (unattended) rather than spinning against a backend that has nothing to say.
NUDGE_LIMIT = 3

# What the repetition guard puts to the model when it catches the run doing the identical act over
# and over. The words live in `prompts/repeat_steer.md` so they are editable; they name the behaviour,
# not a judgment of the model, and offer the two real exits: change approach, or say you are stuck.
REPEAT_STEER = prompts.load("repeat_steer")

# The word for a turn that produced nothing at all. The model never sees its own empty turn enter
# the conversation, so the words report what the harness observed (an empty reply) rather than
# pointing at a turn the model cannot see, and then offer the two real exits: send the needed
# call fresh and in full, or declare completion in a line, since a prose-only turn ends the run.
EMPTY_TURN = prompts.load("empty_turn")

# The structural nudge for a run no operator is watching: a turn with no call — prose or none —
# is asked for the call or the declaration. The words sit in the user's seat the way every core
# word does. It is rendered per turn at the call site, not loaded here: the sentence carries a
# slot the turn's own wire facts fill, and a constant loaded at import would render it with the
# slot left open.

# The completion declaration's name. The loop checks it, the tool is registered under it (cli),
# and the situation prose tells the model about it: one name, read by all three, so the contract
# the model is told and the act the loop honours cannot drift apart.
VERIFY_OBJECTIVE = "verify_objective"

# A turn the output limit ended is not a finished answer, so the loop asks for the rest -- a bounded
# number of times. A model that keeps hitting the limit keeps hitting it, and an unbounded recovery is a
# run that spends its turns forever without ever saying why it stopped. The three prompts answer the three
# shapes a cut-off turn takes (prose, a half-built call, nothing at all); they live in prompts/ because
# they are model-facing instruction, as `repeat_steer.md` is.
CONTINUE_LIMIT = 2
CONTINUE_PROMPTS = {"prose": prompts.load("continue_prose"),
                    "call": prompts.load("continue_call"),
                    "empty": prompts.load("continue_empty")}

# A budget that runs out (turns or time) is not an end: the model is told to wrap up, and the
# grace turns that follow are the leeway to deliver what it has. A run that answers in them ends
# complete, which is the property a bounded run exists to have: a bounded child ends with an
# answer, not with a wall. The words live in `prompts/` because they are model-facing
# instruction, as `repeat_steer` is; the notice that goes beside them names the budget that ran
# out, and that is a record, not an instruction.
WRAP_UP = prompts.load("wrap_up")

# What the first refusal of something askable tells the model about repeating it. Asking is the last
# resort, so a first refusal asks the operator nothing; the sentence says what a repeat costs, and that
# differs depending on whether anyone is there to answer.
ASK_REPEAT = {"attended": prompts.load("ask_repeat_attended"),
              "unattended": prompts.load("ask_repeat_unattended")}


def allow_all(name: str, arguments: dict, *, access=None, address=None) -> Verdict:
    """The default policy. A harness with no policy still appends an `operation` step, so the
    veto point is one call site whether or not anything is confined. It supplies no address, so
    the argument the model typed is what gets recorded."""
    return Verdict(True)


@dataclass
class Command:
    """A command in the queue: its action and arguments, who issued it, its queue identity, and whether
    the block is closed (a cut-off block is refused, not acted on)."""
    action: str
    args: dict
    issuer: str
    queue_id: str = ""
    closed: bool = True


def _with_picture(content: str, part: dict | None):
    """A tool message's content: the text, and beside it the picture when there is one to send.

    Attaching to *this* message is the design: the answer that loaded the file is the one that carries it.
    A server that would rather the picture arrived on a turn the harness authored is a fact about that
    server (measured against one such endpoint, which answers a picture in a tool result with "image_url
    requires a user message"), and it belongs in a live check against `--vision` rather than in a silent
    rewrite of who said what."""
    return content if part is None else [{"type": "text", "text": content}, part]


# --- the render input, serialized ----------------------------------------------
#
# The request a turn asks of the backend is the render input: system, messages, tools, thinking,
# max_tokens. The conversation half of it is the log's stream, recorded step by step; the fixed
# head half is stamped once at run start (`_head_fields`); and every turn carries the fingerprint
# of the whole, so the record says what was sent without copying the stream into itself.


def _spec_wire(spec) -> dict:
    """A tool schema in the logical form the record stores: name, contract, parameters.

    The harness does not rewrite schemas, so this is the schema as derived — the shape the
    `started` step stores as the run's fixed head and the fingerprint hashes. It is the logical
    schema, not the adapter's HTTP form: the wire wraps it in a `function` envelope, and the
    record stores what the harness had, not what the framing added around it."""
    return {"name": spec.name, "description": spec.description, "parameters": spec.parameters}


def _msg_wire(message) -> dict:
    """A message as the record serializes it: every field the wire carries, in a stable shape.

    This is the record's serialization, not the adapter's HTTP form. It exists so a request's
    fingerprint is a function of what the harness had — role, content (an image part by its wire
    bytes, since that is what travels), calls, reasoning — and nothing else: a request rebuilt
    from the record hashes to the same value as one that was sent, while any changed input does
    not."""
    return {"role": message.role, "content": message.content,
            "tool_calls": [list(call) for call in (message.tool_calls or ())],
            "tool_call_id": message.tool_call_id, "reasoning": message.reasoning}


def _request_digest(system, messages, specs, thinking, max_tokens) -> tuple:
    """The fingerprint of the logical request: a digest and a size of its canonical serialization.

    The five render inputs, exactly as handed to the backend, canonicalized and hashed. The digest
    pins what the harness sent: a request re-derived from the record (the fixed head from
    `started`, the stream from the steps) hashes the same while the inputs are unchanged, and
    stops matching the moment one of them moves — a reworded prompt, an edited schema, a changed
    image — which is information, not a failure. It hashes the canonical serialization, not the
    HTTP bytes: the adapter's framing is below the question of what the model was asked."""
    canonical = json.dumps({"system": system,
                            "messages": [_msg_wire(m) for m in messages],
                            "tools": [_spec_wire(s) for s in specs],
                            "thinking": thinking, "max_tokens": max_tokens},
                           sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], len(canonical)


def _prefix_digest(system, specs) -> str:
    """The identity of the fixed head alone — system and tools, with no conversation.

    Stable across a run (the head does not change) and comparable across runs: two runs that
    shared a prompt and a tool set hash to the same value, which is the address for the question
    "did this run ask the same model the same standing question?" The per-turn `req_sha` cannot
    answer it, because it also folds in the conversation."""
    canonical = json.dumps({"system": system, "tools": [_spec_wire(s) for s in specs]},
                           sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class Agent:
    """One run of the loop against one backend."""

    def __init__(self, *, backend: model.Backend, tools: dict, log: Log,
                 policy=allow_all, system: str = "", thinking: str = "none",
                 summary_thinking: str | None = None,
                 max_tokens: int | None = None, max_turns: int = 12, max_runtime: int = 0,
                 grace_turns: int = 3, streaming: bool = True, usage: bool = True,
                 require_reasoning_content: bool = False, reasoning_out: str | None = None,
                 clock=time.monotonic, ask_timeout: float = 0.0,
                 attended: bool = False,
                 run_id: str = "run", root: str = ".", cwd: str | None = None,
                 project_notes_file: str | None = None,
                 queue=None, events=None, guard=True):
        """Set up a run against one backend: the tools and log, the policy and system prose, and
        the limits and flags (max turns, max runtime, grace turns, ask timeout, attendance,
        compaction, guard) that bound it."""
        self.backend = backend
        self.tools = tools
        # What each tool said it touches, read once here rather than per call. The policy dispatches on
        # this instead of on the tool's name, so a tool added tomorrow is judged by the line written next
        # to it; `ext.assemble` has already refused the run if any tool said nothing.
        self.declared = {name: declaration(func) for name, func in tools.items()}
        self.log = log
        self.policy = policy
        self.system = system
        self.thinking = thinking
        # The reasoning level the summary calls run at, when the operator named one: None keeps
        # the run's own level, left alone on every summary attempt (the default — a summary call
        # with no output cap has nothing for reasoning to spend, so the level is not overridden,
        # it is passed through), and a named level applies to all of them (see `sah/compact.py`).
        self.summary_thinking = summary_thinking or None
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.max_runtime = int(max_runtime)
        """Seconds this objective may take before the model is told to wrap up; 0 is no limit.
        The clock it counts against starts when the objective starts (see `_begin`), so a resumed
        run is judged on the time it takes now, not on the conversation it inherits."""
        self.grace_turns = int(grace_turns)
        self.streaming = bool(streaming)
        self.usage = bool(usage)
        # The reasoning wire shape the operator declared for this server, as `streaming` is
        # declared: the backend enforces it on the payload, and `started` stamps it, so a reader of
        # the wire trail can tell why a request wrote the reasoning somewhere the conversation did
        # not come in on, or carried an empty key a turn never filled.
        self.require_reasoning_content = bool(require_reasoning_content)
        self.reasoning_out = reasoning_out
        """The transport's wire knobs, as the operator set them for this run. They are stamped on
        `started` (the log field is the record of what the run talked to the server under) and are
        not read by the loop: the backend owns the wire, the record owns the witness."""
        """Turns the model still gets after a budget runs out, told to wrap up and deliver in
        them. Zero is the hard stop: the budget ends the run at the boundary with no word to the
        model."""
        self._clock = clock
        """The clock the time budget reads. A stand-in seam: a caller may pass a clock that moves
        on demand, so a run of hours is driven by a value, not waited out on the wall."""
        self.ask_timeout = float(ask_timeout)
        """Seconds a question waits for an answer. Zero is the honest default for a harness that
        may be running with nobody watching: the answer to a question nobody hears is no."""
        self.attended = bool(attended)
        """Whether anyone is there to answer. A run is attended when a person sits at a terminal (or has
        passed --ask-timeout, which is a promise to answer); `ask_timeout` on its own is not enough, because
        a spawned child can be given a timeout and still have nobody to hear it. False is the honest
        default: a run that believes it is watched asks questions that only waste the turn."""
        self.run_id = run_id
        self.root = root
        # The working directory the operator launched this run from. The run's own folder says
        # nothing about the workspace the run stood in, and the log is the only place a later
        # reader learns it — so it is stamped on `started`, next to `root`. Captured here, at
        # construction: an in-process child then stands where its parent stood, and a caller may
        # name it instead of inheriting the process's own directory.
        self.cwd = cwd if cwd is not None else os.getcwd()
        # The working directory's instruction file that was framed into `system` at boot, or None. The
        # content is already in the system prose (and stamped there); this is the derivation input's
        # identity -- the audit answer to "what external file entered the prompt" -- recorded as a field
        # a reader can grep rather than parse out of the prose. Present only when the operator granted
        # --project-notes and a file was actually loaded.
        self.project_notes_file = project_notes_file
        self.queue = root if queue is None else queue
        # The event bus: the one point every live chunk is handed to, to fan out to all display
        # consumers. Live only, and never a record — what a client sees streaming and what the log
        # holds are the same facts, but only one is kept, and nothing may be decided from this: it
        # runs before a turn is appended, and a subscriber that raises must not stop the run.
        self.event_subscribers: list = [events] if events is not None else []
        # The repetition guard is a core safety, always on unless a caller deliberately swaps it
        # (pass an instance to tune it, or None to disable it entirely). It only
        # decides; every write it triggers is performed here in the loop, never by the guard.
        self.guard = RepetitionGuard() if guard is True else guard
        self.specs = [spec_for(func) for func in tools.values()]
        # What the run's next request carries, known before the request is sent: the exact system
        # prompt and tool schemas the wire will carry, counted once. The conversation beside them is
        # counted per request (see `compact.Context`) — the run owns the size of its own window, and
        # the backend's number, when it reports one, is the correction, not the source.
        self.context = compact.Context(system, self.specs)
        self.messages: list[model.Message] = []
        self._trace: list[str] = []
        """The log-step id that produced each entry of `messages`, same index. Compaction needs it:
        a `summary` step points at the messages it replaced, and that link only exists if the loop
        kept it. A parallel list, not a field on `Message`, because Message is the wire type and the
        wire has no notion of a log."""
        self.last_usage: dict = {}
        self.compact_enabled = True
        self._last_summary: str | None = None
        """The slot's text from the last compaction of this conversation — or None before the
        first: the next summary merges it into its source (each summary re-summarises the last
        slot), and a resumed conversation re-reads it from the log's summary step. Conversation
        state, not objective state: the slot outlives any single objective."""
        self.pending: list[Command] = []
        self.nudges = 0
        self.stop_requested = ""
        self.last_calls: list[model.ToolCall] = []
        self.last_text = ""
        self.last_error = ""
        self.last_finish = ""           # the server's finish word for the last turn, if one arrived
        self.final_text = ""
        self.last_truncated = False
        self.truncated_kind = ""        # prose | call | empty, for the turn the limit ended
        self._carrying = False          # the next prose continues an answer the limit cut off
        self._truncations = 0           # recoveries spent for this objective
        self._overflow = 0              # room made after a refusal for size, this objective

    # --- coming back ------------------------------------------------------------

    def replay(self) -> int:
        """Rebuild the conversation from the log, and return how many messages came back.

        The unit is the message list, not the turns the model took: one turn can rebuild three
        entries and a rebuilt conversation can hold no assistant turn at all. The count reaches the
        operator in the resume line, so the name here is the name they read.

        Resume means continue, not restart: the message list is reconstructed from what the log
        holds so a run can pick up where it stopped, and pending queue blocks are drained by `run`
        as they always are. When the log carries a `cut` step, `lineage` applies it before the
        walk: the conversation is the prefix in force, and what the operator detached — and the
        act that detached it — stay out of it, so a resume at a step never re-reads the tail it
        cut. Only the model-facing kinds rebuild context — a `summary` stands in
        for everything before it, beside the verbatim backfill it kept standing in (its steps
        stand before the summary in the log, so the walk's tail at that point is the backfill, and
        dropping it would lose the turns the live run still carried) — and a `command`/
        `operation`/`decision` is audit that the model never saw, so replaying it would hand back
        authority the log should not carry.

        It rebuilds *conversation*, never *permission*: the grants in force are the ones this
        invocation was given. Widening is not something a log does to a run.

        A torn final line is tolerated and recorded, not fatal: the run that died mid-append should
        still be continuable, and the step it lost is named in a `notice` so the gap is visible to
        whoever reads this log next.

        The rule is parity, not plausibility: the rebuilt list has to be the list `step()` would have
        held, because a resumed run sends it to the same chat template and the same prefix cache, and a
        near-miss shape is accepted by a lenient server while quietly costing the cache and, for a blank
        assistant turn, inviting the hallucination the loop skips. So one turn is one assistant message
        carrying all of that turn's calls; a turn that produced neither prose nor a call contributes
        nothing; and an entry's trace id is whichever step earns it live — the turn's `message` step for
        an assistant entry, the `call` step for a tool result. A call whose turn header never arrived (a
        log cut mid-turn) is held as a turn of its own rather than dropped, so its result still has
        something to answer."""
        self.messages = []
        self._trace = []
        steps, damage = self.log.read()
        if damage:
            self.log.append("notice", "core", subject="replay", text=damage)
        steps = lineage(steps)
        call_ids: dict[str, str] = {}       # call-step id -> backend tool-call id, so a result
        #                                       links to the call it answers across a reload
        # Live appends ONE assistant message carrying every call of the turn, then one tool message per
        # result in the order the tools ran. The log cannot record it that way: each call is appended,
        # run, and its result appended before the next call is even made, so the steps arrive
        # interleaved (call, result, call, result). The calls of a turn are therefore collected first,
        # keyed by the `message` step they name in `refs`, so the assistant entry can be rebuilt whole.
        calls_by_turn: dict[str, list[tuple]] = {}
        first_index: dict[str, int] = {}      # turn key -> where its first call appears, for summaries
        turn_of_call: dict[str, str] = {}     # call-step id -> the turn it belongs to
        for index, step in enumerate(steps):
            if step["kind"] != "call":
                continue
            cid = step.get("call_id") or step["id"]
            key = step.get("refs") or step["id"]        # a call with no header is its own turn
            call_ids[step["id"]] = cid
            turn_of_call[step["id"]] = key
            calls_by_turn.setdefault(key, []).append((cid, step["name"], step.get("args", {})))
            first_index.setdefault(key, index)

        def emit_turn(turn_id: str, text: str, calls: list, reasoning: str = "",
                      reasoning_field: str = "reasoning_content") -> None:
            """One assistant entry, exactly as `step()` would have added it: carrying its calls when it
            has them, as text alone when it has text, and not at all when it has neither — the blank
            turn the live loop skips must stay skipped across a resume, or the hazard it was dropped
            for comes back on the first resumed turn. `reasoning_field` is the turn's signature, re-read
            from the log so a resume re-sends in the same field the server delivered it in."""
            if calls:
                self.messages.append(model.Message("assistant", text, tool_calls=tuple(calls),
                                                   reasoning=reasoning or None,
                                                   reasoning_field=reasoning_field))
            elif text:
                self.messages.append(model.Message("assistant", text, reasoning=reasoning or None,
                                                   reasoning_field=reasoning_field))
            else:
                return
            self._trace.append(turn_id)

        emitted: set[str] = set()             # turns already put in, so no turn is emitted twice
        for index, step in enumerate(steps):
            kind = step["kind"]
            if kind == "summary":
                # A summary replaces everything before it, mirroring what compact() did live.
                # The step's `stage` marks the layout where the slot is always filled, so both
                # verdicts rewrote the conversation and the frame carries the backfill note; a
                # pre-port frame replayed with a note it never carried would not be the list the
                # original run sent, and a pre-port `failed` step replaced nothing (a failed
                # ladder left the turns standing), so the turns it lists are replayed whole
                # behind it. What stood in front of the slot is the record's own arithmetic —
                # the walk has accumulated everything the live list held before the rewrite, and
                # the step says what it dropped and what it kept — so no content is read from
                # any of it: the remainder is the entry the live loop stood in front of the slot
                # (the opening message in the layout that kept a head, nothing in the layout that
                # summarises it), and a pre-port step, which replaced everything, leaves no
                # remainder to keep.
                if step.get("stage") is not None or step["outcome"] == "ok":
                    ported = step.get("stage") is not None
                    covered = step.get("slot") in ("summary", "cut")
                    kept = step.get("kept", 0) if ported else 0
                    # The tail the cut rung dropped when the backfill itself exceeded the limit:
                    # it stands in the walk between what was dropped and what was kept, so the
                    # remainder the slot stands in front of is the walk minus all three — and a
                    # log the rung never wrote carries no field, which is zero, which is the
                    # arithmetic before the rung existed.
                    cut_tail = step.get("backfill_cut", 0) if ported else 0
                    head = (max(0, len(self.messages) - step["dropped"] - cut_tail - kept)
                            if ported else 0)
                    # The backfill's steps stand before this one in the log (the walk has already
                    # accumulated them), and the live loop keeps them after the rewrite, so the
                    # tail the walk built is the backfill: a resumed run that dropped it would
                    # lose the verbatim turns the live run still carried beside the notes. The
                    # `if kept` keeps the pre-port shape (notes only, no backfill) clear of the
                    # `-0:` slice, which is the whole list.
                    head_msgs, head_trace = self.messages[:head], self._trace[:head]
                    tail = self.messages[-kept:] if kept else []
                    trace_tail = self._trace[-kept:] if kept else []
                    # The frame is rebuilt with the cut the rung made, so a resumed run sees the
                    # same orientation the live run saw: a frame that forgets the gap the record
                    # names is a conversation that changed shape across a restart.
                    self.messages = head_msgs + [model.Message(
                        "user", compact.frame(step["text"],
                                              backfill=kept,
                                              covered=covered,
                                              cut=cut_tail))] + tail
                    self._trace = head_trace + [step["id"]] + trace_tail
                    # turns before the notes are gone with the messages they came from; a turn after
                    # them is still ahead in the walk and must not be caught by this
                    emitted |= {t for t, i in first_index.items() if i < index}
                    self._last_summary = step["text"]
                continue
            if kind == "message":
                if step["role"] != "assistant":
                    self.messages.append(model.Message(step["role"], step["text"]))
                    self._trace.append(step["id"])
                    continue
                emitted.add(step["id"])
                emit_turn(step["id"], step["text"], calls_by_turn.pop(step["id"], []),
                          step.get("reasoning", ""),
                          step.get("reasoning_field", "reasoning_content"))
            elif kind == "result":
                turn = turn_of_call.get(step.get("refs", ""), "")
                if turn and turn not in emitted:
                    # the calls of this turn are in the log but their `message` step is not: a run cut
                    # mid-append, or a turn whose header the summary replaced. Rebuild the turn from its
                    # calls rather than dropping it, or the result answers nothing and a server rejects
                    # the orphan. Its text is unrecoverable, so it comes back empty.
                    emitted.add(turn)
                    emit_turn(turn, "", calls_by_turn.pop(turn, []))
                self.messages.append(model.Message(
                    "tool", _with_picture(step.get("content", ""), self._reattach(step)),
                    tool_call_id=call_ids.get(step.get("refs", "")) or None))
                # the `call` step's id, as the live loop pairs it — not this result's own
                self._trace.append(step.get("refs") or step["id"])
            else:
                continue                          # audit the model never saw: no message, no id
        # calls logged and never answered: the run died between the two. The conversation ends where it
        # did, with the calls outstanding, exactly as the live loop left it.
        for turn, calls in calls_by_turn.items():
            if turn not in emitted:
                emit_turn(turn, "", calls)
        return len(self.messages)

    # --- commands in ----------------------------------------------------------

    def submit(self, action: str, *, args: dict | None = None, issuer: str,
               queue_id: str = "") -> None:
        """The one way anything gets a command in — keypress, RPC call, queue file.

        It only enqueues. Commands take effect at the turn boundary, where they are recorded,
        which is what makes "steer while it works" safe: an in-flight tool call finishes."""
        self.pending.append(Command(action, dict(args or {}), issuer, queue_id))

    def _drain(self) -> None:
        """Pull the queued commands in and decide each one, in turn."""
        for command in self._from_queue():
            self.pending.append(command)
        while self.pending:
            command = self.pending.pop(0)
            fields = {"action": command.action, "args": command.args}
            if command.queue_id:
                fields["queue_id"] = command.queue_id
            step = self.log.append("command", command.issuer, **fields)
            self._decide(command, step)

    def _from_queue(self) -> list[Command]:
        """The queued blocks, in file order. A queued command's issuer is the queue: the file
        attests which queue it came from and never who typed it, so the writer's `from:` line
        is carried as an argument — visible, and not an authority."""
        out = []
        for block in control.pending(self.queue):
            args: dict = {}
            if block.text:
                args["text"] = block.text
            if block.reply:
                args["reply"] = block.reply
            if block.prompt:
                args["prompt"] = block.prompt
            if block.cancel:
                args["cancel"] = True
            if block.sender:
                args["from"] = block.sender
            out.append(Command(block.action, args, "queue", block.id, block.closed))
        return out

    def _mark(self, command: Command) -> None:
        """Consume the queue block, so it is not acted on twice."""
        if command.queue_id:
            control.consume(self.queue, command.queue_id)

    def _decide(self, command: Command, step: dict) -> None:
        """Accept or refuse, then act. The decision is appended before the effect, so a run
        that dies between the two still shows what was authorised.

        Delivery is per action. `steer` carries instruction text, so it is consumed before it
        is applied: applying it twice would inject the words twice, and losing one is
        recoverable because the queue keeps what was sent. `interrupt` and `finish` assert a
        desired state, so applying one twice is harmless and dropping one is the forgetting a
        resume exists to remove — they are consumed after."""
        if not command.closed:
            self.log.append("decision", "core", refs=step["id"], outcome="refused",
                            reason="the queue block has no end line, so it was cut off by a "
                                   "crash; it is consumed and nothing is acted on")
            self._mark(command)
            return
        if command.action not in ACTIONS:
            self.log.append("decision", "core", refs=step["id"], outcome="refused",
                            reason=f"undeclared action {command.action!r}; declared actions "
                                   f"are {', '.join(ACTIONS)}")
            self._mark(command)
            return
        if command.action == "steer":
            self._mark(command)
            text = str(command.args.get("text", ""))
            if not text:
                self.log.append("decision", "core", refs=step["id"], outcome="refused",
                                reason="steer needs args.text; an empty steer changes nothing")
                return
            self.log.append("decision", "core", refs=step["id"], outcome="accepted")
            self.say("user", text, issuer=command.issuer)
            return
        if command.action == "answer":
            self._mark(command)
            # Refs the command step, like every other branch: this answer belongs to no question the run
            # asked, so the request id it carries -- if it carries one -- names a step `_close` already
            # closed, and a decision pointing there reads as a second reply to a question nobody asked.
            # With no `--reply` there is no question to name, so the decision closes nothing; the reply
            # the sender named, when there is one, is kept where it can still be read: in the reason.
            reply = command.args.get("reply", "")
            self.log.append("decision", "core", refs=step["id"],
                            outcome="refused",
                            reason="no question is waiting for an answer at a turn boundary"
                                   + (f" (the sender was replying to {reply}, which this run never asked)"
                                      if reply else "")
                                   + "; a reply is matched where the question was asked")
            return
        self.log.append("decision", "core", refs=step["id"], outcome="accepted",
                        reason=command.action)
        self.stop_requested = command.action
        self._mark(command)

    def say(self, role: str, text: str, *, issuer: str, reasoning: str = "") -> str:
        """One message: appended to the log and to the message list, from the same call, so
        the two cannot disagree about what was said. Returns the log-step id, which compaction
        needs to point at what it replaces."""
        fields = {"role": role, "text": text}
        if reasoning:
            fields["reasoning"] = reasoning
        step_id = self.log.append("message", issuer, **fields)["id"]
        self.messages.append(model.Message(role, text))
        self._trace.append(step_id)
        return step_id

    # --- the event bus --------------------------------------------------------

    def subscribe_event(self, fn) -> None:
        """Add a live consumer of streamed chunks. Notification only; a subscriber that raises is
        isolated, so a renderer's bug can never stop the run it is watching."""
        self.event_subscribers.append(fn)

    def emit(self, event) -> None:
        """Hand one live event to every subscriber. This is the single point all output fans out
        through; each consumer decides what to do with it, and none can push back into the run."""
        for sub in self.event_subscribers:
            try:
                sub(event)
            except Exception:                           # display is not the record; never fatal
                pass

    # --- one turn -------------------------------------------------------------

    def step(self) -> None:
        """Ask for a turn, run what it asks for, record both. This is the seam a subclass
        overrides to change how a turn happens without touching when it happens.

        A turn that did not complete dispatches nothing. That covers both ways a turn fails to finish --
        cut off by the output limit, and broken off by an error mid-stream -- and the rule is the same
        because the reason is: the harness is not entitled to act on an instruction the model never
        finished giving. What the two cases leave in the log differs on purpose, and is written where it
        happens: a cut-off turn is retried, so its calls leave no trace at all; a failed turn is terminal,
        so its calls are recorded as refused, which keeps the call/result pairing a resume needs."""
        # Measured before the request goes out, because the usage that answers it is a count of exactly
        # this composition: the local total and the server's reported count are the two sides of one
        # calibration, and a total measured after the turn would be a count of a different request.
        local_context = self.context.total(self.messages)
        request_had_image = any(compact.message_has_image(m.content) for m in self.messages)
        # The fingerprint of the logical request, measured while the list is still what the request
        # carries: after the turn the assistant message is appended and the same list is no longer
        # the request, so a digest taken later would pin a different composition.
        req_sha, req_chars = _request_digest(self.system, self.messages, self.specs,
                                             self.thinking, self.max_tokens)
        events = self.backend.complete(system=self.system, messages=self.messages,
                                       tools=self.specs, thinking=self.thinking,
                                       max_tokens=self.max_tokens)
        prose: list[str] = []
        reasoning: list[str] = []
        reasoning_field: str | None = None
        calls: list[model.ToolCall] = []
        usage: dict = {}
        self.last_error = ""
        stopped = ""
        cut_call = False

        # How long this turn took, timed around the consumption and not around the call: `complete`
        # returns a generator, so the request is not opened until the loop asks for its first event, and
        # a clock stopped when the call returned would measure the harness holding a promise. Monotonic
        # because a wall clock can step backwards between two readings, and a duration cannot be negative.
        turn_start = time.monotonic()
        for event in events:
            self.emit(event)
            if isinstance(event, model.TextDelta):
                prose.append(event.text)
            elif isinstance(event, model.ReasoningDelta):
                reasoning.append(event.text)
                # The first fragment names the field the turn came in on; the re-send mirrors it.
                reasoning_field = reasoning_field or event.field
            elif isinstance(event, model.ToolCall):
                calls.append(event)
            elif isinstance(event, model.Usage):
                usage = {"tokens_in": event.prompt_tokens, "tokens_out": event.completion_tokens}
                # A backend that reports is a calibration, not just a number: the run knows what it
                # sent, so the ratio between its estimate and the server's count is the correction
                # the estimate runs at from here on. Only a request that carried no image teaches it —
                # an image part is counted by its wire bytes while the server counts it as image
                # tokens, and a ratio learned from such a request would be wrong for every text
                # request after (see `compact.message_has_image`).
                if event.prompt_tokens > 0 and not request_had_image:
                    self.context.calibrate(local_context, event.prompt_tokens)
                # What the server said, field by field, and nothing for what it did not say. Omitted
                # rather than written as None: an optional field carrying None is a refused step, and a
                # server that volunteered no cache count or no rate is not a server that reported a miss
                # at zero speed.
                for _field, _value in (("tokens_cached", event.cached), ("tps", event.tps),
                                       ("prefill_tps", event.prefill_tps)):
                    if _value is not None:
                        usage[_field] = _value
            elif isinstance(event, model.Failed):
                if event.truncated:
                    # The output limit stopped the stream mid-arguments, so the JSON cannot parse. That is
                    # not a model error and not a transport error: it is a turn that needs more room, and
                    # leaving it in last_error would end the run as `failed`.
                    cut_call = True
                    continue
                self.last_error = event.error
            elif isinstance(event, model.Stopped):
                stopped = event.reason

        # A turn the output limit ended is not a finished answer. `finish_reason: length` is the server
        # saying so; a half-built call is the same event seen from the other side. Detecting this by
        # matching error text would break the first time the message changed.
        truncated = cut_call or stopped == "length"
        text = "".join(prose)
        if truncated:
            # Which of the three shapes the cut-off turn took decides what the model is told. Reasoning
            # does not count as something written: a turn that spent the whole budget thinking produced
            # nothing to continue from, and "continue where you stopped" to that model is nonsense -- it
            # needs to answer more briefly. Measured on this server with a small budget: 40 tokens of
            # reasoning, empty content, `finish_reason: length`.
            self.truncated_kind = "call" if cut_call else ("prose" if text else "empty")
            # Its calls are not dispatched. A call whose arguments were cut off cannot be run, and a
            # complete call from a turn the limit ended is not safe to half-run either: the model is about
            # to be asked to try again, and a turn that ran silently while it re-answered would leave the
            # log describing acts this turn never committed to. The message list is not told about them
            # either, so no tool result is expected for a call that never happened.
            calls = []
        else:
            self.truncated_kind = ""

        said = "".join(reasoning) or None      # what the next request may be told this turn reasoned
        fields = {"role": "assistant", "text": text,
                  "outcome": "truncated" if truncated else ("failed" if self.last_error else "ok")}
        # The wire's annotation of the turn: the server's raw finish word, and the fingerprint and
        # size of the logical request the turn was asked with. `finish` is absent when the stream
        # broke before a stop word arrived; the pin is always present, because the request was
        # always made.
        if stopped:
            fields["finish"] = stopped
        fields["req_sha"] = req_sha
        fields["req_chars"] = req_chars
        # This one is ours to know, unlike the counts above: a server that reports no usage still took
        # time, and a turn whose console footer cannot say how long it ran is a turn the operator has to
        # time by hand.
        fields["ms"] = int(round((time.monotonic() - turn_start) * 1000))
        # The window this turn was written against, stamped on the turn rather than remembered from the
        # run: a resume may name a different `--context`, and an occupancy figure taken from a later
        # configuration would be a number about the wrong server. A prompt count without its ceiling is
        # not yet a statement about pressure — 4,000 tokens is nothing at 131,072 and nearly everything
        # at 4,096 — and the ceiling is recorded so the reader never has to guess which run supplied it.
        ceiling = self.backend.capabilities().context_tokens
        if ceiling and ceiling > 0:
            fields["context_max"] = int(ceiling)
        if said:
            fields["reasoning"] = said
            # The signature rides with the reasoning: a resume re-sends in the same field the server
            # delivered it in, and the log is the only place that can tell them apart later.
            fields["reasoning_field"] = reasoning_field or "reasoning_content"
        if self.last_error:
            fields["error"] = self.last_error
        fields.update(usage)
        message = self.log.append("message", "model", **fields)
        self.last_usage = usage

        if calls:
            self.messages.append(model.Message("assistant", text,
                                               tool_calls=tuple((c.id, c.name, c.arguments)
                                                                for c in calls),
                                               reasoning=said,
                                               reasoning_field=reasoning_field or "reasoning_content"))
            self._trace.append(message["id"])
        elif text:
            # Reasoning travels with the turn even though it is not text the model "said": an
            # interleaved-thinking server expects its own reasoning to still be in the conversation when
            # it answers across a tool boundary. Whether to send it is the adapter's call — it is the most
            # expensive text in the window — but whether the conversation *has* it is not a transport
            # question, and dropping it here would make the decision unmakeable downstream.
            self.messages.append(model.Message("assistant", text, reasoning=said,
                                               reasoning_field=reasoning_field or "reasoning_content"))
            self._trace.append(message["id"])
        # else: a turn that said nothing and called nothing is recorded above but NOT put into the
        # conversation, and gains no trace id either. An empty assistant message with no tool_calls is
        # not a silent moment to a chat template — qwen3_coder (and others) render it as a blank turn
        # the next request reads as "the user message is empty, just a conversation start marker", and
        # the model, told to always call a tool, then hallucinates. The turn still happened (it is in
        # the log, reasoning preserved there); it contributes no context. The trace id is appended in
        # each branch above rather than once here because `messages` and `_trace` are paired by index:
        # an id with no message shifts every later pair, and the next compaction would point its
        # summary at the wrong turns.

        self.last_calls = calls
        self.last_text = text
        self.last_finish = stopped
        self.last_truncated = truncated
        # The answer the run reports is the pieces joined: a reply continued after a cut-off turn is one
        # answer spread over two turns, and `run()`'s caller -- an operator, or a parent reading a child's
        # last said -- wants the answer, not its last fragment.
        if text:
            self.final_text = (self.final_text + text) if self._carrying else (text or self.final_text)
        for call in calls:
            self._run(call, message, not_ran=(
                # A turn that died mid-stream is terminal: the run ends `failed` whether or not its calls
                # ran, so acting on them first buys nothing and risks a write or a command issued on an
                # instruction the model never finished. Recorded as refused rather than dropped, because a
                # `call` with no `result` resumes as a conversation with a dangling call, which a chat
                # template rejects. A turn cut off by the limit is the other case and is handled above by
                # clearing `calls` entirely: that one is retried, so it leaves no acts behind.
                "the turn failed before it completed; the call was not run" if self.last_error else ""))

    def _run(self, call: model.ToolCall, message: dict, *, not_ran: str = "") -> None:
        """One tool call through the veto point. Both the verdict and the outcome are
        appended, in that order, and a refusal is also what the model is told."""
        call_step = self.log.append("call", "model", refs=message["id"], name=call.name,
                                    **({"call_id": call.id} if call.id else {}),
                                    args=call.arguments)
        if not_ran:
            # Before the policy: a question about whether to allow something that will not be run is a
            # question with no possible purpose, and in an attended run it would cost an interruption.
            self._verdict_and_answer(call, call_step, allowed=False, reason=not_ran)
            return
        if call.name not in self.tools:
            reason = (f"no tool named {call.name!r}; declared tools are "
                      f"{', '.join(sorted(self.tools))}")
            self._verdict_and_answer(call, call_step, allowed=False, reason=reason)
            return
        func = self.tools[call.name]
        try:
            arguments = arguments_for(func, call.arguments)
        except Exception as e:                      # a malformed call is a verdict, not a crash
            self._verdict_and_answer(call, call_step, allowed=False, reason=str(e))
            return
        access, address_arg = self.declared[call.name]
        verdict = self.policy(call.name, arguments, access=access, address=address_arg)
        if not verdict.allowed and verdict.ask:
            verdict = self._negotiate(verdict, call, call_step)
        self._verdict_and_answer(call, call_step, allowed=verdict.allowed, reason=verdict.reason,
                                 address=verdict.address,
                                 run=lambda: self._call_and_pump(lambda: func(**arguments), call))

    def _call_and_pump(self, produce, call: model.ToolCall) -> str:
        """Run a tool and return what it records. A tool returns a value or yields a stream of chunks;
        both are drained here, and a chunk of a stream is handed to the event bus (stamped with the call
        id) as it arrives, so a long-running tool reports progress live. The recorded result is the
        stream's return value, else the concatenation of its chunks, else the plain return — so a
        non-streaming tool that just returns a string behaves exactly as before.

        `produce` is a thunk rather than a function and its arguments: the call has already passed the
        veto point by the time this runs, and what the loop holds at that moment is a closure over the
        chosen tool and its checked arguments, not a callable it could name."""
        result = produce()
        if not isinstance(result, (Iterator, Generator)):
            return result                                # plain return: no stream, nothing to pump
        parts: list[str] = []
        returned = None
        stream = iter(result)
        try:
            while True:
                # Drain with next() rather than a for-loop on purpose: a `for` swallows the generator's
                # StopIteration, and its .value is the tool's recorded result — the one thing a stream
                # may carry that the displayed chunks do not.
                try:
                    chunk = next(stream)
                except StopIteration as stop:
                    returned = stop.value                    # a generator `return` is the result to record
                    break
                delta = chunk if isinstance(chunk, model.TextDelta) else model.TextDelta(str(chunk))
                if delta.call_id is None and call.id:
                    delta = model.TextDelta(delta.text, call.id)   # stamp; the tool need not know its id
                if delta.text:
                    self.emit(delta)                         # live display of progress
                    parts.append(delta.text)
        finally:
            # A stream the consumer stopped pumping is closed here rather than left for the collector to
            # notice whenever it likes. The one tool that streams is the shell, and a generator parked at
            # a `yield` is a command running with nobody watching it: closing it is what reaches that
            # command's own cleanup (see the `finally` in sah.tools.shell). Closing a finished generator costs
            # nothing and changes nothing, so the ordinary path is unaffected.
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        if returned is not None:
            return returned if isinstance(returned, str) else str(returned)
        return "".join(parts)

    def _verdict_and_answer(self, call: model.ToolCall, call_step: dict, *, allowed: bool,
                            reason: str, address: str = "", run=None) -> None:
        """The verdict, then the outcome, in that order — and a refusal is also what the model is
        told. The address is the resolved place when the veto point looked at the filesystem, and
        the typed argument otherwise, so "what did it touch" has one answer in the log."""
        self.log.append("operation", "core", op=call.name,
                        address=address or str(call.arguments.get("path", "")),
                        outcome="allowed" if allowed else "refused", reason=reason)
        if allowed:
            try:
                content = run()                          # a stream was already drained by the runner
                ok = True
            except Exception as e:
                ok, content = False, f"{call.name} raised {type(e).__name__}: {e}"
        else:
            ok, content = False, reason or "refused"
        # A tool that found a picture answers with `Media`: a sentence for the record and an address for
        # the bytes. Whether the picture may travel is decided here and not by the tool, because it is a
        # fact about the endpoint rather than about the file — and the answer says which of the two
        # happened, since silence about an attachment reads as a file that contained nothing.
        picture = content if isinstance(content, Media) else None
        part, why = (None, "")
        if picture is not None:
            part, why = self._attach(picture)
            content = picture.text + (" — attached to this answer" if part
                                      else f" — not attached: {why}")
        elif not isinstance(content, str):
            content = str(content)
        # Every tool result passes through one bound before it can enter the log or the context, so
        # a tool that forgot to cap itself cannot fill the window; and what the bound clips away is
        # not lost, it is saved where the run owns it and the model can read it back.
        content, truncated, total_bytes, spilled = self._bound(call, content, call_step)
        if not truncated:
            fields = {}
        else:
            fields = {"truncated": True, "bytes": total_bytes}
            if spilled:
                # A pointer only when the file is really there: a log that names a path nobody wrote
                # sends the model, and the next reader, chasing something that does not exist.
                fields["spilled"] = spilled
        if part is not None:
            # The path and not the bytes: this is the receipt of an attachment, and an address is what a
            # resume re-opens to rebuild the same picture — which is also why the field names paths and not
            # steps. Recorded only when the part really exists, because a receipt that claims an upload
            # that never happened teaches the next reader to trust a claim the code never verified.
            fields["images"] = [picture.path]
        self.log.append("result", "core", refs=call_step["id"], ok=ok, content=content, **fields)
        self.messages.append(model.Message("tool", _with_picture(content, part),
                                           tool_call_id=call.id))
        self._trace.append(call_step["id"])
        # Every act funnels through here (dispatched or refused), so this is the one place the guard
        # sees the whole act stream. A refused call counts too: hammering a denial is the same
        # non-convergence the guard is for.
        if self.guard:
            self.guard.observe(call.name, call.arguments, ok)

    def _sees(self) -> bool:
        """Whether this run's endpoint is declared able to see a picture.

        Asked of the live backend per turn, exactly as the window's ceiling is: a resume may name a
        different model, and a capability taken from an earlier turn would be a stale claim about a server
        that is not the one being asked."""
        return bool(self.backend.capabilities().supports_vision)

    def _attach(self, picture: Media):
        """`(part, reason)` — the picture's content part, or why there is none.

        Decided before anything can claim an attachment, because the answer and the record both have to say
        what actually happened: this run's declaration first (a base64 blob uploaded to a server that
        cannot read it has already cost the window its bytes), then the file itself.

        The reason names the operator's flag rather than the server's nature. `supports_vision` is a
        declaration made on a command line, so a sentence about what "the endpoint declares" is both false
        — the server was never asked — and useless to the model reading it, which cannot type a flag. It
        would refuse a run whose front-door default never reached it, about a server that answers a picture
        with the right colour, and send the reader to the wrong machine.
        """
        if not self._sees():
            return None, "this run was started without --vision"
        if picture_kind(picture.path) != picture.mime:
            return None, "the file at that path is gone, or is no longer that picture"
        try:
            return picture.part(), ""
        except OSError as e:
            return None, f"the file could not be read ({type(e).__name__})"

    def _reattach(self, step: dict):
        """Rebuild one recorded attachment for a resumed conversation, from the path the record names.

        The bytes were never in the log, so a resume re-opens the file — which is what a resumed text read
        does too: it shows the file as it is now. A file that is gone, or no longer the picture it was,
        yields no part and the answer stands as the text the log holds.
        """
        paths = step.get("images") or []
        if not paths:
            return None
        path = paths[0]                     # a tool answers with at most one picture
        return self._attach(Media(text="", path=path, mime=picture_kind(path) or ""))[0]

    def _bound(self, call: model.ToolCall, content: str, call_step: dict):
        r"""Bound one result for the context, spilling the whole thing if it overflows.

        A tool declares how it is read by setting `ingest_keep` (command output is `"tail"`; the
        default is the head). An over-cap result is written whole under the run's own directory —
        which the run always owns, so the model can `read` it back with an offset — and the model
        receives the bounded preview plus that path. Returns (for_model, truncated, total_bytes,
        spill_path).

        The file is named after the step id the log minted, never after the id the backend streamed:
        that id is an input like any other, and a path built from an input can name anything. A stream
        sending `..\..\..\..\evil` would write outside the run directory — Windows resolves a
        `result-\..` component lexically, so four levels of `..` land one above the root — a stream
        sending `..\..\pwned` chooses the name inside `out/`, and servers that omit an id share one
        file. On POSIX the same ids raise from `open` instead, outside the tool's `try`, and would end
        the run with no `ended` step. Log ids are unique per log and cannot name anything but a file. A
        spill that cannot be written costs the full output and says so; it does not cost the run."""
        keep = getattr(self.tools.get(call.name), "ingest_keep", "head")
        shown, meta = (ingest.clip_tail if keep == "tail" else ingest.clip_head)(content)
        if not meta["truncated"]:
            return content, False, meta["total_bytes"], ""
        path = os.path.join(self.root, "out", f"{call_step['id']}.txt")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            note = ingest.notice(meta, extra=f"full output could not be saved "
                                            f"({type(exc).__name__}: {exc}) — this preview is all "
                                            f"the model receives, and what the tool did is still "
                                            f"in the log")
            return f"{shown}\n\n{note}", True, meta["total_bytes"], ""
        note = ingest.notice(meta, extra=f"full output saved to {path} — read it with offset")
        return f"{shown}\n\n{note}", True, meta["total_bytes"], path

    # --- asking -----------------------------------------------------------------

    def _negotiate(self, verdict: Verdict, call: model.ToolCall, call_step: dict) -> Verdict:
        """Turn a *repeated* refusal into a question. The first refusal of anything asks nobody.

        Asking is the last resort. A model refused once is told what coming back for the same thing
        costs, and left to finish inside the grants it has; only a second attempt on the same address says
        it cannot, and that is the moment worth interrupting a person for. With nobody watching it is not a
        question at all but an honest failure: the run needed an operator, so it ends saying so instead of
        spending two minutes waiting for a voice and then carrying on as if the answer were a no.

        The policy still decides what is askable (only a policy with a `widen` has somewhere to put a yes),
        and only an answer widens."""
        widen = getattr(self.policy, "widen", None)
        if widen is None or not verdict.ask:
            return verdict                              # nothing askable: the refusal stands alone
        # Keyed on the need, not the arguments: a second read of the same file at another offset is the
        # same need, and treating it as a new one would restart the last-resort count forever.
        key = (self.declared[call.name][0] == "write", verdict.address or verdict.ask)
        if key not in self.refused:
            self.refused.add(key)
            return Verdict(False, verdict.reason + " " + ASK_REPEAT[
                "attended" if self.attended else "unattended"], address=verdict.address)
        if not self.attended:
            return self._needs_operator(verdict, call_step)
        granted, how, issuer = self.ask("widen", verdict.ask, why=verdict.reason,
                                        refs=call_step["id"])
        if not granted:
            return Verdict(False, verdict.reason + f" (asked the operator: {how})",
                           address=verdict.address)
        address = widen(verdict.ask, writing=self.declared[call.name][0] == "write")
        return Verdict(True, f"granted for this run by {issuer}: {address} ({how})",
                       address=address)

    def _needs_operator(self, verdict: Verdict, call_step: dict) -> Verdict:
        """The second attempt at something only an operator can give, in a run that has none.

        The question is still recorded, and closed at once: the `request` step is the log's record of what
        the run needed, and a request left open would read as one still being considered. Nothing here
        waits."""
        step = self.log.append("request", "core", subject="widen", item=verdict.ask,
                               why=verdict.reason, refs=call_step["id"])
        self.log.append("decision", "core", refs=step["id"], outcome="refused",
                        reason="nobody can answer in an unattended run")
        self.blocked = verdict.ask
        return Verdict(False, verdict.reason + " (this run has no operator to ask; the run ends here)",
                       address=verdict.address)

    def ask(self, subject: str, item: str, *, why: str = "",
            refs: str = "") -> tuple[bool, str, str]:
        """Put one question to the operator and wait briefly for an answer.

        Recording the question grants nothing. The request is a step; an answer is a different step
        from a different issuer; the widening happens only if the two line up. With nobody to
        answer, the answer is no — a run does not grant itself because it was patient.

        Returns (granted, what happened, who answered). The issuer of an accepted answer is the
        transport that delivered it, never the name the answerer typed for themselves.
        """
        fields = {"subject": subject, "item": item}
        if why:
            fields["why"] = why
        if refs:
            fields["refs"] = refs
        step = self.log.append("request", "core", **fields)
        return self._wait(step["id"])

    def _wait(self, request_id: str) -> tuple[bool, str, str]:
        """Look for this question's answer in the queue until the deadline. A poll, because the
        queue is a file and the loop owns no thread; a stop command is left pending for the turn
        boundary, since applying it here would abandon a turn mid-flight."""
        deadline = time.monotonic() + self.ask_timeout
        while True:
            for block in control.pending(self.queue):
                if block.closed and block.action in ("interrupt", "finish"):
                    return self._close(request_id, False,
                                       f"a {block.action} arrived while the question was waiting "
                                       "(it is applied at the turn boundary)")
                if block.action != "answer" or not block.closed or block.reply != request_id:
                    continue
                control.consume(self.queue, block.id)
                answer = block.text.strip().split("\n")[0].strip()
                if block.sender:
                    self.log.append("command", "queue", action="answer", queue_id=block.id,
                                    args={"reply": request_id, "text": answer, "from": block.sender})
                else:
                    self.log.append("command", "queue", action="answer", queue_id=block.id,
                                    args={"reply": request_id, "text": answer})
                note = "answered " + (answer if answer else "empty, which is not a yes")
                if block.sender:
                    note += f" by {block.sender}"
                return self._close(request_id, answer.lower() in YES_WORDS, note, issuer="queue")
            if time.monotonic() >= deadline:
                return self._close(request_id, False,
                                   f"no answer within {self.ask_timeout:g}s; a run with no witness "
                                   "does not grant itself")
            time.sleep(POLL_SECONDS)

    def _close(self, request_id: str, granted: bool, note: str,
               issuer: str = "core") -> tuple[bool, str, str]:
        """Every question gets a `decision` that closes it, including one nobody answered: a request
        left open in the log is indistinguishable from one still being considered."""
        self.log.append("decision", issuer, refs=request_id,
                        outcome="accepted" if granted else "refused", reason=note)
        return granted, note, issuer

    # --- the loop -------------------------------------------------------------

    def _begin(self) -> None:
        """Clear the state that belongs to one objective, before starting one.

        `run` is called once per objective — a session calls it again for the next thing the
        operator types — so anything left from the last objective would be acted on again: an
        `interrupt` already requested would end this objective before the backend was ever asked,
        the previous objective's answer would be returned as this one, and nudges spent on an old
        streak would turn this objective's first repetition into a stop instead of a warning.

        What survives is the conversation (`messages`, `_trace`) and the wiring: an objective ends,
        a session continues."""
        self.stop_requested = ""
        self.nudges = 0
        self.last_calls = []
        self.last_text = ""
        self.last_error = ""
        self.last_finish = ""
        self.final_text = ""
        # What this objective has already been refused once. Asking is a last resort, so the first refusal
        # of an address is just a refusal; the set is what recognises the second attempt as the same need.
        self.refused: set = set()
        # Set when the run needs something no one in this run can give it, and `run` ends on it.
        self.blocked = ""
        # Recovery counts per objective, not per session: a second question deserves its own two tries at
        # the output limit rather than inheriting the first question's exhausted budget.
        self._truncations = 0
        # How many times this objective has made room after a refusal for size. One, and the reason is in
        # the loop above: each attempt costs a compaction, which is a loss of conversation.
        self._overflow = 0
        # The budgets judge this objective only, the way the recovery counts do: the turn count and
        # the grace left to spend reset, and the wall the time budget counts against starts now, so
        # a resumed run is judged on the budget its own invocation was given.
        self._turns = 0
        self._budget_reason = ""
        self._grace_spent = 0
        self._objective_start = self._clock()
        self._carrying = False
        self.last_truncated = False
        self.truncated_kind = ""
        if self.guard:
            self.guard.reset()

    def _reach_field(self) -> dict:
        """The reach this run begins with, asked of the policy that will refuse it.

        Read from the policy rather than passed in beside it: a run that recorded one object's grants while a
        second object decided its refusals would carry a log describing a policy it does not have. Asked, not
        required — a policy with no `reach` records nothing, because writing `exec: true` for a harness with
        no confinement would put in the record a claim the code never verified. `sah export` renders this
        field.
        """
        reach = getattr(self.policy, "reach", None)
        return {"grants": reach()} if callable(reach) else {}

    def _head_fields(self) -> dict:
        """The fixed head of the render input, stamped once at run start.

        The exact system prose and tool contract the model is told and offered — the rendered
        bytes, not a pointer at the prompt file. The file and the tool code are mutable world
        state: reword the preamble next month and this run can no longer show what it was told
        unless the record holds the copy. `prefix_digest` is the head's identity; `resident_tokens`
        is the run's own estimate of the window the head occupies before any user turn — the same
        number the compactor's notices report. `max_tokens` is the output budget the head was sent
        under; absent when the operator set none, because an optional field carrying None is a
        refused step. `project_notes_file` is the working directory's instruction file that was
        framed into `system`, present only when the operator granted it and a file was loaded — the
        content is in `system`, this is the derivation input's identity, greppable as a field."""
        fields = {"system": self.system,
                  "tools": [_spec_wire(s) for s in self.specs],
                  "prefix_digest": _prefix_digest(self.system, self.specs),
                  "resident_tokens": self.context.resident()}
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        fields["max_turns"] = self.max_turns
        fields["grace_turns"] = self.grace_turns
        if self.max_runtime > 0:
            # Absent when the operator set none, the way `max_tokens` is: an optional field
            # carrying "no limit" would be a value every reader has to un-interpret.
            fields["max_runtime"] = self.max_runtime
        # Absent at their defaults (streaming, usage requested), present only when the operator
        # turned one off: a grep for `"streaming": false` finds exactly the runs that talked to
        # the server in the other shape, without a field every ordinary run has to carry.
        if not self.streaming:
            fields["streaming"] = False
        if not self.usage:
            fields["usage"] = False
        # The reasoning wire shape the operator declared, the same stamp idiom as `streaming` and
        # `usage`: absent at the defaults (the mirror, no pin), present only on the deviation, so a
        # grep finds exactly the runs whose requests spoke the server's other shape.
        if self.require_reasoning_content:
            fields["require_reasoning_content"] = True
        if self.reasoning_out:
            fields["reasoning_out"] = self.reasoning_out
        if self.project_notes_file is not None:
            fields["project_notes_file"] = self.project_notes_file
        if self.summary_thinking:
            # The operator named a level for the summary calls: the wire fact is stamped the way
            # the other wire knobs are, and absent at the default (the run's own level left alone).
            fields["summary_thinking"] = self.summary_thinking
        return fields

    def run(self, text: str = "", *, issuer: str = "user") -> str:
        """Run to a stop condition and append how it ended. The end state needs no witness:
        the log carries it and the exit code says whether it went badly.

        One objective, beginning from a cleared control state (`_begin`) on a conversation that may
        already be long."""
        self._begin()
        caps = self.backend.capabilities()
        self.log.append("started", "core", run_id=self.run_id, root=str(self.root), cwd=str(self.cwd),
                        # The process behind this log: a run that dies without its `ended` line leaves
                        # a record that claims to be alive, and the PID is the witness a reader on the
                        # same machine can check against the process table.
                        pid=os.getpid(),
                        # The log field is an ident, and the loop's "no preference" is None; naming
                        # the harness's own word keeps a later reader able to enumerate what
                        # posture a run had instead of finding a field missing.
                        model=caps.model, thinking=self.thinking or model.REASONING_NONE,
                        # The window the compaction arithmetic ran against: declared, probed or
                        # the default — the record holds the number, and the CLI's `context`
                        # notice says how it was derived, so a reader of a degenerate compaction
                        # can see the window was a guess rather than a declaration.
                        context=caps.context_tokens,
                        **self._head_fields(), **self._reach_field())
        if text:
            self.say("user", text, issuer=issuer)

        reason = "complete"
        try:
            while True:
                self._drain()
                if self.stop_requested:
                    reason = "interrupted" if self.stop_requested == "interrupt" else "finished"
                    break
                # The budgets speak at the turn boundary, where an interrupt takes effect: neither
                # cuts a turn mid-flight. The first trip is told to the model -- its conversation is
                # the only place a trip can be told -- with the grace turns as the leeway to deliver
                # what it has, and a run that answers in them ends complete. The notice beside the
                # word names the budget that ran out, so the record says why the run changed shape.
                # A trip during grace does not re-fire: once the model is told to wrap up, the grace
                # is the promise it was given, and the run ends when the grace is spent or the model
                # answers.
                if not self._budget_reason:
                    hit = self._budgets_hit()
                    if hit:
                        self._budget_reason = hit[0]
                        self.log.append("notice", "core", subject="nudge",
                                        text=self._budget_notice(hit))
                        if self.grace_turns <= 0:
                            reason = self._budget_reason
                            break
                        self.say("user", WRAP_UP, issuer="core")
                elif self._grace_spent >= self.grace_turns:
                    reason = self._budget_reason
                    break
                self._turns += 1
                self.step()
                if self._budget_reason:
                    # The turn just run spent one of the grace turns the model was owed, whether
                    # it delivered or not: a grace turn is a turn, and a truncated one that retries
                    # spends the way any turn spends.
                    self._grace_spent += 1
                if self.last_error:
                    # A request the server refused for its size is the emergency compaction was built for,
                    # and it arrives the wrong way round: the usage that triggers compaction comes from a
                    # request that succeeded, and a refused one reports none. So the refusal is read as the
                    # signal, room is made, and the turn is asked once more. Once, because a window simply
                    # too small for the task would otherwise compact and retry until the turn budget ran
                    # out, throwing away conversation on a run that was never going to finish.
                    if self._recover_overflow():
                        continue
                    reason = "failed"
                    break
                if self.blocked:
                    # The run asked for something only an operator could give it, and there is no operator
                    # in this run. That is not a completion and not a crash: the log says what was needed.
                    reason = "needs_operator"
                    break
                if self.last_truncated:
                    # The turn was ended by the output limit, so it is not an answer and the run may not
                    # call itself complete. Recover the way the guard recovers: a core `say` (the model is
                    # told, in its own conversation) plus a `notice` (the operator can see why the run is
                    # spending another turn), then ask again -- bounded, because a limit that is too small
                    # stays too small.
                    self._truncations += 1
                    if self._truncations > CONTINUE_LIMIT:
                        self.log.append("notice", "core", subject="nudge",
                                        text=f"the answer was still cut off after {CONTINUE_LIMIT} "
                                             f"continuations; the output limit is too small for this task "
                                             f"-- raise --budget")
                        reason = "truncated"
                        break
                    self.log.append("notice", "core", subject="nudge",
                                    text=f"the turn hit the output limit ({self.truncated_kind}); asking "
                                         f"for the rest ({self._truncations} of {CONTINUE_LIMIT})")
                    self.say("user", CONTINUE_PROMPTS[self.truncated_kind], issuer="core")
                    self._carrying = True
                    continue
                # The trigger does not wait on the server's number to arrive: a backend that reports
                # no usage is exactly the case where the run's own estimate has to speak, and the
                # server's words on a later refusal are the only net that would be left.
                if self.compact_enabled:
                    self._maybe_compact()
                if self.last_calls:
                    # A turn that made calls: check the guard before asking for another turn. It only
                    # decides; the nudge is written here (a core `say` + a `notice`), and the stop is the
                    # loop breaking with a `stalled` reason rather than continuing to spin.
                    #
                    # This consult-and-write shape is exactly the deferred `provide` contract, and this
                    # is where that seam would live if it ever shipped: the core asks a module for a
                    # decision (`escalation()` here), and the core alone performs the write. Today the
                    # module is the single hard-wired guard; a `provide` contract would generalise that
                    # to any registered module that answers a core question -- but it would still give the
                    # responder no path to write, only to advise, so it must not become a disguised
                    # back door. It is deferred because one consumer does not earn a third contract; a
                    # second watcher wanting in here would. See docs and sah.guard.
                    if self.guard:
                        esc = self.guard.escalation()
                        if esc == "stop":
                            self.log.append("notice", "core", subject="nudge",
                                            text=f"stopped: {self.guard.repeating} was repeated "
                                                 f"identically after {self.guard.nudges} nudges to change")
                            reason = "stalled"
                            break
                        if esc == "nudge":
                            self.guard.spend_nudge()
                            self.log.append("notice", "core", subject="nudge",
                                            text=f"the last calls of {self.guard.repeating} were "
                                                 "identical; nudged to change approach")
                            self.say("user", REPEAT_STEER, issuer="core")
                    # The completion declaration lands as a call like any other; what differs is
                    # what the loop does with it: this shape of run ends here, and the act is
                    # recorded beside the ending. The harness does not judge whether the task is
                    # served — the caller that set the task verifies the result against the world;
                    # the log says the declaration happened. It is consulted only on an unattended
                    # run, where the tool is the one offered: an attended run never sees it in the
                    # schema, so a name match there would be a match on a refusal.
                    if not self.attended and any(call.name == VERIFY_OBJECTIVE
                                                 for call in self.last_calls):
                        self.log.append("notice", "core", subject="verify",
                                        text="completion declared: verify_objective called; "
                                             "the run ends")
                        reason = "complete"
                        break
                    continue                      # the model is still working; ask again
                if self.attended:
                    # The operator is the backstop: prose is the delivery, and a turn with words
                    # and no call completes the objective. Only an empty turn is asked about.
                    if self.last_text.strip():
                        reason = "complete"
                        break
                    reason = self._empty_turn()
                    if reason != "nudge":
                        break
                    self.log.append("notice", "core", subject="nudge",
                                    text="the turn produced neither prose nor a tool call")
                    self.say("user", EMPTY_TURN, issuer="core")
                else:
                    # No one is there to read prose as done: a turn with no call — prose or none —
                    # is asked for the call or the declaration, bounded, and a run that cannot
                    # prove done ends blocked rather than complete.
                    reason = self._no_call_turn()
                    if reason != "nudge":
                        self.log.append("notice", "core", subject="nudge",
                                        text=f"no tool call after {NUDGE_LIMIT} nudges; the run "
                                             "cannot prove done and ends blocked")
                        break
                    self.log.append("notice", "core", subject="nudge",
                                    text=("the turn produced prose but no tool call"
                                           if self.last_text.strip()
                                           else "the turn produced neither prose nor a tool call"))
                    # The nudge quotes the turn's own wire facts — the finish word and the
                    # completion count the server reported for it — because a turn with no call
                    # invites an inference about why none came, and the measured facts are the
                    # correction: a model told what actually happened does not invent a truncated
                    # call. The "was not cut off" clause is the machinery's verdict, not a wire
                    # assertion: a turn the output limit ended takes the continue_* prompts above
                    # and never reaches here. The numbers are a projection of a measurement, never
                    # invented — a server that withheld them gets the floor clause (the turn
                    # ended), not a guess about the numbers.
                    finish = self.last_finish
                    tokens_out = self.last_usage.get("tokens_out")
                    evidence = (f"— it ended at {tokens_out} tokens with finish={finish}"
                                if finish and tokens_out is not None else "— it ended")
                    self.say("user", prompts.load("no_call_nudge", evidence=evidence),
                             issuer="core")

        except KeyboardInterrupt:
            # A signal can land anywhere: mid-request, mid-tool, mid-stream. Whatever it lands in the
            # middle of, the record still has to say the run ended -- `recall` and `child_status` ask
            # the log whether a run is over, and a run with no `ended` step is reported as running
            # forever. The code is 128+SIGINT so a caller can tell a run cut short from the graceful
            # stop an operator queued at a turn boundary (which is exit 0 and means "as asked"). Then
            # the signal goes on: the caller owns the session and decides whether it survives.
            self.log.append("ended", "core", reason="interrupted", exit_code=128 + 2)
            raise
        except compact.ContextBlocked:
            # Compaction proved, before spending a summary call, that nothing fits: the head, the
            # smallest slot and the backfill exceed the limit. That is a declared ending, not a
            # crash — the numbers are already on the record (the notice `compact` wrote), the
            # ending is `blocked` with the code a caller can tell apart from a failure, and the
            # run stops where the arithmetic says it must, the conversation untouched.
            self.log.append("ended", "core", reason="blocked", exit_code=1)
            return self.final_text
        except Exception as exc:
            # A crash is an ending the same way a signal is: without its `ended` line the dead run
            # reads as still running, forever, to every reader of the log. The traceback goes on
            # the record beside the ending, and the exception goes on too -- a crash swallowed here
            # would be quieter than the crash it replaces, and the operator still deserves it.
            self.log.append("notice", "core", subject="crash",
                            text="".join(traceback.format_exception(exc)))
            self.log.append("ended", "core", reason="crashed", exit_code=1)
            raise
        self.log.append("ended", "core", reason=reason,
                        exit_code=0 if reason in ("complete", "finished", "interrupted") else 1)
        return self.final_text

    # --- the budgets ------------------------------------------------------------

    def _budgets_hit(self) -> list[str]:
        """The budgets that ran out at this turn boundary: the turn budget first, the time
        budget second; empty while the run is inside both.

        Judged on this objective's own count and clock (see `_begin`), so a resumed run is judged
        on the budget its invocation was given, not on a conversation it inherits. The turns
        budget is consulted first because it is the loop's own unit; when both run out on the
        same boundary the run is named for the one it counts. The time budget is wall time, so a
        turn that spends most of it inside a long tool call counts it there too."""
        hit = []
        if self._turns >= self.max_turns:
            hit.append("max_turns")
        if self.max_runtime > 0 and self._clock() - self._objective_start >= self.max_runtime:
            hit.append("max_runtime")
        return hit

    def _budget_notice(self, hit: list[str]) -> str:
        """The record's words for a budget trip: which budget ran out, with its numbers, and
        what the model was told. A record, not an instruction, so it stays in code the way the
        other notices do."""
        details = []
        if "max_turns" in hit:
            details.append(f"{self._turns} of {self.max_turns} turns")
        if "max_runtime" in hit:
            details.append(f"{int(self._clock() - self._objective_start)}s of {self.max_runtime}s")
        if self.grace_turns > 0:
            unit = "turn" if self.grace_turns == 1 else "turns"
            leeway = f"the model is told to wrap up; {self.grace_turns} grace {unit} left to deliver"
        else:
            leeway = "no grace turns; the run ends"
        return f"budget ran out ({'; '.join(details)}); {leeway}"

    # --- keeping inside the context -------------------------------------------

    def _recover_overflow(self) -> bool:
        """Make room after a server refused the request as too large, and say whether to retry the turn.

        Returns True when the compaction ran — the slot is always filled (a summary, a cut summary,
        or the failed note), so the conversation is always rewritten and the turn is asked once
        more — and a compaction that cannot make the context fit does not return but raises
        `ContextBlocked`, which ends the run `blocked` with the numbers. Nothing here fires for an
        error that is not about size — see `compact.looks_like_overflow` for why that distinction
        is the one that protects the conversation."""
        if (not self.compact_enabled or self._overflow >= 1
                or not compact.looks_like_overflow(self.last_error)):
            return False
        self._overflow += 1
        self.log.append("notice", "core", subject="compact",
                        # The server's own words, window size included, because "it was too big" is not a
                        # diagnosis an operator can act on and this text says which number to raise.
                        text=f"the server refused the request as too large — {self.last_error[:300]} — "
                             f"summarising the oldest turns and retrying this turn once")
        # The compaction either rewrites the conversation (the slot is always filled) or raises
        # ContextBlocked, which ends the run blocked with the numbers: there is no third shape,
        # so the retry is the one.
        self.compact()
        self.last_error = ""
        return True

    def _maybe_compact(self) -> None:
        """Compact if the window is full: the server's count of the last request when it reported
        one, the run's own estimate of the window the next request will carry when it did not. The
        two sources answer the same question — how full is the context — and the fallback exists
        because a backend that reports no usage is not a backend with an empty context; on its own
        number the trigger would be blind to such a run, and the only net left would be the
        server's words on the refusal, which the overflow markers may not match."""
        tokens_in = self.last_usage.get("tokens_in", 0)
        context_tokens = self.backend.capabilities().context_tokens
        prompt = tokens_in if tokens_in > 0 else self.context.effective(self.messages)
        if compact.should_compact(prompt, context_tokens, len(self.messages)):
            if tokens_in <= 0:
                # The decision's source belongs in the record: a compaction fired on the run's own
                # estimate rather than a number the server reported, and the operator reading the
                # log has to be able to tell which side of that line this one fell on.
                self.log.append("notice", "core", subject="compact",
                                text=f"the backend has reported no prompt count, so the trigger fell "
                                     f"back to the run's own estimate of the next request: "
                                     f"{self.context.decomposition(self.messages)} makes ~{prompt} "
                                     f"of a {context_tokens}-token window")
            self.compact()

    def _pre_tokens(self) -> int:
        """The size of the context the next request carries: the server's count of the last
        request when it reported one, else the run's own estimate of what the next request will
        carry. The same source the trigger reads (see `_maybe_compact`): the compaction's
        arithmetic must run on the same number the decision to compact was, or the room is
        computed for a context that is not the one being compacted."""
        tokens_in = self.last_usage.get("tokens_in", 0)
        return tokens_in if tokens_in > 0 else self.context.effective(self.messages)

    def compact(self) -> dict:
        """Summarise the oldest turns into the slot at the head of the conversation, and record
        the act.

        The layout is fixed: the oldest turns — the run's opening turn included; no message is
        carved out — are summarised into a single slot, and the most recent messages survive
        verbatim as the backfill the slot's note names. Keeping the goal standing after the
        rewrite is the summariser's, not a harness privilege: the source is the whole history,
        and a run may open with a greeting and set its goal turns later, or be steered so that
        what it now works toward is not what its first message asked for. The previous summary is
        merged into the next source (each summary re-summarises the last slot), so a second
        compaction sees the first through the source rather than through turns the summary
        replaces.

        The summary call carries no output cap and no tools, and runs at the run's own reasoning
        level unless the operator named a different one for summaries; acceptance is the
        harness's measurement of the answer against the room the window leaves it — the
        server's output gate never decides (see `sah/compact.py`). When the backfill itself
        exceeds the room a compaction must land within, the rung cuts it — the oldest messages,
        in whole units, are dropped until the tail fits, the record names the cut, and the
        frame's note tells the model the gap is a gap. When even the resident load and the
        smallest slot exceed the limit, nothing can fit: the run ends `blocked` with the
        numbers, the conversation untouched, before any summary call is spent that the result
        could not use.

        A compaction that runs always rewrites the conversation — the slot holds a summary, a cut
        summary, or the failed note — so the message list and its trace shrink together: the two
        must never disagree about how many entries there are. The act is then told the way a nudge
        is told: a core message and the same text in a notice beside the record, so every rendered
        surface shows that the context was rewritten — without token numbers, because a number in
        model-facing text is a budget the model can act on; the operator's numbers are in the
        step's `budget` field and the turns' usage. The summary is a model-facing observation and
        carries no authority — grants live only in the live invocation, unchanged by anything
        written here."""
        drop, keep = compact.split(self.messages)
        if not drop:
            return {}
        window = self.backend.capabilities().context_tokens
        if window <= 0:
            # A window with no size stated has no answer to "does it fit", so no compaction: the
            # trigger cannot fire on one either, and the emergency path names the numbers it has.
            return {}
        pre = self._pre_tokens()
        resident = self.context.resident()
        minimum = compact.estimate_tokens(keep)
        limit = compact.limit_for(window, pre)
        # The fit check runs before any summary is asked, and when it fails the backfill is cut
        # before the run is declared blocked: the oldest messages of the verbatim tail, in whole
        # units, are dropped until the tail fits the limit — the last rung's guarantee extended
        # to the tail, so a compaction that runs rewrites the conversation whatever the tail
        # holds, and the declared blocked ending is left for the window that cannot hold the
        # resident load at all. The cut is a record (`backfill_cut` on the summary step) and a
        # word to the model (the frame's cut note): the tail is gone, not hidden.
        failed_note = prompts.load("slot_failed", dropped=len(drop))
        smallest = resident + compact.estimate_text(failed_note) + minimum
        backfill_cut = 0
        if not compact.fits_within(smallest, limit):
            keep, backfill_cut = compact.cut_backfill(
                keep, limit - resident - compact.estimate_text(failed_note))
            minimum = compact.estimate_tokens(keep)
            smallest = resident + compact.estimate_text(failed_note) + minimum
        if not compact.fits_within(smallest, limit):
            # Even with the backfill cut to nothing the resident load and the failed note exceed
            # the limit: the window cannot hold what every request carries. No summary, no cut
            # can make it fit, so the run ends blocked with the numbers, having spent nothing —
            # the operator's answer is a bigger window or a smaller resident load, not another
            # compaction. Naming the numbers is the whole value: an operator can see which of
            # them is too big for the window, and that is a different fix than "compact more".
            self.log.append("notice", "core", subject="compact",
                            text=(f"no compaction can fit the context: the resident load and "
                                  f"the failed note make ~{smallest} tokens "
                                  f"(~{resident} resident — {self.context.decomposition([])}) "
                                  f"against a limit of {limit} (window {window}, pre ~{pre})"
                                  + (f"; the backfill was cut to {len(keep)} messages by the cut "
                                     f"rung and still does not fit"
                                     if backfill_cut else "")
                                  + "; no summary can make it fit, so the run stops"))
            raise compact.ContextBlocked(
                f"no compaction can fit the context: ~{smallest} tokens of resident load and "
                f"smallest slot against a limit of {limit} (window {window}, pre ~{pre})"
                + (f"; the backfill was cut to {len(keep)} messages and still does not fit"
                   if backfill_cut else ""))
        room = compact.room_for(window, pre, resident, minimum)
        slot_text, cost, stage, slot, note = compact.summarise(
            self.backend,
            source_text=compact.source(self._last_summary, drop),
            room=room, thinking=self.summary_thinking or self.thinking,
            dropped=len(drop))
        refs = [t for t in self._trace[:len(drop)] if t]
        # The budget the act ran against, in the record's own words: the limit and what made it,
        # the resident parts split into system prompt and tool schemas — the tool component is
        # the regression guard for the measured defect where the tool surface was ignored in the
        # budget — the backfill, the cut the rung made of it, and the room left the summary. A
        # reader of the log sees the whole arithmetic without re-deriving it.
        budget = (f"limit {limit} (window {window}, pre ~{pre}); "
                  f"{self.context.decomposition([])}; ~{minimum} backfill; "
                  f"~{room} room for the summary"
                  + (f"; backfill cut by {backfill_cut} messages" if backfill_cut else ""))
        step_fields = {"refs": refs, "kept": len(keep), "dropped": len(drop),
                       "text": slot_text, "outcome": "ok" if slot != "failed" else "failed",
                       "stage": stage, "slot": slot, "tokens_cost": cost, "budget": budget}
        if backfill_cut:
            # Absent at zero, the way the other stamps are: a grep for the field finds exactly
            # the compactions that cut the tail, not every compaction.
            step_fields["backfill_cut"] = backfill_cut
        if note:
            # The attempts the slot did not come from, in the server's words: which stage paid for
            # the summary, and what the earlier ones cost, is a fact about the run the operator can
            # only see here. `outcome` says the act completed and filled the slot; `slot` says what
            # the slot holds, and a slot of `failed` is the only `outcome` of `failed` — every
            # other slot text is a summary the act produced.
            step_fields["reason"] = note
        # The step is written before the conversation is rewritten, because the id it mints is what
        # the trace slot needs: the summary is now the oldest entry, and a later
        # compaction builds its `refs` by filtering falsy trace entries. An empty slot here meant
        # that second compaction dropped the first summary without naming it -- and disagreed with
        # replay, which stores the id, so a resumed run carried a different trace from the live one.
        step = self.log.append("summary", "core", **step_fields)
        self._last_summary = slot_text
        frame_msg = model.Message("user", compact.frame(
            slot_text, backfill=len(keep), covered=slot in ("summary", "cut"),
            cut=backfill_cut))
        # The slot takes the conversation's head; the cut is already out of `keep` (the rung
        # returned the tail that survives), so the list and the trace shrink by the drop and
        # the cut together: the two must never disagree about how many entries there are.
        self.messages = [frame_msg] + keep
        self._trace = [step["id"]] + self._trace[len(drop) + backfill_cut:]
        # The act is told the way a nudge is told: a core message, because the conversation is the
        # only place the model can be told its context changed under it, and the same text goes to
        # the record beside it, where every rendered surface shows it. The wording carries no token
        # numbers on purpose: a number in model-facing text is a budget the model can act on
        # (measured in a pi run — a model told its context sat at 80% ended the run itself to
        # "take a break"), and a model that manages the window it was given the task in stops
        # working for a number. The operator's numbers sit where the arithmetic is: the step's
        # `budget` field and the usage the turns around the act report. A compaction that raises
        # ContextBlocked never gets here, so a run that ends blocked is told by the head check's
        # notice, not this one.
        text = prompts.load("compact_done")
        self.log.append("notice", "core", subject="compact", text=text)
        self.say("user", text, issuer="core")
        return step

    def _empty_turn(self) -> str:
        """A turn with no prose and no call, in a run an operator is watching: either a nudge or
        a failure. The unattended shape has its own word (`_no_call_turn`), because prose there
        is not a completion the operator can read."""
        if self.nudges >= NUDGE_LIMIT:
            return "failed"
        self.nudges += 1
        return "nudge"

    def _no_call_turn(self) -> str:
        """The structural nudge of a run no operator is watching: a turn with no call, prose or
        not, is either a nudge or a blocked ending. It shares the attended path's budget — a run
        is one or the other, so one counter serves both without a second state."""
        if self.nudges >= NUDGE_LIMIT:
            return "blocked"
        self.nudges += 1
        return "nudge"
