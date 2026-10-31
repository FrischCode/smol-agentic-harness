"""sah.compact — keep a session inside its context, and say what it cost.

A session fills up. This module decides when that matters, produces the summary that replaces the
oldest turns, and records the whole act — including the attempts that failed — as a `summary` step.
It is deliberately dumb about *content*: it asks the model to summarise and trusts the answer; it is
rigorous about where the size decisions are made, which is the seam the design it replaces got
wrong — the server's output gate there decided whether a summary fit, and a backend that ignored
the gate or misreported the stop silently broke acceptance.

**The summary request carries nothing but the compactor's system prompt and the content to
summarise: no tool schemas, no max_tokens, and no reasoning level but the run's own.** The server is
never given a gate on the answer's length, so nothing on the wire can decide whether the answer
fits — that decision is the harness's, and a reply the output limit cut off is a failed attempt,
its text discarded, not a partial success. With no output cap on the wire there is nothing for
reasoning to spend and nothing to strip, so the call runs at the level the run runs at — left
alone, not overridden — or at the level the operator named for summaries (`--summary-think`):
on a server that thinks heavily the summary's room can be spent on thinking the record will not
keep, and the answer to that is the knob, not an algorithm that amputates the level.

**Acceptance is a measurement against a defined room, not a server verdict.** The room a summary
may occupy is the limit — the smaller of the target share of the window and the reduction of what
the context holds now — with everything every request carries beside it subtracted: the resident
parts (the system prompt and the full tool surface) and the backfill that always survives. The
answer's size is the server's count of the call when it reports one — on a thinking server that
includes its thinking, the deliberate over-measure the knob exists to answer — else the standing
estimate. One predicate, shared by every stage and by the fit check that runs before any stage:
within the limit.

**The stages are bounded and the slot is always filled.** A free templated summary, then the same
source asked for one that fits the room — the prompt names the number — then the latest summary
asked to fit the room, then the last text cut to the room and marked, so the model is told it now
works on potentially missing information. When it is the tail itself that is too big for the
limit, the cut reaches the backfill: the oldest messages of the verbatim tail, in whole units, are
dropped until the tail fits (`cut_backfill`), and the frame's note says the gap is a gap. No path
leaves the conversation untouched and doomed: a compaction that ran rewrites it, whatever the slot
and the tail hold (a summary, a cut summary, or the failed note, beside a tail the cut rung made
fit) — and when the window cannot hold the resident load and the smallest slot at all, nothing can
fit, so `ContextBlocked` ends the run `blocked` with the numbers, before any summary call is spent
that the result could not use.

**No turn is privileged — the goal is the summariser's, not a head the harness keeps.** The first
user message is summarised like the rest: a run may open with a greeting and set its goal turns
later, and a steered or resumed run works toward what its latest instruction says, which the
positional first message would not be. Keeping the goal standing after the rewrite is the
summariser's job — the source is the whole history, and the summary's goal section is what the
next turn reads instead of the turns it replaces. The slot sits first, before the verbatim
backfill, and each summary merges the previous one into its source, so a second compaction sees
the first through the source rather than through turns it replaces.

**The summary carries no authority.** It is model-facing — it goes back as the oldest context the
model sees — and it is a `summary` step, whose kind the record layer already declines from the
authority subset. A compaction must never become a way to rewrite what the run was permitted: the
summary text is dropped into the conversation as an observation, and the grants live only in the
live invocation, exactly as they did before the summarise.

**The run owns the size of its own window; the backend's number is the correction, not the source.**
A run knows exactly what its next request will carry — the system prompt is assembled, the tool
schemas are built, and the conversation is its own list — and it counts that at the standing rate,
per request, as a pure function of the outgoing request rather than as a cumulative counter: a
model that is recomputed from the same state the next request sends cannot drift from it, whatever
compaction or resume has done to the list in between. When a backend reports the prompt tokens of
a request the run measured, the ratio between the two becomes the rate the estimate runs at
(`Context.calibrate`); when a backend reports nothing, the uncalibrated estimate is what the
trigger falls back on — a server that is silent about usage is not a server with an empty context,
it is one the server-number trigger would be blind to, and the only net left for such a run would
be the server's words on the refusal, which the overflow markers may not match.
"""
from __future__ import annotations

import json

from sah import model, prompts

# The share of the window a compaction aims summary + backfill within, and the share of
# what the context holds now that it must leave (see `limit_for`). Both sit well below the
# trigger on purpose: compaction buys headroom, and a result that lands at the trigger is a
# result the next turn refills.
TARGET = 0.4

REDUCING = 0.6

# A tool result longer than this in the summary's source is cut and named: the cap is on the
# request's side (the summary call carries only its system prompt and the source), and a result
# an operator never re-reads need not arrive at the summariser in full.
TOOL_RESULT_CHARS = 2000


class ContextBlocked(Exception):
    """Compaction cannot make the context fit, and no cut can: with the backfill cut to nothing,
    the resident load (the system prompt and the tool surface) and the smallest slot the run can
    write still exceed the limit a compaction must land within — the window cannot hold what
    every request carries, so no summary, no cut of the tail, can make it fit. The run ends
    `blocked` with the numbers — `text` is the sentence the notice and the record carry — the
    conversation untouched: the operator's answer is a bigger window or a smaller resident load,
    and a run that would spend summary calls it cannot win is losing conversation to answer a
    question the arithmetic already answered, so it stops where the arithmetic says it is,
    having spent nothing to find it."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


def limit_for(window: int, pre: int) -> int:
    """The ceiling a compaction result must stay within: the smaller of the target share of the
    window and the reduction of what the context holds now. The two ceilings are one predicate —
    a result is accepted when summary + backfill is within this one number — and the min
    is the whole meaning: a context that is 90% full has less room to give than one that is 82%
    full, whatever the window is."""
    return min(int(TARGET * window), int(pre * (1 - REDUCING)))


def room_for(window: int, pre: int, resident: int, minimum_tokens: int) -> int:
    """What a summary may occupy: the limit with everything every request carries beside it
    subtracted — the resident parts (the system prompt and the full tool surface) and the
    backfill that always survives. No turn is subtracted: keeping the goal standing after the
    rewrite is the summariser's job over the history, not a head the harness reserves. A summary
    bigger than this cannot make the context fit, whatever else it displaces, so it is not
    accepted; a room that is not positive is a compaction with nothing to attempt, and the slot
    takes its smallest text instead."""
    return limit_for(window, pre) - resident - minimum_tokens


def fits_within(tokens: int, limit: int) -> bool:
    """The shared predicate: the number is within the limit. One comparison — the limit already
    is the smaller of the two ceilings, so there is no second check that can drift from the
    first, and the fit check and every stage's acceptance ask the same question of the same
    number."""
    return tokens <= limit

# A server that refuses a request for its size is the one event compaction exists for, and the only one
# it cannot see coming: `TRIGGER` reads the previous request's usage, and a refused request never gets
# to report any. So the refusal has to be recognised from the server's own words.
#
# Measured versus inferred, because a wrong marker here costs a compaction and a compaction loses
# conversation. The first two were read off a live llama.cpp server (127.0.0.1:8081, HTTP 400 body:
# `request (200013 tokens) exceeds the available context size (16128 tokens)`, `"type":
# "exceed_context_size_error"`, `"n_ctx": 16128`). The rest are the shapes the OpenAI and vLLM
# dialects document — `This model's maximum context length is N tokens`, `reduce the length` — and are
# NOT measured on anything reachable from this machine; they are here because a run may be pointed at
# either, and a marker that matches nothing fails safe by ending `failed`.
#
# Every marker is a phrase that *asserts excess* — it names a limit being exceeded or asks the caller to
# shorten — and that is the invariant the table must hold. The temptation when a run fails to recover is
# to add the substring every such error contains,
# which is "context"; that substring is also in errors about configuration, templates and resume ids,
# and a false positive here is not a harmless retry, it is a summarisation of a conversation that did not
# need one. So `n_ctx` is deliberately NOT a marker even though the measured body carries it: it is the
# name of a request field, and a server rejecting a field's value says the opposite of "you filled the
# window".
OVERFLOW_MARKERS = (
    "exceeds the available context size",   # measured: llama.cpp
    "exceed_context_size_error",            # measured: llama.cpp, the JSON `type`
    "maximum context length",               # inferred: OpenAI dialect, unmeasured here
    "reduce the length",                    # inferred: OpenAI dialect, unmeasured here
    "context length exceeded",              # inferred: vLLM-family wording, unmeasured here
)


def looks_like_overflow(error: str) -> bool:
    """Does this failure read like a server refusing the request because it was too big?

    Judged on the server's words because nothing else says it: the harness never knows the window a
    remote server is running with, and `capabilities()` reports what the operator typed, not what the
    process was started with. An empty or unrelated error returns False, which is the safe answer — the
    run ends `failed` and keeps its conversation."""
    text = str(error).lower()
    return any(marker in text for marker in OVERFLOW_MARKERS)


# Compact when the last request used this share of the context. Deliberately below 1.0: leaving
# headroom is cheaper than recovering from a request the server already rejected.
TRIGGER = 0.82

# Never compact a conversation shorter than this many messages — there is nothing worth dropping,
# and a summary of two turns costs a request to lose information.
MIN_MESSAGES = 6

# How many recent messages always survive compaction. The turn the model is mid-way through and its
# tool results must not be summarised away, or the next request cannot see what it just did.
KEEP_RECENT = 6


CHARS_PER_TOKEN = 4
"""The standing estimate: four characters to a token. Wrong in both directions — it under-counts
code and JSON, over-counts prose — but good enough for the only questions it is asked (is this
window mostly full, is this tail most of it), and the alternative is a tokenizer the harness does
not have. A backend that reports its own count corrects the rate through `Context.calibrate`; every
number derived from the raw estimate is written with a `~` so a reader is not told it is exact."""


def estimate_text(text: str) -> int:
    """The token count of one stretch of text at the standing rate."""
    return len(text) // CHARS_PER_TOKEN


def message_tokens(content) -> int:
    """The token count of a message's content, in whatever shape the backend used it.

    Text is counted at the standing rate. A parts list is counted part by part: a text part by its
text, and an image part by the length of the data URL it puts on the wire — that is what the
request carries. On a vision endpoint the server counts the picture as image tokens, orders of
magnitude fewer than its base64 length, so a message that carries an image is over-counted here;
that is the safe direction for the one question these numbers are asked (is the window full), and
a `Context` calibrated against a server that reports corrects it on the text requests that follow.
"""
    if isinstance(content, str):
        return estimate_text(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" or "text" in part:
                total += estimate_text(str(part.get("text", "")))
            elif part.get("type") == "image_url" or "image_url" in part:
                inner = part.get("image_url")
                url = inner.get("url", "") if isinstance(inner, dict) else str(inner or "")
                total += estimate_text(url)
        return total
    return estimate_text(str(content or ""))


def message_has_image(content) -> bool:
    """Whether this message's content carries an image part.

    The calibration guard: the ratio between a server's count and the run's estimate is stable only
    for text, because an image part is counted by its wire bytes while the server counts it as image
tokens. A ratio learned from a request that carried an image would be wrong for every text request
after it, so image-carrying requests are not taught to (see `Context.calibrate`)."""
    if isinstance(content, list):
        return any(isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part)
                   for part in content)
    return False


def estimate_tokens(messages: list[model.Message]) -> int:
    """A rough token count for a stretch of conversation, for a decision that must not send a
    request to make.

    Per message at the standing rate — text by its characters, image parts by their wire bytes
    (see `message_tokens`). Every number derived from it is written with a `~` so a reader is not
    told it is exact; a backend that reports its own count is the one that corrects it, through
    `Context.calibrate`.
    """
    return sum(message_tokens(m.content) for m in messages)


class Context:
    """How big the next request's context is: the resident parts, plus the conversation.

    The resident parts — the system prompt and the tool schemas — are known at run start and do
    not change during it, so they are counted once, from the exact strings the wire carries. The
    conversation is counted from the messages the run is about to send, not from a cumulative
    counter: a model that is a pure function of the outgoing request cannot drift from it, whether
    compaction replaced the oldest turns with a summary or a resume rebuilt the list, and the two
    sides of every estimate (what the run sent, what the run thinks it sent) cannot disagree because
    there is one of each.

    Every number here is the standing estimate until `calibrate` says otherwise: when a backend
    reports the prompt tokens of a request whose composition the run measured, the ratio between
    the two is the correction that makes the estimate track that server, and `effective` applies
    it. It is learned only from image-free requests — see `message_has_image` for why an image
    would poison it.
    """

    def __init__(self, system: str, specs: list, backend=None) -> None:
        self._system = estimate_text(system)
        self._tools = estimate_text(_specs_wire(specs))
        self.backend = backend
        self.ratio: float | None = None
        """The server's count over this run's estimate, for the last image-free request the server
        counted — or None until a backend has reported a count to learn from. An estimate the
        server has not corrected is the standing rate, and pretending to hold a correction would be
        worse than admitting there is none. `backend` is the run's own: the one it asks for its
        usage, the one a summary call would go through, so calibration and the summary's
        acceptance read the same server's numbers."""

    def resident(self) -> int:
        """The system prompt and the tool schemas: what every request carries beside whatever the
        conversation is. Fixed for the run, counted once, in `__init__`."""
        return self._system + self._tools

    def total(self, messages: list[model.Message]) -> int:
        """The whole next request at the standing rate: the resident parts plus the conversation."""
        return self.resident() + estimate_tokens(messages)

    def effective(self, messages: list[model.Message]) -> int:
        """What the next request will cost on this server, as well as can be known without asking.

        The standing estimate corrected by the server's own rate when it has reported one to learn
        from; the standing estimate bare when it has not. This is the fallback the trigger falls
        back to when a backend reports no usage at all — a backend that says nothing is not a
        backend with an empty context."""
        total = self.total(messages)
        return int(total * self.ratio) if self.ratio else total

    def calibrate(self, local: int, server: int) -> None:
        """Learn this server's rate from one request the run measured and the server counted.

        `local` is what `total` said for the exact request that went out, and `server` what the
        backend reported for it. A local count of zero means the request was empty as far as this
        estimate can see — nothing to measure a rate against — and the ratio of such a request is
        not a rate, so it is not learned.
        """
        if local > 0 and server > 0:
            self.ratio = server / local

    def decomposition(self, messages: list[model.Message]) -> str:
        """The window's composition in the words a notice can carry: system prompt, tool schemas
        and conversation, each with the `~` that says estimate. This is the history-versus-
        system-prompt split, made visible wherever the run has to report on the window's size."""
        return (f"~{self._system} of system prompt, ~{self._tools} of tool schemas and "
                f"~{estimate_tokens(messages)} of conversation")


def _specs_wire(specs: list) -> str:
    """The tool schemas as they travel on the wire, in the shape the adapter builds them
    (`sah.openai`), counted once for the run. The count is of what the request carries, not of
    the functions' source: the model pays for what it is sent."""
    return json.dumps([{"type": "function",
                        "function": {"name": spec.name, "description": spec.description,
                                     "parameters": spec.parameters}} for spec in specs])


def occupancy(prompt_tokens: int, context_tokens: int) -> float:
    """Share of the context the last request used, as a float. The trigger compares against this."""
    if context_tokens <= 0:
        return 0.0
    return prompt_tokens / context_tokens


def should_compact(prompt_tokens: int, context_tokens: int, count: int) -> bool:
    """Is the conversation full enough, and long enough, to be worth summarising?"""
    return (count > MIN_MESSAGES and occupancy(prompt_tokens, context_tokens) >= TRIGGER)


def source(prior: str | None, drop: list[model.Message]) -> str:
    """What a summary is made from: the previous summary merged in, and the whole history being
    dropped, a tool result cut to `TOOL_RESULT_CHARS` and named. No turn is carved out of the
    source — the run's opening turn is in it like the rest — because keeping the goal standing
    after the rewrite is the summariser's job, not a head the harness keeps: the source is the
    whole history, and a summary that hid the opening turn would leave the goal to the
    summariser's guess. The merge is the slot's own consequence: each summary re-summarises the
    last one, so a second compaction sees the first through the source rather than through turns
    the summary replaces. The cap is on the request's side: the summary call carries only its
    system prompt and this source, and a result beyond the cap is cut and named so the summary
    says what was lost rather than arriving at the summariser in full."""
    parts = []
    if prior:
        parts.append(prompts.load("compact_prior", prior=prior))
    lines = []
    for m in drop:
        lines.append(prompts.load("compact_turn", label=_label(m), content=_source_content(m)))
    parts.append(prompts.load("compact_history", history="\n".join(lines)))
    return "\n\n".join(parts)


def _label(m: model.Message) -> str:
    """The word the source's turn line carries: the role, and for an assistant turn that reached
    for tools, which tools — the call is part of what the next turn has to know, and a line that
    says only what the turn said hides the half of it that was the asking."""
    if m.role == "tool":
        return "tool result"
    if m.role == "assistant" and m.tool_calls:
        return ("assistant (wants to call: "
                + ", ".join(name for (_id, name, _args) in m.tool_calls) + ")")
    return m.role


def _source_content(m: model.Message) -> str:
    """The turn's text for the source, a tool result cut to the cap behind a line that says what
    was cut: the summariser is told the result was longer, which is a fact worth keeping, and the
    cut is measured in the characters the line will carry, not a token the harness cannot count."""
    content = _text(m.content)
    if m.role == "tool" and len(content) > TOOL_RESULT_CHARS:
        cut = len(content) - TOOL_RESULT_CHARS
        return content[:TOOL_RESULT_CHARS] + prompts.load("compact_truncated", cut=cut)
    return content


def _request(stage: int, room: int, latest: str | None, source_text: str) -> str:
    """The user message of a summary attempt. Stages 1 and 2 ask the source for a summary, the
    second with the compactness sentence that names the room; stage 3 asks the latest summary
    for a version that fits the room. The compactness prose is a prompt file — model-facing text
    an owner retunes per model — templated with the number: for most models the number does not
    matter and for some it is the whole instruction, so it is named either way, and the code
    carries none of the prose itself."""
    if stage == 3:
        return prompts.load("compact_step3", room=room, text=latest)
    text = prompts.load("compact_step1")
    if stage == 2:
        text += "\n" + prompts.load("compact_step2", room=room)
    return text + "\n\n" + source_text


def _one_call(backend: model.Backend, prompt: str, thinking: str | None) -> tuple[str, int, str]:
    """One bare summary attempt: one user message under the compactor's system prompt, no tool
    schemas, no max_tokens. The server is never given a gate, so nothing on the wire can decide
    whether the answer fits (see `summarise`), and a reply the output limit cut off is a failed
    attempt with its text discarded — a too-long summary is discarded and asked to shrink, never
    accepted partial, because a fragment of a thought standing in for the turns it replaced is a
    compaction that forgot and called itself clean. A call that errors, returns nothing, or stops
    on length yields no text; the error names the event so the record can say what the attempt
    cost. Never raises: a backend that cannot answer is a failed attempt, not a crash in the
    loop. The token count is the server's completion count for the call (0 when it reported
    none) — the cost the attempt paid, beside the size the harness measures for acceptance."""
    prose: list[str] = []
    tokens = 0
    error = ""
    try:
        for event in backend.complete(system=prompts.load("compact_system"),
                                      messages=[model.Message("user", prompt)],
                                      tools=[], thinking=thinking, max_tokens=None):
            if isinstance(event, model.TextDelta):
                prose.append(event.text)
            elif isinstance(event, model.Usage):
                tokens = event.completion_tokens
            elif isinstance(event, model.Stopped) and event.reason == "length":
                error = "the summary hit the output limit before it ended"
            elif isinstance(event, model.Failed):
                error = event.error
    except Exception as exc:                      # a summary call that dies is a failed attempt
        error = f"{type(exc).__name__}: {exc}"
    text = "".join(prose).strip()
    if error:
        return "", tokens, error
    if not text:
        return "", tokens, "the model returned no summary text"
    return text, tokens, ""


def _cut(text: str, room: int) -> str:
    """The last stage: the summary that will not fit, cut to the room and marked, so the model
    is told it now works on potentially missing information — the cut and the note of the cut,
    and the note is the point, because a summary that ends mid-sentence without saying the
    record is lossy reads as complete. The cut is measured against what persists: the prose the
    slot keeps, at the standing rate — the server's count includes thinking the slot never
    holds — and the note is budgeted for before the cut, not after it, so the marked summary
    fits the room the unmarked one would have."""
    note = prompts.load("summary_cut")
    budget = max(room - estimate_text(note), 1)
    chars = budget * CHARS_PER_TOKEN
    if len(text) <= chars:
        return text + "\n" + note
    return text[:chars] + "\n" + note


def summarise(backend: model.Backend, *, source_text: str, room: int,
              thinking: str | None = None, dropped: int = 0) -> tuple[str, int, int, str, str]:
    """Produce the slot's text by the stages, and say which stage paid for it.

    Returns (slot_text, cost, stage, slot, note). `slot_text` is what the privileged slot
    receives — a summary, a cut summary, or the failed note; every return path fills the slot,
    because a compaction that ran must rewrite the conversation, and a slot beside an untouched
    context is the doom the guaranteed slot was designed to end. `cost` is the tokens the
    summary calls cost, the server's count when it reported one. `stage` is which attempt
    produced the slot (1 the free summary, 2 the compactness ask, 3 the re-summarise, 4 the cut
    or the failed note) and `slot` its kind. `note` carries the failed attempts' words for the
    record's reason — the attempts the slot did not come from are part of what the act cost.

    `room` is what `room_for` left the summary (see there for what it subtracted); `thinking` is
    the level every attempt runs at — the run's own by default, left alone rather than
    overridden (see the module docstring), or the level the operator named for summaries.
    Acceptance is this module's measurement, never the server's verdict: the attempt's answer is
    sized by the server's count of the call when it reported one — on a thinking server that
    includes its thinking, the deliberate over-measure the `--summary-think` knob exists to
    answer — else the standing estimate, and it fits when it is within `room`. A room that is not
    positive makes no attempt: the slot takes the failed note, and the record says why.
    """
    cost = 0
    complaints: list[str] = []

    def attempt(stage: int, prompt: str) -> tuple[str | None, int]:
        text, tokens, error = _one_call(backend, prompt, thinking)
        nonlocal cost
        cost += tokens
        if error:
            complaints.append(f"stage {stage}: {error}")
            return None, 0
        return text, tokens

    def fits(text: str, tokens: int) -> bool:
        size = tokens if tokens > 0 else estimate_text(text)
        return fits_within(size, room)

    if room > 0:
        text, tokens = attempt(1, _request(1, room, None, source_text))
        if text is not None and fits(text, tokens):
            return text, cost, 1, "summary", "; ".join(complaints)
        text2, tokens = attempt(2, _request(2, room, None, source_text))
        if text2 is not None and fits(text2, tokens):
            return text2, cost, 2, "summary", "; ".join(complaints)
        latest = text2 if text2 is not None else text
        if latest is not None:
            text3, tokens = attempt(3, _request(3, room, latest, None))
            if text3 is not None and fits(text3, tokens):
                return text3, cost, 3, "summary", "; ".join(complaints)
            return _cut(text3 if text3 is not None else latest, room), cost, 4, "cut", \
                "; ".join(complaints)
    if room <= 0:
        complaints.insert(0, f"no attempt was made: the room is {room} tokens")
    return prompts.load("slot_failed", dropped=dropped), cost, 4, "failed", "; ".join(complaints)


def _text(content) -> str:
    """The text of a message's content, in whichever shape the backend used.

    `Message.content` is declared as text *or* a list of content parts, and `str()` is what a type union
    gets by default: measured, a parts list arrived here as `[{'type': 'text', 'text': 'the first part'},
    ...]`, which is Python quoted at the model inside a summary of the conversation. Text parts are joined;
    an image is named rather than dropped, because a summary that forgets an image was in the window lost a
    fact, and naming it costs one word. Anything else is named by its part type instead of printed."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                out.append(f"[{type(part).__name__}]")
            elif part.get("type") == "text" or "text" in part:
                out.append(str(part.get("text", "")))
            elif "image_url" in part or part.get("type") == "image_url":
                out.append("[image]")
            else:
                out.append(f"[{part.get('type', 'part')}]")
        return " ".join(piece for piece in out if piece)
    return str(content)



# The insert `sah/prompts/summary_frame.md` marks where the summary body goes. Substituted by hand rather
# than passed as a template override, because the frame renders with live host facts, and text a model wrote
# should never enter that pass at all — so the guarantee does not depend on the renderer staying single-pass.
_INSERT = "{{summary}}"


def frame(summary: str, backfill: int = 0, covered: bool = True, cut: int = 0) -> str:
    """A summary in the words a run is told it is: a bounded record of turns that are gone, and,
    when the slot is followed by the verbatim tail, the note that tells the model what follows.
    When the compaction's cut rung dropped recent messages from the tail, the cut note says so —
    the gap between the record and the tail is a gap the model has to see, because it cannot
    steer around information it cannot see it is missing.

    The frame is a prompt file, not a constant, because it is model-facing prose an owner should be
    able to retune; `sah/prompts/summary_frame.md` carries its reasons beside it. `backfill` names
    the tail's length (0 for none), and `covered` picks the note's words — the tail is covered by
    the slot when the slot holds a summary, cut or not, and not when it holds the failed note —
    the repetition must read as orientation, not as new events, and a tail the model believes is
    new history is a conversation that changed shape under it. One helper because the live loop
    and the replay build the same message, and a summary that reads differently on a resumed run
    is a conversation that silently changed shape across a restart."""
    text = prompts.load("summary_frame")
    if _INSERT not in text:
        raise ValueError("sah/prompts/summary_frame.md must contain the {{summary}} insert")
    text = text.replace(_INSERT, summary, 1)
    if cut > 0:
        text += "\n\n" + prompts.load("backfill_cut", n=cut)
    if backfill > 0:
        note = prompts.load("backfill_covered" if covered else "backfill_uncovered", n=backfill)
        return text + "\n\n" + note
    return text


def cut_backfill(keep: list[model.Message], budget: int) -> tuple[list[model.Message], int]:
    """The backfill cut to `budget` tokens: the oldest messages, in whole units, are dropped
    until the tail fits, and the count of what was dropped comes back beside the tail.

    The last rung's guarantee extended to the tail: a compaction that runs rewrites the
    conversation whatever the tail holds, so when the tail itself exceeds the room a compaction
    must land within, the tail is cut rather than the run declared blocked. A unit is a message,
    or an assistant turn with tool calls and every tool result that answers it: dropping a unit
    never leaves a result whose call was dropped (an orphan a server rejects) nor a call whose
    results were dropped (a turn the model believes it made that the record half-keeps). The
    head of the backfill is never a tool result — `split` walked the cut back past them — so
    the first unit starts at a whole turn and every drop after it starts where a unit ends.
    With one unit the cut is all or nothing, and a budget even the emptied tail cannot satisfy
    is the caller's to declare: it says the window cannot hold the resident load, which is the
    run's declared ending, not a cut the rung can make."""
    if not keep:
        return keep, 0
    total = estimate_tokens(keep)
    if total <= budget:
        return keep, 0
    i = 0
    while total > budget and i < len(keep):
        end = i
        if keep[i].role == "assistant" and keep[i].tool_calls:
            while end + 1 < len(keep) and keep[end + 1].role == "tool":
                end += 1
        total -= estimate_tokens(keep[i:end + 1])
        i = end + 1
    return list(keep[i:]), i


def split(messages: list[model.Message], keep_recent: int = KEEP_RECENT):
    """(to_summarise, to_keep). No message is privileged: the run's opening turn is in the drop
    like the rest, because keeping the goal standing after the rewrite is the summariser's job
    over the whole history, not a head the harness stands aside (see the module docstring). The
    most recent `keep_recent` messages survive verbatim so the next request still sees the work
    in progress; a message pair (assistant tool-call + its tool results) is never cut in half —
    dropping a call's results leaves the server with an orphan it rejects."""
    if len(messages) <= keep_recent:
        return [], list(messages)
    cut = len(messages) - keep_recent
    # Never leave a role="tool" result whose assistant tool-call was summarised away: walk the cut
    # back until it sits before a non-tool message, or to the start, where the drop is empty and
    # the safe answer is to summarise nothing at all.
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    return list(messages[:cut]), list(messages[cut:])
