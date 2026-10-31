"""sah.model — the backend seam: the message list that goes out, the events that come back.

The harness knows nothing about which server answers. One object with `complete()` and
`capabilities()` is a backend — structural, so there is no class hierarchy and nothing to
register. The adapter never decides policy: it produces events in arrival order and never
retries, never merges, never reorders, and never turns a failure into an empty turn.

`Fake` lives here, beside the seam it scripts, because the loop needs a scripted turn to be
developed against at all, and a scripted backend kept outside the source tends to drift from the
one that ships.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sah.record import Refused

# Roles the OpenAI wire format accepts. Checked before a request is built, so a malformed
# conversation is refused rather than sent to a server to be confused by.
OPENAI_ROLES = ("system", "user", "assistant", "tool")

# The word for "I have no preference about reasoning". It is the harness's own, not a server's: the
# adapter sends it as a JSON null, the one spelling measured to mean "no opinion" rather than HTTP 400.
# `--think none` is the operator's spelling of it, and `thinking=None` is what the loop passes when the
# operator gave no value at all, so the two agree on the wire.
REASONING_NONE = "none"



@dataclass(frozen=True)
class Message:
    """One entry of the message list. `content` is text, or a list of content parts for a
    multimodal turn. `tool_calls` is the assistant turn's structured calls — a tuple of
    (id, name, arguments) — and `tool_call_id` links a role="tool" result to the call it
    answers. `reasoning` is an assistant turn's reasoning, sent back verbatim into
    `reasoning_field` — the wire field the server used to deliver it, so the re-send speaks
    the dialect the turn came in on; the backend's template decides the reasoning's use."""

    role: str
    content: str | list
    tool_calls: tuple | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    reasoning_field: str = "reasoning_content"


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON-schema dict passed verbatim: the
    harness does not rewrite schemas, and derives them from the function, not from a table."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# --- events, in arrival order -------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the assistant's visible text, or of a tool's live output.

    `call_id` is the tool-call id when the fragment is streaming *tool* output and None when it
    is model prose. One chunk type carries both, so a consumer routes a fragment to the right
    place by `call_id` rather than by a second event type — a tool that does not stream simply
    never emits one."""

    text: str
    call_id: str | None = None


@dataclass(frozen=True)
class ReasoningDelta:
    """A fragment of the model's reasoning. Surfaced, never interpreted.

    `field` is the wire field the server carried it in: the re-send writes back into that field,
    not a fixed one, so a turn that arrived in `reasoning` goes out in `reasoning`."""

    text: str
    field: str = "reasoning_content"


@dataclass(frozen=True)
class ToolCall:
    """A complete tool call: `id` as the backend assigned it ("" if none), `arguments` a
    parsed JSON object."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    """Token accounting exactly as the backend reported it.

    `cached` is how many of `prompt_tokens` the server says it did not re-embed. It is the server's
    number, never an estimate made here, and it stays `None` when the server says nothing. Measured on
    one server, the same request took 705 ms to its first token cold and 16 ms once its whole prompt was
    cached — a difference nobody else in the record can see — and writing `0` for a server that never
    answered would make every cold run look like a warm one that missed. Absence is the honest value,
    and the record treats it that way: `tokens_cached` is an optional integer, and an optional field
    carrying `None` is refused.
    """

    prompt_tokens: int
    completion_tokens: int
    cached: int | None = None
    tps: float | None = None
    prefill_tps: float | None = None


@dataclass(frozen=True)
class Stopped:
    """The turn ended. `reason` is the backend's finish reason verbatim."""

    reason: str


@dataclass(frozen=True)
class Failed:
    """Transport or protocol failure. The loop decides policy; the adapter never retries and
    never swallows this into an empty turn.

    `truncated` says the *output limit* broke a tool call mid-arguments, rather than the model
    sending a malformed one. The distinction is the loop's to act on: a call cut off by the limit is
    not run and the model is asked for a smaller one, while a genuinely malformed call is an error to
    report. The adapter cannot know the policy, so it reports only which of the two it saw.
    """

    error: str
    truncated: bool = False


Event = TextDelta | ReasoningDelta | ToolCall | Usage | Stopped | Failed


@dataclass(frozen=True)
class Caps:
    """Declared capabilities. Read once, trusted. The context window is the one value the CLI may
    probe for when the operator declared none (`sah.openai.probe_context`, a boot act beside the
    construction, never in it — a probe at construction would make a dead backend a startup
    blocker); everything else here is the operator's declaration. Every field here is a decision
    something reads; a capability that is only ever written is a claim about a backend that nothing
    keeps true: the declaration goes on being edited while the behaviour it described moves away
    from it."""

    context_tokens: int
    model: str = ""                                   # which model answers, for the record
    supports_vision: bool = False


@dataclass(frozen=True)
class Call:
    """What the loop asked the backend for. Recorded so a replay can show the request that
    produced a turn, not only the turn."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[str, ...]
    thinking: str
    max_tokens: int | None


def check_roles(messages) -> None:
    """Refuse a message list whose roles the wire format does not have, before anything is
    built or sent. The reason names the accepted set."""
    for m in messages:
        if m.role not in OPENAI_ROLES:
            raise Refused(f"message role {m.role!r} is not an OpenAI role "
                          f"({', '.join(OPENAI_ROLES)}); the request is not sent")


@runtime_checkable
class Backend(Protocol):
    """The seam. Structural: any object with these two methods is a backend."""

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        thinking: str = "off",
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        """Produce the next assistant turn as events in arrival order. Must not raise for a
        backend failure — that is a `Failed` event — and must not reorder, merge or drop a
        `ToolCall`."""
        ...

    def capabilities(self) -> Caps:
        """Declared capabilities."""
        ...


def turn_text(*chunks: str) -> list[Event]:
    """A scripted plain turn: the chunks as deltas, then a clean stop."""
    return [*[TextDelta(c) for c in chunks], Stopped("stop")]


def turn_tool(call_id: str, name: str, arguments: dict, then: str = "") -> list[Event]:
    """A scripted turn that calls a tool, optionally followed by prose."""
    events: list[Event] = [ToolCall(call_id, name, arguments)]
    if then:
        events.append(TextDelta(then))
    events.append(Stopped("tool_calls"))
    return events


class Fake:
    """A scripted backend. Each `complete()` consumes one scripted turn; the reasoning and
    request shape are recorded so a caller can see what went out, not just what came back.

    An item of a turn may be an exception instance, which `_stream` raises when it reaches that point.
    A backend that dies mid-stream is a shape several guarantees are written against, and without
    this the caller hand-rolls a generator class to produce it."""

    def __init__(self, turns: list[list[Event]] | None = None, caps: Caps | None = None):
        self.turns = list(turns or [])
        self.caps = caps or Caps(context_tokens=8192)
        self.calls: list[Call] = []

    def capabilities(self) -> Caps:
        """The scripted backend's declared capabilities."""
        return self.caps

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        thinking: str = "off",
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        """Consume one scripted turn: record the request shape, validate the roles, and stream the
        turn's events (raising a scripted exception at its position, if the turn has one)."""
        # Validated here, not inside the generator: a request the wire format cannot carry is
        # refused when it is made, not when it is first pulled.
        check_roles(messages)
        self.calls.append(Call(system=system, messages=tuple(messages),
                               tools=tuple(t.name for t in tools), thinking=thinking,
                               max_tokens=max_tokens))
        return self._stream()

    def _stream(self) -> Iterator[Event]:
        if not self.turns:
            yield Failed("no scripted turn left")
            return
        for event in self.turns.pop(0):
            if isinstance(event, BaseException):
                # Raised at its position, after whatever preceded it: what a caller does next depends on
                # what already arrived, which is the whole content of those tests.
                raise event
            yield event
