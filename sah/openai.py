"""sah.openai — the backend for an OpenAI-compatible endpoint.

Everything vendor-shaped lives here so the rest of the harness never learns that HTTP, SSE and
`reasoning_content` exist. The seam is `model.Backend`: two methods, `capabilities()` and
`complete()`.

Invariants

- **Tool-call deltas accumulate by index and are emitted only when whole.** Servers split
  `arguments` across chunks; a partial call is not an event. Holding a call until it is
  complete is what "never drop a ToolCall" means mechanically, and the loop's stop condition
  depends on it. Emission order at flush is by index.
- **`complete()` never raises and never yields an empty turn.** A connection refused, an HTTP
  error and a socket that dies mid-stream each produce exactly one `Failed`, after whatever was
  already yielded. A message list the wire format cannot carry is the one exception: that is a
  contract violation of the harness, not a backend failure, so it is raised before anything is
  sent.
- **Capabilities are declared; the context window is probed only when the operator declared none.**
  A probe at construction would make starting a session depend on a warm GPU, so the probe is not
  in the constructor: `probe_context` is a boot act the CLI runs when `--context` is unset, and it
  fails soft — a backend that declares no window, or is unreachable, leaves the 32768 default in
  force with a notice, because the probe's answer is a correction for the operator, never a
  startup blocker. It sends no completion, so it cannot displace the server's prefix cache.
- **No retries.** Retry is a loop policy, and a loop that retries a POST whose side effects
  already happened is worse than one that reports the break.
- **Nothing in the message list is rewritten.** `system` is prepended; assistant `tool_calls`,
  `tool_call_id` and reasoning are serialised in the shape the server's dialect expects: the re-send
  mirrors the field the turn came in on, an operator-named `reasoning_out` overrides the mirror for
  the asymmetric servers, and the `require_reasoning_content` pin keeps the key present for the
  servers that validate it. The
  parse core is a pure function of its wire shape — the SSE line stream, or the one non-streaming
  body mapped onto the chunk shape — no transport, no clock, no config inside.
- **An empty `arguments` string is `{}`; truncated JSON or a non-object is a `Failed` naming
  the tool.** A silently dropped call is indistinguishable from a model that asked for nothing.

Decisions

- **Streaming is the default and is a flag, and so is the usage request.** Some servers cannot
  stream at all, and a single-body round trip is its own ask, so `--no-streaming` POSTs the same
  body with `stream: false` and reads one JSON body instead of an SSE line stream. The body is
  mapped onto the chunk shape the SSE parse core reads — a `message` is the one whole `delta`, and a
  tool call without the streaming-only `index` is given one — so the parse core stays one and a
  whole tool call still passes through `_tool_call_from_delta`. The `stream_options` usage request
  is on by default (the compactor calibrates on it and most backends carry it) and `--no-usage`
  omits the field; it rides on streaming, because the spelling is the usage *in the stream* — a
  non-streaming request carrying it is refused by the dialect itself, and a non-streaming answer
  reports its usage in the body regardless.
- **The backend's extras are dropped, and here is the list.** `usage` is parsed into named
  fields (`prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`,
  `completion_tokens_per_second`, `prompt_tokens_per_second`); the rest of a `usage` object
  (`total_tokens`, a draft-decoder's `draft_tokens`, one measured server's whole `timings` block)
  and a chunk's other top-level fields (`timings`, `id`, `created`, `model`, `object`) and choice fields
  beyond `index`/`delta`/`finish_reason` (one server's `eos_reason`, for example) are not surfaced
  anywhere, because the step schema has no field for them and an untyped bag is not a record. To keep
  one, add a named field to `model.Usage` and map it here — do not add a dict.
- **`thinking` names a level, or names nothing, and naming nothing is the default.** Naming nothing
  omits the field: `reasoning_effort` is on the wire only when the operator names a level. The absence
  is the dialect-safe spelling — a server that predates reasoning levels (or validates its body
  strictly) sees a body it can parse in full, and a template-driven server takes the default branch it
  took before the field existed; an explicit value, even a measured-accepted `null`, is a field a
  dialect may not declare, and a template that tests `is defined` reads it as a value. `on` is the
  operator's word for "reason, at whatever level this server defaults to", and it is the omission, not
  a level: a level the harness named would be a guess at the backend's vocabulary, and one measured
  template 400s the OpenAI spelling `high` (its closed set is `xhigh`, `medium`, `low`). Any other
  word (`"off"`, `"low"`, `"xhigh"`, …) is the server's own vocabulary and passes through verbatim.
  `off` is a level, not "no preference": the spec-faithful enum (`none, minimal, low, medium, high,
  xhigh, max`) has no such value, and the server measured here 400s the empty string that would mean
  it. So the operator may ask for `off` and get a silent model.

  What is *not* sent is a guess at the vocabulary, and one measured consequence is worth keeping: on the
  server measured here a low level finishes with `stop` inside a budget a reasoning request would hit
  `length` in. So `finish_reason: "length"` is a signal that fires for one level and not another, and no
  caller may read "the server said `stop`" as "the answer is whole". The harness overrides the
  operator's level in one place only — `compact.FALLBACK`, which is what a dialect without `off` is
  configured around, and a refusal there costs a rung rather than the compaction.
- **An HTTP error body is carried into the reason.** "HTTP 400" without the server's message
  sends the reader to the server's console, which a harness run should not require.
- **The wire capture is a debug artifact, not the record.** `--wire` asks the adapter to append the
  composed request body, one JSON line per request, to a run-local file. It is the one layer that
  answers "did the harness actually compose this correctly?" — the fingerprint in the log pins what
  the harness *had*, and the capture shows what the adapter *built* from it. It records the body,
  never the headers, so the API key (which travels in an `Authorization` header) is never in it, and
  it is written before the POST, so a refused request is captured too — which is the moment it is
  most needed. A failure to write it does not stop the run: the capture is a convenience, and the run
  exists to talk to the server, not to fill a debug file.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Iterator

from sah.model import (Caps, Event, Failed, Message, REASONING_NONE, ReasoningDelta, Refused,
                       Stopped, TextDelta, ToolCall, ToolSpec, Usage, check_roles)

# The stream's own structure; every other top-level chunk field is metadata about it.
# The environment this adapter reads a key from when no key was passed. Named here because two modules
# depend on the same spelling: this one reads it, and `sah.spawn` writes it into a child's environment
# rather than putting a secret on the child's command line. The spelling is a shared contract between
# the two — a rename that would silently cost a spawned run its key is what the constant exists to prevent.
API_KEY_ENV = "SAH_API_KEY"

def _build_opener():
    """The one door a request leaves through, with the environment's proxies removed.

    Supplying a `ProxyHandler` — even an empty one — is what keeps the *default* proxy handler out:
    `build_opener` skips any default class the caller already provided, and the default is the one that
    reads `HTTP_PROXY`/`HTTPS_PROXY`. The empty handler itself is then dropped by `add_handler`, which
    keeps only handlers contributing a `<protocol>_open` method, so the proof is in the http dispatch
    chain rather than in the handler list.

    Without it, a machine with a corporate proxy configured sends every prompt and tool result of a run
    against a local server to that proxy, on its way to a LAN address the proxy frequently cannot reach.
    A proxy between the harness and its own backend is the operator's decision, not an inheritance."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER = _build_opener()

_STREAM_FIELDS = frozenset({"choices"})
_CHOICE_FIELDS = frozenset({"index", "delta", "finish_reason", "logprobs"})


# --- the parse core: a pure function of a line stream --------------------------------

def _tool_call_from_delta(accumulated: dict, allowed: frozenset | None = None) -> Event:
    """Accumulated fragments → one `ToolCall`, or a `Failed` naming the malformation."""
    name = accumulated.get("name", "")
    if not name:
        return Failed("tool call with no function name: the server sent arguments with no name")
    if allowed is not None and name not in allowed:
        return Failed(f"unknown tool {name!r}: not among the tools declared for this turn")
    raw = accumulated.get("arguments", "")
    if not raw:
        parsed: Any = {}
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return Failed(f"malformed tool call arguments for {name!r}: {exc}; raw: {raw[:200]!r}")
    if not isinstance(parsed, dict):
        return Failed(f"tool call arguments for {name!r} are {type(parsed).__name__}, "
                      "not a JSON object")
    return ToolCall(accumulated.get("id", ""), name, parsed)


def _cached_tokens(usage: dict):
    """How many prompt tokens the server says it did not re-embed, or `None` if it did not say.

    Read from `usage.prompt_tokens_details.cached_tokens`, the spelling both measured servers answer
    with: a live llama.cpp server on this box reported `cached_tokens: 328` of 333 prompt tokens on its
    second identical request, and a live tabbyAPI server reported the count there *and* as a top-level
    `cached_tokens`, the two agreeing on every request (0/0 cold, 373/373 warm). One key is read, not
    two, so the two spellings cannot disagree inside this harness.

    A `details` that is not a mapping, or a count that is not a whole number, means "not said" and not
    "a malformed chunk": this is a number worth showing and not one worth failing a decode over.
    `record` declares the field an integer, so a fractional or boolean count is refused here rather than
    at the step, which keeps the two ends agreeing.
    """
    details = usage.get("prompt_tokens_details")
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _rate(usage: dict, key: str):
    """A tokens-per-second count the server measured, as a float, or `None` if it did not say.

    Two shapes need handling: the integer a server may send where the record's field is a float, and the
    string or absent key that is not a measurement. `record` refuses an integer in a float field, and a
    rate that arrives wrong there costs the whole `message` step rather than the number, so the
    conversion happens here.

    A zero passes through. It is what a server reports when it did no work at all: measured against a
    live tabbyAPI server, the request whose whole 373-token context was served from the prompt cache
    answered `prompt_tokens_per_second: 0` — an integer, where the cold request answered `571.93` — with
    `prefill_ms: 0` and a time-to-first-token of 16 ms instead of 705 ms. Deciding that a zero is not
    news belongs to whoever prints it, not to the adapter.
    """
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _chunk_events(chunk: dict, calls: dict, allowed: frozenset | None) -> tuple[list, bool]:
    """One decoded chunk → (events in arrival order, finish seen). Mutates `calls`, which
    holds the fragments accumulated by index."""
    events: list = []
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        events.append(Usage(int(usage.get("prompt_tokens") or 0),
                            int(usage.get("completion_tokens") or 0),
                            cached=_cached_tokens(usage),
                            tps=_rate(usage, "completion_tokens_per_second"),
                            prefill_tps=_rate(usage, "prompt_tokens_per_second")))
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return events, False
    choice = choices[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str) and content:
        events.append(TextDelta(content))
    # The field a server uses to carry reasoning varies by dialect: llama.cpp's Qwen builds use
    # `reasoning_content`, OpenAI-style servers `reasoning`, and some gateways `reasoning_text`. If
    # several carry it at once, the first non-empty one in this order wins: some endpoints duplicate
    # the same text into two of them, and the winner names the field the re-send writes back into.
    for field in ("reasoning_content", "reasoning", "reasoning_text"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            events.append(ReasoningDelta(value, field))
            break
    for fragment in delta.get("tool_calls") or []:
        if not isinstance(fragment, dict):
            continue
        index = int(fragment.get("index", 0))
        held = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if isinstance(fragment.get("id"), str):
            held["id"] = fragment["id"]
        function = fragment.get("function") or {}
        if isinstance(function.get("name"), str):
            held["name"] += function["name"]
        if isinstance(function.get("arguments"), str):
            held["arguments"] += function["arguments"]
    reason = choice.get("finish_reason")
    if reason is not None:
        for index in sorted(calls):
            events.append(_tool_call_from_delta(calls.pop(index), allowed))
        if str(reason) == "length":
            # A turn stopped by the output limit leaves a call holding half a JSON document, which fails
            # to parse. Saying so is the whole point: "malformed arguments" and "the budget ran out
            # mid-call" need different answers from the loop, and only the adapter can tell them apart.
            events = [Failed(e.error, truncated=True) if isinstance(e, Failed) else e for e in events]
        events.append(Stopped(str(reason)))       # the server's reason, verbatim
        return events, True
    return events, False


def parse_sse(lines, allowed: frozenset | None = None) -> Iterator[Event]:
    """An SSE line stream → the turn's events. `allowed` (the declared tool names) turns a
    name the harness never offered into a `Failed` rather than a call nothing will run."""
    calls: dict[int, dict] = {}
    finished = False
    for line in lines:
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue                             # blank line, keep-alive comment, other field
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as exc:
            yield Failed(f"malformed SSE data line: {exc}; {payload[:200]!r}")
            continue
        if not isinstance(chunk, dict):
            yield Failed(f"SSE data line is not a JSON object: {payload[:200]!r}")
            continue
        try:
            events, done = _chunk_events(chunk, calls, allowed)
        except (TypeError, ValueError) as exc:   # a field of a type the format does not allow
            yield Failed(f"malformed chunk: {exc}; {payload[:200]!r}")
            continue
        if done:
            finished = True
        yield from events
    if not finished:
        # The stream ended or said [DONE] with no finish_reason. Flush what accumulated — a
        # dropped call is indistinguishable from a model that made none — then name the break.
        for index in sorted(calls):
            yield _tool_call_from_delta(calls[index], allowed)
        calls.clear()
        yield Failed("the stream ended without a finish reason")


def parse_body(raw: bytes, allowed: frozenset | None = None) -> Iterator[Event]:
    """A non-streaming response body → the turn's events.

    The body is mapped onto the chunk shape the SSE parse core reads — a `message` is the one whole
    `delta`, and a tool call without the streaming-only `index` is given one — so one parse core
    serves both transports: a whole tool call still passes through `_tool_call_from_delta`, and
    truncated arguments under `finish_reason: length` still fail the way the streaming path fails
    them, naming the tool. A body the format cannot carry is a `Failed` rather than a crash, and a
    `usage` the body carries is yielded before the break is named, the way the streaming path
    yields it: a broken answer that still measured itself has fed the calibration.
    """
    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        yield Failed(f"malformed response body: {exc}; {raw[:200]!r}")
        return
    if not isinstance(body, dict):
        yield Failed(f"response body is not a JSON object: {raw[:200]!r}")
        return
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    message = choice.get("message") if choice is not None else None
    if not isinstance(message, dict):
        yield Failed("the response body carries no choices[0].message: a server that cannot answer "
                     "in this shape is a different backend, not a transport the harness can dress up")
        return
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        # The `index` a call accumulates under is a streaming artifact — the fragment's position in
        # the array — and a whole answer has no position but its order. Give each call one the
        # streaming core can accumulate by, honouring an index a server did send.
        indexed = []
        for position, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            call = dict(call)
            call.setdefault("index", position)
            indexed.append(call)
        message = dict(message)
        message["tool_calls"] = indexed
    chunk = {"usage": body.get("usage"),
             "choices": [{"index": 0, "delta": message,
                          "finish_reason": choice.get("finish_reason")}]}
    calls: dict[int, dict] = {}
    try:
        events, done = _chunk_events(chunk, calls, allowed)
    except (TypeError, ValueError) as exc:   # a field of a type the format does not allow
        yield Failed(f"malformed response body: {exc}; {raw[:200]!r}")
        return
    yield from events
    if not done:
        # No finish_reason: flush what accumulated the way the streaming path does, then name the
        # break — a dropped call is indistinguishable from a model that made none.
        for index in sorted(calls):
            yield _tool_call_from_delta(calls[index], allowed)
        yield Failed("the response body carried no finish reason")


def sse_lines(response) -> Iterator[str]:
    """SSE lines from an open response, decoded with replacement: a broken byte is a
    malformation to surface, not a crash. A module function so the transport can be stood in without
    colliding with an attribute the caller wants to name."""
    while True:
        raw = response.readline()
        if not raw:
            return
        yield raw.decode("utf-8", "replace").rstrip("\r\n")



# --- the transport ------------------------------------------------------------------

def _authority(host: str) -> str:
    """A host as a URL authority wants it. An IPv6 literal is bracketed, because a bare colon is a
    delimiter there: `http://::1:8080/` is not a host and a port but an unparseable target that fails
    with a message about the URL rather than about the address. One already bracketed is not doubled."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _derive_base_url(host: str, port: int, base_url: str) -> str:
    """The base a request hangs off: the operator's base when named, else http://host:port/v1.

    The base owns the scheme, the host and any prefix; `api_path` is what hangs off it. Naming
    only host and port keeps the local case exactly what it was, and naming a base is the only
    way to reach a server behind a gateway path."""
    base = (base_url.rstrip("/") if base_url.strip()
            else f"http://{_authority(host)}:{int(port)}/v1")
    if not base.startswith(("http://", "https://")):
        raise Refused(f"base_url must be http(s), not {base!r}: this backend streams "
                      f"OpenAI-dialect SSE over one HTTP POST")
    return base


def _refuse_images(message) -> None:
    """An image addressed to a server that was never declared able to see is refused here, not there.

    The cost of getting this wrong is not the refusal, it is the upload: a base64 image crossing the wire
    and coming back as someone else's HTTP 400, after the window has already paid for the bytes. This is
    the only reader of `supports_vision`, which is what makes it a capability rather than a decoration."""
    content = message.content
    if isinstance(content, list) and any(isinstance(part, dict)
                                         and ("image_url" in part or part.get("type") == "image_url")
                                         for part in content):
        raise Refused("a message carries an image part and this endpoint is declared unable to see; "
                      "run against a vision-capable model (the operator says so with --vision)")


class OpenAI:
    """The backend for one OpenAI-compatible endpoint. Values come from the caller — the CLI,
    an environment — never from a config module the loop would have to know about."""

    def __init__(self, *, model: str, host: str = "127.0.0.1", port: int = 8080,
                 base_url: str = "", api_path: str = "/chat/completions", api_key: str = "",
                 context_tokens: int = 32768, supports_vision: bool = False,
                 preserve_thinking: str = "none", require_reasoning_content: bool = False,
                 reasoning_out: str | None = None, streaming: bool = True,
                 request_usage: bool = True, wire: str | None = None, timeout: float = 600.0):
        self.model = model
        self.host = host
        self.port = int(port)
        # The base owns the scheme, the host and any prefix; `api_path` is what hangs off it. Naming only
        # host and port keeps the local case exactly what it was, and naming a base is the only way to
        # reach a server behind a gateway path — without one the operator's alternative is a rewrite
        # outside the harness, which puts the conversation somewhere the run never chose.
        self.base_url = _derive_base_url(host, self.port, base_url)
        self.api_path = api_path
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        self.context_tokens = int(context_tokens)
        self.supports_vision = bool(supports_vision)
        if preserve_thinking not in ("none", "true", "false"):
            raise Refused(f"preserve_thinking must be none/true/false, not {preserve_thinking!r}: "
                           "none sends no template kwarg (the template's own default governs the "
                           "rendering), true/false send an explicit preserve_thinking kwarg")
        self.preserve_thinking = preserve_thinking
        self.require_reasoning_content = bool(require_reasoning_content)
        # The field a server reads the re-send in, when it differs from the one it delivers in: the
        # name is the operator's declaration about the server, so it is refused where it is named if
        # it is not the shape a dialect's own field name has, and a request can never carry a field
        # name typed by anything the operator did not type.
        if reasoning_out is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reasoning_out):
            raise Refused(f"reasoning_out must look like a dialect field name, not {reasoning_out!r}")
        self.reasoning_out = reasoning_out
        self.streaming = bool(streaming)
        """Stream the answer (the default), or take the single-body round trip: the payload says
        `stream: false` and the response is one JSON body read through `parse_body` instead of an
        SSE line stream. The one knob a server's dialect may refuse outright — some servers cannot
        stream at all — which is why it is a flag and not a fact of the class."""
        self.request_usage = bool(request_usage)
        """Ask the server for usage accounting (the default): on the streaming wire that is the
        `stream_options.include_usage` request, and a `--no-usage` run omits the field a server's
        dialect may not declare. It rides on `streaming`: a non-streaming request carrying it is
        refused by the dialect, and a non-streaming answer reports its usage in the body regardless."""
        self.wire_path = wire
        """The wire capture path, or None. When set, `complete` appends the composed request body here
        (one JSON line per request, before the POST). A debug artifact, not the record — see the module
        docstring for what it is, what it deliberately omits, and why a write failure must not stop a
        run."""
        self.timeout = float(timeout)

    def request_url(self) -> str:
        """The full URL a request is POSTed to: the base plus the api path."""
        return f"{self.base_url}{self.api_path}"

    @property
    def key_in_clear_to_remote(self) -> bool:
        """A key is going out unencrypted to a host that is not this machine.

        Ordinary on a trusted LAN and wrong across the internet, so the harness says it rather than
        refusing it: the local-network case is the common one, and the operator named the host. What it
        must not do is let a run that works be read as evidence that nothing was exposed."""
        if not self.api_key or not self.base_url.startswith("http://"):
            return False
        target = urllib.parse.urlparse(self.base_url).hostname or ""
        return not (target == "localhost" or target.startswith("127.")
                    or target.strip("[]") in ("::1", "0:0:0:0:0:0:0:1"))

    @property
    def accept(self) -> str:
        """The `Accept` header for this transport: the SSE media type while streaming, the plain
        JSON type otherwise — the one header a gateway may key on before the body is read, so it
        must say what the body actually is."""
        return "text/event-stream" if self.streaming else "application/json"

    def capabilities(self) -> Caps:
        """What this endpoint says about itself, declared by whoever configured it.

        `supports_vision` is the operator's `--vision`, and `build_payload` reads it: the declaration has
        to be consulted *before* the bytes go, which is the one thing a route that sends images has to get
        right. Nothing else here is declared that nothing reads — what the adapter does (stream, speak
        OpenAI tool calls) is a fact of this class, and a field asserting it could only ever agree with the
        code beside it or lie."""
        return Caps(context_tokens=self.context_tokens, model=self.model,
                    supports_vision=self.supports_vision)

    def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec],
                 thinking: str = "off", max_tokens: int | None = None) -> Iterator[Event]:
        """Run one turn and yield its events. A malformed message list is raised (a contract
        violation, not a backend failure); the wire capture, if wired, is written before the POST; a
        refused/HTTP/socket break becomes a single `Failed`, never a raise."""
        check_roles(messages)                    # a contract violation is raised, not yielded
        payload = self.build_payload(system, messages, tools, thinking, max_tokens)
        self._write_wire(payload)                # before the POST: the capture survives a refusal
        allowed = frozenset(spec.name for spec in tools)
        try:
            with self.open(payload) as response:
                if self.streaming:
                    yield from parse_sse(sse_lines(response), allowed)
                else:
                    yield from parse_body(response.read(), allowed)
        except Refused:
            raise
        except Exception as exc:                 # refused, HTTP error, socket death
            detail = f" (HTTP {exc.code})" if getattr(exc, "code", None) is not None else ""
            yield Failed(f"transport: {type(exc).__name__}{detail}: {exc}")

    # --- the request ---------------------------------------------------------

    def build_payload(self, system: str, messages: list[Message], tools: list[ToolSpec],
                      thinking, max_tokens) -> dict:
        """The request body. `system` is the first message; nothing else in the list is
        rewritten."""
        out = [{"role": "system", "content": system}]
        for message in messages:
            if not self.supports_vision:
                _refuse_images(message)
            entry: dict = {"role": message.role, "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = [{"id": call_id, "type": "function",
                                        "function": {"name": name,
                                                     "arguments": json.dumps(args, sort_keys=True)}}
                                       for call_id, name, args in message.tool_calls]
            if message.tool_call_id is not None:
                entry["tool_call_id"] = message.tool_call_id
            if message.reasoning:
                # The turn's own reasoning always travels back to the backend, whatever the template
                # does with it: an interleaved-thinking dialect expects its prior reasoning still in
                # the conversation when it answers across a tool boundary, and withholding it would
                # let the harness decide, unilaterally, what a server's template renders. What
                # travels is the model's own text — a fact about the conversation; whether the
                # template renders it into the prompt is the separate decision the
                # `chat_template_kwargs` below may express, and `none` declines to express it.
                # And it travels in the field the server used to deliver it — the turn's signature,
                # captured at parse: a dialect that returned `reasoning` expects `reasoning`, and a
                # fixed name would silently drop the echo on every such server. The signature can
                # only be one of the parse's recognized fields or the de facto default, never server
                # prose, so the request cannot carry a field name the dialect does not know. An
                # operator-named `reasoning_out` overrides the mirror for the asymmetric servers —
                # a server that delivers in one field and reads another — and the override is as
                # safe as the mirror: the name is the operator's own, checked at construction.
                entry[self.reasoning_out or message.reasoning_field] = message.reasoning
            if (message.role == "assistant" and self.require_reasoning_content
                    and "reasoning_content" not in entry):
                # Some servers (DeepSeek's API among them) validate the key's presence on every
                # assistant message, or look it up in the template without a defined-check: the turn
                # that never reasoned is then the turn that breaks the request, so the key is pinned
                # empty rather than absent. The pin is the operator's declaration about the server,
                # never a guess — off by default, so a server whose template only checks definedness
                # never sees an empty think marker it was not asked for.
                entry["reasoning_content"] = ""
            out.append(entry)
        payload: dict = {"model": self.model, "messages": out, "stream": self.streaming}
        if self.streaming and self.request_usage:
            # `stream_options` is the OpenAI spelling of the usage request on the stream, and it is
            # on by default: the compactor calibrates on the counts and most backends carry it. It
            # rides on streaming because the spelling is stream-bound — a non-streaming request
            # carrying it is refused by the dialect itself — and `--no-usage` omits the field,
            # the way a named level is omitted: a field a dialect may not declare is not one the
            # default puts on the wire.
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = [{"type": "function",
                                 "function": {"name": spec.name, "description": spec.description,
                                              "parameters": spec.parameters}} for spec in tools]
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        # The level is the operator's own word, sent as typed, `off` included — an operator asking for
        # a silent model gets a silent model, and a dialect with no `off` is theirs to configure around
        # (a refusal there is a Failed turn carrying the server's own message, or one lost rung inside
        # compaction). `none`/None/`on` name no level and omit the field: the absence is the one
        # spelling a server whose dialect does not declare the field cannot object to, and the empty
        # string that looks like the cheaper way to name nothing answered HTTP 400 where measured.
        if thinking is not None and str(thinking).strip().lower() not in (REASONING_NONE, "on", "true"):
            # The field goes on the wire only when a level is named: naming nothing must not put a field
            # on the wire a server's dialect may not declare (a pre-reasoning server, a strict
            # validator, a template that tests `is defined`), and a harness-chosen level would be a
            # guess at that vocabulary — one measured template 400s the OpenAI spelling `high`.
            payload["reasoning_effort"] = str(thinking)
        # Rendering is the operator's opinion on the template, not on the conversation: `none` sends no
        # kwarg and the template's own default decides whether the reasoning that did travel is rendered
        # into the prompt; `true`/`false` say so explicitly.
        if self.preserve_thinking in ("true", "false"):
            payload["chat_template_kwargs"] = {"preserve_thinking": self.preserve_thinking == "true"}
        return payload

    def _write_wire(self, payload: dict) -> None:
        """Append the composed request to the wire capture, if the operator asked for one.

        The capture holds what the adapter *composed* — the exact body about to cross the wire — which
        is the one layer that answers "did the harness actually compose this correctly?" It is the body,
        never the headers: the API key travels in an `Authorization` header, so a capture that records
        only the body can never record the key. It is written before `open`, so a refused request is
        captured too — the moment the capture is most needed. A failure to write it does not stop the
        run: the capture is a convenience, and the run exists to talk to the server, not to fill a
        debug file."""
        if not self.wire_path:
            return
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "request": payload}, ensure_ascii=False)
        try:
            with open(self.wire_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def open(self, payload: dict):
        """POST and hand back the open response. Raises; `complete` turns that into a `Failed`.

        An HTTP error body is carried into the exception's message: the server's own words are
        the difference between "HTTP 400" and a fixable request."""
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": self.accept}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.request_url(),
                                         data=body, headers=headers, method="POST")
        try:
            return _OPENER.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = "(no body)"
            raise type(exc)(exc.url, exc.code, f"{exc.reason} [server said: {detail}]",
                            exc.headers, exc.fp) from None

def probe_context(base_url: str, model: str, api_key: str = "", api_path: str = "/chat/completions",
                  timeout: float = 5.0) -> int | None:
    """Ask the backend for the context window, and return the first number it declares.

    The answer to an undeclared window: the operator named no `--context`, and the backend may
    know its own size. Two metadata GETs, in order of likelihood: the OpenAI model listing
    (tabbyAPI carries `max_model_len` beside the model it serves, llama.cpp's OpenAI
    compatibility carries `meta.n_ctx`), then the llama.cpp-family `/props` at the server root
    (not under the `/v1` prefix: `n_ctx` there, and under
    `default_generation_settings.n_ctx`). The model entry is matched by id or name, and a
    listing of exactly one entry is taken when the name is not there — a single-model server.
    A value that is not a positive integer is not a number the run can use, and a probe that
    fails in any way — refused connection, timeout, HTTP error, a body the format does not
    carry — returns None rather than raising: the probe is advisory, and its failure is the
    operator's config gap (the default stands and a notice says so), never an error that
    blocks a startup the run itself would survive. It sends no completion, so it cannot
    displace the server's prefix cache; its timeout is short on purpose, because a probe that
    hangs for minutes at boot is a startup that depends on a server that may not be there.
    """
    for value in _probe_models(base_url, api_path, model, api_key, timeout):
        if _is_window(value):
            return value
    for value in _probe_props(base_url, model, api_key, timeout):
        if _is_window(value):
            return value
    return None


def _is_window(value) -> bool:
    """Whether this value is a usable context window: a positive whole number, not a bool (a
    JSON true is an int to Python, and a truth value is not a token count)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _get_json(url: str, api_key: str, timeout: float) -> dict | None:
    """One metadata GET: a JSON object, or None on any failure — the probe's only transport,
    and the one place a probe is allowed to fail quietly."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with _OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _model_entry(body: dict, model: str) -> dict | None:
    """The listing entry the probe's model names: by id, by name (the llama.cpp spelling), or —
    when the server lists exactly one model — the one entry, since a single-model server is the
    common case and the name it was started with may not be the name it lists. Several entries
    and no name is no answer: the probe would be guessing which window belongs to the run."""
    entries = body.get("data")
    if not isinstance(entries, list):
        entries = body.get("models")
    if not isinstance(entries, list) or not entries:
        return None
    for entry in entries:
        if isinstance(entry, dict) and (entry.get("id") == model or entry.get("name") == model):
            return entry
    if len(entries) == 1 and isinstance(entries[0], dict):
        return entries[0]
    return None


def _models_url(base_url: str, api_path: str) -> str:
    """The listing sits where the completions path says it does: the same URL with its final
    `chat/completions` swapped for `models`. A path that does not end that way is the gateway's
    own arrangement, and the best the probe can do is the base's `models` page — a refusal
    there is the None the caller falls back on, never a guess at the gateway's layout."""
    url = base_url + api_path
    marker = "/chat/completions"
    if url.endswith(marker):
        return url[:-len(marker)] + "/models"
    return base_url.rstrip("/") + "/models"


def _probe_models(base_url: str, api_path: str, model: str, api_key: str, timeout: float):
    """The window candidates in the model listing, in the order the fields were measured to
    carry them: tabbyAPI's `max_model_len` (measured, with `meta.n_ctx` agreeing), then
    llama.cpp's `meta.n_ctx` (measured), then the spellings a server of the OpenAI dialect may
    use (`n_ctx`, `context_length`, `max_context_length`) — inferred, unmeasured, and here
    because a run may be pointed at either."""
    body = _get_json(_models_url(base_url, api_path), api_key, timeout)
    entry = _model_entry(body, model) if body is not None else None
    if not isinstance(entry, dict):
        return
    for key in ("max_model_len", "n_ctx", "context_length", "max_context_length"):
        yield entry.get(key)
    meta = entry.get("meta")
    if isinstance(meta, dict):
        yield meta.get("n_ctx")


def _probe_props(base_url: str, model: str, api_key: str, timeout: float):
    """The window candidates in the llama.cpp-family /props, at the server root rather than
    under the /v1 prefix — the path llama.cpp and its forks serve it (measured on both).
    The base is what the adapter posts completions to, so the /v1 suffix is stripped before
    /props is appended; a base without the suffix is left as is, which is the best the probe
    can do with a gateway path, and a refusal there is the None the caller falls back on."""
    origin = base_url[:-3] if base_url.endswith("/v1") else base_url
    body = _get_json(f"{origin}/props?model={urllib.parse.quote(model)}", api_key, timeout)
    if body is None:
        return
    yield body.get("n_ctx")
    settings = body.get("default_generation_settings")
    if isinstance(settings, dict):
        yield settings.get("n_ctx")

