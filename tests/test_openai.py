"""The OpenAI-compatible backend. The parse core is driven with literal SSE lines and the
transport with a stubbed `open`, so every claim here is about bytes on a wire rather than about
a server being up."""
import json
import urllib.error

import pytest

from sah import model, openai
from sah.model import (Caps, Failed, Message, ReasoningDelta, Refused, Stopped, TextDelta, ToolCall,
                       Usage)
from sah.contract import spec_for


class Response:
    def __init__(self, lines=(), body=None):
        self.remaining = list(lines)
        self.body = body

    def readline(self):
        return self.remaining.pop(0) if self.remaining else b""

    def read(self):
        return self.body if self.body is not None else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Stub(openai.OpenAI):
    """A backend whose wire is a list of lines, or one whole body for the non-streaming shape. `payloads`
    shows what would have gone out."""

    def __init__(self, lines=(), error=None, body=None, **kwargs):
        super().__init__(model="test-1", **kwargs)
        self.wire = [line.encode() for line in lines]
        self.body = body
        self.error = error
        self.payloads = []

    def open(self, payload):
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return Response(self.wire, self.body)


def sse(*chunks):
    import json
    return [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]


def delta(**fields):
    return {"choices": [{"index": 0, "delta": fields}]}


# A stream that ends without this is a stream the adapter reports as broken, and the tests below feed a
# usage chunk and nothing else; the name keeps that guarantee out of their way.
FINISHED = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def run(backend, **kwargs):
    kwargs.setdefault("messages", [Message("user", "go")])
    kwargs.setdefault("tools", [])
    return list(backend.complete(system="sys", **kwargs))


def echo(path: str) -> str:
    """A test double for the wire: the docstring is maintainer prose, the description is the model's."""
    return path


echo.description = "Read a file."


SPEC = spec_for(echo)


# --- the wire ----------------------------------------------------------------------------

def test_the_api_key_falls_back_to_the_environment(monkeypatch):
    """A spawned child gets its key through the environment and never through argv, because argv is
    readable by every local user for the life of the process. That only works if this adapter reads the
    same spelling `sah.spawn` writes, so the name is pinned by a literal here and there: a rename on
    either side reddens a test rather than silently costing every spawned run its authentication.
    """
    monkeypatch.setenv("SAH_API_KEY", "sk-from-env")
    assert openai.OpenAI(model="m", host="h", port=1).api_key == "sk-from-env"
    # Both present: what the caller passed wins. Ordering this way round only shows up when the
    # environment is set too, so the case is asserted rather than assumed.
    assert openai.OpenAI(model="m", host="h", port=1, api_key="sk-explicit").api_key == "sk-explicit"
    monkeypatch.delenv("SAH_API_KEY")
    assert openai.OpenAI(model="m", host="h", port=1).api_key == ""
    assert openai.OpenAI(model="m", host="h", port=1, api_key="sk-explicit").api_key == "sk-explicit"


def test_a_call_cut_off_by_the_output_limit_is_reported_as_truncated():
    """A turn the budget ended leaves the call holding half a JSON document, which fails to parse.
    Reported as a plain parse failure the loop can only treat it as a model error; reported as truncated it
    knows to ask for a smaller call and to leave the run's status alone. Only the adapter can tell the two
    apart, because only the adapter sees the finish reason.
    """
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "id": "c1",
                                          "function": {"name": "echo", "arguments": '{"path": "x'}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}),)
    events = run(backend, tools=[SPEC])
    failed = [e for e in events if isinstance(e, Failed)]
    assert failed and failed[0].truncated, f"the limit broke this call; say so: {events}"
    assert isinstance(events[-1], Stopped) and events[-1].reason == "length"


def test_a_malformed_call_from_a_cleanly_finished_turn_is_not_called_truncated():
    """The distinction has a false-positive side. A turn that stopped cleanly and handed over broken JSON
    is a model error; calling that truncation would send the loop asking for a smaller call the model
    cannot make any smaller, three times, and then ending the run blaming the output limit.
    """
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "id": "c1",
                                          "function": {"name": "echo", "arguments": "not json"}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),)
    events = run(backend, tools=[SPEC])
    failed = [e for e in events if isinstance(e, Failed)]
    assert failed and not failed[0].truncated, f"nothing was cut off here: {events}"


def test_a_tool_call_arrives_whole_even_when_the_server_splits_it():
    """Servers send `arguments` in pieces. A client that parses each fragment separately either
    drops the call or runs it with half the arguments."""
    backend = Stub(sse(delta(content="working"),
                       delta(tool_calls=[{"index": 0, "id": "c1",
                                          "function": {"name": "echo", "arguments": '{"pa'}}]),
                       delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "x/y"'}}]),
                       delta(tool_calls=[{"index": 0, "function": {"arguments": "}"}},
                                         ]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),)
    events = run(backend, tools=[SPEC])
    call = [event for event in events if isinstance(event, ToolCall)][0]
    assert (call.id, call.name, call.arguments) == ("c1", "echo", {"path": "x/y"})
    assert isinstance(events[-1], Stopped)


def test_events_arrive_in_the_order_the_server_sent_them():
    backend = Stub(sse(delta(content="a"), delta(reasoning_content="think"),
                       {"usage": {"prompt_tokens": 11, "completion_tokens": 5,
                                  "completion_tokens_per_second": 90.5}},
                       delta(content="b"),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    kinds = [type(event).__name__ for event in run(backend)]
    assert kinds == ["TextDelta", "ReasoningDelta", "Usage", "TextDelta", "Stopped"]


def test_an_empty_fragment_carries_nothing():
    events = run(Stub(sse(delta(content=""), delta(reasoning_content=""), delta(content="real"),
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})))
    assert [type(e).__name__ for e in events] == ["TextDelta", "Stopped"]


def test_reasoning_capture_mirrors_the_field_the_server_used():
    """The dialect names the field that carries reasoning — llama.cpp's Qwen builds use
    `reasoning_content`, OpenAI-style servers `reasoning`, some gateways `reasoning_text` — and the
    capture remembers which one, because the re-send writes back into it. If a server puts the text
    into two of them at once, the first in the order wins and is captured once, not twice; a field
    the dialect does not recognize is not reasoning at all."""
    for field in ("reasoning_content", "reasoning", "reasoning_text"):
        events = run(Stub(sse(delta(**{field: "think"}),
                              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})))
        got = [e for e in events if isinstance(e, ReasoningDelta)]
        assert len(got) == 1 and got[0].text == "think" and got[0].field == field, (field, got)
    events = run(Stub(sse(delta(reasoning_content="same", reasoning="same"),
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})))
    got = [e for e in events if isinstance(e, ReasoningDelta)]
    assert len(got) == 1 and got[0].field == "reasoning_content", got
    events = run(Stub(sse(delta(thought="x"),
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})))
    assert not [e for e in events if isinstance(e, ReasoningDelta)], events


def test_the_require_reasoning_content_pin_keeps_the_key_present():
    """Some servers (DeepSeek's API among them) validate the key's presence on every assistant
    message, or look it up in the template without a defined-check: the turn that never reasoned is
    then the turn that breaks the request. The pin is the operator's declaration about the server —
    off by default (a server whose template only checks definedness should not see an empty think
    marker), on it pins the key empty where the turn did not fill it, and never overwrites the value
    the turn filled. It pins assistant entries only: a user turn is not the server's reasoning."""
    msgs = [Message("user", "hi"), Message("assistant", "working", reasoning="because"),
            Message("assistant", "done")]
    default = openai.OpenAI(model="m").build_payload("sys", msgs, [], "off", None)["messages"]
    # No pin by default: the keyless turn stays keyless
    assert "reasoning_content" not in default[3], default[3]
    pinned = openai.OpenAI(model="m", require_reasoning_content=True).build_payload(
        "sys", msgs, [], "off", None)["messages"]
    assert pinned[2]["reasoning_content"] == "because", pinned[2]      # the value is not overwritten
    assert pinned[3]["reasoning_content"] == "", pinned[3]             # the empty turn is pinned
    assert "reasoning_content" not in pinned[1], pinned[1]             # a user turn is not pinned


def test_the_reasoning_out_override_writes_the_re_send_where_the_server_reads():
    """The asymmetric server delivers the reasoning in one field and reads it back in another (a
    server that answers in `reasoning` but expects `reasoning_content`): the mirror would write the
    echo where it came from, and the operator names where it goes instead. The name is checked at
    construction — a request can never carry a field name the operator did not type."""
    msgs = [Message("assistant", "working", reasoning="because", reasoning_field="reasoning")]
    payload = openai.OpenAI(model="m", reasoning_out="reasoning_content").build_payload(
        "sys", msgs, [], "off", None)
    entry = payload["messages"][1]
    assert entry["reasoning_content"] == "because" and "reasoning" not in entry, entry
    for junk in ("bad-name", "not a field", "x<script>"):
        with pytest.raises(Refused, match="reasoning_out"):
            openai.OpenAI(model="m", reasoning_out=junk)


def test_a_keep_alive_comment_and_a_blank_separator_are_not_events():
    """A blank line is an SSE event separator and a `:` line is a keep-alive; an empty read is
    the end of the stream. The two are one byte apart on the wire, so the client must not
    confuse them."""
    backend = Stub(["\n", ": keep-alive"] + sse(
        delta(content="hi"), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    assert [type(e).__name__ for e in run(backend)] == ["TextDelta", "Stopped"]


# --- malformed protocol ------------------------------------------------------------------

def test_a_name_that_was_never_offered_is_a_failure_not_a_call():
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "function": {"name": "rm_rf",
                                                                  "arguments": "{}"}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
    events = run(backend, tools=[SPEC])
    assert any(isinstance(e, Failed) and "rm_rf" in e.error for e in events)
    assert not any(isinstance(e, ToolCall) for e in events)


def test_truncated_arguments_name_the_tool_and_keep_the_raw():
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "function": {"name": "echo",
                                                                  "arguments": '{"path": '}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
    failure = [e for e in run(backend, tools=[SPEC]) if isinstance(e, Failed)][0]
    assert "echo" in failure.error and "malformed" in failure.error


def test_arguments_that_are_not_an_object_are_refused():
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "function": {"name": "echo",
                                                                  "arguments": "[1,2]"}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
    failure = [e for e in run(backend, tools=[SPEC]) if isinstance(e, Failed)][0]
    assert "not a JSON object" in failure.error


def test_no_arguments_at_all_means_no_arguments():
    backend = Stub(sse(delta(tool_calls=[{"index": 0, "function": {"name": "echo"}}]),
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
    call = [e for e in run(backend, tools=[SPEC]) if isinstance(e, ToolCall)][0]
    assert call.arguments == {}


def test_a_stream_that_stops_mid_flight_flushes_the_call_and_names_the_break():
    """A dropped call is indistinguishable from a model that asked for nothing, so what
    accumulated is emitted before the break is reported."""
    backend = Stub(sse(delta(content="half"),
                       delta(tool_calls=[{"index": 0, "function": {"name": "echo",
                                                                  "arguments": '{"path": "z"}'}}])))
    events = run(backend, tools=[SPEC])
    assert isinstance(events[0], TextDelta) and isinstance(events[1], ToolCall)
    assert isinstance(events[-1], Failed) and "finish reason" in events[-1].error


def test_a_malformed_data_line_is_a_failure_and_the_stream_continues():
    lines = ["data: {not json"] + sse(
        delta(content="still here"), {"choices": [{"index": 0, "delta": {},
                                                   "finish_reason": "stop"}]})
    events = run(Stub(lines))
    assert isinstance(events[0], Failed)
    assert any(isinstance(e, TextDelta) and e.text == "still here" for e in events)


# --- transport ---------------------------------------------------------------------------

def test_a_transport_failure_is_one_failure_and_never_an_empty_turn():
    events = run(Stub(error=OSError("connection refused")))
    assert len(events) == 1 and isinstance(events[0], Failed)
    assert "connection refused" in events[0].error


def test_the_server_s_own_error_message_reaches_the_reason(monkeypatch):
    """`HTTP 400` alone sends the reader to the server's console; the body is usually the fix.
    Driven through the real `open` with the socket layer replaced, since the body capture lives
    there."""
    error = urllib.error.HTTPError("http://x", 400, "Bad Request", None, None)
    error.read = lambda: b'{"error": "this model wants a system message first"}'

    def refuse(request, *args, **kwargs):
        raise error

    monkeypatch.setattr(openai._OPENER, "open", refuse)
    failure = [e for e in run(openai.OpenAI(model="m")) if isinstance(e, Failed)][0]
    assert "HTTP 400" in failure.error and "system message first" in failure.error


def test_a_role_the_wire_format_lacks_is_raised_before_anything_is_sent():
    backend = Stub(sse(delta(content="unreachable")))
    with pytest.raises(Refused, match="not an OpenAI role"):
        run(backend, messages=[Message("wizard", "hi")])
    assert backend.payloads == [], "nothing may be sent once the request is known to be invalid"


# --- the request ---------------------------------------------------------------------------

def test_the_request_shape():
    backend = Stub(sse(delta(content="x")))
    run(backend, messages=[Message("assistant", "said", tool_calls=(("c1", "echo", {"p": 1}),),
                                  reasoning="because"),
                           Message("tool", "out", tool_call_id="c1")],
        tools=[SPEC])
    payload = backend.payloads[0]
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assistant = payload["messages"][1]
    # Reasoning is present on the turn and travels in the request: the model's own prior reasoning always
    # goes back to the backend, whatever the template does with it. Whether the template renders it is the
    # separate `chat_template_kwargs` decision; see
    # `test_reasoning_always_travels_and_rendering_is_the_separate_kwarg`.
    assert assistant["reasoning_content"] == "because"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"p": 1}'
    assert payload["messages"][2]["tool_call_id"] == "c1"
    assert payload["tools"][0]["function"]["name"] == "echo"
    assert payload["stream"] is True and payload["stream_options"]["include_usage"] is True
    assert "max_tokens" not in payload, "omission delegates to the server, it is not a guess"


def test_none_asks_for_no_field_and_off_is_a_level_like_any_other():
    """The owner's ruling, second form: `none` means *send nothing*, and `off` is a real level.

    The first reading made `off` mean "no preference" and `none` mean "do not reason", because the
    OpenAI dialect's enum has `none` and not `off`. Measured against the running server that reading
    got backwards twice over: `none` produced 0 reasoning characters (it means off there, exactly like
    `off`), and the empty-string spelling of "no preference" answered HTTP 400 — that server's own
    words are `reasoning_effort must be off, minimal, low, medium, high, xhigh, or max`. So the word
    for "I have no preference" is the harness's own `none`, and the wire carries it as a JSON null:
    the one spelling that survives both dialects measured: a server that honours the field accepts it
    and reasons, and a server that pins reasoning off by configuration accepts it and stays silent.

    `off` is therefore an ordinary level, passed through verbatim like `low` or `xhigh`. It is not the
    harness's internal request, and the internal sentinel no longer exists to confuse it with.

    `on` is the operator's word for "reason, at whatever level this server defaults to", and the ruling
    is that naming nothing omits the field: a server that predates reasoning levels, or validates its
    body strictly, must see a body it can parse in full, and a template takes the default branch it
    took before the field existed — an explicit value, even a measured-accepted `null`, is a field a
    dialect may not declare.
    """
    for value in (None, "none", "on"):            # naming nothing: the field is not on the wire at all
        omitted = Stub(sse(delta(content="x")))
        run(omitted, thinking=value)
        assert "reasoning_effort" not in omitted.payloads[0], \
            f"{value!r} put a field on the wire a server's dialect may not declare"

    for level in ("off", "low", "medium", "high", "xhigh", "minimal", "max"):
        stub = Stub(sse(delta(content="x")))
        run(stub, thinking=level)
        assert stub.payloads[0]["reasoning_effort"] == level, f"{level!r} was rewritten"


def test_the_internal_reasoning_sentinel_is_gone():
    """A sentinel whose only job was to spell `off` without meaning the operator's `off` has no job
    once `off` is an ordinary level. Leaving it defined would leave two names for one wire value, and
    the pair is what made the first ruling unreadable."""
    assert not hasattr(model, "REASONING_OFF"), \
        "the sentinel outlived the distinction it encoded"


def test_a_budget_is_sent_when_given():
    backend = Stub(sse(delta(content="x")))
    run(backend, max_tokens=400)
    assert backend.payloads[0]["max_tokens"] == 400


def test_no_streaming_says_so_and_carries_no_stream_options():
    """`--no-streaming` exists for the server that cannot stream at all, so the body says
    `stream: false` and carries no `stream_options`: the dialect refuses that field on a
    non-streaming request, so the usage request rides on streaming and goes with it. A non-streaming
    answer reports its usage in the body regardless."""
    backend = Stub(body=json.dumps({"choices": [{"index": 0, "message": {"content": "x"},
                                                 "finish_reason": "stop"}]}).encode(),
                   streaming=False)
    run(backend)
    payload = backend.payloads[0]
    assert payload["stream"] is False
    assert "stream_options" not in payload, "the field rides on streaming; the dialect refuses it otherwise"


def test_no_usage_omits_the_field_while_streaming():
    """`--no-usage` omits `stream_options` the way a named level is omitted when unnamed: a field a
    server's dialect may not declare is one the operator can take off the wire."""
    backend = Stub(sse(delta(content="x")), request_usage=False)
    run(backend)
    payload = backend.payloads[0]
    assert payload["stream"] is True
    assert "stream_options" not in payload


def test_the_accept_header_says_what_the_body_is():
    """The one header a gateway may key on before the body is read must name the shape the response
    will take: SSE while streaming, plain JSON otherwise."""
    assert openai.OpenAI(model="m").accept == "text/event-stream"
    assert openai.OpenAI(model="m", streaming=False).accept == "application/json"


def test_capabilities_are_declared_and_a_server_is_never_probed():
    """Constructing a backend for a host that is not listening must not fail: a probe would
    make starting a session depend on a warm GPU."""
    backend = openai.OpenAI(model="q", host="127.0.0.1", port=1, context_tokens=99)
    caps = backend.capabilities()
    assert isinstance(caps, Caps) and caps.model == "q" and caps.context_tokens == 99
    # Declared blind unless the operator says otherwise — the default the image refusal depends on.
    assert caps.supports_vision is False


def test_the_seam_is_satisfied_structurally():
    from sah.model import Backend
    assert isinstance(openai.OpenAI(model="q"), Backend)


# --- what a chunk may claim about itself, and where the request is allowed to go ---------------

def test_a_null_usage_count_is_zero_and_not_a_failed_turn():
    """A usage object with `"prompt_tokens": null` is a server saying "I did not count", which is
    metadata. `int(None)` raises `TypeError`, `parse_sse` turned that into `Failed("malformed chunk")`,
    the loop recorded `last_error`, and the whole turn ended `failed` — the answer arrived and the run
    was lost over a number nobody needed.

    A null becomes 0, which is what the field means when nothing was counted, and the text of the turn
    survives untouched."""
    events = list(openai.parse_sse(sse(
        delta(content="the answer"),
        {"usage": {"prompt_tokens": None, "completion_tokens": None}},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), frozenset()))
    assert not [e for e in events if isinstance(e, Failed)], f"a metadata field failed the turn: {events}"
    usage = [e for e in events if isinstance(e, Usage)]
    assert usage and usage[0].prompt_tokens == 0 and usage[0].completion_tokens == 0, events
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "the answer"


def test_a_cache_hit_is_read_from_the_key_the_servers_answer_with():
    """How many prompt tokens the server did not re-embed is the one number that separates a 300 ms
    prefill from a 15 ms KV restore, and a turn that took either looks identical on screen.

    The key is `usage.prompt_tokens_details.cached_tokens`, which both measured servers answer with; one
    of them repeats the same figure as a top-level `cached_tokens`, and only one spelling is read so the
    two cannot disagree here."""
    events = list(openai.parse_sse(sse(
        {"usage": {"prompt_tokens": 1927, "completion_tokens": 2,
                   "prompt_tokens_details": {"cached_tokens": 1907}}},
        FINISHED), frozenset()))
    usage = [e for e in events if isinstance(e, Usage)][0]
    assert usage.cached == 1907, events


def test_a_cache_shape_that_is_not_a_count_is_no_count_and_costs_nothing():
    """A missing key, a null `details`, a string, a bool: each means "the server did not say", and none
    of them is a malformed chunk. A number worth showing is not one worth failing a decode over, and
    silence must not arrive as `cached=0` — that would tell the console a warm cache missed."""
    for usage_obj in ({"prompt_tokens": 5, "completion_tokens": 1},
                      {"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": None},
                      {"prompt_tokens": 5, "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": "most of it"}},
                      {"prompt_tokens": 5, "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": True}}):
        events = list(openai.parse_sse(sse({"usage": usage_obj}, FINISHED), frozenset()))
        assert not [e for e in events if isinstance(e, Failed)], events
        assert [e for e in events if isinstance(e, Usage)][0].cached is None, usage_obj


def test_a_rate_arrives_as_a_float_or_not_at_all():
    """A rate is a float in the record and a server may send an integer, so the adapter coerces: `45`
    becomes `45.0` rather than costing the whole `message` step at the taxonomy. A word for a number, or
    no key, is no rate and yields `None`.

    A zero passes through rather than becoming silence here. It is what a server reports when it did no
    work at all — a request whose whole context hit the prompt cache answers `prompt_tokens_per_second: 0`
    — and deciding whether a zero is news belongs to whoever prints the number, not to the adapter."""
    def usage_of(usage_obj):
        events = list(openai.parse_sse(sse({"usage": usage_obj}, FINISHED), frozenset()))
        assert not [e for e in events if isinstance(e, Failed)], events
        return [e for e in events if isinstance(e, Usage)][0]

    got = usage_of({"prompt_tokens": 5, "completion_tokens": 1,
                    "completion_tokens_per_second": 45, "prompt_tokens_per_second": 881.5})
    assert got.tps == 45.0 and isinstance(got.tps, float) and got.prefill_tps == 881.5

    silent = usage_of({"prompt_tokens": 5, "completion_tokens": 1,
                       "completion_tokens_per_second": "fast", "prompt_tokens_per_second": True})
    assert silent.tps is None and silent.prefill_tps is None

    zero = usage_of({"prompt_tokens": 5, "completion_tokens": 1,
                     "completion_tokens_per_second": 0, "prompt_tokens_per_second": 0})
    assert zero.tps == 0.0 and zero.prefill_tps == 0.0


def test_the_backend_never_hands_the_conversation_to_an_environment_proxy(monkeypatch):
    """`HTTP_PROXY` in a shell is a fact about that shell, not about where this run's model lives. With
    `urllib`'s default handling, a corporate proxy silently receives the entire conversation — every
    prompt, every tool result — on its way to a LAN address it often cannot reach at all.

    The backend builds its own opener with no proxies, which is the same ruling the web capability made
    for a different reason: a proxy resolves the name itself, so it is the second opinion an address
    check exists to remove. A machine that genuinely needs a proxy between the harness and its server is
    an operator's decision, expressed on the command line rather than inherited from the environment."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    # Asserted as dispatch, not as handler presence: an empty ProxyHandler contributes no `<protocol>_open`
    # method, so `add_handler` drops it and an assertion that it is installed and empty could never hold.
    # The fresh build happens *inside* the environment being described — an opener read at import would
    # have seen a proxy-free shell and passed against code that proxies.
    plain = urllib.request.build_opener()
    control = any(isinstance(h, urllib.request.ProxyHandler) for h in plain.handle_open.get("http", []))
    assert control, "the environment did not reach a plain opener; this test describes nothing"
    for opener, whence in ((openai._build_opener(), "a fresh opener"),
                           (openai._OPENER, "the shipped opener")):
        for scheme in ("http", "https"):
            chain = opener.handle_open.get(scheme, [])
            assert chain, f"{whence}: nothing would handle {scheme}"
            assert not any(isinstance(h, urllib.request.ProxyHandler) for h in chain), \
                f"{whence}: {scheme} would be routed through a proxy nobody chose"


def test_the_request_goes_out_through_that_one_opener(monkeypatch):
    """The opener is only the answer if the request actually travels through it. Captured here rather
    than trusted from a glance, because `urlopen` and an opener are two doors and only one was closed."""
    seen = []

    class Response:
        def read(self, *_):
            return b"data: [DONE]\n\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(request, *args, **kwargs):
        seen.append(request)
        return Response()

    monkeypatch.setattr(openai._OPENER, "open", fake_open)
    backend = openai.OpenAI(model="m", host="127.0.0.1", port=9091)
    list(backend.complete(system="sys", messages=[Message("user", "go")], tools=[]))
    assert len(seen) == 1, "the request did not go through the module's opener"
    assert seen[0].full_url.endswith("/v1/chat/completions"), seen[0].full_url


# --- where the request goes -------------------------------------------------------------------

def _captured_url(backend, monkeypatch):
    seen = []

    class Response:
        def read(self, *_):
            return b"data: [DONE]\n\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(request, *args, **kwargs):
        seen.append(request)
        return Response()

    monkeypatch.setattr(openai._OPENER, "open", fake_open)
    list(backend.complete(system="sys", messages=[Message("user", "go")], tools=[]))
    assert len(seen) == 1, "the request did not go through the module's opener"
    return seen[0].full_url


def test_a_base_url_reaches_a_provider_that_lives_behind_a_path(monkeypatch):
    """`http://host:port/v1` is one server's shape. A gateway in front of a model serves it under a
    prefix (`https://gw.example.com/llama/v1`), and with no way to name a base the operator's only lever
    is a proxy or a rewrite outside the harness — both of which put the conversation somewhere the run
    never chose.

    The base owns the scheme, the host and the prefix; `api_path` is what hangs off it."""
    backend = openai.OpenAI(model="m", base_url="https://gw.example.com/llama/v1/")
    url = _captured_url(backend, monkeypatch)
    assert url == "https://gw.example.com/llama/v1/chat/completions", url
    assert url.startswith("https://"), "a provider named as https was asked over plaintext"


def test_the_default_base_is_the_host_and_port_the_operator_named(monkeypatch):
    """The control for the last test: with no base given, the ordinary local case must land exactly where
    it always did, down to the `/v1` the OpenAI-dialect servers expect."""
    backend = openai.OpenAI(model="m", host="127.0.0.1", port=9090)
    assert _captured_url(backend, monkeypatch) == "http://127.0.0.1:9090/v1/chat/completions"


def test_an_ipv6_literal_is_bracketed(monkeypatch):
    """A bare colon is a delimiter in a URL authority, so `http://::1:9090/...` is not a host and a port —
    it is an unparseable target that fails with a message about the URL rather than about the address.
    A literal is bracketed on the way in; one the operator already bracketed is not doubled."""
    assert (_captured_url(openai.OpenAI(model="m", host="::1", port=9090), monkeypatch)
            == "http://[::1]:9090/v1/chat/completions")
    assert (_captured_url(openai.OpenAI(model="m", host="[fe80::1]", port=9090), monkeypatch)
            == "http://[fe80::1]:9090/v1/chat/completions")


def test_a_key_travelling_in_clear_to_a_remote_host_is_said_out_loud():
    """`--api-key` over `http://` to a host that is not this machine puts the credential on the wire in
    plaintext, which is ordinary on a trusted LAN and wrong across the internet. The harness does not
    refuse the LAN case — it is the common one and the operator named the host — but it says what is
    happening once, on stderr, where a person will read it, rather than letting a working run be evidence
    that nothing was exposed."""
    remote = openai.OpenAI(model="m", base_url="http://box.example.com/v1", api_key="sk-secret")
    assert remote.key_in_clear_to_remote, "a remote http endpoint with a key read as safe"
    for safe in (openai.OpenAI(model="m", base_url="https://box.example.com/v1", api_key="sk-secret"),
                 openai.OpenAI(model="m", host="127.0.0.1", port=9090, api_key="sk-secret"),
                 openai.OpenAI(model="m", host="localhost", port=9090, api_key="sk-secret"),
                 openai.OpenAI(model="m", base_url="http://box.example.com/v1")):
        assert not safe.key_in_clear_to_remote, safe.request_url()
    # A scheme this backend cannot speak is refused where it is named, not at the first request.
    with pytest.raises(Refused, match="must be http"):
        openai.OpenAI(model="m", base_url="ws://box.example.com/v1", api_key="sk")


def test_reasoning_always_travels_and_rendering_is_the_separate_kwarg():
    """The turn's own reasoning always travels back as reasoning_content — it is the model's own text,
    a fact about the conversation. Whether the template renders it into the prompt is a different
    question: the preserve_thinking kwarg, and only the kwarg.

    `none` (the default) declines to express an opinion — no kwarg travels, and the template's own
    default decides. `true`/`false` express one explicitly. A value outside the three is refused
    where it is named, not at the first request."""
    msgs = [Message("assistant", "working", reasoning="resolve the path first")]
    none = openai.OpenAI(model="m").build_payload("sys", msgs, [], "off", None)
    on = openai.OpenAI(model="m", preserve_thinking="true").build_payload("sys", msgs, [], "off", None)
    off = openai.OpenAI(model="m", preserve_thinking="false").build_payload("sys", msgs, [], "off", None)
    # index 1: the payload leads with the system prompt, and the turn under test is the one after it
    for payload, kwargs in ((none, None), (on, {"preserve_thinking": True}),
                            (off, {"preserve_thinking": False})):
        assert payload["messages"][1]["reasoning_content"] == "resolve the path first", payload["messages"][1]
        assert payload.get("chat_template_kwargs") == kwargs, payload
    with pytest.raises(Refused, match="preserve_thinking"):
        openai.OpenAI(model="m", preserve_thinking="yes")
    # And the re-send writes into the field the turn came in on, not a fixed name: the signature
    # rides with the turn, so a dialect that returned `reasoning` goes back in `reasoning`.
    mirrored = openai.OpenAI(model="m").build_payload(
        "sys", [Message("assistant", "working", reasoning="because", reasoning_field="reasoning")],
        [], "off", None)
    entry = mirrored["messages"][1]
    assert entry["reasoning"] == "because" and "reasoning_content" not in entry, entry

def test_an_image_to_a_blind_server_is_refused_by_the_harness_not_by_the_server():
    """`supports_vision` has to be consulted before the bytes go, or it is a decoration.

    The field already carries its own reasoning: the check is worth more than a base64 blob arriving at a
    server that cannot read it. Until something reads it, a `--vision` run and a blind endpoint differed
    only in how the failure looks -- an HTTP error from someone else's port, after the upload.
    """
    blind = openai.OpenAI(model="m", host="h", port=1, supports_vision=False)
    seeing = openai.OpenAI(model="m", host="h", port=1, supports_vision=True)
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    with pytest.raises(Refused, match="image"):
        blind.build_payload("sys", [Message(role="user", content=parts)], [], "off", None)
    body = seeing.build_payload("sys", [Message(role="user", content=parts)], [], "off", None)
    assert body["messages"][1]["content"] == parts, "a seeing server was not sent the parts it can read"


def test_a_picture_is_attached_to_the_answer_that_loaded_it_and_not_to_another_turn():
    """The harness answers the call that asked for the file, on that call's own message, and does not
    relocate the picture because some server would rather it arrived elsewhere. One endpoint measured on
    this host refuses it in a tool result (`image_url requires a user message and a string URL`), and the
    temptation that follows is a turn the harness authors to carry the bytes; that is refused here because
    `--vision` is the operator's claim about *this* endpoint, and a claim that does not hold belongs in a
    live check, not in a silent rewrite of who said what."""
    seeing = openai.OpenAI(model="m", host="h", port=1, supports_vision=True)
    parts = [{"type": "text", "text": "read x.png (image/png, 73 bytes)"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    body = seeing.build_payload("sys", [Message(role="tool", content=parts, tool_call_id="c1")],
                                [], "off", None)
    entry = body["messages"][1]
    assert entry["role"] == "tool" and entry["tool_call_id"] == "c1", entry
    assert entry["content"] == parts, entry


# --- the wire capture: the one layer that shows what the adapter composed -----------------

def test_the_wire_capture_records_exactly_what_the_adapter_composed(tmp_path):
    """`--wire` exists to answer "did the harness actually compose this correctly?": the log's
    fingerprint pins what the harness *had*, and the capture shows what the adapter *built* from it.
    So the captured request must be byte-for-byte the payload the transport would have sent."""
    wire = tmp_path / "wire.jsonl"
    backend = Stub(lines=sse(delta(content="done"), FINISHED), wire=str(wire))
    events = run(backend)
    assert any(isinstance(e, Stopped) for e in events), "the turn ran"
    assert wire.exists(), "the capture is a file the operator can read"
    lines = wire.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one request, one line"
    entry = json.loads(lines[0])
    assert "ts" in entry
    # The capture is the payload the transport received — the composed body, not a re-derivation.
    assert entry["request"] == backend.payloads[0]
    # ... and it is the composition itself: the system is the first message, nothing rewritten.
    assert entry["request"]["messages"][0] == {"role": "system", "content": "sys"}
    assert entry["request"]["messages"][1] == {"role": "user", "content": "go"}


def test_the_wire_capture_is_written_even_when_the_request_is_refused(tmp_path):
    """A refused request is the moment the capture is most needed, so it is written before the POST:
    the composed body lands in the file even when `open` raises and the turn comes back a `Failed`."""
    wire = tmp_path / "wire.jsonl"
    backend = Stub(lines=(), error=urllib.error.HTTPError("u", 400, "refused", {}, None),
                   wire=str(wire))
    events = run(backend)
    assert any(isinstance(e, Failed) for e in events), "the refusal is reported"
    assert wire.exists(), "the capture survives the refusal it was made for"
    entry = json.loads(wire.read_text(encoding="utf-8").splitlines()[0])
    assert entry["request"] == backend.payloads[0]


def test_the_wire_capture_appends_one_line_per_request(tmp_path):
    """The capture is append-only, like the log it sits beside: a second request adds a line, it
    does not rewrite the first, so a run's whole wire is the file, in order."""
    wire = tmp_path / "wire.jsonl"
    backend = openai.OpenAI(model="test-1", wire=str(wire))
    backend._write_wire({"n": 1})
    backend._write_wire({"n": 2})
    lines = wire.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["request"] for line in lines] == [{"n": 1}, {"n": 2}]


def test_the_wire_capture_is_off_by_default_and_a_noop_when_unset(tmp_path):
    """The capture is a convenience the operator opts into: a run that did not ask for a wire leaves
    no capture, and the write is a safe no-op when there is no path, so the default path is one the
    loop need not think about."""
    backend = openai.OpenAI(model="test-1")
    assert backend.wire_path is None
    backend._write_wire({"model": "test-1"})      # a no-op: no path, nothing written, nothing raised


# --- the single body: the non-streaming round trip --------------------------------


def answer(content=None, reasoning=None, tool_calls=None, finish="stop", usage=None):
    """A non-streaming response body: a `message` where the stream has a `delta`."""
    message: dict = {}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    choice: dict = {"index": 0, "message": message}
    if finish is not None:
        choice["finish_reason"] = finish
    body: dict = {"choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode("utf-8")


def test_a_single_body_answers_through_the_same_parse_core():
    """The body is mapped onto the chunk shape the SSE core reads, so a text answer with a clean stop
    yields the same events the stream does: one delta, one stop, in order."""
    events = list(openai.parse_body(answer(content="x")))
    assert isinstance(events[0], TextDelta) and events[0].text == "x", events
    assert isinstance(events[1], Stopped) and events[1].reason == "stop", events
    assert len(events) == 2, events


def test_a_whole_tool_call_from_a_single_body_passes_the_call_gate():
    """A call a single-body server sends whole still passes through `_tool_call_from_delta`: the
    arguments are parsed, the name is checked against what the turn offered, and the id is kept."""
    raw = answer(tool_calls=[{"id": "c1", "type": "function",
                              "function": {"name": "echo", "arguments": '{"path": "x"}'}}])
    events = list(openai.parse_body(raw, frozenset({"echo"})))
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(calls) == 1, events
    assert calls[0].id == "c1" and calls[0].name == "echo" and calls[0].arguments == {"path": "x"}
    assert any(isinstance(e, Stopped) for e in events), events


def test_two_calls_without_streaming_indices_are_kept_apart():
    """The `index` a call accumulates under is a streaming artifact — the fragment's position in the
    array. A whole answer has no position but its order, so the parse core is given one per call:
    two calls sent whole must arrive two, not one holding the other's name."""
    raw = answer(tool_calls=[{"id": "c1", "type": "function",
                              "function": {"name": "echo", "arguments": "{}"}},
                             {"id": "c2", "type": "function",
                              "function": {"name": "verbose", "arguments": "{}"}}])
    events = list(openai.parse_body(raw, frozenset({"echo", "verbose"})))
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert [(c.id, c.name) for c in calls] == [("c1", "echo"), ("c2", "verbose")], events


def test_a_single_body_with_no_offered_name_fails_like_the_stream():
    """The gate is one: a name the turn never offered is a `Failed` naming the tool, on the wire
    shape that serves it."""
    raw = answer(tool_calls=[{"id": "c1", "type": "function",
                              "function": {"name": "other", "arguments": "{}"}}])
    events = list(openai.parse_body(raw, frozenset({"echo"})))
    failed = [e for e in events if isinstance(e, Failed)]
    assert failed and "other" in failed[0].error, events
    assert not any(isinstance(e, ToolCall) for e in events), events


def test_a_single_body_cut_off_by_the_limit_is_truncated_like_the_stream():
    """`finish_reason: length` under a whole-body call with broken JSON is the same report as on the
    stream: a truncated failure, not a model error — the loop asks for a smaller call either way."""
    raw = answer(tool_calls=[{"id": "c1", "type": "function",
                              "function": {"name": "echo", "arguments": '{"path": "x"' }}],
                 finish="length")
    events = list(openai.parse_body(raw, frozenset({"echo"})))
    failed = [e for e in events if isinstance(e, Failed)]
    assert failed and failed[0].truncated, events
    assert isinstance(events[-1], Stopped) and events[-1].reason == "length", events


def test_a_single_body_carrying_usage_yields_it_before_any_break():
    """A broken answer that still measured itself has fed the calibration: the usage is yielded the
    way the streaming path yields it, and the break is named after it."""
    raw = answer(usage={"prompt_tokens": 10, "completion_tokens": 2}, finish=None)
    events = list(openai.parse_body(raw))
    usage = [e for e in events if isinstance(e, Usage)]
    assert usage and usage[0].prompt_tokens == 10 and usage[0].completion_tokens == 2, events
    failed = [e for e in events if isinstance(e, Failed)]
    assert failed and "finish reason" in failed[0].error, events


def test_a_malformed_single_body_is_a_failure_not_a_crash():
    """A body the JSON grammar refuses is one `Failed` carrying the server's bytes, never an exception
    out of the parse core."""
    events = list(openai.parse_body(b"{not json"))
    assert len(events) == 1 and isinstance(events[0], Failed), events
    assert "malformed response body" in events[0].error


def test_a_single_body_without_a_message_names_the_break():
    """A body with no `choices[0].message` is not an empty answer the loop can nudge on; it is a shape
    the harness cannot dress up, and the failure says so rather than dying on the field access."""
    for raw in (b"{}", b"{\"choices\": []}", b"{\"choices\": [{\"delta\": {\"content\": \"x\"}}]}"):
        events = list(openai.parse_body(raw))
        assert len(events) == 1 and isinstance(events[0], Failed), (raw, events)
        assert "choices[0].message" in events[0].error, events


def test_a_non_streaming_turn_runs_through_complete():
    """The transport branch: `--no-streaming` reads one body through the same `complete` the stream
    uses, so the loop downstream sees one event vocabulary either way — usage first, the way the
    chunk core yields it."""
    backend = Stub(body=answer(content="x", usage={"prompt_tokens": 7,
                                                   "completion_tokens": 3}),
                   streaming=False)
    events = run(backend)
    assert isinstance(events[0], Usage) and events[0].prompt_tokens == 7, events
    assert isinstance(events[1], TextDelta) and events[1].text == "x", events
    assert isinstance(events[2], Stopped) and events[2].reason == "stop", events


# --- the context probe: the window the operator did not declare -----------------

class _ProbeFeed:
    """An `_OPENER.open` stand-in that answers by path, the way the real transport would:
    a dict of path-suffix to body bytes, each served once, and every request recorded so a
    test can say what the probe asked for and in what order."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.asked = []

    def __call__(self, request, *args, **kwargs):
        self.asked.append(request.full_url)
        for suffix, body in self.bodies.items():
            if request.full_url.endswith(suffix):
                import io
                return io.BytesIO(body)
        import urllib.error
        raise urllib.error.URLError("no scripted answer for " + request.full_url)


PROBE_BASE = "http://127.0.0.1:9090/v1"


def _probe(monkeypatch, bodies, model="m"):
    """The base is what the adapter actually posts to: the operator's base or the default
    http://host:port/v1. The listing therefore sits under /v1, beside /chat/completions."""
    feed = _ProbeFeed(bodies)
    monkeypatch.setattr(openai._OPENER, "open", feed)
    return openai.probe_context(PROBE_BASE, model, "sk-test"), feed


def test_the_probe_reads_the_window_tabbyapi_declares(monkeypatch):
    """`max_model_len` is the field tabbyAPI puts on the listing, measured on this box: the value
    is the harness's window from the moment the operator does not declare one, and the probe asks
    for the listing only — a props request would be one more round trip for a number already held."""
    body = b'{"data": [{"id": "m", "object": "model", "max_model_len": 220000, "meta": {"n_ctx": 220000}}]}'
    value, feed = _probe(monkeypatch, {"/v1/models": body})
    assert value == 220000
    assert len(feed.asked) == 1 and feed.asked[0].endswith("/v1/models"), \
        "a listing that answers is answered: the probe stops where it got its number"


def test_the_probe_reads_the_window_llamacpp_declares(monkeypatch):
    """llama.cpp carries the window under `meta.n_ctx` (and `max_model_len` is not its field),
    measured on this box at the gemma endpoint. The probe tries the listing's own fields in
    precedence order and stops at the first it finds."""
    body = b'{"data": [{"id": "gemma", "object": "model", "meta": {"n_ctx": 16128}}]}'
    value, feed = _probe(monkeypatch, {"/v1/models": body}, model="gemma")
    assert value == 16128
    assert len(feed.asked) == 1


def test_the_probe_falls_to_props_when_the_listing_says_nothing(monkeypatch):
    """A listing without a window is not an answer: the probe asks the model's own properties page,
    the field the OpenAI-protocol servers that know their size put there. The props query is named
    with the model — a props page that answers for another model would be a number for the wrong
    endpoint."""
    listing = b'{"data": [{"id": "demo-model", "object": "model"}]}'
    props = b'{"default_generation_settings": {"n_ctx": 131072}}'
    value, feed = _probe(monkeypatch, {"/v1/models": listing,
                                       "/props?model=demo-model": props},
                         model="demo-model")
    assert value == 131072
    assert feed.asked[0].endswith("/v1/models") and feed.asked[1].endswith("/props?model=demo-model"), \
        feed.asked


def test_a_backend_that_declares_no_window_answers_nothing(monkeypatch):
    """The no-window shape, measured: a listing and a props page that carry no window field at all.
    The answer is `None`, not an exception and not a guess — `None` is what says the operator must
    declare the window, and the run is told so at boot."""
    value, feed = _probe(monkeypatch, {"/v1/models": b'{"data": [{"id": "demo-model"}]}',
                                       "/props?model=demo-model": b'{"version": "1"}'},
                         model="demo-model")
    assert value is None
    assert len(feed.asked) == 2


def test_the_probe_counts_only_the_model_it_was_asked_for(monkeypatch):
    """A listing can hold several models, and a number for the wrong one would size the run against
    a window it is not running in. The id match is the gate; a listing with several models and no
    id on any of them is no answer, not the first entry's."""
    body = b'{"data": [{"id": "other", "max_model_len": 999999}, {"id": "m", "max_model_len": 1234}]}'
    assert _probe(monkeypatch, {"/v1/models": body}, model="m")[0] == 1234
    body = b'{"data": [{"max_model_len": 999999}, {"max_model_len": 1234}]}'
    assert _probe(monkeypatch, {"/v1/models": body}, model="m")[0] is None


def test_a_window_field_with_no_number_is_not_a_window(monkeypatch):
    """The probe asks for the value's shape, not just the field's presence: a zero, a fraction, a
    string and a null are fields the server put there for other reasons, and each one is skipped
    for the next candidate field rather than read as a number."""
    body = b'{"data": [{"id": "m", "max_model_len": 0, "meta": {"n_ctx": 12.5}, "context_length": "big"}]}'
    assert _probe(monkeypatch, {"/v1/models": body})[0] is None
    body = b'{"data": [{"id": "m", "max_model_len": null, "context_length": 555}]}'
    assert _probe(monkeypatch, {"/v1/models": body})[0] == 555


def test_a_probe_that_cannot_reach_the_backend_answers_nothing(monkeypatch):
    """A down backend at boot is no different from a silent one: `None`, never an exception, so
    the probe cannot be the thing that stops a run from starting. The run starts at the default
    and says so."""
    import urllib.error
    def refuse(request, *args, **kwargs):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(openai._OPENER, "open", refuse)
    assert openai.probe_context("http://127.0.0.1:9090", "m", "sk-test") is None


def test_construction_never_probes_and_the_probe_is_a_named_call(monkeypatch):
    """The split the constructor keeps: nothing in `OpenAI(...)` touches the network, and the probe
    is a separate named act the CLI makes at boot. Construction on a down backend succeeds, and the
    probe is the only thing that asks — measured by the transport being replaced and never called
    during construction."""
    asked = []

    class _NoOpen:
        def open(self, *args, **kwargs):
            asked.append(1)

    monkeypatch.setattr(openai._OPENER, "open", _NoOpen().open)
    backend = openai.OpenAI(model="m", host="127.0.0.1", port=9090, context_tokens=32768)
    assert asked == []
    assert backend.capabilities().context_tokens == 32768
    assert callable(openai.probe_context)
