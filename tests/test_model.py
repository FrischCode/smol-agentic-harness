"""The seam: what a request must not contain, what comes back, and in what order."""
import pytest

from sah import model
from sah.record import Refused

USER = model.Message("user", "do the thing")


def test_a_role_the_wire_does_not_have_is_refused_when_the_request_is_made():
    """Refused at the call, not at the first pull: a generator that only checks when iterated
    would let "the request is never sent" depend on someone starting the iteration."""
    backend = model.Fake([model.turn_text("x")])
    with pytest.raises(Refused, match="not an OpenAI role"):
        backend.complete(system="", messages=[model.Message("human", "nope")], tools=[])


def test_events_arrive_in_the_order_the_backend_produced_them():
    backend = model.Fake([model.turn_tool("c1", "bash", {"cmd": "ls"}, then="listed")])
    events = list(backend.complete(system="", messages=[USER], tools=[]))
    assert [type(e).__name__ for e in events] == ["ToolCall", "TextDelta", "Stopped"]
    assert events[0].name == "bash" and events[0].arguments == {"cmd": "ls"}
    assert events[2].reason == "tool_calls"


def test_a_backend_that_runs_out_of_script_fails_rather_than_lying():
    """An empty turn looks exactly like a model with nothing to say. A failure has to be a
    failure."""
    backend = model.Fake([])
    events = list(backend.complete(system="", messages=[USER], tools=[]))
    assert [type(e).__name__ for e in events] == ["Failed"]
    assert "scripted" in events[0].error


def test_the_request_shape_is_recorded_not_just_the_response():
    """A replay that shows only what came back cannot answer why the turn went that way."""
    backend = model.Fake([model.turn_text("a", "b"), model.turn_text("c")])
    tools = [model.ToolSpec("bash", "run a command", {"type": "object"})]
    list(backend.complete(system="SYS", messages=[USER], tools=tools, thinking="high",
                          max_tokens=512))
    call = backend.calls[0]
    assert call.system == "SYS" and call.tools == ("bash",)
    assert call.thinking == "high" and call.max_tokens == 512
    assert len(backend.calls) == 1, "one complete() is one recorded request"


def test_reasoning_level_travels_per_request_not_per_backend():
    """The summary call must be able to ask for something different from the run's turn, so
    the level belongs on the request."""
    backend = model.Fake([model.turn_text("x"), model.turn_text("y")])
    list(backend.complete(system="", messages=[USER], tools=[], thinking="high"))
    list(backend.complete(system="", messages=[USER], tools=[], thinking="off"))
    assert [c.thinking for c in backend.calls] == ["high", "off"]


def test_the_seam_is_structural():
    """No inheritance and no registration: anything with the two methods is a backend, and
    anything without them is not."""
    assert isinstance(model.Fake(), model.Backend)

    class NotABackend:
        pass

    assert not isinstance(NotABackend(), model.Backend)


def test_capabilities_are_declared_and_never_probed():
    caps = model.Caps(context_tokens=2048, supports_vision=True)
    assert model.Fake(caps=caps).capabilities() is caps
    # The default is the side that cannot send an image: a capability nobody declared is absent, not assumed.
    assert model.Caps(context_tokens=1).supports_vision is False

def test_a_scripted_turn_can_raise_at_the_point_it_reaches():
    """A backend that dies mid-stream is a shape several guarantees had to stub around.

    `Fake` scripts events, so a test that needed a raise wrote its own generator class. An exception
    instance in the script is raised when the stream reaches it -- after the events before it, which is the
    point: what the harness does next depends on what already arrived.
    """
    fake = model.Fake([[model.TextDelta("first"), RuntimeError("the wire died"),
                        model.TextDelta("never")]])
    stream = fake.complete(system="s", messages=[model.Message(role="user", content="go")], tools=[])
    seen = []
    with pytest.raises(RuntimeError, match="the wire died"):
        for event in stream:
            seen.append(event)
    assert [e.text for e in seen] == ["first"], "the raise jumped over the events before it"
