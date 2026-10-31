"""A tool's schema comes from its signature, and the model's arguments are fitted at the
boundary rather than crashing inside the tool."""
import pytest
from typing import Annotated

from sah import contract
from sah.record import Refused


def greet(name: str, loud: bool = False) -> str:
    """A docstring is for whoever implements the tool: the model sees `greet.description`, and no
    byte of this text reaches it. Keeping them apart is the point — a docstring serving two
    audiences at once degrades for both."""
    return name.upper() if loud else name


greet.description = "Greet someone by name."


def variadic(cmd: str, *rest: str) -> str:
    """Takes *args."""
    return cmd


variadic.description = "Takes a command and any number of names."


def untyped(value) -> str:
    """No annotation."""
    return str(value)


untyped.description = "Take a value and return it as text."


def test_the_schema_is_the_signature():
    spec = contract.spec_for(greet)
    assert spec.name == "greet"
    assert spec.description == "Greet someone by name."
    assert spec.parameters["required"] == ["name"]
    assert spec.parameters["properties"] == {"name": {"type": "string"},
                                             "loud": {"type": "boolean"}}


def test_a_parameter_the_model_cannot_be_told_about_is_refused():
    with pytest.raises(Refused, match="no JSON-mappable annotation"):
        contract.spec_for(untyped)


def test_variadic_tools_are_refused():
    """*args means the schema cannot say what to send, so the tool is wrong, not the schema."""
    with pytest.raises(Refused, match=r"\*args"):
        contract.spec_for(variadic)


def test_an_argument_the_function_does_not_accept_is_refused_by_name():
    with pytest.raises(Refused, match="does not accept"):
        contract.arguments_for(greet, {"name": "x", "volume": 11})


def test_a_missing_required_argument_is_refused_before_the_call():
    with pytest.raises(Refused, match="requires"):
        contract.arguments_for(greet, {})


def test_a_bool_is_not_an_int_and_an_int_is_not_a_bool():
    with pytest.raises(Refused, match="must be a bool"):
        contract.arguments_for(greet, {"name": "x", "loud": 1})
    assert contract.arguments_for(greet, {"name": "x", "loud": True})["loud"] is True


# --- the visibility fix: field meaning and closed sets reach the model --------

Mode = Annotated[str, contract.Param("append or overwrite", ("append", "overwrite"))]


def note(path: Annotated[str, contract.Param("note path")],
         mode: Mode = "append") -> str:
    """The reasoning stays here, where a maintainer finds it: the model never reads a docstring, so
    this is where a decision's *why* lives without taxing the schema the model is sent."""
    return mode


note.description = "Write a note in append or overwrite mode; overwrite needs the note to exist."


def test_a_parameter_description_reaches_the_schema():
    """The bug this guards: a param shown as bare {"type":"string"} forces the model to guess. The
    meaning rides in the type hint and must land in the schema the model is sent."""
    props = contract.spec_for(note).parameters["properties"]
    assert props["path"] == {"type": "string", "description": "note path"}


def test_a_closed_value_set_is_advertised_in_the_schema():
    """mode: string with no enum is a coin flip for the model; the enum is how it learns the legal
    values without a trial call."""
    props = contract.spec_for(note).parameters["properties"]
    assert props["mode"]["enum"] == ["append", "overwrite"]


def test_a_value_outside_the_advertised_enum_is_refused_at_the_boundary():
    """The schema promised only append/overwrite; honour the promise before the tool sees garbage."""
    with pytest.raises(Refused, match="must be one of"):
        contract.arguments_for(note, {"path": "p", "mode": "destroy"})
    assert contract.arguments_for(note, {"path": "p", "mode": "append"})["mode"] == "append"


def test_the_description_is_declared_not_derived():
    """The model's contract is an attribute the author wrote, not text scraped from the source.
    The old rule — first docstring paragraph is the description — made the docstring serve the model
    and the maintainer at once, and every improvement to one audited the other; the split ends that.
    The docstring may say anything, and none of it is sent."""
    note.__doc__ = "A docstring the model never sees: it is for the maintainer alone."
    assert contract.spec_for(note).description == note.description

def test_a_tool_without_a_declared_description_is_refused_at_boot():
    """Every assembled tool passes through `spec_for` at boot (`sah.ext.assemble`), so a tool without
    a declared description fails before a model is ever offered it — silence is unbootable, the same
    idiom that refuses a tool that never declared its access. A fallback to the docstring would keep
    the dual-audience path alive, and that path is the hole."""
    def silent() -> str:
        """A docstring is not a description: it is maintainer prose, and no part of it reaches the
        model. The declaration the model sees has to be written where the declaration lives."""
        return ""
    with pytest.raises(Refused, match="declares no description"):
        contract.spec_for(silent)


# --- declared types are enforced at the boundary, not inside the tool -------------------------

def take(text: str, offset: int = 0, ratio: float = 1.0,
         tags: list = [], blob: dict = {}) -> str:
    """Take the thing."""
    return text


def test_a_string_parameter_that_receives_a_number_is_refused():
    """The hole was exact, not theoretical: `fs.read` hands its argument to `open`, and `open(0)` is
    file descriptor 0 — stdin. A backend sending `{"path": 0}` therefore read the operator's terminal
    input while holding no grant and touching no path, and the veto point was asked to judge a
    location the tool never touched. A declared `str` has to mean a str at the boundary."""
    with pytest.raises(Refused, match=r"take\.text must be a string, got int"):
        contract.arguments_for(take, {"text": 0})
    with pytest.raises(Refused, match=r"take\.text must be a string, got list"):
        contract.arguments_for(take, {"text": ["a"]})
    assert contract.arguments_for(take, {"text": "hi"})["text"] == "hi"


def test_a_number_quoted_as_a_string_is_coerced_because_models_quote_numbers():
    """`offset="5"` is the same intention as `offset=5`; refusing it spends a turn on a typo of form
    rather than of meaning, and small models quote numbers constantly. So the coercion is exact — an
    integral string, nothing else — and what reaches the tool is the int, so no tool has to be
    written to tolerate a string it was never told it might get."""
    assert contract.arguments_for(take, {"text": "hi", "offset": "5"})["offset"] == 5
    assert contract.arguments_for(take, {"text": "hi", "ratio": "1.5"})["ratio"] == 1.5
    for junk in ("5.5", "second", "", " 5", "0x10"):
        with pytest.raises(Refused, match="must be an integer"):
            contract.arguments_for(take, {"text": "hi", "offset": junk})


def test_a_number_that_is_not_a_number_is_refused_by_name():
    """The dot corpus is pinned on purpose: the guard that keeps `"."` (and its signed forms) out of
    the coercion sits one line above the digit check, and a reader who skips it — or an editor who
    drops it — leaves a `float` parameter accepting a string with no digit in it."""
    with pytest.raises(Refused, match="must be an integer, got list"):
        contract.arguments_for(take, {"text": "hi", "offset": []})
    for junk in (".", "+.", "-.", "1.2.3", "1e3", "high"):
        with pytest.raises(Refused, match="must be a number, got str"):
            contract.arguments_for(take, {"text": "hi", "ratio": junk})
    # one side of the dot may be empty — as long as the other carries the digits
    assert contract.arguments_for(take, {"text": "hi", "ratio": "1."})["ratio"] == 1.0
    assert contract.arguments_for(take, {"text": "hi", "ratio": ".5"})["ratio"] == 0.5


def test_a_collection_parameter_must_arrive_as_the_shape_it_declares():
    """A list handed to a `dict` parameter would fail deeper in, where the message names the
    standard library rather than the argument."""
    with pytest.raises(Refused, match=r"take\.tags must be a list, got str"):
        contract.arguments_for(take, {"text": "x", "tags": "a,b"})
    with pytest.raises(Refused, match=r"take\.blob must be a dict, got str"):
        contract.arguments_for(take, {"text": "x", "blob": "[]"})
    assert contract.arguments_for(take, {"text": "x", "tags": ["a"]})["tags"] == ["a"]
