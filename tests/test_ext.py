"""The one registration path: assemble binds tool-contributors, wires watchers, and refuses a
run that would end up with a half-formed or double-meaning tool. A capability is not special because
it is "built in", so these drive the same surface a third-party extension would."""
import pytest

from sah.ext import Extension, assemble
from sah.record import Log, Refused


def one(x: int) -> str:
    """Do a thing with x."""
    return f"did {x}"


def two(y: str) -> str:
    """Echo y."""
    return y


# Neither toy touches a place, which is a declaration like any other: the tests below are about the
# registration path, and an undeclared tool is exactly what that path now refuses.
one.access, one.address = "none", None
two.access, two.address = "none", None
one.description = "Echo x."
two.description = "Echo y."


class Provider(Extension):
    tools = (one, two)


class Watcher(Extension):
    def __init__(self):
        self.seen = []

    def on(self, step):
        self.seen.append(step["kind"])


class Silent(Extension):          # tools only, keeps the no-op default `on`
    tools = (one,)


def test_assemble_builds_the_tool_table_from_contributors(tmp_path):
    tools = assemble([Provider()], Log(tmp_path / "log.jsonl"))
    assert set(tools) == {"one", "two"}


def test_a_tool_that_does_not_declare_its_access_is_refused_at_boot(tmp_path):
    """Confinement used to be a table keyed by tool *name*, so a contributor adding `def cat(path)` got a
    tool that touched the filesystem and was judged by nothing -- silently, because the table's own
    comment said "adding a tool means adding a line here" and nothing enforced it. A default of "allowed"
    makes the next tool's silence a policy hole; making it a boot failure makes the silence loud.
    """
    def cat(path: str) -> str:
        """Read a file."""
        return ""

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused, match="access"):
        assemble([Added()], Log(tmp_path / "log.jsonl"))


def test_a_refusal_to_declare_names_both_attributes(tmp_path):
    """A contributor reading the error should not have to open the source to fix it."""
    def cat(path: str) -> str:
        """Read a file."""
        return ""

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused) as got:
        assemble([Added()], Log(tmp_path / "log.jsonl"))
    assert "address" in str(got.value)


def test_an_access_outside_the_closed_set_is_refused(tmp_path):
    def cat(path: str) -> str:
        """Read a file."""
        return ""

    cat.access, cat.address = "readonly", "path"
    cat.description = "Read a file."

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused, match="access"):
        assemble([Added()], Log(tmp_path / "log.jsonl"))


def test_an_address_that_is_not_a_parameter_is_refused(tmp_path):
    """A typo in the address is the same hole as no declaration: the policy looks for an argument that
    never arrives, and finds nothing to check."""
    def cat(path: str) -> str:
        """Read a file."""
        return ""

    cat.access, cat.address = "read", "folder"
    cat.description = "Read a file."

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused, match="folder"):
        assemble([Added()], Log(tmp_path / "log.jsonl"))


def test_a_tool_judged_against_the_grants_must_name_where_the_place_is(tmp_path):
    """`read` and `write` mean "the policy judges this place", so an address of None is a contradiction:
    the tool would be judged on an empty argument and every call would refuse with "no address was
    given", which reads as a model error rather than as an authoring error caught at boot.
    """
    def cat() -> str:
        """Read the thing."""
        return ""

    cat.access, cat.address = "read", None
    cat.description = "Read the thing."

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused, match="no address"):
        assemble([Added()], Log(tmp_path / "log.jsonl"))


def test_a_tool_without_a_description_is_refused_at_boot(tmp_path):
    """`assemble` calls `spec_for` on every tool before it registers, and `spec_for` refuses a tool
    that never declared what the model is told it does. That is the production path this guards: a
    contributor that ships a tool with a docstring and no `description` attribute fails when the run
    starts, before a model is ever offered a tool it has to guess at. A docstring is maintainer prose
    and is not a fallback, because a fallback would keep the dual-audience path alive."""
    def cat(path: str) -> str:
        """Read a file. This docstring says plenty, and that is exactly the point: none of it is a
        declaration, so it is not enough."""
        return ""

    cat.access, cat.address = "read", "path"

    class Added(Extension):
        tools = (cat,)

    with pytest.raises(Refused, match="declares no description"):
        assemble([Added()], Log(tmp_path / "log.jsonl"))


def test_a_declared_tool_registers_and_reports_its_declaration(tmp_path):
    """The happy path, so the five refusals above cannot pass by refusing everything."""
    from sah.contract import declaration

    def cat(path: str) -> str:
        """Read a file."""
        return ""

    cat.access, cat.address = "read", "path"
    cat.description = "Read a file."

    class Added(Extension):
        tools = (cat,)

    tools = assemble([Added()], Log(tmp_path / "log.jsonl"))
    assert declaration(tools["cat"]) == ("read", "path")


def test_a_tool_only_extension_is_not_subscribed_as_a_watcher(tmp_path):
    log = Log(tmp_path / "log.jsonl")
    before = len(log._subscribers)
    assemble([Silent()], log)
    assert len(log._subscribers) == before, "the no-op default `on` must not subscribe a watcher"


def test_a_watcher_sees_each_step_through_the_log(tmp_path):
    log = Log(tmp_path / "log.jsonl")
    w = Watcher()
    assemble([w], log)
    log.append("message", "core", role="user", text="hi")
    assert w.seen == ["message"], "the `on` contract is notification of every step"


def test_watchers_and_tools_mix_in_one_assembly(tmp_path):
    log = Log(tmp_path / "log.jsonl")
    w = Watcher()
    tools = assemble([Provider(), w], log)
    assert set(tools) == {"one", "two"} and w.seen == []
    log.append("message", "core", role="user", text="hi")
    assert w.seen == ["message"]


def test_a_tool_with_an_underivable_schema_is_refused_at_boot(tmp_path):
    def broken(nope):
        """No type hint on nope."""
        return ""

    class Bad(Extension):
        tools = (broken,)

    with pytest.raises(Refused):
        assemble([Bad()], Log(tmp_path / "log.jsonl"))


def test_two_extensions_claiming_one_tool_name_are_refused(tmp_path):
    class AlsoOne(Extension):
        tools = (one,)

    with pytest.raises(Refused):
        assemble([Provider(), AlsoOne()], Log(tmp_path / "log.jsonl"))


def test_a_watcher_that_raises_does_not_stop_the_record(tmp_path):
    class Exploding(Extension):
        def on(self, step):
            raise RuntimeError("projection bug")

    log = Log(tmp_path / "log.jsonl")
    assemble([Exploding()], log)
    step = log.append("message", "core", role="user", text="still recorded")
    assert step["kind"] == "message", "a subscriber's bug must never lose a step"


def test_a_duck_typed_extension_needs_no_base_class(tmp_path):
    class Duck:
        tools = (one,)

    assert set(assemble([Duck()], Log(tmp_path / "log.jsonl"))) == {"one"}


# --- the door a file passes through (sah.cli._load_extension) ------------------

def _file(tmp_path, body, name="ext_file.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_the_door_loads_a_file_that_defines_extension(tmp_path):
    """The happy path: a file the operator named defines EXTENSION, an instance, and the door hands
    back the pairing the boot box and the record name. The file is self-contained the way a
    contributor's is: it names its own tool, not one from the test module."""
    from sah.cli import _load_extension

    body = ("def shout(msg: str) -> str:\n    return msg\n"
            "shout.access, shout.address = 'none', None\n"
            "shout.description = 'Shout.'\n"
            "class E:\n    tools = (shout,)\nEXTENSION = E()\n")
    path = _file(tmp_path, body)
    loaded_path, objects = _load_extension(path)
    assert loaded_path == path and len(objects) == 1
    assert set(assemble(objects, Log(tmp_path / "log.jsonl"))) == {"shout"}


def test_the_door_accepts_a_tuple_of_instances(tmp_path):
    from sah.cli import _load_extension

    body = ("def shout(msg: str) -> str:\n    return msg\n"
            "shout.access, shout.address = 'none', None\n"
            "shout.description = 'Shout.'\n"
            "class W:\n    def on(self, step): pass\n"
            "class E:\n    tools = (shout,)\nEXTENSION = (E(), W())\n")
    _path, objects = _load_extension(_file(tmp_path, body))
    assert len(objects) == 2


def test_the_door_refuses_a_file_that_defines_no_extension(tmp_path):
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="EXTENSION"):
        _load_extension(_file(tmp_path, "x = 1\n"))


def test_the_door_refuses_a_class_not_an_instance(tmp_path):
    """A class on EXTENSION would bind unbound methods to the table, and the first step would be
    handed to an `on` waiting for its `self` -- a failure that then reads as the step's fault. The
    door names the authoring mistake at boot instead."""
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="class"):
        _load_extension(_file(tmp_path,
                              "class E:\n    tools = ()\nEXTENSION = E\n"))


def test_the_door_refuses_an_object_with_neither_half(tmp_path):
    """An extension with neither a tool nor a watcher would load, assemble, do nothing, and print a
    line in the boot box promising a capability that is not there: a silent no-op door is
    operator error worth a named refusal."""
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="neither"):
        _load_extension(_file(tmp_path, "class E: pass\nEXTENSION = E()\n"))


def test_the_door_refuses_an_empty_tuple(tmp_path):
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="empty"):
        _load_extension(_file(tmp_path, "EXTENSION = ()\n"))


def test_the_door_refuses_a_file_that_does_not_import(tmp_path):
    """A file that raises at import is a named boot refusal, not a traceback: the operator gets the
    exception's words and the fix, and nothing ran."""
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="did not import"):
        _load_extension(_file(tmp_path, "raise RuntimeError('boom')\n"))


def test_the_door_refuses_a_file_that_is_not_there(tmp_path):
    from sah.cli import _load_extension

    with pytest.raises(Refused, match="no such file"):
        _load_extension(str(tmp_path / "never_written.py"))


def test_a_file_tool_that_never_declares_is_refused_by_the_gate(tmp_path):
    """The door reads the file; the gate holds it. A file whose tool never declared how it touches
    the world is refused at assembly exactly like a built-in's would be -- the promise that a
    capability buys no permission by being an extension, proven on the contributor's own file."""
    from sah.cli import _load_extension

    body = ("def shout(msg: str) -> str:\n    return msg\n"
            "class E:\n    tools = (shout,)\nEXTENSION = E()\n")
    _path, (ext_object,) = _load_extension(_file(tmp_path, body))
    with pytest.raises(Refused, match="access"):
        assemble([ext_object], Log(tmp_path / "log.jsonl"))


def test_assembly_hands_every_extension_the_transient_line(tmp_path):
    """The core's door: every assembled extension gets a `status` it can call — the console's
    transient line — and without a console handed in the line is a no-op that says nothing rather
    than a missing attribute that crashes a watcher. The extension proposes the words; the console,
    or its absence, decides what happens to them (the module docstring states what a line is and
    is not)."""
    class Chatter(Extension):
        tools = ()

        def on(self, step):
            self.status.say(f"a line about {step['kind']}")

    spoken = []

    class Spoken:
        def say(self, text):
            spoken.append(text)

    log = Log(tmp_path / "log.jsonl")
    assemble([Chatter()], log, status=Spoken())
    message = log.append("message", "model", role="assistant", text="hello")
    log.append("call", "model", refs=message["id"], name="speak", args={"text": "x"})
    assert spoken == ["a line about message", "a line about call"], \
        "the watcher's word reaches the console through the door"

    spoken.clear()
    chatter = Chatter()
    log2 = Log(tmp_path / "log2.jsonl")
    assemble([chatter], log2)
    assert chatter.status is not None, "the door is always there after assembly"
    message2 = log2.append("message", "model", role="assistant", text="hello")
    log2.append("call", "model", refs=message2["id"], name="speak", args={"text": "x"})
    assert spoken == [], "a no-op door says nothing, and the watcher still runs"
