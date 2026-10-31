"""The MCP client extension (extensions/mcp_client.py), driven through the door a run uses, against a
scripted MCP server on a pipe and a scripted MCP endpoint on a URL — both of them in this file, the
server started by the extension itself and the endpoint never reaching a socket at all.

Why a child process at all, in a battery whose rule is that it does not start one: the thing under test
is a client whose entire job is a child process and two pipes. A double that answered in-process would
test the wrapper functions and never the wire — the order the handshake's messages go out in, that an
answer is matched to its request by id, that closing stdin is what ends a server. So the process belongs
to the code under test, and this file starts nothing itself (the structural guard is on *this file*
calling a spawn). What keeps the battery's other promises: the server is stdlib and scripted here, and
its transport is a pipe, so no port is opened and no network is reached; the only clocks are ceilings a
test shortens rather than waits out.

The server speaks the wire exactly as the extension's docstring states it, and it *checks* the client as
it goes: an `initialize` whose params are not the promised ones, or a `tools/call` carrying a property
its schema never advertised or a null for one it did, is answered with a JSON-RPC error — so a client
that drifts fails loudly instead of passing a test that had stopped comparing anything.

The script is inlined rather than shared with the tree's opt-in live check: a battery file and an
out-of-tree check travel apart, and a shared fixture would make each depend on a folder the other may not
have. The duplication is the price of each standing alone, and it is one script.

The HTTP half states the same division of proof with a smaller mechanism, because there is less to reach:
every byte the client sends over HTTP goes through one function of the file under test (`_http_post`), so
the battery replaces that one function with a peer that plays the protocol from dicts and bytes. No
socket, no thread, no process, and no import of any of them here — which is the rule the guard at the end
of this module checks against this file's own text. What the peer proves is the client's protocol logic:
that a body and not a header decides JSON from a stream; that a stream's blocks are parsed and the message
carrying the request's id is the one found; that a session issued is a session echoed; that a redirect is
refused and its destination never asked; and that a status, a refused call and a blown deadline each
arrive as the sentence the moment of the run owes — a refusal at boot, the tool's own text during it. What
no peer can prove is the seam itself: that a real socket carries those bytes and a real endpoint answers
them, which is the opt-in check the operator runs against a server of their own. That is why this file
calls exactly one function that is not itself, and why that function is the transport.
"""
import inspect
import json
import pathlib
import sys

import pytest

from sah.cli import _load_extension
from sah.contract import declaration, spec_for
from sah.ext import Extension, assemble
from sah.record import Log, Refused

EXTENSION_FILE = pathlib.Path(__file__).resolve().parent.parent / "extensions" / "mcp_client.py"

# The scripted server. argv[1] chooses the behaviour, so one script covers a working server, a server
# that lies about one tool, a server that dies, and a server that says nothing:
#
#   normal     two honest tools (`echo`, `add`) and a notification sent before two of the answers, so a
#              client that mistook a notification for an answer would hang and say so
#   six        one tool carrying all six mappable types, three required, plus a closed enum
#   error      `tools/call` is refused with a JSON-RPC error
#   iserror    `tools/call` answers with a result that says the work failed
#   blocks     `tools/call` answers with two text blocks and one image block
#   blob       one honest tool, one with a property of type `null`, one with a property that declares no
#              type at all: the honest one must still load
#   oddnames   one honest tool, one with no description, one with a property name no signature can hold
#   die        exits with code 3 before it answers anything
#   silent     reads forever and never answers
#   hang       handshakes, then never answers a `tools/call`
_SERVER = r'''
"""A scripted MCP server on stdio: JSON-RPC 2.0, one message per line, nothing else on either channel."""
import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

ECHO = {"name": "echo", "description": "Say the given text back, exactly as it arrived.",
        "inputSchema": {"type": "object",
                        "properties": {"text": {"type": "string", "description": "The words."}},
                        "required": ["text"]}}
ADD = {"name": "add", "description": "Add two integers and answer the total.",
       "inputSchema": {"type": "object",
                       "properties": {"a": {"type": "integer", "description": "The first addend."},
                                      "b": {"type": "integer", "description": "The second addend."},
                                      "label": {"type": "string",
                                                "description": "Words to put before the total."}},
                       "required": ["a", "b"]}}
SIX = {"name": "everything", "description": "One property of every type the mapping carries.",
       "inputSchema": {"type": "object",
                       "properties": {"text": {"type": "string", "description": "Required string."},
                                      "count": {"type": "integer", "description": "Required integer."},
                                      "flag": {"type": "boolean", "description": "Required boolean."},
                                      "ratio": {"type": "number", "description": "Optional number."},
                                      "extra": {"type": "object", "description": "Optional object."},
                                      "items": {"type": "array", "description": "Optional array."},
                                      "mode": {"type": "string", "enum": ["left", "right"],
                                               "description": "A closed set."}},
                       "required": ["text", "count", "flag"]}}
TOOLS = {
    "normal": [ECHO, ADD],
    "six": [SIX],
    "error": [ECHO],
    "iserror": [ECHO],
    "blocks": [ECHO],
    "blob": [ECHO,
             {"name": "nullable", "description": "A property no signature can carry.",
              "inputSchema": {"type": "object",
                              "properties": {"thing": {"type": "null"}}, "required": ["thing"]}},
             {"name": "untyped", "description": "A property that never said.",
              "inputSchema": {"type": "object", "properties": {"thing": {}}, "required": []}}],
    "oddnames": [ECHO,
                 {"name": "silent_about_itself", "inputSchema": {"type": "object", "properties": {}}},
                 {"name": "spacey", "description": "A property name no signature holds.",
                  "inputSchema": {"type": "object",
                                  "properties": {"not a name": {"type": "string"}},
                                  "required": ["not a name"]}}],
    "die": [],
    "silent": [],
    "hang": [ECHO],
}

THE_NAME = {"error": "faulty", "iserror": "faulty", "blocks": "blocksy", "blob": "blobby",
            "oddnames": "oddnames", "six": "sixfold", "hang": "hanging"}.get(MODE, "demo")

# The introduction the transport promises: anything else is a client that drifted, and the only honest
# answer to that is an error the boot will refuse with.
INITIALIZE = {"protocolVersion": "2025-06-18", "capabilities": {},
              "clientInfo": {"name": "sah-mcp-client", "version": "0.1"}}


def send(message):
    sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
    sys.stdout.flush()


def told(rid, value):
    send({"jsonrpc": "2.0", "id": rid, "result": value})


def refused(rid, message):
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": message}})


def notify(text):
    send({"jsonrpc": "2.0", "method": "notifications/message",
          "params": {"level": "info", "data": text}})


def expect(name, arguments, allowed, required):
    """The check every call passes here: only advertised properties, and every required one present.

    An argument that arrived as null is named, because the client's rule is that a property the model
    did not fill is left out of the call rather than sent as a null the schema never promised."""
    unknown = sorted(set(arguments) - set(allowed))
    if unknown:
        raise ValueError("%s was sent %s, which it never advertised" % (name, ", ".join(unknown)))
    missing = sorted(key for key in required if key not in arguments)
    if missing:
        raise ValueError("%s is missing %s" % (name, ", ".join(missing)))
    nulled = sorted(key for key, value in arguments.items() if value is None)
    if nulled:
        raise ValueError("%s was sent null for %s" % (name, ", ".join(nulled)))


def do_echo(arguments):
    expect("echo", arguments, ["text"], ["text"])
    if not isinstance(arguments["text"], str):
        raise ValueError("echo.text must be a string")
    return arguments["text"]


def do_add(arguments):
    expect("add", arguments, ["a", "b", "label"], ["a", "b"])
    for key in ("a", "b"):
        if not isinstance(arguments[key], int) or isinstance(arguments[key], bool):
            raise ValueError("add.%s must be an integer" % key)
    total = arguments["a"] + arguments["b"]
    if "label" in arguments:
        if not isinstance(arguments["label"], str):
            raise ValueError("add.label must be a string")
        return "%s: %d" % (arguments["label"], total)
    return str(total)


def do_everything(arguments):
    expect("everything", arguments, ["text", "count", "flag", "ratio", "extra", "items", "mode"],
           ["text", "count", "flag"])
    if not isinstance(arguments["text"], str) or not isinstance(arguments["count"], int):
        raise ValueError("everything.text and .count must be a string and an integer")
    if not isinstance(arguments["flag"], bool):
        raise ValueError("everything.flag must be a boolean")
    for key, kind in (("ratio", float), ("extra", dict), ("items", list)):
        if key in arguments and not isinstance(arguments[key], kind):
            raise ValueError("everything.%s must be a %s" % (key, kind.__name__))
    # The property names that arrived are the answer: an optional the caller left out is not in here.
    return ",".join(sorted(arguments))


def call(rid, params):
    name = (params or {}).get("name")
    arguments = (params or {}).get("arguments") or {}
    if MODE == "error":
        refused(rid, "this server refuses every call")
        return
    if MODE == "iserror":
        told(rid, {"content": [{"type": "text", "text": "the thing the server could not do"}],
                   "isError": True})
        return
    if MODE == "blocks":
        told(rid, {"content": [{"type": "text", "text": "the first half"},
                               {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                               {"type": "text", "text": "the second half"}], "isError": False})
        return
    if MODE == "hang":
        return                                  # the answer never comes; the client's ceiling does
    handlers = {"echo": do_echo, "add": do_add, "everything": do_everything}
    if name not in handlers:
        refused(rid, "no such tool: %r" % (name,))
        return
    notify("a notification sent while the answer is still owed")
    try:
        told(rid, {"content": [{"type": "text", "text": handlers[name](arguments)}],
                   "isError": False})
    except ValueError as exc:
        refused(rid, str(exc))


def main():
    if MODE == "die":
        sys.exit(3)
    if MODE == "silent":
        for line in sys.stdin.buffer:
            pass                                # reads everything, answers nothing, forever
        return
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        rid = message.get("id")
        if rid is None:
            continue                            # a notification: nothing is owed back for it
        method = message.get("method")
        if method == "initialize":
            if message.get("params") != INITIALIZE:
                refused(rid, "the client introduced itself as %s"
                        % (json.dumps(message.get("params")),))
                continue
            told(rid, {"protocolVersion": INITIALIZE["protocolVersion"],
                       "capabilities": {"tools": {"listChanged": False}},
                       "serverInfo": {"name": THE_NAME, "version": "1.0"}})
        elif method == "tools/list":
            notify("a notification sent before the listing, while the listing is still owed")
            told(rid, {"tools": TOOLS[MODE]})
        elif method == "tools/call":
            call(rid, message.get("params"))
        else:
            refused(rid, "no such method: %r" % (method,))


main()
'''


def _script(tmp_path) -> str:
    """The scripted server as a file this run can name in the knob: the knob separates argv words with
    commas, so a server's source cannot travel inside it — only a path can."""
    path = pathlib.Path(tmp_path) / "scripted_mcp_server.py"
    path.write_text(_SERVER, encoding="utf-8")
    return str(path)


def _server(tmp_path, mode: str = "normal") -> list:
    """The argv the knob names for one scripted server in one behaviour."""
    return [sys.executable, _script(tmp_path), mode]


def _load(monkeypatch, *argvs):
    """Load the extension through the operator's own door, with `SAH_MCP_STDIO` naming these servers
    (each an argv), and return the loaded module beside the extension instance it built.

    Naming no argv means naming no knob at all: the shape of a box where nothing was asked for. Every MCP
    knob is cleared first, including the HTTP pair this file's second half sets, because a knob this
    machine's environment happens to carry would be a server the battery did not name — and the door's
    import is a boot, so it would be reached before any test could say otherwise.
    """
    for name in ("SAH_MCP_STDIO", "SAH_MCP_HTTP", "SAH_MCP_HTTP_HEADERS", "SAH_MCP_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    if argvs:
        monkeypatch.setenv("SAH_MCP_STDIO", ";".join(",".join(argv) for argv in argvs))
    _path, (client,) = _load_extension(str(EXTENSION_FILE))
    return sys.modules[type(client).__module__], client


def _tool(client, name):
    """One generated wrapper, by the name it was advertised under. The tools are generated, so a test
    reaches them through the table rather than importing a function that was never written down."""
    return next(tool for tool in client.tools if tool.__name__ == name)


def test_the_handshake_completes_and_the_advertised_tools_join_the_table(tmp_path, monkeypatch):
    """The whole promise in one pass of the door: launch, introduce, list — and the server's tools are
    in the run's table under the names, with the descriptions and the declarations every other tool
    carries, through the same gate the built-ins pass."""
    _module, client = _load(monkeypatch, _server(tmp_path))
    try:
        tools = assemble([client], Log(tmp_path / "log.jsonl"))
        assert set(tools) == {"echo", "add"}
        assert declaration(tools["echo"]) == ("self", None)
        spec = spec_for(tools["echo"])
        assert spec.description.startswith("Say the given text back")
        assert spec.parameters["properties"]["text"] == {"type": "string", "description": "The words."}
        assert spec.parameters["required"] == ["text"]
        assert client.findings == ("demo connected, 2 tools advertised",), \
            "the connection is a finding the core records"
    finally:
        client.close()


def test_a_server_initiated_notification_is_not_mistaken_for_an_answer(tmp_path, monkeypatch):
    """The scripted server sends a notification while the listing and while each call are still owed.
    An answer is matched by id, so a client that read a notification as the reply would take
    `notifications/message` for a tool result — and here it would hang, or answer the wrong thing."""
    _module, client = _load(monkeypatch, _server(tmp_path))
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo", "add"]
        assert _tool(client, "echo")(text="the words the notification did not answer") == \
            "the words the notification did not answer"
    finally:
        client.close()


def test_the_six_advertised_types_become_the_same_schema(tmp_path, monkeypatch):
    """`string`/`integer`/`number`/`boolean`/`object`/`array` become `str`/`int`/`float`/`bool`/`dict`/
    `list` and back again: what `sah.contract.spec_for` derives from a generated signature is the schema
    the server advertised — types, which properties are required, what each one means, and the closed
    set it named."""
    _module, client = _load(monkeypatch, _server(tmp_path, "six"))
    try:
        parameters = spec_for(_tool(client, "everything")).parameters
        assert {name: field["type"] for name, field in parameters["properties"].items()} == {
            "text": "string", "count": "integer", "flag": "boolean", "ratio": "number",
            "extra": "object", "items": "array", "mode": "string"}
        assert parameters["required"] == ["text", "count", "flag"]
        assert parameters["properties"]["count"]["description"] == "Required integer."
        assert parameters["properties"]["mode"]["enum"] == ["left", "right"]
        signature = inspect.signature(_tool(client, "everything"))
        assert signature.parameters["text"].default is inspect.Parameter.empty
        assert signature.parameters["ratio"].default is None, "optional means it may be left unsaid"
        # The server answers with the property names it was sent, which is the mapping proven on the
        # wire rather than only in the schema: three filled, four left unsaid, four absent.
        assert _tool(client, "everything")(text="t", count=1, flag=True) == "count,flag,text"
        assert _tool(client, "everything")(text="t", count=1, flag=True, ratio=0.5,
                                          extra={"k": 1}, items=[1, 2], mode="left") == \
            "count,extra,flag,items,mode,ratio,text"
    finally:
        client.close()


def test_an_optional_the_model_left_unfilled_is_left_out_of_the_call(tmp_path, monkeypatch):
    """An optional property the model did not fill is not sent at all — and the scripted server refuses
    any call that carries a null, so this goes red the moment a wrapper forwards its default instead of
    omitting it. Filling the same property sends it, unchanged and in place."""
    _module, client = _load(monkeypatch, _server(tmp_path))
    try:
        add = _tool(client, "add")
        assert add(a=2, b=3) == "5"
        assert add(a=2, b=3, label="the total") == "the total: 5"
    finally:
        client.close()


def test_a_call_round_trip_returns_the_servers_text(tmp_path, monkeypatch):
    """`tools/call` out, the server's own text back, verbatim: the pair of tools a real server would
    advertise, answering what they advertise."""
    _module, client = _load(monkeypatch, _server(tmp_path))
    try:
        assert _tool(client, "echo")(text="MCP over a pipe") == "MCP over a pipe"
        assert _tool(client, "add")(a=40, b=2) == "42"
    finally:
        client.close()


def test_a_json_rpc_error_comes_back_as_the_answer_that_names_it(tmp_path, monkeypatch):
    """The server answers the call with a JSON-RPC error, and the run is told: the code, the server's
    own words, and no exception anywhere near the loop — a failure the model can read is a failure the
    run can change course about."""
    _module, client = _load(monkeypatch, _server(tmp_path, "error"))
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "-32601" in answer and "this server refuses every call" in answer
        assert answer.startswith("mcp:"), "the answer says where it came from"
    finally:
        client.close()


def test_a_result_that_says_it_failed_comes_back_as_the_answer_that_names_it(tmp_path, monkeypatch):
    """`isError` is not a broken transport and is not reported as one: it is the server saying the work
    did not go off, and the text it gave with it is what the run reads."""
    _module, client = _load(monkeypatch, _server(tmp_path, "iserror"))
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "reported an error" in answer and "the thing the server could not do" in answer
    finally:
        client.close()


def test_a_non_text_block_is_stated_as_dropped_and_the_text_still_arrives(tmp_path, monkeypatch):
    """v1 carries text. A picture in the answer is named in the result, not quietly discarded: a run
    reading a short reply has to know the server sent more than it was given."""
    _module, client = _load(monkeypatch, _server(tmp_path, "blocks"))
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "the first half" in answer and "the second half" in answer
        assert "(one image block dropped — v1 is text)" in answer
    finally:
        client.close()


def test_two_servers_advertising_one_name_are_a_boot_refusal(tmp_path, monkeypatch):
    """The collision is the boot gate's to refuse — the same mechanism that stops an extension shadowing
    a built-in, with nothing new built for the MCP case. Two servers meaning different things by the word
    `echo` is a run no operator can reason about."""
    _module, client = _load(monkeypatch, _server(tmp_path), _server(tmp_path))
    try:
        with pytest.raises(Refused) as caught:
            assemble([client], Log(tmp_path / "log.jsonl"))
        assert "echo" in str(caught.value)
    finally:
        client.close()


def test_a_named_server_that_dies_before_the_handshake_is_a_boot_refusal_that_names_it(tmp_path,
                                                                                        monkeypatch):
    """The run must not start against a server that is already gone, and the refusal has to name which
    one — with the command the operator named it — because that line is what points at the fix."""
    argv = _server(tmp_path, "die")
    with pytest.raises(Refused) as caught:
        _load(monkeypatch, argv)
    message = str(caught.value)
    assert " ".join(argv) in message, "the refusal names the server by what the operator named it"
    assert "exit code 3" in message or "closed" in message, "and it names why"


def test_a_named_server_that_never_answers_is_a_boot_refusal_naming_the_ceiling(tmp_path, monkeypatch):
    """A boot that waits forever is a run that never started, so the handshake has a ceiling and running
    past it is the same named refusal. The ceiling is shortened here, not waited out at thirty seconds."""
    module, _client = _load(monkeypatch)               # loaded with nothing to connect to
    argv = _server(tmp_path, "silent")
    monkeypatch.setattr(module, "HANDSHAKE_SECONDS", 0.5)
    monkeypatch.setenv("SAH_MCP_STDIO", ",".join(argv))
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "the handshake got no answer within 0.5 s" in message
    assert " ".join(argv) in message


def test_a_tool_with_an_unrepresentable_property_is_skipped_and_the_rest_load(tmp_path, monkeypatch):
    """One tool carrying a `null` is no reason to lose the server: the unmappable tool is skipped and the
    reason carries the four parts a reader needs — server, tool, property, type — as a finding the
    record holds. A server with fifty tools must not die because one of them cannot be written as a
    signature."""
    _module, client = _load(monkeypatch, _server(tmp_path, "blob"))
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo"]
        findings = list(client.findings)
        assert any("blobby" in t for t in findings), "the server is named"
        assert any("nullable" in t and "'thing'" in t and "'null'" in t for t in findings), \
            "the unmappable tool is named with its property and its type"
        assert any("untyped" in t and "None" in t for t in findings), \
            "a property that declared no type is skipped too"
    finally:
        client.close()


def test_a_tool_no_signature_can_hold_is_skipped_and_the_rest_load(tmp_path, monkeypatch):
    """The two other ways a server can advertise something this harness cannot offer — a tool with no
    description (which `sah.contract.spec_for` refuses at boot) and a property whose name is not a name
    (which no signature can hold) — skip that tool, record which and why, and cost nothing else."""
    _module, client = _load(monkeypatch, _server(tmp_path, "oddnames"))
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo"]
        findings = list(client.findings)
        assert any("silent_about_itself" in t and "description" in t for t in findings)
        assert any("spacey" in t and "not a name" in t for t in findings)
    finally:
        client.close()


def test_no_server_named_loads_the_extension_and_contributes_nothing(tmp_path, monkeypatch, capfd):
    """The knob is an act, and a box where it was not pulled is not an error: the extension loads, has
    no tools, says nothing, and the run starts (sah/ext.py: an extension with no tools is legal)."""
    _module, client = _load(monkeypatch)
    assert assemble([client], Log(tmp_path / "log.jsonl")) == {}
    assert client.tools == ()
    assert client.findings == (), "no server, no finding: nothing found, nothing handed up"
    assert capfd.readouterr().err == "", "silence, not a shrug"


def test_the_watcher_closes_the_server_when_the_run_ends(tmp_path, monkeypatch):
    """The ending the log records is the ending the servers get: stdin closed, and a server that reads
    EOF as the session being over has gone by the time the step is written."""
    _module, client = _load(monkeypatch, _server(tmp_path))
    log = Log(tmp_path / "log.jsonl")
    assemble([client], log)
    assert client.servers[0].proc.poll() is None, "running while the run runs"
    log.append("ended", "core", reason="complete", exit_code=0)
    assert client.servers[0].proc.poll() is not None, "the child left with the run"


def test_a_call_that_outruns_its_ceiling_answers_with_the_ceiling(tmp_path, monkeypatch):
    """A hung server must not hang a run: the wrapper answers with the ceiling it hit, naming the tool,
    and the loop goes on with that as the tool result."""
    module, _client = _load(monkeypatch)
    monkeypatch.setenv("SAH_MCP_STDIO", ",".join(_server(tmp_path, "hang")))
    monkeypatch.setenv("SAH_MCP_TIMEOUT", "1")
    client = module.from_env()
    try:
        answer = _tool(client, "echo")(text="are you there")
        assert "within 1 s" in answer and "echo" in answer
        assert answer.startswith("mcp:")
    finally:
        client.close()


def test_the_extension_is_an_extension_whose_watcher_is_its_own(tmp_path, monkeypatch):
    """Cheap to state, and it is the difference between a watcher and a no-op: `sah.ext.assemble`
    subscribes an extension only when its `on` is not the contract's default."""
    _module, client = _load(monkeypatch)
    assert isinstance(client, Extension)
    assert type(client).on is not Extension.on


# --- the scripted endpoint ------------------------------------------------------------------------
#
# The HTTP half needs no server, because the client under test reaches HTTP through exactly one function
# (`_http_post`): replacing that function replaces the socket. Nothing is spawned to answer it, no port is
# opened for it, and the script is dicts and bytes inside this process.
#
# It checks the client as it goes, the way the scripted pipe server does: it answers only the endpoint (or
# endpoints) a test named, so a second URL is a failure and not a follow-up request; it insists that the
# introduction and the required request headers are the ones the transport promises; it answers with the
# id the client asked with, so a client that matched on anything else would hang rather than pass; it
# refuses a request after the first that did not echo a session id it issued; and its `tools/call` refuses
# a property a schema never advertised or a null standing in for an optional left unsaid.

_END = "http://scripted-endpoint.test/mcp"       # never resolved: the peer answers before any lookup

_HELLO = {"protocolVersion": "2025-06-18", "capabilities": {},
          "clientInfo": {"name": "sah-mcp-client", "version": "0.1"}}
"""The introduction the transport promises, which the peer holds the client to, as the pipe server does."""

_ECHO = {"name": "echo", "description": "Say the given text back, exactly as it arrived.",
         "inputSchema": {"type": "object",
                         "properties": {"text": {"type": "string", "description": "The words."}},
                         "required": ["text"]}}
_ADD = {"name": "add", "description": "Add two integers and answer the total.",
        "inputSchema": {"type": "object",
                        "properties": {"a": {"type": "integer", "description": "The first addend."},
                                       "b": {"type": "integer", "description": "The second addend."},
                                       "label": {"type": "string",
                                                 "description": "Words to put before the total."}},
                        "required": ["a", "b"]}}
_INTRO = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": False}},
          "serverInfo": {"name": "scripted", "version": "1.0"}}
_SHAPES = {"echo": (["text"], ["text"]), "add": (["a", "b", "label"], ["a", "b"])}


def _bytes(payload) -> bytes:
    """A message as the bytes it would travel as."""
    return json.dumps(payload).encode("utf-8")


def _text(text) -> dict:
    """One text block, in the shape a `tools/call` result carries it."""
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _answered(result, status: int = 200, headers=()):
    """Script entry: this `result`, as a JSON body, under the id the client asked with.

    The scripted headers are empty by default, which is the measured endpoint's shape — a response with no
    `Content-Type` on it at all — so a client that trusted that header would never parse the answer here.
    """
    def answer(message, url):
        return status, list(headers), _bytes({"jsonrpc": "2.0", "id": message.get("id"),
                                              "result": result})
    return answer


def _refused(code: int, text: str, status: int = 200):
    """Script entry: the server's considered refusal — a JSON-RPC `error`, which is an answer, and not a
    transport that broke."""
    def answer(message, url):
        return status, [], _bytes({"jsonrpc": "2.0", "id": message.get("id"),
                                   "error": {"code": code, "message": text}})
    return answer


def _status(status: int, body=b"", headers=()):
    """Script entry: this status, these headers, this body — whatever the wire would have carried."""
    def answer(message, url):
        blob = body if isinstance(body, bytes) else body.encode("utf-8")
        return status, list(headers), blob
    return answer


def _sse(result, *before, split: bool = False, status: int = 200):
    """Script entry: an SSE body — the blocks a test names, then this `result` under the asked-for id.

    `before` is where the parts of the format a client must step over go: a `:` comment, an `event:`/`id:`/
    `retry:` frame, or a notification block with no id (a dict is written as one `data:` line, a string is
    a block as it stands). `split` sends the message across two `data:` lines, because the rule worth
    pinning is that a block's data lines are joined with a newline before anything is parsed."""
    def answer(message, url):
        payload = json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
        half = len(payload) // 2
        # A data-line boundary is a newline, and a newline is not legal inside a JSON string: walk the
        # break forward to the next point outside the quotes, which is where a real stream would break.
        while half < len(payload) and payload[:half].count('"') % 2:
            half += 1
        data = f"data: {payload}" if not split else f"data: {payload[:half]}\ndata: {payload[half:]}"
        blocks = [one if isinstance(one, str) else f"data: {json.dumps(one)}" for one in before]
        return status, [("Content-Type", "text/event-stream")], (
            "\n\n".join(blocks + [data]) + "\n\n").encode("utf-8")
    return answer


def _raises(exc):
    """Script entry: the outbound call fails, and this is the error it fails with.

    The errors a socket raises travel out of `_http_post` untouched, so the script raises them too — that
    is the shape the client has to handle, rather than a wait a battery would have to sit through."""
    def answer(message, url):
        raise exc
    return answer


def _echoes(message, url):
    """Script entry: the endpoint's own `tools/call`, answering the two tools it advertises as they
    advertise — after the check every call passes here, which is the pipe server's check for the same
    reason: only advertised properties, every required one present, and no null standing in for an
    optional the caller left unsaid."""
    params = message.get("params") or {}
    name = params.get("name")
    arguments = params.get("arguments") or {}
    allowed, required = _SHAPES.get(name, ([], []))
    unknown = sorted(set(arguments) - set(allowed))
    missing = sorted(key for key in required if key not in arguments)
    nulled = sorted(key for key, value in arguments.items() if value is None)
    assert not (unknown or missing or nulled), (
        f"{name} was called with {arguments!r}: unknown={unknown} missing={missing} null={nulled}")
    total = str(arguments["a"] + arguments["b"]) if name == "add" else arguments["text"]
    return 200, [], _bytes({"jsonrpc": "2.0", "id": message["id"], "result": _text(total)})


_DEFAULT = {"initialize": _answered(_INTRO),
            "notifications/initialized": _status(202),
            "tools/list": _answered({"tools": [_ECHO, _ADD]}),
            "tools/call": _echoes}
"""The endpoint every HTTP test gets unless a script says otherwise: an honest handshake, a 202-empty
notification, two advertised tools, and calls that answer what they advertise."""


class _Peer:
    """The scripted endpoint: what `_http_post` becomes when there is no socket behind it.

    A script maps a method name to one answer, or to a list of them consumed in order and then repeated
    from the last, so one script can play a whole handshake and then the shape one call is about. Each
    answer is a callable of the request message and the URL it arrived at (two endpoints named in one knob
    answer apart), returning the `(status, header pairs, body bytes)` triple the real function returns.
    `script` is a plain dict, editable after the peer is installed — an error class that lives in the file
    under test can only be named once that file has been loaded.

    `session`, when a test gives one, is the session id this endpoint issues: it goes on every answer, and
    any request after the first that did not echo it is a failure named for the header, which is what a
    client that forgot it deserves — a message about the header, not a hang or an unrelated error.
    """

    def __init__(self, script=None, session=None, urls=(_END,)):
        self.script = dict(_DEFAULT)
        self.script.update(script or {})
        self.session = session
        self.urls = tuple(urls)
        self.calls: list[dict] = []      # one per request: url, headers, decoded message, ceiling

    def __call__(self, url, headers, body, timeout):
        assert url in self.urls, f"the client posted to {url!r}, which no test named"
        message = json.loads(body)
        self.calls.append({"url": url, "headers": dict(headers), "message": message,
                           "timeout": timeout})
        assert headers.get("Content-Type") == "application/json", "the POST the transport requires"
        assert "text/event-stream" in headers.get("Accept", ""), "a client must say it can read a stream"
        if self.session is not None and len(self.calls) > 1:
            assert headers.get("Mcp-Session-Id") == self.session, (
                f"this endpoint issued session {self.session!r} and request {len(self.calls)} "
                f"arrived without echoing it: {sorted(headers)}")
        if message.get("method") == "initialize":
            assert message.get("params") == _HELLO, (
                f"the client introduced itself as {json.dumps(message.get('params'))}")
        entry = self.script.get(message.get("method"))
        assert entry is not None, f"the script has no answer for {message.get('method')!r}"
        answers = entry if isinstance(entry, list) else [entry]
        chosen = answers.pop(0) if len(answers) > 1 else answers[0]
        status, response_headers, response_body = chosen(message, url)
        if self.session is not None:
            response_headers = list(response_headers) + [("Mcp-Session-Id", self.session)]
        return status, list(response_headers), response_body


def _load_http(monkeypatch, script=None, urls=(_END,), session=None, headers=None, stdio_argv=()):
    """The extension, loaded through the door and then booted against a scripted endpoint.

    Two steps, and the reason is the door's own shape: `_load_extension` runs the file, and the file's last
    line is `EXTENSION = from_env()`, so the boot happens during the import and a peer can only be put in
    place afterwards. So the door loads the module with every knob empty — nothing named, nothing reached —
    the peer then replaces the one function in it that would touch a socket, and the boot is run again the
    way the file runs it: `from_env()`, this time naming the endpoint. The instance the import built is
    closed on the way out, because the boot under test is the next one and two live instances of one
    extension in one process is a leak rather than a case.

    Everything else is unchanged on the way through: the same `from_env`, the same `McpClient`, the same
    wrappers and the same boot gate. Only the socket is scripted, and that is the whole point of the seam
    being one function.
    """
    module, built = _load(monkeypatch, *stdio_argv)
    built.close()                                   # the door's own client: nothing named, nothing to end
    peer = _Peer(script=script, session=session, urls=urls)
    monkeypatch.setattr(module, "_http_post", peer)
    monkeypatch.setenv("SAH_MCP_HTTP", ";".join(urls))
    if headers is not None:
        monkeypatch.setenv("SAH_MCP_HTTP_HEADERS", headers)
    return module, peer


def test_a_scripted_endpoint_handshakes_and_its_tools_join_the_table(tmp_path, monkeypatch):
    """The HTTP promise in one pass: one POST per message, in the order the transport fixes, the answer
    read out of a body whose response carried no `Content-Type` (the measured endpoint's shape), the name
    taken from `serverInfo`, and the advertised tools in the run's table through the same gate as
    everything else — the transport under them invisible to the declaration as to the model.
    """
    module, peer = _load_http(monkeypatch)
    client = module.from_env()
    try:
        assert [entry["message"]["method"] for entry in peer.calls] == [
            "initialize", "notifications/initialized", "tools/list"]
        assert "id" not in peer.calls[1]["message"], "a notification is asked for nothing"
        assert client.servers[0].name == "scripted", "the name the endpoint gave, not the URL it got"
        assert client.servers[0].session is None, "an endpoint that issues no session is served"
        tools = assemble([client], Log(tmp_path / "log.jsonl"))
        assert set(tools) == {"echo", "add"}
        assert declaration(tools["echo"]) == ("self", None)
        assert client.findings == ("scripted connected, 2 tools advertised",), \
            "the connection is a finding the core records"
    finally:
        client.close()


def test_an_endpoint_that_issues_a_session_has_it_echoed_on_every_later_request(monkeypatch):
    """The other half of the session rule: a server that does issue one is answered by a client that
    carries it on every later request. The peer refuses any request after the first that arrives without
    it, so a client that stored the header and forgot to send it fails here naming the header — and the
    first request is asserted to carry none, because it cannot echo a session it has not been given."""
    module, peer = _load_http(monkeypatch, session="session-7")
    client = module.from_env()
    try:
        assert client.servers[0].session == "session-7"
        assert "Mcp-Session-Id" not in peer.calls[0]["headers"]
        assert _tool(client, "echo")(text="with a session") == "with a session"
        assert [entry["headers"].get("Mcp-Session-Id") for entry in peer.calls[1:]] == \
            ["session-7"] * (len(peer.calls) - 1)
    finally:
        client.close()


def test_a_listing_answered_as_a_stream_is_found_and_the_rest_is_skipped(monkeypatch):
    """An SSE answer is a run of blocks and one of them is the reply. Ahead of it here: a `:` comment, an
    `event:`/`id:`/`retry:` frame, and a whole notification block with no id — none of which carries a
    message this request asked for. The answer's own JSON is then split across two `data:` lines, because
    the rule is that a block's data lines are joined with a newline and only then parsed."""
    script = {"tools/list": _sse({"tools": [_ECHO, _ADD]},
                                  ": a comment the client has to step over",
                                  "event: message\nid: 17\nretry: 3000",
                                  {"jsonrpc": "2.0", "method": "notifications/progress",
                                   "params": {"progressToken": 1, "progress": 0}},
                                  split=True)}
    module, peer = _load_http(monkeypatch, script)
    client = module.from_env()
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo", "add"]
        assert len(peer.calls) == 3, "the listing was answered once, by the one stream"
    finally:
        client.close()


def test_a_call_answered_as_a_stream_and_one_answered_as_json_give_the_same_text(monkeypatch):
    """Two shapes of the same answer, one result: the server's text verbatim, whether it came as a JSON
    body or inside a stream. The first call is answered over SSE and the second as a body, and both are
    the endpoint's own echo, so this is the pipe half's round trip read through both HTTP shapes."""
    def streamed(message, url):
        return _sse(_text(message["params"]["arguments"]["text"]))(message, url)

    module, _peer = _load_http(monkeypatch, {"tools/call": [streamed, _echoes]})
    client = module.from_env()
    try:
        assert _tool(client, "echo")(text="over a stream") == "over a stream"
        assert _tool(client, "add")(a=40, b=2) == "42"
    finally:
        client.close()


def test_a_call_the_endpoint_says_it_failed_names_the_failure_and_raises_nothing(monkeypatch):
    """`isError` is the server saying the work did not go off, on HTTP exactly as on the pipe: the answer
    the run reads, with the server's words beside it, and no exception reaching the loop."""
    module, _peer = _load_http(monkeypatch, {"tools/call": _answered(
        {"content": [{"type": "text", "text": "the thing the endpoint could not do"}],
         "isError": True})})
    client = module.from_env()
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "reported an error" in answer and "the thing the endpoint could not do" in answer
    finally:
        client.close()


def test_a_json_rpc_error_to_an_http_call_comes_back_as_the_answer_that_names_it(monkeypatch):
    """A JSON-RPC error is the endpoint's considered answer to the call, and the run is told in its terms:
    the code, the message, and the prefix that says where the sentence came from."""
    module, _peer = _load_http(monkeypatch,
                               {"tools/call": _refused(-32601, "this endpoint refuses every call")})
    client = module.from_env()
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "-32601" in answer and "this endpoint refuses every call" in answer
        assert answer.startswith("mcp:"), "the answer says where it came from"
    finally:
        client.close()


def test_a_non_text_block_in_an_http_answer_is_stated_as_dropped(monkeypatch):
    """Text is still all this client carries, and the words it uses to say so are the pipe half's, because
    the reader of the answer does not care which wire the picture arrived on: it cares that one did."""
    module, _peer = _load_http(monkeypatch, {"tools/call": _answered(
        {"content": [{"type": "text", "text": "the first half"},
                     {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                     {"type": "text", "text": "the second half"}], "isError": False})})
    client = module.from_env()
    try:
        answer = _tool(client, "echo")(text="anything")
        assert "the first half" in answer and "the second half" in answer
        assert "(one image block dropped — v1 is text)" in answer
    finally:
        client.close()


def test_a_notification_answered_202_with_an_empty_body_costs_no_line_of_stderr(monkeypatch, capfd):
    """202 and nothing else is what the measured endpoint answers the `initialized` notification with, and
    the only honest response to the normal shape of an answer is silence on the operator stream: the boot's
    one line about the connection it made goes to the record where boot facts belong, and nothing is said
    about a step it did not mind."""
    module, _peer = _load_http(monkeypatch)
    client = module.from_env()
    try:
        assert capfd.readouterr().err == "", "the routine boot owes the operator stream nothing"
        assert client.findings == ("scripted connected, 2 tools advertised",), \
            "and the connection is where a boot fact belongs: the record"
    finally:
        client.close()


def test_a_notification_answered_with_a_json_rpc_error_is_recorded_and_the_boot_goes_on(monkeypatch,
                                                                                        capfd):
    """A 200 carrying an error to a message that asked for nothing is recorded and stepped over: the
    endpoint answered the handshake, which is what had to be proven, and it is the tool listing that
    decides whether this server has anything to offer. Refusing the boot here would be refusing a
    server that listed its tools a moment later. The complaint costs the operator stream no line: it
    is in the record, beside where the connection is recorded."""
    module, _peer = _load_http(monkeypatch, {"notifications/initialized": _refused(
        -32601, "the endpoint disliked the notification")})
    client = module.from_env()
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo", "add"], "the boot continued"
        assert any("the endpoint disliked the notification" in text for text in client.findings), \
            "and the complaint is in the record"
        assert capfd.readouterr().err == "", "a stepped-over complaint costs the operator stream no line"
    finally:
        client.close()


def test_a_boot_stage_405_is_a_named_refusal_naming_the_endpoint_and_the_status(monkeypatch):
    """A server of the older two-channel dialect answers a Streamable HTTP POST this way, and what it gets
    is the refusal rather than a client guessing at a dialect it was never told to speak: the URL the
    operator typed, the status that answered it, and the body's own words — the same shape of sentence the
    pipe half says about an argv it could not launch."""
    module, _peer = _load_http(monkeypatch, {"initialize": _status(405, "Method Not Allowed")})
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert _END in message, "the refusal names the server by what the operator named it"
    assert "405" in message and "Method Not Allowed" in message, "and it names why"


def test_a_500_whose_body_is_a_json_rpc_error_names_that_error_and_the_status(monkeypatch):
    """An error status whose body has the sense of a JSON-RPC error is quoted as that error and not as two
    kilobytes of whatever else was in there: the server's message, and the status that carried it."""
    script = {"tools/list": _status(500, json.dumps({"jsonrpc": "2.0", "id": 2,
                                                     "error": {"code": -32603,
                                                               "message": "the listing is broken"}}))}
    module, _peer = _load_http(monkeypatch, script)
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "500" in message and "the listing is broken" in message and _END in message


def test_an_endpoint_that_refuses_the_connection_is_a_boot_refusal_naming_it(monkeypatch):
    """The failure a wrong port or a stopped server produces: no answer at all, and a refusal that quotes
    the endpoint and names the failure the way the socket named it, so the operator can see which of the
    two it was."""
    module, peer = _load_http(monkeypatch,
                              {"initialize": _raises(ConnectionRefusedError("refused"))})
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert _END in message and "ConnectionRefusedError" in message
    assert len(peer.calls) == 1, "one attempt, refused, and no retry invented"


def test_an_endpoint_that_fails_closes_the_stdio_sibling_that_had_connected(tmp_path, monkeypatch):
    """A run that will not start must not start with a server running underneath it, and a failure on one
    transport can arrive after the other transport's server was already handshaken. The witness below is
    why this is a fact and not a race: the close is recorded by the transport as it happens, before the
    refusal leaves `from_env`, rather than inferred from when an operating system notices a closed pipe."""
    argv = _server(tmp_path)
    module, _peer = _load_http(monkeypatch, {"initialize": _raises(ConnectionRefusedError("refused"))},
                               stdio_argv=(argv,))
    closed: list = []

    class Witness(module.StdioServer):
        """A stdio transport that says when it was closed, so the ordering is recorded, not guessed."""

        def close(self, *args):
            closed.append(self.command)
            return super().close(*args)

    monkeypatch.setattr(module, "StdioServer", Witness)
    with pytest.raises(Refused):
        module.from_env()
    assert closed == [" ".join(argv)], "the sibling was closed before the refusal left"


def test_a_redirect_is_refused_and_the_url_it_names_is_never_asked(monkeypatch):
    """The operator named a destination; a `Location` is another one. Following it would carry the run,
    its calls and its headers to a URL nobody typed, so the redirect is reported and the exchange ends —
    and the peer, which refuses any URL a test did not name, is what proves the second request the
    tempting thing to send was never sent."""
    module, peer = _load_http(monkeypatch, {"initialize": _status(
        302, "", [("Location", "http://somewhere-else.test/mcp")])})
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "302" in message and _END in message
    assert "somewhere-else.test" in message, "the refusal says where it was being sent"
    assert len(peer.calls) == 1, "one POST: the location was named, never asked"


def test_a_property_with_no_type_skips_its_tool_and_the_endpoint_still_loads(monkeypatch):
    """Measured on a real endpoint: a property whose schema says `anyOf` carries no `type` word at all, and
    that is a property no signature can carry. It is skipped with the reason, like any other, and the
    endpoint's other tools load — the mapping is not widened for unions, because a tool the harness
    half-understands is a contract the model has to guess at."""
    union = {"name": "slider", "description": "Set a slider, or say nothing.",
             "inputSchema": {"type": "object",
                             "properties": {"level": {"anyOf": [{"type": "number"},
                                                                {"type": "null"}]}},
                             "required": []}}
    module, _peer = _load_http(monkeypatch, {"tools/list": _answered({"tools": [_ECHO, union]})})
    client = module.from_env()
    try:
        assert [tool.__name__ for tool in client.tools] == ["echo"]
        findings = list(client.findings)
        assert any("scripted" in t for t in findings), "the endpoint is named"
        assert any("slider" in t and "'level'" in t and "None" in t for t in findings)
    finally:
        client.close()


def test_two_endpoints_named_in_one_knob_both_contribute_to_one_table(tmp_path, monkeypatch):
    """One knob, two endpoints, one tool table: each is handshaken in turn, each contributes what it
    advertises, and both are closed by the one ending. A name the two of them both claim would be the
    boot gate's collision — the pipe servers meet that rule and these meet it here, with nothing
    transport-specific about it."""
    one, two = "http://one-endpoint.test/mcp", "http://two-endpoint.test/mcp"

    def listing(message, url):
        return _answered({"tools": [_ECHO] if url == one else [_ADD]})(message, url)

    module, _peer = _load_http(monkeypatch, {"tools/list": listing}, urls=(one, two))
    client = module.from_env()
    try:
        assert [server.command for server in client.servers] == [one, two]
        assert set(assemble([client], Log(tmp_path / "log.jsonl"))) == {"echo", "add"}
        assert client.findings.count("scripted connected, 1 tools advertised") == 2
    finally:
        client.close()


def test_a_header_the_operator_named_arrives_on_every_request(monkeypatch):
    """`SAH_MCP_HTTP_HEADERS` is the door a credential comes through: what the operator names is sent on
    every request to every endpoint they named, split on the first colon so a value full of colons
    survives, and fixed at boot — so a model can call a tool that needs the key, and can neither read the
    key nor point it at another host."""
    module, peer = _load_http(monkeypatch, headers="Authorization: Bearer token-7; X-Trace: on")
    client = module.from_env()
    try:
        assert _tool(client, "echo")(text="authenticated") == "authenticated"
        for entry in peer.calls:
            assert entry["headers"]["Authorization"] == "Bearer token-7"
            assert entry["headers"]["X-Trace"] == "on"
    finally:
        client.close()


def test_a_header_pair_with_no_colon_is_refused_before_anything_is_reached(monkeypatch):
    """A half-written credential is not something to guess at, and it is refused before an endpoint is
    asked or a server launched: a boot that was going to fail should fail with nothing running, and a boot
    that quietly dropped the header would send unauthenticated requests at a server the operator believed
    they had authorised."""
    module, peer = _load_http(monkeypatch, headers="Bearer token-7 with no colon and no name")
    with pytest.raises(Refused) as caught:
        module.from_env()
    assert "SAH_MCP_HTTP_HEADERS" in str(caught.value)
    assert peer.calls == [], "the knob was read before anything was reached"


def test_an_endpoint_named_without_an_http_scheme_is_a_named_refusal(monkeypatch):
    """`ws://`, `ftp://`, and a bare host and port with the scheme left out are not destinations this
    client will reinterpret into one: the naming is read exactly, and the refusal quotes what was named."""
    module, peer = _load_http(monkeypatch, urls=("ftp://somewhere-else.test/mcp",))
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "ftp://somewhere-else.test/mcp" in message and "http" in message
    assert peer.calls == [], "nothing was asked of an endpoint the client refused to name a destination"


def test_a_stream_that_dribbles_past_its_deadline_answers_with_the_ceiling(monkeypatch):
    """The wall deadline is a timeout and not a hang, and at call stage it is the tool's own text: the
    ceiling it ran out of, the tool it ran out on, and a loop that goes on. The peer raises what the body
    read raises when the deadline passes between two lines — the failure `_http_post` promises, rather
    than a wait the battery would have to sit through."""
    module, peer = _load_http(monkeypatch)
    peer.script["tools/call"] = _raises(module._TimedOut)
    client = module.from_env()
    try:
        answer = _tool(client, "echo")(text="are you there")
        assert "within 120 s" in answer and "echo" in answer
        assert answer.startswith("mcp:"), "the answer says where it came from"
    finally:
        client.close()


def test_an_endpoint_that_never_answers_the_handshake_refuses_naming_the_ceiling(monkeypatch):
    """The same blown deadline at boot is the same named refusal the pipe half gives a server that reads
    and never answers, quoting the ceiling it ran out of — shortened here rather than waited out."""
    module, peer = _load_http(monkeypatch)
    peer.script["initialize"] = _raises(module._TimedOut)
    monkeypatch.setattr(module, "HANDSHAKE_SECONDS", 0.5)
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "the handshake got no answer within 0.5 s" in message
    assert _END in message


def test_a_stream_that_ends_without_the_answer_is_a_named_wire_failure(monkeypatch):
    """A well-formed stream of messages that never carries the one asked for is not an answer and not a
    timeout: the answer is not coming, and the boot says which endpoint it was waiting on. A notification
    is the shape this happens with in practice, and dropping it is right — but dropping the last message
    in the stream cannot end the wait silently."""
    quiet = "data: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                                   "params": {"requestId": "2"}}) + "\n\n"
    module, _peer = _load_http(monkeypatch, {"tools/list": _status(
        200, quiet, [("Content-Type", "text/event-stream")])})
    with pytest.raises(Refused) as caught:
        module.from_env()
    message = str(caught.value)
    assert "nothing carrying the id it asked for" in message and _END in message


def test_the_runs_ending_closes_both_transports(tmp_path, monkeypatch):
    """The step the log records ends every server the extension owns, of either transport, and the
    endpoint's turn is the flag and nothing else: closing it asks the endpoint for nothing, because the
    connection each message opened was closed with its answer."""
    module, peer = _load_http(monkeypatch)
    client = module.from_env()
    log = Log(tmp_path / "log.jsonl")
    assemble([client], log)
    server = client.servers[0]
    assert server._closed is False, "live while the run runs"
    log.append("ended", "core", reason="complete", exit_code=0)
    assert server._closed is True, "the endpoint was closed with the run"
    assert len(peer.calls) == 3, "and closing it asked it nothing"
    client.close()


def test_headers_named_without_an_endpoint_name_nothing(monkeypatch, capfd):
    """A header is not a destination. Naming a credential and no endpoint is a knob half-pulled: the
    extension loads, contributes nothing, says nothing, and reaches nothing — because the act that widens
    the run is naming somewhere to go, and there is nowhere to go."""
    module, peer = _load_http(monkeypatch, headers="Authorization: Bearer token-7", urls=())
    client = module.from_env()
    assert client.servers == [] and client.tools == ()
    assert capfd.readouterr().err == "", "silence, not a shrug"
    assert peer.calls == []


def test_empty_slots_between_semicolons_name_no_endpoint(monkeypatch):
    """The knob is a table, and a blank row in a table is not a row: `;  ;url;` names one endpoint, and
    the boot connects to one. The pipe knob's rule, enforced by the same absence of machinery."""
    module, _peer = _load_http(monkeypatch, urls=("", "   ", _END))
    client = module.from_env()
    try:
        assert len(client.servers) == 1
        assert client.servers[0].command == _END
    finally:
        client.close()


def test_the_battery_itself_starts_no_process_and_opens_no_socket():
    """The guard this file's first paragraph promises, stated as a rule on this file's text: the process
    belongs to the code under test and the socket belongs to the code under test, and a test that spawned
    its own server or opened its own port would be testing something else while appearing to test this.

    The names are assembled from halves, because a file that wrote down the whole name of the thing it is
    forbidding would forbid itself: this guard reads this file, and the strings it looks for appear here
    only in the halves they are built from. The rule is on the import statement and not on the word,
    because this file has always spoken about the socket it does not open."""
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    forbidden = [f"import {head + tail}" for head, tail in (("sub", "process"), ("soc", "ket"),
                                                             ("http.", "server"))]
    # The from-import is an import statement too, and a guard that missed it would be rule-in-prose,
    # rule-in-practice.
    forbidden += [f"from {head + tail} import" for head, tail in (("sub", "process"), ("soc", "ket"),
                                                                  ("http.", "server"))]
    forbidden.append("Po" + "pen")
    for name in forbidden:
        assert name not in source, (
            f"{name} belongs to the client under test, not to the file that tests it")
