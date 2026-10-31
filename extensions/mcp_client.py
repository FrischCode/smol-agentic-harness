"""extensions.mcp_client — the Model Context Protocol client extension: the servers the operator names,
over either transport, and the tools they advertise joining the run's table.

What it is. The operator names one or more MCP servers in the environment: a command to launch, or an
endpoint to post to. At boot — while this file is being imported and its `EXTENSION` built — each named
server is reached, the handshake is run, the tools the server advertises are read, and one generated
wrapper per tool is contributed through the same door and the same boot gate (sah.ext.assemble) every
other tool passes: the signature is the schema (sah/contract.py), the declaration is read before a model
is ever offered the tool, and a name a built-in or another server already holds is the existing named
refusal, not a shadow. The model sees the server's own tool names, the server's own descriptions, and the
types it can send. It never sees the transport.

Two transports, one seam. `StdioServer` and `HttpServer` are the whole of the difference between a
launched child on two pipes and an endpoint on a URL, and what they share is five methods and three
attributes: `connect()`, `request(method, params, ceiling, label)`, `notify(method)`, `call(tool,
arguments)`, `close()`, and `command` / `name` / `advertised` (`_Server` states the seam). `McpClient`,
`_wrapper`, `from_env` and `_servers` hold a list of that shape and never ask which class an entry is, so
the transport is the pipe and swapping it changes no other line here. A third transport — the older
two-channel dialect, or whatever replaces this one — is another class on the same five methods.

Generated wrappers, precisely. An advertised `inputSchema` becomes a signature: `string`→`str`,
`integer`→`int`, `number`→`float`, `boolean`→`bool`, `object`→`dict`, `array`→`list`; a property named
in `required` gets no default, an optional one defaults to `None` and is left out of the call's
`arguments` when it arrives as `None` (an argument the model did not send is not an argument the server
should be told was null); a property `description` and a closed `enum` ride into the schema on `Param`,
which is where this harness puts a field's meaning. A tool that cannot be mapped is **skipped** and the
reason is handed up as a finding the core records beside the load notice — server, tool, property,
type — because a server with fifty tools must not die over one unrepresentable blob, and a tool whose
schema was half-written is a contract the model has to guess at. The same applies to a property this
cannot name in a signature (an empty name, a name with
a space, a Python keyword) and to a tool advertised with no description: `sah.contract.spec_for` refuses
both at boot, and a skip that says why keeps the other forty-nine. A real server's unions show up here
as a property with no `type` word at all (the schema says `anyOf` instead), and they are skipped like any
other property no signature can carry: the endpoint's other tools load, and the reason is in the record.

The wire (MCP stdio transport: JSON-RPC 2.0, one message per line on the child's stdin and stdout, no
other bytes on either channel — the client writes UTF-8 and reads UTF-8):

    → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18",
         "capabilities":{},"clientInfo":{"name":"sah-mcp-client","version":"0.1"}}}
    ← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"...","capabilities":{...},
         "serverInfo":{"name":"...","version":"..."}}}
    → {"jsonrpc":"2.0","method":"notifications/initialized"}          (no id; nothing comes back)
    → {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    ← {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"...","description":"...",
         "inputSchema":{"type":"object","properties":{...},"required":[...]}}]}}
    → {"jsonrpc":"2.0","id":N,"method":"tools/call","params":{"name":"...","arguments":{...}}}
    ← {"jsonrpc":"2.0","id":N,"result":{"content":[{"type":"text","text":"..."}],"isError":false}}
    ← {"jsonrpc":"2.0","id":N,"error":{"code":-32601,"message":"..."}}

The wire (MCP Streamable HTTP, protocol 2025-06-18: one endpoint, one message per POST, the answer on
the same response — the handshake order, the id matching, the meaning of an `error` and the flattening
of `content` are the stdio ones, unchanged):

    → POST http://host:port/mcp
      Content-Type: application/json
      Accept: application/json, text/event-stream
      Mcp-Session-Id: <only when the server issued one>
      {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18",
       "capabilities":{},"clientInfo":{"name":"sah-mcp-client","version":"0.1"}}}
    ← 200  {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"...","capabilities":{...},
           "serverInfo":{...}}}                  (a JSON body, or the same message inside an SSE stream)
    → POST .../mcp  {"jsonrpc":"2.0","method":"notifications/initialized"}          (no id)
    ← 202  (empty body — the shape the measured endpoint answers, and the shape to expect)
    → POST .../mcp  {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    ← 200  {"jsonrpc":"2.0","id":2,"result":{"tools":[...]}}
    → POST .../mcp  {"jsonrpc":"2.0","id":N,"method":"tools/call",
                     "params":{"name":"...","arguments":{...}}}
    ← 200  {"jsonrpc":"2.0","id":N,"result":{"content":[...],"isError":false}}
       or  {"jsonrpc":"2.0","id":N,"error":{"code":...,"message":"..."}}

The body decides, not the header. A server answers a request with either a JSON body or an SSE stream,
and it announces which by a `Content-Type` header it is entitled not to send — the endpoint this client
was written against answers with no `Content-Type` at all, and with no `Mcp-Session-Id` either — so the
adjudication reads the bytes: after leading whitespace, a body beginning `{` or `[` is JSON and anything
else is an SSE stream. An SSE body is blocks separated by a blank line; within a block the `data:` lines
carry the message (joined with a newline, then parsed as JSON) while `event:`, `id:`, `retry:` and `:`
comment lines carry none. The stream is read until the message whose id matches the request; a
notification in it is skipped, and a stream that ends without the match is a wire failure, not a timeout
— the answer is not coming. One message per POST: this client never batches, so there is no array of
requests to correlate and no partial acceptance to reason about.

Session. If a response — the handshake's, in every server that has one — carries `Mcp-Session-Id` it is
stored and echoed in the header of every later request to that endpoint; if none does, none is sent. The
measured endpoint is stateless and issues no session, and a client that required one would refuse servers
that work. No session renewal: an endpoint that expires a session mid-run is met as the wire failure it
is, and renewal is a later class on the same seam.

Status. 200 is parsed as above. 202 is a notification's answer: an empty body is the normal shape, and a
body carrying a JSON-RPC error is handed up as a finding while the boot goes on — the endpoint has already
answered `initialize`, and `tools/list` is the gate that matters. 3xx is a named refusal and is never
followed: the operator named the destination, and a client that followed a redirect would carry the run
and every call it makes to a URL nobody typed. 4xx/5xx is a wire failure naming the status and the head
of the body, or that body's JSON-RPC error text when it is one. A server of the older two-channel
dialect answers this POST with 405 or 404 and meets exactly that sentence, rather than a guess at what it
would have preferred.

Time. The ceiling is the one `SAH_MCP_TIMEOUT` already means and it keeps its stdio meaning — a ceiling on
a hung server, not a budget for honest work — and the handshake's own ceiling is still
`HANDSHAKE_SECONDS`. On HTTP the ceiling is both the socket operation timeout and a wall deadline
re-checked between the lines of the body read, so a stream that dribbles past it is a timeout and not a
hang.

Connection. One fresh connection per POST, closed before the answer leaves the handler: no pooling, no
keep-alive, no persistent channel, because this extension carries no state between calls and a pool is
state. `close()` on an HTTP server is therefore the no-op that sets its closed flag — there is no child to
end and no pipe to EOF, which is worth saying only because the stdio half of the ending is so much louder
that a reader will assume this half does something too. `http.client`, not `urllib`: `urllib` follows
redirects, which is refused above, and gives no per-operation deadline on a stream. `http://` and
`https://` both, and an `https` certificate is verified the way `ssl` verifies by default — no knob turns
that off, because an endpoint's authenticity is not the model's to choose and not this file's to waive.

Ids are the client's, one counter per server, and an answer is matched to its request by id: a message
with no id (a server-initiated notification) or an id that was never asked for is not the answer to
anything, and is dropped. This client keeps one request outstanding per server, so there is nothing
else to match.

Configuration, from the environment (a .env file feeds these the way it feeds SAH_HOST; a value the
operator exported or a flag set wins over the file):

    SAH_MCP_STDIO     semicolon-separated servers, each a comma-separated argv, e.g.
                      python,servers/echo.py;python,servers/add.py
                      An argv word containing a comma or a semicolon cannot be named this way — the
                      value is a table, not a language, so no escape dialect is invented for it; point
                      the knob at a wrapper script that carries the awkward argument instead.
    SAH_MCP_HTTP      semicolon-separated endpoints, each a full URL, e.g.
                      http://127.0.0.1:8791/mcp;https://mcp.example.org/mcp
                      The same table-not-a-language rule: a URL containing a semicolon cannot be named
                      here, and none is invented for it. A name that is not an `http` or `https` URL is
                      a named refusal at boot, quoting what was named.
    SAH_MCP_HTTP_HEADERS
                      optional; extra request headers sent on every request to every named endpoint, as
                      semicolon-separated `Name: Value` pairs (split on the first colon, both halves
                      stripped) — the operator's door to a Bearer token or an API-key header, so the
                      destination and its keys stay the operator's act, fixed at boot, with only the
                      tool and its arguments left for the model to choose. A pair with no colon (or no
                      name) is a named refusal at boot, read before any server is started for it. The
                      protocol's own headers (Content-Type, Accept, Mcp-Session-Id) are written after
                      these, because those are what the wire requires and these are the extras.
    SAH_MCP_TIMEOUT   seconds a `tools/call` may take before the wrapper answers with the timeout
                      instead of going on waiting (default 120). It applies to both transports. It is a
                      ceiling on a hung server, not a budget for honest work, and it is not the
                      handshake's ceiling — see below.

No server and no endpoint named → the extension loads, contributes nothing, stays silent: an extension
with no tools is legal (sah/ext.py).

Boot failure: a server that will not launch, an endpoint that cannot be reached, one that answers with a
redirect or an error status or a JSON-RPC error to the handshake or the tool listing, or one that runs
out the 30-second handshake ceiling is a **named refusal at boot** (the same class of failure a
mis-shaped extension file already is: `sah.cli._load_extension` turns it into exit 2 with the reason, and
nothing runs). The message names the server by the command, or the URL, the operator named it with, and
the reason. Servers that did connect are closed before the refusal leaves, whichever transport they are
on, so a run that will not start starts with no processes left behind and no channel half open.

v2, and what is deliberately not in it. **Both transports, one seam** — a launched child and a named
endpoint sit behind `_Server`, and everything above the seam is shared. What is out, on purpose:
**the legacy SSE two-channel transport** — the 2024-11 spec's shape, where a server is met by a GET that
opens a stream and requests are posted to a second URL; a server of that flavour answers this POST with
405 or 404 and gets that named refusal rather than a client guessing at its dialect, and support for it is
a later class behind the same seam. **Server-initiated streams** — the client never opens a GET stream:
the one thing the transport would push at this harness is a tool-list change, and the tool table is built
once, at boot, because a tool that left mid-run would take a table entry the model was told it had, which
is the same reason the notification is dropped on stdio. **Connection pooling** (a pool is state, and
there is none to keep) **and session renewal** (an expired session is the wire failure it reports).
**Batched requests** — one message per POST. **Redirects** — refused, above. **Text results only** — a
`content` block that is not `text` is stated in the answer as dropped ("(one image block dropped — v1 is
text)") rather than passed through as nothing or failed over, so the run knows what it did not get;
returning a picture is v3, and `sah.contract.Media` is where it would land.

Trust. A stdio server is a subprocess this file starts from a command the **operator** named in the
environment, so it is in the same trust class as this extension file itself: the operator's act chooses
the program, and no value the model sends changes which program runs. An HTTP endpoint is the same class
of widening bought the same way: a URL and its headers, named by the operator in the environment and read
at boot, so the model directs a call and never a destination, and nothing it can say turns this client
into a proxy for a host nobody named. Both are real widenings — the server can do whatever its own code
does — and every call still passes the veto point: contributing a tool buys no permission, because the
gate is on the call and not on who registered it (sah/ext.py).

Lifecycle: what the handshake found — the connection (`<serverInfo name> connected, N tools advertised`),
a dialect the server answered with, a notification it answered with an error — is handed up as findings
the core records beside the load notice, rather than said to stderr, because a boot fact the record does
not hold is a boot fact the operator's audit cannot find. The one stderr this file keeps for itself is
the ending's: a stdio child that does not exit when its stdin closes is stopped, and the stop is stated
where a stop is stated. The watcher sees every step as it is written, and the step that matters is the
run's ending, which closes the servers — both kinds, in the one list. Closing a stdio server is: close
the child's stdin — the
transport's own ending, a well-behaved server exits on EOF — with `sah.process.kill_tree` as the backstop
a short grace later. Closing an endpoint is the flag and nothing else, because nothing was left open. The
assumption that makes the killed-not-ended case safe: a pipe handle does not outlive its holder, so if
this process dies the child's stdin still EOFs, and a server that obeys the transport leaves with it.
"""
from __future__ import annotations

import http.client
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Annotated

from sah.contract import Param
from sah.ext import Extension
from sah.process import kill_tree
from sah.record import Refused

PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "sah-mcp-client"
CLIENT_VERSION = "0.1"

# The handshake's ceiling, and it is short on purpose: a server that cannot say what it is within half
# a minute is not going to say it during a call, and a run that begins by hanging is a run that never
# began. A boot that waits forever is worse than a boot that refuses. It is the transport's own ceiling —
# the same half minute covers an endpoint that accepts the connection and then says nothing, because the
# message being waited for and the length it is owed are facts about the handshake, not about the pipe.
HANDSHAKE_SECONDS = 30.0

# A call's ceiling, deliberately wider than the handshake's: a tool the operator named may legitimately
# spend minutes building something. The ceiling exists to turn a hung server into an answer the run can
# read and act on, not to budget honest work, so it is generous and it is a knob.
CALL_SECONDS = 120.0

# How long a server gets to exit on its own after its stdin closes, before the backstop runs. Two
# seconds is long enough for a server that flushes on EOF and shorter than anyone will notice.
CLOSE_GRACE_SECONDS = 2.0

# How much of an error status's body is worth quoting. A proxy's error page is two kilobytes of markup,
# and the first line of it is the fact the reader wants; the rest is a wall behind the word "Bad".
BODY_HEAD_CHARS = 200

JSON_TYPE = {"string": str, "integer": int, "number": float, "boolean": bool,
             "object": dict, "array": list}
"""The advertised JSON type -> the Python type whose derived schema says the same thing.

These six and no others, because `sah.contract.spec_for` derives a tool's schema from a signature, and
a property of any other type (`null`, `anyOf`, an untyped blob) has no signature to derive from. The
mapping round-trips exactly: `_JSON_TYPE` in sah/contract.py turns each of these back into the word the
server used, so the schema the model is shown is the schema the server advertised.
"""


class McpError(Exception):
    """One server's wire failed: it would not start, it could not be reached, it closed, it spoke no
    JSON, it ran out its clock.

    Internal to this file. It becomes a named refusal at boot (from_env turns it into `Refused`) and a
    tool's own answer during a run (`call` turns it into text), because the two moments owe the operator
    different things: a boot that must not start, and a run that must be told."""


class _TimedOut(Exception):
    """The wait ran out. Carries nothing: only the waiter knows what it was waiting for and what
    ceiling it set, and those are the two parts worth reading."""


def _error_text(error) -> str:
    """A JSON-RPC `error` member as one readable clause: `json-rpc -32601: no such tool`."""
    if not isinstance(error, dict):
        return f"json-rpc error {error!r}"
    return (f"json-rpc {error.get('code', '?')}: {error.get('message') or 'the server gave no message'}"
            + (f" {error.get('data')!r}" if error.get("data") not in (None, "") else ""))


def _block_phrase(counts: dict) -> str:
    """`one image block, 2 audio blocks` for {image: 1, audio: 2} — a count of one spelled out.

    Spelling the one out is the wording this file promises ("one image block dropped"), and the sentence
    it lands in is read by a model as prose about what it did not receive, not as a count to parse."""
    parts = []
    for kind in sorted(counts):
        n = counts[kind]
        parts.append(f"{'one' if n == 1 else n} {kind} block" + ("" if n == 1 else "s"))
    return ", ".join(parts)


def _flatten(content) -> str:
    """A `tools/call` result's `content` blocks as the text a tool returns: every text block verbatim,
    joined; everything else counted and stated as dropped.

    A block of a kind v1 cannot carry is never passed through as nothing and never fatal: the answer says
    what was left out, so a run reading a short reply knows the server sent more than it heard rather
    than concluding the server was terse.
    """
    parts: list[str] = []
    dropped: dict[str, int] = {}
    for block in content if isinstance(content, list) else []:
        if (isinstance(block, dict) and block.get("type") == "text"
                and isinstance(block.get("text"), str)):
            parts.append(block["text"])
            continue
        kind = block.get("type") if isinstance(block, dict) else None
        kind = kind if isinstance(kind, str) and kind.strip() else "untyped"
        dropped[kind] = dropped.get(kind, 0) + 1
    text = "\n".join(parts)
    if dropped:
        note = f"({_block_phrase(dropped)} dropped — v1 is text)"
        text = f"{text}\n{note}" if text else note
    return text


class _Server:
    """The seam: what one server owes the rest of this file, whichever transport carries it.

    Four attributes, read by everything above: `command` — how the operator named this server, an argv
    or a URL, quoted by every failure about it, because a reader of the refusal has to be able to find
    the thing they typed; `name` — the server's own, once `serverInfo` gives one, and its naming until
    then; `advertised` — what `connect()` read, consumed once to build the tool table; `findings` —
    what the handshake found on the way, in the order found, open with the handshake and read once by
    the client, which hands it up for the record.

    Five methods, and `connect()` and `call()` are here because the handshake's order and the meaning of
    an answer are the protocol's rules, shared by every transport, and two copies of them would be two
    places to keep in step. `request()` and `notify()` are the wire, and `close()` is each transport's own
    ending — which for a stateless endpoint is nothing at all, a fact worth stating only because silence
    about it reads like an omission.

    One instance per server, because an id counter and whatever the transport holds between calls (a
    child and its reader thread, or nothing) belong to one server: a run pointed at two servers must not
    have one counter answering both, and it must not have one dead pipe closing the other's tools.
    """

    def connect(self) -> None:
        """Reach the server, introduce ourselves, announce ourselves as ready, and ask what it has.

        The order the transport fixes: `initialize`, then the `initialized` notification the server is
        waiting for before it will serve anything, then `tools/list`. A JSON-RPC `error` to either of the
        two requests is a refusal by the server and stops the boot: a server that will not say what it
        has cannot be offered to a model as anything.
        """
        # The handshake's own record: what was found on the way, in the order found. A server that
        # fails the handshake is closed and the boot refused, so a half list is never read.
        self.findings: list = []
        self._begin()
        answer = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
        }, HANDSHAKE_SECONDS, label="the handshake")
        if "error" in answer:
            raise McpError(f"{self.command}: the server refused the handshake "
                           f"({_error_text(answer['error'])})")
        intro = answer.get("result")
        intro = intro if isinstance(intro, dict) else {}
        info = intro.get("serverInfo")
        info = info if isinstance(info, dict) else {}
        spoken = info.get("name")
        if isinstance(spoken, str) and spoken.strip():
            self.name = spoken.strip()
        version = intro.get("protocolVersion")
        if isinstance(version, str) and version and version != PROTOCOL_VERSION:
            # Stated, not fatal: the transport lets a server answer with a version it supports rather
            # than the one asked for, and a boot that refused every server not on the current word
            # would refuse servers that work. What the operator needs is to know which dialect the run
            # is actually talking.
            self.findings.append(
                f"{self.name} answered protocol {version!r} instead of {PROTOCOL_VERSION!r}; "
                f"continuing on the version it answered with")
        self.notify("notifications/initialized")
        listed = self.request("tools/list", {}, HANDSHAKE_SECONDS, label="the tool listing")
        if "error" in listed:
            raise McpError(f"{self.command}: the server refused tools/list "
                           f"({_error_text(listed['error'])}); a server that will not say what it has "
                           "cannot be offered as tools")
        result = listed.get("result")
        tools = (result or {}).get("tools") if isinstance(result, dict) else None
        self.advertised = [tool for tool in tools if isinstance(tool, dict)] if tools else []
        self.findings.append(f"{self.name} connected, {len(self.advertised)} tools advertised")

    def call(self, tool: str, arguments: dict) -> str:
        """`tools/call`, and the answer flattened to the text a tool returns.

        It does not raise. A JSON-RPC `error`, an `isError` result, a dead pipe and a blown ceiling all
        come back as the answer naming the failure: a tool that raises takes the run down with it, and a
        tool that reports lets the run read what the server said and change course. `isError` is the
        server's own "this call did not work" and is worded as that, with whatever text came with it —
        it is a fact about the tool's work, where a JSON-RPC error is a fact about the call.
        """
        try:
            answer = self.request("tools/call", {"name": tool, "arguments": arguments},
                                  self.call_timeout, label=f"tools/call {tool!r}")
        except McpError as exc:
            return f"mcp: {exc}"
        if "error" in answer:
            return (f"mcp: {self.name} refused {tool} ({_error_text(answer['error'])}); the server "
                    "answered the call, and not with a result")
        result = answer.get("result")
        if not isinstance(result, dict):
            return f"mcp: {self.name} answered {tool} with no result object"
        text = _flatten(result.get("content"))
        if result.get("isError"):
            return f"mcp: {tool} reported an error: {text or 'the server gave no reason'}"
        return text or f"mcp: {self.name} answered {tool} with no content"


class StdioServer(_Server):
    """One launched server and its stdio wire: the child, the reader thread, the request loop.

    What the base half owes this class is the child's life: `_begin()` starts it and `close()` ends it.
    Everything protocol-shaped that happens in between happens on the base, on the same five methods an
    endpoint is driven through.

    The thread exists for exactly one reason — a blocking pipe read has no deadline, and `select()`
    will not wait on a pipe on Windows, where it works on sockets only — and it does exactly one thing:
    move bytes from the child's stdout into a queue, line by line, until EOF. It is the only part of
    this file shared between threads, the queue is the only thing passed across it, and nothing it does
    can change what the run did, which is the whole of what a thread in a synchronous harness may do.
    Draining in a thread rather than on demand also means a server that logs notifications to stdout
    never blocks on a full pipe buffer while the client is waiting on a different answer.
    """

    def __init__(self, argv: list[str], call_timeout: float = CALL_SECONDS):
        self.argv = list(argv)
        self.command = " ".join(self.argv)     # what the operator named; every failure quotes this
        self.name = self.command               # until the server gives its own name, it has no other
        self.call_timeout = call_timeout
        self.proc: subprocess.Popen | None = None
        self.advertised: list[dict] = []
        self._lines: queue.Queue = queue.Queue()
        self._next_id = 0
        self._closed = False

    # --- the wire ------------------------------------------------------------------------

    def _pump(self, pipe, out: queue.Queue) -> None:
        """Move lines from the server's stdout into the queue until the pipe closes (None at the end).

        Runs on a daemon thread, so a server that outlives its usefulness can never hold this process
        open: the ending is the queue's EOF, and the ending of the process is allowed to arrive first."""
        while True:
            try:
                line = pipe.readline()
            except (OSError, ValueError):
                line = b""              # the handle went away under us: that is the closed pipe talking
            if not line:
                out.put(None)
                return
            out.put(line)

    def _send(self, message: dict) -> None:
        """One message down the child's stdin: JSON, one line, UTF-8, flushed now.

        `ensure_ascii` stays on (json's default) so the bytes on the wire are ASCII whatever the console
        thinks it is; a non-ASCII answer comes back as an escape and decodes to the same text."""
        if self.proc is None or self.proc.stdin is None:
            raise McpError(f"{self.command}: the server is not connected")
        blob = (json.dumps(message) + "\n").encode("utf-8")
        try:
            self.proc.stdin.write(blob)
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(f"{self.command}: {self._died() or 'its stdin refused the request'} "
                           f"({exc.__class__.__name__})") from exc

    def _died(self) -> str:
        """Why the pipe is quiet, when the process can answer that: an exit code beats a shrug."""
        code = self.proc.poll() if self.proc is not None else None
        if code is None:
            return ""
        return f"the server had already ended (exit code {code})"

    def _read(self, deadline: float) -> dict:
        """The next message off the wire, or `McpError` when the pipe closes and `_TimedOut` when the
        deadline passes (the waiter names the stage and its own ceiling — a pipe read cannot know them).

        The transport promises one message per line and nothing else on the channel, so a line that is
        not one JSON object is a breach of the transport, named as one and not skipped: a client that
        quietly ignored garbage would also quietly ignore the half of a response it needed to see."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _TimedOut
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty:
            raise _TimedOut from None
        if line is None:
            raise McpError(f"{self.command}: {self._died() or 'the server closed its stdout'}")
        text = line.decode("utf-8", "replace").strip()
        try:
            message = json.loads(text)
        except ValueError:
            raise McpError(f"{self.command}: a line that is not JSON arrived on stdout "
                           f"({text[:120]!r})") from None
        if not isinstance(message, dict):
            raise McpError(f"{self.command}: a JSON value that is not a message arrived on stdout "
                           f"({text[:120]!r})")
        return message

    def request(self, method: str, params: dict, ceiling: float, label: str = "") -> dict:
        """One JSON-RPC request and the answer that carries its id.

        Anything else that arrives while waiting is not an answer to this request: a notification has no
        id, and an id never asked for was not asked for. Both are dropped, because this client keeps one
        request outstanding per server and there is no other reading of a message that is not the reply.
        The answer is returned whether it is a `result` or an `error` — an `error` is the server's
        considered answer, and what to make of it belongs to the caller, not to the wire.
        """
        self._next_id += 1
        wanted = self._next_id
        self._send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params})
        deadline = time.monotonic() + ceiling
        while True:
            try:
                message = self._read(deadline)
            except _TimedOut:
                raise McpError(f"{self.command}: {label or method} got no answer within "
                               f"{ceiling:g} s") from None
            if message.get("id") == wanted:
                return message

    def notify(self, method: str) -> None:
        """One notification out: no id, and nothing is owed back for it."""
        self._send({"jsonrpc": "2.0", "method": method})

    # --- boot -------------------------------------------------------------------------------

    def _begin(self) -> None:
        """Start the server the operator named, with the two halves of the wire attached.

        No detach flags, on purpose: the server's life is the run's life. A child started for a session
        must not outlive it — which is the opposite of what `sah/process.py`'s `popen_detach_kwargs` is
        for (that is for a run sold on outliving its parent), and an MCP server detached from the run
        whose tools it is would be a process with no reason and no owner.

        The server's stderr is left inherited: the transport says nothing but messages may travel on
        stdout, so a server's own diagnostics belong to the operator's terminal, where they are already,
        rather than to the wire or to a buffer nobody reads before a pipe fills.
        """
        try:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except (OSError, ValueError) as exc:
            raise McpError(f"{self.command}: the server could not be started "
                           f"({exc.__class__.__name__}: {exc})") from exc
        threading.Thread(target=self._pump, args=(self.proc.stdout, self._lines),
                         daemon=True, name=f"mcp-{self.command[:40]}").start()

    # --- the ending -------------------------------------------------------------------------

    def close(self, grace: float = CLOSE_GRACE_SECONDS) -> None:
        """Close the server: its stdin first, then the backstop.

        Closing stdin is the whole of the polite ending — the transport's own EOF, which a well-behaved
        server reads as the session being over and exits on. `sah/process.py`'s `kill_tree` is the
        backstop for the one that does not: it never raises, and on Windows it walks the server's own
        children (which a bare `terminate` does not), so a server that spawned something does not leave
        it behind either.

        Only the backstop is loud. A run that ends quietly should not spend a line saying so; a server
        that ignored the ending is the fact worth printing.
        """
        if self._closed:
            return
        self._closed = True
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass                        # a dead child's stdin is already closed; that was the aim
        stopped_early = True
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            stopped_early = False
        if not stopped_early:
            kill_tree(self.proc, detached=False)
            print(f"mcp: {self.name} did not exit when its stdin closed; it was stopped",
                  file=sys.stderr, flush=True)


# --- the HTTP transport's own parts --------------------------------------------------------------

def _header(headers, name: str) -> str:
    """A response header, looked up without regard to case: `Mcp-Session-Id` and `mcp-session-id` are the
    same header, and which one arrives is the server's choice about casing, not a fact about the session.

    `headers` is the iterable of `(name, value)` pairs `http.client` answers with. An absent header is
    `""`, which is what every caller here wants: no session, or no `Location`, is the absence, not an
    error of its own."""
    wanted = name.lower()
    for key, value in headers or ():
        if key.lower() == wanted:
            return value
    return ""


def _body_messages(body: bytes, command: str) -> list[dict]:
    """What an answer's bytes carry, as messages — adjudicated by the body and not by a header.

    The transport lets a server answer with a JSON body or with an SSE stream, and the header that would
    say which is one a real server is known to omit, so the first non-whitespace character decides: `{`
    or `[` is JSON, anything else is a stream. An SSE block's `data:` lines are joined with a newline and
    parsed as one message; its `event:`, `id:`, `retry:` and `:` comment lines are skipped, and a block
    with no `data:` line carries nothing.

    A body that claims to be JSON and is not is a breach of the transport and says so, as it does on the
    stdio half: a client that quietly ignored it would also quietly ignore the half of an answer it needed
    to see. A JSON array is read as the messages it holds, which is what a batching server sends; this
    client never sends a batch, so an array arriving is the server's answer, not a request for support.
    """
    text = body.decode("utf-8", "replace")
    if text.lstrip()[:1] in ("{", "["):
        try:
            value = json.loads(text)
        except ValueError:
            raise McpError(f"{command}: an answer that is not JSON arrived ({text[:120]!r})") from None
        found = value if isinstance(value, list) else [value]
        for candidate in found:
            if not isinstance(candidate, dict):
                raise McpError(f"{command}: a JSON value that is not a message arrived "
                               f"({text[:120]!r})") from None
        return found
    messages: list[dict] = []
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        data = [line[5:].removeprefix(" ") for line in block.split("\n") if line.startswith("data:")]
        payload = "\n".join(data)
        if not payload.strip():
            continue            # a frame with no data line is a heartbeat, an `event:` alone, a comment
        try:
            message = json.loads(payload)
        except ValueError:
            raise McpError(f"{command}: an SSE data block that is not JSON arrived "
                           f"({payload[:120]!r})") from None
        if not isinstance(message, dict):
            raise McpError(f"{command}: a JSON value that is not a message arrived in the stream "
                           f"({payload[:120]!r})") from None
        messages.append(message)
    return messages


def _status_text(body: bytes) -> str:
    """What an error status's body is worth in a refusal: its JSON-RPC error text when it is one, and
    otherwise the head of it.

    The head and not the body, because an intermediary's error page is kilobytes of markup whose first
    line is the fact the reader wants, and a refusal that quoted the rest would bury the status code the
    operator has to act on."""
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return "the endpoint gave no body"
    try:
        value = json.loads(text)
    except ValueError:
        return text[:BODY_HEAD_CHARS]
    if isinstance(value, dict) and "error" in value:
        return _error_text(value["error"])
    return text[:BODY_HEAD_CHARS]


def _split_url(url: str) -> tuple:
    """The endpoint as `(https, host, port, path)` — or the named reason the naming is not one.

    Split here rather than with a URL library, because the library this file might have reached for is
    the one that follows redirects (refused above), and four parts are all any decision in this file
    needs. `http` and `https` are the only schemes a Streamable HTTP endpoint is named with; anything else
    — a `ws:` URL, an `ftp:` one, a bare `host:port` with the scheme left out — is a named refusal that
    quotes what was named, because a client that quietly corrected the operator's naming would be choosing
    a destination rather than reaching the one that was stated.
    """
    scheme, separator, rest = url.partition("://")
    if not separator or scheme.lower() not in ("http", "https"):
        raise McpError(f"{url}: an endpoint must be named as http://... or https://... "
                       "(this client speaks one URL per server and no other scheme)")
    https = scheme.lower() == "https"
    host, _, target = rest.partition("/")
    if host.startswith("["):                                  # an IPv6 literal keeps its brackets
        hostname, _, tail = host.rpartition("]")
        port_text = tail[1:] if tail.startswith(":") else tail
    else:
        hostname, _, port_text = host.partition(":")
    if not hostname:
        raise McpError(f"{url}: the endpoint names no host")
    if port_text:
        if not port_text.isdigit():
            raise McpError(f"{url}: {port_text!r} is not a port number")
        port = int(port_text)
    else:
        port = 443 if https else 80
    return https, hostname, port, ("/" + target) if target else "/"


def _http_post(url: str, headers: dict, body: bytes, timeout: float) -> tuple:
    """The one outbound function: one POST, on a connection that ends before this one returns.

    Every byte this file sends over HTTP goes through here, which is what makes the seam testable and
    what keeps it honest: a check of the protocol's logic replaces this function and nothing else, and the
    socket itself — the real connect, the real bytes — is proven only against a real endpoint. It opens
    the connection for this one request, sends the body, and answers `(status, response-header pairs,
    body bytes)`, closing the connection before it answers: nothing here outlives the call, because
    nothing here is allowed to.

    The whole body is read, line by line, with the wall deadline re-checked between the lines — a
    `read()` of a stream has one timeout on the socket and no ceiling over the whole read, so a server
    that sends a byte a minute would satisfy the first and never reach the second.

    The failures travel up as they happened (`OSError`, `http.client.HTTPException`, `_TimedOut`), the way
    the stdio half lets a broken pipe travel to the code that reads it: the words that make a failure
    legible belong to whoever knows what was being asked for, and that is the caller.
    """
    https, host, port, path = _split_url(url)
    connection = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(
        host, port, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        status = response.status
        response_headers = response.getheaders()
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() > deadline:
                raise _TimedOut
            line = response.readline()
            if not line:
                break
            chunks.append(line)
        return status, response_headers, b"".join(chunks)
    finally:
        connection.close()


class HttpServer(_Server):
    """One endpoint the operator named and the Streamable HTTP wire to it: the handshake, one POST per
    message, and an ending that is nothing.

    There is no channel between calls and nothing is pooled: each message opens its connection and closes
    it with its answer, which is why this class holds no handle, no thread and no buffer, and why `close()`
    below is the flag and no act. A server that keeps a session is carried by the one header it issued
    (`session`), echoed on every later request; the endpoint this client was written against issues none,
    so the field stays `None` and no header is sent.

    The header the run sends is the operator's `SAH_MCP_HTTP_HEADERS` plus the two the wire requires, and
    it is the same on every request: the destination and its credentials are fixed at boot, and no
    argument a model writes changes either.
    """

    def __init__(self, url: str, extra_headers: dict | None = None,
                 call_timeout: float = CALL_SECONDS):
        _split_url(url)                                # the operator's naming is checked before it is used
        self.url = url
        self.command = url                             # every failure quotes this, as the argv does on stdio
        self.name = url                                # until the server gives its own name, it has no other
        self.headers = dict(extra_headers or {})
        self.call_timeout = call_timeout
        self.session: str | None = None
        self.advertised: list[dict] = []
        self._next_id = 0
        self._closed = False

    # --- the wire ------------------------------------------------------------------------

    def _begin(self) -> None:
        """Nothing to open: every message opens its own connection and closes it with its answer.

        The hook exists because the handshake's first step is where a stdio server starts its child, and
        an endpoint that is named rather than launched has no equivalent act to take here."""

    def _post(self, message: dict, ceiling: float, label: str) -> tuple:
        """One message to the endpoint, and the triple `_http_post` answered with.

        The one place this transport's raw failures become `McpError`, which is what keeps the
        boot-vs-call split where the stdio half put it: `_TimedOut` becomes the ceiling it ran out of,
        naming the stage (the waiter knows that and the socket does not), and anything the connection or
        the HTTP layer raises is named by class and message. A session id is stored from the first answer
        that carries one — the handshake's, on every server that has one — and echoed from then on.
        """
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        if self.session is not None:
            headers["Mcp-Session-Id"] = self.session
        method = str(message.get("method") or "")
        try:
            status, response_headers, body = _http_post(
                self.url, headers, json.dumps(message).encode("utf-8"), ceiling)
        except _TimedOut:
            raise McpError(f"{self.command}: {label or method} got no answer within "
                           f"{ceiling:g} s") from None
        except (OSError, http.client.HTTPException) as exc:
            raise McpError(f"{self.command}: {label or method} never reached the endpoint, or its "
                           f"answer never came back ({exc.__class__.__name__}: {exc})") from exc
        if self.session is None:
            self.session = _header(response_headers, "Mcp-Session-Id") or None
        return status, response_headers, body

    def _check_status(self, status: int, headers, body: bytes, label: str) -> None:
        """The status codes that are not an answer: a redirect refused, an error status named.

        A redirect is never followed. The operator named this endpoint; a `Location` is a destination the
        run would choose for itself, and carrying a run — and the calls it makes, and the headers it
        sends — to a URL nobody typed is not a transport detail, it is the run leaving its grants. So the
        redirect is reported, with the status that carried it, and this response ends here.

        A 404 or 405 gets one extra clause, because it is the most likely thing this message will ever
        have to say: it is what a server of the older two-channel dialect answers a Streamable HTTP POST
        with, and an operator meeting it needs to know the client spoke one dialect, not that the server
        is broken.
        """
        if 300 <= status < 400:
            location = _header(headers, "Location")
            raise McpError(f"{self.command}: {label} was answered with a redirect (status {status})"
                           + (f" to {location!r}" if location else "")
                           + "; a redirect is never followed — the endpoint the operator named is the "
                             "destination, and this client will not carry a run to a URL nobody typed")
        if status >= 400:
            raise McpError(f"{self.command}: {label} was answered with status {status}: "
                           f"{_status_text(body)}"
                           + (" — this client speaks the Streamable HTTP transport and no older dialect"
                              if status in (404, 405) else ""))

    def request(self, method: str, params: dict, ceiling: float, label: str = "") -> dict:
        """One JSON-RPC request as one POST, and the answer carrying its id out of the response.

        The body is a JSON message or an SSE stream and both are read the same way (`_body_messages`);
        what arrives without the id this request used is dropped for the stdio reason — a notification has
        no id and an id never asked for was not asked for. Nothing left that matches is a wire failure,
        and the two shapes that takes are stated apart, because an empty body to a request (a 202, say)
        is a different server's mistake than a stream that ended short of the answer.
        """
        self._next_id += 1
        wanted = self._next_id
        status, headers, body = self._post(
            {"jsonrpc": "2.0", "id": wanted, "method": method, "params": params}, ceiling, label)
        self._check_status(status, headers, body, label or method)
        messages = _body_messages(body, self.command)
        for message in messages:
            if message.get("id") == wanted:
                return message
        if not body.strip():
            raise McpError(f"{self.command}: {label or method} was answered with status {status} and "
                           "no body at all, and a request is owed the message carrying its id")
        raise McpError(f"{self.command}: {label or method} was answered with status {status} and "
                       "nothing carrying the id it asked for")

    def notify(self, method: str) -> None:
        """One notification out: no id, and nothing is owed back for it.

        The only notification this client sends is the boot's, so the ceiling on its POST is the
        handshake's. A 2xx is the answer a notification deserves, and the body is read only to look for a
        JSON-RPC error: one carried is handed up as a finding and the boot goes on, because the
        endpoint already answered `initialize` and it is `tools/list` that decides whether this server
        has anything to offer. A 3xx or an error status is the failure `_check_status` names, whichever
        asked.
        """
        status, headers, body = self._post({"jsonrpc": "2.0", "method": method}, HANDSHAKE_SECONDS,
                                           label=f"the {method} notification")
        self._check_status(status, headers, body, f"the {method} notification")
        for message in _body_messages(body, self.command):
            if "error" in message:
                self.findings.append(
                    f"{self.name} answered the {method} notification with "
                    f"{_error_text(message['error'])}; continuing, because it answered the handshake "
                    f"and the tool listing is the gate that matters")
                return

    # --- the ending -------------------------------------------------------------------------

    def close(self) -> None:
        """End this server, which is to say: mark it closed.

        There is no child to end and no pipe to EOF — every connection this class opened was closed before
        its answer was handed back — so the only thing left to do is the one thing `close()` always means
        here: stop treating the server as live, so a second close is as quiet as the first. Stated because
        the stdio half of the ending is loud (stdin, a grace, a process tree stopped) and a reader who has
        just read that will otherwise look for the half that is not here.
        """
        self._closed = True


class _Skip(Exception):
    """One advertised tool cannot be offered to a model. Carries the reason, in the shape the operator
    needs: server, tool, property, type. The rest of the server's tools still load."""


def _wrapper(server: _Server, advertised: dict):
    """Build the tool for one advertised tool: the name exactly as advertised, the description from the
    server, and a signature mapped from the advertised `inputSchema` — because in this harness the
    signature *is* the schema (sah/contract.py), so a wrapper is generated by describing its parameters
    and not by writing a schema beside them.

    Raises `_Skip` for the properties and tools no signature can carry. The skip is per tool and the
    reason rides the record as a finding, so one unmappable tool costs one tool and not the server's
    whole table.
    """
    name = advertised.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _Skip(f"{server.command}: a tool advertised with no name cannot be offered to a model")
    description = advertised.get("description")
    if not isinstance(description, str) or not description.strip():
        raise _Skip(f"{server.command}, tool {name!r}: advertised with no description, and "
                    "sah.contract.spec_for refuses a tool the model would have to guess at")
    schema = advertised.get("inputSchema")
    if schema is None:
        schema = {}                       # no schema at all is a tool that takes nothing, which maps
    if not isinstance(schema, dict):
        raise _Skip(f"{server.command}, tool {name!r}: its inputSchema is not an object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise _Skip(f"{server.command}, tool {name!r}: its inputSchema.properties is not an object")
    required = schema.get("required") if isinstance(schema.get("required"), list) else []

    annotations: dict = {}
    parameters: list[inspect.Parameter] = []
    for prop, rule in properties.items():
        rule = rule if isinstance(rule, dict) else {}
        json_name = rule.get("type")
        if json_name not in JSON_TYPE:
            raise _Skip(f"{server.command}, tool {name!r}: property {prop!r} has type "
                        f"{json_name!r}; the types a signature can carry are "
                        + ", ".join(sorted(JSON_TYPE)))
        python_type = JSON_TYPE[json_name]
        field_description = rule.get("description")
        enum = rule.get("enum")
        meta = None
        if ((isinstance(field_description, str) and field_description.strip())
                or (isinstance(enum, list) and enum)):
            # `Param` is where this harness keeps a field's meaning, and a closed set the server
            # advertised is enforced at the boundary rather than left for the server to misread.
            meta = Param(description=(field_description.strip()
                                      if isinstance(field_description, str) else ""),
                         enum=tuple(enum) if isinstance(enum, list) and enum else ())
        hint = Annotated[python_type, meta] if meta is not None else python_type
        annotations[prop] = hint
        try:
            parameters.append(inspect.Parameter(
                prop, inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if prop in required else None,
                annotation=hint))
        except (TypeError, ValueError):
            # A name no signature can hold (empty, a space, a Python keyword) is not a tool this
            # harness can advertise: the signature is the schema, and a schema the harness half wrote
            # is a promise the operator cannot see.
            raise _Skip(f"{server.command}, tool {name!r}: property {prop!r} is not a name a "
                        "signature can carry") from None

    def wrapper(**given):
        """One `tools/call`, and the server's answer as text.

        `given` is a name no schema will use, because everything the model sent arrives by keyword and
        this picks it up by name. A value that is `None` is an optional property the model did not fill,
        and it is left out of `arguments` rather than sent as a null the server was not promised."""
        arguments = {key: value for key, value in given.items() if value is not None}
        return server.call(name, arguments)

    wrapper.__name__ = name               # exactly as advertised: a collision is the boot's to refuse
    wrapper.__qualname__ = name
    wrapper.__doc__ = (f"Generated MCP wrapper for the {server.name!r} server's {name!r}. "
                       "Documentation for whoever reads it: the model is shown the description.")
    wrapper.__annotations__ = annotations
    wrapper.__signature__ = inspect.Signature(parameters=parameters, return_annotation=str)
    wrapper.description = description.strip()
    # `self`, the class the self-confined tools sit in: the destination is the operator's act — the
    # server was named in the environment and reached at boot, and no value the model sends chooses
    # or changes it — so the model directs the arguments and never the place, nothing new has grown
    # that the policy has not already seen, and there is no argument naming a place to declare.
    wrapper.access, wrapper.address = "self", None
    return wrapper


class McpClient(Extension):
    """The extension: the servers the operator named, on either transport, their tools, and the ending.

    One extension over every server, because `sah.ext.assemble` builds the run's one tool table and a
    name a second server claims must collide *there* — the existing named refusal is the mechanism, and
    nothing is invented here to reconcile two servers that mean different things by the same word.

    The `servers` list holds both transports side by side, and nothing in this class asks which one an
    entry is: the tools come from `advertised`, the calls go through `call`, and the ending goes through
    `close` — including the endpoint whose close is nothing, which is the right thing for it to do there
    as well as anywhere else.

    The tools are generated, and they are still plain functions with a signature, a description and a
    declaration: nothing about arriving over a wire exempts a tool from the gate every other tool passes.

    What the handshake found — the connection, a dialect the server answered with, a notification it
    answered with an error — and every tool it refused with the reason ride `findings` up to the core,
    which records them beside the load notice: the extension's own words, the core's stamp.
    """

    def __init__(self, servers=(), call_timeout: float = CALL_SECONDS):
        self.servers = list(servers)
        self.call_timeout = call_timeout
        tools = []
        findings: list = []
        for server in self.servers:
            findings.extend(server.findings)
            for advertised in server.advertised:
                try:
                    tools.append(_wrapper(server, advertised))
                except _Skip as exc:
                    findings.append(str(exc))
        self.tools = tuple(tools)
        self.findings = tuple(findings)
        self._closed = False

    def on(self, step: dict) -> None:
        """See each step as it is written. Notification only — the one step this acts on is the run's
        ending, and what it does with it is end the servers it owns, not anything in the run.

        Every other step is passed in silence: the connection was announced when it was made, and a
        watcher that narrated each call would be noise the log already carries.
        """
        if step.get("kind") == "ended":
            self.close()

    def close(self) -> None:
        """End every server, of either kind. Idempotent, because the ending may be seen and also called
        by hand."""
        if self._closed:
            return
        self._closed = True
        for server in self.servers:
            server.close()


def _servers(spec: str, call_timeout: float) -> list[StdioServer]:
    """Every launched server the stdio knob names, each connected before the next is tried.

    A failure closes what already connected and lets the failure through: a run that will not start must
    not start with servers running underneath it, and there is no partial run to keep them for.
    """
    servers: list[StdioServer] = []
    try:
        for entry in spec.split(";"):
            argv = [word.strip() for word in entry.split(",") if word.strip()]
            if not argv:
                continue                # an empty slot between separators names no server
            server = StdioServer(argv, call_timeout=call_timeout)
            server.connect()
            servers.append(server)
    except McpError:
        for server in servers:
            server.close()
        raise
    return servers


def _http_servers(spec: str, headers: dict, call_timeout: float) -> list[HttpServer]:
    """Every endpoint the HTTP knob names, each handshaken before the next is tried — the stdio half's
    rule repeated, for the same reason: a boot that refuses has to refuse with nothing left behind.

    The endpoints share the one header table, because `SAH_MCP_HTTP_HEADERS` names headers and not
    destinations: an operator who names two endpoints and one token means that token on both, which is
    what they mean by naming it once. An empty slot between separators names no endpoint, and a name that
    is not an http or https URL is the refusal `_split_url` states, arriving here as a boot refusal that
    closes whatever connected before it.
    """
    servers: list[HttpServer] = []
    try:
        for entry in spec.split(";"):
            url = entry.strip()
            if not url:
                continue
            server = HttpServer(url, headers, call_timeout=call_timeout)
            server.connect()
            servers.append(server)
    except McpError:
        for server in servers:
            server.close()
        raise
    return servers


def _headers_from_env() -> dict:
    """`SAH_MCP_HTTP_HEADERS` as a header table: the extras the operator adds to every request.

    The door a Bearer token or an API-key header comes through, and the reason it is a knob read at boot
    and not an argument: the destination and its credentials are the operator's act, so a model can call
    a tool that needs them and can neither read them nor point them elsewhere. Split on the first colon,
    both halves stripped, so `Authorization: Bearer x` arrives as the header it names and a value
    containing colons — a JWT, a URL — survives intact.

    A pair with no colon, or with no name before it, is a named refusal: a half-written credential is not
    something to guess at, and a boot that silently dropped it would send requests missing an
    authorisation the operator believed they had given.
    """
    headers: dict[str, str] = {}
    for pair in os.environ.get("SAH_MCP_HTTP_HEADERS", "").split(";"):
        if not pair.strip():
            continue                                        # an empty slot names no header
        name, colon, value = pair.partition(":")
        if not colon or not name.strip():
            raise Refused("mcp: SAH_MCP_HTTP_HEADERS is a semicolon-separated list of 'Name: Value' "
                          f"headers; {pair.strip()!r} is not one")
        headers[name.strip()] = value.strip()
    return headers


def from_env() -> McpClient:
    """The extension built from the environment, the way a run's operator configures it.

    Read at construction (import time, per run), so a value this run's process holds wins over whatever
    the machine's .env carries beneath it. No server and no endpoint named is not a mistake: the
    extension loads, contributes nothing and says nothing — and headers named with no endpoint to put
    them on contribute nothing either, because a header is not a destination and naming one is not an act
    that starts anything.

    The knobs are read before a single server is reached: a timeout that is not a number and a header that
    is not a pair are refused with nothing running, rather than after a child was started for a boot that
    was going to fail anyway.
    """
    raw_timeout = os.environ.get("SAH_MCP_TIMEOUT", "").strip()
    try:
        call_timeout = float(raw_timeout) if raw_timeout else CALL_SECONDS
    except ValueError:
        raise Refused(f"mcp: SAH_MCP_TIMEOUT must be a number of seconds; {raw_timeout!r} is not one") \
            from None
    headers = _headers_from_env()
    servers: list = []
    try:
        servers = _servers(os.environ.get("SAH_MCP_STDIO", ""), call_timeout)
        servers += _http_servers(os.environ.get("SAH_MCP_HTTP", ""), headers, call_timeout)
    except McpError as exc:
        # The one place a wire failure becomes the harness's own boot refusal: exit 2 with this sentence,
        # no run, and (via the two helpers and this loop) nothing left running underneath the run that
        # would not start — of either transport, whichever one failed in the other's middle.
        for server in servers:
            server.close()
        raise Refused(f"mcp: {exc}") from None
    return McpClient(servers, call_timeout=call_timeout)


# The one name the door reads. An instance, built now, at boot: the servers are launched and handshaken
# before the run exists, so a server that will not answer is a named refusal with nothing running
# (sah.cli._load_extension), and the tools it advertised are in the table the model is offered.
EXTENSION = from_env()
