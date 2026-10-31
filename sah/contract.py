"""sah.contract — a tool is a plain type-hinted function; its schema is derived, never declared.

There is no parameter table: the signature is the truth, so a schema cannot drift from the
function it describes, and a tool author writes Python rather than a dialect. The hint carries the
field's *meaning* too, via `Annotated[str, Param("...")]`, because a parameter the model sees only
as `{"type": "string"}` is a parameter it has to guess at. The type hint stays the single source of
the schema; the `Param` rides inside it rather than in a parallel table that could fall out of step.

The schema carries the contract (types, required, field meaning, closed value sets); the system
prompt carries judgment (when to reach for a tool, how to use it well). This module does the schema.

A tool receives plain values and returns text — or one `Media`, which is that text plus a picture the
model is to be shown. It gets no log handle, no session, no policy — anything a tool could reach, it
could bypass the veto point with. A tool that needs context is built by a closure that already has it.

The signature is the schema; three attributes beside the function are its *declaration* (`access`,
`address`, `description`: see `ACCESS`, `declaration` and `spec_for`), saying how it touches the
world, which argument names the place, and what the model is told it does. The schema is what the
model sees; the declaration is what the policy judges; and a tool that never declared is refused at
boot rather than running unconfined or being half-described.

The docstring is not part of the declaration. It is documentation for whoever reads the source —
the why, the cross-references, the history of a decision — and no byte of it is sent to a model.
That split is deliberate: a docstring serving two audiences at once (the model via the schema and
the maintainer in the source) degrades for both, because every improvement to one is a tax on the
other. The model gets the contract from `description`; the maintainer gets the reasoning from the
docstring; neither pays for the other's prose.
"""
from __future__ import annotations

import base64
import inspect
import typing
from dataclasses import dataclass
from typing import Annotated as Annotated  # re-exported so tools import Param + Annotated from here

from sah.model import ToolSpec
from sah.record import Refused


PICTURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
"""The file kinds a tool may hand to the model as a picture, told by their first bytes.

An extension is a claim and the bytes are the fact: a file named `photo.png` that holds prose must not be
attached as a picture, and a JPEG saved as `capture.dat` must not arrive as escapes. BMP and WebP are
matched in `picture_kind`, where their weak leading bytes get a second look."""


def picture_kind(path: str):
    """The mime kind if this file *is* one of the picture kinds, else None.

    A failed open returns None rather than a refusal: this is a look at a file, and whoever reads it is the
    one that owes whatever went wrong. The same question is asked at a resume, where the file is re-opened
    to rebuild a picture the log only named — and a file that is gone, or is no longer a picture, answers
    None and leaves the answer as text.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return None
    for magic, mime in PICTURES:
        if head.startswith(magic):
            return mime
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    # `BM` is two characters that prose could begin with, so the weak magic is checked against the header
    # beside it: a real BMP carries four zero reserved bytes here, which text does not.
    if head.startswith(b"BM") and head[6:10] == b"\x00\x00\x00\x00":
        return "image/bmp"
    return None


@dataclass(frozen=True)
class Media:
    """A tool's text answer plus one picture the model should be shown.

    The two halves are for two different readers. `text` is what the record keeps and what the model reads
    as the answer (`read: x.png (image/png, 170 KB)`); `path` is where the bytes are, so the log holds a
    place it can re-open rather than a megabyte of base64 that every later reader pays for — and a resumed
    run rebuilds the picture the way it already re-reads a text file it was shown once.

    Whether a picture may travel is not the tool's decision and is not made here. A tool reads a file and
    says what it found; the loop holds the endpoint's declaration and withholds the bytes when the server
    cannot see, saying so in the answer rather than returning silence that reads as an empty file.
    """

    text: str
    path: str
    mime: str

    def part(self) -> dict:
        """The content part, with the bytes read now rather than when the file was first looked at.

        A data URL and not a `file:` URL because a server is another program, and often another machine:
        measured against one endpoint that answers a `file:` url with "image URL must use HTTPS or a base64
        data URL". Encoding at build time is also what keeps the record holding an address instead of the
        bytes themselves.
        """
        with open(self.path, "rb") as handle:
            blob = base64.b64encode(handle.read()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{self.mime};base64,{blob}"}}


@dataclass(frozen=True)
class Verdict:
    """What the veto point decided.

    `reason` is read twice: it is the log's explanation and the text the model is told, because a
    model shown only "refused" retries the identical call. `address` is the resolved place the
    call concerns, supplied by whoever checked it — a verdict that has looked at the filesystem
    knows more precise coordinates than the argument the model typed.

    `ask` turns a refusal into a question: text worth putting to the operator, naming the smallest
    act that would let the call proceed. Empty means there is nothing to ask — some refusals are
    the policy standing firm, and a run that asked about every one of them would be nagging."""

    allowed: bool
    reason: str = ""
    address: str = ""
    ask: str = ""


ACCESS = ("read", "write", "exec", "self", "none")
"""How a tool touches the world, declared beside the tool instead of tabled by its name.

    read / write    a place the policy judges against the read or write grants.
    exec            the exec axis. Its address is a working directory, which is not a limit.
    self            the tool confines itself — a memory tree, a runs root, a child id, a web address
                    guard. The policy allows it and records the place it names.
    none            the tool touches no place at all.
"""


def declaration(func) -> tuple:
    """The `(access, address)` pair a tool declares, validated. `ext.assemble` calls this at boot, so a
    contributor that ships an undeclared tool fails when the run starts rather than running unconfined.

    A table keyed by tool name would leave any tool someone forgot to add — `cat`, `edit`, an
    extension's — confined by nothing and silently allowed: "allowed unless someone remembered" is a hole
    waiting for the next tool. The shape of the fix is to make the silence unbootable — a tool that
    declares no access is refused at boot — rather than to ask the next author to remember a line.
    """
    access = getattr(func, "access", None)
    address = getattr(func, "address", None)
    if access is None:
        raise Refused(f"{func.__name__} declares no access: a tool says how it touches the world. Set "
                      "`<tool>.access` to one of " + ", ".join(repr(a) for a in ACCESS) +
                      " and `<tool>.address` to the argument that names the place, or None")
    if access not in ACCESS:
        raise Refused(f"{func.__name__} declares access={access!r}; the closed set is "
                      + ", ".join(repr(a) for a in ACCESS))
    params = inspect.signature(func).parameters
    if address is not None and address not in params:
        raise Refused(f"{func.__name__} declares address={address!r}, which is not one of its parameters "
                      f"({', '.join(params)}); a policy that looks for an argument that never arrives "
                      "finds nothing to check")
    if access in ("read", "write") and address is None:
        raise Refused(f"{func.__name__} is judged against the grants (access={access!r}) but names no "
                      "address; say which argument holds the place")
    return access, address

_JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}


_UNANNOTATED = object()   # sentinel: a parameter with no hint, refused with a named error


@dataclass(frozen=True)
class Param:
    """A parameter's meaning, declared inside its type hint.

    `description` is shown to the model in the schema — the whole reason it exists: a bare
    `{"type": "string"}` forces guessing, and a frontier model that guesses about an enum or a
    mode looks identical to a weak model that misused the tool. `enum`, when set, is both advertised
    in the schema and enforced at the boundary: a value outside it is refused with a named error
    rather than handed to the tool to misinterpret.
    """
    description: str = ""
    enum: tuple = ()


def _signature(func) -> dict:
    """name -> (base json type, Param or None) for each parameter, from the type hints.

    `Annotated[T, Param(...)]` collapses to its base `T` for the JSON type and hands back the
    `Param`; a plain hint is its own base with no `Param`. This is read with `include_extras` so
    the metadata survives — without it, `get_type_hints` silently strips `Annotated` and the
    descriptions vanish, which is precisely the failure this exists to prevent."""
    hints = typing.get_type_hints(func, include_extras=True)
    out = {}
    for name in inspect.signature(func).parameters:
        hint = hints.get(name, _UNANNOTATED)
        meta = getattr(hint, "__metadata__", None)
        if meta is None:
            out[name] = (hint, None)
        else:
            out[name] = (hint.__origin__, next((m for m in meta if isinstance(m, Param)), None))
    return out


def spec_for(func) -> ToolSpec:
    """The JSON schema implied by a function signature, its `Param` hints and its declared
    `description`.

    A parameter with a default is optional; a parameter without a JSON-mappable annotation is
    refused, because an untyped parameter means the model is guessing what to send. The tool-level
    description is not derived from anything in the source — it is declared, beside `access` and
    `address`, and a tool without one is refused here, at boot, for the same reason an undeclared
    tool is: the model would be shown a contract it has to guess at, and a schema the harness half
    wrote is a lie the operator cannot see."""
    description = getattr(func, "description", None)
    if not isinstance(description, str) or not description.strip():
        raise Refused(f"{func.__name__} declares no description: the model is shown the tool as a "
                      "contract, and one it has to guess at is one it can misuse. Set "
                      f"`{func.__name__}.description` beside its `access`/`address` declaration to "
                      "the paragraph that states what it does and what comes back; the docstring is "
                      "for whoever reads the source, and no part of it reaches the model")
    types = _signature(func)
    signature = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise Refused(f"{func.__name__} takes *args/**kwargs; declare each parameter so "
                          "the schema says what the model must send")
        base, meta = types[name]
        if base not in _JSON_TYPE:
            raise Refused(f"{func.__name__}.{name} has no JSON-mappable annotation "
                          f"(got {base!r}); declare str, int, float, bool, list or dict")
        field = {"type": _JSON_TYPE[base]}
        if meta is not None:
            if meta.description:
                field["description"] = meta.description
            if meta.enum:
                field["enum"] = list(meta.enum)
        properties[name] = field
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return ToolSpec(
        name=func.__name__,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
    )


def _integral(value: str) -> bool:
    """True for exactly what `int()` accepts without guessing: an optional single sign and ASCII
    digits. `" 5"`, `"0x10"`, `"5.0"` and `"٣"` are not integers here — a digit-shaped string a
    model invented is a refusal naming the argument, not a value the harness inferred."""
    body = value[1:] if value[:1] in "+-" else value
    return bool(body) and body.isdigit() and body.isascii()


def _numeric(value: str) -> bool:
    """True for an optional sign and at most one dot with a digit on at least one side of it:
    `"1"`, `"1.5"`, `".5"` and `"1."` pass; `"."`, `"+."` and `"-."` (no digit on either side) and
    `"1.2.3"` (two dots) do not. No exponent form on purpose: `"1e3"` is refused with the argument
    named rather than silently widened, because a coercion that itself needs rules has stopped
    being a convenience and become a parser."""
    body = value[1:] if value[:1] in "+-" else value
    whole, _dot, rest = body.partition(".")
    if not whole and not rest:
        return False
    return all(not part or (part.isdigit() and part.isascii()) for part in (whole, rest))


def arguments_for(func, arguments: dict) -> dict:
    """Fit a model's arguments to the function: unknown names, missing required ones and wrong
    types are refused here, at the boundary, rather than as a TypeError from inside the tool.

    Types are enforced because a declared type is the only promise the schema makes that a server
    cannot break quietly. `fs.read` hands its argument to `open`, and `open(0)` is file descriptor 0 —
    stdin — so a hint that says `str` and a call that sends `0` lets a backend read the operator's
    terminal input with no grant held and no path for the veto point to confine, which then gets
    consulted about a location the tool never touched. The check runs before the policy, so a wrongly
    typed argument never reaches the veto point either: there is nothing to grant.

    One kindness: a number quoted as a string (`offset="5"`) is coerced, because models quote numbers
    constantly and refusing it spends a turn on a typo of form rather than of meaning. The coercion
    is exact — see `_integral`/`_numeric` — and `sent` carries the converted value, so no tool has to
    tolerate a type it was never told about."""
    types = _signature(func)
    signature = inspect.signature(func)
    unknown = sorted(set(arguments) - set(signature.parameters))
    if unknown:
        raise Refused(f"{func.__name__} does not accept {unknown}; it accepts "
                      f"{sorted(signature.parameters)}")
    missing = [n for n, p in signature.parameters.items()
               if p.default is inspect.Parameter.empty and n not in arguments]
    if missing:
        raise Refused(f"{func.__name__} requires {missing}")
    sent = dict(arguments)
    for name, value in list(sent.items()):
        expected, meta = types[name]
        if expected is bool and not isinstance(value, bool):
            raise Refused(f"{func.__name__}.{name} must be a bool, got {type(value).__name__}")
        if expected in (int, float) and isinstance(value, bool):
            raise Refused(f"{func.__name__}.{name} must be {_JSON_TYPE[expected]}, not a bool")
        if expected is str and not isinstance(value, str):
            raise Refused(f"{func.__name__}.{name} must be a string, got {type(value).__name__}")
        elif expected is int and not isinstance(value, int):
            if isinstance(value, str) and _integral(value):
                sent[name] = int(value)
            else:
                raise Refused(f"{func.__name__}.{name} must be an integer, got "
                              f"{type(value).__name__}")
        elif expected is float and not isinstance(value, (int, float)):
            if isinstance(value, str) and _numeric(value):
                sent[name] = float(value)
            else:
                raise Refused(f"{func.__name__}.{name} must be a number, got "
                              f"{type(value).__name__}")
        elif expected is list and not isinstance(value, list):
            raise Refused(f"{func.__name__}.{name} must be a list, got {type(value).__name__}")
        elif expected is dict and not isinstance(value, dict):
            raise Refused(f"{func.__name__}.{name} must be a dict, got {type(value).__name__}")
        # A declared enum is a promise in the schema; honour it here so a value the schema said was
        # impossible never reaches the tool to be silently misinterpreted. It judges the value the
        # tool will actually receive, so a coerced number is checked as the number it became.
        if meta is not None and meta.enum and sent[name] not in meta.enum:
            raise Refused(f"{func.__name__}.{name} must be one of {list(meta.enum)}, "
                          f"got {sent[name]!r}")
    return sent
