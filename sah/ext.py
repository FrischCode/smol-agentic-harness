"""sah.ext — the one surface a contributor is registered through.

Two contracts, and only two, are what anything (the built-ins or a third party) gets:

- **tools** — plain type-hinted functions added to the run's tool set. A tool returns text; it never
  sees the log, the policy or the loop, and every call still passes the veto point. So a capability is
  not smuggled in by being "an extension": contributing a tool buys no permission the operator did not
  grant, because the gate is on the call, not on who registered it.
- **on(step)** — see each step as it is written. Notification only: the return value is ignored and
  there is no way back into the loop. This is how a projection (the terminal, the verdict floor, the
  ask-notifier) reports what happened without being able to cause it.

An extension may have either half, both, or (uselessly) neither. There is deliberately no third
contract that lets a plugin answer a core question (`provide`) — that category is real but has no
inhabited use yet, so it is not built. The loop already *consults* one advisor (the repetition guard,
which returns a decision and writes nothing); if a second ever wants in, the seam would generalise
that consult-and-core-writes shape into a `provide` contract, and `sah.agent`'s guard check is where
that hook would live. Nor can an extension register a command: commands are inbound
state changes the *core* accepts or refuses (the control queue), and giving a plugin a channel to
inject one would hand it the one write path the whole design reserves for the core.

Beside the two contracts there are two surfaces that are data rather than doors. The first is
`findings`, a tuple of facts (one non-empty string each) an extension may hand in at boot. The core
reads them once, after
the load notice, and records each fact as a `notice` step it stamps, sequences and subjects itself —
the extension cannot write the log (the stamp belongs to whoever appends), and the subject is the
core's word the same way, because `notice.subject` is a closed vocabulary declared in the core's
taxonomy (sah/record.py): a plugin that widened a vocabulary the reader can enumerate would be
writing the taxonomy, not a finding. The word is `extension`, the load line's own, so the file's
facts ride beside its load line; the extension's own naming rides in the words of the fact itself.
A finding is not a contract: nothing in the loop reads it, nothing answers it, and it reaches no
model (a `notice` is not `to_model`), so it is not the parked `provide` either, which answers a core
question. It is the boot's narration getting a second voice: what a loaded file found at its own
startup (a server reached, a tool it refused and why) is a fact a later reader of the record should
hold, and the record is the core's to write.

The second is `status`, the console's transient line — the mirror image: a door the core hands out
rather than one the extension hands in. The core builds one per run (sah.render.Status) and sets it on
every extension at assembly, and an extension's `say` renders a line the console draws and the record
does not hold — not a step (nothing is stamped, sequenced or exported), not the operator's stream (it
lands on the console's face, never on stderr), and it reaches no model. It is the live class: a
watcher's observation the moment a step lands, or the progress a long tool owes mid-call. Where a
finding and a line split is where the record and the console split: a finding is a fact a later
reader should hold, and the record is where a later reader looks; a line is a word for the operator
who is watching now, and where there is no one watching there is no line, because the fact it points
at is in the record. The core owns the channel and its rendering the way it owns the record's stamps:
the extension proposes the words, the way a finding proposes a fact. A line, like a finding, carries
its own naming in its words: the console's face has no subject column the core could fill, and an
operator reading two files' lines has to tell them apart from the lines themselves — so a line begins
with its speaker, and the core, which owns the channel and the mark, leaves the words to the
extension that chose them.

`assemble` is the single place tools get bound and watchers get wired, so a run's whole capability set
is one readable list rather than a pile of `dict.update` calls, and the built-ins use the exact path a
plugin would. A tool whose schema cannot be derived fails here, at boot, rather than reaching a model
as a half-formed tool.
"""
from __future__ import annotations

from sah.record import Refused
from sah.contract import declaration, spec_for
from sah.render import Status


class Extension:
    """Contributes tools and/or watches steps. Subclass it, or pass any object duck-typed the same
    way (a tool table's worth of `tools`, an `on` method). Both halves are optional. Two surfaces
    ride along as data, not doors: `findings`, the boot facts the core records, and `status`, the
    console's transient line the core hands out at assembly (the module docstring states why
    neither is a contract)."""

    tools: tuple = ()          # type-hinted functions; their schemas are derived at assembly
    findings: tuple = ()       # boot facts (one non-empty string each) the core records as notices
    status: Status | None = None  # the console's transient line; the core sets it at assembly

    def on(self, step: dict) -> None:   # the `on` contract: notification, ignored
        """See each step as it is written. Default is no-op, so a tool-only extension need not define
        it; the presence of a non-default `on` is what makes `assemble` subscribe the extension."""


def assemble(extensions, log, status: Status | None = None) -> dict:
    """Build the tool table and wire every watcher — the whole registration path.

    Returns name -> function. A tool with an un-derivable schema raises `Refused` here (before a
    model is ever offered it); a name claimed by two extensions raises rather than one silently
    shadowing the other. Watchers are subscribed to `log`, which isolates one that raises: a
    projection's bug must never stop the record.

    The console's transient line is handed in here too: `status` is set on every extension (a no-op
    one when the caller has no console), so a watcher — or a long tool's body — can `say` a line to
    the watching operator without ever touching the record. The core owns the channel and its
    rendering; the extension proposes the words (the module docstring states the rest)."""
    tools: dict[str, callable] = {}
    door = status if status is not None else Status()
    for ext in extensions:
        ext.status = door
        for func in getattr(ext, "tools", ()):
            spec_for(func)                              # a bad tool is caught now, not by the model
            declaration(func)                           # ...nor one that never said what it touches
            if func.__name__ in tools:
                raise Refused(f"two extensions both provide {func.__name__!r}; a run may not have "
                              "one tool mean two things")
            tools[func.__name__] = func
        on = getattr(ext, "on", None)
        if on is not None and getattr(type(ext), "on", None) is not Extension.on:
            log.subscribe(on)                           # only a real watcher, not the no-op default
    return tools


def findings_of(extension) -> tuple:
    """The boot facts one extension hands up, validated the way a tool's schema is: a finding that
    does not carry is a named refusal at boot, not a garbage step in the record and not a traceback
    the operator has to decode.

    A finding is one non-empty string: a fact, in the words the extension found it, said in a
    sentence that names what it is about. The record's subject is the core's (the closed vocabulary
    is the reader's enumeration, and a plugin does not widen it), so the extension's own naming
    travels in the fact itself. This decides which hands are clean enough to be taken from; what the
    core does with a clean finding is the caller's act."""
    raw = getattr(extension, "findings", ())
    if not isinstance(raw, tuple):
        raise Refused(f"an extension's findings must be a tuple of one-fact strings, "
                      f"got {type(raw).__name__}")
    for finding in raw:
        if not isinstance(finding, str) or not finding.strip():
            raise Refused(f"every finding is one non-empty string stating a fact, got {finding!r}")
    return raw
