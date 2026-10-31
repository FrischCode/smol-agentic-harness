"""sah.record — the step taxonomy, its field discipline, and its serialization.

One line of the log is one step. A step's type is its `kind`; each kind declares a CLOSED
field set here, and a step may not carry a field its kind does not declare. The field table
is not presentation: the main reader of old logs is a model asked to look something up, so a
field name is the address a future question uses and renaming one is a breaking change.

Every step carries three stamped fields — `id`, `ts`, `issuer` — supplied by whoever appends,
never by the plugin whose work the step describes. Whether a step enters the model's context
is `to_model`, a property of the kind and not of the step.

Serialization is one JSON object per line: `dumps` and `loads` are the only places the wire
format exists, and `loads` validates, so a line the taxonomy cannot name is refused rather
than quietly skipped.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import pathlib
import sys

# --- field kinds -------------------------------------------------------------

TEXT = "text"        # prose, one fact, never a rendered sentence
IDENT = "ident"      # a name from a vocabulary the reader can enumerate
INT = "int"
FLOAT = "float"      # a quantity the world measured, not a number derived on read
BOOL = "bool"
JSON = "json"        # a structured value with no declared internals (arguments, grants)
REF = "ref"          # the id of one step
REFS = "refs"        # the ids of several steps, in order
PATHS = "paths"      # several places on disk, in order. Not ids: a path names nothing in the log
DIGEST = "digest"    # a hash of content stored elsewhere
PATH = "path"
TIME = "time"        # ISO-8601; the only time field is always `ts`

_PY_FOR = {
    TEXT: str, IDENT: str, DIGEST: str, PATH: str, TIME: str, REF: str,
    INT: int,
    FLOAT: float,
    BOOL: bool,
    JSON: (dict, list),
    REFS: list,
    PATHS: list,
}

# Names that may never be declared. `detail` is a dumping ground for rendered sentences — a
# rendered sentence is not a fact, and a fact inside one is ungreppable. `stamp`/`started`/
# `duration`/`time` are how one clock acquired four names; a derived duration is computed on
# read, never stored.
BANNED_FIELDS = frozenset({"detail", "stamp", "started", "duration", "time"})

# The stamping belongs to whoever appends, so a plugin cannot supply or forge it.
STAMPED = ("id", "ts", "issuer")


class Refused(Exception):
    """A step the taxonomy will not accept. The message names the way to fix it."""


class Kind:
    """A declared step kind: its name, its closed field table, and the to_model / must_surface
    flags that decide whether its steps enter the model's context and which of their fields a
    display level may not drop."""
    __slots__ = ("name", "fields", "to_model", "must_surface")

    def __init__(self, name, fields, to_model=False, must_surface=()):
        self.name = name
        self.fields = fields                    # field -> (field kind, required)
        self.to_model = to_model                # does replay put this in the model's context
        self.must_surface = frozenset(must_surface)


def kind(name, table, *, to_model=False, must_surface=()):
    """Declare one kind. The field table is positional, so no field name — `name`, `text`,
    `to_model` — can shadow a control argument."""
    return Kind(name, dict(table), to_model, must_surface)


# --- the taxonomy: twelve kinds ----------------------------------------------

KINDS = {k.name: k for k in (
    kind("message", {
        "role": (IDENT, True), "text": (TEXT, True), "reasoning": (TEXT, False),
        # The wire field the server used to deliver the turn's reasoning; the re-send mirrors it
        "reasoning_field": (IDENT, False),
        "outcome": (IDENT, False),
        "error": (TEXT, False), "model": (IDENT, False), "tokens_in": (INT, False),
        "tokens_out": (INT, False), "tokens_cached": (INT, False), "ms": (INT, False),
        # Rates the server measured about the request that is this step. A float is admissible for a
        # quantity the world measured, which is what separates it from the rendered strings the banned
        # names collected: these are the server's own token-per-second counts of its own work, and a
        # reader who wants a duration still derives it, as the ban on `duration` requires.
        "tps": (FLOAT, False), "prefill_tps": (FLOAT, False),
        "context_used": (INT, False), "context_max": (INT, False),
        # The wire's annotation of the turn: the server's raw finish word, and the fingerprint and
        # size of the logical request the turn was asked with. The request's content is the
        # conversation itself (the stream), so the pin is a digest and a count, not a copy of it.
        "finish": (IDENT, False), "req_sha": (DIGEST, False), "req_chars": (INT, False)},
        to_model=True, must_surface=("outcome",)),

    kind("call", {                                  # the model asking for a tool
        "refs": (REF, True), "call_id": (IDENT, False), "name": (IDENT, True),
        "args": (JSON, False)},
        to_model=True),

    kind("result", {
        "refs": (REF, True), "ok": (BOOL, True), "content": (TEXT, False),
        "images": (PATHS, False), "truncated": (BOOL, False), "bytes": (INT, False),
        "spilled": (PATH, False)},
        to_model=True, must_surface=("ok",)),

    kind("command", {                               # an inbound control act
        "action": (IDENT, True), "args": (JSON, False), "queue_id": (IDENT, False)}),

    # `subject`, never `kind`: a step's own type is already `kind`, and a subtype field may
    # not shadow the word that names it.
    kind("request", {                               # an outbound ask from the core
        "subject": (IDENT, True), "item": (TEXT, True), "why": (TEXT, False),
        "refs": (REF, False)}),

    kind("decision", {                              # closes a command or a request
        "refs": (REF, True), "outcome": (IDENT, True), "reason": (TEXT, False)},
        must_surface=("outcome",)),

    kind("cut", {                                   # a resume that detaches the tail it held
        # The step the conversation continues from. After a `cut`, the run's conversation is the
        # prefix ending at this step, and what the file held after it is detached: still in the
        # file for the audit, out of every conversation `lineage` rebuilds. The file is never
        # truncated — a cut is recorded, not performed — so a log cut twice shows both lineages
        # and both acts. Not model-facing: the operator who cut may be steering what comes next,
        # and a model told a cut happened would be told about the operator's edit.
        "to": (REF, True)}),

    kind("operation", {                             # a mutation attempted on an address
        "op": (IDENT, True), "address": (PATH, True), "outcome": (IDENT, True),
        "reason": (TEXT, False), "prior_digest": (DIGEST, False), "bytes": (INT, False)},
        must_surface=("outcome",)),

    kind("summary", {                               # compaction, including its failed attempts
        "refs": (REFS, True), "kept": (INT, True), "dropped": (INT, True),
        "text": (TEXT, True), "outcome": (IDENT, True), "reason": (TEXT, False),
        "tokens_cost": (INT, False),
        # Which attempt produced the slot (1 the free summary, 2 the compactness ask, 3 the
        # re-summarise, 4 the cut or the failed note), and the kind of text the slot holds.
        "stage": (INT, False), "slot": (IDENT, False),
        # Messages the cut rung dropped from the head of the backfill when the tail itself
        # exceeded the limit — absent at zero, the way the other stamps are: a grep for the
        # field finds exactly the compactions that cut the tail, and a resume skips those
        # messages on replay. The cut is a content-free count: the messages stand in the log
        # whole, and the count is what the walk's arithmetic needs to leave them out.
        "backfill_cut": (INT, False),
        # The budget the act ran against, in the record's own words: the limit and its two
        # ceilings, the resident parts split into system prompt and tool schemas (the tool
        # component is the regression guard for the measured defect where the tool surface was
        # ignored in the budget), the backfill, and the room left the summary.
        "budget": (TEXT, False)},
        to_model=True),

    kind("notice", {"subject": (IDENT, True), "text": (TEXT, True)}),

    kind("started", {
        "run_id": (IDENT, True), "root": (PATH, True), "cwd": (PATH, False),
        # The process behind the log: a run that dies without its `ended` line leaves a record that
        # claims to be alive, and the PID is the witness a reader on the same machine can check
        # against the process table.
        "pid": (INT, False),
        "model": (IDENT, False),
        "thinking": (IDENT, False), "grants": (JSON, False), "resumed_from": (IDENT, False),
        # The fixed head of the render input, stamped once: the exact system prose and tool
        # contract the model was told and offered, and the budget it was sent under. The prompt
        # file and the tool code are mutable world state, so the record holds the copy; the two
        # digests below are the head's identity and the run's own estimate of its window share.
        # `project_notes_file` is the working directory's instruction file framed into `system` at
        # boot (a mutable external file, like the prompt files); its content is in `system`, this is
        # the derivation input's identity. Absent unless the operator granted --project-notes and a
        # file was loaded.
        "system": (TEXT, False), "tools": (JSON, False), "max_tokens": (INT, False),
        # The window the compaction arithmetic ran against — declared, probed from the backend,
        # or the default; the CLI's `context` notice says which, and the number is stamped here
        # so a reader of a degenerate compaction never has to guess which window the run
        # believed it had.
        "context": (INT, False),
        "max_turns": (INT, False), "grace_turns": (INT, False), "max_runtime": (INT, False),
        # The transport's wire knobs, stamped the way `max_runtime` is: absent at their defaults
        # (streaming, usage requested), present only when the operator turned one off, so a grep for
        # `"streaming": false` finds exactly the runs that talked to the server in the other shape.
        "streaming": (BOOL, False), "usage": (BOOL, False),
        # The reasoning wire shape the operator declared for this server, stamped the same way as
        # `streaming` and `usage`: absent at the defaults (the mirror, no pin), present only when
        # the operator turned one on, so a reader of the wire trail can tell why a request wrote
        # the reasoning somewhere the conversation did not come in on.
        "reasoning_out": (IDENT, False), "require_reasoning_content": (BOOL, False),
        "prefix_digest": (DIGEST, False), "resident_tokens": (INT, False),
        # The reasoning level the summary calls run at, present only when the operator named one
        # (`--summary-think`): absent at the default, which is the run's own level left alone, the
        # same stamp idiom as `streaming` and `usage`.
        "project_notes_file": (PATH, False), "summary_thinking": (IDENT, False)}),

    kind("ended", {"reason": (IDENT, True), "exit_code": (INT, True)}),
)}

# Four kinds are model-facing: that is the entire context surface.
TO_MODEL = frozenset(k.name for k in KINDS.values() if k.to_model)

# Kinds whose accepted facts a resume re-applies as authority. Empty by decision: what is in
# force comes from the invocation, so `decision` and `operation` are audit, and the only thing
# replay re-applies is conversation content. A member is added deliberately — with the reasoning
# written down — not silently, so the set does not grow by accident.
ESTABLISHING: frozenset = frozenset()

# Closed vocabularies for the `ident` fields that subtype a step or carry a verdict. `ident` is
# documented as "a name from a vocabulary the reader can enumerate" — which is only true if someone
# owns the list. A (kind, field) named here accepts only these values, on write and on strict read;
# adding one is a deliberate, greppable edit to the taxonomy, so a reader asking "what kinds of
# notice exist?" or "which runs failed?" answers it from this dict rather than by scanning every log
# or prefix-matching a field.
#
# What belongs here is a set the *code* fixes. What stays open on purpose is a set that grows with
# the world the harness is pointed at — a tool `name`, an `op`, a model — because a wrong enum
# refuses a legitimate future value, which is worse than leaving the field open. The verdict fields
# were left open longest for the weakest reason: an outcome that carried a sentence ("failed: the
# socket died") is not a verdict anyone can group by, and every distinct error became a new outcome.
VOCABULARY = {
    # `nudge` = the repetition guard stepped in or a budget ran out (the wrap-up notice); `spawn` =
    # a child was launched; `replay` = a resume rebuilt this conversation over a step it could not
    # read; `context` = the window the run runs against was not declared by the operator — probed
    # from the backend, or the default — and the record says which, so a degenerate compaction is
    # readable as a config gap, not a mystery. The last exists so a gap in the history is a fact
    # inside the history, not a detail known only to whoever was watching.
    ("notice", "subject"): frozenset({"nudge", "spawn", "replay", "compact", "context",
                                       # the completion declaration landing on an unattended run
                                       "verify",
                                       # a crash's traceback, recorded beside its `ended` line
                                       "crash",
                                       # an operator-named extension file loading at boot: which
                                       # file, and the tools it brought into the run's table
                                       "extension"}),
    # How a run stopped, as opposed to the exit code it stopped with. `complete` = the objective
    # was answered; `finished` = the operator said stop at a turn boundary; `interrupted` = a signal
    # landed mid-run; `stalled` = the turn produced neither prose nor a call, repeatedly; `truncated`
    # = the answer kept hitting the output limit; `max_turns` / `max_runtime` = the turn or the time
    # bound ran out and the grace turns the model was told to wrap up in were spent without an answer
    # (an answer inside the grace ends the run `complete`, which is the point of the grace);
    # `failed` = the backend or the run itself broke; `needs_operator` = an ask came to a second
    # attempt with nobody
    # watching, so the run stopped and left the question in the log rather than waiting out a timeout
    # for a voice that does not exist. That outcome is not one the loop's own branches name, which is the
    # argument for closing the set: a closed vocabulary lists the outcomes the code paths do not label.
    ("ended", "reason"): frozenset({"complete", "finished", "interrupted", "failed", "stalled",
                                     "max_turns", "max_runtime", "truncated", "needs_operator",
                                     # no-call nudges exhausted on a run no operator is watching:
                                     # the run cannot prove done, and must not end complete
                                     "blocked",
                                     # the run died of an exception: the traceback is on the record
                                     # beside this line, and the exception was re-raised
                                     "crashed"}),
    ("decision", "outcome"): frozenset({"accepted", "refused"}),
    # `allowed`, not `ok`: a policy verdict and a tool's own success are different facts, and the
    # log has always named them with different fields.
    ("operation", "outcome"): frozenset({"allowed", "refused"}),
    ("summary", "outcome"): frozenset({"ok", "failed"}),
    # The kind of text the slot holds: a summary a stage produced, a cut one (the last text cut to
    # the room and marked), or the failed note (no attempt produced text). The backfill note's
    # covered/uncovered words ride on this.
    ("summary", "slot"): frozenset({"summary", "cut", "failed"}),
    # One assistant turn's verdict. `truncated` is the server saying the answer was cut off, which
    # is neither success nor failure and must not be smoothed into either.
    ("message", "outcome"): frozenset({"ok", "failed", "truncated"}),
    # The dialect's own reasoning field names. The re-send mirrors whatever the log records here,
    # so a corrupted log cannot put a fourth field name on the wire.
    ("message", "reasoning_field"): frozenset({"reasoning_content", "reasoning", "reasoning_text"}),
}

# (kind, field) pairs no display level may drop: the verdicts.
MUST_SURFACE = frozenset((k.name, f) for k in KINDS.values() for f in k.must_surface)


def _self_check() -> None:
    """Reject an impossible taxonomy at import, before any log is written."""
    reserved = BANNED_FIELDS | set(STAMPED) | {"kind"}
    for spec in KINDS.values():
        bad = sorted(set(spec.fields) & reserved)
        if bad:
            raise Refused(f"{spec.name} declares forbidden field(s) {bad}: time is `ts`, a "
                          "rendered sentence is not a fact, and the stamping is core-owned")
        for field, (fk, _required) in spec.fields.items():
            if fk not in _PY_FOR:
                raise Refused(f"{spec.name}.{field} has unknown field kind {fk!r}")
        for field in spec.must_surface:
            if field not in spec.fields:
                raise Refused(f"{spec.name}.must_surface names {field!r}, which it does not "
                              "declare, so it cannot be surfaced")
    for (k, f), words in VOCABULARY.items():
        if k not in KINDS:
            raise Refused(f"VOCABULARY names kind {k!r}, which is not declared")
        spec = KINDS[k]
        if f not in spec.fields or spec.fields[f][0] is not IDENT:
            raise Refused(f"VOCABULARY names {k}.{f}, which is not a declared ident field")
        if not words or not all(isinstance(w, str) and w for w in words):
            raise Refused(f"VOCABULARY[{k}.{f}] must be a non-empty set of non-empty strings")


_self_check()


def now() -> str:
    """The current UTC time as an ISO-8601 string, to the second — the value the `ts` stamp holds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make(kind_name: str, issuer: str, seq: int, /, **fields) -> dict:
    """Build one validated, stamped step, ready for `dumps`.

    `issuer` and `seq` come from whoever appends: the core stamps who acted and the log owns
    the sequence, so an id is never a plugin's claim about itself."""
    if kind_name not in KINDS:
        raise Refused(f"unknown step kind {kind_name!r}; declared kinds are {sorted(KINDS)}")
    if not isinstance(issuer, str) or not issuer:
        raise Refused("issuer must be a non-empty string naming who acted")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise Refused("seq must be a non-negative int owned by the log")

    spec = KINDS[kind_name]
    unknown = sorted(set(fields) - set(spec.fields))
    if unknown:
        raise Refused(f"{kind_name} does not declare {unknown}; declared fields are "
                      f"{sorted(spec.fields)}")
    missing = sorted(f for f, (_fk, required) in spec.fields.items()
                     if required and f not in fields)
    if missing:
        raise Refused(f"{kind_name} requires {missing}")

    step = {"kind": kind_name, "id": f"{seq:06d}-{kind_name}", "ts": now(), "issuer": issuer}
    for field, value in fields.items():
        fk = spec.fields[field][0]
        if fk is REFS and not all(isinstance(v, str) for v in value):
            raise Refused(f"{kind_name}.{field} must be a list of step ids")
        if fk is PATHS and not all(isinstance(v, str) and v for v in value):
            raise Refused(f"{kind_name}.{field} must be a list of paths")
        if not _fits(fk, value):
            raise Refused(f"{kind_name}.{field} must be {fk}, got {type(value).__name__}")
        _check_vocab(kind_name, field, value)
        step[field] = value
    return step


def _check_vocab(kind_name: str, field: str, value) -> None:
    """Enforce a closed vocabulary declared in VOCABULARY, if this (kind, field) has one."""
    words = VOCABULARY.get((kind_name, field))
    if words is not None and value not in words:
        raise Refused(f"{kind_name}.{field} must be one of {sorted(words)}, got {value!r}")


def _fits(field_kind: str, value) -> bool:
    expected = _PY_FOR[field_kind]
    if isinstance(value, bool) is (expected is not bool):   # bool is a subclass of int
        return False
    if not isinstance(value, expected):
        return False
    return (field_kind is not REFS or all(isinstance(v, str) for v in value)) and (
        field_kind is not PATHS or all(isinstance(v, str) and v for v in value))


def dumps(step: dict) -> str:
    """Serialise one step to a single JSON line: keys sorted, non-ASCII kept literal. The one
    place the wire format's serialisation exists."""
    return json.dumps(step, ensure_ascii=False, sort_keys=True)


def loads(line: str, *, strict: bool = True) -> dict:
    """Parse and validate one log line. A line the taxonomy cannot name is refused.

    With ``strict=False`` the reader is lenient about *vocabulary* and not about damage: a kind this
    build has never heard of, or a field on a known kind that it predates, is kept rather than refused,
    because a projection's job is to show a log it may not fully understand. A declared field of the
    wrong type is still refused — that is corruption, not the future, and dressing it up as an
    unrecognised word would let a torn line into a report as though it were a new feature. The audit
    reader never passes the flag: "what did this run do" may not be answered by a step nothing checked.
    """
    try:
        obj = json.loads(line)
    except ValueError as e:
        raise Refused(f"malformed log line: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("kind"), str):
        raise Refused("log line is not a step object with a `kind`")
    return validate(obj) if strict else _lenient(obj)


def _lenient(step: dict) -> dict:
    """Check what this build can check and keep the rest. See `loads`."""
    spec = KINDS.get(step["kind"])
    if spec is None:
        return step                                   # a future kind: its own fields are the rendering
    for field, value in step.items():
        if field in STAMPED or field == "kind":
            continue
        declared = spec.fields.get(field)
        if declared is None:
            continue                                  # a field this build predates; the projector prints it
        if not _fits(declared[0], value):
            raise Refused(f"{step['kind']}.{field} must be {declared[0]}")
        # No vocabulary check here, on purpose. A word this build cannot name may be the future (a
        # kind or a field it predates, which is what leniency exists for) or it may be history: a log
        # written before a set was closed holds a value the closing made unnameable, and a projector
        # that refused it would make closing a vocabulary destroy history. The audit reader never
        # reaches this function, so the strict rule still holds where the audit needs it.
    return step


def read(path, *, tolerate_damage: bool = False, strict: bool = True) -> tuple[list[dict], str]:
    """Every step in a log file, plus one sentence about the lines this reader chose to drop.

    `read` is the one reader for a log file, and the policy is the flag: the strict form refuses a
    damaged line outright, the tolerant form names and drops it. The two answers to "a damaged
    line" — refuse, or skip-and-report — are both legitimate, for different readers.

    Tolerance covers a damaged line *anywhere*, not only at the end. Tail-only tolerance is
    useless: the repair in `append` closes a torn line, so the next step any run writes moves the
    damage into the middle of the file, and a log that crashed once would then be unreadable by
    `export`, `recall`, resume and the REPL for the rest of its life, however many good lines it
    holds. The reason the repair (rather than a reader) is what makes mid-file damage survivable
    is that, with it, an unreadable line is self-contained — its neighbours parse, and skipping it
    loses exactly the one step it holds.

    So the tolerant form is honest rather than strict: it names every line it dropped, with its
    position, and the strict form — `steps()`, which the audit path uses — still refuses the file
    outright. A projection that skips a step and says so can be questioned; one that skips it
    silently is how an audit trail becomes decoration. A log that does not exist is an empty history
    with nothing to report, not an error."""
    file = pathlib.Path(path)
    try:
        lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return [], ""
    steps, dropped = [], []
    for index, line in enumerate(lines):
        try:
            steps.append(loads(line, strict=strict))
        except Refused as exc:
            if not tolerate_damage:
                raise
            dropped.append(f"line {index + 1} of {len(lines)}: {exc}")
    if not dropped:
        return steps, ""
    named = "; ".join(dropped[:3])
    extra = f" (+{len(dropped) - 3} more)" if len(dropped) > 3 else ""
    return steps, (f"{len(dropped)} line(s) unreadable and dropped — {named}{extra}"
                   "; a writer that died mid-append leaves one of these")


def lineage(steps) -> list[dict]:
    """The run's steps with every cut applied: the conversation in force, and nothing of what a
    cut detached.

    A `cut` step is an instruction to the reader, not a step of the conversation: it says the
    conversation is now the prefix ending at the step it names, and the steps the file held after
    that step are detached. Walking in file order and truncating at each cut composes — a later
    cut may aim at a step of a lineage an earlier cut created, and a detached step never
    re-enters, because it is gone from the list before a later cut runs. Cut steps themselves do
    not survive in the answer either: they are how the operator acted on the run, a fact of the
    raw file and the audit, not of the conversation, and a model reading a run must never be told
    a cut happened — the operator who cut may be editing what comes next.

    This is the projection every conversation consumer rebuilds from — `replay`, `result`,
    `recall`, the reach note — which is why it sits beside `read` rather than in each of them.
    The raw file is the record: `export` and the audit read it whole, cut steps and all. A cut
    whose target is not in the list is a corruption, not a gap to skip: the projection cannot be
    answered honestly, so it refuses, and the file itself is untouched, so the audit still reads
    it whole.
    """
    valid: list[dict] = []
    for step in steps:
        if step["kind"] == "cut":
            target = step.get("to", "")
            index = next((i for i, prior in enumerate(valid) if prior["id"] == target), None)
            if index is None:
                raise Refused(f"cut step {step['id']} names {target!r}, which is not in the "
                              f"conversation it detaches a tail from")
            valid = valid[: index + 1]
        else:
            valid.append(step)
    return valid


def open_calls(steps) -> list[dict]:
    """The call steps of a list that hold no result in it.

    A `result` links to its `call` by naming the call step's id in `refs` — the pairing `replay`
    reads — so a list that stops mid-exchange (after a turn's header, or after one of its calls)
    names its outstanding calls here. A run that dies mid-turn may carry them: the file is all it
    will ever have, and `replay` holds such a turn of its own so the result still has something to
    answer. A deliberate cut may not: the operator chose the stop, and a conversation that ends
    owing a tool result is refused by the server it is sent to, so the stop is refused with its
    outstanding calls named instead of written as a cut.
    """
    answered = {step.get("refs", "") for step in steps if step["kind"] == "result"}
    return [step for step in steps if step["kind"] == "call" and step["id"] not in answered]


def validate(step: dict) -> dict:
    """Check a parsed step against its kind: stamping, declared fields, value types."""
    name = step["kind"]
    if name not in KINDS:
        raise Refused(f"step claims undeclared kind {name!r}")
    spec = KINDS[name]
    for stamped in STAMPED:
        if stamped not in step:
            raise Refused(f"{name} step is missing the stamped field {stamped!r}")
    extra = sorted(set(step) - set(spec.fields) - set(STAMPED) - {"kind"})
    if extra:
        raise Refused(f"{name} step carries undeclared field(s) {extra}")
    for field, value in step.items():
        if field in STAMPED or field == "kind":
            continue
        fk = spec.fields[field][0]
        if not _fits(fk, value):
            raise Refused(f"{name}.{field} must be {fk}")
        _check_vocab(name, field, value)
    return step


class Log:
    """The append-only step log: one file, one sequence, one way in.

    Every step becomes a fact through `append`, which mints the sequence and takes the issuer
    from whoever owns the act. Nothing in the harness hands a `Log` to a plugin or a tool — a
    step is a fact only once the core writes it, which is why a client cannot manufacture an
    accepted grant by injecting a record, and why replay re-applies conversation content
    rather than claims.

    Subscribers see each step the moment it is written, so a projection can report a verdict as
    it lands rather than after the fact. A subscriber receives the dict that was written and
    cannot change it — a projection that wanted to edit a fact would need the file. A subscriber
    that raises is reported and skipped: a client's bug must not stop the record, and a record is
    what the run exists to leave behind.
    """

    def __init__(self, path, *, echo=None):
        self.path = pathlib.Path(path)
        self.seq = self._continuation()
        self._subscribers = [echo] if echo is not None else []

    def _continuation(self) -> int:
        """The number the next step gets, continuing whatever this file already holds.

        The sequence is a property of the log, not of this object: a resumed run, `sah export` and
        the REPL each build a Log over the same file, and step ids are what `refs` points at. A
        counter that restarted at zero would give two steps one name, and an operator answering
        `--reply <id>` could be answering an older question wearing the same number.

        Every line counts, including one too damaged to parse: a torn final line is a step whose id
        was already spent. Where the two disagree, the highest id in the file wins — that is what a
        line two steps were once glued into looks like."""
        try:
            steps = [line for line in self.path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        except OSError:
            return 0                                    # no log yet: this one starts at zero
        for line in reversed(steps):
            try:
                return max(len(steps), int(loads(line)["id"].split("-", 1)[0]) + 1)
            except (Refused, KeyError, ValueError):
                continue                                # damage is counted above, not parsed here
        return len(steps)

    def subscribe(self, sink) -> None:
        """See every step from now on, as it is written. This is the whole surface a projection
        gets: notification, with no way back into the loop."""
        self._subscribers.append(sink)

    def append(self, kind: str, issuer: str, **fields) -> dict:
        """Write one step: mint the sequence, stamp and validate it, append it (closing a torn
        final line first, so a kill mid-append damages exactly one line), and notify the
        subscribers. Returns the step that was written."""
        step = make(kind, issuer, self.seq, **fields)
        self.seq += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            # One step per line is the format, and a run killed mid-append leaves a line with no
            # newline at the end. Closing that line before writing keeps the damage to the one
            # line it damaged: glued onto the next step it would poison a line in the middle of
            # the log, which every reader refuses.
            if self._ends_mid_line():
                handle.write("\n")
            handle.write(dumps(step) + "\n")
        for sink in self._subscribers:
            try:
                sink(step)
            except Exception as exc:
                print(f"sah: a step subscriber failed on {step['kind']}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return step

    def _ends_mid_line(self) -> bool:
        """True when the file has content and its last byte is not a newline — a torn final line.
        Only the last byte is read, so the check does not scale with the size of the log."""
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                if handle.tell() == 0:
                    return False
                handle.seek(handle.tell() - 1)
                return handle.read(1) != b"\n"
        except OSError:
            return False                                # no file: nothing to close

    def steps(self) -> list[dict]:
        """Every step, validated on the way in. A line the taxonomy cannot name is a refusal,
        never a skipped line — silence is how an audit trail becomes decorative. This is the audit
        form; a reader that projects a run wants `read()`, which tolerates a torn tail and reports
        it."""
        steps, _ = read(self.path)
        return steps

    def read(self, *, strict: bool = True) -> tuple[list[dict], str]:
        """The steps, and what was lost to get them: `(steps, "")` for an intact log, `(steps,
        sentence)` when lines were too damaged to parse. For anything that *reports* on a run —
        resume, export, a status line. Dropping them is safe to show precisely because the ids in
        the surviving lines are still true, so a gap in the sequence stays visible.

        `strict=False` asks the lenient vocabulary policy — see the module's `loads` — for a reader whose
        whole job is to render a log it may not fully understand."""
        return read(self.path, tolerate_damage=True, strict=strict)   # the module function, not this method

    def of_kind(self, kind: str) -> list[dict]:
        """Every step of one kind, in log order."""
        return [step for step in self.steps() if step["kind"] == kind]

    def verdicts(self) -> list[tuple[str, str]]:
        """The facts no display level may drop, read straight from the log so a client's
        filtering can be checked against what was actually recorded."""
        out = []
        for step in self.steps():
            for field in KINDS[step["kind"]].must_surface:
                if field in step:
                    out.append((f"{step['kind']}.{field}", step[field]))
        return out
