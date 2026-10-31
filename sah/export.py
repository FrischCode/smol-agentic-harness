"""sah.export — project a run's log into a readable Markdown report.

A projection, not a second writer: it reads the same `log.jsonl` every other consumer reads and
renders it. There is no separate export state and no re-derivation of meaning — the log is the
record, this is one way of showing it. So the report can never disagree with the transcript; if the
two looked different, the export would be a second source of truth, which is the failure mode this
whole tree is organised to avoid.

Rendering follows the replay principle: a kind it knows gets a purpose-built line; a kind it does
not know (a future schema, or an older export reading a newer log) falls back to printing its
declared fields rather than dropping them silently. A report that quietly omits a step type is worse
than one that prints it crudely. That fallback is only reachable if the reader in front of it agrees:
the log reader is asked here in its lenient vocabulary form, which keeps a kind or a field this build
predates instead of refusing the file. Read strictly — the way the audit path reads it — one future
step would make the whole report unreadable, and the crude line would exist in a function no caller
could reach.

A run nobody is watching leaves this projection in its own folder as `report.md` at the ending
(`write_report`): the reader who arrives after the fact should not have to know this command exists
to read the run.
"""
from __future__ import annotations

import json
import os

from sah import confine
from sah.record import KINDS, Log
from sah.render import trailer_shape

_META = ("id", "ts", "issuer", "kind")          # stamped columns, not content to render


def _args(step) -> str:
    return json.dumps(step.get("args", {}), ensure_ascii=False, sort_keys=True)


def _indented(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines()) or "    (empty)"


def _message(step) -> list[str]:
    if step.get("outcome") == "failed" or step.get("error"):
        return [f"> ⚠ turn failed: {step.get('error', '')}"]
    # The author, not the role, names the voice: the core puts words in the user's seat (a nudge,
    # a wrap-up, a future word), and the report says the core spoke them. The role is the shape the
    # wire carries; the issuer is who spoke, and the reader is owed the who. The rule keys on the
    # issuer, never on what the word is about: the channel owns the face.
    role = step.get("role", "")
    who = ("core" if step.get("issuer") == "core"
           else {"user": "you", "assistant": "agent"}.get(role, role))
    text = (step.get("text") or "").strip()
    lines = [f"**{who}:** {text}"] if text else []
    # The status footer the terminal closes the turn's box with, projected from the same step: the
    # usage the footer reads is already on the record, so the report shows the window filling as it
    # fills -- the last footer before a silence is what tells a long request from a hang. A turn the
    # server never annotated gets no footer, and the failed branch above keeps its warning as the
    # turn's only face.
    trailer = trailer_shape(step)
    if trailer:
        lines.append(f"*{trailer}*")
    return lines


def _call(step) -> list[str]:
    return [f"▶ `{step.get('name', '')}` {_args(step)}"]


def _result(step, names: dict[str, str]) -> list[str]:
    # A result carries no name of its own — the name belongs to the call it answers, reached through
    # `refs`. Falling back to a literal "output" made a report of ten different tools read as ten
    # blocks titled `output:`, which throws away the one label a reader has to work with.
    mark = "✓" if step.get("ok") else "✗"
    label = names.get(step.get("refs", ""), "output")
    content = (step.get("content") or "").rstrip()
    return [f"{mark} {label}:", _indented(content)] if content else [f"{mark} {label} (no output)"]


def _operation(step) -> list[str]:
    # A human report needs the refusals, not the routine allowances; the allowed act is already the
    # call/result above it. An exec disclosure or a refusal is the part that earns its line.
    if step.get("outcome") == "refused":
        return [f"! refused `{step.get('op', '')}` — {step.get('reason', '')}"]
    if step.get("reason") == confine.EXEC_DISCLOSURE:
        # The exact sentence the policy emits, compared as a constant: matching a phrase from inside it
        # printed any step that happened to contain those words — a refusal quoting the disclosure back,
        # for instance — as the italic allowance line, and a rewording in the policy dropped this one.
        return [f"  _{step['reason']}_"]
    return []


def _request(step) -> list[str]:
    return [f"? asked ({step.get('subject', '')}) `{step.get('item', '')}` — {step.get('why', '')}"]


def _decision(step) -> list[str]:
    return [f"↳ decided `{step.get('outcome', '')}` — {step.get('reason', '')}"]


def _notice(step) -> list[str]:
    return [f"• {step.get('subject', 'notice')}: {step.get('text', '')}"]


def _summary(step) -> list[str]:
    # The verdict and its reason are separate fields, and the report shows both: `failed` alone tells an
    # operator that compaction gave up, which is not a thing they can act on.
    verdict = step.get("outcome", "") + (f": {step['reason']}" if step.get("reason") else "")
    head = f"### summary ({verdict}, kept {step.get('kept', '?')}, dropped {step.get('dropped', '?')})"
    return [head, "", (step.get("text") or "").strip()]


# Unknown kinds fall back to this rather than vanishing: print what the step declared.
def _generic(step) -> list[str]:
    body = ", ".join(f"{k}={v!r}" for k, v in step.items() if k not in _META)
    return [f"· {step['kind']}: {body}"]


def _undeclared(step) -> list[str]:
    """Fields a *known* kind carries that this build's taxonomy never offered.

    A newer log adds fields to kinds that already exist, and a projector that renders only the fields it
    knows would drop information the log holds without saying so. Printing them crudely is the same
    bargain `_generic` strikes for a kind it cannot name: shown, and visibly crude."""
    spec = KINDS.get(step["kind"])
    if spec is None:
        return []
    extra = sorted(set(step) - set(spec.fields) - set(_META))
    if not extra:
        return []
    return ["  _" + ", ".join(f"{k}={step[k]!r}" for k in extra) + "_"]


_HANDLERS = {
    "message": _message, "call": _call, "result": _result, "operation": _operation,
    "request": _request, "decision": _decision, "notice": _notice, "summary": _summary,
}


def render(steps) -> str:
    """Turn a run's steps into Markdown. Pure: same steps, same text; writes nothing.

    Consecutive step blocks are separated by exactly one blank line — the turn seam a reader
    needs — never inside a block, and never doubled."""
    openings = [s for s in steps if s["kind"] == "started"]
    started = openings[0] if openings else {}
    endings = [s for s in steps if s["kind"] == "ended"]
    ended = endings[-1] if endings else None
    # A REPL session writes one pair per objective, so more than one opening means the boundaries between
    # objectives are content the reader needs, not header material: showing only the first start and the
    # last end collapsed a four-turn session into one run that ended once.
    session = len(openings) > 1
    names = {s["id"]: s.get("name", "") for s in steps if s["kind"] == "call" and "id" in s}

    lines = [f"# run {started.get('run_id', 'run')}", ""]

    meta = []
    if started.get("model"):
        meta.append(f"model: {started['model']}")
    if started.get("resumed_from"):
        meta.append(f"resumed from: {started['resumed_from']}")
    if started.get("grants"):
        meta.append(f"grants: {json.dumps(started['grants'], ensure_ascii=False)}")
    if ended and not session:
        meta.append(f"ended: {ended['reason']} (exit {ended['exit_code']})")
    if meta:
        lines.append(" · ".join(meta))
        lines.append("")

    for step in steps:
        if step["kind"] == "started":
            if not (session and step is not started):
                continue
            block = [f"*run {step.get('run_id', 'run')} begins"
                     + (f" · model {step['model']}" if step.get("model") else "") + "*"]
        elif step["kind"] == "ended":
            if not session:
                continue
            block = ["---", f"*ended: {step['reason']} (exit {step['exit_code']})*"]
        elif step["kind"] == "result":
            block = _result(step, names) + _undeclared(step)
        else:
            block = _HANDLERS.get(step["kind"], _generic)(step) + _undeclared(step)
        # One blank line before each non-empty block, and only when the previous line is not
        # already a blank: that keeps the separation at exactly one even where a block renders
        # nothing or carries a blank line of its own (a summary), and it is what keeps the
        # `---` boundary above a blank line — a rule line directly under a text line parses as
        # a setext heading underline, not a thematic break.
        if block and lines and lines[-1] != "":
            lines.append("")
        lines.extend(block)

    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir: str, err=None) -> bool:
    """The `report.md` an unattended run leaves in its own folder: the same projection `sah export`
    renders, written by the harness as the run's last act, for every ending — the bad run's report
    is the one a caller arriving after the fact most wants to read.

    A projection, not a second record: the header names its source and its step count, and the log
    remains the record. It is written only when the log holds an `ended` step — a run still in
    flight has no report, and a partial one would look finished to exactly the reader it is for.
    The write is atomic (tmp + replace): a reader never sees a half-written report the way a torn
    line would sit in the log."""
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.isfile(log_path):
        return False
    steps, damage = Log(log_path).read(strict=False)
    if damage and err is not None:
        print(f"log damage: {damage}", file=err)
    if not any(step.get("kind") == "ended" for step in steps):
        return False
    body = f"> rendered from log.jsonl · {len(steps)} steps · the log is the record\n\n"
    body += render(steps)
    tmp = os.path.join(run_dir, "report.md.tmp")
    final = os.path.join(run_dir, "report.md")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(tmp, final)
    if err is not None:
        print(f"wrote {final}", file=err)
    return True
