"""sah.render — the human's live view: a projection of the run's steps and streamed chunks.

There are three projections of one truth: JSONL is the record, Markdown (`sah.export`) is the
portable report a human or a model reads back, and this is the console — **purely for a person
watching a run happen, right now**. It is deliberately the throwaway one: nothing a consumer decides
here is durable, and if the renderer is absent or buggy the JSONL still holds everything and the
verdict floor still reaches stderr. That is what lets this file be decorative in ways the record
must never be.

It is a *subscriber*, not a participant, and reaches the loop through two ordinary doors:

- **`on(step)`** — the authoritative steps, as written (this is a normal extension door; the loop
  wires it). Structure comes from here: open a tool box on `call`, close it on `result`, render a
  compaction `summary`, a `notice`.
- **`on_event(event)`** — the live chunk stream (prose, reasoning, and a streaming tool’s output),
  wired by the client through the event bus. This is what makes progress visible *as it happens* —
  the reason streaming is first-class: a build or download that runs for minutes must not go silent.

There is a third channel, and it is not a door into the loop at all: **`transit(text)`**, the transient
line. A watcher (or a long tool’s body) speaks it to the operator who is watching now; the console
renders it and nothing else does — it is not a step (nothing is stamped, sequenced or exported), not
the operator’s stream (it lands on this face, never on stderr — a refusal on stdout is the bug, and a
line this console draws is not a refusal), and it reaches no model. The fact a line points at is the
step it points at, and the record holds that, so a line is a word, not a fact — and where the console
is not showing the run’s own mechanics, or is not there at all, a line says nothing on purpose.

Display semantics come from the step schema, not a kind-by-kind guess: a kind this renderer does not
recognise is still shown by its declared fields, so adding a record type can never make output
vanish. Visual treatment (which kinds appear,
boxes or lines, colour) is chosen by a `View`, not scattered `if mode ==` branches.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime

from sah import model

# The one width the console still measures anything against: a boot-box line is clipped to it, so a
# machine path long enough to wrap cannot break the block into an unreadable shape. The rules themselves
# are not drawn at any width -- an open rule is its label, a close rule is its footer -- because nothing
# bounds content by the frame (a model's paragraph runs past any width this file picks), so a rule padded
# to it drew a right edge the body ignores.
BOX_WIDTH = 68
_META = ("id", "ts", "issuer", "kind")
CLEAN_OUTCOME = ("", "ok", "allowed", "accepted")

_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"
_RED, _GREEN, _YELLOW, _CYAN, _MAGENTA, _WHITE = (
    "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[97m")


@dataclass(frozen=True)
class View:
    """What a display mode shows, as data. Adding a mode is a row in `VIEWS`, not new branches."""

    stream: bool          # paint the live chunk stream (prose/reasoning/tool output) as it arrives
    boxes: bool           # framed, role-coloured turn boxes; otherwise compact one-line output
    prose: bool           # show the assistant's prose
    reasoning: bool       # show reasoning, dim
    tool_output: bool     # show a tool's result body, not just that it was called
    verdicts: bool        # surface refusals inline (the verdict floor guarantees them regardless)
    meta: bool            # show the run's own mechanics: a call, a notice, a transient line, an end line
    labels: bool          # say who is speaking (`you` / `agent`) on the compact one-line modes


# `meta` and `labels` are what separates a view a program reads from one a person reads. `text` is the
# pipe `sah-call` builds, and `off` is what a child runs with while its stdout goes to a file: in both,
# stdout is data, so the run's own mechanics do not belong in it. Verdicts are not lost by that — the
# verdict floor and the questions surface write to **stderr**, which is the operator's stream. A refusal
# on stdout would be the bug; a refusal on stderr is the design. The transient line is the third of
# these: it is the console's face, so it renders where the mechanics render (the `meta` views) and
# says nothing where they do not — a line for an operator who is watching now, and the record holds
# the fact where there is no one watching.

VIEWS = {
    "normal": View(stream=True, boxes=True, prose=True, reasoning=True, tool_output=True,
                   verdicts=True, meta=True, labels=True),
    "quiet": View(stream=False, boxes=False, prose=True, reasoning=False, tool_output=False,
                  verdicts=True, meta=True, labels=True),
    "text": View(stream=False, boxes=False, prose=True, reasoning=False, tool_output=False,
                 verdicts=False, meta=False, labels=False),
    "off": View(stream=False, boxes=False, prose=False, reasoning=False, tool_output=False,
                verdicts=True, meta=False, labels=False),
}


def _wants_color(stream) -> bool:
    """Colour only for a real terminal, and never when the operator asked not to. A piped or
    captured stream gets plain text, so logs and child output stay free of escape codes.

    `TERM=dumb` is that request, made by pagers, emulators and screen readers: the check reads the
    environment's *value* (`os.environ.get("TERM") != "dumb"`), so a dumb terminal gets plain text and the
    environment is actually consulted.
    """
    return (hasattr(stream, "isatty") and stream.isatty()
            and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb")


def _hhmmss(step=None) -> str:
    """The clock the operator is looking at, not the clock the record keeps.

    `ts` stores UTC — the record's rule, so a log moved between machines does not drift — but a person
    reading a run's timeline compares it with the clock on their wall, so the stamp is converted to the
    local timezone (via `astimezone()`) rather than sliced raw. A stamp this cannot read still falls back:
    a projection should not
    break over a cosmetic line.
    """
    ts = (step or {}).get("ts", "")
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return datetime.now().strftime("%H:%M:%S")


class Terminal:
    """The console projection. Duck-typed as an extension (`on`); the client also wires `on_event`."""

    def __init__(self, out=None, err=None, view: str = "normal", *, color: bool | None = None):
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.view = VIEWS[view] if isinstance(view, str) else view
        self._color = _wants_color(self.out) if color is None else color
        self._assistant_open = False         # a prose box is open awaiting its message step
        self._tool_open = False              # a tool box is open awaiting its result step
        self._tool_streamed = False          # this tool's output already painted live from chunks
        self._line_open = False              # is the cursor mid-line (raw prose wrote no newline)?
        self._reason_open = False            # reasoning streamed inline; a break is owed before prose

    # --- boot ---------------------------------------------------------------

    def header(self, items, *, title: str = "S   ╱─╲   H") -> None:
        """The boot box: the SAH wordmark as the title (the A drawn as ``╱─╲``), one fact per line --
        endpoint, the live bounds, the run id and its control-file path, in one dim block, so a run's
        identity is stated in one place rather than scattered across separate prints. Box view only: a
        one-shot, a quiet/text
        view, or a redirected child (a run with no witness) prints nothing. Dim throughout -- it is a
        label on the run, not content, so it must not compete with the turn boxes below it."""
        if not self.view.boxes:
            return
        # The block is dim; only the wordmark is lit (bright white) -- a lit label on a dim block, so it
        # reads as the run's name rather than as content. The title carries no fill and the block carries
        # no bottom rule: it closes by the first turn box below it, the same way a footerless turn does.
        self.out.write("\n" + self._c("╭─ ", _DIM) + self._c(title, _BOLD + _WHITE) + "\n")
        for item in items:
            self.out.write(self._c(item[: BOX_WIDTH - 1], _DIM) + "\n")
        self._line_open = False

    # --- primitives ---------------------------------------------------------

    def _c(self, text: str, code: str) -> str:
        return text if not self._color else f"{code}{text}{_RESET}"

    def _open_box(self, label: str, step=None, color=_GREEN) -> None:
        """The open rule is a label and nothing else: `╭─ 22:37:19 · YOU`.

        Padded out to `BOX_WIDTH` with dashes, the rule would claim the box is that wide — but nothing in
        this file makes the claim true: no content is bounded by the frame, a model's paragraph runs
        straight past it. So the rule is a label, with no fill drawing a right edge the body ignores.

        The bottom rule keeps its corner for a different reason: it is the footer's attachment to the body
        above it, the one glyph that says these figures describe that turn and not the next one.

        A label longer than the frame is printed in full, never clipped. With no fill there is no visible
        reference width for an over-long label to overflow, so the guarantee is a stated contract rather
        than something the frame shows.
        """
        self.out.write("\n" + self._c(f"╭─ {_hhmmss(step)} · {label}", color) + "\n")
        self._line_open = False

    def _close_box(self, trailer: str = "", color=_GREEN) -> None:
        """The close rule is the footer and nothing else: `╰─ 3647/32768 CTX · ... · 7.1s`.

        With nothing to say it says nothing -- no rule is written at all: a bare corner under a body is an
        ornament claiming an edge the text above it never respected.

        Only terminate the current line if something left it open -- raw streamed prose writes no newline
        and must be ended before the footer -- but newline-terminated content is already ended, and an
        unconditional newline here would open a gap between the body and its own footer (a box whose text
        streamed would look different from one whose text arrived whole).
        """
        if self._line_open:
            self.out.write("\n")
        if trailer:
            self.out.write(self._c("╰─ ", color) + self._c(trailer, _DIM) + "\n")
        self._line_open = False
        self._reason_open = False              # a closed box owes no break to the next one

    def _body(self, text: str, color: str = "") -> None:
        self.out.write(self._c(text.rstrip("\n"), color) if color else text.rstrip("\n"))
        self.out.write("\n")
        self._line_open = False

    def _standalone(self, text: str, color: str = "") -> None:
        """A top-level block that is not a box (a verdict, a notice, an unknown kind).

        In box mode it gets the same blank line above it that a box header carries, so a verdict is
        spaced like a turn rather than butting against the box before it -- without this, boxes are
        separated by a gap but the verdict is not, which reads as a bug. In the compact modes every
        line is single-spaced on purpose (density is the point), so it gets none."""
        if self.view.boxes:
            self.out.write("\n")
        self._body(text, color)

    # --- the transient channel ----------------------------------------------

    def transit(self, text: str) -> None:
        """The transient line: a word a watcher (or a long tool’s body) speaks to the operator who is
        watching now. It is the run’s own mechanics, so it renders where the mechanics render and says
        nothing where they do not (the views that keep stdout as data). Dim with its own mark: it is a
        caption, and a caption must not compete with the turn boxes.

        The word carries its speaker: the console’s face has no subject the core could fill (the core
        owns the channel and the mark, not the words), so a line that does not name its speaker reads
        as the run’s voice. The extension names itself in the words, the way a finding carries its own
        naming — the core stays dumb about the words and forgiving about the line (below).

        Forgiving by design — a cosmetic channel that raised would fail a tool call over a caption:
        a line that is blank says nothing, and a line with newlines in it says several."""
        if not self.view.meta:
            return
        for line in str(text).splitlines():
            if line.strip():
                self._standalone(self._c(f"… {line}", _DIM))

    # --- streaming chunks (the live channel) --------------------------------

    def on_event(self, event) -> None:
        """The live chunk stream (prose, reasoning, and a streaming tool's output): paint a delta as it
        arrives when the view shows it, tracking the cursor position so a line-break is owed exactly when
        one is owed."""
        if not self.view.stream:
            return
        if isinstance(event, model.ReasoningDelta):
            if self.view.reasoning:
                self._ensure_assistant()
                # Reasoning arrives as token-level deltas, not lines, so -- like prose -- it streams
                # inline. Routing it through _body terminated every delta with a newline and laid the
                # thought out one word per line.
                self.out.write(self._c(event.text, _DIM))
                self.out.flush()
                # A delta that ends in a newline leaves the cursor at column 0, not mid-line. This is
                # tracked honestly so the prose branch can decide whether it owes a line-break before
                # the blank separator: qwen3_coder closes its reasoning block with a trailing "\n"
                # delta, a thought that just stops leaves no newline, and both must read the same.
                self._line_open = not event.text.endswith("\n")
                self._reason_open = True
        elif isinstance(event, model.TextDelta):
            if event.call_id is not None:                       # a tool's live output, if shown at all
                if self.view.tool_output and self._tool_open:
                    self._body(event.text, _DIM)
                    self._tool_streamed = True
            elif self.view.prose:
                self._ensure_assistant()
                # A thinking turn separates its thought from its answer with exactly one blank line,
                # owed once -- on the first prose chunk -- whenever reasoning was shown this message.
                # End the thought's line first if it did not already end in a newline, then write the
                # blank, so the separation is identical whether qwen3_coder closed the reasoning block
                # with its trailing "\n" delta or the thought simply stopped mid-line. Cleared after,
                # so later prose chunks flow normally and no further blanks appear.
                if self._reason_open:
                    if self._line_open:
                        self.out.write("\n")
                    self.out.write("\n")
                    self._reason_open = False
                self.out.write(event.text)                      # prose streams raw, no decoration
                self.out.flush()
                self._line_open = not event.text.endswith("\n")  # honest about the cursor position

    def _ensure_assistant(self) -> None:
        if not self._assistant_open:
            self._open_box("ASSISTANT", color=_GREEN)
            self._assistant_open = True

    # --- authoritative steps (the record channel) ---------------------------

    def on(self, step: dict) -> None:
        """The authoritative-step channel: route a step to its `_kind` handler, or the generic fallback
        (which renders any kind by its declared fields) when there is none."""
        handler = getattr(self, f"_{step['kind']}", None)
        if handler is not None:
            handler(step)                        # a bound method already carries self
        else:
            _generic(self, step)                 # the fallback is module-level, so pass self

    def _message(self, step: dict) -> None:
        if step.get("role") != "assistant":
            if step.get("role") == "user":
                # The face follows the author, not the role: a word the core puts in the user's seat
                # (a nudge, a wrap-up, a future word) is the core's voice, and the view says so with
                # its own label and frame. The wire carries it as a user message on purpose — that is
                # where the model's attention is — and the record stamps the issuer on purpose — so a
                # view that showed the word in the operator's box would tell them they typed it. The
                # rule keys on the issuer, never on what the word is about: the channel owns the face,
                # and any future core word inherits it. Operator input — the objective typed (issuer
                # `user`) and a steer from a second shell (issuer `queue`) — keeps the operator's face.
                core = step.get("issuer") == "core"
                label, color = ("CORE", _MAGENTA) if core else ("YOU", _CYAN)
                if self.view.boxes:
                    self._open_box(label, step, color)
                    self._body(step.get("text", ""))
                    self._close_box(color=color)
                elif self.view.prose and self.view.labels:
                    # An echoed objective is the caller's own input handed back to it. In the piped views
                    # the caller already has it — it typed it — so printing it there is not context, it is
                    # the pipe lying about where the answer starts.
                    self._body(self._c("core " if core else "you  ", color) + step.get("text", ""))
            return
        # normal mode already streamed the prose; here we only close its box. Other modes render
        # from the record, because they do not stream.
        if self._assistant_open:
            self._close_box(trailer_shape(step))
            self._assistant_open = False
            return
        if not self.view.prose and not self.view.reasoning:
            return
        text = (step.get("text") or "").strip()
        reasoning = step.get("reasoning")
        if not text and not reasoning:
            return
        if self.view.boxes:
            self._open_box("ASSISTANT", step, _GREEN)
            if reasoning and self.view.reasoning:
                self._body(self._c(reasoning.rstrip(), _DIM))
            if text:
                self._body(text)
            self._close_box(trailer_shape(step))
        else:
            if reasoning and self.view.reasoning:
                self._body(self._c("reason ", _DIM) + reasoning.rstrip())
            if text and self.view.prose:
                # The label is for a reader who needs to tell the two voices apart. A pipe does not,
                # and a prefix it has to strip is a bug the caller pays for once per line.
                self._body(self._c("agent ", _GREEN) + text if self.view.labels else text)

    def _call(self, step: dict) -> None:
        args = _compact(step.get("args", {}))
        self._tool_streamed = False                             # a fresh box: has not streamed yet
        if self.view.boxes:
            self._open_box(f"TOOL · {step.get('name', '?')}", step, _YELLOW)
            if args:
                self._body(self._c(args, _DIM))
            self._tool_open = True
        elif self.view.meta:
            self.out.write("  " + self._c(f"→ {step.get('name', '?')}", _YELLOW)
                           + (self._c(f" {args}", _DIM) if args else "") + "\n")

    def _result(self, step: dict) -> None:
        ok = step.get("ok")
        failed = not ok
        if ok:
            trailer = "ok"
        else:
            first = step.get("content", "").splitlines()
            trailer = f"failed — {first[0][:60]}" if first else "failed"
        extra = _bytes_note(step)
        if extra:
            trailer += extra
        if self.view.boxes:
            # A streamed tool already showed its body live; reprinting it here would double it. A
            # plain (non-streamed) tool shows its body now, because nothing streamed it earlier.
            if self.view.tool_output and step.get("content") and not self._tool_streamed:
                self._body(step["content"], _RED if failed else _DIM)
            self._close_box(trailer, _RED if failed else _YELLOW)
            self._tool_open = False
        elif self.view.tool_output and step.get("content"):
            self._body(self._c(("  ✗ " if failed else "  ✓ ") + step["content"].splitlines()[0][:80],
                               _RED if failed else _DIM))

    def _summary(self, step: dict) -> None:
        counts = f"{step.get('kept', '?')} kept · {step.get('dropped', '?')} dropped"
        verdict = step.get("outcome", "")
        # A rung that worked states its arithmetic and nothing else: `ok` adds no fact that the absence of
        # a failure does not already carry, and `OK · KEPT 12` read as two verdicts because the count and
        # the outcome shared the word. A failed one leads with the word and its sentence, the way a footer
        # leads with `truncated`.
        if verdict in CLEAN_OUTCOME:
            head = f"summary ({counts})"
        else:
            reason = f": {step['reason']}" if step.get("reason") else ""
            head = f"summary ({verdict}{reason} · {counts})"
        if self.view.boxes:
            self._open_box(head.upper(), step, _MAGENTA)
            self._body(step.get("text", ""), _DIM)
            self._close_box(color=_MAGENTA)
        elif self.view.meta:
            self._body(self._c(head, _MAGENTA))

    def _notice(self, step: dict) -> None:
        if not self.view.meta:
            return
        if self.view.verdicts or step.get("subject") != "nudge":
            self._standalone(self._c(f"• {step.get('subject', 'notice')}: {step.get('text', '')}", _MAGENTA))

    # `request`, `decision` and `command` have no handler on purpose. They are the three kinds a person
    # most needs and a pipe must never receive, and both halves are already true: the generic fallback
    # renders any kind by its declared fields (so none is dropped, which is that fallback's whole job),
    # and the fallback now honours `meta`. Handlers here would print what the fallback prints, in a shape
    # no test could tell apart — a line that cannot fail is not a guarantee, so the fallback stays the
    # one path for them.

    def _operation(self, step: dict) -> None:
        """Deliberately silent: the result box already carries the outcome, and the verdict floor
        guarantees a refusal reaches stderr. Printing every permitted call is noise that trains the
        eye to scroll past the one thing it should not. (A kind with no handler at all still shows
        via the generic fallback — this is an intentional suppression, not an omission.)"""

    def _started(self, step: dict) -> None:
        """Silent: the operator's banner already states the run's bounds, grants and model."""

    def _ended(self, step: dict) -> None:
        if self.view.meta and self.view.verdicts:
            color = _GREEN if step.get("reason") == "complete" else _RED
            self._standalone(self._c(f"◆ {step.get('reason')} (exit {step.get('exit_code')})", color))

def trailer_shape(step: dict) -> str:
    """The status footer of an assistant's turn, as one shape that is never cut to fit.

    The segments, in the shape a person reads: `3647/32768 CTX · ¤0 ›125 ↑881 ↓45/s TOK · 7.1s`. The
    window pair first, unmarked, because it is the number being watched — how full the model's window
    is decides whether the next step is possible, which a token count on its own does not say. Then
    one space-separated group of the token figures: `¤`, the part of that prompt the server served
    from its own cache — measured end to end through this console, 1149 of a turn's 1212 prompt tokens
    came from the server's cache and what had taken 2.6 s took 0.7 s, and nothing else on the line
    would have said why — and `›`, what came out: both counts, and neither gets an arrow, because
    the arrows are the rates' marks. Then the two rates the server measured for itself, `↑` prefill
    and `↓` decode, with `/s` on the last because it covers both, and `TOK` to name the group. The
    wall time last. A stated cache count of zero prints (`¤0`) and a server that never volunteered
    one prints no mark: "it cached nothing" and "it said nothing" are different facts, and the
    console is where a run that never hits its cache gets noticed.

    The arrows mark rates and nothing else, and `›` is the only mark a count gets: a count carrying an
    arrow is read as a speed by anyone who has seen this line, which is the difference between a footer
    and one that lies about its own units. A server that reports no rate gets no arrows and no `?`
    placeholder either — an unknown speed is not a speed of zero, and `?` asserts a countable thing
    was measured and lost. `¤` is U+00A4 and `›` is U+203A; both encode in cp1252, so the marks are not
    the reason a box would fail to print on a stream that cannot carry its frame.

    Nothing is dropped for width and no segment is cut to fit: the footer is one plain line, so an
    unusually long one wraps in the terminal like the prose above it does and stands as a long line
    in the report — either way it costs a line and loses no figure.
    """
    bits = []
    if step.get("tokens_in") is not None:
        # The ceiling may be absent — no `--context` was named, or the backend declared no window —
        # and then the count stands alone rather than carrying a `/0` that reads as an empty window.
        ctx = f"{step['tokens_in']}"
        if step.get("context_max"):
            ctx += f"/{step['context_max']}"
        bits.append(ctx + " CTX")
    # One group, in the order the server produced the figures: what was served from the cache, what
    # came out, and the two rates. Each rate prints only when the server reported one above zero, since
    # a server that measured nothing says so with a zero and an unknown speed is not a speed of zero.
    # Nothing here is divided out of `ms`: that spans the whole request, so a rate computed from it
    # would be labelled decode and be something else.
    group = []
    if step.get("tokens_cached") is not None:
        group.append(f"¤{step['tokens_cached']}")
    if step.get("tokens_out") is not None:
        group.append(f"›{step['tokens_out']}")
    if step.get("prefill_tps"):
        group.append(f"↑{step['prefill_tps']:.0f}")
    if step.get("tps"):
        group.append(f"↓{step['tps']:.0f}/s")
    if group:
        bits.append(" ".join(group) + " TOK")
    if step.get("ms") is not None:
        bits.append(f"{step['ms'] / 1000:.1f}s")
    outcome = step.get("outcome")
    if outcome and outcome not in CLEAN_OUTCOME:
        bits.insert(0, outcome)
    return " · ".join(bits)



def _compact(args: dict) -> str:
    import json
    s = json.dumps(args, sort_keys=True, ensure_ascii=False) if args else ""
    return s if len(s) <= 90 else s[:87] + "…"


def _bytes_note(step: dict) -> str:
    if step.get("truncated") and step.get("bytes"):
        return f" · {step['bytes']} bytes (spilled)" if step.get("spilled") else " · truncated"
    return ""


def _generic(self, step: dict) -> None:
    """A kind this renderer predates is shown by its declared fields, never dropped — a report that
    silently omits a record type is how refusals once vanished. Crude beats invisible.

    "Invisible" is the constraint that matters in the piped views, though: crude there is not a rough
    rendering of an answer, it is a line of the run's internals arriving in a program's stdin. The
    kind is still in the log, which is where a later reader looks for it.
    """
    if not getattr(self.view, "meta", True):
        return
    body = ", ".join(f"{k}={v!r}" for k, v in step.items() if k not in _META)
    self._standalone(self._c(f"· {step['kind']}: {body}", _DIM))


class RenderExtension:
    """Bundles a Terminal as an extension so its `on(step)` is wired through the one registration
    path. The client additionally calls `agent.subscribe_event(terminal.on_event)` for the live
    channel — streaming is the client's own wiring, not a contract a plugin is handed."""

    def __init__(self, terminal: Terminal):
        self.terminal = terminal
        self.tools = ()

    def on(self, step: dict) -> None:
        """The extension door: forward the step to the Terminal."""
        self.terminal.on(step)


class Status:
    """The console's transient line, bundled for an extension to hold.

    This is the door the client hands to a third-party extension at assembly (`sah.ext.assemble`):
    an extension that wants to say a word to the operator who is watching now holds a `Status` and
    calls `say`, naming its speaker in the word itself — the line’s face has no subject the core
    could fill, so a line that does not name its speaker reads as the run’s voice (`transit` states
    the rule). The line is rendered by the `Terminal` it wraps and held nowhere else — it is not a
    step, not the operator's stream, and it reaches no model (the module docstring states why each
    of those is the point, not an accident). Wrapping a `Terminal` rather than handing the extension
    one is the same reason `RenderExtension` wraps: the extension gets `say` and nothing else — no
    `out` handle to write raw, no `on` to fake, no box it does not own.

    Built with a `None` terminal it is a quiet no-op: a run with no witness (a redirected child, a
    pipe that keeps stdout as data) says nothing on purpose, because the fact a line points at is
    the step it points at, and the record holds that.
    """

    def __init__(self, terminal: Terminal | None = None):
        self.terminal = terminal

    def say(self, text: str) -> None:
        """Say a line to the watching operator. The terminal decides whether it is drawn at all
        (the view's `meta`); this only hands the words over. Blank says nothing; a newline says
        several. It never raises — a caption that refused would fail a tool call over decoration."""
        if self.terminal is None:
            return
        self.terminal.transit(text)
