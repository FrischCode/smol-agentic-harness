"""sah.live — the run's stream as it arrives, on disk.

The record holds the whole step, and the console holds the moment; this is the middle thing a run
writes when it is asked to: the fragments as they arrive, one line each, to `live.jsonl` in the
run folder. A trace of arrival, not a second record — the lines are outside the record's closed
vocabulary, a reader of `log.jsonl` never sees them, and the record's steps settle every fact a
fragment carries (the whole turn, the whole tool output, the whole thought). Off by default: a
run that does not ask for it leaves no second file, and the record's claim on the run stays whole.

The reader it serves is remote — a UI that tails the folder — so a fragment is live the moment it
is on disk: the handle ends with the line, the same open-per-write discipline the log keeps (a
kill mid-line damages exactly one line of a file nobody parses by the record's rules, and a
reader's delete blocks nothing, because the trace holds no handle between fragments).
"""

from __future__ import annotations

import json

from . import model, record


class Live:
    """One run's fragments, appended to one file as the event bus hands them out.

    The bus carries `TextDelta` for prose and for streaming tool output alike, `ReasoningDelta`
    for the reasoning the turn thinks, and the turn's whole steps. This consumer keeps the
    fragments and passes the steps: a prose or tool fragment stands as the `delta` line its kind
    is owed — a fragment with no `call_id` is the model's prose, one with a `call_id` is the
    output of the call it names, the same routing the display uses, so a live file and a console
    agree on where each fragment belongs — and a reasoning fragment stands as the `reasoning`
    line its kind is owed, with no call beside it, because the reasoning a turn thinks is not
    the output of a call.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def on_event(self, event) -> None:
        """One bus event. Prose, tool, and reasoning fragments are written as they arrive, each
        as the line its kind is owed; the whole steps pass over — the record settles them — and
        a fragment that is empty has nothing to arrive."""
        if isinstance(event, model.TextDelta):
            if not event.text:
                return
            line = {"kind": "delta", "ts": record.now(), "text": event.text}
            if event.call_id:
                line["call"] = event.call_id
        elif isinstance(event, model.ReasoningDelta):
            # The turn's own thought: its line says the kind, the moment it arrived, and the word —
            # and no call, because a line that carries a call is a line the output of the call it
            # names stands in, and the reasoning a turn thinks stands before any call it makes.
            if not event.text:
                return
            line = {"kind": "reasoning", "ts": record.now(), "text": event.text}
        else:
            return
        with open(self.path, "a", encoding="utf-8") as handle:
            # The open stands the handle down with the write: created at the first fragment, so a
            # run that streams nothing leaves no file, and a reader that tails an absent file and
            # a present one reads the same. A disk that refuses the open raises, and it is the
            # bus that isolates a subscriber that raises — the trace never stops the run it is
            # tracing.
            handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
