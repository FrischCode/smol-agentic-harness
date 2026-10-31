"""sah.guard — the repetition guard.

Failure shape it exists for: the run does not converge because it keeps doing the *identical* thing
and expecting a different answer. Catching that must not require understanding the task, so the guard
reasons only about **declared properties of the act stream** — the tool name, a digest of its
arguments, and whether the call succeeded — and never about the content or intent of anything. A guard
that began inferring what the run "means" would be a separate contract with its own correctness
problem; this one is a counter over an equality key, and that limitation is the whole reason it can be
trusted to run unattended.

It escalates in the two small steps that are worth acting on, and no more:

- after `threshold` identical consecutive acts → a **nudge**: one steering message to the model and a
  `notice` for the record, so the human sees the run was watching itself. The streak then resets, so a
  single long streak is not nudged on every turn — it takes another full `threshold` to escalate again.
- once `steer_limit` nudges are spent and it is *still* repeating → **stop**: the loop ends as
  `stalled`. Two nudges then stop is the whole policy; a third tier of escalating warnings is theatre.

It only *decides*. It never writes: the core performs every write (the nudge is a `say`, the record is
a `notice`, the stop is the loop breaking). Keeping detection pure is what lets it be tested without a
model and reasoned about without the loop.
"""
from __future__ import annotations

import hashlib
import json


def act_key(name: str, arguments: dict, ok: bool) -> tuple:
    """The identity of one act, from declared properties only.

    Argument *order* must not matter (the same call written two ways is the same call), so the
    arguments are canonicalised before hashing. They are hashed rather than stored so a megabyte of
    arguments does not sit in the guard's memory — the digest is for equality, never for display."""
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return (name, digest, bool(ok))


class RepetitionGuard:
    """Counts consecutive identical acts and says when to nudge or stop. Pure: it holds a few
    integers and a key, and writes nothing."""

    def __init__(self, *, threshold: int = 3, steer_limit: int = 2):
        if threshold < 2:
            raise ValueError("a repetition threshold below 2 would nudge the first ever call")
        if steer_limit < 1:
            raise ValueError("steer_limit below 1 means 'never nudge', which is just no guard")
        self.threshold = threshold
        self.steer_limit = steer_limit
        self._last = None
        self._streak = 0
        self._steers = 0
        self._name = ""

    def reset(self) -> None:
        """Forget the current streak and the nudges spent on it. The core calls this at the start of
        each objective: acts counted in one objective are not evidence that the next one is going
        the same way, and nudges already spent there must not buy the next one a stop instead of a
        warning. Detection stays pure — this clears memory, it writes nothing."""
        self._last = None
        self._streak = 0
        self._steers = 0
        self._name = ""

    def observe(self, name: str, arguments: dict, ok: bool) -> None:
        """Record one act. Any change of the identity key resets the streak, so a run that adapts —
        even slightly — is never nudged. A repeat is only ever *identical* repeats."""
        key = act_key(name, arguments, ok)
        if key == self._last:
            self._streak += 1
        else:
            self._last = key
            self._streak = 1
            self._name = name

    def escalation(self) -> str | None:
        """`None`, `"nudge"`, or `"stop"` — derived purely from the streak and how many nudges are
        spent. Idempotent between acts: it does not consume state, so calling it twice is safe; the
        core calls `spend_nudge()` when it actually acts on a nudge."""
        if self._streak >= self.threshold:
            return "stop" if self._steers >= self.steer_limit else "nudge"
        return None

    def spend_nudge(self) -> None:
        """Called by the core after it delivers a nudge: the streak resets (so the same long streak
        takes another `threshold` to escalate again) and one nudge is spent (so exhausting
        `steer_limit` is what eventually turns a crossing into a stop)."""
        self._streak = 0
        self._steers += 1

    @property
    def repeating(self) -> str:
        """The tool name of the current streak, for the text of the nudge."""
        return self._name

    @property
    def nudges(self) -> int:
        """The nudges spent so far on the current streak."""
        return self._steers
