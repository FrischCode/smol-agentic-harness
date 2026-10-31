"""sah.tools.clock — the current time, on demand.

The date a run started is in its instructions and in its first log line, and both are frozen there. A
run that works for hours, or one resumed the next day, carries a prompt that states a date it may no
longer be — and a model has no way to tell the two apart from inside. The alternative, putting the live
time in the system prompt, costs every run its prefix cache (a prompt that changes every turn is a
prompt nothing can cache) and spends tokens on a fact most turns never consult. A tool asks for the
time at the moment time matters.

The answer is three lines: `utc`, `local` and `epoch`.

- **Both zones, and the offset spelled out.** An offset a reader has to infer is an offset a reader
  guesses. `local` carries it as written by `isoformat`, so `+01:00` is in the text rather than in a
  comment about the machine.
- **`epoch` is the one time given as a number.** The log's own stamps are ISO-8601 and stay that way,
  but a second count since 1970 is what other programs speak — another tool's log, a file's
  modification time, an API's field — so reporting it here makes a duration one `calc` subtraction
  instead of a date parse. Whole seconds, to match the two lines above it.
- **Each line carries its own zone's weekday, as a name.** The numeric form is a fact about one
  library's enum (is Sunday 0 or 6?) and a machine's locale decides what `%A` spells, so the names are
  spelled below and the answer depends on neither. A weekday is not a fact about an instant, only about
  a zone and an instant: around 23:00 local the two lines answer "what day is it" differently, and both
  answers are correct for the zone they sit in. The local one is what a run scheduling anything wants.

Second precision, deliberately. A model does not schedule anything to the microsecond, and a clock that
reported `123456` would have the run believe it knows something the operating system's clock does not
hold still.
"""
from __future__ import annotations

from datetime import datetime, timezone

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
"""Spelled here rather than asked of the C library: `%A` answers in the machine's locale, and a run
whose weekday comes back in a language nobody chose is a run that has to guess what it was told."""


def _now() -> datetime:
    """The instant to report, with the machine's offset attached.

    A function rather than a call inside the tool, so the instant is a single frozen seam — the one
    place to hold time steady. The tool has no parameter a model could use to lie about time.
    """
    return datetime.now().astimezone()


def clock() -> str:
    """The date in your instructions was true when this run began and does not move; ask here when the
    answer has to be *now*. The epoch line is the same instant as a number, so a duration against a
    second count from anywhere else is one arithmetic call.
    """
    now = _now()
    utc = now.astimezone(timezone.utc)
    return (f"utc:   {utc.isoformat(timespec='seconds').replace('+00:00', 'Z')} "
            f"({_WEEKDAYS[utc.weekday()]})\n"
            f"local: {now.isoformat(timespec='seconds')} ({_WEEKDAYS[now.weekday()]})\n"
            f"epoch: {int(now.timestamp())}")


clock.description = ("Read the current date and time: UTC and local (each with its weekday) and Unix "
                     "epoch seconds.")


# The clock is read, not written, and it is nobody's file: `none`, so the policy has no address to
# judge and the tool needs no flag. A run that cannot ask what time it is will still answer from the
# date in its prompt, which is the one thing this exists to correct.
clock.access, clock.address = "none", None


class Clock:
    """The clock, always offered: touching no place, it is neither a reach nor a widening.

    Cheap enough to advertise unconditionally — one line of schema — and the run that needs it is the
    one resuming a days-old log, which is exactly the run that cannot tell from its own prompt that the
    date in it has gone stale.
    """

    tools = (clock,)
