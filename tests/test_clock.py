"""The clock's promises: one instant told three ways, and no way for a caller to choose the instant.

Every test freezes `sah.tools.clock._now`. A test that reads the real clock cannot assert what the lines say,
goes red on the last day of a month, and proves nothing about a zone it happens not to be in — the seam
exists so the format is testable at all.

What the format owes, in order: the two zones must not be reconcilable only by arithmetic a reader has
to get right; the weekday belongs to a zone rather than to an instant (they disagree around local
midnight, and both answers are correct); and the epoch line must be the *same* instant, since a
duration is the reason a number is offered next to two strings.
"""
from datetime import datetime, timedelta, timezone
import re

import pytest

from sah.tools import clock
from sah.confine import Confine
from sah.record import Refused
from sah.contract import arguments_for, declaration, spec_for

# 23:30 on a Monday *local*, which is 06:30 Tuesday in UTC: the one shape of instant where the two
# lines must disagree about the day, because a weekday is a fact about a zone and a moment.
MONDAY_NIGHT = datetime(2026, 10, 26, 23, 30, 0, 123456, tzinfo=timezone(timedelta(hours=-7)))


def _frozen(monkeypatch, when):
    monkeypatch.setattr(clock, "_now", lambda: when)


def test_one_instant_is_told_three_ways(monkeypatch):
    _frozen(monkeypatch, datetime(2026, 10, 27, 9, 14, 7, tzinfo=timezone(timedelta(hours=1))))
    assert clock.clock() == (
        "utc:   2026-10-27T08:14:07Z (Tuesday)\n"
        "local: 2026-10-27T09:14:07+01:00 (Tuesday)\n"
        "epoch: 1793088847")


def test_each_line_carries_the_weekday_of_its_own_zone(monkeypatch):
    """At 23:30 in UTC-7 the UTC line says Tuesday and the local line says Monday, and neither is wrong.

    A single weekday would have to belong to one of the two zones while sitting beside the other, and
    the run that asks "is tonight the weekend" is asking about the wall clock in front of it."""
    _frozen(monkeypatch, MONDAY_NIGHT)
    utc, local, _epoch = clock.clock().splitlines()
    assert utc == "utc:   2026-10-27T06:30:00Z (Tuesday)", utc
    assert local == "local: 2026-10-26T23:30:00-07:00 (Monday)", local


def test_the_local_offset_is_in_the_text(monkeypatch):
    """An offset the reader has to infer is an offset the reader guesses, and a log line that reads
    `09:14` with no offset cannot be compared with another machine's `09:14` at all."""
    for hours, spelled in ((0, "+00:00"), (5, "+05:00"), (-7, "-07:00"), (9, "+09:00")):
        _frozen(monkeypatch, datetime(2026, 10, 27, 9, 14, 7, tzinfo=timezone(timedelta(hours=hours))))
        assert spelled in clock.clock()


def test_the_weekday_names_are_the_seven_in_order(monkeypatch):
    """Spelled in this module rather than asked of `%A`, which answers in whatever language the machine
    was configured for. The names here are the answer on every box; the test pins the mapping, so an
    off-by-one in `weekday()` is caught rather than becoming a run that believes it is Thursday."""
    monday = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
    for day, name in enumerate(("Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday")):
        _frozen(monkeypatch, monday + timedelta(days=day))
        assert clock.clock().splitlines()[0].endswith(f"({name})")


def test_the_epoch_line_is_the_same_instant_as_the_two_strings(monkeypatch):
    """The reason a number sits beside them: a duration against a second count from anywhere else is one
    arithmetic call rather than a date parse. It is the same instant or it is a lie about a timestamp."""
    _frozen(monkeypatch, MONDAY_NIGHT)
    lines = dict(line.split(":", 1) for line in clock.clock().splitlines())
    epoch = int(lines["epoch"])
    assert epoch == int(MONDAY_NIGHT.timestamp())
    stamp = lines["utc"].strip().split()[0].replace("Z", "+00:00")
    assert int(datetime.fromisoformat(stamp).timestamp()) == epoch


def test_the_answer_is_second_precise(monkeypatch):
    """Microseconds are dropped, not rounded away by accident: the operating system's clock is not held
    still to that precision, and a run handed `123456` would believe it knows more than it does."""
    _frozen(monkeypatch, MONDAY_NIGHT)
    answer = clock.clock()
    assert "." not in answer, answer                      # no fractional seconds anywhere
    stamps = re.findall(r"\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", answer)
    assert len(stamps) == 2, answer                       # both clocks, both closed by a zone


def test_no_argument_a_model_sends_can_choose_the_instant():
    """The tool takes nothing. A `when` parameter would let a run write a timestamp it made up into the
    log beside one the machine supplied, and the record would stop being evidence about when anything
    happened."""
    assert spec_for(clock.clock).parameters["properties"] == {}
    with pytest.raises(Refused, match="does not accept"):
        arguments_for(clock.clock, {"fmt": "%Y"})
    with pytest.raises(Refused, match="does not accept"):
        arguments_for(clock.clock, {"when": "2020-01-01"})


def test_the_seam_hands_back_an_aware_instant():
    """Every other test here replaces `_now`, so none of them can prove what `_now` does. A naive
    `datetime.now()` passes the whole format battery above and prints a local time whose offset nobody
    knows — the one fact the local line exists to carry.

    So this asserts a *property* of the real clock rather than a value from it: awareness holds at every
    instant, so it cannot go red on a boundary date or in a boundary zone. It is the only place the
    battery reads the machine's clock, and it is what the seam owes that a frozen test cannot see.
    """
    when = clock._now()
    assert when.tzinfo is not None and when.utcoffset() is not None, \
        "a naive instant has no offset to print, so the local line would be an unlabelled time"


def test_the_clock_is_read_through_one_seam(monkeypatch):
    """The battery freezes `_now`, so the tool must ask the clock and nothing else — a second
    `datetime.now()` anywhere in the tool would answer from the real machine while the test believes it
    froze time, and would go red on a boundary date instead of here."""
    _frozen(monkeypatch, MONDAY_NIGHT)
    calls = []

    def spy():
        calls.append(1)
        return MONDAY_NIGHT
    monkeypatch.setattr(clock, "_now", spy)
    clock.clock()
    assert calls == [1]


def test_the_clock_touches_no_place_and_is_offered_to_a_run_granted_nothing():
    """Reading a clock opens nothing, so there is no address for the veto point and no flag to wait for.
    The run that most needs the time is the one resumed over a stale prompt, which is also the run least
    likely to hold a grant."""
    assert declaration(clock.clock) == ("none", None)
    assert Confine()("clock", {}, access="none", address=None).allowed
