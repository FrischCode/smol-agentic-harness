"""The compaction stages, the room they run against, and the trigger and split that feed them.

A scripted backend drives every test, so these assert the harness's own behaviour — what the
summary request carries on the wire, what a too-long answer does, which stage paid for the slot,
what the room arithmetic subtracts — and never a server's. The seam these exist for: the server's
output gate used to decide whether a summary fit, and a backend that ignored the gate or
misreported the stop silently broke acceptance. Acceptance is the harness's measurement now, and
every budget assertion is that inversion kept from returning."""

import pytest

from sah import compact, model, prompts
from sah.model import Caps, Message, TextDelta, ToolSpec, Usage


def scripted(*replies_and_errors):
    """A backend that answers each complete() with one scripted turn. A reply may be a text
    (then a Usage of 50 prompt / len(text) completion and a clean stop ride with it) or an
    exception/event list for the failed and cut-off shapes."""
    turns = []
    for item in replies_and_errors:
        if isinstance(item, str):
            turns.append([TextDelta(item), Usage(50, len(item)), model.Stopped("stop")])
        else:
            turns.append(item)
    return model.Fake(turns, caps=Caps(context_tokens=1000, model="t"))


def msgs(n, role="user"):
    return [Message(role, f"turn {i}") for i in range(n)]


# --- the bare call: nothing but the prompt and the content on the wire -------

def test_a_summary_call_carries_no_output_budget_no_tools_or_foreign_prompt():
    """A4: the summary call carries ONLY the summarization system prompt and the content. No
    max_tokens (the server is never given a gate), no tool schemas (a summariser asked to write
    notes should not be handed a hammer), and the compactor's own system prompt, not the run's."""
    backend = scripted("notes")
    compact.summarise(backend, source_text="the source", room=100)
    call = backend.calls[-1]
    assert call.max_tokens is None, "the server is never given a gate on the answer's length"
    assert call.tools == ()
    assert call.system == prompts.load("compact_system")
    assert len(call.messages) == 1 and call.messages[0].role == "user"
    assert "the source" in call.messages[0].content


def test_the_summary_call_runs_at_the_level_it_is_given():
    """A8 at the module seam: the level is left alone — whatever the caller names (the run's own
    by default, the operator's when the knob set one) is what every attempt carries. A mutant
    that substitutes its own word reddens here on any level that is not that word."""
    for level in ("xhigh", "low", "off", None):
        backend = scripted("notes")
        compact.summarise(backend, source_text="s", room=100, thinking=level)
        assert backend.calls[-1].thinking == level, \
            f"{level!r} was not left alone: the call asked for {backend.calls[-1].thinking!r}"


# --- acceptance: the harness's measurement, not the server's verdict ---------

def test_acceptance_prefers_the_servers_count_of_the_call():
    """A1: the estimate says the prose fits and the telemetry says the call cost more than the
    room (a thinking server counts its thinking, which the slot will not keep) — the telemetry
    wins, stage 1 is rejected, and stage 2 is asked. The over-measure is deliberate; the escape
    is the knob, not a smaller estimate."""
    backend = scripted("a" * 400, "b" * 40)
    backend.turns[0] = [TextDelta("a" * 400), Usage(50, 200), model.Stopped("stop")]  # 100 est, 200 server
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=100)
    assert stage == 2 and kind == "summary" and slot == "b" * 40
    assert note == "", "a too-long-but-clean attempt is not a failure; the stage says who paid"


def test_acceptance_falls_back_to_the_estimate_when_the_server_reports_nothing():
    """A server silent about usage is not a server with an empty answer: with no count to prefer,
    the standing estimate decides, and 400 characters is 100 tokens at the rate, so it fits a
    room of 100 at stage 1."""
    backend = model.Fake([[TextDelta("c" * 400), model.Stopped("stop")],
                          [TextDelta("d" * 160), model.Stopped("stop")]],
                         caps=Caps(context_tokens=1000, model="t"))
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=100)
    assert stage == 1 and slot == "c" * 400


def test_a_length_stop_discards_the_text_and_fails_the_attempt():
    """A reply the output limit cut off is not a partial success: the text is discarded and the
    attempt failed — a too-long summary is discarded and asked to shrink, never accepted partial,
    because a fragment of a thought standing in for the turns it replaced is a compaction that
    forgot and called itself clean."""
    backend = model.Fake([[TextDelta("the user aske"), model.Stopped("length")],
                          [TextDelta("the whole note"), Usage(50, 30), model.Stopped("stop")]],
                         caps=Caps(context_tokens=1000, model="t"))
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=100)
    assert stage == 2 and slot == "the whole note"
    assert "output limit" in note, f"the cut-off attempt's words were dropped: {note!r}"
    second = backend.calls[1].messages[0].content
    assert "about 100 tokens" in second, "the compactness ask names the room, not just asks for less"


# --- the stages: bounded, in order, and the slot always filled ---------------

def test_the_second_stage_asks_the_source_for_a_fitting_summary():
    """Stage 2 re-asks the same source with the compactness sentence (a prompt file that names
    the number) — not a new prompt and not a smaller budget: the wire difference between stage 1
    and stage 2 is exactly that sentence."""
    backend = scripted("a" * 400, "the fit")
    backend.turns[0] = [TextDelta("a" * 400), Usage(50, 200), model.Stopped("stop")]
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="SOURCE-MARKER", room=100)
    assert stage == 2
    first, second = (c.messages[0].content for c in backend.calls[:2])
    assert "SOURCE-MARKER" in first and "SOURCE-MARKER" in second
    assert "about 100 tokens" not in first and "about 100 tokens" in second


def test_the_third_stage_summarises_the_latest_summary_not_the_source():
    """A6: stage 3 re-summarises the latest summary (stage 2's output) to the room — a summary
    that did not fit is still mostly already-dropped detail, and asking the model to cut its own
    notes is the shrink that keeps what mattered."""
    backend = scripted("A" * 400, "B" * 400, "the fit")
    backend.turns[0] = [TextDelta("A" * 400), Usage(50, 200), model.Stopped("stop")]
    backend.turns[1] = [TextDelta("B" * 400), Usage(50, 200), model.Stopped("stop")]
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="SOURCE-MARKER", room=100)
    assert stage == 3 and slot == "the fit"
    third = backend.calls[2].messages[0].content
    assert "B" * 400 in third, "stage 3 asks the latest summary to fit, not the source"
    assert "SOURCE-MARKER" not in third


def test_the_cut_stage_truncates_the_last_text_and_marks_it():
    """A7: when nothing fits, the last text is cut to the room and the cut is noted — the model
    is told it now works on potentially missing information. The marked summary must fit the room
    (the note is budgeted for before the cut), and it cuts the last text, not the first: A and B
    are already gone with the attempts that made them."""
    backend = scripted("A" * 400, "B" * 400, "C" * 400)
    for turn in backend.turns:
        turn[1] = Usage(50, 500)
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=100)
    assert stage == 4 and kind == "cut"
    assert compact.estimate_text(slot) <= 100, "the marked summary does not fit the room"
    assert slot.endswith(prompts.load("summary_cut")), "the note of the cut rides with the cut"
    assert "C" in slot and "A" not in slot and "B" not in slot


def test_no_text_at_all_takes_the_failed_note():
    """Every attempt failed (the backend cannot summarise) and the slot still gets its text: the
    failed note, stating how much of the earlier history is missing. The record carries the
    attempts' words, one per attempt, in the reason beside the verdict."""
    backend = model.Fake([[model.Failed("the engine died")]] * 3,
                         caps=Caps(context_tokens=1000, model="t"))
    slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=100, dropped=14)
    assert stage == 4 and kind == "failed"
    assert "14 earlier turns" in slot
    # Two attempts, not three: stage 3 re-summarises the latest summary, and with no text at all
    # there is no latest to ask — the old tree's guard, kept for the idealised pseudocode's sake.
    assert note.count("the engine died") == 2, f"one complaint per attempt: {note!r}"


def test_a_nonpositive_room_makes_no_attempt():
    """A room that is not positive is a compaction with nothing to attempt: no call is spent
    (the smallest possible cost — a summary call its result could not use is the doom this
    avoids), the slot takes the failed note, and the record says why."""
    for room in (0, -5):
        backend = model.Fake([], caps=Caps(context_tokens=1000, model="t"))
        slot, cost, stage, kind, note = compact.summarise(backend, source_text="s", room=room,
                                                          dropped=14)
        assert backend.calls == [], "a call was spent for a room its result could not use"
        assert stage == 4 and kind == "failed"
        assert "no attempt was made" in note, f"the record does not say why: {note!r}"


def test_a_backend_that_raises_fails_the_attempt_not_the_caller():
    """A complete() that dies is a failed attempt, caught: the caller gets an outcome it can
    record instead of an exception through the harness, and the record names what the attempt
    cost, per attempt."""
    class Boom:
        def capabilities(self):
            return Caps(context_tokens=1000, model="t")

        def complete(self, **kw):
            raise RuntimeError("socket died")

    slot, cost, stage, kind, note = compact.summarise(Boom(), source_text="s", room=100, dropped=14)
    assert kind == "failed" and stage == 4
    # Two dead attempts, not three: stage 3 needs a latest summary to re-summarise, and a backend
    # that dies never produces one.
    assert note.count("socket died") == 2, f"each dead attempt names itself: {note!r}"


# --- the room: what the window leaves the summary ----------------------------

def test_the_limit_is_the_smaller_of_the_target_share_and_the_reduction():
    """A context that is 90% full has less room to give than one that is 82% full, whatever the
    window is: the reduction of what the context holds now bounds the result too, and the min is
    the whole predicate."""
    assert compact.limit_for(10000, 9000) == 3600      # min(4000, 3600)
    assert compact.limit_for(10000, 8000) == 3200
    assert compact.limit_for(10000, 10000) == 4000     # the target is the smaller
    assert compact.limit_for(0, 9000) == 0             # no window, no ceiling


def test_the_room_subtracts_what_every_request_carries_beside_the_summary():
    """The budget correction: the room is computed on what is LEFT after the resident load, not on
    the raw window — an oversized system prompt or tool surface eats the window, so the summary's
    room is the limit minus the resident parts and the minimum backfill. No turn is subtracted:
    keeping the goal standing after the rewrite is the summariser's job over the history, not a
    head the harness reserves."""
    assert compact.room_for(10000, 9000, 3000, 2000) == -1400   # nothing to attempt
    assert compact.room_for(10000, 9000, 3000, 400) == 200
    assert compact.room_for(10000, 8000, 3000, 400) == -200     # the reduction bites sooner


def test_fits_within_is_one_comparison_at_the_boundary():
    """The predicate is a single comparison against the limit (which already is the smaller of
    the two ceilings): at the limit it fits, one over it does not — and there is no second check
    a stage could pass while the fit check fails."""
    assert compact.fits_within(100, 100) is True
    assert compact.fits_within(101, 100) is False


def test_the_tool_surface_is_inside_the_resident_and_counted():
    """The measured defect the port keeps counted: an earlier generation sized the summary's
    budget on the raw window and never saw the tool surface the request carries — descriptions
    above all, the strongest form of 'tool descriptions ignored in the budget'. The guard the
    spec names: the tool component of resident is not zero, so the room subtracts what the
    request really pays."""
    spec = ToolSpec("read", "A description long enough to be a fact, not a rounding error: "
                            + "x" * 400,
                    {"type": "object", "properties": {}})
    ctx = compact.Context(system="", specs=[spec])
    assert ctx.resident() > 0, "with nothing but tools, the tool surface alone must still count"
    assert "~0 of tool schemas" not in ctx.decomposition([]), \
        "the decomposition's tool component is zero: the surface was not counted"
    limit = compact.limit_for(10000, 9000)
    assert compact.room_for(10000, 9000, ctx.resident(), 0) == limit - ctx.resident()


# --- the source: what a summary is made from ---------------------------------

def test_the_source_carries_the_prior_and_the_whole_history():
    """No turn is carved out of the source — the run's opening turn is in it like the rest,
    because keeping the goal standing after the rewrite is the summariser's job, not a head the
    harness keeps — beside the previous summary merged in (each summary re-summarises the last
    slot); the history is labelled so the model knows what it is reading, including which tools
    an assistant turn reached for."""
    drop = [Message("user", "the goal"), Message("assistant", "a"),
            Message("assistant", "", tool_calls=(("c1", "read", {}),)),
            Message("tool", "the file", tool_call_id="c1")]
    src = compact.source("the earlier notes", drop)
    assert "merge it in" in src and "the earlier notes" in src
    assert "[user]: the goal" in src, "the opening turn is summarisable content, not a carved-out head"
    assert "[assistant]: a" in src
    assert "wants to call: read" in src
    assert "[tool result]: the file" in src


def test_a_tool_result_in_the_source_is_cut_to_the_cap_and_named():
    """The cap is on the request's side: the summary call carries only its system prompt and the
    source, and a result an operator never re-reads need not arrive at the summariser in full —
    but the cut is named, because the summariser being told the result was longer is a fact worth
    keeping."""
    drop = [Message("tool", "x" * 5000, tool_call_id="c1")]
    src = compact.source(None, drop)
    assert "[... truncated 3000 chars]" in src
    assert "x" * 2000 in src and "x" * 2001 not in src


# --- the frame: what the slot says to the model ------------------------------

def test_the_frame_carries_the_backfill_note_when_a_tail_follows():
    """The slot is followed by the verbatim tail, and the model has to be told the repetition is
    a repetition — an unexplained repeat reads as new events. `covered` picks the note's words:
    the tail is covered by the slot when it holds a summary, and not when it holds the failed
    note. No tail means no note: the frame stands alone."""
    framed = compact.frame("BODY", backfill=6, covered=True)
    assert "BODY" in framed and "6 messages" in framed and "covered above" in framed
    uncovered = compact.frame("BODY", backfill=6, covered=False)
    assert "6 messages" in uncovered and "covered above" not in uncovered
    assert "kept verbatim" not in compact.frame("BODY")


def test_the_frame_names_the_cut_when_the_rung_dropped_recent_messages():
    """The cut note is the orientation duty for the gap the rung made: the model has to see that
    the conversation between the record and the tail is incomplete, because it cannot steer
    around information it cannot see it is missing. It sits beside the backfill's note (the tail
    it stands in front of), and a frame the rung never cut carries no note at all."""
    framed = compact.frame("BODY", backfill=2, covered=True, cut=4)
    assert "4 of the most recent messages" in framed, framed
    assert "potentially missing information" in framed, framed
    assert "kept verbatim" in framed, "the cut note does not replace the backfill's note"
    # The note stands before the backfill note: it names the gap the tail's note follows.
    assert framed.index("potentially missing") < framed.index("kept verbatim"), framed
    # No cut, no note: the ordinary frame is byte-identical to before the rung existed.
    assert compact.frame("BODY", backfill=2, covered=True, cut=0) == compact.frame(
        "BODY", backfill=2, covered=True)
    # A cut with no surviving tail still says the gap is a gap.
    assert "potentially missing information" in compact.frame("BODY", backfill=0, cut=3)


# --- the trigger --------------------------------------------------------------

def test_occupancy_is_a_share_of_the_context():
    assert compact.occupancy(82, 100) == pytest.approx(0.82)
    assert compact.occupancy(0, 0) == 0.0                        # a backend that declares no size


def test_compaction_fires_near_full_and_not_before():
    assert compact.should_compact(90, 100, 20) is True
    assert compact.should_compact(82, 100, 20) is True          # exactly at the trigger: it fires
    assert compact.should_compact(50, 100, 20) is False          # half full: nothing to do


def test_a_short_conversation_is_never_summarised():
    """Two turns are not worth a request that loses information. Length gates the trigger alongside
    occupancy."""
    assert compact.should_compact(99, 100, 3) is False


# --- the split ----------------------------------------------------------------

def test_recent_turns_survive_verbatim_and_the_first_message_is_in_the_drop():
    """No message is privileged: of 20 messages the drop is 14 — the oldest, the run's opening
    turn included — and the kept tail is the six most recent. Keeping the goal standing after the
    rewrite is the summariser's job over the whole history, not a head the harness stands aside."""
    drop, keep = compact.split(msgs(20))
    assert len(keep) == 6 and len(drop) == 14
    assert msgs(20)[0] in drop, "the opening message is summarisable content, not a protected head"
    assert drop[-1] is not keep[0], "the drop and the backfill share no message"


def test_the_walk_back_never_leaves_a_result_without_its_call():
    """The cut walks back so a kept tool result never opens on one whose call was summarised —
    the orphan a server rejects — and the walk may cross the opening message on its way: it is
    content like the rest, not a boundary the cut must keep clear of."""
    convo = [Message("user", "the goal"),
             Message("assistant", "", tool_calls=(("c1", "read", {}),)),
             Message("tool", "r1", tool_call_id="c1"),
             Message("assistant", "", tool_calls=(("c2", "read", {}),)),
             Message("tool", "r2", tool_call_id="c2")]
    drop, keep = compact.split(convo, keep_recent=1)
    assert keep[0].role != "tool", "a kept tail must not open on a result whose call was dropped"
    assert convo[0] in drop, "the opening message is content the walk may summarise"


def test_a_tool_result_is_never_cut_from_its_call():
    """Dropping a call's results leaves the server an orphan tool message it rejects. The cut walks
    back so a role='tool' result never ends up summarised while its assistant call is kept."""
    convo = (msgs(4) + [Message("assistant", "", tool_calls=(("c1", "read", {}),)),
                        Message("tool", "the file", tool_call_id="c1")])
    drop, keep = compact.split(convo, keep_recent=2)            # would cut through the tool result
    assert keep and keep[0].role != "tool"                       # cut moved before the call


# --- the cut rung: the backfill itself exceeds the limit ------------------------

def test_a_backfill_that_fits_the_budget_is_left_whole():
    """The rung only runs when the fit check fails: a tail inside its budget is the ordinary case,
    and a cut that changes nothing would mark the record with a cut that did not happen."""
    keep = msgs(6)
    out, dropped = compact.cut_backfill(keep, compact.estimate_tokens(keep))
    assert out == list(keep) and dropped == 0


def test_the_cut_drops_oldest_until_the_tail_fits_and_counts_what_went():
    """The oldest messages go first — the most recent are the work in progress the next request
    has to see — and the count of what was dropped comes back beside the tail so the record can
    name it. The estimate, not the wire, is what the rung works on: it is the standing rate, and
    the budget the rung is given is the limit's arithmetic at the same rate."""
    keep = [Message("user", "old"), Message("user", "older"),
            Message("user", "x" * 4000), Message("user", "new")]
    # The budget is the estimate of the tail the cut should leave: the assertion is on the
    # behaviour (the two oldest go, the count comes back), not on the estimate's own arithmetic.
    budget = compact.estimate_tokens([keep[2], keep[3]])
    out, dropped = compact.cut_backfill(keep, budget)
    assert [m.content for m in out] == ["x" * 4000, "new"], out
    assert dropped == 2, "the count is the messages gone from the head, for the record's field"


def test_a_call_and_its_results_drop_as_one_unit():
    """Dropping an assistant's calls but keeping its results leaves the server an orphan it
    rejects; dropping the results but keeping the call leaves a turn the model believes it made
    that the record half-keeps. The unit is the whole: call and every result it answered drop
    together, and the tail the cut leaves never opens on a result whose call is gone."""
    keep = [Message("assistant", "", tool_calls=(("c1", "read", {}), ("c2", "grep", {}))),
            Message("tool", "x" * 4000, tool_call_id="c1"),
            Message("tool", "y" * 4000, tool_call_id="c2"),
            Message("user", "the goal")]
    out, dropped = compact.cut_backfill(keep, 10)
    assert out == [keep[3]], "the unit is the whole: call and both results, not one of them"
    assert dropped == 3
    # The unit at the head of a tail: dropping it cannot leave a result without its call.
    assert all(m.role != "tool" for m in out[:1] or [out[0]]), out


def test_a_budget_the_whole_tail_cannot_satisfy_empties_it_and_reports_it():
    """One unit that alone exceeds the budget is all or nothing: the tail empties and the caller
    sees it — the count is the whole tail — because the only reading of a tail that cannot be cut
    to fit is that the window cannot hold the resident load, which is the run's declared ending,
    not a cut the rung can make."""
    keep = [Message("user", "x" * 4000)]
    out, dropped = compact.cut_backfill(keep, 10)
    assert out == [] and dropped == 1
    out, dropped = compact.cut_backfill(keep, -5)               # a negative budget is the same case
    assert out == [] and dropped == 1
    assert compact.cut_backfill([], 10) == ([], 0)


# --- CMP-1: what counts as a refusal for size ---

MEASURED_REFUSAL = ('transport: HTTPError (HTTP 400): Bad Request [server said: {"error":{"code":400,'
                    '"message":"request (200013 tokens) exceeds the available context size (16128 '
                    'tokens), try increasing it","type":"exceed_context_size_error","n_ctx":16128}}]')


def test_a_refusal_for_size_is_recognised_from_the_measured_server_words():
    assert compact.looks_like_overflow(MEASURED_REFUSAL) is True
    # The OpenAI dialect, recorded as inferred: documented wording, not measured on anything reachable.
    assert compact.looks_like_overflow(
        'transport: HTTPError (HTTP 400): [server said: {"error":{"message":"This model\'s maximum '
        'context length is 8192 tokens. However, you requested 10102 tokens"}}]') is True


def test_an_error_that_only_names_the_window_is_not_a_refusal_for_filling_it():
    """Constructed, not captured — and it exists because of what the marker set originally contained.

    The measured llama.cpp body carries `"n_ctx": 16128` as a JSON key, so `n_ctx` looks like a free
    marker. It is not one: a server validating a *request field* says something like the text below, and
    that run's prompt may be a hundred tokens long. Compacting it would summarise a short conversation
    for nothing, which is the loss this whole finding is trying to make rare. A marker has to assert that
    the request was too big, not merely appear in an error that mentions size.
    """
    config_error = ('transport: HTTPError (HTTP 400): Bad Request [server said: '
                    '{"error":{"message":"n_ctx must be a positive integer",'
                    '"code":"invalid_request_error"}}]')
    assert compact.looks_like_overflow(config_error) is False

    # Constructed on the same reasoning, and this one is the case that a loose detector cannot survive:
    # the word "context" is in it. A chat template that fails to render refuses a request of any size,
    # so a run that compacted here would summarise its own conversation to fix someone else's file.
    template_error = ("transport: HTTPError (HTTP 500): Internal Server Error [server said: "
                      "{\"error\":{\"message\":\"chat template error: the context variable "
                      "is undefined in this template\",\"type\":\"server_error\"}}]")
    assert compact.looks_like_overflow(template_error) is False

    assert compact.looks_like_overflow(
        "transport: ConnectionResetError: connection reset by peer") is False
    assert compact.looks_like_overflow("the stream ended without a finish reason") is False
    assert compact.looks_like_overflow("") is False


def test_every_overflow_marker_asserts_that_the_request_was_too_big():
    """The rule the last test's reasoning generalises, stated where it can be enforced.

    A future contributor adding a marker will add a *substring*, and the substring that tempts them is
    the one that appears in every such error — "context" — which is also the one that appears in errors
    about configuration, templates and resume ids. So the table itself is checked: each marker must claim
    excess, by naming a limit being exceeded or asking the caller to shorten. That is a weaker test than
    a corpus of real servers would be, and much stronger than a comment saying "be specific"."""
    assert compact.OVERFLOW_MARKERS, "an empty table matches nothing, so no run ever recovers"
    for marker in compact.OVERFLOW_MARKERS:
        assert marker == marker.lower(), f"marker {marker!r} would never match a lowercased error"
        assert ("exceed" in marker or "maximum context" in marker or "reduce the length" in marker), (
            f"marker {marker!r} names size without asserting excess")

def test_a_message_of_parts_is_carried_as_its_text_not_as_its_python():
    """The declared type says content may be a list of parts; the summariser must read that type.

    `_text` is what turns a message into summary input. Measured before the fix, a parts list came out as
    `[{'type': 'text', 'text': 'the first part'}, ...]` -- the summary of a run that carried parts would
    quote Python at the model. An image part is named rather than dropped: a summary that silently forgets
    an image was there loses a fact, and naming it costs one word.
    """
    parts = [{"type": "text", "text": "the first part"},
             {"type": "image_url", "image_url": {"url": "file:///x.png"}},
             {"type": "text", "text": "the tail"}]
    got = compact._text(parts)
    assert "the first part" in got and "the tail" in got, got
    assert "{'type'" not in got and "image_url" not in got, f"a parts list was rendered as Python: {got!r}"
    assert "image" in got.lower(), f"an image part vanished without a trace: {got!r}"


# --- the local model: what the run's next request actually carries -------------------------


def test_the_estanding_rate_counts_four_characters_to_a_token():
    assert compact.estimate_text("abcd") == 1
    assert compact.estimate_text("") == 0
    assert compact.estimate_text("a" * 400) == 100


def test_a_message_is_measured_by_what_it_holds_not_by_being_one_message():
    """The trigger and the room arithmetic both speak in tokens, and the only cheap stand-in for a
    tokenizer the harness does not have is a character count. What matters is the *shape*: a
    400-character result and a 4-character one must not weigh the same."""
    assert compact.estimate_tokens([Message("user", "a" * 400)]) == 100
    assert compact.estimate_tokens([Message("user", "abcd"), Message("user", "abcd")]) == 2
    assert compact.estimate_tokens([]) == 0


def test_an_image_part_is_counted_by_its_wire_bytes_not_its_name():
    """A picture travels as a data URL, and the estimate counts what travels: the base64, not a
    filename. On a vision endpoint the server counts far fewer tokens (image tokens), so this
    over-counts — the safe direction for the one question these numbers are asked, and corrected
    by calibration on the text requests that follow."""
    url = "data:image/png;base64," + "A" * 400
    parts = [{"type": "text", "text": "a picture"},
             {"type": "image_url", "image_url": {"url": url}}]
    assert compact.message_tokens(parts) == (len("a picture") + len(url)) // 4
    assert compact.message_has_image(parts) is True
    assert compact.message_has_image("plain text") is False
    assert compact.message_has_image([{"type": "text", "text": "x"}]) is False


def test_the_resident_parts_are_counted_once_and_do_not_move():
    """The system prompt and the tool schemas ride along on every request but are fixed for the
    run: counted once at construction, independent of whatever the conversation is."""
    spec = ToolSpec("read", "Read a file.", {"type": "object", "properties": {}})
    ctx = compact.Context(system="s" * 400, specs=[spec])
    resident = ctx.resident()
    assert resident > 0
    assert ctx.total([]) == resident
    assert ctx.total([Message("user", "abcd")]) == resident + 1
    assert ctx.resident() == resident, "the resident parts moved with the conversation"


def test_an_uncalibrated_context_is_the_standing_estimate():
    ctx = compact.Context(system="", specs=[])
    assert ctx.ratio is None
    assert ctx.effective([Message("user", "a" * 400)]) == 100


def test_calibration_learns_the_servers_rate_and_applies_it():
    ctx = compact.Context(system="s" * 400, specs=[])
    local = ctx.total([Message("user", "a" * 400)])
    ctx.calibrate(local, 2 * local)
    assert ctx.ratio == pytest.approx(2.0)
    assert ctx.effective([Message("user", "a" * 400)]) == 2 * local
    # A later report overwrites: the rate is what this server does now, not what it did once.
    ctx.calibrate(local, local // 2)
    assert ctx.effective([Message("user", "a" * 400)]) == local // 2


def test_calibration_never_learns_from_a_request_it_cannot_measure():
    """A local count of zero is an empty request as far as the estimate can see — nothing to
    measure a rate against — and the server's count of it is not a rate for other requests."""
    ctx = compact.Context(system="", specs=[])
    ctx.calibrate(0, 42)
    assert ctx.ratio is None
    ctx.calibrate(100, 0)
    assert ctx.ratio is None, "a server count of zero is not a measurement"


def test_the_decomposition_names_the_history_and_the_system_prompt_apart():
    """The split the operator is owed — how much of the window is the run's standing prompt and
    how much is the conversation — in the words a notice carries, each marked estimate."""
    ctx = compact.Context(system="s" * 400, specs=[])
    text = ctx.decomposition([Message("user", "a" * 400)])
    assert "~100 of system prompt" in text
    assert "~100 of conversation" in text
    assert "~0 of tool schemas" in text
