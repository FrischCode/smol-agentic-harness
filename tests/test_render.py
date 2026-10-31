"""The console renderer. It is the throwaway projection, so the tests pin the promises that make it
safe to be decorative: it shows the real content, marks failures, never drops a record type it
predates, goes silent when told, and never paints colour onto a pipe. Display is cosmetic, so these
assert shape, not pixels."""
import io
from dataclasses import replace

from sah import record, render
from sah.model import ReasoningDelta, TextDelta


def term(view="normal", color=False):
    out = io.StringIO()
    return render.Terminal(out, io.StringIO(), view, color=color), out


def step(kind, **fields):
    return {"kind": kind, "id": f"1-{kind}", "ts": "2026-10-24T05:08:31+00:00", "issuer": "core",
            **fields}


def test_normal_frames_an_assistant_turn():
    t, out = term()
    t.on_event(TextDelta("hello there"))
    t.on(step("message", role="assistant", text="hello there", tokens_in=10, tokens_out=3, ms=1234))
    text = out.getvalue()
    assert "hello there" in text and "ASSISTANT" in text
    assert "╭" in text and "╰" in text                       # a framed box
    assert "10 CTX · ›3 TOK · 1.2s" in text                  # the footer reads declared turn fields


FOOTER = dict(role="assistant", text="done", outcome="ok", tokens_in=3647, context_max=32768,
              tokens_cached=0, tokens_out=125, prefill_tps=881.4, tps=45.2, ms=7100)


def test_the_footer_states_the_window_the_counts_the_rates_and_the_wall_time():
    """One line, in the order an operator reads it: how full the window is, then the token group, then
    how long the turn took. `¤` marks the part of the prompt the server served from its own cache, `›`
    what came out, and the two arrows are the two rates the server measured for itself — prefill and
    decode, the `/s` standing on the last because it covers both."""
    t, out = term()
    t.on(step("message", **FOOTER))
    assert "3647/32768 CTX · ¤0 ›125 ↑881 ↓45/s TOK · 7.1s" in out.getvalue()


def test_an_unreported_rate_leaves_no_arrow_and_no_placeholder():
    """A server that reports no throughput (llama.cpp answers with counts only) yields a footer with no
    arrows at all. A `?` would assert that a countable thing was measured and lost, and a `0/s` would
    assert a speed of zero; an unknown speed is neither."""
    t, out = term()
    t.on(step("message", role="assistant", text="x", tokens_in=1909, context_max=32768,
              tokens_cached=0, tokens_out=2, ms=2200))
    text = out.getvalue()
    assert "1909/32768 CTX · ¤0 ›2 TOK · 2.2s" in text
    assert "↑" not in text and "↓" not in text and "?" not in text


def test_a_rate_the_server_measured_as_zero_prints_no_arrow():
    """A server that did no work reports a rate of zero — measured, a request whose whole context hit the
    prompt cache answered `prompt_tokens_per_second: 0` — and a footer that printed `↑0 ↓0/s` would say
    the machine ran at zero speed rather than that it never measured a speed at all."""
    t, out = term()
    t.on(step("message", role="assistant", text="x", tokens_in=100, tokens_out=2,
              prefill_tps=0.0, tps=0.0))
    text = out.getvalue()
    assert "↑" not in text and "↓" not in text, text


def test_a_server_that_said_nothing_about_its_cache_prints_no_cache_mark():
    """A stated 0 prints (`¤0`) and silence prints nothing: "the server cached nothing" and "the server
    never said" are different facts, and printing a zero for silence would make every cold run read as
    a warm one that missed."""
    t, out = term()
    t.on(step("message", role="assistant", text="x", tokens_in=100, tokens_out=2))
    assert "¤" not in out.getvalue()


def test_a_count_without_its_ceiling_prints_the_count_alone():
    """No `--context` named means no ceiling is known, and the occupancy figure stands alone rather than
    carrying a `/0` that reads as an empty window."""
    t, out = term()
    t.on(step("message", role="assistant", text="x", tokens_in=100, tokens_out=2))
    assert "100 CTX" in out.getvalue() and "/0" not in out.getvalue()


def test_the_outcome_word_leads_the_footer():
    """`truncated` is the fact the line most needs to carry, so it leads and it is last to be dropped."""
    t, out = term()
    t.on(step("message", **dict(FOOTER, outcome="truncated")))
    assert "truncated · 3647/32768 CTX" in out.getvalue()


def test_the_footer_keeps_every_figure_even_at_digits_no_server_has():
    """Nothing is cut for width. The ladder that used to drop segments existed because a padded bottom
    rule let one long footer push the right edge out and make every frame below it look broken; the rule
    is a plain line now, so the worst a huge footer costs is a wrap in the terminal, and a wrap loses a
    figure nothing else can recover. Asserted at absurd magnitudes on purpose: this is the claim that the
    line is complete, and it has to be tested where the old code would have trimmed."""
    t, out = term()
    t.on(step("message", role="assistant", text="x", outcome="truncated", tokens_in=99999999,
              context_max=99999999, tokens_cached=99999999, tokens_out=9999999,
              prefill_tps=999999.0, tps=9999.0, ms=9999900))
    footer = _footer_of(out)
    for figure in ("truncated", "99999999/99999999 CTX", "¤99999999", "›9999999",
                   "↑999999", "↓9999/s", "TOK", "9999.9s"):
        assert figure in footer, f"{figure!r} is missing: {footer}"


def _footer_of(out) -> str:
    """The footer line of the box that was just drawn."""
    return [ln for ln in out.getvalue().splitlines() if ln.startswith("╰")][0]


def test_the_close_rule_is_the_footer_and_nothing_else():
    """The footer follows its corner with no padding around it and no stub after it. Neither half can be
    inferred from a "did the footer appear?" check -- a padded rule carries the same text -- so the line
    is asserted as a whole."""
    t, out = term()
    t.on(step("message", **FOOTER))
    footer = _footer_of(out)
    assert footer.startswith("╰─ ") and not footer.rstrip().endswith("─"), footer
    assert "  " not in footer and " · " in footer, footer


def test_a_box_with_nothing_to_say_prints_no_close_rule_at_all():
    """A turn whose step carries no figures -- a user's own words, which the server never tokenised --
    has no footer, and then no rule is drawn rather than a bare corner. The glyph is what a footer
    attaches to; on its own it claims an edge below content that stopped being true when the dashes did."""
    t, out = term()
    t.on(step("message", role="user", text="hello there", issuer="user"))
    lines = out.getvalue().splitlines()
    assert "hello there" in lines
    assert not [ln for ln in lines if ln.startswith("╰")], lines


def _head_of(out) -> str:
    """The open rule of the box that was just drawn."""
    return [ln for ln in out.getvalue().splitlines() if ln.startswith("╭")][0]


def test_the_open_rule_ends_with_its_label_and_carries_no_fill():
    """The dashes after a box's label claimed the box was `BOX_WIDTH` columns wide, a claim this renderer
    does not honour: nothing bounds content by the frame, so the fill stated a width the body ignores.
    The open rule is the label — time and role — and it ends there."""
    s = step("message", role="user", text="hello", issuer="user")
    t, out = term()
    t.on(s)
    assert _head_of(out) == f"╭─ {render._hhmmss(s)} · YOU", _head_of(out)


def test_a_core_word_in_the_users_seat_wears_the_cores_face():
    """The core puts words in the user's seat (a nudge, a wrap-up, a future word): the wire carries
    them as a user message because that is where the model's attention is, and the record stamps the
    issuer because the operator must not read the harness's voice as their own typing. The face
    follows the issuer — the core's label in the framed view, the core's prefix in the labelled
    compact one — never what the word happens to be about: the channel owns the face, so any future
    core word inherits it, and operator input keeps the operator's face."""
    word = step("message", role="user", text="emit that call fresh now")
    t, out = term()
    t.on(word)
    assert _head_of(out) == f"╭─ {render._hhmmss(word)} · CORE", _head_of(out)
    t2, out2 = term(view="quiet")
    t2.on(word)
    assert "core emit that call fresh now" in out2.getvalue(), out2.getvalue()
    theirs = step("message", role="user", text="what is in this folder", issuer="user")
    t3, out3 = term()
    t3.on(theirs)
    assert _head_of(out3) == f"╭─ {render._hhmmss(theirs)} · YOU", _head_of(out3)


def test_the_core_frame_carries_its_own_colour():
    """The label alone would let a future edit repaint the core's frame in the operator's cyan and no
    shape test would notice: the point of the channel is that the two voices are told apart at a
    glance, and colour is half of that. Pinned by the palette constant, the way the disclosure test
    pins the policy constant — by identity, not by a copy of the bytes."""
    word = step("message", role="user", text="emit that call fresh now")
    t, out = term(color=True)
    t.on(word)
    core_lines = [ln for ln in out.getvalue().splitlines() if "· CORE" in ln]
    assert core_lines and render._MAGENTA in core_lines[0], out.getvalue()


def test_a_tool_header_ends_with_the_tool_name_for_the_same_reason():
    t, out = term()
    t.on(step("call", name="read", args={"path": "x"}, call_id="c1"))
    assert _head_of(out).endswith("· TOOL · read"), _head_of(out)


def test_an_over_long_header_is_printed_in_full_rather_than_cut_to_the_frame():
    """Nothing bounds a header by the frame, and with no fill on the open rule there is no longer a
    visible reference width against which an over-long one shows as an overflow. So the promise that a
    label is never truncated lives here instead of on screen."""
    t, out = term()
    long_name = "m" * 80
    t.on(step("call", name=long_name, args={}, call_id="c1"))
    assert _head_of(out).endswith(long_name), _head_of(out)


def test_streamed_prose_is_not_printed_twice_when_the_message_step_lands():
    t, out = term()
    t.on_event(TextDelta("hi"))
    t.on(step("message", role="assistant", text="hi"))
    assert out.getvalue().count("hi") == 1                    # painted once from the stream, box just closed


def test_a_tool_box_opens_on_call_fills_live_and_closes_on_result():
    t, out = term()
    t.on(step("call", name="read", args={"path": "/x"}, call_id="c1"))
    t.on_event(TextDelta("file body", call_id="c1"))          # live output into the open box
    t.on(step("result", refs="1-call", ok=True, content="file body"))
    text = out.getvalue()
    assert "TOOL · read" in text and '"path": "/x"' in text
    assert "file body" in text and text.count("file body") == 1   # streamed once, not repeated at result
    assert "╰" in text


def test_a_failed_result_is_marked_failed():
    t, out = term()
    t.on(step("call", name="run", args={}, call_id="c1"))
    t.on(step("result", refs="1-call", ok=False, content="boom: it broke"))
    text = out.getvalue()
    assert "failed" in text and "boom" in text


def test_an_unknown_kind_is_never_dropped():
    """A record type this renderer predates must still appear, by its declared fields — this is the
    exact class of failure (refusals vanishing) the old renderer had."""
    t, out = term()
    t.on(step("telemetry", gpu="5090", seconds=12))
    text = out.getvalue()
    assert "telemetry" in text and "gpu='5090'" in text


def test_off_mode_is_silent():
    t, out = term(view="off")
    t.on_event(TextDelta("should not appear"))
    t.on(step("message", role="assistant", text="should not appear"))
    t.on(step("result", refs="x", ok=False, content="nope"))
    assert out.getvalue() == ""


def test_quiet_mode_is_one_line_and_never_frames():
    t, out = term(view="quiet")
    t.on(step("message", role="assistant", text="done"))
    t.on(step("call", name="read", args={"path": "/x"}, call_id="c1"))
    text = out.getvalue()
    assert "done" in text and "→ read" in text
    assert "╭" not in text and "╰" not in text


def test_colour_is_never_painted_onto_a_pipe():
    # a StringIO is not a tty, so even the colour default must produce no escape codes
    out = io.StringIO()
    t = render.Terminal(out, io.StringIO(), "normal", color=None)   # color=None -> autodetect (not a tty)
    t.on(step("call", name="read", args={}, call_id="c1"))
    assert "\033[" not in out.getvalue()


def test_colour_appears_when_requested():
    t, out = term(color=True)
    t.on(step("call", name="read", args={}, call_id="c1"))
    assert "\033[" in out.getvalue()


def test_reasoning_is_shown_only_when_the_view_asks():
    on, out_on = term()                                        # normal shows reasoning
    on.on_event(ReasoningDelta("thinking hard"))
    assert "thinking hard" in out_on.getvalue()

    offview = render.VIEWS["quiet"]                            # quiet does not
    t, out = term()
    t.view = offview
    t.on_event(ReasoningDelta("thinking hard"))
    assert "thinking hard" not in out.getvalue()


def test_a_permitted_operation_and_started_are_not_echoed_as_noise():
    """Suppressed on purpose (the result box carries the outcome; the floor guarantees refusals),
    yet these are KNOWN kinds with handlers — the generic fallback is only for kinds with none."""
    t, out = term()
    t.on(step("operation", op="bash", outcome="allowed", reason="granted"))
    t.on(step("started", run_id="r", root="."))
    assert out.getvalue() == ""


def _line_before(out, prefix):
    """The line immediately preceding the first line that starts with `prefix`. A blank line above
    a block is the separator; this is how the tests read spacing without matching pixels."""
    lines = out.getvalue().splitlines()
    idx = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    return lines[idx - 1] if idx else None


def test_a_standalone_block_is_spaced_like_a_box_in_box_mode():
    """The verdict (and any non-box line) is separated from what precedes it by the same blank line
    a box header gets. Without it, boxes have a gap between them but the verdict butts against the
    last box -- the spacing inconsistency this pins shut."""
    t, out = term()
    t.on(step("call", name="bash", args={}, call_id="c1"))
    t.on(step("result", refs="1-call", ok=True, content="done"))
    t.on(step("ended", reason="complete", exit_code=0))
    assert _line_before(out, "◆") == "", "a verdict is preceded by a blank line in box mode"


def test_compact_modes_stay_single_spaced_around_a_standalone_block():
    """The mirror, so the fix does not create a new inconsistency: quiet/text modes are dense by
    design, so a verdict there is NOT padded with a blank line."""
    t, out = term(view="quiet")
    t.on(step("message", role="assistant", text="all good"))
    t.on(step("ended", reason="complete", exit_code=0))
    assert _line_before(out, "◆") != "", "compact modes keep single-spaced lines"


def test_a_box_is_tight_its_body_hugs_the_bottom_border():
    """No gap between a box's content and its own closing border. A streamed-prose box and a
    whole-content box must look the same -- an unconditional newline before the border gave the
    whole-content one a stray blank the streamed one never had."""
    t, out = term()
    t.on(step("call", name="ls", args={}, call_id="c1"))
    t.on(step("result", refs="1-call", ok=True, content="onefile"))
    lines = out.getvalue().splitlines()
    idx = next(i for i, line in enumerate(lines) if line.startswith("╰"))
    assert lines[idx - 1] == "onefile", "the body sits directly above the border, no blank between"


def _term(view):
    out = io.StringIO()
    return render.Terminal(out, io.StringIO(), view, color=False), out


def test_off_prints_nothing_and_no_prose_leaks():
    """`off` means verdicts only. A reasoning flag must not punch a hole in it -- the transcript
    lives in the log, not on the console of a run that asked to be silent."""
    t, out = _term(render.VIEWS["off"])
    t.on({"kind": "message", "id": "1-message", "ts": "2026-10-24T05:00:00+00:00", "issuer": "core",
          "role": "assistant", "text": "the answer", "reasoning": "thinking"})
    t.on({"kind": "message", "id": "2-message", "ts": "2026-10-24T05:00:01+00:00", "issuer": "core",
          "role": "user", "text": "the ask"})
    got = out.getvalue()
    for word in ("agent", "you", "reason", "the answer", "the ask"):
        assert word not in got, f"off leaked {word!r}"


def test_reasoning_adds_only_the_reasoning_line_not_prose():
    """--reasoning reveals the reasoning stream; it must not also reveal the prose the view turned
    off. Each view flag governs its own content, independently."""
    from dataclasses import replace
    t, out = _term(replace(render.VIEWS["off"], reasoning=True))
    t.on({"kind": "message", "id": "1-message", "ts": "2026-10-24T05:00:00+00:00", "issuer": "core",
          "role": "assistant", "text": "the answer", "reasoning": "thinking hard"})
    got = out.getvalue()
    assert "thinking hard" in got
    assert "the answer" not in got and "agent" not in got


def test_streamed_reasoning_stays_on_one_line_and_prose_breaks_after():
    """Reasoning deltas are tokens, not lines. Terminating each with a newline laid the whole
    thought out one word per line. They must concatenate inline, and prose must start fresh."""
    t, out = _term(render.VIEWS["normal"])
    for chunk in ("The ", "user ", "wants ", "a greeting."):
        t.on_event(ReasoningDelta(chunk))
    for chunk in ("Hi", " there", "!"):
        t.on_event(TextDelta(chunk))
    got = out.getvalue()
    assert "The user wants a greeting." in got          # one line, not "The\n user\n wants\n ..."
    assert "wants\n" not in got and "user\n" not in got  # no newline split between tokens
    assert "Hi there!" in got and "\nHi there!" in got   # prose begins on its own line


def test_reasoning_is_split_from_prose_by_exactly_one_blank_line():
    """qwen3_coder closes its reasoning block with a trailing "\n" delta. A thinking turn shows the
    thought, one blank line, then the answer -- exactly one blank, never zero (glued) and never two.
    The reasoning block's own newline must not be double-counted, so prose adds exactly one more."""
    t, out = _term(replace(render.VIEWS["normal"], reasoning=True))
    for chunk in ("Simple task.", " ", "Write it.", "\n"):     # note the trailing-newline delta
        t.on_event(ReasoningDelta(chunk))
    t.on_event(TextDelta("Rivers are veins."))
    got = out.getvalue()
    assert "Simple task. Write it.\n\nRivers are veins." in got, got   # thought, one blank, answer
    assert "\n\n\nRivers" not in got                            # not a double blank


def test_one_blank_separates_thought_from_answer_even_when_reasoning_stops_mid_line():
    """When the last reasoning delta had no newline the cursor is mid-line, so the separator must
    first end the thought's line and then write the blank -- the same single blank a newline-terminated
    thought gets. Prose must not run onto the reasoning line, and must not get two blanks."""
    t, out = _term(replace(render.VIEWS["normal"], reasoning=True))
    for chunk in ("thinking", " about it"):
        t.on_event(ReasoningDelta(chunk))
    t.on_event(TextDelta("the answer"))
    got = out.getvalue()
    assert "thinking about it\n\nthe answer" in got             # thought, one blank, answer
    assert "\n\n\nthe answer" not in got                        # not a double blank


def test_only_the_first_prose_chunk_after_reasoning_owes_the_blank():
    """The blank separates the thought from the answer, not one sentence from the next: later prose
    chunks flow on as they arrive with no further blank line."""
    t, out = _term(replace(render.VIEWS["normal"], reasoning=True))
    t.on_event(ReasoningDelta("thought\n"))
    t.on_event(TextDelta("First. "))
    t.on_event(TextDelta("Second."))
    assert "thought\n\nFirst. Second." in out.getvalue(), out.getvalue()


def test_the_boot_box_titles_the_run_with_the_wordmark_and_draws_only_to_a_witness():
    items = ["model @ host:9090  think=on", "exec: granted", "rw: /proj", "web: public only"]
    t, out = term("normal")
    t.header(items)
    text = out.getvalue()
    # the wordmark titles the block
    assert text.startswith("\n\u256d\u2500 S   \u2571\u2500\u2572   H\n"), text
    # No fill after the title and no bottom rule under the items: the block closes by the first turn box
    # below it, the same way a turn with no footer closes by the next one.
    assert "\u2570" not in text, text
    for item in items:
        assert f"\n{item}\n" in text                     # one fact per line, never joined onto one
    t, out = term("normal", color=True)
    t.header(items)
    assert render._WHITE in out.getvalue()               # the wordmark is lit white on the dim frame
    for view in ("text", "quiet", "off"):                # a one-shot / redirected child has no witness
        t, out = term(view)
        t.header(items)
        assert out.getvalue() == ""                      # draws nothing a pipe would carry


# --- RND-1: what a pipe is allowed to contain -------------------------------

# One step of every kind the record can hold, so a kind added tomorrow is covered by these tests
# without anyone remembering to add a case. Each carries the fields a real step of that shape has;
# extra fields are what the generic fallback prints, which is exactly the noise under test.
def _every_kind():
    fields = dict(text="the answer", subject="widen", item="/etc", name="read",
                  args={"path": "/etc"}, ok=True, outcome="ok", reason="complete",
                  exit_code=0, refs="1", action="steer", role="assistant")
    return [step(kind, **fields) for kind in sorted(record.KINDS)]


def test_the_text_view_is_the_answer_and_nothing_else():
    """`sah-call "…" | grep x` is a program reading the answer. The launcher promises the pipe holds
    the answer alone, and `text` is the view that pipe is built with, so the promise is testable in
    one line: stdout is the assistant's text and a newline, and nothing else.

    The lines this refuses are not decoration. An echoed objective is the caller's own input handed
    back to it, an `agent ` prefix makes the first line lie about where it starts, and a generic
    `· request: …` line appears only when a run asked a question or was steered -- which is precisely
    when a caller has no way to predict its own pipe has changed shape.
    """
    t, out = term("text")
    for s in _every_kind():
        t.on(s)
    t.on(step("message", role="user", text="what is in this folder", issuer="user"))
    assert out.getvalue() == "the answer\n", f"the pipe carried: {out.getvalue()!r}"


def test_the_off_view_leaves_stdout_empty_for_every_kind():
    """A child runs with view `off` and its stdout redirected to `child.out`. Verdicts are not lost by
    staying quiet here -- the verdict floor and the questions surface write them to **stderr**, which
    is a different stream and the operator's, not the data's."""
    t, out = term("off")
    for s in _every_kind():
        t.on(s)
    assert out.getvalue() == "", f"off wrote to stdout: {out.getvalue()!r}"


def test_quiet_keeps_the_speaker_labels_a_person_reads_them_by():
    """`quiet` is for a person at a terminal, where `you`/`agent` is the difference between a question
    and an answer. Silencing it there would fix the pipe by breaking the reader, so the two views are
    separated by a flag rather than by a shared "compact" notion."""
    t, out = term("quiet")
    t.on(step("message", role="user", text="what is in this folder", issuer="user"))
    t.on(step("message", role="assistant", text="three files"))
    got = out.getvalue()
    assert "you  what is in this folder" in got, got
    assert "agent three files" in got, got


def test_a_person_still_sees_the_ask_the_decision_and_the_steer():
    """The pipe goes quiet; a person does not. A run that asked a question, refused something, or took
    a steer must show it in the two views a human reads, or the renderer has hidden the only three
    steps that explain why an answer arrived late or not at all."""
    for view in ("normal", "quiet"):
        t, out = term(view)
        t.on(step("request", subject="widen", item="/etc", why="the run cannot read it yet"))
        t.on(step("decision", refs="1", outcome="refused", reason="no answer"))
        t.on(step("command", action="steer", args={"text": "use the log"}))
        got = out.getvalue()
        assert "/etc" in got, f"{view} hid what the run asked for: {got!r}"
        assert "refused" in got, f"{view} hid the decision: {got!r}"
        assert "steer" in got, f"{view} hid the command: {got!r}"


def test_the_piped_views_still_show_no_generic_line_for_a_kind_they_predate():
    """The generic fallback exists so a record type older than the renderer is never silently dropped,
    and it must survive in the human views. It must not reach the pipe, where it reads as the answer.
    """
    predated = step("a_kind_not_in_the_record", subject="something", text="value")
    for view, seen in (("normal", True), ("quiet", True), ("text", False), ("off", False)):
        t, out = term(view)
        t.on(predated)
        assert ("a_kind_not_in_the_record" in out.getvalue()) is seen, \
            f"{view}: {out.getvalue()!r}"


def test_a_dumb_terminal_gets_no_colour_even_on_a_tty(monkeypatch):
    """The condition this replaces read `"TERM" not in ("dumb",)`, which asks whether the four-letter
    string `"TERM"` is a member of a tuple containing `"dumb"`. That is true whatever the environment
    holds, so the operator's terminal was never consulted and `TERM=dumb` — the setting a pager, an
    emulator or a screen reader uses to say "no escape codes" — was silently ignored.

    Asserted in both directions on purpose: with a plausible TERM the answer must still be colour, or
    the test would pass by making the check always false, which is the same bug mirrored."""

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert render._wants_color(Tty()), "a real terminal with a real TERM must still be painted"
    monkeypatch.setenv("TERM", "dumb")
    assert not render._wants_color(Tty()), "TERM=dumb is the operator asking for no escape codes"


def test_a_box_clock_reads_the_clock_the_operator_is_looking_at():
    """A turn's box header sliced the stamp's own text after the `T`, which is UTC by the record's rule
    (`ts` stores time in UTC), while the fallback for a step with no stamp used local time. On this box
    (UTC+1 in summer) the two sources disagreed by an hour, so a run's timeline read an hour away from
    the wall clock beside it, and the disagreement only appeared when you compared them.

    Both numbers here derive from one instant, so no wall clock is read mid-test and no second boundary
    can make this flaky: the expected value is the same instant converted to local, computed
    independently of the function under test."""
    from datetime import datetime, timezone

    instant = datetime.now(timezone.utc)
    ts = instant.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rendered = render._hhmmss(step("message", role="assistant", text="hi", outcome="ok", ts=ts))
    assert rendered == instant.astimezone().strftime("%H:%M:%S"), (
        f"the renderer showed {rendered} for an instant at "
        f"{instant.astimezone().strftime('%H:%M:%S')} local")


def test_a_summary_that_worked_states_its_arithmetic_and_no_verdict_word():
    """`OK · KEPT 12 · DROPPED 4` read as two verdicts, because the outcome and the count shared the word
    "kept". The numbers belong to the counts and the verdict is the absence of a failure: the counts lead,
    and `ok` prints nothing."""
    t, out = term()
    t.on(step("summary", outcome="ok", kept=12, dropped=4, text="a summary"))
    head = [ln for ln in out.getvalue().splitlines() if ln.startswith("╭")][0]
    assert head.endswith("SUMMARY (12 KEPT · 4 DROPPED)"), head


def test_a_failed_summary_shows_its_reason_on_the_console():
    """The console projection is the one an operator is looking at when a compaction gives up, so it carries
    the same fact the report does. Splitting the verdict from its sentence was right and would otherwise have
    silently made the reason unshowable in both projections."""
    t, out = term()
    t.on(step("summary", outcome="failed", kept=2, dropped=5, text="(no summary produced)",
              reason="the server refused the request: too big"))
    # The box head is drawn upper-cased, so the comparison is made on the lower-cased text: the promise is
    # that the reason is showable, not that it keeps a particular case.
    text = out.getvalue().lower()
    assert "failed" in text and "too big" in text, text


def test_a_transient_line_renders_as_a_caption_where_it_is_spoken():
    """The transient channel: a line a watcher (or a long tool's body) speaks to the operator who is
    watching now. It lands on the console's face under the step it is about — its own mark, spaced
    like a top-level line, never glued to the box above it (the gluing a raw print used to do)."""
    t, out = term()
    t.on(step("call", refs="1-message", name="speak", args={"text": "hi"}))
    t.transit("the run asks to say — hi")
    text = out.getvalue()
    assert "… the run asks to say — hi" in text
    assert text.index("… the run asks to say") > text.index("TOOL · speak"), \
        "the caption lands where it is spoken: under the ask, before the answer"
    assert '}\n\n… the run asks to say' in text, "the caption is spaced, not glued"


def test_a_transient_line_says_nothing_where_stdout_is_data():
    """The line is the run's own mechanics, so it renders where the mechanics render (the `meta`
    views) and says nothing where they do not: a line for an operator who is watching now, and the
    record holds the fact where there is no one watching. The compact views keep the density: no
    blank line above the caption."""
    t, out = term(view="quiet")
    t.transit("a quiet line")
    assert "… a quiet line" in out.getvalue() and out.getvalue().count("\n\n…") == 0
    for view in ("text", "off"):
        t, out = term(view=view)
        t.transit("a line that must not say")
        assert out.getvalue() == "", f"{view} keeps stdout as data"


def test_a_multiline_say_says_each_line_and_a_blank_says_nothing():
    """Forgiving by design — a cosmetic channel that raised would fail a tool call over a caption:
    a line with newlines in it says several, and a line that is blank says nothing at all."""
    t, out = term()
    t.transit("first\n\nsecond")
    assert "… first" in out.getvalue() and "… second" in out.getvalue()
    t, out = term()
    t.transit("   \n")
    assert out.getvalue() == ""


def test_a_status_with_no_terminal_says_nothing_and_never_raises():
    """A run with no witness says nothing on purpose: the console is not there, the line is a word,
    and the word is dropped rather than a missing attribute that crashes a watcher."""
    render.Status().say("no one is watching")
    render.Status().say("")
    render.Status().say("first\nsecond")
