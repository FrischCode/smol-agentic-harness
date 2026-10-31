"""The prompt files and their loader. The guarantees: the shipped files exist, `load` renders their
inserts (and a caller override wins), and no shipped file leaves a known placeholder unfilled."""
import datetime as dt
import pathlib

from sah import prompts

PROMPT_DIR = pathlib.Path(prompts.__file__).parent


def test_the_shipped_prompt_files_exist():
    for name in ("preamble", "agents", "repeat_steer"):
        assert (PROMPT_DIR / f"{name}.md").is_file()


def test_load_returns_non_empty_rendered_text():
    for name in ("preamble", "agents", "repeat_steer"):
        text = prompts.load(name)
        assert text.strip(), f"{name}.md rendered empty"
        assert "{{" not in text, f"{name}.md left an unfilled insert"


def test_the_compaction_prompt_files_exist_and_their_numbered_inserts_are_named():
    """The compactor's prose lives in prompt files, not in `sah/compact.py`: the compactness ask is
    a file an owner retunes per model, and it names the room's number through the `{{room}}` insert
    — harmless for most models, load-bearing for some, and a number hard-coded in the code is the
    constant the spec refuses. The frame's body is its own insert, substituted by hand (model prose
    must not enter the template's pass), and the failed note and the cut note carry the words that
    orient the model to what is missing."""
    for name in ("compact_system", "compact_step1", "compact_step2", "compact_step3",
                 "compact_prior", "compact_history", "compact_turn",
                 "compact_truncated", "slot_failed", "summary_cut", "summary_frame",
                 "backfill_covered", "backfill_uncovered", "compact_done"):
        assert (PROMPT_DIR / f"{name}.md").is_file(), \
            f"{name}.md is missing: the compactor's prose lives in code"
        assert prompts.load(name, room=100, text="SOMETHING", prior="p",
                            history="h", label="l", content="c", cut=3, dropped=3, n=3).strip(), \
            f"{name}.md renders empty"
    for name in ("compact_step2", "compact_step3"):
        raw = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
        assert "{{room}}" in raw, f"{name}.md does not name the room's number"
    assert "100" in prompts.load("compact_step2", room=100)
    assert prompts.load("compact_step3", room=100, text="SOMETHING").count("SOMETHING") == 1
    assert prompts.load("summary_frame").count("{{summary}}") == 1, \
        "the frame must have exactly one body slot, substituted by hand"
    assert "missing" in prompts.load("summary_cut").lower(), \
        "the cut note does not tell the model it now works on potentially missing information"
    assert "not summarized" in prompts.load("slot_failed", dropped=3)
    act = prompts.load("compact_done")
    assert act and not any(c.isdigit() for c in act), \
        "the act notice carries a number: a token count in model-facing text is a budget the model can act on"


def test_the_agents_file_carries_a_live_host_fact():
    text = prompts.load("agents")
    assert str(dt.datetime.now().year) in text          # the {{datetime}} insert was filled


def test_a_caller_override_wins_over_the_live_fact():
    assert "999 cpus" in prompts.load("agents", **{"sys.cpu": "999"})


def test_no_shipped_prompt_file_has_an_unresolved_known_insert():
    """Every `{{datetime...}}` / `{{sys.*}}` written in a prompt file must be one the parser fills, so
    a typo in prose is caught here rather than shown to a model as literal braces."""
    for md in PROMPT_DIR.glob("*.md"):
        rendered = prompts.load(md.stem)
        for key in ("{{datetime", "{{sys."):
            assert key not in rendered, f"{md.name} names an insert the parser does not fill"


# --- PRM-1: a prompt file serves two readers, and only one is charged ---

def test_the_standing_guidance_holds_no_note_written_for_the_owner():
    """`agents.md` is the file an owner edits to retune the agent — and every sentence of it is sent to
    the model on every request. "Edit this file to set the agent's default behaviour" is documentation
    about a file the model cannot see, costs tokens on each call, and invites a run to go looking for
    something that is not in its grants.

    So the note belongs in the file, where the owner reads it, and must not survive the load.
    """
    text = prompts.load("agents")
    assert "<!--" not in text and "-->" not in text, "a comment reached the model"
    for phrase in ("Edit this file", "default behaviour for your model", "placeholder parser"):
        assert phrase not in text, f"an owner-only note reached the model: {phrase!r}"


def test_a_comment_in_a_prompt_file_never_reaches_the_model(monkeypatch, tmp_path):
    """The mechanism, not just this one file: an owner who annotates a prompt with `<!-- ... -->` has
    written it for the next human, and the loader is what has to know that."""
    (tmp_path / "annotated.md").write_text(
        "Be brief.\n<!-- a note to whoever edits this next: keep it under a paragraph -->\n"
        "Say what you did.\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    got = prompts.load("annotated")
    assert got == "Be brief.\nSay what you did.", repr(got)


def test_a_comment_cannot_hide_a_fact_from_the_operator_either(monkeypatch, tmp_path):
    """Stripping must not become a second renderer that mangles what is left: the inserts outside the
    comment still resolve, and the ones inside it do not leak as braces or as values."""
    (tmp_path / "mixed.md").write_text(
        "Host: {{sys.machine}}.\n<!-- built on {{sys.machine}}, do not send this -->\n",
        encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    got = prompts.load("mixed", **{"sys.machine": "BOX-7"})
    assert got == "Host: BOX-7.", repr(got)


def test_the_model_facing_fact_survives_the_strip():
    """The host line is the reason the file renders at all: a run told what machine it is on can pick a
    path separator and a plausible budget. Deleting the owner note must not take that with it."""
    text = prompts.load("agents")
    assert "cpus" in text and "memory" in text, f"the host fact went with the note: {text!r}"
    assert str(dt.datetime.now().year) in text


def test_a_note_on_its_own_line_leaves_no_blank_line_behind(monkeypatch, tmp_path):
    """Removing a whole-line comment has to take the line, not just its text, or the file grows a blank
    paragraph exactly where an owner once wrote a note — and the paragraph structure the author did
    write is the thing that must survive untouched."""
    (tmp_path / "paragraphs.md").write_text("First.\n\nSecond.\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("paragraphs") == "First.\n\nSecond."
    (tmp_path / "annotated2.md").write_text("First.\n<!-- why this order -->\n\nSecond.\n",
                                            encoding="utf-8")
    assert prompts.load("annotated2") == "First.\n\nSecond."


def test_a_comment_inside_a_line_is_removed_and_the_line_survives(monkeypatch, tmp_path):
    """The other shape: an aside at the end of a sentence. Only the comment goes; the text the model
    needs stays, on the same line."""
    (tmp_path / "inline.md").write_text(
        "Say what you did.<!-- the owner asked for brevity here -->\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("inline") == "Say what you did."


def test_a_broken_insert_in_a_note_cannot_break_the_prompt(monkeypatch, tmp_path):
    """Why stripping runs *before* rendering rather than after. `{{datetime:%Q}}` is not a format the
    clock accepts, and the renderer raises on it -- so an owner who leaves a note mentioning an insert,
    with one mistyped character, would take down every run that loads this file if the note were
    rendered first and removed second. The comment is not sent, so it must also not be evaluated."""
    (tmp_path / "note.md").write_text(
        "Be brief.\n<!-- the date insert is {{datetime:%Q}} until I fix it -->\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("note") == "Be brief."


def test_two_notes_do_not_take_the_text_between_them(monkeypatch, tmp_path):
    """The pattern has to stop at the first `-->`. Greedy matching makes two comments one comment, and
    the guidance between them silently leaves the prompt — a missing instruction is exactly the kind of
    change nobody notices until a run starts misbehaving."""
    (tmp_path / "twonotes.md").write_text(
        "<!-- first note -->\nBe brief.\n<!-- second note -->\nSay what you did.\n",
        encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("twonotes") == "Be brief.\nSay what you did."


def test_a_note_spanning_several_lines_leaves_no_ragged_gap(monkeypatch, tmp_path):
    """A note about *why* something is worded a certain way runs to several lines, which is the shape
    the shipped `agents.md` actually uses. A pass that fails to match it leaves the comment's own
    newlines behind: nothing wrong reaches the model, but the prompt grows ragged, and the whole point
    of taking the line with the comment is that the file's spacing survives the note being removed."""
    (tmp_path / "long.md").write_text(
        "Be brief.\n<!--\n  why: the model rambles on this host,\n  so brevity is a rule not a style.\n"
        "-->\nSay what you did.\n", encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("long") == "Be brief.\nSay what you did."


def test_two_notes_on_one_line_do_not_eat_the_sentence_between(monkeypatch, tmp_path):
    """The two-per-line case for the inline pass, which the separate-line test cannot reach: the
    whole-line pass handles comments that own a line, so a greedy inline pattern only shows its damage
    when both notes sit inside one line. Greedy there deletes the instruction between them, and a
    deleted instruction is invisible until a run starts misbehaving."""
    (tmp_path / "inline2.md").write_text(
        "Be brief.<!-- owner: short --> Say what you did.<!-- owner: past tense -->\n",
        encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("inline2") == "Be brief. Say what you did."


def test_a_note_in_the_middle_of_a_line_does_not_join_the_lines(monkeypatch, tmp_path):
    """The distinction the line anchor earns. A comment sitting at the end of one line, with the next
    instruction on the next line, must not take that newline with it: without the anchor, removal
    glues two sentences into "Be brief.Say what you did." Nothing forbidden appears, and a model can
    usually read through it — which is why it has to be asserted rather than noticed.
    """
    (tmp_path / "joiner.md").write_text("Be brief.<!-- owner note -->\nSay what you did.\n",
                                        encoding="utf-8")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    assert prompts.load("joiner") == "Be brief.\nSay what you did."
