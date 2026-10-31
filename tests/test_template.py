"""The tiny placeholder parser. The guarantees worth pinning: known inserts fill, the datetime can
be formatted, an unknown insert is left untouched (never eaten), and a real doc full of them renders
with no unresolved known braces left behind."""
import datetime as dt
import pathlib

from sah import prompts, template

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_sys_shell_reports_the_shell_a_run_actually_gets():
    """A template that says `cmd.exe` while commands are being handed to Git Bash teaches the model the
    wrong syntax on purpose, from a document it trusts. The insert therefore reports the resolved shell,
    the same object the argv and the tool description come from.
    """
    from sah.tools import shell
    assert template.render("{{sys.shell}}") == shell.SHELL.name


def test_known_sys_facts_fill():
    out = template.render("os={{sys.os}} arch={{sys.machine}} cpus={{sys.cpu}}")
    assert "{{" not in out and "=" in out
    assert "os= cpu" not in out            # facts came back non-empty on the test host


def test_datetime_expands_to_iso_and_a_format_pattern():
    fixed = {"datetime": dt.datetime(2026, 3, 5, 14, 7, 9, tzinfo=dt.timezone.utc)}
    assert template.render("{{datetime}}", fixed).startswith("2026-03-05T14:07:09")
    assert template.render("{{datetime:%Y-%m-%d}}", fixed) == "2026-03-05"


def test_a_format_pattern_with_colons_survives():
    fixed = {"datetime": dt.datetime(2026, 1, 2, 9, 30, tzinfo=dt.timezone.utc)}
    assert template.render("{{datetime:%H:%M}}", fixed) == "09:30"


def test_an_unknown_insert_is_left_exactly_as_written():
    # Prose/code may contain braces this parser was never asked to touch; eating them would corrupt.
    assert template.render("json: {{not_a_fact}} stays") == "json: {{not_a_fact}} stays"


def test_format_on_a_non_date_is_ignored_not_an_error():
    out = template.render("{{sys.os:nonsense}}")
    assert "nonsense" not in out and "{{" not in out   # expanded, format ignored


def test_context_overrides_a_fact():
    assert template.render("{{sys.host}}", {"sys.host": "testbox"}) == "testbox"


def test_render_file_reads_and_expands(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("on {{sys.system}} at {{datetime:%Y}}", encoding="utf-8")
    out = template.render_file(str(p))
    assert "{{" not in out and str(dt.datetime.now().year) in out


def test_the_shipped_prompt_has_no_unresolved_known_inserts():
    """`sah/prompts/agents.md` is the file whose `{{...}}` inserts are really filled — the prompt
    loader renders it at load — so a typo in an insert name would reach the model as a literal
    brace. This checks the shipped copy, not a fixture, because the bug is a misspelled key in
    prose nobody executes.

    The assertion is on what the model receives, which since PRM-1 is the *stripped* text: a note to
    the owner sits in a comment and is never sent, so holding it to the insert rules would fail on
    prose the model cannot see. The precondition still reads the file, because "this prompt stopped
    shipping inserts" is a fact about the file."""
    text = (REPO / "sah" / "prompts" / "agents.md").read_text(encoding="utf-8")
    assert "{{" in text, "the prompt stopped shipping inserts; this test has nothing to check"
    loaded = prompts.load("agents")
    for key in ("{{datetime", "{{sys."):
        assert key not in loaded, f"the prompt names an insert `{key}` the parser does not fill"


def test_the_capacity_probe_is_consulted_once_per_prompt_load(monkeypatch):
    """The host fact for RAM asked the platform twice for one number -- once for the condition, once for the
    value. Counted rather than asserted by value: the value never changed, so no value assertion can catch a
    second syscall, and a fact read twice is also a fact that can disagree with itself mid-sentence."""
    calls = []
    monkeypatch.setattr(template, "_total_ram_gib", lambda: calls.append(1) or 128)
    assert template.facts()["sys.ram"] == "128 GiB"
    assert len(calls) == 1, f"the probe was consulted {len(calls)} times for one prompt load"


def test_an_unreadable_capacity_says_unknown_rather_than_nothing(monkeypatch):
    """The fallback is the whole reason the helper exists, so it is asserted where the probe is wrapped --
    otherwise a mutant that returns "None GiB" or drops the branch survives."""
    monkeypatch.setattr(template, "_total_ram_gib", lambda: None)
    assert template.facts()["sys.ram"] == "unknown"
