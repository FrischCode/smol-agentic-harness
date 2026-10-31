"""The terminal client. Two things are asserted here because they are the two the client exists
to honour: a verdict cannot be scrolled past, and the client decides nothing."""
import io
import pathlib
import sys

import pytest


from sah import cli, control, openai, prompts, render
from sah.agent import Agent, Verdict
from sah.confine import Confine
from sah.model import Caps, Fake, turn_text, turn_tool
from sah.record import Log


def explode(text: str) -> str:
    """Return the text."""
    raise RuntimeError("the tool died")


def echo(text: str) -> str:
    """Return the text."""
    return text


def verbose(text: str) -> str:
    """Fail at length."""
    raise RuntimeError("y" * 4000)


# Doubles that take text declare that they touch no place, which is a declaration like any other, and
# say what they do, which the loop reads for every tool it is given.
explode.access, explode.address = "none", None
echo.access, echo.address = "none", None
verbose.access, verbose.address = "none", None
explode.description = "Return the text."
echo.description = "Return the text."
verbose.description = "Fail at length."


def build(turns, root, policy=None, tools=None):
    # `attended`: the sessions these tests drive through `cli.repl` have a person at the prompt,
    # so prose completes; a constructed Agent's own default is the unattended contract.
    log = Log(root / "log.jsonl")
    backend = Fake(turns, caps=Caps(context_tokens=100, model="test-1"))
    return Agent(backend=backend, tools=tools if tools is not None else {}, log=log, root=root,
                 policy=policy or (lambda name, arguments, **_: Verdict(True)), attended=True)


def test_text_streams_to_the_terminal(tmp_path):
    """Live text must reach the terminal as it arrives. This asserted through `cli.stream`, a hand-painted
    shim that nothing in the shipped path called — so it could stay green while the projection `build`
    actually subscribes streamed nothing. It now asserts through `render.Terminal`, which is what the
    client wires, and the shim is gone."""
    out, err = io.StringIO(), io.StringIO()
    agent = build([turn_text("hel", "lo")], tmp_path)
    terminal = render.Terminal(out, io.StringIO(), "normal", color=False)
    agent.subscribe_event(terminal.on_event)
    cli.repl(agent, iter(["go", "quit"]), out, err)
    assert "hello" in out.getvalue(), out.getvalue()


def test_the_floor_prints_a_refusal_even_when_the_model_talks_over_it(tmp_path):
    """The model answering cheerfully after a refusal is the case where a client hides the
    refusal, so the verdict goes to stderr regardless of what else was displayed."""
    agent = build([turn_tool("c1", "echo", {"text": "x"}), turn_text("all done, nothing happened")
                   ], tmp_path,
                  policy=lambda name, arguments, **_: Verdict(False, "no writing in this run"),
                  tools={"echo": echo})
    err = io.StringIO()
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)     # build wires the floor now, not repl
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    assert "operation.outcome=refused — no writing in this run" in err.getvalue()
    assert "all done" in str(agent.final_text)


def test_a_clean_run_announces_nothing(tmp_path):
    err = io.StringIO()
    cli.repl(build([turn_text("fine")], tmp_path), iter(["go", "quit"]), io.StringIO(), err)
    assert "outcome" not in err.getvalue() and ".ok=" not in err.getvalue()


def test_a_successful_tool_call_announces_nothing(tmp_path):
    """A tool that worked is a non-event on the floor's channel (stderr = needs attention). This is
    the case that hid a real bug: `result.ok` is a bool, and matching True against the string CLEAN
    set let every success print `result.ok=True`, so refusals scrolled past their own noise."""
    agent = build([turn_tool("c1", "echo", {"text": "hi"}), turn_text("done")], tmp_path,
                  tools={"echo": echo})
    err = io.StringIO()
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    assert ".ok=" not in err.getvalue(), "a green tool call is not a verdict needing attention"


def test_a_failed_tool_call_is_loud_on_the_floor(tmp_path):
    """The mirror: ok=False IS something to miss, so it must reach stderr even with no renderer."""
    def bad(text: str) -> str:
        raise RuntimeError("the tool broke")
    bad.access, bad.address = "none", None
    bad.description = "Raise to fail."
    agent = build([turn_tool("c1", "bad", {"text": "x"}), turn_text("sorry")], tmp_path,
                  tools={"bad": bad})
    err = io.StringIO()
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    assert "result.ok=False" in err.getvalue()


def test_a_verdict_is_printed_once(tmp_path):
    agent = build([turn_text("x")], tmp_path)
    agent.run("go")
    err = io.StringIO()
    floor = cli.Floor(agent.log, err)
    assert floor.flush() == []
    agent.run("go again")
    floor.flush()  # drain the report for "go again" so the next flush comes back empty
    assert floor.flush() == [], "the floor reports what it has not reported, not everything again"


def test_a_tool_that_dies_is_surfaced_by_result_not_by_prose(tmp_path):
    """`result.ok` is a declared verdict field, so a crash reaches the operator even though the
    model goes on to summarise the failure in its own words."""
    agent = build([turn_tool("c1", "explode", {"text": "x"}), turn_text("that failed, sorry")],
                  tmp_path, tools={"explode": explode})
    err = io.StringIO()
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    assert "result.ok=False — explode raised RuntimeError: the tool died" in err.getvalue()


def test_an_allowed_operation_is_not_announced(tmp_path):
    """The floor is for verdicts that need attention: printing every permitted call trains the
    operator to scroll past it, which is how refusals get hidden again."""
    agent = build([turn_tool("c1", "echo", {"text": "x"}), turn_text("done")], tmp_path,
                  tools={"echo": echo})
    err = io.StringIO()
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    assert "operation.outcome" not in err.getvalue()
    assert any(s["kind"] == "operation" for s in agent.log.steps()), "it is still recorded"


def test_long_failure_text_is_not_dumped_on_the_operator(tmp_path):
    agent = build([turn_tool("c1", "verbose", {"text": "x"}), turn_text("sorry")], tmp_path,
                  tools={"verbose": verbose})
    err = io.StringIO()
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)
    cli.repl(agent, iter(["go", "quit"]), io.StringIO(), err)
    line = [l for l in err.getvalue().splitlines() if l.startswith("result.ok")][0]
    assert len(line) < 260 and line.endswith("…")


def test_a_run_that_ends_badly_says_so(tmp_path):
    """A Fake runs out of script on the second request: the run failed, and the client reads
    that from the log rather than from a return value."""
    err = io.StringIO()
    cli.repl(build([turn_text("first")], tmp_path), iter(["go", "again", "quit"]),
             io.StringIO(), err)
    assert "run ended: failed" in err.getvalue()


def test_an_empty_line_starts_nothing(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    cli.repl(build([turn_text("fine")], tmp_path), iter(["", "quit"]), out, err)
    assert "empty line" in err.getvalue()
    assert out.getvalue() == "", "an empty line must not reach the model"


def test_quit_leaves_the_process_alone(tmp_path):
    agent = build([turn_text("unreached")], tmp_path)
    cli.repl(agent, iter(["quit"]), io.StringIO(), io.StringIO())
    assert agent.log.steps() == [] or all(s["kind"] != "started" for s in agent.log.steps())


# --- the other half of the CLI: talking to someone else's run -------------------------

def test_send_writes_one_block_and_reports_where_it_went(tmp_path, monkeypatch):
    monkeypatch.delenv("SAH_USER", raising=False)
    out, err = io.StringIO(), io.StringIO()
    code = cli.build(["send", "--run-dir", str(tmp_path), "--from", "ada", "steer",
                      "the other file"], out, err)
    assert code == 0
    assert "queued" in out.getvalue() and str(tmp_path) in out.getvalue()
    block = control.pending(tmp_path)[0]
    assert (block.action, block.text, block.sender) == ("steer", "the other file", "ada")


def test_send_defaults_to_a_steer_and_takes_the_identity_from_the_environment(tmp_path,
                                                                              monkeypatch):
    monkeypatch.setenv("SAH_USER", "ada")
    cli.build(["send", "--run-dir", str(tmp_path)], io.StringIO(), io.StringIO())
    block = control.pending(tmp_path)[0]
    assert (block.action, block.text, block.sender) == ("steer", "", "ada")


def test_send_refuses_to_write_a_command_nobody_signed(tmp_path, monkeypatch):
    monkeypatch.delenv("SAH_USER", raising=False)
    err = io.StringIO()
    code = cli.build(["send", "--run-dir", str(tmp_path), "interrupt"], io.StringIO(), err)
    assert code == 2
    assert control.pending(tmp_path) == [], "an anonymous command is not written at all"


def test_send_carries_a_cancel_and_a_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("SAH_USER", "ada")
    cli.build(["send", "--run-dir", str(tmp_path), "--cancel", "--prompt", "urgent",
               "interrupt"], io.StringIO(), io.StringIO())
    block = control.pending(tmp_path)[0]
    assert block.action == "interrupt" and block.cancel is True and block.prompt == "urgent"


def test_a_verdict_is_surfaced_when_it_lands_not_when_the_run_ends(tmp_path):
    """A live run dumped every refusal in a block after the model had finished talking. A verdict
    that arrives after the prose has scrolled past is a verdict that arrived too late to act on,
    so the floor is called as each step is written."""
    agent = build([turn_tool("c1", "echo", {"text": "x"}), turn_text("done")], tmp_path,
                  policy=lambda name, arguments, **_: Verdict(False, "no calls in this run"),
                  tools={"echo": echo})
    err = io.StringIO()
    # The floor is wired through the one registration path (it no longer subscribes itself), and it
    # is wired BEFORE the probe below so it prints first — that ordering is the thing under test.
    cli.ext.assemble([cli.Floor(agent.log, err)], agent.log)
    seen = []
    agent.log.subscribe(lambda step: seen.append((step["kind"], "refused" in err.getvalue())))
    agent.run("go")
    assert seen[0] == ("started", False)
    assert ("operation", True) in seen, "the refusal was printed as it was recorded"
    assert seen[-1] == ("ended", True)


def test_a_client_attached_later_is_not_silently_blind(tmp_path):
    agent = build([turn_tool("c1", "echo", {"text": "x"}), turn_text("done")], tmp_path,
                  policy=lambda name, arguments, **_: Verdict(False, "no calls in this run"),
                  tools={"echo": echo})
    agent.run("go")                       # no client attached yet: nothing was announced
    err = io.StringIO()
    floor = cli.Floor(agent.log, err)
    assert any(line.startswith("operation.outcome=refused") for line in floor.flush())


def test_the_console_s_codepage_does_not_lose_what_the_model_said(tmp_path):
    """A Windows console inherits a codepage that cannot hold an em dash and substitutes a box.
    The stream is fixed instead of the text, because a client that mangles the run's words is not
    a projection of it."""
    handle = open(tmp_path / "out.txt", "w", encoding="cp1252")
    cli.use_utf8(handle)
    handle.write("Iron gate holds fast—")
    handle.close()
    assert "fast—" in (tmp_path / "out.txt").read_text(encoding="utf-8")


def test_a_capture_buffer_is_left_alone(tmp_path):
    cli.use_utf8(io.StringIO())           # no reconfigure to be had; nothing raised


# --- the system prompt: the model is shown the live bounds --------------------

def _sit(**over):
    kw = dict(run_id="r", root="/runs/r", model="m", host="h", port=1,
              policy=Confine(read_roots=["/datadir"], write_roots=[], allow_exec=False), memory="")
    kw.update(over)
    return cli.situation(**kw)


def test_the_default_prompt_names_the_harness_and_the_situation_carries_the_stop_rule():
    """A model must be told what it is, and how its run ends. The stop rule is a per-run fact —
    attendance decides whether prose completes — so it lives in the situation, not the static
    preamble: a preamble that stated one rule would be silent about the other shape of run."""
    assert "S.A.H" in cli.PREAMBLE and "Smol Agentic Harness" in cli.PREAMBLE
    assert "calls no tool" not in cli.PREAMBLE


def test_the_situation_shows_what_the_model_cannot_infer_from_its_tools():
    """Grants, exec, and memory are state the tool list does not carry. A model left to guess them
    reaches for scratch and hallucinated memory folders — the exact live failure this replaced."""
    s = _sit()
    assert "datadir" in s                        # read root survives resolution
    assert "may not run shell commands" in s
    assert "no memory tree" in s


def test_the_situation_reflects_exec_and_memory_when_present():
    s = _sit(policy=Confine(read_roots=["/rd"], write_roots=["/wroot"], allow_exec=True), memory="memroot")
    assert "may run shell commands" in s
    assert "memory tree" in s and "memroot" in s
    assert "wroot" in s                           # write root shown too


def test_a_refusal_is_explained_as_unganted_not_impossible():
    """The model must know a refusal means 'not granted, tell the user', not 'retry harder' — the
    line that stops the retry-the-identical-call loop a bare 'refused' provokes. The standing
    judgment is stated once, in the file world; the situation carries only the per-run consequence.
    """
    agents = prompts.load("agents")
    assert "refused" in agents and "not granted" in agents


def test_the_situation_names_the_shell_the_exec_grant_actually_gives():
    """A grant to "run shell commands" does not say what will parse them. A model that assumes POSIX on a
    cmd.exe box -- or cmd syntax on Git Bash -- spends turns discovering the answer, so the situation
    carries the same resolved shell the argv is built from, with what it means for syntax. And a run with
    no exec grant is told no shell, because naming one would be a lie about its bounds.
    """
    from sah.tools import shell
    s = _sit(policy=Confine(read_roots=["/rd"], write_roots=[], allow_exec=True))
    assert shell.SHELL.name in s and shell.SHELL.hint in s
    assert shell.SHELL.name not in _sit(), "no shell is named to a run that cannot run one"


def test_a_custom_system_replaces_the_preamble_but_never_the_situation():
    """--system is verbatim for the author's voice, but dropping the run's real bounds to honor it
    would hide from the model the constraints it is being asked to respect."""
    s = cli.assemble_system("You are a pirate.", run_id="r", root="/r", model="m", host="h",
                            port=1, policy=Confine(read_roots=["/xroot"]), memory="")
    assert s.startswith("You are a pirate.")
    assert "Smol Agentic Harness" not in s          # generic preamble gone (name is spelled S.A.H,
                                                    # so the phrase is what marks it)
    assert "<situation>" in s and "xroot" in s   # live bounds still present


def test_the_default_system_is_preamble_then_situation_then_agents():
    """Three stacked parts in order: the author's voice, the live bounds the model cannot infer, then
    the editable standing guidance. The bounds must sit before the guidance so nothing in the guidance
    can contradict what the run is actually allowed to touch."""
    s = cli.assemble_system("", run_id="r", root="/r", model="m", host="h", port=1,
                            policy=Confine(read_roots=["/yroot"]), memory="")
    assert s.startswith("You are S.A.H")
    assert "<situation>" in s and "yroot" in s
    assert s.rstrip().endswith(prompts.load("agents"))          # standing guidance appended last
    assert s.index("<situation>") < s.index(prompts.load("agents"))


def test_the_default_prompt_has_no_unfilled_inserts():
    """The agents file carries {{...}} host inserts; by the time it reaches the model they must be
    real values, not literal braces the model would read as instructions."""
    s = cli.assemble_system("", run_id="r", root="/r", model="m", host="h", port=1,
                            policy=Confine(read_roots=["/r"]), memory="")
    assert "{{" not in s


# --- the tool surface a real build advertises ---------------------------------

def _captured_agent(monkeypatch, argv):
    """Run cli.build far enough to wire everything, but stop at repl so no stdin is read and no
    backend is called; return the Agent it built."""
    box = {}
    monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: box.setdefault("agent", agent) or 0)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
    cli.build(argv, io.StringIO(), io.StringIO())
    return box["agent"]


def _names(agent):
    return {spec.name for spec in agent.specs}


def test_spawn_is_advertised_to_the_model(tmp_path, monkeypatch):
    """The bug this guards: spawn was registered AFTER the Agent derived its tool schemas, so the
    model was correctly told it had no spawn tool. A tool that exists but is not advertised is a
    tool that does not exist as far as the model is concerned."""
    agent = _captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)])
    assert "spawn" in _names(agent)


def test_arithmetic_and_the_clock_are_offered_without_a_flag(tmp_path, monkeypatch):
    """Neither tool touches a place, so neither has a flag and neither waits on a grant. The run that
    needs them is an ordinary one: without `calc` a run with no shell has to do its sums in prose, and a
    run resumed over a stale prompt cannot tell that the date in its instructions has gone stale.

    Pinned at the build rather than only in the tool modules because the failure is registration order —
    the same shape as the spawn bug above, where a tool joined the table after the schemas had been
    derived and the model was correctly told the tool did not exist.
    """
    names = _names(_captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)]))
    assert {"read", "write", "ls", "calc", "clock"} <= names


def test_the_declaration_tool_is_offered_only_to_a_run_no_operator_watches(tmp_path, monkeypatch):
    """The completion declaration belongs to the run whose exit code is the only signal its caller
    gets: an attended run completes in words, and a tool named for ending the run beside a
    conversation is a foot a person never needs. Withholding has to reach the schema, the way the
    exec tools' do — a model that never sees it cannot call it. `--ask-timeout` models the attended
    run: a promise to answer from somewhere else puts an operator in the seat."""
    plain = _names(_captured_agent(monkeypatch,
                                   ["run", "--run-dir", str(tmp_path), "--ask-timeout", "5"]))
    assert "verify_objective" not in plain
    unattended = _names(_captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)]))
    assert "verify_objective" in unattended


def test_a_run_that_may_not_spawn_is_not_offered_the_child_tools(tmp_path, monkeypatch):
    """Withholding a capability has to reach the schema, not just the policy: a model told about `spawn`
    that then finds it refused wastes a turn, and a model that never sees it can plan around it. The same
    rule the exec tools follow -- absent from the list, not present and gated."""
    names = _names(_captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path), "--no-spawn"]))
    for tool in ("spawn", "child_status", "child_wait"):
        assert tool not in names, f"{tool} offered to a run that was told it may not start children"


# --- the extension door: a file the operator names ------------------------------

_DEMO_EXTENSION = str(pathlib.Path(__file__).resolve().parent.parent / "extensions" / "speak.py")


def _build_captured(monkeypatch, argv):
    """Like `_captured_agent`, but keeps the console handles and the exit code, so the boot box,
    the record and the refusal path can all be read back."""
    out, err = io.StringIO(), io.StringIO()
    box = {}

    def _stop(agent, *a, **k):
        box["agent"] = agent
        return 0

    monkeypatch.setattr(cli, "repl", _stop)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
    code = cli.build(argv, out, err)
    return code, box.get("agent"), out.getvalue(), err.getvalue()


def test_a_named_extension_file_is_advertised_to_the_model(tmp_path, monkeypatch):
    """The demo travels the whole road a third party would: named on the command line, loaded at
    boot, offered in the schema. A file the operator named that the model is not told about is a
    tool that does not exist as far as the model is concerned."""
    agent = _captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path),
                                          "--extension", _DEMO_EXTENSION])
    assert "speak" in _names(agent)


def test_the_boot_box_names_a_loaded_extension_and_the_model_is_told(tmp_path, monkeypatch):
    """What is in force is printed: the banner states the file and the tools it brings, and the
    model is told the same thing in its <situation>, so no run acts on a capability the operator
    never saw opened."""
    code, agent, out, _err = _build_captured(monkeypatch,
                                             ["run", "--run-dir", str(tmp_path),
                                              "--view", "normal",
                                              "--extension", _DEMO_EXTENSION])
    assert code == 0
    assert "extension:" in out and "speak" in out, "the banner states what the invocation opened"
    assert "an extension is loaded from" in agent.system


def test_a_loaded_extension_is_recorded_beside_the_log(tmp_path, monkeypatch):
    """A later reader of a degenerate run sees who else was in the tool table, from the record
    alone."""
    _code, agent, _out, _err = _build_captured(monkeypatch,
                                               ["run", "--run-dir", str(tmp_path),
                                                "--extension", _DEMO_EXTENSION])
    steps, _ = agent.log.read()
    notices = [s for s in steps if s["kind"] == "notice" and s.get("subject") == "extension"]
    assert any("speak" in s["text"] for s in notices)


def test_a_loaded_extensions_findings_are_recorded_beside_its_load_line(tmp_path, monkeypatch):
    """A boot fact a loaded file found is recorded beside the load line it came from, under the
    core's own subject and the core's stamp: the extension hands the fact up, the record holds it,
    and a later reader of a degenerate run sees it from the record alone."""
    finding_file = tmp_path / "finding_ext.py"
    finding_file.write_text(
        "class FindingDemo:\n"
        "    tools = ()\n"
        "    findings = ('the demo found a fact at boot',)\n"
        "EXTENSION = FindingDemo()\n",
        encoding="utf-8")
    _code, agent, _out, _err = _build_captured(monkeypatch,
                                               ["run", "--run-dir", str(tmp_path),
                                                "--extension", str(finding_file)])
    steps, _ = agent.log.read()
    notices = [s for s in steps if s["kind"] == "notice"]
    assert any(s.get("subject") == "extension" and "finding_ext.py" in s["text"]
               for s in notices), "the load line is the record's"
    assert any(s.get("subject") == "extension" and s.get("text") == "the demo found a fact at boot"
               for s in notices), "the fact rides beside its load line, under the core's own subject"


def test_a_malformed_finding_is_a_named_refusal_at_boot(tmp_path, monkeypatch):
    """A finding that does not carry — not a non-empty string stating a fact — is a named boot refusal
    the way a bad tool is: a named line, exit 2, no run built. Not a garbage step in the record, and
    not a traceback the operator has to decode."""
    bad_file = tmp_path / "bad_finding.py"
    bad_file.write_text(
        "class BadFinding:\n"
        "    tools = ()\n"
        "    findings = (123,)\n"
        "EXTENSION = BadFinding()\n",
        encoding="utf-8")
    code, agent, _out, err = _build_captured(monkeypatch,
                                             ["run", "--run-dir", str(tmp_path),
                                              "--extension", str(bad_file)])
    assert code == 2 and agent is None
    assert "finding" in err


def test_a_named_extensions_transient_line_reaches_the_console_and_not_the_record(tmp_path, monkeypatch):
    """The whole door end to end, the way a run wires it: a file named by the operator, its
    watcher's `status.say` rendered by the console the run builds — the line on the console's face,
    and absent from the record, which holds the fact the line points at (the step it is about, not
    the word)."""
    chatty = tmp_path / "chatty_ext.py"
    chatty.write_text(
        "class Chatty:\n"
        "    tools = ()\n"
        "    def on(self, step):\n"
        "        self.status.say('a transient line from a named file')\n"
        "EXTENSION = Chatty()\n",
        encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    box = {}

    def _stop(agent, *a, **k):
        box["agent"] = agent
        return 0

    monkeypatch.setattr(cli, "repl", _stop)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
    code = cli.build(["run", "--run-dir", str(tmp_path), "--view", "normal",
                      "--extension", str(chatty)], out, err)
    agent = box["agent"]
    assert code == 0 and agent is not None
    agent.log.append("message", "model", role="assistant", text="a step after boot")
    text = out.getvalue()
    assert "… a transient line from a named file" in text, \
        "the console renders the line"
    assert text.index("… a transient line from a named file") < text.index("a step after boot"), \
        "the word precedes its rendering: it lands before the box of the step it is about, " \
        "not inside or after it"
    steps, _ = agent.log.read()
    assert not any(s.get("kind") == "notice"
                   and "a transient line from a named file" in str(s.get("text", "")) for s in steps), \
        "and the record does not hold the word: the line is a word, not a fact"


def test_a_missing_extension_file_is_a_named_refusal_with_nothing_running(tmp_path, monkeypatch):
    """The refusal is the operator's own command-line mistake, answered the way the rest are: a
    named line, exit 2, no run built, nothing written."""
    code, agent, _out, err = _build_captured(monkeypatch,
                                             ["run", "--run-dir", str(tmp_path),
                                              "--extension", str(tmp_path / "never_written.py")])
    assert code == 2 and agent is None
    assert "no such file" in err


def test_an_extension_file_whose_tool_clashes_with_a_builtin_is_refused(tmp_path, monkeypatch):
    """The gate is on the registration, not on who registered: a file that provides `clock` claims
    a name the built-ins hold, and the run boots with a named line instead of one tool meaning two
    things."""
    bad = tmp_path / "clash.py"
    bad.write_text("def clock() -> str:\n    return 'x'\n"
                   "clock.access, clock.address = 'none', None\n"
                   "clock.description = 'A clash.'\n"
                   "class E:\n    tools = (clock,)\nEXTENSION = E()\n", encoding="utf-8")
    code, agent, _out, err = _build_captured(monkeypatch,
                                             ["run", "--run-dir", str(tmp_path),
                                              "--extension", str(bad)])
    assert code == 2 and agent is None
    assert "two extensions both provide" in err


def test_an_env_named_extension_is_loaded_when_the_flag_names_none(tmp_path, monkeypatch):
    """The machine's .env names a file the way it names the backend (SAH_EXTENSION): the operator's
    own machine config, stated in the boot box like anything else. A run that names none on the
    command line gets it."""
    monkeypatch.setenv("SAH_EXTENSION", _DEMO_EXTENSION)
    agent = _captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)])
    assert "speak" in _names(agent)


def test_the_flag_outranks_the_env_named_extension(tmp_path, monkeypatch):
    """Precedence is the standard one: a value the live invocation sets wins over the machine's
    default. An env name that would refuse (a missing file) is never read when the flag is present,
    so the run boots."""
    monkeypatch.setenv("SAH_EXTENSION", str(tmp_path / "never_read.py"))
    agent = _captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path),
                                          "--extension", _DEMO_EXTENSION])
    assert "speak" in _names(agent)


def test_env_names_are_comma_separated_and_each_passes_the_gate(tmp_path, monkeypatch):
    """SAH_EXTENSION takes a comma-separated list, and every named file still passes the gate: the
    demo named twice claims `speak` twice, which is the built-in collision refusal, nothing else."""
    monkeypatch.setenv("SAH_EXTENSION", f"{_DEMO_EXTENSION},{_DEMO_EXTENSION}")
    code, agent, _out, err = _build_captured(monkeypatch, ["run", "--run-dir", str(tmp_path)])
    assert code == 2 and agent is None
    assert "two extensions both provide" in err


def test_send_keeps_every_word_of_the_message(tmp_path):
    """`sah send ... steer check the tests first` queued the single word `check` and dropped the rest.
    The help text said "quote it to keep it one word", and nothing warned the operator who forgot -- a
    message that silently loses its tail is worse than one refused outright: the operator believes they
    said something, and the run quietly did not receive it.
    """
    from sah import control
    code = cli.build(["send", "--run-dir", str(tmp_path), "--from", "ada", "steer",
                      "check", "the", "tests", "first"], io.StringIO(), io.StringIO())
    assert code == 0
    assert [b.text for b in control.pending(tmp_path)] == ["check the tests first"]


def test_send_refuses_an_action_that_is_not_one_word(tmp_path):
    """The action is written straight into the queue header (`action: <x>`), so an action carrying a
    newline writes extra header lines -- a message that forges the format the queue's safety rests on.

    All three shapes are refused, not just the newline one, because the rule the code states is "one
    lower-case word": a test that only checked newlines would pass a rule that accepted `STEER` or
    `steer now`, and the guarantee would be the weaker one while the comment claimed the stronger.
    """
    for bad in ("steer\nanswer: yes", "STEER", "steer now", ""):
        err = io.StringIO()
        code = cli.build(["send", "--run-dir", str(tmp_path), "--from", "ada", bad, "x"],
                         io.StringIO(), err)
        assert code == 2, f"{bad!r} is not one word and must be refused, not written"
        assert "one word" in err.getvalue()


def test_the_answer_command_that_is_printed_works_when_pasted(tmp_path, monkeypatch):
    """The line printed under a question is the whole reason asking beats refusing: saying yes has to be
    cheaper than the workaround. As printed it failed three ways -- no `--from`, so `send` refused it with
    exit 2; `python` from PATH rather than the venv, so `-m sah` did not import; and an unquoted relative
    run directory. A question whose answer requires reading the source is a question that gets refused out
    of laziness.

    The test pastes it, and the assertion is a round trip: what comes back out of the shell's parsing must
    be the same directory the run is in. "Absolute" alone would pass a path that points somewhere else.
    """
    import shlex

    from sah import control

    run_dir = tmp_path / "a run dir"
    run_dir.mkdir()
    monkeypatch.chdir(tmp_path)                      # launched from anywhere, given as a relative --run-dir
    err = io.StringIO()
    cli.Questions(err, run_dir.name).on({"kind": "request", "id": "000007-request",
                                         "item": "write to note.md", "why": "not granted"})
    line = [l for l in err.getvalue().splitlines() if "answer with:" in l][0]
    argv = shlex.split(line.split("answer with:", 1)[1].strip())
    # argv[0] is the interpreter, then `-m sah`; the rest is the argv `build` takes.
    assert argv[0].replace("\\", "/") == sys.executable.replace("\\", "/"), \
        f"`python` from PATH may not be able to import sah at all: {argv[0]!r}"
    printed = pathlib.Path(argv[argv.index("--run-dir") + 1])
    assert printed.as_posix() == run_dir.absolute().as_posix(), \
        f"what parses back is not this run: {printed} != {run_dir.absolute()}"
    assert "--from" in argv, "send refuses an answer that names nobody"
    code = cli.build(argv[3:], io.StringIO(), io.StringIO())
    assert code == 0, f"the printed command does not work when pasted: {argv}"
    blocks = control.pending(str(run_dir))
    assert [b.action for b in blocks] == ["answer"] and blocks[0].reply == "000007-request"


def test_a_run_dir_at_a_filesystem_root_still_prints_a_pasteable_path(tmp_path):
    """Why the printed path uses slashes. A root is what `abspath` answers with a trailing separator
    (`C:\\`), and inside double quotes a trailing backslash escapes the closing quote. Measured on this
    host: `--run-dir "D:\\db\\"` reached python's argv as `D:\\db" --from ada` -- the quote escaped and the
    next argument was swallowed whole. Forward slashes are accepted by every shell on every OS and by
    Python on Windows, so they cannot fail this way.
    """
    import shlex

    root = pathlib.Path(sys.executable).anchor        # 'C:\\' on Windows, '/' on POSIX
    err = io.StringIO()
    cli.Questions(err, root).on({"kind": "request", "id": "000001-request", "item": "x", "why": ""})
    line = [l for l in err.getvalue().splitlines() if "answer with:" in l][0]
    argv = shlex.split(line.split("answer with:", 1)[1].strip())
    assert argv[argv.index("--run-dir") + 1].rstrip("/\\") == root.rstrip("/\\"), \
        f"the path did not survive the quotes: {argv}"
    assert "--from" in argv and "answer" in argv, \
        f"an escaped quote swallowed the arguments after the path: {argv}"


def test_a_resumed_run_does_not_spawn_a_second_child_into_the_firsts_folder(tmp_path, monkeypatch):
    """The child id came from a counter held in the process, so a resumed run minted `child1` again: the new
    child appended to the first child's log, and a status read answered with the OLD run's `ended` step --
    the parent acted on a result that was never its child's.

    `Spawner.next_id` is tested on its own, and this test still mattered: replacing the binding in `build`
    with a counter passes every other test in the battery (measured), because no other test resumes.
    """
    import pathlib

    from sah import model
    from sah import spawn as spawn_mod
    from sah.model import Caps, turn_text, turn_tool

    made = []

    class P:
        pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(spawn_mod, "subprocess_spawn",
                        lambda argv, root, api_key="": (made.append(root), P())[1])

    def go(resume: bool) -> int:
        """One one-shot run whose single act is a spawn call. A fresh run and a resumed one, same folder."""
        script = [turn_tool("c1", "spawn", {"objective": "work", "model": "", "max_turns": 0}),
                  turn_text("done")]
        monkeypatch.setattr(cli, "OpenAI", lambda **kw: model.Fake(script, caps=Caps(10_000)))
        # The window must hold the resident load (system prompt + the full tool surface, ~2.8k
        # tokens): a window smaller than the resident load is one the fit check declares blocked,
        # which is the compaction contract's job to test, not this one — this test is about the
        # child id the spawner mints across a resume.
        argv = ["run", "--run-dir", str(tmp_path), "--task", "spawn one child", "--ask-timeout",
                "5", "--no-color"]
        if resume:
            argv.append("--resume")
        return cli.build(argv, io.StringIO(), io.StringIO())

    assert go(resume=False) == 0
    assert go(resume=True) == 0
    names = [pathlib.Path(root).name for root in made]
    assert len(names) == 2, f"expected two children across the two runs, got {names}"
    assert names[0] != names[1], f"the resumed run reused {names[0]}, so one folder holds two children's logs"
    assert names[1].endswith("-child2"), names


def test_the_environment_veto_withholds_the_child_tools_too(tmp_path, monkeypatch):
    """A child started by a shell command line the model wrote carries no flag, so the environment is the
    other half of the bound. Narrowing from the environment is allowed where widening never is: a variable
    can only take a capability away, never hand one out."""
    monkeypatch.setenv("SAH_NO_SPAWN", "1")
    names = _names(_captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)]))
    assert "spawn" not in names and "child_status" not in names


def test_the_boot_box_shows_the_spawn_veto_the_run_actually_has(tmp_path, monkeypatch):
    """The operator reads this block to decide whether to widen the run, so a box announcing `spawn: on`
    for a run whose child tools were withheld is the same failure as a receipt naming a backup that was
    never written: a claim the code never checked. The box is suppressed for a pipe, so the test captures
    the header call instead of pretending to have a terminal.
    """
    box = {}

    class Term:
        def __init__(self, *a, **k):
            pass

        def header(self, lines):
            box["lines"] = list(lines)

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(cli.render, "Terminal", Term)
    monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: 0)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))

    cli.build(["run", "--run-dir", str(tmp_path)], io.StringIO(), io.StringIO())
    assert any(l.startswith("spawn: on") for l in box["lines"]), box["lines"]
    box.clear()
    cli.build(["run", "--run-dir", str(tmp_path), "--no-spawn"], io.StringIO(), io.StringIO())
    assert any(l.startswith("spawn: off") for l in box["lines"]), box["lines"]


def test_every_tool_a_run_is_offered_declares_its_access_truthfully(tmp_path, monkeypatch):
    """The battery cannot see a third-party extension, so two checks cover declaration: `ext.assemble`
    refuses an undeclared tool at boot, and this walks the tools a real build actually assembles and asks
    whether the declaration matches the signature.

    The rule is a name heuristic on purpose: a parameter called `path`, `url`, `cwd`, `folder` or `*_dir`
    is a place, and a tool that touches a place may not declare `none`. That is the shape of the hole this
    closes -- a `def cat(path)` declared `none` reads as harmless and is not.
    """
    import inspect

    from sah import contract as T

    PLACE = ("path", "file", "dir", "folder", "cwd", "url", "root", "subpath")
    agent = _captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path), "--exec",
                                         "--memory", str(tmp_path / "mem"),
                                         "--runs-root", str(tmp_path / "runs"), "--web"])
    assert len(agent.tools) >= 10, f"the build gave almost nothing to check: {sorted(agent.tools)}"
    for name, func in sorted(agent.tools.items()):
        access, address = T.declaration(func)             # raises if missing or invalid
        params = inspect.signature(func).parameters
        marks_place = [p for p in params if p in PLACE or p.endswith("_path") or p.endswith("_dir")]
        if access == "none":
            assert not marks_place, f"{name} touches {marks_place} but declares no place"
        elif address is not None:
            assert address in params, f"{name} declares address={address}, not a parameter"
        if marks_place:
            assert address in marks_place, (
                f"{name} names a place ({marks_place}) but declares address={address!r}; the policy "
                f"records and judges what it is told, so an address that is not the place is a log "
                f"that lies")


def test_the_situation_says_whether_children_may_be_started():
    """A model that is not told the bound reaches for a tool it does not have, or reports a capability the
    run lacks. The situation states both ways, so the absence is explained rather than discovered."""
    off = _sit(spawn=False)
    assert "spawn" in off.lower()
    assert "cannot start child runs" in off
    assert "cannot start child runs" not in _sit(), "a run that may spawn was told it may not"


def test_the_situation_states_the_completion_contract_both_ways():
    """The stop rule is per-run: attendance decides whether prose completes. The situation states
    it both ways, so a model never works under a completion rule nobody told it — the unattended
    run is told the declaration it is offered, the attended run the prose contract it gets."""
    attended = _sit(attended=True)
    assert "read as finished" in attended
    assert "verify_objective" not in attended
    unattended = _sit()
    assert "read as finished" not in unattended
    # The unattended contract is pinned whole: it is the owner's sentence, and a paraphrase is a
    # different contract — the declaration the model is told to make is not one the harness may
    # soften.
    assert ("EVERY run ends with verify_objective. No exceptions. After your final response, "
            "call verify_objective. Do not stop with a text-only response — the harness will "
            "re-prompt you. This is not optional; it is the only way to end the run."
            in unattended)


def test_exec_tools_only_appear_with_the_exec_flag(tmp_path, monkeypatch):
    plain = _names(_captured_agent(monkeypatch, ["run", "--run-dir", str(tmp_path)]))
    assert "bash" not in plain
    withexec = _names(_captured_agent(monkeypatch,
                                      ["run", "--run-dir", str(tmp_path), "--exec"]))
    assert "bash" in withexec


def test_memory_tools_only_appear_with_a_memory_root(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    names = _names(_captured_agent(monkeypatch,
                                   ["run", "--run-dir", str(tmp_path), "--memory", str(mem)]))
    assert {"mem_tree", "mem_read", "mem_grep", "mem_note"} <= names


def test_the_advertised_schema_carries_field_meanings(tmp_path, monkeypatch):
    """The whole point of the visibility fix, asserted end-to-end through a real build: a model
    reading the tool list sees that mem_note.mode is a closed set, not a bare string it must guess."""
    agent = _captured_agent(monkeypatch,
                            ["run", "--run-dir", str(tmp_path), "--memory", str(tmp_path / "m")])
    note = next(s for s in agent.specs if s.name == "mem_note")
    assert note.parameters["properties"]["mode"]["enum"] == ["append", "create", "overwrite"]
    assert note.parameters["properties"]["path"]["description"]


def test_a_bare_positional_after_run_is_the_objective():
    """`sah run "say hi"` used to parse the positional into `rest`, never read it, and open the
    interactive prompt -- the run looked hung when it had just dropped its task on the floor.

    The middle assertion here read `rest=["ignored"]` and expected the word to disappear, which named the
    defect in the fixture's own vocabulary. Nothing is ignored now: `--task` plus further words is one
    objective, because that is the shape `sah.bat` hands over on every multi-word prompt. The old
    expectation was not a competing spec, it was the bug, so it is rewritten rather than kept."""
    import argparse
    assert cli._objective(argparse.Namespace(task="", rest=["say", "hi"])) == "say hi"
    assert cli._objective(argparse.Namespace(task="explicit", rest=["and", "more"])) == "explicit and more"
    assert cli._objective(argparse.Namespace(task="", rest=[])) == ""     # nothing -> interactive


def _seeded_runs(monkeypatch, tmp_path, extra):
    """Drive cli.build with a task and capture what reaches run() vs repl().

    repl() is replaced with one that drains its `lines` iterator, so a seed turn and a typed turn
    flow through the same path and this sees the seed as repl's first line -- not a separate run()."""
    seen = {}
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
    monkeypatch.setattr(cli.Agent, "run", lambda self, t: seen.setdefault("runs", []).append(t))

    def fake_repl(agent, lines, out, err):
        seen["repl"] = True
        seen["first_line"] = next(iter(lines), None)     # the seed, if one was handed in
        return 0

    monkeypatch.setattr(cli, "repl", fake_repl)
    cli.build(["run", "--run-dir", str(tmp_path), "--task", "do a thing", *extra],
              io.StringIO(), io.StringIO())
    return seen


def test_a_one_shot_task_runs_once_and_exits(tmp_path, monkeypatch):
    """The default --task behavior a spawned child relies on: run it, return, never open a prompt.
    If this regressed to interactive, an unattended child would block forever on a keyboard it has."""
    seen = _seeded_runs(monkeypatch, tmp_path, [])
    assert seen.get("runs") == ["do a thing"]         # ran the task once
    assert "repl" not in seen                          # never opened an interactive prompt


def test_interactive_runs_the_seed_then_stays_open(tmp_path, monkeypatch):
    """`agent "task" -i` is the product shape: run the objective, then keep the prompt. The seed must
    reach repl as its first line (one run path), not as a separate one-shot run() before the loop."""
    seen = _seeded_runs(monkeypatch, tmp_path, ["-i"])
    assert seen.get("repl") is True                    # it went to the REPL, not a bare one-shot run
    assert seen.get("first_line") == "do a thing"      # the objective seeds the loop's first read
    assert "runs" not in seen                          # build did NOT run it out-of-band; repl owns it


def test_default_view_follows_the_witness_not_task_presence():
    """A human at a console (output is a TTY) gets boxes even for a one-shot; output redirected to
    a file or pipe -- a spawned child's sink -- has no live reader and stays off."""
    assert cli._default_view(True) == "normal"
    assert cli._default_view(False) == "off"


def _wire_renderer(agent, out):
    """Attach a Terminal the way build() does: on(step) via the log, on_event via the bus."""
    terminal = cli.render.Terminal(out, io.StringIO(), cli.render.VIEWS["normal"], color=False)
    agent.log.subscribe(cli.render.RenderExtension(terminal).on)
    agent.subscribe_event(terminal.on_event)
    return terminal


def test_the_repl_adds_no_blank_line_of_its_own(tmp_path):
    """The renderer owns block spacing (a leading blank, none trailing). repl used to also
    print a blank after each turn, which showed as TWO blank lines after a turn's verdict.
    The client is a projection; it must not add spacing the renderer already owns."""
    out = io.StringIO()
    agent = build([turn_text("first."), turn_text("second.")], tmp_path)
    _wire_renderer(agent, out)
    cli.repl(agent, iter(["go", "again", "quit"]), out, io.StringIO())
    text = out.getvalue()
    assert "\n\n\n" not in text, "two consecutive blank lines -- a second spacing owner crept in"


def test_turns_are_separated_by_exactly_one_blank(tmp_path):
    """A verdict ends one turn, the next turn opens a box: between them there must be exactly
    one blank line, not zero (glued) and not two (double-spaced)."""
    out = io.StringIO()
    agent = build([turn_text("one."), turn_text("two.")], tmp_path)
    _wire_renderer(agent, out)
    cli.repl(agent, iter(["a", "b", "quit"]), out, io.StringIO())
    lines = out.getvalue().split("\n")
    verdicts = [i for i, l in enumerate(lines) if l.startswith("◆")]
    assert verdicts, "no verdict line rendered"
    for v in verdicts:
        # whatever follows a verdict is a blank line, and the line after that is NOT blank
        if v + 2 < len(lines):
            assert lines[v + 1] == "" and lines[v + 2] != "", \
                f"expected exactly one blank after verdict at line {v}"


def test_the_floor_says_when_part_of_the_record_is_missing(tmp_path):
    """The floor is the surface no display level may silence, so it is the surface that has to say
    a step could not be read: an operator shown nothing is an operator being told the record was
    fine."""
    agent = build([turn_text("x")], tmp_path)
    agent.run("go")
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"ended","id":"000099-end')            # a writer died mid-append
    err = io.StringIO()
    lines = cli.Floor(agent.log, err).flush()
    assert any("log damage" in line for line in lines), lines


def test_exporting_a_damaged_log_says_so(tmp_path):
    """`sah export` writes the account a person reads afterwards. If a step could not be parsed,
    the note travels with the Markdown; otherwise the file reads as a complete history of a run
    whose history is incomplete."""
    agent = build([turn_text("x")], tmp_path)
    agent.run("go")
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"ended","id":"000099-end')
    out, err = io.StringIO(), io.StringIO()
    rc = cli.build(["export", "--run-dir", str(tmp_path)], out, err)
    assert rc == 0, "a damaged log is still worth exporting"
    assert out.getvalue().strip(), "the projection still rendered"
    assert "log damage" in err.getvalue(), err.getvalue()


# --- the exit code a script can trust ------------------------------------------

def _one_shot(monkeypatch, tmp_path, turns, argv=()):
    """A real one-shot through `build`: scripted backend, no repl, no stdin. Returns the code the
    process would exit with."""
    monkeypatch.setattr(cli, "OpenAI",
                        lambda **k: Fake(turns, caps=Caps(context_tokens=100, model="t")))
    return cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "go",
                      "--view", "off", *argv], io.StringIO(), io.StringIO())


def test_the_situation_says_what_a_repeat_costs_in_this_run():
    """The standing refusal judgment lives in the prompt files; the situation carries only the
    per-run consequence of the last resort, because that is the one part that differs between an
    attended and an unattended run. The model is the one deciding whether to repeat, so the
    sentence has to describe the run it is in.
    """
    assert "last resort" in prompts.load("agents")
    attended = _sit(attended=True)
    unattended = _sit()
    assert "has an operator" in attended and "pauses until they answer" in attended
    assert "has no operator" in unattended and "ends the run as failed" in unattended
    assert "pauses until they answer" not in unattended, \
        "a run with nobody to answer promised the model that someone would be asked"


def _booted(monkeypatch, tmp_path, argv=()):
    """A one-shot whose boot box is captured. `out` is not a tty, which is exactly the state of a piped
    run and of every spawned child -- so this is the unattended case unless a flag says otherwise. The
    view is asked for explicitly because the default follows the witness, and with no witness there would
    be no box to read.
    """
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(context_tokens=100, model="t")))
    out = io.StringIO()
    cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "go", "--view", "normal",
               *argv], out, io.StringIO())
    return out.getvalue()


def test_a_piped_one_shot_says_asks_end_the_run(monkeypatch, tmp_path):
    """The run states out loud what an ask will do, because the operator is the one who has to decide
    whether that is acceptable before relying on the answer.
    """
    assert "asks: end the run" in _booted(monkeypatch, tmp_path)


def test_passing_a_timeout_is_a_promise_to_answer(monkeypatch, tmp_path):
    """`--ask-timeout` is how an operator says 'someone is watching even though my output is piped' --
    the ask check of the opt-in live suite (kept out of the shipped tree on purpose) runs a piped
    one-shot and answers from outside, and could not work otherwise. It has to be visible in the
    boot box, since it changes what the run does with a refusal.
    """
    booted = _booted(monkeypatch, tmp_path, ["--ask-timeout", "7"])
    assert "asks: wait 7s" in booted


def test_a_child_is_not_handed_a_timeout_it_cannot_use(monkeypatch, tmp_path):
    """An explicit timeout means 'attended', so a spawner that passed one would raise a child from
    failing honestly on a needed address to waiting minutes for a voice that does not exist. The default
    is None for that reason, and the argv built for a child must not set it.
    """
    from sah import spawn as spawn_mod

    spawner = spawn_mod.Spawner(runs_root=str(tmp_path), model="m", host="h", port=1)
    argv = spawner.build_command("child1", "do the thing")
    assert "--ask-timeout" not in argv


def test_a_clean_one_shot_exits_zero(monkeypatch, tmp_path):
    """The exit code is the only signal a run no operator is watching hands its caller: 0 means the
    run declared completion. Prose alone is not a declaration — a one-shot that says `done` in words
    is nudged, bounded, and ends 1, so a script chained on the exit code cannot be fooled by a model
    that talks instead of acts."""
    assert _one_shot(monkeypatch, tmp_path, [turn_tool("c", "verify_objective", {})]) == 0
    assert _one_shot(monkeypatch, tmp_path / "prose", [turn_text("done")]) == 1


def test_a_failed_one_shot_exits_one_so_a_script_can_tell(monkeypatch, tmp_path):
    """`sah-call "x" && deploy` is the entire reason an exit code exists. The run drew `failed
    (exit 1)` on screen and the process still returned 0 — invisible to precisely the caller that
    has no screen: a spawn, a cron line, a `&&`."""
    assert _one_shot(monkeypatch, tmp_path, []) == 1      # out of script: the run failed


def test_an_unattended_one_shot_leaves_its_report_in_the_folder(tmp_path, monkeypatch):
    """The reader arriving after the fact should not have to know `sah export` exists: a run nobody
    is watching leaves the projection in its own folder next to the record, under a header that
    keeps the hierarchy legible — the log is the record, the file is the reading of it."""
    assert _one_shot(monkeypatch, tmp_path, [turn_tool("c", "verify_objective", {})]) == 0
    report = tmp_path / "run" / "report.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert text.startswith("> rendered from log.jsonl · ")
    assert "the log is the record" in text
    assert "verify_objective" in text
    assert "ended: complete (exit 0)" in text


def test_an_attended_one_shot_leaves_no_report_in_the_folder(tmp_path, monkeypatch):
    """Attendance, not task presence, gates the report: with an operator reachable, the live view
    is the report and the file is the operator's to ask for — so an attended one-shot that completes
    the way an attended run completes (prose) still leaves the folder to the log alone."""
    assert _one_shot(monkeypatch, tmp_path, [turn_text("done")],
                     argv=("--ask-timeout", "5")) == 0
    assert not (tmp_path / "run" / "report.md").exists()


def test_a_crashed_one_shot_still_leaves_its_report_in_the_folder(tmp_path, monkeypatch):
    """The bad run's report is the one a caller most wants, and the crash path is the one that
    used to leave nothing: the catch-all logs the ending and re-raises, and the report is written
    on the way out — so the projection holds the crash notice and the crashed ending beside the
    record they came from, and the traceback still reaches the operator."""
    class WireDeath:
        """Raises where a backend should answer: the class the crash guard was written for."""

        def __init__(self):
            self.caps = Caps(context_tokens=100, model="t")

        def capabilities(self):
            return self.caps

        def complete(self, *, system, messages, tools, thinking="off", max_tokens=None):
            raise RuntimeError("the wire died")

    monkeypatch.setattr(cli, "OpenAI", lambda **k: WireDeath())
    # `cli.build` directly, not `_one_shot`: the helper re-pins its own scripted backend and would
    # replace the wire death with a quiet one. Same argv shape, same path through the one-shot
    # branch — the crash reaches the `finally` that writes the report and keeps on propagating.
    with pytest.raises(RuntimeError, match="the wire died"):
        cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "go", "--view", "off"],
                  io.StringIO(), io.StringIO())
    report = tmp_path / "run" / "report.md"
    assert report.is_file(), "the crashed run left no report for the caller arriving after it"
    text = report.read_text(encoding="utf-8")
    assert "the wire died" in text
    assert "ended: crashed (exit 1)" in text


def test_the_code_comes_from_the_record_and_not_from_a_variable(monkeypatch, tmp_path):
    """The client asks the log what happened — the same question every later reader asks — so the
    number a shell sees and the number `sah recall` shows cannot disagree."""
    code = _one_shot(monkeypatch, tmp_path, [])
    steps, _ = Log(tmp_path / "run" / "log.jsonl").read()
    ended = [s for s in steps if s["kind"] == "ended"][-1]
    assert ended["reason"] == "failed" and code == ended["exit_code"] == 1


def test_a_run_that_reached_no_end_is_not_reported_as_success(monkeypatch, tmp_path):
    """A loop that dies before its closing step leaves no `ended` at all. Absent is not the same as
    zero: nothing in the record says the run went well, so nothing may claim it did."""
    monkeypatch.setattr(cli, "OpenAI",
                        lambda **k: Fake([turn_text("x")], caps=Caps(context_tokens=100, model="t")))
    monkeypatch.setattr(cli.Agent, "run", lambda self, task: None)   # a run that recorded no end
    assert cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "go", "--view", "off"],
                     io.StringIO(), io.StringIO()) == 1


def test_a_session_still_exits_zero_after_an_objective_ended_badly(tmp_path):
    """The other half of the decision. A session holds many objectives and one ending badly is not
    the session failing; exiting 1 mid-REPL would tell a shell the interactive run never happened."""
    err = io.StringIO()
    assert cli.repl(build([turn_text("first")], tmp_path), iter(["go", "again", "quit"]),
                    io.StringIO(), err) == 0


def test_a_signal_stops_the_turn_and_keeps_the_session(tmp_path):
    """Ctrl+C stops a turn; it does not take the session down with a traceback.

    The usual REPL contract, and the reason the loop rather than the terminal owns the signal: the
    operator typed into a session, and a session that dies mid-objective loses the prompt they were
    about to use. The turn's log still records the stop (`run` writes the `ended` step), so the next
    objective begins from a record that says what happened."""
    def boom(text: str) -> str:
        """The signal arrives while this tool runs."""
        raise KeyboardInterrupt

    boom.access, boom.address = "none", None
    boom.description = "Take the signal while running."

    agent = build([turn_tool("c1", "boom", {"text": "x"}), turn_text("still here")], tmp_path,
                  tools={"boom": boom})
    err = io.StringIO()
    assert cli.repl(agent, iter(["go", "again", "quit"]), io.StringIO(), err) == 0
    assert "interrupted" in err.getvalue(), "the operator is told the turn stopped, not shown a traceback"
    assert "still here" in str(agent.final_text), "the session went on to the next objective"


def test_the_knobs_reach_the_agent_that_will_act_on_them(tmp_path, monkeypatch):
    """`--think` and `--summary-think` are parsed for two different consumers: the loop reads the level,
    compaction reads the summaries-only level. A build that parses a flag and never hands it to the Agent
    is indistinguishable from one that never parsed it, so this reads the constructor arguments of the
    Agent a real build makes.

    `none` has to arrive as None, not as the string: the operator's spelling and the loop's own
    "no opinion" must agree on the wire, or a spawned child's `--think none` reasons and the same
    preference expressed by omitting the flag does not. The summary knob is absent by default: a run
    that names no level for summaries keeps its own, and a constructor that invented one would be the
    mutation that overrides the level A8 leaves alone.
    """
    seen = {}

    class Watch(cli.Agent):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "Agent", Watch)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))
    monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: 0)

    cli.build(["run", "--run-dir", str(tmp_path / "a")], io.StringIO(), io.StringIO())
    assert seen["thinking"] is None, \
        "the default run expresses no preference; `off` names a level a strict dialect has never heard of"
    assert seen.get("summary_thinking") is None, \
        "no level named for summaries: the run keeps its own, and the default must be that absence"

    seen.clear()
    cli.build(["run", "--run-dir", str(tmp_path / "b"), "--think", "none"], io.StringIO(), io.StringIO())
    assert seen["thinking"] is None, "`--think none` did not become the loop's no-opinion"

    seen.clear()
    cli.build(["run", "--run-dir", str(tmp_path / "c"), "--think", "xhigh",
               "--summary-think", "low"], io.StringIO(), io.StringIO())
    assert seen["thinking"] == "xhigh" and seen["summary_thinking"] == "low"


def test_the_streaming_and_usage_knobs_reach_the_agent_and_the_backend(tmp_path, monkeypatch):
    """Two consumers of the same flags: the loop's record stamps the knobs on `started`, and the
    backend composes the wire from them. A build that parsed the flag and handed it to neither is
    indistinguishable from one that never parsed it, so both constructor argument sets are read off
    a real build — on by default, off when the operator says so."""
    seen = {}
    wire = {}

    class Watch(cli.Agent):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "Agent", Watch)
    monkeypatch.setattr(cli, "OpenAI",
                        lambda **k: (wire.update(k), Fake([], caps=Caps(100)))[1])
    monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: 0)

    cli.build(["run", "--run-dir", str(tmp_path / "a")], io.StringIO(), io.StringIO())
    assert seen["streaming"] is True and seen["usage"] is True
    assert wire["streaming"] is True and wire["request_usage"] is True

    seen.clear()
    wire.clear()
    cli.build(["run", "--run-dir", str(tmp_path / "b"), "--no-streaming", "--no-usage"],
              io.StringIO(), io.StringIO())
    assert seen["streaming"] is False and seen["usage"] is False
    assert wire["streaming"] is False and wire["request_usage"] is False


def test_a_run_shows_the_compaction_word_only_when_it_is_not_the_default(tmp_path, monkeypatch):
    """The boot box is where an operator checks what the run will do. Naming the fallback only when it
    differs from the default keeps the box honest without adding a line every run pays for -- and an
    override that shows up nowhere is the silent-ignore failure this session already met once with shells.
    """
    box = {}

    class Term:
        def __init__(self, *a, **k):
            pass

        def header(self, lines):
            box["lines"] = list(lines)

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(cli.render, "Terminal", Term)
    monkeypatch.setattr(cli, "repl", lambda agent, *a, **k: 0)
    monkeypatch.setattr(cli, "OpenAI", lambda **k: Fake([], caps=Caps(100)))

    cli.build(["run", "--run-dir", str(tmp_path)], io.StringIO(), io.StringIO())
    assert not any("summary think=" in l for l in box["lines"]), box["lines"]
    box.clear()
    cli.build(["run", "--run-dir", str(tmp_path), "--summary-think", "low"], io.StringIO(), io.StringIO())
    assert any("summary think=low" in l for l in box["lines"]), box["lines"]


def test_the_base_url_reaches_the_backend_it_was_named_for(tmp_path, monkeypatch):
    """A flag the client parses and drops is a flag that lies in `--help`. Captured at the seam the
    backend is built, because the whole point of a base URL is where the conversation goes."""
    seen = {}
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: (seen.update(kw), Fake([], caps=Caps(100)))[1])
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path),
               "--base-url", "https://gw.example.com/llama/v1"],
              io.StringIO(), io.StringIO())
    assert seen.get("base_url") == "https://gw.example.com/llama/v1", seen


def test_a_credential_leaving_in_clear_is_said_once_and_a_backend_that_says_nothing_is_not_a_crash():
    """The warning is a client act with two failure modes, and both are asserted: silent exposure, and a
    check that assumes every backend knows about transport.

    The second is not hypothetical — the first version asked for the attribute directly and every run
    against a scripted backend died on an `AttributeError`, which is the particular way a security notice
    becomes a crash on the path that exercises the harness most."""
    remote = openai.OpenAI(model="m", base_url="http://box.example.com/v1", api_key="sk-secret")
    err = io.StringIO()
    cli.warn_cleartext(remote, err)
    assert "clear text" in err.getvalue() and "https" in err.getvalue(), err.getvalue()

    err = io.StringIO()
    cli.warn_cleartext(openai.OpenAI(model="m", base_url="https://box.example.com/v1",
                                     api_key="sk-secret"), err)
    cli.warn_cleartext(openai.OpenAI(model="m", host="127.0.0.1", port=9090, api_key="sk-secret"), err)
    cli.warn_cleartext(Fake([], caps=Caps(100)), err)          # a backend with no opinion about transport
    assert err.getvalue() == "", f"warned where nothing was exposed, or crashed: {err.getvalue()!r}"


def test_a_run_started_against_a_cleartext_endpoint_says_so_before_it_starts(tmp_path, monkeypatch):
    """The unit test of the warning proves what it prints; it cannot prove the run ever asks. Deleting
    the call from the command left every one of those assertions green, which is the difference between a
    feature and a function nobody invokes.

    Driven through the real constructor with only the socket replaced, so the decision, the wording and
    the channel are the ones an operator would see."""
    class Response:
        def read(self, *_):
            return b"data: [DONE]\n\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(openai._OPENER, "open", lambda request, *a, **k: Response())
    err = io.StringIO()
    cli.build(["run", "--run-dir", str(tmp_path), "--task", "go", "--no-color", "--max-turns", "1",
               "--base-url", "http://box.example.com/v1", "--api-key", "sk-secret"],
              io.StringIO(), err)
    assert "clear text" in err.getvalue(), f"the run said nothing about the credential:\n{err.getvalue()}"


def test_the_thinking_and_vision_switches_reach_the_backend(tmp_path, monkeypatch):
    """Both knobs existed on the constructor and neither was reachable: a switch nobody can set is not a
    capability, it is a default with a comment. `--vision` in particular declares a server that can see,
    which is the precondition for any future image route and is false unless the operator can say it."""
    seen = {}
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: (seen.update(kw), Fake([], caps=Caps(100)))[1])
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path / "a"),
               "--preserve-thinking", "--vision"], io.StringIO(), io.StringIO())
    assert seen.get("preserve_thinking") == "true" and seen.get("supports_vision") is True, seen
    seen.clear()
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path / "b"),
               "--preserve-thinking", "false"], io.StringIO(), io.StringIO())
    assert seen.get("preserve_thinking") == "false", seen
    seen.clear()
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path / "c")], io.StringIO(), io.StringIO())
    assert seen.get("preserve_thinking") == "none", seen


def test_the_reasoning_wire_switches_reach_the_backend(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: (seen.update(kw), Fake([], caps=Caps(100)))[1])
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path / "a"),
               "--require-reasoning-content", "--reasoning-out", "reasoning"],
              io.StringIO(), io.StringIO())
    assert seen.get("require_reasoning_content") is True, seen
    assert seen.get("reasoning_out") == "reasoning", seen
    seen.clear()
    cli.build(["run", "--task", "go", "--run-dir", str(tmp_path / "b")], io.StringIO(), io.StringIO())
    assert seen.get("require_reasoning_content") is False and seen.get("reasoning_out") is None, seen


def test_the_resume_line_names_the_unit_it_counts(tmp_path, monkeypatch):
    """`resumed <run>: N turns back` printed a message count under the name "turns". A line about history
    that misnames its unit teaches the operator to discount the number, and it was the number the operator
    needed to decide whether the right run had come back."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("hi")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                     "--resume"],
                     io.StringIO(), err) == 0
    line = [l for l in err.getvalue().splitlines() if l.startswith("resumed ")]
    assert line, f"the resume line never printed: {err.getvalue()!r}"
    assert "turns back" not in line[0], line[0]
    assert "messages" in line[0], line[0]


def test_a_system_file_that_is_not_there_is_a_message_and_not_a_traceback(tmp_path):
    """A prompt file named on the command line and absent from the disk is a mistake in a command line.
    The harness answers mistakes with `sah: ...` and exit 2 — as it does for an answer with no `--reply`,
    for instance — but this path opened the file bare, so the operator got a Python traceback whose only
    mention of the offending argument was a path in an `open()` frame."""
    err = io.StringIO()
    code = cli.build(["run", "--run-dir", str(tmp_path), "--task", "x",
                      "--system-file", str(tmp_path / "nope.md")], io.StringIO(), err)
    assert code == 2, f"exit {code}: a command-line mistake must be distinguishable from a crash"
    assert err.getvalue().startswith("sah:"), err.getvalue()
    assert "system-file" in err.getvalue() and "nope.md" in err.getvalue(), err.getvalue()


def test_every_word_the_operator_typed_reaches_the_run(tmp_path, monkeypatch):
    """The front door passes `--task %*`, so `sah "rename these"` arrives as `--task rename these` — and
    the second word was dropped silently. The sh sibling joins them, so the two launchers disagreed about
    what one sentence meant, and the word that went missing was the operator's, in the one place a run is
    not allowed to guess: its objective.

    Checked against the log rather than the parsed namespace, because the loss this describes happens
    between the two."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "rename", "--ask-timeout", "5", "these",
                     "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    said = " ".join(s.get("text", "") for s in Log(tmp_path / "log.jsonl").steps()
                    if s["kind"] == "message")
    assert "rename these" in said, f"the run recorded an objective of: {said!r}"


# --- what a run was allowed to touch, recorded and re-stated on resume -----------------------------
#
# A resumed run does not inherit reach: permission comes from the live invocation, never from data.
# That rule was previously unfalsifiable, because the reach itself was never recorded — `started` declares
# a `grants` field and nothing wrote it, so the question "did the run I am continuing have a shell?" had no
# answer in the log or anywhere else. Measured before this: run 1 printed `exec: granted (Git Bash)`, the
# resume printed no such line, and neither the operator nor the model was told the shell had gone. The
# model's side is worse than silence: it gets back a conversation full of `bash` calls and a schema with no
# `bash`, which reads as a capability that broke rather than one that was never re-granted.

def _started(tmp_path) -> dict:
    """The last `started` step: the reach of the most recent run in this folder."""
    return [s for s in Log(tmp_path / "log.jsonl").steps() if s["kind"] == "started"][-1]


def _reach(tmp_path, *flags):
    """One refused-free run into `tmp_path` with the given extra flags, returning stderr."""
    from sah.openai import Fake as Fake  # the Fake the module-level monkeypatch uses
    err = io.StringIO()
    rc = cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                   *flags],
                   io.StringIO(), err)
    assert rc == 0, err.getvalue()
    return err


def test_a_run_records_which_folders_it_may_touch_and_whether_it_has_a_shell(tmp_path, monkeypatch):
    """The grant axes are the two that matter — which folders, and whether there is a shell — so a log that
    cannot answer them cannot be audited, and cannot tell a later reader that reach changed."""
    from sah.confine import resolve

    read_dir, write_dir = tmp_path / "notes", tmp_path / "out"
    read_dir.mkdir()
    write_dir.mkdir()
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                      "--grant", str(read_dir), "--write", str(write_dir), "--exec"],
                     io.StringIO(), io.StringIO()) == 0
    grants = _started(tmp_path).get("grants")
    assert grants is not None, "the run recorded no reach at all"
    assert resolve(str(read_dir)) in grants["read"] and resolve(str(write_dir)) in grants["write"], grants
    assert grants["exec"] is True, grants


def test_a_run_without_exec_records_that_it_has_none(tmp_path, monkeypatch):
    """Recording only the widening would make a plain run look identical to an old log that recorded
    nothing, which is the ambiguity the resume line has to resolve."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    grants = _started(tmp_path).get("grants")
    assert grants is not None and grants["exec"] is False, grants


def test_a_resume_that_lost_a_grant_says_so(tmp_path, monkeypatch):
    """The scenario: a run given `--exec` to do shell work, resumed later without it. The conversation comes
    back full of shell results, the tool is gone, and nothing said either fact. A resumed run is not
    entitled to a shell it was not given again — that part is correct — but the operator must be told the
    rule fired, naming the flag that would restore it."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                     "--exec"],
                     io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                     "--resume"],
                     io.StringIO(), err) == 0
    text = err.getvalue()
    assert "--exec" in text, f"the resume said nothing about the shell it did not restore: {text!r}"
    # The direction is asserted too: a line that names the lost flag but calls it an addition would still
    # contain "--exec", and would tell the operator the opposite of what happened.
    assert "does not grant" in text.lower() and "adds" not in text.lower(), text


def test_a_resume_that_widened_reach_says_so_too(tmp_path, monkeypatch):
    """Losing reach mid-task is confusing; gaining it quietly is worse. Both directions get a line, because
    the resume is exactly the moment nobody is reading flags."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                      "--exec", "--resume"], io.StringIO(), err) == 0
    text = err.getvalue()
    assert "--exec" in text and "reach" in text.lower(), f"the widening printed nothing: {text!r}"
    assert "adds" in text.lower(), f"a widening described as a loss reads as a warning about safety: {text!r}"


def test_a_resume_that_changes_nothing_says_nothing_about_reach(tmp_path, monkeypatch):
    """A warning that fires every resume is a line the operator stops reading. Silence means unchanged."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    for task in ("one", "two"):
        err = io.StringIO()
        assert cli.build(["run", "--run-dir", str(tmp_path), "--task", task, "--ask-timeout", "5",
                          "--no-color", "--exec",
                          *(["--resume"] if task == "two" else [])], io.StringIO(), err) == 0
        assert "reach" not in err.getvalue().lower(), f"{task}: {err.getvalue()!r}"


def test_a_resume_into_an_older_log_says_the_earlier_reach_is_unrecorded(tmp_path, monkeypatch):
    """Logs written before reach was recorded cannot be compared, and silence there would recreate the exact
    failure: an operator who reads nothing concludes nothing changed."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    log = Log(tmp_path / "log.jsonl")
    steps = log.steps()
    stripped = [{k: v for k, v in s.items() if k != "grants"} for s in steps]
    (tmp_path / "log.jsonl").write_text("".join(__import__("json").dumps(s) + "\n" for s in stripped),
                                        encoding="utf-8")
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                     "--resume"],
                     io.StringIO(), err) == 0
    text = err.getvalue()
    # Asserted as facts, not as a word: the line must say the comparison could not be made and must still
    # name what this invocation grants. An earlier draft grepped for "unrecorded", which the sentence says
    # as "recorded no grants" — a test pinned to one phrasing punishes better prose.
    assert "no grants" in text.lower() and "compare" in text.lower(), \
        f"a log with no recorded reach printed nothing: {text!r}"
    assert "--read" in text, f"the line did not name the reach that IS in force: {text!r}"


def test_a_run_does_not_report_its_own_folder_as_a_grant(tmp_path, monkeypatch):
    """A run can always read inside its own run directory — that is standing ownership, not a grant, and the
    policy keeps the two apart for exactly that reason. Reported as a grant it would make every run look
    self-widening, and a reader comparing two runs' reach would see a difference where the operator gave
    nothing."""
    from sah.confine import resolve

    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    run_dir = tmp_path / "r"
    assert cli.build(["run", "--run-dir", str(run_dir), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    grants = [s for s in Log(run_dir / "log.jsonl").steps() if s["kind"] == "started"][-1]["grants"]
    assert resolve(str(run_dir)) not in grants["read"], \
        f"standing ownership was recorded as a grant: {grants}"


def test_a_resume_compares_against_the_run_it_continues(tmp_path, monkeypatch):
    """A folder's runs can differ in reach, and the comparison is against the most recent one — that is the
    session being continued — not whatever the first run in the folder was given.

    The narrower run gets here by resuming: dropping a second run into an existing folder is now refused, so
    this is the only route that leaves two recorded reaches in one log."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                     "--exec"],
                     io.StringIO(), io.StringIO()) == 0
    # A resume that does not re-grant exec: reach narrowed, and that is the last reach the folder records.
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                     "--resume"],
                     io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "three", "--ask-timeout", "5",
                     "--no-color", "--resume"],
                     io.StringIO(), err) == 0
    assert "reach" not in err.getvalue().lower(), \
        f"compared against the wrong run: {err.getvalue()!r}"


# --- one folder per run: the merge is not a state you can enter by accident ------------------------
#
# `--run-dir` defaulted to `runs/scratch`, and the README quickstart runs without `--run-dir`, so the
# documented first command, run twice, silently merged two runs into one log. Measured: one log, two
# `started`/`ended` pairs, both objectives inside, while `recall` indexed the folder as ONE run carrying the
# first objective and the last verdict; and a resume of it printed "2 messages back" — two consecutive user
# turns with nothing between them, as if the first had never been answered.
#
# The shape below — a fresh folder per run — makes the ambiguity unconstructible, so no refusal is
# needed on the main path and no new flag exists. The refusal that remains is for the case the doors
# cannot generate — naming a folder that already holds a run.

def log_obj(path):
    from sah.record import read
    steps, _lost = read(path)
    return steps


def _fake(monkeypatch):
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))


def test_two_runs_with_no_folder_named_do_not_share_a_log(tmp_path, monkeypatch):
    """The documented invocation, twice. Each run must own its folder and its history."""
    _fake(monkeypatch)
    monkeypatch.setenv("SAH_RUNS", str(tmp_path / "runs"))
    for task in ("count the pdfs", "rename the pdfs"):
        assert cli.build(["run", "--task", task, "--ask-timeout", "5",
                         "--no-color"], io.StringIO(), io.StringIO()) == 0
    logs = sorted((tmp_path / "runs").glob("*/log.jsonl"))
    assert len(logs) == 2, f"expected two run folders, found: {[p.parent.name for p in logs]}"
    for log in logs:
        objectives = [s.get("text") for s in log_obj(log)
                      if s["kind"] == "message" and s.get("role") == "user"]
        assert len(objectives) == 1, \
            f"one log carries {len(objectives)} objectives: {objectives}"


def test_the_minted_folder_name_orders_by_the_clock_it_claims(tmp_path, monkeypatch):
    """The name is how a later reader finds the newest run — the door's `-r` will read it off a directory
    listing rather than from a pointer file — so the stamp may not be locale-shaped. Fixed-format UTC basic
    ISO, asserted here rather than in a launcher, because a shell script cannot be executed by the battery."""
    import re
    import time as _time

    _fake(monkeypatch)
    root = tmp_path / "runs"
    monkeypatch.setenv("SAH_RUNS", str(root))
    assert cli.build(["run", "--task", "one", "--ask-timeout", "5",
                     "--no-color"], io.StringIO(), io.StringIO()) == 0
    name = root.iterdir().__next__().name
    assert re.fullmatch(r"r\d{8}-\d{6}-[0-9a-f]{6}", name), f"unexpected run folder name: {name!r}"
    _time.strptime(name[1:16], "%Y%m%d-%H%M%S")     # must parse, or a listing cannot be trusted to sort


def test_naming_a_folder_that_holds_a_run_is_refused_until_resume_is_said(tmp_path, monkeypatch):
    """Naming a folder that already holds a run is either a mistake or a deliberate continuation, and the two
    have different effects on the history. Guessing merged them: the merge was invisible in `recall` and
    produced a conversation with two objectives and no answer between them."""
    _fake(monkeypatch)
    run_dir = tmp_path / "r"
    assert cli.build(["run", "--run-dir", str(run_dir), "--task", "first", "--ask-timeout", "5",
                     "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    before = (run_dir / "log.jsonl").read_text(encoding="utf-8")
    err = io.StringIO()
    rc = cli.build(["run", "--run-dir", str(run_dir), "--task", "second", "--ask-timeout", "5", "--no-color"],
                   io.StringIO(), err)
    assert rc == 2, f"an existing run folder was appended to (exit {rc})"
    assert "--resume" in err.getvalue(), f"the refusal did not name the way through: {err.getvalue()!r}"
    assert (run_dir / "log.jsonl").read_text(encoding="utf-8") == before, "the refusal still wrote"


def test_an_empty_folder_named_on_purpose_is_not_refused(tmp_path, monkeypatch):
    """An empty directory is not a run: a spawned child's folder is created before its first step, and
    refusing it would make every spawn fail on the folder the parent just made."""
    _fake(monkeypatch)
    run_dir = tmp_path / "child1"
    run_dir.mkdir()
    assert cli.build(["run", "--run-dir", str(run_dir), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     io.StringIO(), io.StringIO()) == 0


def test_send_still_addresses_a_folder_that_holds_a_run(tmp_path, monkeypatch):
    """`send` exists to reach the run in that folder from a second shell — refusing it would remove the
    command whose whole job is naming an existing run. The refusal belongs to `run`, not to every mode."""
    _fake(monkeypatch)
    run_dir = tmp_path / "r"
    assert cli.build(["run", "--run-dir", str(run_dir), "--task", "first", "--ask-timeout", "5",
                     "--no-color"],
                     io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    rc = cli.build(["send", "--run-dir", str(run_dir), "--from", "ada", "steer", "check tests"],
                   io.StringIO(), err)
    assert rc == 0, f"send was refused: exit {rc} — {err.getvalue()!r}"

def test_a_custom_system_loses_its_owner_annotations_too(tmp_path):
    """Comment stripping belongs to prompt files, not only to the shipped ones.

    The rule in `sah/prompts/__init__.py` is that every byte of a prompt is sent, so an owner's notes are
    written as HTML comments and removed on the way out. An operator who copies `preamble.md` to write
    their own inherits those comments -- and every note they add -- unless the custom path renders through
    the same removal. Decided: one rule, both paths.
    """
    custom = "You are a pirate.<!-- a note to self, never for the model --> Arr."
    s = cli.assemble_system(custom, run_id="r", root="/r", model="m", host="h", port=1,
                            policy=Confine(read_roots=["/xroot"]), memory="")
    assert "never for the model" not in s, "a custom --system carried its owner's comments to the model"
    assert s.startswith("You are a pirate."), "stripping took the text with it"
    assert "Arr." in s


# --- the global override: --no-confine ---------------------------------------

class _Witnessed(io.StringIO):
    """An output the run believes a person is watching: the boot box is drawn only for a witness, so a
    plain capture buffer would prove nothing about what the operator is shown."""

    def isatty(self) -> bool:
        return True


def test_the_open_axis_is_recorded_and_shown_as_itself(tmp_path, monkeypatch):
    """Opening a run up is legitimate; doing it silently is not. Both the log and the boot box have to say
    the wall is down, or a later reader cannot tell an open run from one that named a folder — and a root
    list printed beside `rw:` would be a statement about drives nobody granted."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    out, err = _Witnessed(), io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                      "--no-confine"], out, err) == 0, err.getvalue()
    assert _started(tmp_path)["grants"]["unrestricted"] is True, _started(tmp_path).get("grants")
    box = out.getvalue()
    # The line itself, not a substring: an open run has no roots to print, and the cwd it no longer
    # consults appearing beside `rw:` would describe a policy that is not in force.
    rw = [line for line in box.splitlines() if line.startswith("rw:")]
    assert rw == ["rw: anywhere this user can reach (--no-confine)"], box


def test_a_resume_that_loses_the_open_axis_says_so(tmp_path, monkeypatch):
    """The axis is re-typed like every other one, so a resume of an open run is a confined run. The
    operator must see that, because the conversation coming back is full of reads from everywhere."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                      "--no-confine"], io.StringIO(), io.StringIO()) == 0
    err = io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5", "--no-color",
                      "--resume"], io.StringIO(), err) == 0
    text = err.getvalue()
    assert "--no-confine" in text, f"the resume said nothing about the axis it dropped: {text!r}"
    assert "does not grant" in text.lower() and "adds" not in text.lower(), text


def test_an_open_run_says_the_wall_is_down_instead_of_naming_no_roots():
    """`<situation>` is how the model learns its bounds. Telling a run that may read the machine that it
    "may read under: nothing" sends it to ask permission for the thing it is already allowed to do — and
    an ask is the last resort the prompt spends a paragraph on."""
    s = _sit(policy=Confine(read_roots=[], write_roots=[], unrestricted=True))
    assert "--no-confine" in s, s
    assert "you may read under: nothing" not in s, s


def test_the_boot_box_says_whether_the_run_can_send_a_picture(tmp_path, monkeypatch):
    """`--vision` is a capability a run either has or has not, and until now nothing on the console said
    which way it fell. The only place it appeared was inside a refusal, after a picture had already been
    read — where it read as a fact about the server rather than about this invocation. Measured on a run
    whose front-door default never reached it: the operator saw the refusal and had nothing to compare it
    against, because `exec`, `spawn`, `asks`, `rw` and `web` all have a line and vision had none."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    off, err = _Witnessed(), io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color"],
                     off, err) == 0, err.getvalue()
    assert "vision=off" in off.getvalue(), off.getvalue()

    on = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path / "second"), "--task", "one", "--ask-timeout", "5",
                     "--no-color", "--vision"], on, io.StringIO()) == 0
    assert "vision=on" in on.getvalue(), on.getvalue()


def test_vision_can_be_declined_as_well_as_declared(tmp_path, monkeypatch):
    """A door that passes `--vision` by default has to leave the operator a way to say no, and there was
    none: the flag was `store_true`, so `--vision` was the only spelling and a run could not decline what
    its front door asserted. `--no-vision` is that way out, and the last word on the line wins, which is
    the same property that lets `sah run --think none` narrow the door's `--think on`.

    The last-wins case is the one that matters here and is asserted on its own: a negative flag that only
    worked in isolation would still leave a door run unable to opt out."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    declined, err = _Witnessed(), io.StringIO()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                     "--no-vision"],
                     declined, err) == 0, err.getvalue()
    assert "vision=off" in declined.getvalue(), declined.getvalue()

    narrow = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path / "second"), "--task", "one", "--ask-timeout",
                     "5", "--no-color", "--vision", "--no-vision"], narrow, io.StringIO()) == 0
    assert "vision=off" in narrow.getvalue(), f"the door's flag outranked the operator's: {narrow.getvalue()}"


def test_an_attended_run_waits_for_the_answer_rather_than_for_a_deadline(monkeypatch, tmp_path):
    """A person at a terminal is the one case where a question can be answered at any hour, and the run
    used to answer its own question by giving up after 120 seconds — a fuse nobody set and one that turns
    a refusal into a silent `no` the moment the operator reaches for a glass of water. Waiting is the
    honest default for an attended run; `--ask-timeout N` remains the way to set a real deadline, and an
    unattended run still ends itself rather than waiting for a voice that does not exist."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    # An attended run is an interactive one, and the REPL reads a line after the task. An empty stream is
    # the closed-pipe case -- `input()` answers it with EOFError and the session ends, which is the behaviour
    # `test_a_session_ends_when_the_stream_it_reads_is_gone` holds open for the other three.
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    out = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--no-color", "-i"],
                     out, io.StringIO()) == 0
    assert "asks: wait indefinitely" in out.getvalue(), out.getvalue()


def test_the_model_line_carries_the_model_s_own_capabilities(monkeypatch, tmp_path):
    """`think` and `vision` are both facts about the endpoint this run is talking to, so they read as one
    line about that endpoint; `exec`, `spawn` and `rw` are reach, and `asks` is what a refusal does. The
    order is the reading order an operator uses — who am I talking to, then what happens if it says no —
    and it is asserted rather than left to the order of a list literal.

    Vision moved here from a line of its own, where it read as a reach axis alongside `exec`.
    The window states beside the line about the endpoint it bounds: a number the operator did
    not type is still a fact about the endpoint, and the line says how it was derived."""
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    out = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5", "--no-color",
                     "--vision"],
                     out, io.StringIO()) == 0
    lines = [l.strip() for l in out.getvalue().splitlines() if l.strip()]
    head = next(i for i, l in enumerate(lines) if "think=" in l)
    assert "vision=on" in lines[head], lines
    assert lines[head + 1].startswith("context:"), lines
    assert lines[head + 2].startswith("asks:"), lines


class _TrackingFake(Fake):
    """A double that echoes the context window the way the real adapter's `capabilities()` does:
    the constructor's `context_tokens` as its declaration, and a `context_tokens` attribute the
    CLI may correct at boot (the probe's answer) as the window it reports. Without the echo the
    double would state the window the CLI decided against — which is the one thing a test of the
    CLI's window resolution must not do."""

    def capabilities(self):
        return Caps(context_tokens=getattr(self, "context_tokens", self.caps.context_tokens),
                    model=self.caps.model, supports_vision=self.caps.supports_vision)


def _window_fake(**kw):
    """The double carries the adapter's probe surface (base_url and api_key) beside its window,
    so the CLI's probe path runs against it the way it runs against the real adapter."""
    backend = _TrackingFake([turn_tool("c", "verify_objective", {})],
                            caps=Caps(context_tokens=kw.get("context_tokens", 100),
                                      model=kw.get("model", "t")))
    backend.base_url = "http://127.0.0.1:9090/v1"
    backend.api_key = "sk-test"
    return backend


def test_an_undeclared_window_is_probed_and_the_record_says_so(monkeypatch, tmp_path):
    """The operator named no `--context`, and the backend knows its own size: the probe's answer is
    the window the run runs against — on the `started` step, so a reader of a later compaction sees
    the number's origin — and the boot box names it as probed, not declared. The probe is a boot
    act, not a constructor act: the backend is built with the default, and the probed number is
    what the run is handed, because the window is a run fact the record keeps."""
    seen = {}
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: (seen.update(kw), _window_fake(**kw))[-1])
    probed = {}

    def probe(base_url, model, api_key="", api_path="/chat/completions", timeout=5.0):
        probed.update(base_url=base_url, model=model, api_path=api_path)
        return 250_000

    monkeypatch.setattr(cli.openai, "probe_context", probe)
    out = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "one", "--model", "demo-model",
                      "--no-color"], out, io.StringIO()) == 0
    assert probed["model"] == "demo-model", "the probe is asked for the model the run names"
    assert seen["context_tokens"] == 32768,\
        "the constructor gets the default; the probed number is the correction, on the attr"
    assert "context: 250000 (probed from the backend)" in out.getvalue(), out.getvalue()
    steps = Log(tmp_path / "run" / "log.jsonl").steps()
    started = [st for st in steps if st["kind"] == "started"][-1]
    assert started["context"] == 250_000, "the window is a run fact, on the started step"
    notices = [st for st in steps if st["kind"] == "notice" and st.get("subject") == "context"]
    assert len(notices) == 1 and "probed from the backend" in notices[0]["text"], notices


def test_a_window_the_backend_does_not_declare_is_the_default_said_out_loud(monkeypatch, tmp_path):
    """A backend that declares no window (a measured shape) leaves the default in force —
    and says so in two places a later reader of a degenerate compaction will look: the boot box,
    which the operator sees, and the record, which the reader does. The notice is what makes a
    later `blocked` readable as a config gap instead of a mystery, and it names the remedy: set
    --context to the real value."""
    monkeypatch.setattr(cli, "OpenAI", _window_fake)
    monkeypatch.setattr(cli.openai, "probe_context", lambda *a, **kw: None)
    out = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "one", "--model", "demo-model",
                      "--no-color"], out, io.StringIO()) == 0
    box = out.getvalue()
    # The remedy leads the box line, because the box truncates at the terminal's width: the
    # operator sees the state and the fix, and the full cause (the backend declared none and the
    # probe found none) is in the record notice the test reads next.
    assert "context: 32768 (default — set SAH_CONTEXT" in box, box
    steps = Log(tmp_path / "run" / "log.jsonl").steps()
    started = [st for st in steps if st["kind"] == "started"][-1]
    assert started["context"] == 32768, "the default is still the run's window, and it is on the record"
    notices = [st for st in steps if st["kind"] == "notice" and st.get("subject") == "context"]
    assert len(notices) == 1 and "the default" in notices[0]["text"], notices
    assert "--context" in notices[0]["text"], "the remedy is in the record, not only the box"


def test_a_declared_window_is_not_probed(monkeypatch, tmp_path):
    """The operator's number is the operator's: a declared `--context` is never replaced, and the
    probe is never made — a probe that ran over a declaration would be the harness deciding the
    window the operator already named, and the boot box says the number is declared, not found."""
    def refuse(*a, **kw):
        raise AssertionError("the probe ran over a declared window")

    monkeypatch.setattr(cli, "OpenAI", _window_fake)
    monkeypatch.setattr(cli.openai, "probe_context", refuse)
    out = _Witnessed()
    assert cli.build(["run", "--run-dir", str(tmp_path / "run"), "--task", "one", "--model", "demo-model",
                      "--context", "4096", "--no-color"], out, io.StringIO()) == 0
    assert "context: 4096 (declared)" in out.getvalue(), out.getvalue()
    steps = Log(tmp_path / "run" / "log.jsonl").steps()
    started = [st for st in steps if st["kind"] == "started"][-1]
    assert started["context"] == 4096
    assert [st for st in steps if st["kind"] == "notice"
        and st.get("subject") == "context"] == [],\
        "a declaration is a declaration: nothing to record about where it came from"



class _Raises:
    """A stream whose every read fails, the way a real one does once it is gone."""

    def __init__(self, exc):
        self.exc = exc

    def readline(self, *args):
        raise self.exc


def test_a_session_ends_when_the_stream_it_reads_is_gone(monkeypatch, tmp_path):
    """The REPL catches `EOFError` and `KeyboardInterrupt` around `input()`, which is what the builtin
    documents and what a terminal produces. It is not what a vanished stream produces. Measured on this box
    (CPython 3.12, Windows), because the class differs per condition and one guess would cover one case:

      * stdin's fd closed (`0<&-`)           -> OSError:    [Errno 9] Bad file descriptor
      * no stream to read at all             -> RuntimeError: lost sys.stdin
      * the stream object closed             -> ValueError: I/O operation on closed file.
      * an empty pipe, or Ctrl+Z             -> EOFError    (the case already handled)

    A detached child and a parent whose pipe closed are both in one of the first three states by
    construction, and any of them escaping the loop is a traceback where a session should have ended. The
    exit stays 0 for the same reason EOF leaves it 0: a session is not its worst turn.
    """
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    cases = {
        "fd closed": _Raises(OSError(9, "Bad file descriptor")),\
        "stream lost": None,\
        "stream closed": _Raises(ValueError("I/O operation on closed file.")),
    }
    for name, stream in cases.items():
        run_dir = tmp_path / name.replace(" ", "_")
        monkeypatch.setattr(sys, "stdin", stream)
        out = _Witnessed()
        assert cli.build(["run", "--run-dir", str(run_dir), "--task", "one", "--no-color", "-i"],
                         out, io.StringIO()) == 0, f"a vanished {name} crashed the session"


def test_the_wire_flag_wires_the_capture_path_to_the_backend(tmp_path, monkeypatch):
    """`--wire` is a debug option: the operator opts in, and the run's composed requests are captured
    to `<run-dir>/wire.jsonl`. The flag must actually reach the backend — a parsed flag that is never
    passed to the adapter is a switch that does nothing, and the one layer it exists to expose is
    exactly the layer a wrong wiring would leave blind."""
    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return Fake([turn_text("done")], caps=Caps(100))

    wired = tmp_path / "wired"
    monkeypatch.setattr(cli, "OpenAI", fake_openai)
    code = cli.build(["run", "--wire", "--run-dir", str(wired), "--task", "one", "--ask-timeout", "5",
                     "--no-color"],
                     io.StringIO(), io.StringIO())
    assert code == 0, "the one-shot run completed"
    assert seen["wire"] == str(wired / "wire.jsonl"), "the flag did not reach the backend"
    # And without the flag the capture is off: the backend is built with no wire path at all.
    seen.clear()
    cli.build(["run", "--run-dir", str(tmp_path / "plain"), "--task", "one", "--ask-timeout", "5",
              "--no-color"],
              io.StringIO(), io.StringIO())
    assert seen["wire"] is None, "a run that did not ask for a wire must not capture one"


def test_the_project_notes_reader_takes_precedence_and_stops_at_the_cwd(tmp_path):
    """First present wins in the fixed order (override > AGENTS > CLAUDE), and the scope is the cwd
    alone: the reader never walks up, so a file in a parent directory cannot leak into a run confined
    to a subdirectory."""
    (tmp_path / "CLAUDE.md").write_text("legacy")
    assert cli._load_project_notes(str(tmp_path))["path"].endswith("CLAUDE.md")
    (tmp_path / "AGENTS.md").write_text("standard")
    assert cli._load_project_notes(str(tmp_path))["path"].endswith("AGENTS.md"), "AGENTS beats CLAUDE"
    (tmp_path / "AGENTS.override.md").write_text("override")
    assert cli._load_project_notes(str(tmp_path))["path"].endswith("AGENTS.override.md")
    # Nothing present, nothing loaded.
    assert cli._load_project_notes(str(tmp_path / "empty")) is None
    # A subdirectory does not inherit the parent's file: scope is the one directory.
    (tmp_path / "child").mkdir()
    assert cli._load_project_notes(str(tmp_path / "child")) is None


def test_the_system_assembly_frames_the_notes_and_only_when_granted(tmp_path):
    """The notes become a labeled block appended project-last (below the situation and the global
    agents guidance), and the path is named in <situation>. A custom --system still gets the block
    (the operator granted it; an author's prompt must not withhold it); a run that did not grant it
    gets no block at all, and the situation says so."""
    (tmp_path / "AGENTS.md").write_text("use the project's make target")
    notes = cli._load_project_notes(str(tmp_path))
    kwargs = dict(run_id="r", root=str(tmp_path), model="m", host="h", port=1, memory="")
    granted = cli.assemble_system(None, project_notes=notes,
                                  policy=Confine(project_notes=True), **kwargs)
    assert "use the project's make target" in granted, "the verbatim content is in the prompt"
    assert "is the working directory's own notes" in granted
    assert granted.index("own notes, loaded for convenience") > granted.index("<situation>"), \
        "the block is appended below the situation, project-last"
    assert notes["path"] in granted, "the path is named"
    # A custom --system still carries the block.
    custom = cli.assemble_system("MY PREAMBLE", project_notes=notes,
                                 policy=Confine(project_notes=True), **kwargs)
    assert "MY PREAMBLE" in custom and "own notes, loaded for convenience" in custom
    # Not granted: no block, and the situation states the absence.
    plain = cli.assemble_system(None, project_notes=None,
                                policy=Confine(project_notes=False), **kwargs)
    assert "own notes, loaded for convenience" not in plain
    assert "are not loaded" in plain


def test_a_granted_run_records_the_notes_it_loaded(tmp_path, monkeypatch):
    """End to end: a run granted --project-notes in a cwd that holds a CLAUDE.md loads it, and the
    record shows the act -- the grant on started.grants, the file on started.project_notes_file, and
    the framed content inside started.system. A later reader can audit what external content entered
    the prompt from the log alone."""
    (tmp_path / "CLAUDE.md").write_text("run `make test` before committing; never edit generated/")
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    code = cli.build(["run", "--project-notes", "--run-dir", str(run_dir), "--task", "one",
                      "--ask-timeout", "5", "--no-color"], io.StringIO(), io.StringIO())
    assert code == 0, "the run completed"
    started = next(s for s in Log(str(run_dir / "log.jsonl")).read()[0] if s["kind"] == "started")
    assert started["grants"]["project_notes"] is True, "the grant is recorded"
    assert started["project_notes_file"].endswith("CLAUDE.md"), "the loaded file is recorded"
    assert "run `make test` before committing" in started["system"], "the framed content is in the head"
    assert "is the working directory's own notes" in started["system"]


def test_a_run_without_the_grant_loads_nothing(tmp_path, monkeypatch):
    """The grant is off by default: a cwd that holds a CLAUDE.md still loads nothing when the operator
    did not pass --project-notes, so the default run has no external-content surface."""
    (tmp_path / "CLAUDE.md").write_text("these rules must not leak in ungranted")
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    cli.build(["run", "--run-dir", str(run_dir), "--task", "one", "--no-color"],
              io.StringIO(), io.StringIO())
    started = next(s for s in Log(str(run_dir / "log.jsonl")).read()[0] if s["kind"] == "started")
    assert started["grants"]["project_notes"] is False
    assert "project_notes_file" not in started, "no file was loaded, so the field is absent"
    assert "these rules must not leak in ungranted" not in started["system"]


def test_the_reach_words_name_the_project_notes_grant():
    """The reach words are the flags the operator types, so a resume that gains or loses the grant
    names it the way it would re-type it."""
    base = {"read": [], "write": [], "exec": False, "unrestricted": False}
    assert "--project-notes" in cli._reach_words({**base, "project_notes": True})
    assert "--project-notes" not in cli._reach_words({**base, "project_notes": False})


def test_the_budget_flags_reach_the_agent(tmp_path, monkeypatch):
    """A switch nobody can set is not a capability: --max-runtime and --grace-turns must reach
    the loop with the values the operator typed, beside --max-turns."""
    seen = {}
    real = cli.Agent

    def spy(**kw):
        seen.update(kw)
        return real(backend=Fake([turn_text("ok")], caps=Caps(100)),
                    **{k: v for k, v in kw.items() if k != "backend"})

    monkeypatch.setattr(cli, "Agent", spy)
    code = cli.build(["run", "--run-dir", str(tmp_path), "--task", "go", "--ask-timeout", "5", "--no-color",
                      "--max-turns", "5", "--max-runtime", "90", "--grace-turns", "1"],
                     io.StringIO(), io.StringIO())
    assert code == 0
    assert seen["max_turns"] == 5 and seen["max_runtime"] == 90 and seen["grace_turns"] == 1


def test_the_default_budgets_are_the_run_s_own_defaults(tmp_path, monkeypatch):
    seen = {}
    real = cli.Agent

    def spy(**kw):
        seen.update(kw)
        return real(backend=Fake([turn_text("ok")], caps=Caps(100)),
                    **{k: v for k, v in kw.items() if k != "backend"})

    monkeypatch.setattr(cli, "Agent", spy)
    cli.build(["run", "--run-dir", str(tmp_path), "--task", "go", "--no-color"],
              io.StringIO(), io.StringIO())
    assert seen["max_turns"] == 12 and seen["grace_turns"] == 3 and seen["max_runtime"] == 0


def test_a_negative_budget_is_refused_before_the_run_exists(tmp_path):
    """A negative budget is not a budget: zero is the meaningful edge (no work, wrap up now;
    no wall), and the refusal names the flag before the run exists, the way the other
    argument refusals do."""
    for flag in ("--max-turns", "--max-runtime", "--grace-turns"):
        err = io.StringIO()
        code = cli.build(["run", "--task", "go", "--run-dir", str(tmp_path), f"{flag}=-1"],
                         io.StringIO(), err)
        assert code == 2, flag
        assert flag in err.getvalue(), f"the refusal did not name {flag}:\n{err.getvalue()!r}"


# --- resuming at a step: the cut is resolved and refused before it is written --------------------
#
# `--resume <step>` is the jump-back: the conversation is rebuilt from the prefix ending at the
# step the operator names, and the later steps are detached — kept in the log for the audit, never
# in any conversation again. The file is never truncated, which is what makes the jump safe: the
# record shows both lineages and the act between them. The refusals are mechanical, in the order
# they are checked: the number is read, the target resolves against the conversation in force,
# the target is not the run's ending, and the prefix owes no tool result.


def _log_at(tmp_path, *steps):
    log = Log(tmp_path / "log.jsonl")
    for kind, issuer, fields in steps:
        log.append(kind, issuer, **fields)
    return str(tmp_path)


def test_a_resume_at_a_closed_exchange_returns_the_target_and_the_detached_count(tmp_path):
    root = _log_at(tmp_path, ("message", "user", {"role": "user", "text": "a"}),
                   ("message", "model", {"role": "assistant", "text": "b"}),
                   ("ended", "core", {"reason": "complete", "exit_code": 0}))
    target, detached = cli._resolve_cut(root, "1", io.StringIO())
    assert target == "000001-message"
    assert detached == 1, "the ended step is what this cut detaches"


def test_a_resume_number_outside_the_conversation_in_force_is_named(tmp_path):
    root = _log_at(tmp_path, ("message", "user", {"role": "user", "text": "a"}),
                   ("ended", "core", {"reason": "complete", "exit_code": 0}))
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        cli._resolve_cut(root, "9", err)
    assert exc.value.code == 2
    assert "not in this run's conversation in force" in err.getvalue()


def test_a_resume_at_a_detached_step_is_refused_with_the_line_named(tmp_path):
    root = _log_at(tmp_path,
                   ("message", "user", {"role": "user", "text": "a"}),
                   ("message", "user", {"role": "user", "text": "b"}),
                   ("message", "user", {"role": "user", "text": "c"}),
                   ("cut", "core", {"to": "000000-message"}),
                   ("message", "user", {"role": "user", "text": "d"}))
    err = io.StringIO()
    with pytest.raises(SystemExit):
        cli._resolve_cut(root, "2", err)
    assert "detached continuation" in err.getvalue()


def test_a_resume_at_the_ended_step_is_refused(tmp_path):
    root = _log_at(tmp_path, ("message", "user", {"role": "user", "text": "a"}),
                   ("ended", "core", {"reason": "complete", "exit_code": 0}))
    err = io.StringIO()
    with pytest.raises(SystemExit):
        cli._resolve_cut(root, "1", err)
    assert "ending" in err.getvalue()


def test_a_resume_that_would_leave_a_call_owed_its_result_is_refused(tmp_path):
    root = _log_at(tmp_path,
                   ("message", "user", {"role": "user", "text": "a"}),
                   ("call", "model", {"refs": "000000-message", "name": "read"}),
                   ("message", "user", {"role": "user", "text": "b"}))
    err = io.StringIO()
    with pytest.raises(SystemExit):
        cli._resolve_cut(root, "1", err)
    text = err.getvalue()
    assert "owed a result" in text and "read (000001-call)" in text, \
        "the refusal names the call that would be left outstanding"


def test_a_resume_number_that_is_not_a_number_is_named(tmp_path):
    root = _log_at(tmp_path, ("message", "user", {"role": "user", "text": "a"}))
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        cli._resolve_cut(root, "abc", err)
    assert exc.value.code == 2
    assert "step number" in err.getvalue()


def test_a_resume_at_a_step_detaches_the_tail_and_steers_from_the_prefix(tmp_path, monkeypatch):
    """End to end: jump back to a step and steer again, with no file surgery. The detached
    tail stays in log.jsonl for the audit, the conversation the second invocation sends is the
    prefix plus the new objective — never the tail — and the line says how many steps came off.
    """
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: Fake([turn_text("done")], caps=Caps(100)))
    assert cli.build(["run", "--run-dir", str(tmp_path), "--task", "one", "--ask-timeout", "5",
                      "--no-color"], io.StringIO(), io.StringIO()) == 0
    log = Log(tmp_path / "log.jsonl")
    target = next(s for s in log.steps()
                  if s["kind"] == "message" and s.get("role") == "assistant")["id"]
    second = Fake([turn_text("steered")], caps=Caps(100))
    monkeypatch.setattr(cli, "OpenAI", lambda **kw: second)
    err = io.StringIO()
    code = cli.build(["run", "--run-dir", str(tmp_path), "--task", "two", "--ask-timeout", "5",
                      "--no-color", "--resume", target.split("-")[0]], io.StringIO(), err)
    assert code == 0
    line = [l for l in err.getvalue().splitlines() if l.startswith("resumed ")]
    assert line and "step(s) detached" in line[0], err.getvalue()
    steps = log.steps()
    cut = next(s for s in steps if s["kind"] == "cut")
    assert cut["to"] == target, "the cut names the step the conversation continues from"
    assert any(s["kind"] == "ended" for s in steps[: steps.index(cut)]), \
        "the first run's ending stands in the file before the cut — the tail was kept, not erased"
    sent = [m.content for m in second.calls[0].messages if m.role in ("user", "assistant")]
    assert sent == ["one", "done", "two"], f"the tail came back with the request: {sent}"
