"""The operator's channel on a live session: the slash commands. The guarantees a line at the
prompt owes: a slash line never reaches the model; every outcome — done, refused, listed — is
a word (a notice beside the record, the aid's own line for a list); /think changes the level
the next turn runs at; /compact runs the act the trigger runs, and says so; /backto cuts the
conversation before the prompt it names, detaches the tail the log keeps, and the operator
types the replacement. A session the command refuses still stands: a refusal ends the command,
not the session."""
import io
import os
from types import SimpleNamespace

from sah import cli, console, local
from sah.agent import Agent, Verdict
from sah.model import Caps, Fake, Message, Stopped, TextDelta, Usage, turn_text
from sah.record import Log, Refused

ENDPOINT = {"model": "test-1", "host": "127.0.0.1", "port": 8080, "base_url": "",
            "api_key": "", "think": "none", "context": 100, "vision": False}


def build(turns, root, **kwargs):
    """A Fake-backed run the session's commands can reach, the way `test_cli.build` builds one."""
    log = Log(root / "log.jsonl")
    backend = Fake(turns, caps=Caps(context_tokens=100, model="test-1"))
    return Agent(backend=backend, tools=kwargs.pop("tools", {}), log=log, root=root,
                 policy=kwargs.pop("policy", (lambda name, arguments, **_: Verdict(True))),
                 attended=True, **kwargs)


def session(agent, **kwargs):
    """A session standing on the run, on the endpoint in force (the boot's resolved values)."""
    return console.Session(agent=agent, endpoint=dict(ENDPOINT), **kwargs)


def make_spawner(tmp_path, **overrides):
    """A stand-in for the child spawner the /fork command builds its argv from: the fields the
    command reads, plus the two methods it calls (next_id, child_dir). A child the test would
    never actually launch — subprocess_spawn is patched where a launch is the act under test."""
    base = dict(model="test-1", host="127.0.0.1", port=8080, api_key="", think="none",
                context=100, max_turns=30, grace_turns=3, read_roots=["/work"],
                write_roots=["/out"], allow_exec=False, runs_root=str(tmp_path / "runs"),
                memory="", summary_thinking=None, max_runtime=0, allow_web=False,
                web_private=False, search_url="")
    base.update(overrides)
    ns = SimpleNamespace(**base)
    ns.next_id = lambda parent: "child1"
    ns.child_dir = lambda run_id: os.path.join(ns.runs_root, run_id)
    return ns


def command_notices(agent):
    steps, _ = agent.log.read()
    return [s for s in steps if s["kind"] == "notice" and s["subject"] == "command"]


def run_session(agent, lines, out=None, err=None, **session_kwargs):
    """Drive the REPL with the lines: the real `cli.repl`, the real dispatch, the record the
    assertions read afterwards."""
    out, err = out or io.StringIO(), err or io.StringIO()
    cli.repl(agent, iter(lines), out, err, session=session(agent, **session_kwargs))
    return out, err


# --- the channel itself ---------------------------------------------------------


def test_a_slash_line_never_reaches_the_model(tmp_path):
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["/think none", "go", "quit"])
    assert len(agent.backend.calls) == 1            # only "go" went out
    assert agent.backend.calls[0].messages[-1].content == "go"
    assert command_notices(agent)                   # the command's word is on the record


def test_an_unknown_command_is_refused_and_the_session_stands(tmp_path):
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["/frobnicate", "go", "quit"])
    notice = command_notices(agent)[0]["text"]
    assert "not a command" in notice and "/backto" in notice
    assert agent.backend.calls[-1].messages[-1].content == "go"    # the session went on


def test_the_quit_words_still_quit(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/think", "quit"])
    assert not agent.backend.calls


# --- /think ---------------------------------------------------------------------


def test_think_sets_the_level_the_next_turn_runs_at(tmp_path):
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["/think xhigh", "go", "quit"])
    assert agent.backend.calls[0].thinking == "xhigh"
    notice = command_notices(agent)[0]["text"]
    assert "think: xhigh" in notice and "was none" in notice


def test_think_none_omits_the_field(tmp_path):
    agent = build([turn_text("ok")], tmp_path, thinking="xhigh")
    out, err = run_session(agent, ["/think none", "go", "quit"])
    assert agent.backend.calls[0].thinking is None
    assert "think: none (was xhigh)" in command_notices(agent)[0]["text"]


def test_think_with_no_word_reports_the_level_in_force(tmp_path):
    agent = build([], tmp_path, thinking="xhigh")
    out, err = run_session(agent, ["/think", "quit"])
    assert not agent.backend.calls
    assert "think: xhigh" in command_notices(agent)[0]["text"]


def test_think_reports_a_pinned_summary_level_it_does_not_touch(tmp_path):
    agent = build([], tmp_path, thinking="xhigh", summary_thinking="low")
    out, err = run_session(agent, ["/think", "quit"])
    assert "the summary calls keep their own level: low" in command_notices(agent)[0]["text"]


def test_think_refuses_a_multiword_level_and_changes_nothing(tmp_path):
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["/think x high", "go", "quit"])
    assert "a level is one word" in command_notices(agent)[0]["text"]
    assert agent.thinking == "none"
    assert agent.backend.calls[0].thinking == "none"   # the run's default, unchanged


def test_think_updates_the_spawners_child_level(tmp_path):
    agent = build([], tmp_path)
    spawner = SimpleNamespace(model="test-1", host="127.0.0.1", port=8080,
                              api_key="", think="none", context=100)
    out, err = run_session(agent, ["/think high", "quit"], spawner=spawner)
    assert spawner.think == "high"


# --- /compact -------------------------------------------------------------------


def stuff(agent, count, chars=100):
    """A conversation long enough for the act to have something to do."""
    for i in range(count):
        agent.messages.append(Message("user" if i % 2 == 0 else "assistant", "t" * chars))


def test_compact_refuses_a_short_conversation(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/compact", "quit"])
    notice = command_notices(agent)[0]["text"]
    assert "nothing to drop yet" in notice
    assert agent.messages == []                        # the conversation stands untouched
    steps, _ = agent.log.read()
    assert not [s for s in steps if s["kind"] == "summary"]


def test_compact_runs_the_act_and_the_record_carries_the_numbers(tmp_path):
    """A window the summary cannot fit (the room is the window's own arithmetic, unchanged by
    the bypass) still lands the guaranteed slot: the act rewrites the conversation, the `summary`
    step holds the stage and the slot's kind, the cut rung's drop is stamped on it, and the
    notice carries the operator's numbers."""
    agent = build([], tmp_path)
    stuff(agent, 8)
    out, err = run_session(agent, ["/compact", "quit"])
    steps, _ = agent.log.read()
    summary = [s for s in steps if s["kind"] == "summary"]
    assert len(summary) == 1
    assert summary[0]["dropped"] == 2
    # No scripted summary answer was left to take: the slot holds the failed note, and the cut
    # rung dropped the backfill to the room, so the slot stands alone.
    assert summary[0]["slot"] == "failed" and summary[0]["stage"] == 4
    assert summary[0].get("backfill_cut") == 6 and summary[0]["kept"] == 0
    notice = command_notices(agent)[0]["text"]
    assert "compact: ran" in notice and "2 turn(s) summarised" in notice
    # the slot stands with the act's marker: the backfill gone by the cut rung
    assert len(agent.messages) == 2
    assert "compacted" in agent.messages[0].content


def test_compact_a_window_that_cannot_fit_says_so_and_the_session_stands(tmp_path):
    """The resident load and the smallest slot exceed the limit: the act raises ContextBlocked,
    the console catches it, nothing changes, and the session goes on — the operator's answer is
    a bigger window, not a dead session."""
    agent = build([turn_text("ok")], tmp_path, system="s" * 200)
    stuff(agent, 8)
    out, err = run_session(agent, ["/compact", "go", "quit"])
    notice = command_notices(agent)[0]["text"]
    assert "no compaction can fit the window" in notice and "nothing was changed" in notice
    # the 8 stuffed messages stand untouched, and "go" ran its exchange beside them
    assert len(agent.messages) == 10
    assert all(m.content == "t" * 100 for m in agent.messages[:8])
    assert agent.backend.calls[0].messages[-1].content == "go"


def test_compact_takes_no_argument(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/compact now", "quit"])
    assert "takes no argument" in command_notices(agent)[0]["text"]


# --- /backto --------------------------------------------------------------------


def test_backto_without_a_number_lists_the_prompts_in_force(tmp_path):
    agent = build([turn_text("a"), turn_text("b")], tmp_path)
    out, err = run_session(agent, ["one", "two", "/backto", "quit"])
    assert "your prompts in force" in err.getvalue()
    assert "one" in err.getvalue() and "two" in err.getvalue()
    steps, _ = agent.log.read()
    assert not [s for s in steps if s["kind"] == "cut"]   # the list is an aid, not an act


def test_backto_cuts_before_the_first_prompt_and_the_replacement_starts_fresh(tmp_path):
    agent = build([turn_text("one"), turn_text("two"), turn_text("three")], tmp_path)
    out, err = run_session(agent, ["one", "two", "/backto 1", "three", "quit"])
    steps, _ = agent.log.read()
    cut = [s for s in steps if s["kind"] == "cut"][0]
    started = [s for s in steps if s["kind"] == "started"][0]
    assert cut["to"] == started["id"]                # the boundary before prompt 1 is the run's start
    # The replacement turn sees a fresh conversation: nothing of the old timeline
    call = agent.backend.calls[2]
    assert [m.content for m in call.messages] == ["three"]
    notice = command_notices(agent)[0]["text"]
    assert "prompt 1 and 7 step(s) after it detached" in notice


def test_backto_cuts_before_a_later_prompt_and_keeps_what_came_before(tmp_path):
    agent = build([turn_text("one"), turn_text("two"), turn_text("three")], tmp_path)
    out, err = run_session(agent, ["one", "two", "/backto 2", "three", "quit"])
    # The boundary is the step before prompt 2's message: its objective's `ended` marker
    # detaches with the tail it belongs to, so the cut lands at the previous exchange's end.
    call = agent.backend.calls[2]
    assert [m.role for m in call.messages] == ["user", "assistant", "user"]
    assert call.messages[0].content == "one"
    assert call.messages[-1].content == "three"
    steps, _ = agent.log.read()
    assert [s for s in steps if s["kind"] == "cut"]    # the act is on the record
    # The detached tail is still in the file: the audit reads it whole
    all_steps, _ = agent.log.read()
    assert sum(1 for s in all_steps if s["kind"] == "message") == 6    # the old four, plus the
    # exchange the replacement ran beside the cut


def test_backto_out_of_range_is_refused_and_the_session_stands(tmp_path):
    agent = build([turn_text("one"), turn_text("two")], tmp_path)
    out, err = run_session(agent, ["one", "/backto 99", "go", "quit"])
    notice = command_notices(agent)[0]["text"]
    assert "prompt 99" in notice and "1 operator prompt" in notice
    assert agent.backend.calls[-1].messages[-1].content == "go"


def test_backto_a_number_that_is_not_a_number_is_refused(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/backto abc", "quit"])
    assert "not a prompt number" in command_notices(agent)[0]["text"]


# --- /fork --------------------------------------------------------------------


def test_fork_carries_the_conversation_and_launches_a_fork_run(tmp_path, monkeypatch):
    """/fork with no number carries the conversation in force whole and launches a `sah fork`
    (not a `sah run`): the argv names the fork verb, the source is the run's own log, the step is
    the conversation's last step, the objective rides the argv, and the standing grants travel
    with it. The notice names the fork's folder; the source is never launched or written."""
    agent = build([[TextDelta("ok"), Usage(5, 2), Stopped("stop")]], tmp_path)
    agent.run("one")
    # the conversation's end, read before the fork command adds its notice: the branch is the
    # conversation in force's last step, and the notice the command records comes after it
    steps, _ = agent.log.read()
    branch_end = steps[-1]["id"].split("-")[0]
    spawner = make_spawner(tmp_path)
    captured = {}
    monkeypatch.setattr(console, "subprocess_spawn",
                        lambda argv, root, api_key="": captured.update(argv=argv, root=root))
    out, err = run_session(agent, ["/fork do it the other way", "quit"], spawner=spawner)
    argv = captured["argv"]
    assert argv[2:4] == ["sah", "fork"], f"the fork is not a sah fork: {argv}"
    assert str(agent.log.path) in argv, "the source is not the run's own log"
    assert argv[argv.index("--step") + 1] == branch_end, "the branch is not the conversation's end"
    assert " ".join(argv[argv.index("--task") + 1:]) == "do it the other way"
    assert "--grant" in argv and "--write" in argv, "the standing grants are not forwarded"
    notices = command_notices(agent)
    assert any("fork:" in n["text"] and "child1" in n["text"] for n in notices)


def test_fork_with_a_number_branches_before_the_named_prompt(tmp_path, monkeypatch):
    """/fork <N> branches at the boundary before the operator's Nth prompt (the same boundary
    /backto cuts to), so the fork carries the conversation up to that boundary and diverges from
    there. The step the argv carries is the boundary's, not the prompt's."""
    agent = build([[TextDelta("a"), Usage(5, 2), Stopped("stop")],
                   [TextDelta("b"), Usage(5, 2), Stopped("stop")]], tmp_path)
    agent.run("one")
    agent.run("two")
    spawner = make_spawner(tmp_path)
    captured = {}
    monkeypatch.setattr(console, "subprocess_spawn",
                        lambda argv, root, api_key="": captured.update(argv=argv))
    out, err = run_session(agent, ["/fork 1 redo it", "quit"], spawner=spawner)
    argv = captured["argv"]
    assert argv[2:4] == ["sah", "fork"]
    # the boundary before prompt 1 is the run's started step (the first prompt has nothing before
    # it but the head) — the branch is the head, not the prompt
    steps, _ = agent.log.read()
    started_seq = next(s["id"].split("-")[0] for s in steps if s["kind"] == "started")
    assert argv[argv.index("--step") + 1] == started_seq, \
        f"the branch is not the boundary before the named prompt: {argv}"


def test_fork_without_a_spawner_is_refused_and_the_session_stands(tmp_path):
    """A run that cannot spawn cannot launch a fork from the session: the command says so and
    the session stands — the operator's escape is the `sah fork` shell door, which the notice
    names."""
    agent = build([[TextDelta("ok"), Usage(5, 2), Stopped("stop")]], tmp_path)
    agent.run("one")
    out, err = run_session(agent, ["/fork do it", "quit"])       # no spawner
    notices = command_notices(agent)
    assert any("cannot spawn" in n["text"] and "sah fork" in n["text"] for n in notices)


def test_fork_without_an_objective_is_refused(tmp_path, monkeypatch):
    """A fork with no objective is a fork that would carry the conversation and then do nothing:
    the command refuses it and says what the fork owes."""
    agent = build([[TextDelta("ok"), Usage(5, 2), Stopped("stop")]], tmp_path)
    agent.run("one")
    spawner = make_spawner(tmp_path)
    captured = {}
    monkeypatch.setattr(console, "subprocess_spawn",
                        lambda argv, root, api_key="": captured.update(argv=argv))
    out, err = run_session(agent, ["/fork 1", "quit"], spawner=spawner)
    assert not captured, "a fork with no objective launched"
    assert any("needs an objective" in n["text"] for n in command_notices(agent))


# --- /model ---------------------------------------------------------------------


def test_model_literal_switches_the_backend_and_rebinds_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models", lambda root=None: None)
    agent = build([], tmp_path)
    rebind = lambda m, h, p: f"SYS {m} @ {h}:{p}"
    out, err = run_session(agent, ["/model other-model", "quit"], rebind=rebind)
    assert agent.backend.model == "other-model"
    assert agent.backend.host == "127.0.0.1" and agent.backend.port == 8080
    assert agent.system == "SYS other-model @ 127.0.0.1:8080"
    notice = command_notices(agent)[0]["text"]
    assert "other-model" in notice and "literal model name" in notice
    # no turn was run against the new backend: the record holds no messages at all
    steps, _ = agent.log.read()
    assert not [s for s in steps if s["kind"] == "message"]


def test_model_a_profile_rides_its_endpoint_fields_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models",
                        lambda root=None: {"flash": {"model": "flash-next",
                                                     "host": "10.0.0.9", "port": 9090,
                                                     "context": 5000}})
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/model flash", "quit"])
    assert agent.backend.model == "flash-next"
    assert agent.backend.host == "10.0.0.9" and agent.backend.port == 9090
    assert agent.backend.context_tokens == 5000      # named by the profile: no probe was spent
    notice = command_notices(agent)[0]["text"]
    assert "profile 'flash'" in notice and "was test-1 @ 127.0.0.1:8080" in notice


def test_model_a_field_the_profile_leaves_silent_stays_as_the_run_has_it(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models",
                        lambda root=None: {"name": {"model": "only-a-name"}})
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/model name", "quit"])
    assert agent.backend.model == "only-a-name"
    assert agent.backend.host == "127.0.0.1" and agent.backend.port == 8080
    assert agent.backend.context_tokens == 100       # the run's in-force window, endpoint unmoved


def test_model_an_unknown_id_is_a_literal_on_the_endpoint_in_force(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models",
                        lambda root=None: {"flash": {"model": "flash-next"}})
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/model nope", "quit"])
    assert agent.backend.model == "nope"
    assert agent.backend.host == "127.0.0.1" and agent.backend.port == 8080
    assert "literal model name" in command_notices(agent)[0]["text"]


def test_model_a_malformed_table_is_a_named_refusal(tmp_path, monkeypatch):
    def boom(root=None):
        raise Refused("data/models.json is not JSON: {")
    monkeypatch.setattr(local, "load_models", boom)
    agent = build([], tmp_path)
    orig_backend = agent.backend
    out, err = run_session(agent, ["/model flash", "quit"])
    assert "data/models.json is not JSON" in command_notices(agent)[0]["text"]
    assert agent.backend is orig_backend            # nothing switched


def test_model_switch_updates_the_spawners_child_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models",
                        lambda root=None: {"flash": {"model": "flash-next",
                                                     "host": "10.0.0.9", "port": 9090,
                                                     "think": "xhigh", "context": 5000}})
    agent = build([], tmp_path)
    spawner = SimpleNamespace(model="test-1", host="127.0.0.1", port=8080,
                              api_key="", think="none", context=100)
    out, err = run_session(agent, ["/model flash", "quit"], spawner=spawner)
    assert spawner.model == "flash-next" and spawner.host == "10.0.0.9"
    assert spawner.port == 9090 and spawner.think == "xhigh" and spawner.context == 5000


def test_model_with_no_word_lists_the_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models",
                        lambda root=None: {"flash": {"model": "flash-next",
                                                     "host": "10.0.0.9", "port": 9090}})
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/model", "quit"])
    assert "profiles (data/models.json)" in err.getvalue()
    assert "flash" in err.getvalue() and "flash-next" in err.getvalue()
    assert not command_notices(agent)                # the list is an aid, not an act


def test_model_with_no_table_and_no_word_says_what_the_shape_is(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models", lambda root=None: None)
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/model", "quit"])
    assert "no data/models.json" in command_notices(agent)[0]["text"]


def test_model_refuses_a_multiword_id(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_models", lambda root=None: None)
    agent = build([], tmp_path)
    orig_backend = agent.backend
    out, err = run_session(agent, ["/model flash now", "quit"])
    assert "an id is one word" in command_notices(agent)[0]["text"]
    assert agent.backend is orig_backend    # nothing switched


# --- /quit, /new, /export, /live -----------------------------------------------


def test_the_quit_words_take_the_slash_spelling_too(tmp_path):
    """The channel's surface includes the door out: /quit and /exit end the session the way
    quit and exit do — no objective runs, and no notice, because the last objective's `ended`
    step is already the record of how the session closed."""
    for word in ("/quit", "/exit"):
        agent = build([], tmp_path)
        out, err = run_session(agent, [word])
        assert not agent.backend.calls
        assert not command_notices(agent)


def test_new_detaches_the_conversation_and_the_fresh_objective_starts_clean(tmp_path):
    agent = build([turn_text("one"), turn_text("two"), turn_text("three")], tmp_path)
    out, err = run_session(agent, ["one", "two", "/new", "three", "quit"])
    steps, _ = agent.log.read()
    assert [s for s in steps if s["kind"] == "cut"]    # the act is on the record, same step as backto
    # the fresh objective sees an empty conversation: nothing of the old timeline
    assert [m.content for m in agent.backend.calls[2].messages] == ["three"]
    notice = command_notices(agent)[0]["text"]
    assert "new: the conversation detached" in notice and "type the fresh objective" in notice


def test_new_on_a_conversation_without_an_operator_prompt_says_so_and_cuts_nothing(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/new", "quit"])
    notice = command_notices(agent)[0]["text"]
    assert "no operator prompt" in notice
    steps, _ = agent.log.read()
    assert not [s for s in steps if s["kind"] == "cut"]    # the refusal is a word, not an act


def test_new_takes_no_argument(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/new now", "quit"])
    assert "takes no argument" in command_notices(agent)[0]["text"]


def test_export_writes_the_snapshot_and_the_record_carries_the_word(tmp_path):
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["do it", "/export", "quit"])
    report = (agent.root / "report.md").read_text(encoding="utf-8")
    # the header marks the snapshot: the run is still going, and the projection says so
    assert "snapshot" in report.splitlines()[0]                # the header says what it is


def test_live_reports_the_window_and_records_nothing(tmp_path):
    """A query is a line, not a record: the vitals print on the transient channel, the log gains
    no command notice, and no turn runs."""
    agent = build([], tmp_path)
    stuff(agent, 8)
    out, err = run_session(agent, ["/live", "quit"])
    text = err.getvalue()
    assert "live:" in text and "window tokens" in text
    assert "trigger" in text
    assert "next objective: 12 turns, 3 grace" in text
    assert not agent.backend.calls
    assert not command_notices(agent)    # the query records nothing


def test_live_names_the_last_request_in_the_servers_own_numbers(tmp_path):
    """A turn that reports usage teaches the run the server's rate: /live shows the last
    request's server numbers and the calibration the run learned from them, instead of an
    estimate nobody has checked."""
    turn = [TextDelta("here is the summary"), Usage(2011, 144), Stopped("stop")]
    agent = build([turn], tmp_path)
    out, err = run_session(agent, ["summarize the notes in this folder", "/live", "quit"])
    text = err.getvalue()
    assert "last request: 2011 tokens in, 144 out (server-reported)" in text
    assert "calibrated" in text and "uncorrected" not in text


def test_live_names_an_uncorrected_estimate_when_the_server_never_reported(tmp_path):
    """A backend that reports no usage leaves the estimate bare, and /live says so: a reading
    that pretended to carry a correction it never learned would be a number with a fake pedigree."""
    agent = build([turn_text("ok")], tmp_path)
    out, err = run_session(agent, ["go", "/live", "quit"])
    text = err.getvalue()
    assert "uncorrected" in text
    assert "no request yet" in text


def test_live_without_a_window_says_the_trigger_cannot_fire(tmp_path):
    agent = build([], tmp_path)
    agent.backend.caps = agent.backend.caps.__class__(context_tokens=0,
                                                     model=agent.backend.caps.model)
    out, err = run_session(agent, ["/live", "quit"])
    assert "no window in force" in err.getvalue() and "cannot fire" in err.getvalue()


def test_live_takes_no_argument(tmp_path):
    agent = build([], tmp_path)
    out, err = run_session(agent, ["/live now", "quit"])
    assert "takes no argument" in command_notices(agent)[0]["text"]
