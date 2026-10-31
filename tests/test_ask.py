"""The ask: a refusal that can become a question, and a question a command can answer.

No test spawns a second process. The "operator" writes to the same queue file the loop polls, and
the loop is synchronous, so a pre-seeded matching answer is read on the next poll; the timeout path
uses a tiny ask_timeout with no answer present.
"""

from sah import control
from sah import model
from sah.agent import Agent, YES_WORDS
from sah.record import Log


def save(path: str, text: str) -> str:
    """Write text to a path."""
    return f"saved {path}"


TOOLS = {"save": save}
# The double writes to a path the model typed, which is what the ask/widen tests need the policy to see.
save.access, save.address = "write", "path"
save.description = "Write text to a path."


def build(turns, tmp_path, *, policy, ask_timeout=0.0, attended=False):
    log = Log(tmp_path / "log.jsonl")
    backend = model.Fake(turns, caps=model.Caps(context_tokens=1000, model="test-1"))
    agent = Agent(backend=backend, tools=TOOLS, log=log, policy=policy, max_turns=4,
                  run_id="r1", root=tmp_path, queue=tmp_path / "control.md",
                  ask_timeout=ask_timeout, attended=attended)
    return agent


def kinds(agent):
    return [s["kind"] for s in agent.log.steps()]


def find(agent, kind):
    return [s for s in agent.log.steps() if s["kind"] == kind]


def test_a_refusal_with_no_ask_just_refuses(tmp_path):
    """ask='' is the default; the loop must not invent a question the policy never offered."""
    from sah.contract import Verdict

    def policy(name, args, **_):
        return Verdict(False, "outside the run root")

    turn = [[model.ToolCall("c1", "save", {"path": "x", "text": "t"}),
             model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=policy)
    agent.run()
    assert "request" not in kinds(agent)
    refused = [r for r in find(agent, "result") if not r["ok"]]
    assert refused and "outside the run root" in refused[0]["content"]


class Askable:
    """A policy that refuses but offers an ask, the way `Confine` does for an address it could widen."""

    def __init__(self):
        self.widened: list = []

    def __call__(self, name, args, **_):
        from sah.contract import Verdict
        return Verdict(False, "not granted", address="secret", ask="secret")

    def widen(self, address, *, writing):
        self.widened.append((address, writing))
        return address


def test_a_first_refusal_asks_nobody_even_when_the_policy_offers_an_ask(tmp_path):
    """What CLI-4 replaced: a policy that offered an ask used to interrupt the operator on the *first*
    refusal. A model that was refused and moved on had already cost a question, and a piped one-shot or a
    spawned child -- with nobody to answer -- spent the whole timeout before the no arrived. Asking is the
    last resort, so a first refusal is a refusal, plus one line saying what a repeat would cost.
    """
    turn = [[model.ToolCall("c1", "save", {"path": "secret", "text": "t"}),
             model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=Askable())
    agent.run()
    assert "request" not in kinds(agent), "the first refusal asked a question; asking is the last resort"
    refused = [r for r in find(agent, "result") if not r["ok"]]
    assert refused and "Nobody can grant" in refused[0]["content"], \
        "the model was refused without being told what repeating the call does"


def test_the_second_attempt_at_the_same_need_is_what_asks(tmp_path, monkeypatch):
    """A model that comes back for the same thing has said it cannot finish without it. That is the
    question worth asking -- once, and not before.

    `ask` is replaced rather than answered through the queue because this asserts *when* the loop asks; the
    round trip with a queued answer is `test_an_accepted_ask_widens_confine_and_the_retry_proceeds`.
    """
    asked: list = []

    def fake_ask(subject, item, **_kw):
        asked.append(item)
        return False, "answered no", "queue"

    turn = [[model.ToolCall("c1", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")],
            [model.ToolCall("c2", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=Askable(), attended=True)
    monkeypatch.setattr(agent, "ask", fake_ask)
    agent.run()
    assert asked == ["secret"], f"asking fired {len(asked)} time(s): {asked}; it belongs to the repeat"


def test_a_run_with_no_operator_fails_at_the_second_attempt_instead_of_waiting(tmp_path):
    """Unattended, a repeated need is not a question that goes unanswered -- it is the honest end of the
    run. The behaviour this replaces held each turn for the timeout and then carried on as if the no were
    an answer, so a child touching three paths outside the grant stalled six minutes to achieve what it
    could have reported at once: that it needed an operator.

    The question is still recorded and closed at once, because the `request` step is the log's record of
    what the run needed.
    """
    turn = [[model.ToolCall("c1", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")],
            [model.ToolCall("c2", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=Askable(), ask_timeout=120.0)   # unattended: nobody is waited for
    agent.run()

    req = find(agent, "request")
    assert len(req) == 1 and req[0]["item"] == "secret"
    close = [d for d in find(agent, "decision") if d["refs"] == req[0]["id"]]
    assert close and close[0]["outcome"] == "refused" and "unattended" in close[0]["reason"]
    ended = find(agent, "ended")[-1]
    # Exit 1 is the half that matters to whatever launched the run: a child that needed an operator and
    # had none is not a success, and `ended.exit_code` is the only place the exit code is ever read from.
    assert ended["reason"] == "needs_operator" and ended["exit_code"] == 1
    assert len(find(agent, "message")) == 2, \
        "the run asked the backend again after the need was raised -- it should have stopped"


def test_another_offset_in_the_same_file_is_the_same_need(tmp_path, monkeypatch):
    """The last-resort count keys on the address and the access, not on the arguments typed. Two writes to
    one path with different text is one need twice; counting arguments would restart the last resort
    forever, so a model could ask an operator a hundred questions about one file.
    """
    asked: list = []

    def fake_ask(subject, item, **_kw):
        asked.append(item)
        return False, "answered no", "queue"

    turn = [[model.ToolCall("c1", "save", {"path": "secret", "text": "a"}), model.Stopped("tool_calls")],
            [model.ToolCall("c2", "save", {"path": "secret", "text": "b"}), model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=Askable(), attended=True)
    monkeypatch.setattr(agent, "ask", fake_ask)
    agent.run()
    assert asked == ["secret"], f"different arguments were treated as a different need: {asked}"


def test_a_policy_without_widen_is_not_asked_around(tmp_path):
    """An ask the loop literally cannot grant (no widen hook) is noise; the refusal stands and no
    question goes out."""
    from sah.contract import Verdict

    def policy(name, args, **_):
        return Verdict(False, "hard no", address="x", ask="x")

    assert not hasattr(policy, "widen")
    turn = [[model.ToolCall("c1", "save", {"path": "x", "text": "t"}),
             model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=policy)
    agent.run()
    assert "request" not in kinds(agent)


def test_an_answer_on_the_queue_grants_and_closes_the_request(tmp_path):
    """Full round trip driven through ask(): a matching reply block, delivered by the queue,
    accepted, and it names the answerer rather than pretending to verify them."""
    agent = build([], tmp_path, policy=lambda *a, **kw: None, ask_timeout=0.05)
    control.append(str(tmp_path / "control.md"), "answer", text="yes", sender="root",
                   reply="000000-request")       # ask() is the first step in a fresh log
    granted, note, issuer = agent.ask("widen", "secret.txt", refs="x")
    assert granted is True
    assert issuer == "queue"                       # the transport, not "root"
    assert "root" in note                          # the claim is recorded, not verified
    close = [d for d in find(agent, "decision") if d["refs"] == "000000-request"]
    assert close[0]["outcome"] == "accepted" and close[0]["issuer"] == "queue"


def test_an_answer_is_matched_by_reply_id_not_by_existing(tmp_path):
    """A reply aimed at a different question must not grant this one."""
    agent = build([], tmp_path, policy=lambda *a, **kw: None, ask_timeout=0.05)
    control.append(str(tmp_path / "control.md"), "answer", text="yes", reply="999999-request")
    granted, note, issuer = agent.ask("widen", "thing", refs="x")
    assert granted is False and issuer == "core"   # the 999999 answer is not for 000000


def test_only_a_yes_word_grants(tmp_path):
    agent = build([], tmp_path, policy=lambda *a, **kw: None, ask_timeout=0.5)
    control.append(str(tmp_path / "control.md"), "answer", text="maybe later", reply="000000-request")
    granted, note, _ = agent.ask("widen", "thing", refs="x")
    assert granted is False
    assert "maybe" in note                         # the non-answer is recorded verbatim


def test_an_answer_arriving_at_a_turn_boundary_is_refused(tmp_path):
    """A stray reply with no question waiting is recorded and refused, never silently dropped."""
    agent = build([], tmp_path, policy=lambda *a, **kw: None)
    control.append(str(tmp_path / "control.md"), "answer", text="yes", reply="000000-request")
    agent.run()
    assert any(c["action"] == "answer" for c in find(agent, "command"))
    assert any("boundary" in d.get("reason", "") for d in find(agent, "decision"))


def test_a_stop_command_while_waiting_ends_the_wait(tmp_path):
    """A finish arriving mid-wait closes the question 'no' and stays pending for the boundary,
    so it is applied at the turn edge, not abandoned."""
    agent = build([], tmp_path, policy=lambda *a, **kw: None, ask_timeout=30.0)
    control.append(str(tmp_path / "control.md"), "finish", text="", cancel=True)
    granted, note, issuer = agent.ask("widen", "thing", refs="1-request")
    assert granted is False and "finish" in note
    # the finish was not consumed by the ask; it is still pending for the loop boundary
    assert [b.id for b in control.pending(str(tmp_path / "control.md"))]


def test_the_last_resort_counter_belongs_to_one_objective(tmp_path, monkeypatch):
    """A session runs several objectives over one conversation. A refusal that was already chased to a
    question in the *last* objective must not make this objective's first refusal look like a repeat --
    otherwise, from the second question onward the model is asked-for-grant on its first try at anything,
    which is the exact behaviour CLI-4 removed, arriving by a different route.
    """
    asked: list = []

    def fake_ask(subject, item, **_kw):
        asked.append(item)
        return False, "answered no", "queue"

    one = [model.ToolCall("c1", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]
    two = [model.ToolCall("c2", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]
    three = [model.ToolCall("c3", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]
    four = [model.ToolCall("c4", "save", {"path": "secret", "text": "t"}), model.Stopped("tool_calls")]
    done = [model.TextDelta("done"), model.Stopped("stop")]
    agent = build([one, two, done, three, four, done], tmp_path, policy=Askable(), attended=True)
    monkeypatch.setattr(agent, "ask", fake_ask)
    agent.run("first objective")
    agent.run("second objective")
    # One ask per objective, and it belongs to the second attempt within that objective: the repeat of
    # objective one, and the repeat of objective two. The first refusal of either asked nobody.
    assert asked == ["secret", "secret"], \
        f"asking fired {len(asked)} times across two objectives: {asked}"


def test_an_answered_ask_widens_the_axis_the_tool_declares(tmp_path, monkeypatch):
    """The loop must widen the axis the *tool declared*, not the axis its name suggests.

    `writing` used to mean `call.name == "write"`, which is true of exactly one tool in this tree and
    false of any other that writes — and an ask answered yes on the wrong axis grants something the
    operator never saw. So the tool here is named `save`, declares `write`, and the policy records which
    axis the loop asked it to widen.

    The answer is supplied at the `ask` seam rather than through the queue file: a queued answer must
    carry the request's id, and an id the loop has not minted yet cannot be written in advance — a test
    that guessed the sequence number would break on any added step for reasons that have nothing to do
    with what this asserts.
    """
    from sah.contract import Verdict

    file = str(tmp_path / "note.txt")

    class Policy:
        def __init__(self):
            self.widened: list = []

        def __call__(self, name, args, **_):
            return Verdict(False, "not granted yet", address=file, ask=file)

        def widen(self, address, *, writing):
            self.widened.append((address, writing))
            return address

    policy = Policy()
    turn = [[model.ToolCall("c1", "save", {"path": file, "text": "t"}), model.Stopped("tool_calls")],
            [model.ToolCall("c2", "save", {"path": file, "text": "t"}), model.Stopped("tool_calls")]]
    agent = build(turn, tmp_path, policy=policy, attended=True)
    monkeypatch.setattr(agent, "ask", lambda *a, **k: (True, "answered yes", "queue"))
    agent.run()

    assert policy.widened, "an answered ask never reached the policy"
    assert policy.widened[0] == (file, True), (
        f"a tool declared write widened {policy.widened[0]!r}; the write axis is the one the operator "
        f"was asked about")


def test_yes_words_are_a_closed_set():
    assert {"yes", "y", "allow", "grant"} <= YES_WORDS
    assert "no" not in YES_WORDS and "maybe" not in YES_WORDS


def test_an_accepted_ask_widens_confine_and_the_retry_proceeds(tmp_path):
    """The real round trip against the real policy: confine refuses a read outside the grant, the
    operator answers yes on the queue, confine widens to that one address, and the same read that
    was just refused is now allowed -- while its sibling in the same folder is still refused. That
    last clause is the whole point of asking instead of advising a parent grant."""
    from sah import confine as confine_mod
    from sah.confine import Confine

    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("secret", encoding="utf-8")
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("no", encoding="utf-8")

    policy = Confine(read_roots=[str(root)], write_roots=[])
    assert not policy("read", {"path": str(outside)}, access="read", address="path").allowed
    # refused, and it offers an ask

    agent = build([], tmp_path / "widening", policy=policy, ask_timeout=0.05)
    control.append(str(tmp_path / "widening" / "control.md"), "answer", text="yes",
                   reply="000000-request")
    granted, note, issuer = agent.ask("widen", confine_mod.resolve(str(outside)),
                                      why="one file", refs="c1-call")
    assert granted and issuer == "queue"

    policy.widen(confine_mod.resolve(str(outside)), writing=False)
    assert policy("read", {"path": str(outside)}, access="read", address="path").allowed
    # the one address now reads
    assert not policy("read", {"path": str(sibling)}, access="read", address="path").allowed
    # not the folder around it


def test_an_open_deadline_still_honours_the_answer(tmp_path):
    """`math.inf` as the timeout is how an attended run says "a person may answer at any hour", and the
    loop must not read the absence of a deadline as an expiry: the poll runs before the deadline is
    consulted, so an answer waiting on the queue is taken on the first pass whatever the deadline says.

    Honest about what this does NOT cover: because the poll precedes the check, no answer-present test can
    tell `deadline = monotonic() + inf` from `deadline = monotonic()`. The arithmetic is one line and is
    checked by mutation at the CLI level (the default), not here."""
    import math

    agent = build([], tmp_path, policy=lambda *a, **kw: None, ask_timeout=math.inf)
    control.append(str(tmp_path / "control.md"), "answer", text="yes", reply="000000-request")
    granted, note, issuer = agent.ask("widen", "secret.txt", refs="x")
    assert granted is True and issuer == "queue"
    assert "no answer" not in note, note
