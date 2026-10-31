"""The taxonomy is a decision, so it is tested as one: the set of kinds, which of them the
model sees, which fields carry a verdict, and the refusals that keep a step honest."""
import json

import pytest

from sah import record


ALL_KINDS = {
    "message", "call", "result", "command", "request", "decision", "operation",
    "summary", "notice", "started", "ended", "cut",
}


def test_the_kind_set_is_exactly_the_declared_twelve():
    assert set(record.KINDS) == ALL_KINDS


def test_four_kinds_are_model_facing_and_that_is_the_whole_context_surface():
    assert record.TO_MODEL == {"message", "call", "result", "summary"}


def test_the_authority_subset_is_empty_by_decision():
    """A resume re-applies conversation content, not authority: what is in force comes from
    the invocation. Adding a member here means the reasoning lives beside the set it protects."""
    assert record.ESTABLISHING == frozenset()


def test_the_verdict_fields_are_exactly_four():
    assert record.MUST_SURFACE == {
        ("message", "outcome"), ("result", "ok"),
        ("decision", "outcome"), ("operation", "outcome"),
    }


def test_no_kind_declares_a_banned_field():
    for name, spec in record.KINDS.items():
        assert not set(spec.fields) & record.BANNED_FIELDS, f"{name} declares a banned field"


def test_no_kind_shadows_the_word_that_names_its_type():
    for name, spec in record.KINDS.items():
        assert "kind" not in spec.fields and "issuer" not in spec.fields, name


def test_a_step_may_not_carry_an_undeclared_field():
    with pytest.raises(record.Refused, match="does not declare"):
        record.make("notice", "core", 1, subject="nudge", text="y", detail="sneaked in")


def test_required_fields_are_enforced():
    with pytest.raises(record.Refused, match="requires"):
        record.make("operation", "core", 1, op="write", address="/tmp/x")


def test_an_int_field_rejects_a_bool():
    with pytest.raises(record.Refused, match="must be int"):
        record.make("operation", "core", 1, op="write", address="/tmp/x",
                    outcome="allowed", bytes=True)


def test_a_bool_field_rejects_an_int():
    with pytest.raises(record.Refused, match="must be bool"):
        record.make("result", "core", 1, refs="000001-call", ok=1)


def test_a_rate_the_server_measured_is_a_declared_float():
    """A throughput figure is a quantity the world measured, which is the line between it and the
    rendered strings the banned names collected: `tps` is the server's own count of its own work, and a
    reader who wants a duration still derives it. Only a float is a rate — an integer, a word or a bool
    is not a measurement this taxonomy will store, and refusing it here is what keeps the adapter (which
    coerces) and the record agreeing."""
    step = record.make("message", "model", 1, role="assistant", text="hi", outcome="ok",
                       tps=45.2, prefill_tps=881.0)
    assert record.loads(record.dumps(step)) == step        # a rate survives the wire
    for not_a_rate in ("45", 45, True, None):
        with pytest.raises(record.Refused, match="must be float"):
            record.make("message", "model", 1, role="assistant", text="hi", tps=not_a_rate)


def test_a_rate_is_declared_only_where_a_turn_produced_it():
    """The rates describe one request, so they live on `message` and nowhere else — a `result` has no
    throughput, and a `started` that carried a rate would be a number about a request that had not
    happened."""
    assert record.KINDS["message"].fields["tps"][0] == record.FLOAT
    assert record.KINDS["message"].fields["prefill_tps"][0] == record.FLOAT
    for name in ("call", "result", "started", "ended", "summary"):
        assert not {"tps", "prefill_tps"} & set(record.KINDS[name].fields), name


def test_the_caller_may_not_supply_the_stamping():
    """`id`, `ts` and `issuer` are derived by whoever appends. Passing them in would let the
    code describing an act also choose who did it."""
    with pytest.raises(record.Refused):
        record.make("notice", "core", 1, subject="nudge", text="y", ts="2020-01-01T00:00:00+00:00")
    with pytest.raises(record.Refused, match="issuer"):
        record.make("notice", "", 1, subject="nudge", text="y")


def test_an_unknown_kind_is_refused_by_name():
    with pytest.raises(record.Refused, match="unknown step kind 'turn'"):
        record.make("turn", "core", 1, role="user", text="the old schema is gone")


def test_refs_is_a_list_of_ids():
    with pytest.raises(record.Refused, match="list of step ids"):
        record.make("summary", "core", 1, refs=[1, 2], kept=3, dropped=4, text="s",
                    outcome="ok")


def test_a_step_survives_the_wire_unchanged():
    step = record.make("message", "model", 7, role="assistant", text="hi",
                       outcome="ok", tokens_out=3, context_used=90, context_max=1000)
    assert record.loads(record.dumps(step)) == step


def test_one_step_is_one_line():
    line = record.dumps(record.make("notice", "core", 1, subject="nudge", text="y"))
    assert "\n" not in line and line.startswith("{")


def test_a_line_the_taxonomy_cannot_name_is_refused_not_skipped():
    with pytest.raises(record.Refused, match="undeclared kind"):
        record.loads('{"kind": "telemetry", "id": "1", "ts": "t", "issuer": "core", "data": {}}')


def test_a_parsed_step_with_an_unnamed_field_is_refused():
    step = record.make("notice", "core", 1, subject="nudge", text="y")
    step["extra"] = "drift"
    with pytest.raises(record.Refused, match="undeclared field"):
        record.validate(step)


def test_a_subscriber_sees_each_step_as_it_is_written(tmp_path):
    log = record.Log(tmp_path / "log.jsonl")
    seen = []
    log.subscribe(lambda step: seen.append((step["kind"], step["id"])))
    log.append("message", "user", role="user", text="one")
    log.append("notice", "core", subject="nudge", text="two")
    assert seen == [("message", "000000-message"), ("notice", "000001-notice")]


def test_a_subscriber_cannot_break_the_record(tmp_path, capsys):
    """A client's bug must not stop the log: the step is written first, and the failure is
    reported rather than swallowed."""
    log = record.Log(tmp_path / "log.jsonl")

    def explode(step):
        raise KeyError("the subscriber is broken")

    log.subscribe(explode)
    step = log.append("message", "user", role="user", text="kept")
    assert log.steps() == [step]
    assert "a step subscriber failed" in capsys.readouterr().err


def test_a_subscriber_only_sees_what_is_already_written(tmp_path):
    """A projection reports facts, so it must not be able to announce a step the file does not
    hold yet: the line reaches the log first, and only then does anything hear about it."""
    log = record.Log(tmp_path / "log.jsonl")
    lines_at_report = []
    log.subscribe(lambda step: lines_at_report.append(
        (step["id"], sum(1 for _ in open(tmp_path / "log.jsonl", encoding="utf-8")))))
    log.append("message", "user", role="user", text="one")
    log.append("notice", "core", subject="nudge", text="two")
    assert lines_at_report == [("000000-message", 1), ("000001-notice", 2)]


# --- the closed vocabulary: ident fields that subtype a step --------------------

def test_a_notice_subject_must_be_from_the_declared_vocabulary():
    """`ident` promises a name the reader can enumerate; only a owned list makes that true. A free
    string here means a future 'what notices exist?' question has no answer short of a full scan."""
    with pytest.raises(record.Refused, match="must be one of"):
        record.make("notice", "core", 1, subject="spwn", text="typo'd subject")
    assert record.make("notice", "core", 1, subject="spawn", text="ok")["subject"] == "spawn"
    assert record.make("notice", "core", 1, subject="replay", text="ok")["subject"] == "replay"
    # A run that made room after a server refused the request for its size says so here. It is a notice
    # and not a nudge: a nudge is the guard warning a run that repeats, this is the harness spending a
    # request to forget on purpose, and a reader asking "why did this run summarise?" needs the two
    # separable.
    assert record.make("notice", "core", 1, subject="compact", text="ok")["subject"] == "compact"


def test_a_vocabulary_violation_is_caught_on_read_not_just_write():
    """A log line hand-edited or written by an older build with an unknown subject must not load as
    if it were fine -- the reader trusts the vocabulary, so loads enforces it too."""
    line = '{"kind": "notice", "id": "0-x", "ts": "t", "issuer": "core", "subject": "bogus", "text": "y"}'
    with pytest.raises(record.Refused, match="must be one of"):
        record.loads(line)


def test_an_open_ident_field_still_accepts_any_name():
    """Not every ident is closed: a tool `name` grows with the tools, and a wrong enum there would
    refuse a legitimate new tool. Only (kind, field) pairs listed in VOCABULARY are constrained."""
    step = record.make("call", "model", 1, refs="0-message", name="brand_new_tool", args={})
    assert step["name"] == "brand_new_tool"


def test_the_taxonomy_rejects_a_vocabulary_for_a_field_it_does_not_own():
    """A vocabulary that names a non-ident or undeclared field is a bug in the taxonomy itself, so
    it fails at import, before any log is written."""
    saved = dict(record.VOCABULARY)
    try:
        record.VOCABULARY[("notice", "text")] = frozenset({"only this"})   # text is TEXT not IDENT
        with pytest.raises(record.Refused, match="not a declared ident field"):
            record._self_check()
    finally:
        record.VOCABULARY.clear()
        record.VOCABULARY.update(saved)


# --- the sequence belongs to the file, not to the object -------------------------------------

def test_a_reopened_log_continues_the_sequence(tmp_path):
    """Resume, `sah export` and the REPL each open a fresh Log over the same log. A counter that
    restarted at zero would mint ids that `refs` already points at, so an operator answering
    `--reply <id>` could be answering an older question that happens to carry the same number."""
    path = tmp_path / "log.jsonl"
    record.Log(path).append("notice", "core", subject="nudge", text="first session")
    step = record.Log(path).append("notice", "core", subject="nudge", text="second session")
    assert step["id"] == "000001-notice"
    assert [s["id"] for s in record.Log(path).steps()] == ["000000-notice", "000001-notice"]


def test_a_reopened_log_counts_a_line_it_cannot_parse(tmp_path):
    """A run killed mid-append leaves a torn final line whose id was already spent. Reissuing it
    would give one log two steps under one name, so the next number comes from the lines in the
    file rather than from the lines that happen to parse — and an append never glues a new step
    onto a line that was cut off without its newline."""
    path = tmp_path / "log.jsonl"
    record.Log(path).append("notice", "core", subject="nudge", text="whole")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"notice","id":"000001-noti')          # killed here: no newline
    step = record.Log(path).append("notice", "core", subject="nudge", text="after")
    assert step["id"] == "000002-notice"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, "the new step starts its own line instead of welding onto the torn one"
    assert lines[1] == '{"kind":"notice","id":"000001-noti'


# --- one reader, tolerant only of the tail it can name ---------------------------------------

def test_a_torn_tail_is_dropped_and_named_rather_than_refused(tmp_path):
    """A run killed mid-append leaves a final line with no end. Every reader that projects a run —
    resume, `sah export`, the REPL, a parent asking after a child — would otherwise say nothing at
    all about a run that is merely damaged. The tolerant form drops that one line and says it did;
    the strict form keeps refusing, because an audit that skips lines is decoration."""
    path = tmp_path / "log.jsonl"
    log = record.Log(path)
    log.append("notice", "core", subject="nudge", text="one")
    log.append("notice", "core", subject="nudge", text="two")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"notice","id":"000002-noti')          # killed here: no newline
    steps, damage = record.read(path, tolerate_damage=True)
    assert [s["id"] for s in steps] == ["000000-notice", "000001-notice"]
    assert damage and "line 3 of 3" in damage, f"a line was dropped in silence: {damage!r}"
    with pytest.raises(record.Refused):
        record.read(path)                                     # the audit form still refuses
    with pytest.raises(record.Refused):
        record.Log(path).steps()


def test_damage_is_dropped_and_named_wherever_it_sits_not_only_at_the_end(tmp_path):
    """Refusing mid-file damage looked safer than skipping it, and the argument was gluing: a torn
    line welded onto its neighbour carried a good step down with it. That argument retired with the
    repair in `append`, which closes a torn line first -- so an unreadable line is now self-contained
    and skipping it loses exactly the one step it holds. Refusing it instead would make the log that
    crashed once unreadable forever, because the next step any run writes moves the damage off the
    end. Tolerance is therefore for damage anywhere, named line by line; the audit form still
    refuses the whole file."""
    path = tmp_path / "log.jsonl"
    log = record.Log(path)
    for text in ("one", "two", "three"):
        log.append("notice", "core", subject="nudge", text=text)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"kind":"notice","id":"000001-noti'        # damage, and not at the end
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    steps, damage = record.read(path, tolerate_damage=True)
    assert [s["id"] for s in steps] == ["000000-notice", "000002-notice"]
    assert "line 2 of 3" in damage, f"the dropped line was not named: {damage!r}"
    assert "1 line(s)" in damage
    with pytest.raises(record.Refused):
        record.read(path)                                   # the audit form still refuses
    lines[2] = '{"kind":"notice","id":"000002-no'          # a second loss
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    steps, damage = record.read(path, tolerate_damage=True)
    assert [s["id"] for s in steps] == ["000000-notice"]
    assert "2 line(s)" in damage and "line 3 of 3" in damage


def test_lenience_covers_vocabulary_and_not_damage(tmp_path):
    """The distinction the projection reader turns on: a word this build has never seen is the future and
    gets kept, while a declared field holding the wrong type is corruption and is still refused.

    Without that line the lenient reader is a general "parse whatever arrives" mode, and a torn line
    would reach a report dressed up as an unfamiliar feature — which is precisely the confusion between
    an old export and a broken log that the two reader policies exist to keep apart."""
    path = tmp_path / "log.jsonl"
    record.Log(path).append("started", "core", run_id="r1", root="/r")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "result", "id": "z", "ts": "t", "issuer": "core",
                                 "refs": "c1", "ok": "yes"}) + "\n")
    with pytest.raises(record.Refused) as exc:
        record.loads(open(path, encoding="utf-8").readlines()[1])
    assert "must be" in str(exc.value)
    steps, damage = record.read(path, tolerate_damage=True, strict=False)
    assert [s["kind"] for s in steps] == ["started"], "a mistyped declared field was kept as a step"
    assert "1 line(s)" in damage, "the loss was not reported"


def test_a_line_without_a_kind_the_reader_can_name_is_not_a_step(tmp_path):
    """Lenient does not mean permissive about shape: `kind` is what a step *is*, and a line whose kind is
    a number, or absent, or a JSON array wearing a step's clothes, is not a step in either policy."""
    for line in ('{"kind": 5}', '{"issuer": "core"}', '[{"kind": "notice"}]', '"notice"'):
        for strict in (True, False):
            with pytest.raises(record.Refused):
                record.loads(line, strict=strict)


def test_a_whole_log_reports_no_damage(tmp_path):
    """The note is a report, not a ritual: a caller prints it only when a line was actually lost, so
    the ordinary case stays silent and a log that never existed is an empty history, not an error."""
    path = tmp_path / "log.jsonl"
    record.Log(path).append("notice", "core", subject="nudge", text="one")
    steps, damage = record.read(path, tolerate_damage=True)
    assert len(steps) == 1 and damage == ""
    assert record.read(tmp_path / "absent.jsonl", tolerate_damage=True) == ([], "")


def test_the_reach_field_takes_a_structure_and_never_a_sentence(tmp_path):
    """`started.grants` describes a policy rather than an act, and it now has both a writer (the loop, asked
    of the policy that refuses) and a reader (`sah export`, which had carried the line since it was written
    and rendered nothing). A rendered sentence in this field would be the banned `detail` dumping ground in a
    new coat — a list of roots is greppable, `read+write+exec on .` is not."""
    log = record.Log(tmp_path / "log.jsonl")
    reach = {"read": ["x:/notes"], "write": [], "exec": True}
    step = log.append("started", "core", run_id="r", root=".", grants=reach)
    assert step["grants"] == reach
    back = record.Log(tmp_path / "log.jsonl").steps()[0]
    assert back["grants"] == reach, "structure must survive the round trip, not arrive rendered"
    with pytest.raises(record.Refused):
        log.append("started", "core", run_id="r2", root=".", grants="read+write+exec")


# --- closed vocabularies -------------------------------------------------------

def test_the_verdict_fields_the_code_already_fixes_are_closed():
    """`ident` is documented as "a name from a vocabulary the reader can enumerate", which is only true if
    somebody owns the list.

    These five are fixed by the code, not by a growing set of tools: the way a run ended, and the verdicts a
    decision, an operation and a summary carry. Leaving them open meant a reader asking "which runs failed"
    had to prefix-match, because an error sentence glued onto a verdict turns every error into a new outcome.
    """
    for pair in {("ended", "reason"), ("decision", "outcome"), ("operation", "outcome"),
                 ("summary", "outcome"), ("summary", "slot"), ("message", "outcome")}:
        assert pair in record.VOCABULARY, \
            f"{pair[0]}.{pair[1]} is a verdict the code fixes but nobody enumerates"


def test_the_way_a_run_ended_is_a_verdict_word():
    """Declared TEXT while every value the loop produces was one of seven words — which is why it could not be
    closed as it stood."""
    assert record.KINDS["ended"].fields["reason"][0] == record.IDENT
    for word in sorted(record.VOCABULARY[("ended", "reason")]):
        step = record.make("ended", "core", 1, reason=word, exit_code=0)
        assert record.loads(record.dumps(step))["reason"] == word
    with pytest.raises(record.Refused) as got:
        record.make("ended", "core", 1, reason="the model stalled twice, sorry", exit_code=1)
    assert "ended" in str(got.value) and "reason" in str(got.value)


def test_a_verdict_with_a_sentence_glued_to_it_is_refused():
    """The exact shape the ladder used to return: a verdict plus the server's words in one field. The sentence
    belongs in `reason`, which the kind declares; the verdict stays one word."""
    with pytest.raises(record.Refused) as got:
        record.make("summary", "core", 1, refs=[], kept=1, dropped=2, text="t",
                    outcome="failed: the socket died", reason="the socket died")
    assert "summary" in str(got.value) and "outcome" in str(got.value)


def test_the_closed_sets_are_exactly_the_words_the_loop_produces():
    """A wrong enum refuses a legitimate future value, which is worse than leaving a set open — so the
    sets are pinned as a decision, and an addition has to be made here deliberately rather than
    discovered at runtime."""
    assert record.VOCABULARY[("ended", "reason")] == {
        "complete", "finished", "interrupted", "failed", "stalled", "max_turns", "max_runtime",
        # the last found by closing the set, not by recall
        "truncated", "max_turns", "truncated", "needs_operator", "blocked", "crashed"}
    assert record.VOCABULARY[("decision", "outcome")] == {"accepted", "refused"}
    assert record.VOCABULARY[("operation", "outcome")] == {"allowed", "refused"}
    assert record.VOCABULARY[("summary", "outcome")] == {"ok", "failed"}
    assert record.VOCABULARY[("summary", "slot")] == {"summary", "cut", "failed"}
    assert record.VOCABULARY[("message", "outcome")] == {"ok", "failed", "truncated"}


def test_the_slot_kind_is_a_verdict_word():
    """The slot field is the kind of text the privileged slot holds, and a `summary` step whose slot
    names something the loop never writes is a record describing a compaction that did not happen:
    the same closed-vocabulary rule as `outcome`, applied to the slot's kind."""
    for word in sorted(record.VOCABULARY[("summary", "slot")]):
        step = record.make("summary", "core", 1, refs=[], kept=1, dropped=2, text="t",
                           outcome="ok", slot=word, stage=4, budget="limit 63; room 7")
        assert record.loads(record.dumps(step))["slot"] == word
    with pytest.raises(record.Refused) as got:
        record.make("summary", "core", 1, refs=[], kept=1, dropped=2, text="t",
                    outcome="ok", slot="a mostly complete summary")
    assert "summary" in str(got.value) and "slot" in str(got.value)


def test_a_log_written_before_the_set_was_closed_still_projects(tmp_path):
    """Closing a vocabulary has a cost, and it lands on old logs: a summary written before the failure verdict
    was separated stored `outcome: "failed: <error>"`, which this build cannot name.

    The audit reader must still refuse it — "what did this run do" is not answered by a step nothing checked.
    A projection keeps it and prints what the step declared, because a log the build predates is the case
    leniency exists for. Refusing both ways would make closing a set destroy history, and a set that cannot be
    closed without a data migration is a set that never gets closed."""
    line = json.dumps({"kind": "summary", "id": "000001-summary", "ts": record.now(), "issuer": "core",
                       "refs": [], "kept": 1, "dropped": 2, "text": "(no summary produced)",
                       "outcome": "failed: the socket died"})
    path = tmp_path / "log.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(record.Refused):
        record.read(str(path))
    steps, damage = record.read(str(path), tolerate_damage=True, strict=False)
    assert [s["outcome"] for s in steps] == ["failed: the socket died"], damage
    assert damage == "", f"a kept line was reported as lost: {damage}"

def test_a_list_of_the_wrong_things_is_not_reported_as_a_list():
    """The REFS check that looks duplicated is the only thing making that refusal readable.

    `_fits` does validate the elements, so removing the check in `make` keeps a refusal — and measured, the
    refusal it keeps is `summary.refs must be <class 'list'>, got list`: the declared kind for REFS is
    `list`, so the generic message asserts the value is a list about a value that is one and never names
    what was wrong with its elements. Removing the "duplicate" reddens this test and the older one beside
    it. Pinned so the second cleanup does not succeed where the first was stopped.
    """
    with pytest.raises(record.Refused) as got:
        record.make("summary", "core", 1, refs=["ok", 7], kept=1, dropped=2, text="s", outcome="ok")
    assert "got list" not in str(got.value), f"the refusal called a list something else: {got.value}"


def test_the_pictures_a_result_attached_are_paths_and_not_step_ids(tmp_path):
    """`images` names the files whose bytes were handed to the model, not steps. Declared as REFS it read
    as "the steps this result refers to", and the element check that keeps REFS a list of ids refuses a
    path — so the field could never hold what its name promises, which is how a field ends up declared and
    unwritten. PATHS is its own kind: a list of places, no id vocabulary claimed.

    The field is also what a resumed run re-reads to rebuild a picture, so the round trip is asserted, not
    just the write."""
    log = record.Log(tmp_path / "log.jsonl")
    said = log.append("message", "model", role="assistant", text="read it", outcome="ok")
    call = log.append("call", "model", refs=said["id"], name="read", args={"path": "x.png"})
    log.append("result", "core", refs=call["id"], ok=True, content="read x.png", images=["x.png"])
    assert log.steps()[-1]["images"] == ["x.png"], log.steps()[-1]
    with pytest.raises(record.Refused, match="images"):
        record.make("result", "core", 2, refs=call["id"], ok=True, content="x", images="x.png")
    # and an element that is not a path is refused *as a path*, so the refusal says what the field wanted
    with pytest.raises(record.Refused, match="images must be a list of paths"):
        record.make("result", "core", 2, refs=call["id"], ok=True, content="x", images=["x.png", 4])


def test_the_head_fields_and_the_wire_pin_round_trip(tmp_path):
    """The head of the render input (`system`, `tools`, `max_tokens`) and the per-turn wire pin
    (`finish`, `req_sha`, `req_chars`) are declared fields that round-trip through the log. They
    are optional, so a line that lacks them — an old log — still reads: closing the record must
    not refuse the history it was written for."""
    log = record.Log(tmp_path / "log.jsonl")
    log.append("started", "core", run_id="r1", root="/r",
               system="you are a test",
               tools=[{"name": "echo", "description": "d", "parameters": {"type": "object"}}],
               max_tokens=512, prefix_digest="a" * 16, resident_tokens=10)
    log.append("message", "model", role="assistant", text="hi",
               finish="stop", req_sha="b" * 16, req_chars=123)
    steps = log.steps()
    assert steps[0]["system"] == "you are a test"
    assert steps[0]["tools"][0]["name"] == "echo"
    assert steps[0]["max_tokens"] == 512
    assert steps[0]["prefix_digest"] == "a" * 16
    assert steps[0]["resident_tokens"] == 10
    assert steps[1]["finish"] == "stop"
    assert steps[1]["req_sha"] == "b" * 16
    assert steps[1]["req_chars"] == 123
    # Optional and lenient with history: a `started` line with no head still reads.
    log.append("started", "core", run_id="r2", root="/r")
    r2 = [s for s in log.steps() if s.get("run_id") == "r2"][0]
    assert "system" not in r2 and "tools" not in r2 and "max_tokens" not in r2


def test_the_head_and_the_pin_are_refused_where_they_do_not_belong():
    """The head is on `started` and the per-turn pin is on `message`, and nowhere else: a `result`
    has no request fingerprint (a result is not a request), and a `call` has no finish word (a call
    is what the model asked for, not how the server stopped). A field on the wrong kind is a record
    describing a fact another step owns."""
    with pytest.raises(record.Refused, match="does not declare"):
        record.make("result", "core", 1, refs="000001", ok=True, content="x", req_sha="a" * 16)
    with pytest.raises(record.Refused, match="does not declare"):
        record.make("call", "model", 2, refs="000001", name="read", finish="stop")


# --- cuts: the projection a cut leaves in force -----------------------------------------------


def _msg(seq, text="m", role="user"):
    return record.make("message", "model", seq, role=role, text=text)


def test_lineage_cuts_the_conversation_at_the_step_it_names():
    """A `cut` says the conversation is the prefix ending at the step it names: the tail the
    file still holds is out of the projection, and the cut step itself is an instruction, not
    a step of the conversation, so it is out too."""
    steps = [_msg(0, "a"), _msg(1, "b"), _msg(2, "c"),
             record.make("cut", "core", 3, to="000000-message"),
             _msg(4, "d")]
    assert [s["id"] for s in record.lineage(steps)] == ["000000-message", "000004-message"]


def test_lineage_composes_a_cut_onto_a_cut():
    """A later cut may aim at a step of the lineage an earlier cut created, and a step the
    first cut detached never re-enters: the walk truncates what is in force, not the file."""
    steps = [_msg(0), _msg(1), _msg(2),
             record.make("cut", "core", 3, to="000001-message"),
             _msg(4), _msg(5), _msg(6),
             record.make("cut", "core", 7, to="000005-message")]
    assert [s["id"] for s in record.lineage(steps)] == [
        "000000-message", "000001-message", "000004-message", "000005-message"]


def test_lineage_refuses_a_cut_that_names_no_step_of_the_conversation_in_force():
    """A cut whose target the list does not hold is a corruption, not a gap to skip: the
    projection cannot be answered honestly, so it refuses, and the file itself is untouched,
    so the audit still reads it whole."""
    steps = [_msg(0), record.make("cut", "core", 1, to="000005-message")]
    with pytest.raises(record.Refused, match="not in the conversation"):
        record.lineage(steps)


def test_open_calls_is_empty_when_every_call_in_the_list_has_its_result():
    steps = [record.make("call", "model", 0, refs="000000-message", name="read"),
             record.make("result", "core", 1, refs="000000-call", ok=True)]
    assert record.open_calls(steps) == []


def test_open_calls_names_a_call_its_result_is_not_in_the_list():
    steps = [record.make("call", "model", 0, refs="000000-message", name="read"),
             _msg(1)]
    assert [s["id"] for s in record.open_calls(steps)] == ["000000-call"]
