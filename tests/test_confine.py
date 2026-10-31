"""The veto point. Every test here is about a *place*: what was granted, where the address really
is, and whether the refusal tells the reader which act would have permitted it."""
import os
import pathlib
import re

import pytest

from sah import confine
from sah.tools import fs
from sah.confine import Confine, inside, resolve


@pytest.fixture
def yard(tmp_path):
    """A folder that is granted, and a folder outside it."""
    granted = tmp_path / "granted"
    granted.mkdir()
    (granted / "note.md").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets").write_text("no", encoding="utf-8")
    return granted, outside


def test_a_tool_is_confined_by_what_it_declares_not_by_its_name(yard):
    """The table this replaces was `ADDRESS_ARG = {"read": "path", "write": "path", "ls": "path"}`, so a
    tool called `cat` was unconfined no matter what it opened. Dispatching on the declaration means the
    next filesystem tool is confined by the line written next to it, and forgetting that line is a boot
    failure rather than a hole.
    """
    granted, outside = yard
    policy = Confine(read_roots=[granted])
    verdict = policy("cat", {"path": str(outside / "secrets")}, access="read", address="path")
    assert not verdict.allowed
    assert "--grant" in verdict.reason
    assert policy("cat", {"path": str(granted / "note.md")}, access="read", address="path").allowed


def test_a_declared_write_is_judged_on_the_write_axis_whatever_its_name(yard):
    """`writing` used to mean `name == "write"`, so any other tool that wrote was judged as a reader and
    the write grant was never consulted."""
    granted, outside = yard
    policy = Confine(read_roots=[granted])            # readable somewhere, nothing granted for writes
    verdict = policy("put", {"path": str(outside / "secrets")}, access="write", address="path")
    assert not verdict.allowed
    assert "--write" in verdict.reason, "a write must be refused with the act that widens writes"


def test_a_declared_exec_takes_the_exec_axis_whatever_its_name(yard):
    granted, _ = yard
    verdict = Confine(read_roots=[granted])("sh", {"command": "ls"}, access="exec", address="cwd")
    assert not verdict.allowed and "--exec" in verdict.reason
    verdict = Confine(allow_exec=True)("sh", {"command": "ls"}, access="exec", address="cwd")
    assert verdict.allowed and "does not bind" in verdict.reason


def test_a_tool_that_confines_itself_is_allowed_and_its_place_is_recorded_as_typed(yard, monkeypatch):
    """`memory`, `recall`, `spawn` and `web` police their own ground, so the policy must not judge them.
    It still records the place they name, because the log is what a later reader asks "what did that run
    touch" -- and it records it *as typed*, because `a.md` is relative to the memory tree, a child id to
    the run's children/, and a URL to no folder at all. Resolving one against the working directory puts
    a path in the log that does not exist anywhere.
    """
    granted, _ = yard
    verdict = Confine()("mem_read", {"path": str(granted / "note.md")}, access="self", address="path")
    assert verdict.allowed
    assert verdict.address == str(granted / "note.md")

    monkeypatch.chdir(granted)
    tree_relative = Confine()("mem_read", {"path": "notes/a.md"}, access="self", address="path")
    assert tree_relative.address == "notes/a.md", (
        f"a tree-relative note was recorded as {tree_relative.address!r}, a place it is not")


def test_an_exec_call_records_the_directory_it_ran_in(yard):
    """The shell is the one tool whose address the policy declines to judge and still reports: `bash`
    with `cwd` empty ran in the run's folder, and a log that recorded nothing there would lose the only
    filesystem fact the call had."""
    granted, _ = yard
    verdict = Confine(allow_exec=True)("bash", {"command": "dir", "cwd": str(granted)},
                                       access="exec", address="cwd")
    assert verdict.allowed and verdict.address == str(granted)
    empty = Confine(allow_exec=True)("bash", {"command": "dir"}, access="exec", address="cwd")
    assert empty.allowed and not empty.address


def test_a_tool_that_touches_no_place_is_allowed_without_one():
    verdict = Confine()("runs", {}, access="self", address=None)
    assert verdict.allowed and not verdict.address


def test_an_access_the_policy_does_not_understand_is_refused_not_allowed():
    """`tools.declaration` checks the closed set at boot, so reaching the policy with anything else is a
    bug. A bug in the policy must land on the safe side of its own default."""
    verdict = Confine(read_roots=["/"])("x", {"path": "/etc/passwd"}, access="mystery", address="path")
    assert not verdict.allowed


def test_a_granted_address_is_allowed_and_comes_back_resolved(yard):
    granted, _ = yard
    verdict = Confine(read_roots=[granted])("read", {"path": str(granted / "note.md")},
                                            access="read", address="path")
    assert verdict.allowed
    assert verdict.address == resolve(str(granted / "note.md"))


def test_an_address_outside_every_grant_is_refused_naming_the_act(yard):
    granted, outside = yard
    verdict = Confine(read_roots=[granted])(
        "read", {"path": str(outside / "secrets")}, access="read", address="path")
    assert not verdict.allowed
    assert "--grant" in verdict.reason, "a refusal the reader cannot act on is a dead end"
    assert resolve(str(outside / "secrets")) in verdict.reason
    assert "asked as" in verdict.reason, "the place judged and the place typed are both shown"


def test_nothing_granted_says_so_instead_of_complaining_about_the_path(yard):
    granted, _ = yard
    verdict = Confine()("read", {"path": str(granted / "note.md")}, access="read", address="path")
    assert not verdict.allowed and "nothing is granted" in verdict.reason


def test_a_write_needs_a_write_grant_even_where_reads_are_granted(yard):
    granted, _ = yard
    verdict = Confine(read_roots=[granted])(
        "write", {"path": str(granted / "new.md")}, access="write", address="path")
    assert not verdict.allowed and "--write" in verdict.reason


def test_a_write_grant_contains_its_own_read_grant(yard):
    granted, _ = yard
    policy = Confine(write_roots=[granted])
    assert policy("read", {"path": str(granted / "note.md")}, access="read", address="path").allowed
    assert policy("write", {"path": str(granted / "new.md")}, access="write", address="path").allowed


def test_a_relative_path_is_measured_against_where_the_run_started(yard, monkeypatch):
    granted, outside = yard
    monkeypatch.chdir(granted)
    assert Confine(read_roots=[granted])("read", {"path": "note.md"}, access="read", address="path").allowed
    monkeypatch.chdir(outside)
    verdict = Confine(read_roots=[granted])("read", {"path": "secrets"}, access="read", address="path")
    assert not verdict.allowed


def test_dot_dot_does_not_walk_out_of_a_grant(yard, monkeypatch):
    granted, outside = yard
    monkeypatch.chdir(granted)
    verdict = Confine(read_roots=[granted])("read", {"path": os.path.join("..", "outside",
                                                                          "secrets")},
                                             access="read", address="path")
    assert not verdict.allowed and "--grant" in verdict.reason


def test_a_symlink_is_judged_at_where_it_points(tmp_path, monkeypatch):
    """The link is simulated by replacing `realpath`, because creating one needs privilege on
    Windows. What is under test is the rule, not the filesystem: the policy judges the resolved
    location, so a name inside a grant that lands outside it is refused and one that lands inside
    is allowed."""
    granted = str(tmp_path / "granted")
    escaping = os.path.join(granted, "link")
    staying = os.path.join(granted, "inner")
    targets = {os.path.normcase(escaping): os.path.join(str(tmp_path), "outside", "secrets"),
               os.path.normcase(staying): os.path.join(granted, "note.md")}
    monkeypatch.setattr(confine.os.path, "realpath",
                        lambda path: targets.get(os.path.normcase(path), path))

    verdict = Confine(read_roots=[granted])("read", {"path": escaping}, access="read", address="path")
    assert not verdict.allowed
    # The meaning, not the wording: the refusal has to say that the place meant is somewhere else, and name
    # that place. Pinning a phrase here would be the defect the finding was about — a notice that names a
    # mechanism it cannot support is worse than one that names none.
    assert "resolve" in verdict.reason.lower(), verdict.reason
    assert os.path.normcase(os.path.join(str(tmp_path), "outside", "secrets")) \
        in verdict.reason, verdict.reason
    assert Confine(read_roots=[granted])("read", {"path": staying}, access="read", address="path").allowed


def test_a_grant_holds_against_repointing_a_path_without_moving_it(yard):
    """Dots, doubled separators and a trailing slash name the same place. A boundary that could
    be escaped by punctuation is not a boundary."""
    granted = str(yard[0])
    policy = Confine(read_roots=[granted + os.sep])
    assert policy("read", {"path": os.path.join(granted, ".", "note.md")},
                  access="read", address="path").allowed
    assert policy("read", {"path": granted + os.sep + os.sep + "note.md"},
                  access="read", address="path").allowed


def test_a_tool_with_no_address_has_nothing_to_confine():
    """Confinement is about places. Inventing a rule for a tool that touches none is theatre -- and the
    declaration is what says so now, rather than the tool's absence from a table nobody maintained."""
    assert Confine()("summarize", {"text": "hello"}, access="none", address=None).allowed


def test_an_address_argument_that_was_not_given_is_refused_not_assumed():
    verdict = Confine(read_roots=["/"])("read", {}, access="read", address="path")
    assert not verdict.allowed and "needs args.path" in verdict.reason


# --- the axis a shell sits on ---------------------------------------------------------

def test_execution_is_refused_unless_it_is_granted():
    verdict = Confine(read_roots=["/"])("bash", {"command": "ls"}, access="exec", address="cwd")
    assert not verdict.allowed and "--exec" in verdict.reason


def test_a_granted_shell_says_what_it_costs_every_single_time():
    """Path confinement does not bind a shell command, so the disclosure rides on the verdict and
    lands in the operation step of every call rather than in a manual."""
    verdict = Confine(read_roots=["/"], allow_exec=True)(
        "bash", {"command": "ls"}, access="exec", address="cwd")
    assert verdict.allowed
    assert "path confinement does not bind" in verdict.reason


# --- the predicate ---------------------------------------------------------------------

def test_inside_is_the_folder_or_beneath_it_not_merely_a_string_prefix(tmp_path):
    root = resolve(str(tmp_path / "run"))
    assert inside(root + os.sep + "a", [root])
    assert inside(root, [root])
    assert not inside(resolve(str(tmp_path / "running")), [root]), \
        "a name that starts like the grant is not inside it"


def test_a_path_on_another_drive_is_not_inside(tmp_path):
    """On Windows, comparing paths on different drives raises rather than answering; the answer
    a refusal needs is 'not inside'."""
    assert not inside("f:\\somewhere\\else", [resolve(str(tmp_path))])


# --- the tools themselves ---------------------------------------------------------------

def test_the_tools_do_the_thing_and_say_what_happened(tmp_path):
    target = tmp_path / "deep" / "note.md"
    first = fs.write(str(target), "one\ntwo\n")
    assert first.startswith("created") and "8 bytes" in first
    assert fs.read(str(target)) == "one\ntwo"
    assert fs.write(str(target), "three").startswith("replaced"), \
        "destroying a file should not require a second call to find out"
    assert fs.ls(str(tmp_path)) == "deep/"


def test_a_read_can_be_paged(tmp_path):
    target = tmp_path / "lines.txt"
    target.write_text("\n".join(f"l{n}" for n in range(10)), encoding="utf-8")
    out = fs.read(str(target), offset=3, count=2)
    assert out.startswith("l2\nl3")
    assert "more remain" in out and "offset=5" in out, "a page that stopped says where to continue"


def test_listing_says_which_names_are_folders(tmp_path):
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    (tmp_path / "afolder").mkdir()
    listing = fs.ls(str(tmp_path)).splitlines()
    assert "afolder/" in listing
    assert any(line.startswith("afile (1 bytes)") for line in listing)


def test_listing_a_file_says_it_is_not_a_folder(tmp_path):
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")
    assert "not a folder" in fs.ls(str(target))


def test_an_empty_folder_says_so_rather_than_printing_nothing(tmp_path):
    assert "is empty" in fs.ls(str(tmp_path))


# --- the veto point wired into the loop --------------------------------------------------

def test_the_loop_records_the_resolved_address_not_the_typed_one(tmp_path, monkeypatch):
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    granted = tmp_path / "granted"
    granted.mkdir()
    (granted / "note.md").write_text("hello", encoding="utf-8")
    agent = Agent(backend=Fake([turn_tool("c1", "read", {"path": "note.md"}), turn_text("done")],
                               caps=Caps(100, model="t")),
                  tools={"read": fs.read}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(read_roots=[granted]))
    monkeypatch.chdir(granted)
    agent.run("read the note")
    operation = [s for s in agent.log.steps() if s["kind"] == "operation"][0]
    assert operation["address"] == resolve("note.md")
    assert operation["outcome"] == "allowed"
    result = [s for s in agent.log.steps() if s["kind"] == "result"][0]
    assert result["ok"] is True and result["content"] == "hello"


def test_the_model_is_told_the_same_refusal_the_log_holds(tmp_path):
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    agent = Agent(backend=Fake([turn_tool("c1", "read", {"path": "/etc/shadow"}),
                               turn_text("I cannot read that")], caps=Caps(100, model="t")),
                  tools={"read": fs.read}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(read_roots=[tmp_path]))
    agent.run("read it")
    operation = [s for s in agent.log.steps() if s["kind"] == "operation"][0]
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert operation["outcome"] == "refused"
    assert tool_message.content == operation["reason"]
    assert "--grant" in tool_message.content


def test_a_refused_call_touches_nothing(tmp_path):
    """The gate is a gate: a call the veto point refused must not have happened, whatever the log
    says about it. A check performed after the act would pass every other test in this file."""
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    granted = tmp_path / "granted"
    granted.mkdir()
    target = granted / "new.md"
    agent = Agent(backend=Fake([turn_tool("c1", "write", {"path": str(target), "content": "x"}),
                               turn_text("done")], caps=Caps(100, model="t")),
                  tools={"write": fs.write}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(read_roots=[granted]))        # readable, not writable
    agent.run("write it")
    assert not target.exists(), "the refusal was recorded and the file was still written"
    assert [s for s in agent.log.steps() if s["kind"] == "operation"][0]["outcome"] == "refused"


def test_nothing_reaches_the_comparison_unnormalised(tmp_path):
    """Both sides of `inside` come from `resolve`, which is why a spelling difference the
    filesystem ignores cannot become a boundary. Asserted as an identity because case folding is a
    no-op on POSIX: the portable claim is that whatever enters the comparison is already
    normalised, and on a case-insensitive filesystem that is what stops a drive letter and a
    folder name from becoming two different places."""
    target = resolve(str(tmp_path / "Granted" / "Note.MD"))
    assert target == os.path.normcase(target)
    if os.path.normcase("A") == "a":       # a filesystem that folds case
        assert resolve(str(tmp_path / "GRANTED")) == resolve(str(tmp_path / "granted"))


def test_a_refused_call_records_the_resolved_place_too(tmp_path):
    """A live run showed allowed calls carrying the resolved address and refused ones carrying the
    typed string. The place that was judged is the interesting fact about a refusal, so both
    outcomes record it."""
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    granted = tmp_path / "granted"
    granted.mkdir()
    asked = os.path.join(str(granted), "..", "elsewhere", "note.txt")
    agent = Agent(backend=Fake([turn_tool("c1", "write", {"path": asked, "content": "x"}),
                               turn_text("blocked")], caps=Caps(100, model="t")),
                  tools={"write": fs.write}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(read_roots=[granted]))
    agent.run("write it")
    operation = [s for s in agent.log.steps() if s["kind"] == "operation"][0]
    assert operation["outcome"] == "refused"
    assert operation["address"] == resolve(asked)
    assert operation["address"].endswith("elsewhere" + os.sep + "note.txt")


def test_a_refused_read_also_records_the_place_judged(tmp_path):
    """The write case is covered above; the read refusal is a separate return, and a branch with
    no test is a branch that changes underneath you."""
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    asked = str(tmp_path / "private" / "secrets.txt")
    agent = Agent(backend=Fake([turn_tool("c1", "read", {"path": asked}), turn_text("no")],
                               caps=Caps(100, model="t")),
                  tools={"read": fs.read}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(read_roots=[tmp_path / "granted"]))
    agent.run("read it")
    operation = [s for s in agent.log.steps() if s["kind"] == "operation"][0]
    assert operation["outcome"] == "refused"
    assert operation["address"] == resolve(asked)


# --- the exec axis -------------------------------------------------------------------

def test_bash_is_refused_without_the_exec_axis(yard):
    """Running something is not a path act, so it is not covered by a read/write grant. A run with
    folders granted but no --exec still cannot open a shell."""
    policy = Confine(read_roots=[yard], write_roots=[])
    verdict = policy("bash", {"command": "ls"}, access="exec", address="cwd")
    assert not verdict.allowed
    assert "--exec" in verdict.reason


def test_bash_is_allowed_with_exec_and_the_cost_is_stated_every_time(yard):
    """The disclosure is not a banner printed once at boot where nobody reads it; it is the verdict
    reason, so it lands in the operation step of every single shell call and renews itself."""
    policy = Confine(read_roots=[], write_roots=[], allow_exec=True)
    verdict = policy("bash", {"command": "rm -rf /"}, access="exec", address="cwd")
    assert verdict.allowed
    assert "does not bind" in verdict.reason        # the sentence that says confinement is now advisory


def test_the_exec_disclosure_says_what_it_costs(yard):
    """'exec granted' is not the disclosure. The disclosure is the consequence: path confinement no
    longer binds, because a command can touch anything the user can."""
    assert "advisory" in confine.EXEC_DISCLOSURE or "does not bind" in confine.EXEC_DISCLOSURE
    assert "anything this user can" in confine.EXEC_DISCLOSURE


def test_granting_exec_does_not_grant_paths_and_vice_versa(yard):
    """The axes are independent. --exec is not a write grant, and a write grant is not a shell: each
    is refused with the name of the act that would actually permit it."""
    granted, _ = yard
    exec_only = Confine(allow_exec=True)
    assert not exec_only("write", {"path": str(granted / "x"), "content": "x"},
                         access="write", address="path").allowed
    paths_only = Confine(read_roots=[granted], write_roots=[granted])
    assert paths_only("write", {"path": str(granted / "x"), "content": "x"},
                      access="write", address="path").allowed
    assert not paths_only("bash", {"command": "ls"}, access="exec", address="cwd").allowed


def test_an_exec_call_records_its_disclosure_in_the_operation_step(tmp_path):
    """The whole point of putting the disclosure in the reason is that it survives into the log. If
    the loop dropped the reason for an allowed call, the disclosure would be a lie."""
    from sah.agent import Agent
    from sah.model import Caps, Fake, turn_text, turn_tool
    from sah.record import Log

    def _bash(command: str, cwd: str = "", timeout: int = 120) -> str:
        return "exit: 0"

    _bash.access, _bash.address = "exec", "cwd"
    _bash.description = "Run a shell command and return its output."

    agent = Agent(backend=Fake([turn_tool("c1", "bash", {"command": "echo hi"}), turn_text("done")],
                               caps=Caps(100, model="t")),
                  tools={"bash": _bash}, log=Log(tmp_path / "log.jsonl"), root=tmp_path,
                  policy=Confine(allow_exec=True))
    agent.run("run it")
    op = [s for s in agent.log.steps() if s["kind"] == "operation"][0]
    assert op["outcome"] == "allowed"
    assert "does not bind" in op["reason"]


# --- ownership: a parent can read its own children, and that is not a grant ----

def test_a_parent_reads_inside_its_own_child_tree_without_a_grant(tmp_path):
    """spawn hands back a child log path that usually sits outside the run's read grants. Ownership
    closes that gap: the run's own children tree is always readable, no --grant involved."""
    children = tmp_path / "children"
    (children / "kid1").mkdir(parents=True)
    (children / "kid1" / "log.jsonl").write_text("{}", encoding="utf-8")
    policy = Confine(read_roots=[], write_roots=[], owned_roots=[str(children)])
    v = policy("read", {"path": str(children / "kid1" / "log.jsonl")}, access="read", address="path")
    assert v.allowed and "child" in v.reason


def test_ownership_is_read_only(tmp_path):
    """Owning a child means inspecting it, not writing into its folder. If a parent could write its
    child's run dir it could forge the child's log -- the record it is meant to be reading."""
    children = tmp_path / "children"
    children.mkdir()
    policy = Confine(read_roots=[], write_roots=[], owned_roots=[str(children)])
    v = policy("write", {"path": str(children / "kid1" / "log.jsonl")}, access="write", address="path")
    assert not v.allowed


def test_ownership_does_not_appear_as_a_read_grant(tmp_path):
    """The distinction the user asked for: parents *own* children, they were not *granted* them. If
    owned leaked into readable_roots, the banner and <situation> would advertise a widening that
    nobody performed."""
    children = tmp_path / "children"
    children.mkdir()
    policy = Confine(read_roots=[], write_roots=[], owned_roots=[str(children)])
    assert policy.readable_roots == []            # the grants are still empty
    assert policy.owned_roots                      # ownership is tracked apart


def test_ownership_does_not_reach_outside_the_child_tree(tmp_path):
    children = tmp_path / "children"
    children.mkdir()
    policy = Confine(read_roots=[], write_roots=[], owned_roots=[str(children)])
    v = policy("read", {"path": str(tmp_path / "notachild" / "x.txt")}, access="read", address="path")
    assert not v.allowed


def test_a_dot_dot_that_escapes_a_grant_says_so(tmp_path):
    """The refusal is supposed to name the gap between the name typed and the place meant.

    It never fired for a dot-dot path, because the asked-for path is made absolute *before* the two names are
    compared, and making a path absolute collapses `..` — so the comparison compared a name that had already
    lost the thing being reported. This is the same escape as a symlink, judged identically and explained
    never."""
    granted = tmp_path / "granted"
    granted.mkdir()
    typed = str(granted / ".." / "elsewhere" / "note.md")
    verdict = Confine(read_roots=[str(granted)])("read", {"path": typed}, access="read", address="path")
    assert not verdict.allowed
    assert "resolve" in verdict.reason.lower(), \
        f"a dot-dot escape explained nothing about resolving: {verdict.reason!r}"


def test_a_name_whose_only_change_is_its_spelling_does_not_claim_a_symlink(tmp_path, monkeypatch):
    """Windows `realpath` expands an 8.3 short name, and nothing else about the path moved.

    The old wording asserted "a symlink" for that, which teaches the operator to disbelieve the notice on the
    case it exists for. A name that resolves somewhere else should say it resolves somewhere else; naming the
    mechanism is a claim this function cannot support."""
    granted = tmp_path / "granted"
    granted.mkdir()
    typed = str(granted / "SHORTN~1.MD")
    monkeypatch.setattr(confine.os.path, "realpath",
                        lambda path: path.replace("SHORTN~1.MD", "a much longer name.md"))
    verdict = Confine(read_roots=[str(granted / "sub")])("read", {"path": typed},
                                                        access="read", address="path")
    assert not verdict.allowed
    assert "symlink" not in verdict.reason.lower(), verdict.reason


def _advice(reason: str, flag: str) -> str:
    """The path a refusal tells the operator to grant."""
    m = re.search(rf"--{flag} (\S+)", reason)
    assert m, f"the refusal no longer names a way in: {reason!r}"
    return m.group(1)



def test_a_refusal_advises_the_address_that_was_asked_for(tmp_path):
    """The advice must name the file, because that is what the run asked for and what `widen` grants.

    The module states the rule in `widen`: "a refusal that suggests granting a parent teaches the operator
    to grant a drive". `inside()` accepts a file as a root, so naming the file costs the operator nothing
    they would not have typed anyway, and naming the folder is a widening the run never requested.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "a.md"
    target.write_text("x", encoding="utf-8")
    verdict = Confine(read_roots=[tmp_path / "granted"])("write_file", {"path": str(target)},
                                                         access="write", address="path")
    assert not verdict.allowed
    advised = _advice(verdict.reason, "write")
    assert pathlib.Path(advised).resolve() == target.resolve(), \
        f"the refusal advised granting {advised!r}, not the {target.name} that was asked for"


def test_the_address_a_refusal_names_really_opens_the_thing_it_named(tmp_path):
    """An advice the operator cannot follow is a dead end written as guidance.

    So the test does what the operator does: read the path out of the refusal, start a policy with that
    grant, and make the same call. It must pass -- and its sibling must not, or the "smallest act" was a
    lie and the grant opened the folder after all.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "a.md"
    sibling = elsewhere / "b.md"
    target.write_text("x", encoding="utf-8")
    first = Confine()("write_file", {"path": str(target)}, access="write", address="path")
    advised = _advice(first.reason, "write")

    widened = Confine(write_roots=[str(pathlib.Path(advised).resolve())])
    assert widened("write_file", {"path": str(target)}, access="write",
                   address="path").allowed, "granting the advised path did not permit the call"
    assert not widened("write_file", {"path": str(sibling)}, access="write",
                       address="path").allowed, "the advised grant opened the folder, not the file"


def test_a_read_refusal_advises_the_address_that_was_asked_for(tmp_path):
    """The read side advises `--grant`, and is held to the same rule as the write side."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "a.md"
    target.write_text("x", encoding="utf-8")
    verdict = Confine()("read_file", {"path": str(target)}, access="read", address="path")
    assert not verdict.allowed
    advised = _advice(verdict.reason, "grant")
    assert pathlib.Path(advised).resolve() == target.resolve(), \
        f"the read refusal advised {advised!r} instead of the file asked for"


# --- the global override: --no-confine ---------------------------------------

def test_the_open_axis_reaches_any_place_and_still_names_it(yard):
    """`--no-confine` is the operator taking the wall down for one run: a USB stick mounted at a letter
    nobody thought to grant, a second drive, a folder three links away. It is an *axis* rather than a
    giant root, so the record keeps saying which address a call touched — an open run whose log says
    nothing about places is unauditable in exactly the way a confined one is not."""
    granted, outside = yard
    policy = Confine(unrestricted=True)
    verdict = policy("read", {"path": str(outside / "secrets")}, access="read", address="path")
    assert verdict.allowed, verdict.reason
    assert verdict.reason == confine.OPEN_DISCLOSURE, "an allowance must state the reason it gave"
    assert verdict.address == resolve(str(outside / "secrets"))
    assert policy("write", {"path": str(granted / "new.txt")}, access="write",
                  address="path").allowed


def test_the_open_axis_still_refuses_a_call_that_named_no_place():
    """With the wall down there is nothing left to check against, and the temptation is to allow
    everything before looking at the arguments. A call with no address is a broken call: allowing it
    records an act that touched an unnamed place, which is the one entry a later reader cannot use."""
    verdict = Confine(unrestricted=True)("read", {}, access="read", address="path")
    assert not verdict.allowed and "no address" in verdict.reason


def test_the_open_axis_does_not_borrow_the_exec_axis():
    """Two axes, two costs. Opening every path is not granting a shell, and a policy that let one flag
    imply the other would make `exec: off` in the boot box a sentence about neither."""
    verdict = Confine(unrestricted=True)("shell", {"command": "dir"}, access="exec", address="command")
    assert not verdict.allowed and "--exec" in verdict.reason


def test_the_open_axis_is_recorded_as_an_axis_and_not_as_a_root():
    """`reach()` is what `started.grants` holds, and two different facts must stay distinguishable in
    it: this run may touch everything, and this run may touch nothing. Both have empty root lists, so
    without the axis they record identically — and a resume comparing them would report no change."""
    open_reach = Confine(unrestricted=True).reach()
    assert open_reach["unrestricted"] is True, open_reach
    assert open_reach["read"] == [] and open_reach["write"] == []
    assert Confine(read_roots=["x"]).reach()["unrestricted"] is False


def test_the_project_notes_axis_is_recorded_and_no_veto_consults_it(tmp_path):
    """`--project-notes` is a reach axis, so it is on `reach()` (recorded on `started.grants`,
    compared on resume) like exec and unrestricted. But it is consumed at boot (it steers prompt
    assembly), so no tool call consults it: a read within the grant is allowed exactly as before,
    with or without the axis. It is a grant in the reach sense, not a veto that widens file access."""
    assert Confine(project_notes=True).reach()["project_notes"] is True
    assert Confine().reach()["project_notes"] is False, "off by default: a run that did not ask loads nothing"
    with_file = str(tmp_path / "a.md")
    (tmp_path / "a.md").write_text("x")
    for notes in (True, False):
        verdict = Confine(read_roots=[str(tmp_path)], project_notes=notes)(
            "read", {"path": with_file}, access="read", address="path")
        assert verdict.allowed, "the axis must not change a file-access verdict"
