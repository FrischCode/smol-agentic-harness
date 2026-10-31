"""Structural rules the project promises. These run against the tree, not against behaviour,
so they hold when there is almost nothing in it yet — and they fail the moment a promise is
traded away for convenience."""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "sah"
TESTS = ROOT / "tests"
# This file lists the strings it forbids, so scanning it would report its own patterns.
SELF = pathlib.Path(__file__).resolve()

# Imported by conftest.py to make the package importable; not a dependency.
_HARNESS = {"conftest"}


def _sources(folder, skip_self=False):
    found = [p for p in folder.rglob("*.py") if p.stem not in _HARNESS]
    return [p for p in found if not (skip_self and p.resolve() == SELF)]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_core_imports_stdlib_and_sah_only():
    """A dependency is a decision, not a convenience: nothing outside the standard library
    may be imported by core, and nothing may reach outside the package."""
    allowed = set(sys.stdlib_module_names) | {"sah"}
    for path in _sources(SRC):
        for module in _imports(path):
            assert module in allowed, f"{path.relative_to(ROOT)} imports {module!r}"


def test_the_battery_has_no_skips_and_no_xfails():
    """A test that cannot fail is not in the battery, and a known-red test is a lie about
    the state of the tree."""
    for path in _sources(TESTS, skip_self=True):
        text = path.read_text(encoding="utf-8")
        for marker in ("skipif", "xfail", "pytest.skip"):
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_the_battery_runs_in_process():
    """The battery is fast and deterministic by construction: no child processes, no sockets,
    no real clocks. Importing a module to reach one of its exception types is allowed —
    spawning through it is not. Anything needing a process boundary gets its own slower
    harness."""
    forbidden = {("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
                 ("subprocess", "check_output"), ("os", "system"), ("os", "popen"),
                 ("socket", "socket"), ("threading", "Thread")}
    for path in _sources(TESTS):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                assert pair not in forbidden, f"{path.name} calls {pair[0]}.{pair[1]}()"


def test_no_marker_debt_in_the_tree():
    """A TODO is a decision deferred without a reason. Either it is written down or it is done."""
    for path in list(_sources(SRC)) + list(_sources(TESTS, skip_self=True)):
        text = path.read_text(encoding="utf-8")
        for marker in ("TODO", "FIXME", "XXX"):
            assert marker not in text, f"{path.relative_to(ROOT)} carries a {marker}"


def test_the_taxonomy_is_declared_in_exactly_one_place():
    """Two registries with different vocabularies is the failure this project exists to
    avoid, so the kind table may not be duplicated or shadowed."""
    owners = [p.relative_to(ROOT).as_posix() for p in _sources(SRC)
              if "KINDS = {" in p.read_text(encoding="utf-8")]
    assert owners == ["sah/record.py"], owners




def _import_graph() -> dict:
    """Edges between `sah` modules, read from the syntax tree. Nothing is imported, so a cycle is found
    by a test that cannot itself fail to run because of the cycle it is looking for.

    Nodes are keyed by import path (`sah.agent`, `sah.tools.calc`), not file stem: a split module
    leaves two files that share a stem (`sah/spawn.py` and `sah/tools/spawn.py`), and a stem-keyed
    graph would keep one and silently drop the other, so the test would stop watching a module
    without saying so."""
    package = ROOT / "sah"
    mods = {}
    for f in package.rglob("*.py"):
        name = "sah." + f.relative_to(package).with_suffix("").as_posix().replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-len(".__init__")]
        mods[name] = f
    graph: dict[str, set[str]] = {name: set() for name in mods}
    for name, path in mods.items():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "sah" or node.module.startswith("sah.")):
                for alias in node.names:
                    # `from sah import x` (x is a submodule) and `from sah.x import y` (an edge to the
                    # module, and to the submodule itself where one is named).
                    for candidate in (node.module, node.module + "." + alias.name):
                        if candidate in graph:
                            graph[name].add(candidate)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in graph:
                        graph[name].add(alias.name)
    return graph


def _cycles(graph: dict) -> list[list[str]]:
    """Every cycle reachable by depth-first search, as the path that closed it. Marked rather than
    memoised by path: a diamond is not a cycle and must not be reported as one."""
    state: dict[str, int] = {}
    found: list[list[str]] = []

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                found.append(stack + [node, nxt])
            elif state.get(nxt) is None:
                visit(nxt, stack + [node])
        state[node] = 2

    for node in sorted(graph):
        if state.get(node) is None:
            visit(node, [])
    return found


def test_the_core_import_graph_has_no_cycles():
    """A cycle is why an import hides inside a function, and an import that hides inside a function is
    how a cycle becomes permanent: nothing downstream can see the edge, so nobody notices the second one.

    Three modules import a `sah` module from inside a method, and the honest reading of each is different
    — one needs the package object to find its own folder, one catches a failed import so a label can
    never break a run — so the rule cannot be "no function-level imports". What it can be, and what those
    two cases do not violate, is that the modules form a DAG. When that stops being true the answer is to
    break the cycle, not to add a lazy import, and this names the path that closed."""
    graph = _import_graph()
    cycles = _cycles(graph)
    assert not cycles, "import cycle: " + " -> ".join(cycles[0])


def _code_and_checks():
    """The files that must explain themselves line by line: the code and the battery.

    `AGENTS.md` is deliberately outside this set. It is contributor-facing history, and it documents the
    previous tree this repo was written against — a rule that forbade naming that tree would forbid
    documenting the discipline that governs it. The self-explanatory promise is about
    reading the *source* without a second document, so that is where it is enforced. This file is exempt
    too, and for the plain reason it states at the top: it lists the strings it forbids."""
    return list(_sources(SRC)) + list(_sources(TESTS, True))


def _tracked_prose():
    """Documentation that ships: the top-level notes."""
    return sorted(ROOT.glob("*.md"))


def test_nothing_tracked_cites_the_throwaway_folder():
    """A pointer into a folder a fresh clone does not have is a dead end written as evidence.

    The throwaway folder is gitignored on purpose — it holds half-finished experiments and the mutation
    harnesses, and their scripts exist only on the machine that ran them. A reader cannot re-run one,
    cannot inspect a mutant, cannot check the claim; they get a path to nothing. What travels is the
    claim and what it reddened. A harness worth keeping has a tracked home, not a place in the
    throwaway folder — the console measurement sits in the battery for exactly that reason.

    This one scans prose as well as code. This file is exempt, for the reason stated at its top: it lists
    the strings it forbids, so scanning it would report its own patterns rather than a violation."""
    paths = (list(_sources(SRC)) + list(_sources(TESTS, True))
             + sorted(ROOT.glob("*.md")))
    for path in paths:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert "scratch/" not in line, (
                f"{path.relative_to(ROOT)}:{n} cites the throwaway folder, which no clone of this repo "
                "will contain; state the claim, and give a reusable harness a tracked home")


def test_the_review_is_a_journal_and_not_a_source():
    """The review records what was measured, what was retracted and what is still inferred. That is a
    journal — read it to learn the history of a decision, never to learn the decision.

    Citing it as a source inverts the relationship. A reader told to "see the review" for how a detector
    works must open a second document to understand the line in front of them, and the harness stops being
    self-explanatory at exactly the place where someone decided the explanation was too long to keep in the
    code that has to carry it. Where a claim exists only in the review, the code cannot support it."""
    for path in _code_and_checks() + _tracked_prose():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert "docs/review" not in line, (
                f"{path.relative_to(ROOT)}:{n} sends a reader to the review to learn a fact; the fact "
                "belongs here, and the review keeps the history")


_OUTSIDE_AUTHORITY = ("kmem", "sah_old", "sah_fossil", "the oracle", "the old sah")


def test_no_outside_project_carries_authority_in_the_code():
    """Two kinds of sentence fail here. "Lifted from X" tells a reader the reason lives in someone else's
    tree, which they do not have and cannot check. A sentence making a previous implementation's history
    stand in for a reason has the same defect: the reader cannot open what is being cited.

    Neither is load-bearing. The memory module can say why a generated index is a liability; a test can say
    why a body must be inflated before it is decoded. What another project once did is not evidence about
    this one, and a harness sold as readable in full cannot rest on files the reader lacks.

    Endpoints are exempt because they are not appeals to authority — `SAH_SEARCH_URL`, a loopback host and
    the RFC a live check fetches are targets a run contacts, and are named as targets."""
    for path in _code_and_checks():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            hit = next((w for w in _OUTSIDE_AUTHORITY if w in low), None)
            assert not hit, (f"{path.relative_to(ROOT)}:{n} leans on {hit!r}; state the behaviour and its "
                             "reason in this tree — if the claim is about another system, say it was "
                             "measured and against what")




def test_no_file_is_left_for_the_collector_to_close():
    """`open(...)` without a `with` hands the handle to the garbage collector: on CPython that means "when
    this stops being referenced", on another implementation "whenever". For a short read nobody notices,
    which is how the shape spreads; around a write, or when one run touches the same note twice, it is a
    locked file on Windows and an error with no line in it. `with` costs nothing and turns the lifetime
    into a fact a reader can see in the line.
    """
    offenders = []
    for path in _sources(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        held = {id(item.context_expr) for node in ast.walk(tree) if isinstance(node, ast.With)
                for item in node.items if isinstance(item.context_expr, ast.Call)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or id(node) in held:
                continue
            # The builtin only. `os.open` returns a descriptor with its own discipline, and `_OPENER.open`
            # is a method that hands back a response the caller wraps — flagging those would be the guard
            # crying wolf, and a guard that does that gets widened away the first time it annoys someone.
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (f"opened without a context manager: {', '.join(sorted(offenders))}; the handle "
                           "should end with the block that needed it")

def test_a_declared_capability_is_read_or_it_does_not_exist():
    """Closed set is a test, not a comment: a field on the seam that nobody consults is a claim that rots.

    Measured before this rule existed: two Caps fields had zero attribute loads anywhere outside
    `model.py`, and a third had one -- its own re-statement inside the `Caps(...)` call that reports it.
    That is how a seam starts to lie. The declaration keeps being maintained; nothing keeps being true.
    """
    model_py = pathlib.Path(__file__).resolve().parent.parent / "sah" / "model.py"
    tree = ast.parse(model_py.read_text(encoding="utf-8"), filename=str(model_py))
    caps = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Caps")
    fields = {a.target.id for a in caps.body
              if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)}
    assert fields, "Caps declares no fields to hold to account"

    reads = {f: 0 for f in fields}
    for path in _sources(SRC):
        if path.name == "model.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Attribute) and node.attr in reads \
                    and not isinstance(node.ctx, ast.Store):
                reads[node.attr] += 1
    never = sorted(f for f, n in reads.items() if n == 0)
    assert not never, f"declared and never consulted: {never}"


def test_tool_declarations_live_only_in_the_tools_package():
    """A tool's declaration is its boundary, and the declaration is the marker: an assignment to
    `access` anywhere outside `sah/tools/` is machinery wearing a tool's face, and only the boot
    gate — which a reader of the tree may never read — would catch it. The tree itself must say so.

    The scan is an AST walk for assignments whose target carries the `access` attribute, the
    declaration's shape, not a string search: prose about the marker must not make a file a
    violator."""
    found = []
    for path in _sources(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                declared = any(isinstance(sub, ast.Attribute) and sub.attr == "access"
                               for target in targets for sub in ast.walk(target))
                if declared:
                    found.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert found, "no tool declaration found under sah/: the marker moved or the scan broke"
    outside = [loc for loc in found if not loc.startswith("sah/tools/")]
    assert not outside, f"tool declarations outside sah/tools/: {outside}"


def test_the_tools_package_catalog_names_every_module():
    """The package's catalog is the one place a reader is told what capabilities exist, so it must
    agree with the tree exactly: a module that is not named is undiscoverable, and a name with no
    module is a promise the tree does not keep.

    The catalog is read from the `sah.tools.<name>` spellings in the docstring, so adding a
    capability costs one word, and a module that moves or is renamed breaks the test in the same
    commit rather than rotting silently."""
    package = SRC / "tools"
    init = package / "__init__.py"
    doc = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"), filename=str(init))) or ""
    present = {p.stem for p in package.glob("*.py") if p.name != "__init__.py"}
    named = set(re.findall(r"sah\.tools\.(\w+)", doc))
    assert named == present, (
        f"the catalog and the package disagree: named {sorted(named)}, present {sorted(present)}")
