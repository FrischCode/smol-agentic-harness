"""sah.spawn -- start a child run that is addressable, not attached.

`spawn` starts another SAH run and returns immediately with its id. The child writes to its own
run folder and its own log; the spawner reads that log back later. This is the answer to the one
question that covers both unattended runs and child output -- *who is the reader*. A run never
assumes a human is watching, so a child's output is not a terminal buffer the parent happens to be
looking at: it is a log, which is the record. The log stays the only ground truth a reader audits
against; the Markdown an unattended run leaves beside it at the ending (`report.md`, written by
the harness) is a projection for reading, not a second source — the parent's audit path renders
from the child's log itself, never from the child's file.

The model-facing capability built on this machinery -- the `spawn` tool a run offers its model, and
the `child_status`/`child_wait` reads -- is declared in `sah.tools.spawn`.

**The child is addressable, not attached.** Dying with the parent is a rule about children whose
output a client is watching. A child with its own log is meant to outlive the parent that started
it -- that is the point of running it separately -- so it is launched in its own process group and
is never killed when the parent exits. The two cases stay named apart: *attached* (a shell child,
a tool the parent waits on) dies with the parent; *addressable* (a spawned run) does not.

**The schema exposes a task, never a scope.** A schema generated from a run-config vocabulary
advertises `run_id`, `sessions_dir` and `manifest`, and a model call could write a child's grants —
so this tool takes the child's *objective* and, at most, a model override. The run id, its folder,
and the backend it talks to are chosen by the harness, from the spawner's own live configuration:
a child runs under what the current invocation has, not under a widening the caller invents. If a
child needs a folder the parent does not have, that is a new invocation with new grants, said out
loud.

**A child inherits the parent's grants, because that is what "run under me" means** -- but
*inheritance here is the harness passing its own flags to the child's command line*, visible in the
child's own banner and log, not a manifest file the child trusts. A child never gets more than the
parent has, and it can be given less. Widening still can't be inherited: the child's command is
built from the parent's current grants, and there is no channel through which a caller adds one.

**A child does not inherit `spawn`.** Grants pass down because "run under me" means exactly that, but
hand-off ability compounds: children are detached by design, so one looping model could start a line of
runs each starting a line of runs, all on the same backend, nothing bounding the total. A run may hand
work off once; it may not pass on the ability. The harness therefore appends `--no-spawn` to every
child's command line — where the rest of the inheritance is already visible — and sets `SAH_NO_SPAWN=1`
in the child's environment so a child with `--exec` cannot start a sah run through its own shell without
the veto. Both directions narrow and neither widens. The cost is the declared cost of exec: a child with
a shell can unset that variable.
"""
from __future__ import annotations

import os
import sys
import time

from sah import openai, process, record


class Spawner:
    """Launches child SAH runs on the parent's behalf.

    Everything the child needs to talk to the same backend and to be confined the way the parent is
    is captured at construction from the parent's live configuration. None of it is a parameter of
    the `spawn` tool, so a model call cannot reach any of it -- the tool's whole argument surface is
    the child's task."""

    def __init__(self, *, runs_root: str, model: str, host: str, port: int, api_key: str = "",
                 think: str = "none", summary_thinking: str | None = None,
                 context: int = 32768, read_roots=(), write_roots=(),
                 allow_exec: bool = False, memory: str = "", max_turns: int = 12,
                 max_runtime: int = 0, grace_turns: int = 3,
                 allow_web: bool = False, web_private: bool = False, search_url: str = "",
                 unrestricted: bool = False):
        self.runs_root = os.path.abspath(str(runs_root))
        self.model = model
        self.host = host
        self.port = int(port)
        self.api_key = api_key
        self.think = think
        self.summary_thinking = summary_thinking or None
        self.context = int(context)
        self.read_roots = list(read_roots)
        self.write_roots = list(write_roots)
        self.allow_exec = bool(allow_exec)
        # Held and deliberately never put on a child's argv. The parent's open axis (`--no-confine`) is the
        # widest thing a run can hold, and a spawner that forwarded it would turn one operator's decision at
        # one terminal into the standing reach of every run beneath it — each recording `unrestricted:
        # false`, because forwarding a widening is exactly the drift the child's argv is assembled to
        # prevent. This field is where that prevention lives: held, and never forwarded.
        self.unrestricted = bool(unrestricted)
        # Resolved here, against this process's working directory -- the only one that means anything for
        # the path the operator typed. The child runs with its own run folder as cwd, so a relative
        # `--memory notes` would create a fresh empty tree under children/<id>/notes while the parent's
        # tree stayed untouched: the child would silently remember nothing, and remember it somewhere new.
        # `runs_root` is abspath'd above for the same reason.
        self.memory = os.path.abspath(memory) if memory else ""
        self.max_turns = int(max_turns)
        self.max_runtime = int(max_runtime)
        # The child's wall is the parent's wall: a child that outlives the operator's time budget is
        # the exact thing the budget exists to prevent, and inheritance here is the harness passing
        # its own flags, visible on the child's command line.
        self.grace_turns = int(grace_turns)
        # The child's leeway is the parent's: the wrap-up is part of the budget's meaning, so a
        # child that inherits the turn ceiling inherits the word that goes with it.
        self.allow_web = bool(allow_web)
        self.web_private = bool(web_private)
        self.search_url = search_url

    def child_dir(self, run_id: str) -> str:
        """The run folder a child with this id writes to, under the runs root."""
        return os.path.join(self.runs_root, run_id)

    def build_command(self, run_id: str, prompt: str, *, model: str = "", max_turns: int = 0):
        """The argv that launches the child run. Built from the parent's configuration; the only
        per-call choices are the objective and an optional model/turn-count override -- never a
        path, a grant, or a scope."""
        root = self.child_dir(run_id)
        argv = [sys.executable, "-m", "sah", "run", "--run-dir", root,
                "--model", model or self.model, "--host", self.host, "--port", str(self.port),
                "--context", str(self.context), "--think", self.think,
                "--max-turns", str(max_turns or self.max_turns),
                # Always forwarded: the leeway is part of the turn ceiling's meaning, so a child
                # that inherits the ceiling inherits the wrap-up that goes with it.
                "--grace-turns", str(self.grace_turns),
                # Not inherited: see the module docstring. Unconditional, so no child can be built
                # without it, whatever flags its parent was started with. `--no-confine`, when the parent
                # had it, is likewise absent from this list on purpose — see `self.unrestricted`.
                "--no-spawn"]
        # The API key is deliberately absent from this list. The prompt is on the command line because
        # an unattended child should advertise what it was told to do; a secret has the opposite
        # requirement, since argv is readable by every local user for the life of the process and lands
        # in anything that collects process telemetry. `subprocess_spawn` passes it in the environment.
        for grant in self.read_roots:
            argv += ["--grant", grant]
        for grant in self.write_roots:
            argv += ["--write", grant]
        if self.allow_exec:
            argv += ["--exec"]
        if self.memory:
            argv += ["--memory", self.memory]
        if self.summary_thinking:
            # The level the parent named for its summary calls, forwarded the way a set wall is:
            # absent when the parent kept its own level there, so the child inherits the default
            # (keep its own level) the same way the run does.
            argv += ["--summary-think", self.summary_thinking]
        if self.max_runtime:
            # 0 means the operator set no wall; forwarding a 0 would put a flag on the child's
            # line that says nothing, while a set wall travels the way the grants do.
            argv += ["--max-runtime", str(self.max_runtime)]
        if self.allow_web:                       # a child can never gain web the parent lacks
            argv += ["--web"]
        if self.web_private:                     # ...nor the private-address widening it was not given
            argv += ["--web-private"]
        if self.search_url:
            argv += ["--search-url", self.search_url]
        if prompt:
            argv += ["--task", prompt]
        return argv

    def next_id(self, parent_id: str) -> str:
        """The next child id for this parent, read from the folder the ids are named after.

        A counter living in the parent process restarts when the run is resumed, and the next child then
        reuses an earlier child's folder and log: `child_status` reads the previous child's `ended` step
        and reports the new child finished before it had started, so the parent acts on a stale result.
        The filesystem is the source of truth instead.

        The suffix is compared as a number, because comparing names answers `child9` once `child10`
        exists and would hand out an id already in use. The count is scoped to the parent id, since one
        folder holds the children of several runs.
        """
        prefix = f"{parent_id}-child"
        used = [int(cid[len(prefix):]) for cid in self.child_ids()
                if cid.startswith(prefix) and cid[len(prefix):].isdigit()]
        return f"{prefix}{max(used, default=0) + 1}"

    def spawn(self, run_id: str, prompt: str, *, model: str = "", max_turns: int = 0):
        """Start the child detached and return (proc, run_dir). Detached means addressable: the
        child outlives the parent, so this never waits and never attaches to the parent's signals.
        The prompt goes on the command line, so it is visible in the child's banner and process
        table -- an unattended child says what it was told to do.

        An existing folder is a refusal, not a reuse. `next_id` avoids collisions, but two parents can
        resume the same run, and a folder can be left by hand; appending to a child's log is the stale
        read this whole path exists to prevent.
        """
        root = self.child_dir(run_id)
        try:
            os.makedirs(root)
        except FileExistsError as exc:
            raise ValueError(f"a child folder for {run_id!r} already exists, so this id is taken; a second "
                             f"child writing there would append to the first child's log and a status read "
                             f"would answer with the older run's result") from exc
        argv = self.build_command(run_id, prompt, model=model, max_turns=max_turns)
        proc = subprocess_spawn(argv, root, api_key=self.api_key)
        return proc, root

    # --- inspecting the children this run owns --------------------------------
    #
    # Reading a child is done through its LOG, not a live process handle: the log is authoritative,
    # survives the parent restarting, and needs no witness — which is exactly the "did these finish,
    # and how" query the design owes an unattended run. A parent normally does none of this; it is
    # the always-available escape hatch.

    def child_ids(self) -> list[str]:
        """The child ids this run owns: the folder names under the runs root."""
        if not os.path.isdir(self.runs_root):
            return []
        return sorted(name for name in os.listdir(self.runs_root)
                      if os.path.isdir(os.path.join(self.runs_root, name)))

    def _steps(self, child_id: str) -> tuple[list[dict], str]:
        """A child's steps, through the one reader the log owns: a child still running may be
        mid-write, and a status query must not fail because of that. The second value names what
        was dropped, so the report says 'partly readable' rather than looking whole."""
        return record.read(os.path.join(self.runs_root, child_id, "log.jsonl"),
                           tolerate_damage=True)

    def wait(self, child_id: str, timeout: float = 60.0) -> str:
        """Block until this child records an `ended` step, or `timeout` seconds pass, then return the
        same report `status` gives. The sanctioned way to wait is *bounded*: the poll holds this run's
        turn (a long tool call, like a slow `bash`), so it can never hang a run on a child that never
        finishes. On timeout it returns the child's current state plus a note to re-check or wait
        again, leaving the choice with the caller rather than spinning forever.

        The wait reads the log, not a live handle: a child may have been launched by an earlier
        session, and finishing is a fact the log records, not a process to join."""
        _guard_child_id(child_id)
        ids = self.child_ids()
        if child_id not in ids:
            known = ", ".join(ids) or "none yet"
            return f"no child run {child_id!r} in this run's tree. your children: {known}"
        timeout = max(0.0, min(float(timeout), 300.0))      # bounded: a caller cannot wait unawares
        start = time.monotonic()
        while True:
            steps, damage = self._steps(child_id)
            if any(s["kind"] == "ended" for s in steps):
                waited = time.monotonic() - start
                note = f"  (waited {waited:.0f}s)" if waited >= 1 else ""
                return _describe(steps, child_id, damage) + note
            if time.monotonic() - start >= timeout:
                return (_describe(steps, child_id, damage)
                        + f"  — still running after {timeout:.0f}s; check later with "
                        + "child_status or wait again")
            time.sleep(0.5)

    def status(self, child_id: str = "") -> str:
        """One-line-each status for all children, or detail for one. This is the cheap completion
        query, answered from the logs rather than a process table, so it works across a parent
        restart and with nobody watching."""
        if child_id:
            _guard_child_id(child_id)
            if child_id not in self.child_ids():
                known = ", ".join(self.child_ids()) or "none yet"
                return f"no child run {child_id!r} in this run's tree. your children: {known}"
            steps, damage = self._steps(child_id)
            return _describe(steps, child_id, damage)
        ids = self.child_ids()
        if not ids:
            return "this run has no children. start one with the spawn tool."
        lines = []
        for cid in ids:
            steps, damage = self._steps(cid)
            lines.append(_describe(steps, cid, damage))
        return "\n".join(lines)


def _package_parent() -> str:
    """The folder that must be on PYTHONPATH for `python -m sah` to import the package. The parent
    run finds sah only because its working directory is the repo; a child launched with a different
    cwd needs the location named explicitly, because the package is a source tree, not an install."""
    import sah
    return os.path.dirname(os.path.dirname(os.path.abspath(sah.__file__)))


def subprocess_spawn(argv, root, api_key: str = ""):
    """Launch a detached child run process. Split out so the launch is a single stand-in seam: the
    argv and the detach can be checked without forking a live agent, which would be slow and flaky —
    and the command line and the detach are the contract worth checking.

    The child is started the way an unattended process must be: its own group, no console of the
    parent's to be killed along with, and stdin at the null device -- a run whose operator reads its log
    has nobody at a keyboard, and inherited stdin would put it in front of the parent's terminal.
    """
    import subprocess
    env = dict(os.environ)
    parent = _package_parent()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = parent + (os.pathsep + existing if existing else "")
    # The command-line flag does not cover a child that starts a sah run through its own shell tool,
    # because that command line is the model's to write. The environment travels with the process.
    env["SAH_NO_SPAWN"] = "1"
    if api_key:
        # The one channel the process table does not read. The child needs no flag for it, because
        # `openai.OpenAI` falls back to this variable -- the shared name is the contract.
        env[openai.API_KEY_ENV] = api_key
    log_path = os.path.join(root, "child.out")
    with open(log_path, "ab") as sink:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=sink,
                                stderr=subprocess.STDOUT, cwd=root, env=env,
                                **process.popen_unattended_kwargs())


def _guard_child_id(child_id: str) -> None:
    """A child id is a single folder name under the run's own children tree, never a path. Rejecting
    anything that is not one clean path component (a separator of either kind, or traversal) keeps a
    status query inside the tree the caller owns -- the same intrinsic boundary the memory tree
    enforces, not an fs grant. basename-equality is the platform-neutral form: it catches the
    active separator, and the explicit slash checks catch the other kind too."""
    if (not child_id or child_id in (".", "..") or child_id != os.path.basename(child_id)
            or "/" in child_id or "\\" in child_id):
        raise ValueError("child id must be a single name, not a path")


def _describe(steps: list[dict], child_id: str, note: str = "") -> str:
    """One child's state, read from its steps: did it finish and how, what it last said, and
    whether it is waiting on a question nobody answered. The log carries all of it; nothing here
    needs the live process. `note` carries what the reader had to drop, so a partly readable child
    is reported as one."""
    if not steps:
        return f"{child_id}: no log yet (starting, or never launched)"
    ended = next((s for s in reversed(steps) if s["kind"] == "ended"), None)
    if ended:
        state = f"{ended['reason']} (exit {ended['exit_code']})"
    else:
        state = "running"
    parts = [f"{child_id}: {state}" + (f"  (note: {note})" if note else "")]
    last_said = next((s["text"] for s in reversed(steps)
                      if s["kind"] == "message" and s.get("role") == "assistant" and s["text"]), None)
    if last_said:
        parts.append(f"  last said: {last_said.strip().splitlines()[0][:120]}")
    failed = [s for s in steps if s["kind"] == "message" and s.get("outcome") == "failed"]
    if failed:
        parts.append(f"  {len(failed)} failed turn(s): {failed[-1].get('error', '')[:80]}")
    asks = [s for s in steps if s["kind"] == "request"]
    if asks:
        parts.append(f"  asked {len(asks)} question(s); a child with no operator is refused by default")
    return "\n".join(parts)
