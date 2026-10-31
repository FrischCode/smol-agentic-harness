"""sah.confine — the one place that decides whether a call happens.

One callable, because there is exactly one question worth asking at a veto point: *where is the
thing this call touches, and was that place given to the run?* Everything else — how a tool is
declared, how arguments are fitted, what gets recorded — belongs to the modules that own it, and
this module refuses to duplicate it.

The rules, in order:

1. **A tool declares how it touches the world, and that declaration is what is judged.** `read` and
   `write` name a place checked against the grants; `exec` takes the separate exec axis; `self` means the
   tool confines its own ground (a memory tree, a runs root, a web address guard) and the place it names
   is recorded rather than judged; `none` touches no place. The declaration travels with the tool (see
   `sah.contract.declaration`), because a table keyed by *name* left any tool someone forgot to add —
   `cat`, `edit`, an extension's — confined by nothing and silently allowed.
2. **The address is resolved before it is judged.** `..` and symlinks are collapsed to the real
   location, so the policy asks where the file actually is rather than where the caller said it
   was. A relative path is resolved against the working directory the run started in.
3. **Reads need a read grant; writes need a write grant.** A write grant contains its own read
   grant, since a run that may write a folder must be able to read what is in it.
4. **Execution is a separate axis and it says what it costs.** Granting a shell makes path
   confinement advisory: any command can write anywhere. That sentence is returned as the
   verdict's reason, so it lands in the `operation` step of every shell call — a disclosure that
   renews itself each time rather than a line in a manual nobody reads.
5. **A refusal names the act that would have permitted it.** "Permission denied" sends the reader
   to the source. The reason names the flag and the path, because the useful shape of a refusal
   is a repair.
6. **A parent owns its child runs, and ownership is not a grant.** A run that spawned a child can
   always read that child's log and artifacts, because it is *theirs* — not because a flag said so.
   This is its own axis beside the invocation's grants (a fixed root covering the run's own
   `children/` tree), deliberately kept out of the read-grant list so it never reads as a widening
   someone performed. It is read-only (owning a child is inspecting it, not writing into it) and it
   cannot be turned off, because it is not a permission the run was given but a fact of having
   spawned. Normally a parent never touches it: `spawn` hands back the child's handle, and the
   artifact usually lands in a shared folder — inspection is the always-available escape hatch.

7. **The wall may be taken down, by hand, for one run.** `--no-confine` opens every path this user can
   reach, for the mounted drive nobody thought to grant and the folder three links away. It is its own
   axis rather than a giant root, so the record still names the address each call touched; it grants no
   shell (the axes keep their separate costs); it says so in the boot box, in `<situation>` and in
   `started.grants`; and it is never forwarded to a child, because one decision at one terminal must not
   become the standing reach of every run beneath it.

Grants come from the invocation and only from the invocation — the roots and the open axis alike.
Nothing here reads them from a log or a config file: widening is an act someone performs, not a state a
run inherits because a previous run happened to be wider.
"""
from __future__ import annotations

import os

from sah.contract import Verdict

EXEC_DISCLOSURE = ("execution is granted in this run, so path confinement does not bind this "
                   "call: a shell command may touch anything this user can")
OPEN_DISCLOSURE = ("path confinement is off for this run (--no-confine): this call may touch "
                   "anything this user can")


def resolve(raw: str) -> str:
    """The real location of an address: absolute, `..` collapsed, symlinks followed. Case is
    folded where the filesystem ignores it, so a grant cannot be escaped by respelling it."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(str(raw)))))


def inside(target: str, roots) -> bool:
    """Is `target` the root or beneath it? Both are already resolved."""
    for root in roots:
        if target == root:
            return True
        try:
            if os.path.commonpath([root, target]) == root:
                return True
        except ValueError:      # different drives on Windows, or mixed UNC and local paths
            continue
    return False


class Confine:
    """The policy the loop calls: `verdict = confine(name, arguments, access=..., address=...)`.

    The declaration comes from the loop, which read it from the tool the model called — the tool never
    decides its own permission, and the policy does not have to know any tool's name.

    `read_roots` and `write_roots` arrive from the invocation. Empty read roots means the run can
    see no file at all, which is a legitimate way to start a session and is stated as such in
    every refusal rather than worked around.

    `unrestricted` is the other end of the same dial: the operator took the wall down, so no place is
    judged. It is not a root list made of every drive — see rule 7 in the module docstring — and it is
    kept beside the roots rather than inside them so that "everything" and "nothing" stay two facts."""

    def __init__(self, read_roots=(), write_roots=(), allow_exec: bool = False, owned_roots=(),
                 unrestricted: bool = False, project_notes: bool = False):
        self.read_roots = [resolve(root) for root in read_roots]
        self.write_roots = [resolve(root) for root in write_roots]
        self.allow_exec = bool(allow_exec)
        self.unrestricted = bool(unrestricted)
        # Read-only, always-allowed roots that are NOT grants: the run's own child tree. Kept apart
        # from read_roots so the operator's banner and the model's <situation> report them as
        # ownership, never as a granted widening someone performed.
        self.owned_roots = [resolve(root) for root in owned_roots]
        # A reach axis no per-call veto consults: it decides whether the working directory's own
        # instruction file is loaded into the system prompt at boot. It is a grant in the reach sense
        # (it comes from the invocation, it is recorded and compared on resume), but `__call__` never
        # looks at it, because no tool call needs it. Kept beside the veto axes rather than in a second
        # object so the run's whole reach stays one readable source.
        self.project_notes = bool(project_notes)

    @property
    def readable_roots(self) -> list[str]:
        """The roots a read is judged against: the read roots plus the write roots (a write grant
        contains its own read grant), de-duplicated in order."""
        return list(dict.fromkeys(self.read_roots + self.write_roots))

    def reach(self) -> dict:
        """What this policy will let the run do, as data rather than as a verdict.

        Two readers, one source: the loop records this as `started.grants`, and a later invocation compares
        itself against what the log holds. Neither restates the axes on its own, because a run whose record
        described one policy while a different object refused its calls would be auditable only by luck.

        The scope is the reach this run *began* with. An accepted ask widens the policy for the rest of the
        run, and that widening is its own pair of steps (an `ask`, a `decision` with `outcome: accepted`) —
        so the final reach of a run is this value plus the decisions that followed it, never this value alone.

        `owned_roots` is absent on purpose: a run's own folder is a standing read-only ownership, not a
        grant, and folding it in would report a reach the operator never gave.

        `unrestricted` is present for the opposite reason. Without it an open run and a run granted
        nothing both record empty root lists, and a resume comparing the two would report no change in
        a reach that changed completely.

        `project_notes` is a reach axis no veto consults: it decides whether the working directory's
        own instruction file is loaded into the prompt at boot. It is present for the same reason as
        `unrestricted` — a run that granted it and a run that did not both record the same root lists,
        and a resume comparing the two would otherwise report no change in a reach that changed."""
        return {"read": sorted(self.read_roots), "write": sorted(self.write_roots),
                "exec": self.allow_exec, "unrestricted": self.unrestricted,
                "project_notes": self.project_notes}

    def __call__(self, name: str, arguments: dict, *, access: str, address) -> Verdict:
        if access == "exec":
            if not self.allow_exec:
                return Verdict(False, f"running {name} is not granted in this run; start it with "
                                      "--exec, which also makes path confinement advisory")
            # The working directory is recorded as typed. It is not judged, so there is nothing to
            # resolve it against, and an empty cwd is honest as nothing rather than as the run's folder.
            return Verdict(True, EXEC_DISCLOSURE, address=str(arguments.get(address, "")))

        if access in ("self", "none"):
            # The tool judges its own ground, so there is nothing to decide here. A self-confined tool
            # still names the place it worked on, and it is recorded **as typed**: a memory note's
            # `a.md` is relative to the tree, a child id to the run's children/, a URL to nowhere on
            # disk. Resolving any of them against the working directory would write a location that
            # does not exist into the one field a later reader trusts to ask "what did that run touch".
            # No namespace prefix either, because the step already carries the tool's name.
            if access == "self" and address:
                raw = str(arguments.get(address, ""))
                if raw:
                    return Verdict(True, "", address=raw)
            return Verdict(True)

        if access not in ("read", "write"):
            # `tools.declaration` checks the closed set at boot, so arriving here means a bug. A bug in
            # the policy lands on the refused side, which is the one side that cannot damage anything.
            return Verdict(False, f"{name} declares access={access!r}, which this policy does not "
                                  "understand; the call is refused rather than allowed by default")

        field = address
        raw = str(arguments.get(field, ""))
        if not raw:
            return Verdict(False, f"{name} needs args.{field}; no address was given, so nothing "
                                  "can be checked")

        target = resolve(raw)
        if self.unrestricted:
            # Judged against nothing, but still resolved and still recorded: an open run whose log cannot
            # answer "what did it touch" is unauditable in exactly the way a confined one is not, and the
            # disclosure renews on every call rather than sitting in a manual nobody reads.
            return Verdict(True, OPEN_DISCLOSURE, address=target)
        writing = access == "write"
        # A parent may always read inside its own children tree, before any grant is consulted.
        # This is ownership, not a read grant, so it applies to reads only and never to writes.
        if not writing and self.owned_roots and inside(target, self.owned_roots):
            return Verdict(True, "a parent may always read its own child runs", address=target)
        roots = self.write_roots if writing else self.readable_roots
        if inside(target, roots):
            return Verdict(True, "", address=target)

        asked = os.path.abspath(os.path.expanduser(raw))
        # Two facts about the name as typed, not about the name after this function has tidied it. `abspath`
        # collapses `..`, so a comparison of the two resolved names can never see a dot-dot escape; and
        # `realpath` on Windows expands an 8.3 short name, which is a change of spelling and not a link. The
        # gap is therefore looked for in the typed string, and the notice names the name that was typed
        # rather than the place meant -- the verdict already states the place it judged, so repeating it
        # would be prose for the sake of it. Naming the mechanism (a symlink) would be a claim this function
        # cannot support, and a notice that cries symlink on a short name teaches a reader to disbelieve the
        # case the notice exists for.
        dotted = os.pardir in raw.replace("\\", "/").split("/")
        elsewhere = resolve(asked) != os.path.normcase(asked)
        via = "" if not (dotted or elsewhere) else f" (typed as {raw}, which resolves elsewhere)"
        # The advice names the exact address, never the folder around it. `inside()` takes a file as a
        # root, `widen` grants exactly what was asked, and the run asked for this file: handing out the
        # parent is a widening nobody requested, and — the reason in `widen` — it teaches the operator to
        # grant a drive. The two refusals used to advise `os.path.dirname(target)`, which contradicted the
        # module's own ruling while the asking path beside it did the narrow thing.
        if writing:
            return Verdict(False, f"writing {target}{via} is not granted; this run has no write "
                                  f"grant covering it ({_say(self.write_roots, 'write')}) — "
                                  "start it with --write " + target,
                           address=target, ask=target)
        return Verdict(False, f"reading {target}{via} is not granted; "
                              f"{_say(self.readable_roots, 'read')}"
                              " — start the run with --grant " + target
                        + " (asked as " + asked + ")", address=target, ask=target)

    def widen(self, address: str, *, writing: bool) -> str:
        """Grant one resolved address for the rest of the run, and return it.

        The smallest act is the address that was asked for, never the folder around it. That is
        the point of asking rather than advising: a refusal that suggests granting a parent teaches
        the operator to grant a drive."""
        target = resolve(address)
        (self.write_roots if writing else self.read_roots).append(target)
        return target


def _say(roots, verb: str) -> str:
    if not roots:
        return f"nothing is granted for {verb} in this run"
    return "granted for " + verb + ": " + ", ".join(roots)
