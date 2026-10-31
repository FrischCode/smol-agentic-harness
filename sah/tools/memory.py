"""sah.tools.memory — a library the run reads and appends to, not a log it rewrites.

The shape is deliberately small, and nothing in it is generated from the notes. A generated index can be
wrong about the note it indexes, and a reader cannot tell which of the two they are being asked to trust.
So: a folder tree is the taxonomy, the files inside are append-only notes, and four tools move through it —

    mem_tree    browse the structure (folders + notes, with line counts)
    mem_read    read one note by path — the exact bytes, ground truth, never a summary
    mem_grep    search every note for a phrase, with file:line hits — the librarian walk
    mem_note    write: append (default), create (refuses if it exists), overwrite (destructive,
                needs a reason, and backs up the old content first)

**Notes are append-only, and every entry carries provenance.** A timestamp, a session id, and a
`src:` tag go in an HTML comment the reader can ignore and a search can find: `grep src:user` locates
every user-issued line without parsing prose. Two things make that promise hold, and neither is free.
The stamp shape belongs to the core: a comment opener arriving in note text is defused to `&lt;!--`, so
content cannot author a stamp and pass itself off as someone else's. (The words `src:user` in prose still
match a search — only a stamp can be *authored*, and prose is not deleted to protect a substring.) And
`src` is a *claim* the writer made — the tool records who the writer said the content came from, not who
vouched for it; a run can type `src="user"`, and nothing in a log can tell that from the real thing.
Append-only is not caution — it is what lets a note be a *record of what was believed when*, so two
entries can disagree and the reader sees the disagreement instead of a silent rewrite. Overwrite
exists but is gated on a stated reason, and
copies the previous content to a timestamped `.bak` before replacing it — refusing the write when that
copy cannot be made, because "I meant to replace it" and "I destroyed it by accident" look identical
afterward unless the tool makes you say which.

**The tree is its own boundary.** A memory path cannot escape the memory root — `../`, absolute paths
and symlink tricks are refused — so the notes never reach into the filesystem the run's other grants
cover. That confinement is intrinsic to these tools, which is why they are absent from the policy's
address table: a memory note is not an arbitrary file the operator granted, it is an entry in a
namespace whose root the operator chose when they pointed `--memory` at a folder. Reading notes is
inert; the only thing a memory tool does to the world is write inside the tree.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import Annotated

from sah import ingest
from sah.contract import Param

# The separator between entries in a note: two newlines, so an appended entry starts on its own
# line and the stamp `format_entry` writes above it stays separately addressable by a search.
ENTRY_SEP = "\n\n"

# The filesystem's answer about where bytes really live, named at module level so the containment
# boundary can be stood in without the privileges a real link requires on Windows. Every containment
# question goes through it, so standing it in moves the whole boundary and cannot leave one tool behind.
_realpath = os.path.realpath


def _inside(path: str, root: str) -> bool:
    """Whether `path`'s real location is the root or below it.

    One decision, asked of the filesystem rather than inferred from the entry: an entry can be a link, and
    on Windows a directory junction is not even reported as one, so `is_symlink()` is not the question.
    Every walk, list and write consults this, because a boundary held in two functions is a habit, and
    habits drift apart."""
    try:
        base = _realpath(root)
        return os.path.commonpath([base, _realpath(path)]) == base
    except ValueError:                       # different drives on Windows; nothing is common
        return False


def _defuse(content: str) -> str:
    """Neutralise a comment opener in note text, so content cannot author a provenance stamp.

    `grep src:user` is only a real index of sources if the stamp shape is something a note body cannot
    type. Left raw, a line of text could invent `src:user`, or a whole entry with a fabricated time and
    session, and no search could tell the two apart. `&lt;!--` keeps the words, shows a reader that a
    comment was there, and stops opening a comment."""
    return content.replace("<!--", "&lt;!--")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_src(raw: str) -> str:
    """Reduce a claimed source to a safe token: user | assistant | joint | external:<name> | unknown.

    An unknown or empty claim becomes 'unknown' rather than an error — the write still happens, but
    the provenance line says honestly that nobody vouched for it. 'external:<name>' keeps the name so
    a citation to an outside source survives the normalization."""
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s.startswith("src:"):
        s = s[4:].strip()
    s = "".join(s.split())
    if s in ("", "unknown"):
        return "unknown"
    if s in ("user", "assistant", "joint") or s.startswith("external:"):
        return s
    return "unknown"


def format_entry(sid: str, content: str, src: str = "", header: str = "") -> str:
    """One entry with its provenance stamp on top.

    `header` is a second core-authored line (the overwrite reason) and is not defused; `content` is
    somebody else's text and is."""
    lines = [f"<!-- {_now()} [{sid}] src:{normalize_src(src)} -->"]
    if header:
        lines.append(header)
    lines.append(_defuse(content).strip())
    return "\n".join(lines) + "\n"


class Tree:
    """A memory tree rooted at one folder, which the notes can never leave.

    `sid` is the session id that appears in every provenance line; it is chosen by whoever builds
    the tree (usually the run id), never by the caller of a tool, so a note's session is a fact
    about the run and not the model's claim about itself."""

    def __init__(self, root, sid: str = "run"):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.sid = sid

    def ensure(self) -> None:
        """Create the tree's root folder if it is not there yet."""
        os.makedirs(self.root, exist_ok=True)

    # --- the one boundary ----------------------------------------------------

    def resolve(self, raw: str) -> str:
        """An absolute path inside the root, or refuse. This is the whole confinement rule: the same
        check before every read, list and write, so a note can never be written outside the tree by
        any tool. An absolute path is refused rather than silently relocated under the root -- a
        caller who typed `/etc/passwd` should hear 'no', not get a surprise file at <root>/etc."""
        s = str(raw).strip()
        cleaned = s.replace("\\", "/").lstrip()
        if cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
            raise ValueError(f"the path {raw!r} is absolute; memory paths are tree-relative")
        cleaned = cleaned.lstrip("/")
        if not cleaned:
            raise ValueError("a memory path is required")
        # A path ending in `.` or `..` names a place you stand and look from, not a note. Refusing those
        # by name is what the blanket `strip(".")` was reaching for, and did by renaming: it turned
        # `.drafts/x.md` into `drafts/x.md`, so the write landed somewhere nobody typed and the receipt
        # named that other place as though it had been the request. A leading dot is a folder name.
        last = cleaned.rstrip("/").rsplit("/", 1)[-1]
        if last in ("", ".", ".."):
            raise ValueError(f"the path {raw!r} names a folder to stand on, not a note to read")
        full = os.path.normcase(os.path.join(_realpath(self.root), cleaned))
        root = _realpath(self.root)
        if not _inside(full, root):
            raise ValueError(f"the path {raw!r} leaves the memory tree; notes stay inside it")
        return full

    def rel(self, full: str) -> str:
        """A path's form relative to the tree root, with forward slashes (the form the tools report)."""
        return os.path.relpath(full, self.root).replace("\\", "/")

    # --- browse ---------------------------------------------------------------

    def tree(self, subpath: str = "", max_depth: int = 3) -> str:
        """A directory listing of the tree: folders and notes, with line/byte counts. Structure to
        navigate by, not contents to read."""
        base = self.resolve(subpath) if subpath else _realpath(self.root)
        lines = [f"# {self.rel(base) if subpath else '(root)'}\n"]
        self._list(base, 1, "", max_depth, lines)
        return "\n".join(lines)

    def _list(self, dir_path: str, depth: int, prefix: str, max_depth: int, out: list) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return
        for e in entries:
            # Asked first, before the entry is trusted as either a folder or a note: a link out is neither
            # of the things the listing means to show, and recursing into it put another directory's
            # names and counts inside a tree whose promise is that it is its own boundary.
            if not _inside(e.path, self.root):
                out.append(f"{prefix}{e.name}{'/' if e.is_dir() else ''}  (outside the tree, not read)")
                continue
            if e.is_dir(follow_symlinks=False):
                out.append(f"{prefix}{e.name}/")
                self._list(e.path, depth + 1, prefix + "  ", max_depth, out)
            elif e.is_dir():
                # A link to a folder that is inside the tree. Following it would show the same notes twice
                # under two paths, and a link to an ancestor would be walked until the depth limit — but
                # dropping it silently would leave a folder that is refused reading exactly like a folder
                # that is empty, which is the lie this listing exists to avoid.
                out.append(f"{prefix}{e.name}/  (link, not followed)")
            elif e.name.endswith(".md") and e.name != "INDEX.md":
                lines, size = _stats(e.path)
                out.append(f"{prefix}{e.name}  ({lines}L, {size}B)")

    # --- read -----------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 200) -> str:
        """The exact bytes of one note, ground truth and never a summary, because a summary of a
        note is one step away from the drift this whole design refuses. offset/limit page a long
        note; the header says where in the file you are."""
        full = self.resolve(path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"{path} is not a note; use mem_tree to browse")
        with open(full, encoding="utf-8") as h:
            all_lines = h.read().split("\n")
        # 1-based, because `read`, `run_log` and this tool answer the same question — "show me from line
        # N" — in the same session, and one of them meaning N-1 is how a note comes to be cited from the
        # wrong line. A 0 or a negative is a guess, not a wrap: it pages from the top.
        offset = max(1, int(offset))
        start = offset - 1
        window = all_lines[start:start + max(1, int(limit))]
        # Capped to what a result can carry, because `limit` alone is not a bound the loop can rely
        # on: an over-long page is clipped at the choke point and spilled, copying a note this tool can
        # already page with `offset`. The header counts the lines actually shown, so the range it names
        # is the range the caller received.
        shown, meta = ingest.clip_head("\n".join(window), max_lines=ingest.BODY_LINES,
                                       max_bytes=ingest.BODY_BYTES)
        last = start + meta["shown_lines"]     # clip_head counts what it kept, either way
        head = f"# {self.rel(full)} — lines {start + 1}-{last} of {len(all_lines)}"
        if last < len(all_lines):
            return f"{head}\n\n{shown}\n\n[page full; continue with offset={last + 1}]"
        return f"{head}\n\n{shown}"

    # --- search ---------------------------------------------------------------

    def grep(self, pattern: str, subpath: str = "", max_results: int = 40) -> str:
        """Every note scanned for the phrase (case-insensitive substring), hits as path:line. The
        librarian walk: use it when you know what was said and not where."""
        base = self.resolve(subpath) if subpath else _realpath(self.root)
        needle = str(pattern).lower()
        matches = []
        files, skipped_files, skipped_dirs = _walk(base, self.root)
        for f in files:
            if f == "INDEX.md":
                continue
            try:
                with open(os.path.join(self.root, f), encoding="utf-8") as h:
                    content = h.read()
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(content.split("\n")):
                if needle in line.lower():
                    matches.append(f"{f}:{i + 1}: {line.strip()}")
                    if len(matches) >= max(1, int(max_results)):
                        break
            if len(matches) >= max(1, int(max_results)):
                break
        where = self.rel(base) if subpath else "(root)"
        # "No matches" is a claim about a set of files. When the set was smaller than the tree — a link
        # out, which is exactly the entry that cannot be shown — the answer has to say so, or the reader
        # concludes the fact was never written when the truth is that it was never looked at.
        gap = _skipped_note(skipped_files, skipped_dirs)
        if not matches:
            return f"No matches for {pattern!r} in {where}.{gap}"
        return f"# {len(matches)} match(es) for {pattern!r}\n\n" + "\n".join(matches) + gap

    # --- write ----------------------------------------------------------------

    def note(self, path: str, content: str, mode: str = "append", src: str = "",
             reason: str = "") -> str:
        """Write one entry to a note. Returns a line describing the new state; raises on refusal.

        append     add to the note, creating it if new. The default, and the common case.
        create     a brand-new note; refuse if one is already there, so two unrelated facts cannot
                   be merged by a path typo.
        overwrite  replace the note entirely — DESTRUCTIVE, so it requires a non-empty `reason`, and
                   the previous bytes are copied to a timestamped `.bak` first. If that copy cannot be
                   made the overwrite is refused. The reason is recorded in the
                   note itself, so the tree carries why it was rewritten, not just that it was."""
        full = self.resolve(path)
        # A note is what the tree can find, and the tree finds `*.md`. Writing `todo.txt` answered
        # "Written: todo.txt" with the bytes really on disk, and then nothing — not `mem_tree`, not
        # `mem_grep` — could ever surface it again. The alternative, quietly writing `todo.md` instead, is
        # the rename this module refuses for dot-paths, so the refusal is the honest half: say no, and
        # name the word that works.
        if not full.lower().endswith(".md"):
            raise ValueError(f"a note is a .md file; {self.rel(full)!r} would be invisible to mem_tree "
                             f"and mem_grep. Write it as "
                             f"{os.path.splitext(self.rel(full))[0] + '.md'!r} instead")
        mode = mode or "append"
        if mode not in ("append", "create", "overwrite"):
            raise ValueError(f"unknown mode {mode!r}; use append, create or overwrite")
        if mode == "overwrite" and not reason.strip():
            raise ValueError(f"overwriting {self.rel(full)} needs a reason — this is the one "
                             "destructive write, say why (the previous content is copied to a .bak first)")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        entry = format_entry(self.sid, content, src)
        kept = ""                                  # the backup this call actually made, named in the receipt
        if mode == "create":
            if os.path.exists(full):
                raise ValueError(f"{self.rel(full)} already exists; use append to add to it or "
                                 "overwrite to replace it (destructive)")
            with open(full, "w", encoding="utf-8") as h:
                h.write(entry)
        elif mode == "overwrite":
            if os.path.exists(full):
                # Before the write, not after: if the copy cannot be made, the note stays as it was.
                kept = self.rel(_backup(full))
            entry = format_entry(self.sid, content, src,
                                 header=f"<!-- overwrite reason: {_defuse(reason.strip())} -->")
            with open(full, "w", encoding="utf-8") as h:
                h.write(entry)
        else:
            existing = ""
            if os.path.exists(full):
                with open(full, encoding="utf-8") as h:
                    existing = h.read()
            prefix = ENTRY_SEP if existing else ""
            needs_nl = "\n" if (existing and not existing.endswith("\n")) else ""
            with open(full, "a", encoding="utf-8") as h:
                h.write(needs_nl + prefix + entry)
        lines, size = _stats(full)
        # The receipt names the file this call actually wrote, because the claim is the guarantee: a
        # receipt that says "kept in x.bak" while x.bak was never written is worse than silence.
        extra = f" (previous content kept in {kept})" if kept else ""
        return f"Written: {self.rel(full)} ({mode}) — now {lines} lines, {size} bytes{extra}."


# --- helpers, no state -------------------------------------------------------

def _stats(path: str) -> tuple[int, int]:
    try:
        with open(path, "rb") as h:
            data = h.read()
    except OSError:
        return (0, 0)
    return (data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1), len(data))


def tools_for(tree: "Tree") -> dict:
    """The four memory tools bound to a tree, as plain typed functions for the tool registry. Each
    returns a string (a note, a listing, hits, a write receipt) and raises on refusal — the loop
    turns a raise into a failed result the model reads, so a refusal is actionable prose, not a
    crash. Typed hints are the schema: the model sees exactly these parameters.

    These declare `self`: the tree is their whole boundary — a note cannot leave it by construction, so
    the run's file grants neither widen nor restrict them, and the policy lets them through while
    recording the note each one worked on. Reading is
    inert; the only world-affecting write stays inside a folder the operator chose with --memory."""

    def mem_tree(subpath: Annotated[str, Param("folder to list inside; empty lists the root")] = "",
                 max_depth: Annotated[int, Param("how deep to walk")] = 3) -> str:
        """The counts are the only fact a listing can give about a note's contents without opening it:
        a note that is 12 lines and one that is 4,000 are answered differently by `mem_read`, and the
        count is what tells them apart at the cheap end. A link out of the tree is named rather than
        walked or dropped, because a listing whose promise is that it is the tree's boundary must not
        show another directory's names inside it."""
        return tree.tree(subpath, max_depth)

    mem_tree.description = ("Browse the memory tree: folders and notes with line counts. "
                            "Structure to navigate by. Pass a subpath (e.g. 'projects') to list inside it.")

    def mem_read(path: Annotated[str, Param("tree-relative note path, e.g. projects/x.md")],
                 offset: Annotated[int, Param("first line, 1-based")] = 1,
                 limit: Annotated[int, Param("max lines")] = 200) -> str:
        """Exact, never a summary: a note changed since it was last seen is exactly what the run must
        see, and a paraphrase would let the run act on what a summary said rather than what the note
        says. Paging by line keeps a long note reachable without a second tool: offset and limit are
        the same shape `read` carries, so one way of reading covers both the tree and the files."""
        return tree.read(path, offset, limit)

    mem_read.description = ("Read one note by its tree-relative path. Returns the exact text, never a "
                            "summary. offset/limit page a long note; offset counts lines from 1.")

    def mem_grep(pattern: Annotated[str, Param("phrase to find, case-insensitive substring")],
                 subpath: Annotated[str, Param("limit the search to this subpath")] = "",
                 max_results: Annotated[int, Param("max hits to return")] = 40) -> str:
        """A substring, not a regular expression: notes are prose, and the phrase the run is looking
        for is what was said, not a pattern to be matched. The search should fail by finding nothing,
        not by failing to parse — and a phrase that is not in any note is itself the answer to "was
        this ever said"."""
        return tree.grep(pattern, subpath, max_results)

    mem_grep.description = ("Search every note for a phrase (case-insensitive), hits as path:line. "
                           "The librarian walk: use it when you know what was said and not where it lives.")

    def mem_note(path: Annotated[str, Param("tree-relative note path to write")],
                 content: Annotated[str, Param("markdown to add")],
                 mode: Annotated[str, Param("append adds; create fails if the note exists; overwrite "
                                            "replaces it and needs reason",
                                            ("append", "create", "overwrite"))] = "append",
                 src: Annotated[str, Param("who this content is from: user | assistant | joint | "
                                           "external:<name>")] = "",
                 reason: Annotated[str, Param("required for overwrite: why replacing is intended")]
                 = "") -> str:
        """One write carries one source, because the stamp is a search axis: mixed sources under one
        stamp would be found as one voice, which is the lie `src` exists to prevent — mixed sources go
        in separate writes. `overwrite` is the only act that destroys, and it does so under a
        precondition: the .bak copy happens before the write and a failed copy is a refused write, so
        "recoverable" is a fact about the copy, not a hope."""
        return tree.note(path, content, mode, src, reason)

    mem_note.description = ("Write to the memory tree (append default / create / overwrite). The path "
                            "must end in .md — that is what the other memory tools look for. Append adds "
                            "a provenance-stamped entry and creates the note if new; create makes a "
                            "brand-new note and refuses if one exists; overwrite replaces the note "
                            "entirely — destructive: it requires reason, copies the previous content to a "
                            "timestamped .bak first (and refuses the write if that copy fails), so the "
                            "old version stays recoverable. One write = one source: src is stamped so the "
                            "tree can be searched by source, and sources mix in separate writes.")

    # Declared `self` because the boundary is the tree, not the run's grants: the policy lets these
    # through and resolves the place named, so the log can still answer "what did that run read".
    mem_tree.access, mem_tree.address = "self", "subpath"
    mem_read.access, mem_read.address = "self", "path"
    mem_grep.access, mem_grep.address = "self", "subpath"
    mem_note.access, mem_note.address = "self", "path"

    return {"mem_tree": mem_tree, "mem_read": mem_read, "mem_grep": mem_grep,
            "mem_note": mem_note}


# the one filesystem act a backup performs, named so the copy can be stood in (to fail)
_COPY = shutil.copyfile


def _backup(path: str) -> str:
    """Copy the current note somewhere recoverable before an overwrite, and say where.

    The copy is the *precondition* for a destructive write, not a courtesy that runs if convenient: a
    backup that fails and is swallowed leaves the overwrite to destroy content the interface promises to
    keep — and the receipt then states a falsehood about a file that was never written. A locked file, a
    full disk or a permissions problem are ordinary on Windows, so refusing costs a retry while proceeding
    costs the note.

    One file per overwrite rather than one per note, stamped in UTC so the names sort oldest first. With a
    single `.bak`, a second overwrite destroyed the first version, and "nothing silently disappears" was
    true only of the most recent loss.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    target = f"{path}.{stamp}.bak"
    n = 2
    while os.path.exists(target):                       # two overwrites inside one second
        # Digits appended, not `-1`: a name is the only record of which version is older, and `-` (0x2D)
        # sorts before the `.` (0x2E) that follows the stamp, which would order the names youngest-first.
        target = f"{path}.{stamp}{n}.bak"
        n += 1
    try:
        _COPY(path, target)
    except OSError as exc:
        raise OSError(f"overwrite refused: the current content could not be copied to a backup ({exc}); "
                      "nothing was replaced") from exc
    return target


def _skipped_note(files: int, dirs: int) -> str:
    """The sentence that turns a silent omission into a disclosure, or '' when nothing was omitted."""
    bits = []
    if files:
        bits.append(f"{files} file(s)")
    if dirs:
        bits.append(f"{dirs} folder(s)")
    return f"\n\n[{', '.join(bits)} outside the tree were not searched]" if bits else ""


def _walk(base: str, tree_root: str) -> tuple[list, int, int]:
    """Every note under `base` that really lives inside the tree, made relative to the root (what a reader
    greps for), sorted for a deterministic order; plus the counts of what was left out.

    `os.walk` lists and descends into what a link points at — a linked file's contents, and, measured on
    Windows where a directory junction is not reported as a link at all, a linked directory's whole
    subtree. So containment is asked of the filesystem for every candidate rather than inferred from the
    entry, and anything outside is counted, not dropped quietly."""
    out, skipped_files, skipped_dirs = [], 0, 0
    for dirpath, dirnames, filenames in os.walk(base):
        keep = []
        for d in sorted(dirnames):
            # A folder outside the tree has nothing inside it that is inside, so it is not descended into.
            if _inside(os.path.join(dirpath, d), tree_root):
                keep.append(d)
            else:
                skipped_dirs += 1
        dirnames[:] = keep
        for name in sorted(filenames):
            if not name.endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            if not _inside(full, tree_root):
                skipped_files += 1
                continue
            out.append(os.path.relpath(full, tree_root).replace("\\", "/"))
    return sorted(out), skipped_files, skipped_dirs


class _MemoryExtension:
    """The memory tools bound to one tree, wrapped as an extension (duck-typed: `assemble` reads
    `.tools`). Offered only when the run was given a `--memory` tree."""

    def __init__(self, funcs):
        self.tools = tuple(funcs)


def extension(tree: "Tree"):
    """The memory capability for a given tree, as an extension for `sah.ext.assemble`."""
    return _MemoryExtension(tools_for(tree).values())
