"""sah.tools.recall — read the record of past runs. This is *why* the log's fields are a stable query
interface: the main reader of old runs is the model itself, asking "what did we work on four sessions
ago", "pull up yesterday's session", "what failed last week."

**Recall reads raw run logs; it is not the memory tree.** Memory is curated, append-only notes the
agent writes on purpose. Recall is the transcript of what actually happened, read-only. The two are
complementary and stay apart: recall never writes, and a note is not a run and vice versa. Confusing
them would put an un-curated firehose where a distilled fact belongs.

**The runs root is its own boundary**, exactly like the memory tree and a parent's own directory. A
run id is a single folder name under the root the operator chose with `--runs-root`, never a path —
traversal, absolute paths and separators are refused — so recall reaches only the run folders in that
namespace, whatever the run's other grants cover. It is offered only when a root is named, because a
run that was not pointed at a history has none to consult.

A run's log is rendered by `sah.export` (one renderer, not a second), then paged through
`sah.ingest` like any large source, so reading a long run cannot flood the window.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from sah import export, ingest, record
from sah.contract import Param


def _guard_run_id(run_id: str) -> None:
    """One folder name under the runs root, never a path — the same intrinsic boundary a parent uses
    to read its own children, kept independent so a recall query cannot climb out of the history."""
    if (not run_id or run_id in (".", "..") or run_id != os.path.basename(run_id)
            or "/" in run_id or "\\" in run_id):
        raise ValueError("run id must be a single name, not a path")


def _steps(run_dir: str) -> tuple[list[dict], str]:
    """A run's steps, through the one reader the log owns: a run still going may be mid-append, and
    a history query must report on it rather than crash on a half-written last record. The second
    value is what was dropped, so the row can say so instead of looking intact.

    Read with the lenient vocabulary policy, because a history question about a run written by a newer
    build is answered by a report, not by a refusal it cannot act on.

    The steps come back through `lineage`: a cut is an operator act, not a step of the conversation, so
    what a history query asks about is the line in force, and the detached tail — and the cut that
    detaches it — stay in the raw file, where the audit finds them. A cut the log cannot project is
    reported in the note and shown uncut, rather than refusing the query it cannot answer.
    """
    steps, dropped = record.read(os.path.join(run_dir, "log.jsonl"), tolerate_damage=True, strict=False)
    try:
        steps = record.lineage(steps)
    except record.Refused as cut:
        dropped = (dropped + " " if dropped else "") + f"cut not projectable: {cut}; the run is shown uncut"
    return steps, dropped


def _iso_days_back(since: str):
    """Turn a `since` filter into a cut-off date string (YYYY-MM-DD), or None for 'no filter'.
    Accepts `Nd` (last N days) or a plain ISO date. Returns '' to signal a value that could not be
    understood, so the caller says so rather than silently returning everything."""
    since = since.strip()
    if not since:
        return None
    if since[-1:].lower() == "d" and since[:-1].isdigit():
        return (datetime.now(timezone.utc) - timedelta(days=int(since[:-1]))).strftime("%Y-%m-%d")
    if len(since) >= 10 and since[4] == "-" and since[7] == "-":
        return since[:10]
    return ""


_UNDATED = ""      # the peek's "I cannot tell", chosen to sort before every real date


def _describe(steps: list[dict], run_id: str, note: str = "") -> str:
    """One line per run for the index: when, what model, how it ended, how much happened, and the
    objective it was given. All of it is already in the log; nothing here re-derives a verdict."""
    started = next((s for s in steps if s["kind"] == "started"), None)
    ended = next((s for s in reversed(steps) if s["kind"] == "ended"), None)
    day = (started or ended or steps[0])["ts"][:10] if steps else "?"
    model = (started or {}).get("model", "?")
    verdict = f"{ended['reason']} (exit {ended['exit_code']})" if ended else "no end recorded"
    objective = next((s.get("text", "") for s in steps
                      if s["kind"] == "message" and s.get("role") == "user"), "")
    objective = objective.strip().splitlines()[0][:90] if objective.strip() else "(no objective)"
    calls = sum(1 for s in steps if s["kind"] == "call")
    return (f"- {run_id}  ·  {day}  ·  {model}  ·  {verdict}  ·  {calls} call(s)\n"
            f"    {objective}" + (f"\n    note: {note}" if note else ""))


class Runs:
    """The history rooted at one directory whose subfolders are run directories (each with a log)."""

    def __init__(self, runs_root: str):
        self.root = os.path.abspath(str(runs_root))

    def run_ids(self) -> list[str]:
        """The run ids under this root: the folder names that contain a `log.jsonl`, sorted."""
        if not os.path.isdir(self.root):
            return []
        return sorted(name for name in os.listdir(self.root)
                      if os.path.isfile(os.path.join(self.root, name, "log.jsonl")))

    def _steps(self, run_id: str) -> tuple[list[dict], str]:
        return _steps(os.path.join(self.root, run_id))

    def _peek_day(self, run_id: str) -> str:
        """The date the run's own first step names, read from one line -- or `_UNDATED` when it cannot say.

        The index orders by day, and a folder name is only a reliable proxy for a day while every id was
        minted with a timestamp in it; a hand-named folder breaks that. So the day is asked of the log.

        This is not a second reader and it decides nothing about the log's health. It parses one line under
        the policy the full reader uses and, on any doubt, says it cannot tell, so the caller opens the whole
        file rather than dropping a run it could not date: a run missing from a history index is worse than a
        run costing one extra read.

        What it says when ignorant is `_UNDATED`, the empty string, and that is a sorting decision rather than
        a formatting one. The first candidate was "?", which reads as unknown to a person and sorts *above*
        every digit to a comparison -- so a log with a torn first line presented itself as the newest history
        on the box. Unknown must sort last in a newest-first list, because last is where a reader expects what
        they cannot place."""
        try:
            with open(os.path.join(self.root, run_id, "log.jsonl"), encoding="utf-8") as handle:
                line = handle.readline().strip()
            return record.loads(line, strict=False)["ts"][:10]
        except Exception:
            return _UNDATED

    @staticmethod
    def _haystack(run_id: str, steps: list[dict]) -> str:
        """What `grep` is allowed to match: the run id, the model, and what the operator asked for.

        A repr of the steps (result bodies, reasoning, the contents of files the run read, and all the
        punctuation of nested dicts) would match every run that merely *touched* a file holding the word,
        which is not a history of intents, and it would miss any objective asked later in a long session,
        which is exactly the thing a caller is trying to find again.

        Every user message counts, not just the first: a session is a conversation, and an index that can
        only see its opening line cannot answer "which run was about the quay bridge"."""
        parts = [run_id]
        for step in steps:
            if step["kind"] == "started":
                parts.append(str(step.get("model", "")))
            elif step["kind"] == "message" and step.get("role") == "user":
                parts.append(str(step.get("text", "")))
        return " ".join(parts).lower()

    def index(self, since: str = "", grep: str = "", limit: int = 20) -> str:
        """Newest-first list of past runs, optionally filtered by date and a search term."""
        cutoff = _iso_days_back(since)
        if cutoff == "":
            return f"recall: I could not read since={since!r}; use 'Nd' (e.g. 7d) or a date like 2026-10-20."
        ids = self.run_ids()
        if not ids:
            return f"no run logs under {self.root}. point --runs-root at a folder of run directories."
        term = (grep or "").lower()
        # Order on the cheap fact first, then open as few logs as the page needs.
        #
        # Measured on a root of 60 logs at 208 KiB each: reading every one took 700 ms per call, and 87% of
        # the per-log cost was parsing lines the caller never sees. Streaming was the suggested remedy and
        # cannot work here -- a line has to be parsed to be judged, so the parse is the cost and not the
        # reading. The saving available is in how much gets opened at all: one line gives the day, the day
        # does the ordering and the date filter, and only then are logs opened.
        dated = [(self._peek_day(rid), rid) for rid in ids]        # (day, run id), and kept that way round
        candidates = [(day, rid) for day, rid in dated
                      if cutoff is None or day == _UNDATED or day >= cutoff]
        # Descending on (day, id). `_UNDATED` is the empty string, which is below every date, so a run that
        # could not be dated lands last and stays a candidate: an undated run in the list is a fact, one
        # silently missing is a gap.
        candidates.sort(key=lambda r: (r[0], r[1]), reverse=True)

        rows: list[tuple[str, str, str]] = []
        opened = 0
        cap = max(1, int(limit))
        for day, rid in candidates:
            # With no search term the page fills from the newest end and the rest goes unread. With one,
            # every candidate is opened: `grep` is a question about all of history, and a search that quietly
            # searched only the page it was going to show would answer "no runs matched" about logs it never
            # read. The asymmetry is the caller's promise -- a page is a page, a search is a search.
            if not term and len(rows) >= cap:
                break
            steps, damage = self._steps(os.path.join(self.root, rid))
            opened += 1
            if term and term not in self._haystack(rid, steps):
                continue
            rows.append((day, rid, _describe(steps, rid, damage)))

        shown = rows[:cap]
        if not shown:
            return f"no runs matched (since={since or 'any'}, grep={grep or 'none'})."
        head = f"{len(candidates)} run(s) under {self.root}"
        if not term and opened < len(candidates):
            head += (f"; opened {opened} of {len(candidates)} logs to fill this page"
                     f" ({len(candidates) - opened} not read -- raise limit, or use grep, to reach them)")
        elif len(rows) > len(shown):
            head += f" (showing {len(shown)}; raise limit or narrow with since/grep)"
        lines = [head, ""]
        lines += [line for _, _, line in shown]
        lines.append("read any one in full with run_log(run_id).")
        return "\n".join(lines)

    def run_log(self, run_id: str, offset: int = 1, count: int = 0) -> str:
        """A single run rendered to Markdown (via `sah.export`), paged with offset so a long run is
        never a flood. Returns a page plus a `continue with offset=N` note when there is more."""
        _guard_run_id(run_id)
        if run_id not in self.run_ids():
            known = ", ".join(self.run_ids()[-15:]) or "none yet"
            return f"no run {run_id!r} under {self.root}. recent runs: {known}"
        steps, damage = self._steps(run_id)
        text = export.render(steps)
        if damage:                                   # a projection must not look intact when it is not
            text = f"(note: the log is damaged — {damage})\n\n" + text
        lines = text.splitlines()
        start = max(1, int(offset))
        if start > len(lines):
            return f"(offset {start} is past the end; this run has {len(lines)} lines)"
        take = int(count) if int(count) > 0 else None
        window = lines[start - 1:] if take is None else lines[start - 1:start - 1 + take]
        # Clip at the BODY limits, not the result limits: the `continue with offset=` line below is
        # part of the result, and a page that arrives over the limit is re-clipped at the choke point —
        # which drops that line, at the end — and then spilled, copying a log the model was paging.
        shown, meta = ingest.clip_head("\n".join(window), max_lines=ingest.BODY_LINES,
                                       max_bytes=ingest.BODY_BYTES)
        if meta["truncated"]:
            nxt = start + meta["shown_lines"]
            return shown + "\n\n" + ingest.notice(meta, extra=f"continue with run_log(run_id, offset={nxt})")
        end = start - 1 + len(window)
        if end < len(lines):
            return shown + (f"\n\n[end of this page; run has {len(lines)} lines — "
                            f"continue with run_log(run_id, offset={end + 1})]")
        return shown


def tools_for(store: "Runs") -> dict:
    """The read-only recall tools bound to one runs root."""

    def runs(since: Annotated[str, Param("only runs from this date: 'Nd' (e.g. 7d) or "
                                         "'2026-10-20'); empty = all")] = "",
            grep: Annotated[str, Param("case-insensitive term matched against the run id, the "
                                       "model and what was asked for (every objective in the run, "
                                       "not just the first); empty = no filter")] = "",
            limit: Annotated[int, Param("max runs to list (default 20, newest first)")] = 20) -> str:
        """An index, not a transcript: it is what a run *was* (when, what it was asked, how it ended),
        which is the question asked of a history before the cost of reading one is worth spending. The
        grep matches every objective in a run, not only the first, because the run that was asked two
        things and then changed course is found by either of them."""
        return store.index(since=since, grep=grep, limit=limit)

    runs.description = ("List past runs under this run's runs-root, newest first: id, date, model, how "
                        "each ended, how many tool calls, and the objective it was given. Use it to "
                        "answer 'what have I run recently / last week / that failed'. Then read one in "
                        "full with run_log. This reads the raw transcripts of past runs — it is not the "
                        "memory tree, which holds curated notes.")

    def run_log(run_id: Annotated[str, Param("the run folder name to read (one name, not a path)")],
                offset: Annotated[int, Param("1-based line to start from, to page a long run")] = 1,
                count: Annotated[int, Param("max lines this page; 0 means up to the size cap")] = 0) -> str:
        """The transcript is the report: refusals and asks are in it, so a run that ended in a refusal
        is readable as a refusal rather than as a failure of unknown cause. Paging follows the same
        contract as `read` and `runs` — the capped page names the offset that continues — so one way
        of reading covers the log, the files and the history."""
        return store.run_log(run_id, offset=offset, count=count)

    run_log.description = ("Read one past run's full transcript as Markdown — its objective, every "
                           "tool call and result, refusals and asks, and how it ended. Paged: a long "
                           "run says where it stopped and the offset to continue from, so reading a "
                           "big history never overflows the window.")

    # Declared `self`: a run id is one folder name under the runs root, never a path, so there is no
    # place for the file grants to judge. The id read is recorded, which is the half a log needs.
    runs.access, runs.address = "self", None
    run_log.access, run_log.address = "self", "run_id"

    return {"runs": runs, "run_log": run_log}


class _RecallExtension:
    """The recall tools bound to one runs root, wrapped as an extension (duck-typed: `assemble` reads
    `.tools`). Offered only when the run was given a `--runs-root`, because a run not pointed at a
    history has none to consult."""

    def __init__(self, funcs):
        self.tools = tuple(funcs)


def extension(runs: "Runs"):
    """The recall capability for a given runs root, as an extension for `sah.ext.assemble`."""
    return _RecallExtension(tools_for(runs).values())
