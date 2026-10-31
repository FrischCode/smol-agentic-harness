#!/usr/bin/env bash
# ============================================================================
#  sah.sh -- the SAH front door, POSIX sibling of sah.bat: the shape you reach
#  for from any folder.
#
#      ~/mydir$  sah "rename these files by content"
#
#  It is ONE machine's opinion of good defaults, assembled into flags the real CLI
#  already understands. No logic of its own -- read it, edit it. Everything here is
#  overridable by your own flag or environment variable.
#
#  Two verbs share these defaults, split by what the output is FOR:
#      sah "task"        run it, then KEEP TALKING -- a session you watch (boxes)
#      sah               an empty interactive session in this folder
#      sah-call "task"   run it, print the answer as text, exit  (see sah-call.sh)
#
#  A plain `sah "task"` gives you (each is the shim's choice, not core's):
#    - a NEW session          (a fresh run dir under the runs root)
#    - a grant on THIS folder (read + write + exec on the current directory)
#    - the server's own reasoning default (SAH_THINK overrides: off | on | a level the backend speaks, e.g. xhigh)
#    - the full loadout       (memory + recall + web + exec + spawn)
#    - interactive            (runs the task, then stays open -- "anything else?")
#
#  Also:
#      sah -r [STEP]      resume the last session; with a step, cut back to that step and steer again
#      sah --help          the real CLI help (the door's flags ride along and change nothing)
#      sah run --task "x"  the CLI's own verbs, still with this file's loadout
#
#  Overrides (env): SAH_THINK SAH_RUNS SAH_PYTHON -- and the machine's host/port/model from the
#  .env at $ROOT/.env (the core reads it; a shell value or a flag wins over it).
#  SAH_RAW=1 drops every flag this file adds (see raw)
#  One-shot is set by sah-call.sh via SAH_ONCE=1 -- the only difference between the
#  two verbs, kept in one place so they cannot drift.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The venv layout differs by platform (POSIX bin/python, Windows Scripts/python.exe),
# and this runs under Git Bash on Windows too, so look for both. SAH_PYTHON wins.
if   [ -n "${SAH_PYTHON:-}" ];          then PY="$SAH_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ];   then PY="$ROOT/.venv/bin/python"
else                                         PY="$ROOT/.venv/Scripts/python.exe"
fi
if [ ! -x "$PY" ]; then
    echo "sah.sh: no venv at $ROOT/.venv (bin/python or Scripts/python.exe)" >&2
    echo "        create it first:  uv venv && uv pip install -e ." >&2
    exit 1
fi

# Policy defaults: an already-set env var wins. The machine's endpoint + model come from the
# .env at $ROOT/.env (the core reads it at start), so this shim carries no machine-specific
# address -- a fresh clone reads its own .env. SAH_THINK and SAH_RUNS are the shim's generic
# defaults; it builds the --think / --memory / --runs-root flags from them.
export SAH_THINK="${SAH_THINK:-on}"
SAH_RUNS="${SAH_RUNS:-$ROOT/runs}"

# Let `python -m sah` import the package no matter which folder we work in.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The current directory is the working folder AND the grant.
CWD="$(pwd)"
mkdir -p "$SAH_RUNS"

# The newest run folder under the runs root, by name. The CLI mints those names (fixed-format UTC, so a
# listing ordered by name is ordered by clock) and refuses to write into a folder that already holds a run,
# which is what lets this door drop its pointer file: `last.txt` went stale the first time a folder was
# renamed, and a stale pointer resumes the wrong conversation.
latest() {
    for d in "$SAH_RUNS"/*/log.jsonl; do
        [ -f "$d" ] && printf '%s\n' "$(dirname "$d")"
    done | sort | tail -1 || true
}

# The shared defaults (every flag both verbs use): cwd read+write+exec, reasoning
# on, full loadout. The view is the ONE line that differs -- a session shows boxes,
# a one-shot call shows the answer as text.
args=(--grant "$CWD" --write "$CWD" --exec
      --think "$SAH_THINK"
      --memory "$SAH_RUNS/.memory"
      --runs-root "$SAH_RUNS"
      --web --vision)
# Vision is a default because the operator asked for it and I kept removing it. The caveat stays beside
# the line rather than in a commit message: the endpoint pinned above answers a picture inside the TOOL
# RESULT with HTTP 400 (measured: the vision check of the opt-in live suite exits 1, and the server
# itself says "image_url requires a user message and a string URL"), while the same picture on a user
# turn goes through clean.
# So an image `read` in a door run against that endpoint fails at the NEXT request until core moves the
# bytes onto a user turn. `--no-vision` declines the default per run; `--vision` is what it overrides.
# --think (server-side reasoning quality) is shared; --reasoning (SHOWING the thought) is the
# session's, not the call's. A one-shot reasons just as hard but prints only the answer as text, so
# `sah-call "x" | grep` gets a clean pipe. The view line is the whole verb difference.
if [ -n "${SAH_ONCE:-}" ]; then view=(--view text); else view=(--view normal --reasoning); fi

# The endpoint and bounds are NOT echoed here -- the CLI's boot box (sah.render.header) draws them, so
# an interactive run's first line is that box. A one-shot (text view) prints no box and, with no echo
# here, its stdout is the answer alone -- a clean pipe.

# Routing. A POSIX shell does NOT re-parse "$*", so a metacharacter in the prompt
# reaches --task as one argument and is never run as a command.
dispatch() {
    if [ -n "${SAH_ONCE:-}" ]; then
        exec "$PY" -m sah run "${args[@]}" "${view[@]}" --task "$*"
    else
        exec "$PY" -m sah run "${args[@]}" "${view[@]}" -i --task "$*"
    fi
}

# A leading dash or a subcommand means the CLI's own verbs, and the CLI's own verbs still get this file's
# loadout: an invocation that names its own endpoint, or adds --no-confine, is a door run, not an escape
# from one. Measured before this existed: `sah --no-confine run --host ... --port ...` fell through to the
# raw CLI and booted with exec off, web off, vision off, think=none and no memory tree.
#
# The CLI is one flat parser whose first positional is `run|send|export` and every entry in `args` is an
# option, so prefixing them cannot swallow a positional or reorder anything. A repeated scalar takes the
# LAST word, which is how `--think none` and `--no-vision` narrow this file's assertions; --grant and
# --write append and cannot be narrowed, and that is the one thing SAH_RAW is genuinely for.
with_args() {
    # No "${view[@]}" here on purpose, so `sah run --task "x"` takes the CLI's own default, which is `off`
    # for a one-shot -- the log is the report. `sah "x"` above passes a view, so an objective run draws the
    # box and a verb run says nothing unless asked. Ask with --view normal; --view off keeps it silent.
    exec "$PY" -m sah "${args[@]}" "$@"
}

# SAH_RAW=1: none of this file's flags. The env block above still applies, because it ran before
# routing. The runs and think values are printed (not described) so the line cannot drift from what
# this file holds. The machine's endpoint + model come from the .env at $ROOT/.env (the core reads
# it), and the log records the model that answered -- so this echo names the .env rather than the
# values it holds. To stderr: `sah-call "x" | grep` is a pipe.
raw() {
    echo "[sah] SAH_RAW: no flags from this file - grant, write, exec, memory, runs-root, web, think, vision" >&2
    echo "[sah]   machine config (host/port/model) from $ROOT/.env; runs=$SAH_RUNS think=$SAH_THINK" >&2
    exec "$PY" -m sah "$@"
}

# The only way out of the loadout. Checked before routing so it means the same thing for every verb.
[ -n "${SAH_RAW:-}" ] && raw "$@"

if [ "$#" -eq 0 ]; then
    if [ -n "${SAH_ONCE:-}" ]; then
        echo "sah-call: a single call needs an objective.  e.g.  sah-call 'what is in this folder'" >&2
        exit 2
    fi
    exec "$PY" -m sah run "${args[@]}" "${view[@]}"
fi
case "$1" in
    -r|--resume)
        RUN="$(latest)"
        [ -n "$RUN" ] || { echo "sah.sh: nothing to resume yet -- no run folder under $SAH_RUNS" >&2; exit 1; }
        shift
        # The one word after -r is the jump-back: the CLI's --resume <step>. A word the CLI cannot
        # read as a step number is refused by the CLI's own named line; more than one word is not
        # a step, and a dropped word is a silent gap, so the door refuses it here rather than let
        # it vanish.
        resume=(--resume)
        if [ "$#" -gt 0 ]; then
            [ "$#" -eq 1 ] || { echo "sah.sh: -r takes at most a step number (the step's digits, as logged)" >&2; exit 2; }
            resume=(--resume "$1")
        fi
        echo "[sah] resuming $RUN"
        # Grants are never resumed -- re-issue this run's cwd grant the same way.
        exec "$PY" -m sah run --run-dir "$RUN" "${args[@]}" "${view[@]}" "${resume[@]}" ;;
    run|send|export|--help|-h)
        with_args "$@" ;;
    -*)
        with_args "$@" ;;                             # the door's flags, then whatever was typed
    *)
        dispatch "$*" ;;                              # a bare prompt: run it, then stay open
esac
