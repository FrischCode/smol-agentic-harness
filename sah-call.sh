#!/usr/bin/env bash
# ============================================================================
#  sah-call.sh -- the single-call front door, POSIX sibling of sah-call.bat:
#  fire one objective, get the answer as clean text, exit.
#
#      ~/mydir$  sah-call "summarize what is in this folder"
#      ~/mydir$  sah-call "list the .py files" | grep model
#
#  Identical defaults to sah.sh -- new session, cwd read+write+exec, reasoning on,
#  full loadout -- with two differences, both from SAH_ONCE: it does NOT stay open
#  after the task, and it shows text (the answer) instead of boxes.
#
#  Every default lives in sah.sh. This is a wrapper on purpose: the two verbs share
#  one body, so their defaults can never drift apart.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAH_ONCE=1
exec bash "$ROOT/sah.sh" "$@"
