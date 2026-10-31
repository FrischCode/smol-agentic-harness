"""sah.prompts — the model-facing text, kept in editable `.md` files, not Python constants.

The prompt a run sends is data its owner should be able to read and retune without opening a `.py`
file: `preamble.md` (what the agent is and how it ends a turn), `agents.md` (standing guidance,
appended below the live situation), `repeat_steer.md` (what the guard says to a run that repeats).
Each is a file in this folder; this module only reads one and expands its `{{...}}` inserts.

`load` renders through `sah.template`, so a prompt file can carry host facts true at launch (the
`{{datetime}}`, `{{sys.*}}` inserts in `agents.md`). A caller may pass keyword overrides — a frozen
value, or a future per-model context — and unknown inserts are left untouched, so a prompt
file that happens to contain braces of its own is not corrupted.

Every byte of these files is sent to the model, so an owner's annotations are written as HTML comments
and `strip_comments` removes them before the text is rendered. A prompt file is therefore editable with
its reasons beside it, and the model is charged only for what it is being told.
"""
from __future__ import annotations

import pathlib
import re

from sah import template

_DIR = pathlib.Path(__file__).parent

# Two passes because the two shapes want different care: a comment occupying a line of its own goes with
# that line, so no paragraph break appears where the note was; a comment inside a line is just removed.
_WHOLE_LINE_COMMENT = re.compile(r"^[ \t]*<!--.*?-->[ \t]*\n?", re.S | re.M)
_INLINE_COMMENT = re.compile(r"<!--.*?-->", re.S)


def strip_comments(text: str) -> str:
    """Remove HTML comments, so a prompt file can carry notes for its owner without paying for them.

    This runs before the placeholder parser on purpose: a fact written inside a comment is then never
    resolved, so it cannot leak as a value or as a pair of braces. Stripping after the render would also
    hide it, but it would resolve it first, which is the half worth refusing."""
    return _INLINE_COMMENT.sub("", _WHOLE_LINE_COMMENT.sub("", text))


def load(name: str, **context) -> str:
    """Return `sah/prompts/<name>.md` with its comments removed, its inserts rendered, and surrounding
    whitespace stripped.

    `context` overrides named facts for this render only; anything not overridden falls back to the
    live host facts. Surrounding whitespace is stripped so the caller controls the blank lines
    between prompt sections."""
    text = (_DIR / f"{name}.md").read_text(encoding="utf-8")
    return render(strip_comments(text), context or None).strip()


def render(text: str, context: dict | None = None) -> str:
    """Strip and render text that did not come from a shipped file, by the same rule.

    `--system` is the operator's own prose. It is still a prompt, so it still gets the comment removal
    that `load` applies — otherwise the rule is a property of this package's files rather than of prompts,
    and a custom preamble copied from one quietly carries its notes to the model."""
    return template.render(strip_comments(text), context or None)
