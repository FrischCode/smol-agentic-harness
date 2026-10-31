# SAH — Contributor Notes

Developer notes for **SAH - Smol Agentic Harness**. Read before editing.

The product overview is in `README.md`; the reasoning behind every decision is in the relevant source. 

## What this is

An agentic harness: Stdlib-only core, synchronous loop, one append-only log, CLI interface.

## Portability axiom

Everything must be copyable to another machine. The venv (`.venv`, uv) is a copyable directory. 
Core imports stdlib and `sah` only — no runtime dependency may enter `pyproject.toml` `dependencies` 
without the decision and its reason written in `pyproject.toml` first.

Both Windows and POSIX are targeted and must be supported.

## Self-explanatory

The tree carries every reason a reader needs — a comment, a docstring. Nothing bound to outside source.

## Environment & commands

- Python `>=3.11`. Own venv: `./.venv` (uv).
- Run the tool: `.venv/Scripts/python.exe -m sah run …` (there is no console script; the module
  is the entry point — `sah/__main__.py`).
- A run talks to any OpenAI-compatible server (llama.cpp, vLLM, tabbyAPI, Strata…) via `--host/--port`.
- Machine config lives in a gitignored `.env` at the tree root (`.env.example` is the committed
  shape). The run reads it at start and fills any `SAH_*` variable not already in the environment —
  a value the shell or a flag sets wins over the file. The shipped defaults are generic (the backend
  on `127.0.0.1:8080`, a local SearXNG for search); this box's real endpoints are the `.env`'s.
- Lint: ruff lives in the venv (dev-only; `dependencies` stays `[]`). `ruff check .` validates, and
  `ruff check --fix .` applies the safe auto-fixes. There is no routine `ruff format`: the layout is
  the house's deliberate style, so the repo is **lint-enforced, not formatter-enforced** — that is the
  guarantee, and it is not a universal formatting guarantee. The config is `[tool.ruff]` in
  `pyproject.toml` (line-length 110); the ignores (`E741`/`E731` deliberate terseness, `I001` import
  order — pure imports, and its fix re-wraps long lists) are one rule at a time so the exclusion
  cannot conceal unrelated findings; `I` stays selected so other isort errors still fire. Suppression
  belongs in this config (`ignore`, or `per-file-ignores` when it is file-specific), never as an inline
  `# noqa: <code>` in the source — ruff is optional, and a rule code in the source is a reference to an
  external source (the Self-explanatory rule applies). Where an import is a deliberate re-export, prefer
  the self-alias idiom (`from … import X as X`) to any suppression at all.

## Prompts are files, not constants

All model-facing prose lives in `sah/prompts/*.md` (a package; `__init__.py` is the loader) so the
agent's instructions are editable without opening a `.py`. Every byte of those files is sent to the
model, so an owner's own annotations are written as HTML comments and `prompts.strip_comments` removes
them **before** the placeholder parser runs — after would evaluate them, and the parser raises on a
malformed insert, so one mistyped character in a note would break every run. The system prompt is three 
stacked parts, in this order: `preamble.md` (what the agent is, the stop rule) → the live `<situation>` 
(built in `cli.situation` from real state, never from a file) → `agents.md` (standing guidance, appended 
last, below the bounds so it cannot contradict them). `--system` replaces only the preamble. Other
model-facing strings are files too (`repeat_steer.md` = the guard's nudge). Audit/log prose (the
`notice` texts) stays in code — it is a record, not an instruction, and is not model-facing.

A prompt file may carry placeholder inserts (`sah/template.py`): the `datetime` and `sys.*` keys are
filled at render by `prompts.load`. Keep the shipped `agents.md` free of the harness name — the test
`test_a_custom_system_replaces_the_preamble` asserts a custom `--system` drops the generic preamble,
detected by the phrase "Smol Agentic Harness" (the name itself is spelled `S.A.H`, so the literal
substring `SAH` never appears anywhere in the prompt).

## Docstrings & comments

Standalone, local, universal: readable with only the file open and nothing else. No task/session/
date/incident prose ("as discussed", "the bug from yesterday"). A comment that references another
module names it. When a non-obvious decision needs to survive, the *reason* goes in a comment,
not a memory note — the idea must travel with the source (see the parked `provide` seam).

## Editing

Keep this file token-dense: no prose, no filler, every line earns its tokens.

## Static 

This file is not to be modified without operator consent.