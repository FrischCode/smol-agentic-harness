"""sah.local — the machine-local layer: the data/ directory at the tree root.

The tree ships its defaults — the prompts in `sah/prompts`, the endpoint read from the
environment — and the data/ directory is where the operator's machine states its own: a
`models.json` naming the endpoints the operator reaches for by id, and prompt overrides, a
`data/<name>.md` that stands in for the shipped `sah/prompts/<name>.md`.

It is the machine-config class of thing, the .env's own: the repo ships the shape
(`data/models.example.json`), not the values, and the directory's contents are gitignored the
way `.env` is. Everything here is a file the operator wrote, so the reading is strict: a
models.json that does not hold the shape is a named refusal, not a partial load — a profile
table the harness half-reads is a table the operator believes it has — and an override that is
absent leaves the shipped text in force, untouched.

The directory sits at the tree root — this module's grandparent — rather than in the working
directory, so it is found from wherever the launcher stands (the front door runs from any
folder), and so it travels with the tree the portability axiom copies: a machine's data/ is
that machine's endpoints, and a copy of the tree without one is a fresh machine's tree, on the
shipped defaults.
"""
from __future__ import annotations

import json
import os

from sah.record import Refused

# The tree root: two levels up from this file's folder (sah/local.py -> sah -> root). The
# editable install keeps the package inside the tree, so the file's own location is the
# tree's location.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A profile's closed field set: name -> the exact type the field holds. Closed like the record's
# step vocabulary — a key the shape does not name is a refusal naming the key, not a silent
# ignore: a profile the operator mistypes (contex, viison) that the harness quietly drops would
# switch the endpoint with a window or a vision declaration the operator never made, and the
# record would say nothing.
PROFILE_FIELDS = {
    "model": str,      # the wire name the request carries — the one required field
    "host": str,
    "port": int,
    "base_url": str,
    "api_key": str,
    "think": str,
    "think_levels": list,  # the level words the profile's server takes: the operator's statement
    "context": int,
    "vision": bool,
}


def data_dir(root: str | None = None) -> str:
    """The data/ directory: the tree root's, or the one under `root` when the caller names one."""
    return os.path.join(root if root is not None else ROOT, "data")


def parse_profile(profile_id: str, value) -> dict:
    """One profile from the models table, checked against the closed field set.

    `model` is the one required field — a profile that names no model selects nothing — and
    every other field the shape names is optional, its absence meaningful: a field the profile
    leaves silent stays as the run has it (the caller's chain decides what that is). The
    returned dict holds exactly the fields the file held, nothing defaulted into place."""
    if not profile_id:
        raise Refused("data/models.json: a profile needs a name — an empty id selects nothing")
    if not isinstance(value, dict):
        raise Refused(f"data/models.json: profile {profile_id!r} must be an object of fields, "
                      f"not a {type(value).__name__}")
    for key in value:
        if key not in PROFILE_FIELDS:
            raise Refused(f"data/models.json: profile {profile_id!r} holds field {key!r}, which "
                          f"the shape does not name ({', '.join(sorted(PROFILE_FIELDS))})")
    if not isinstance(value.get("model"), str) or not value["model"]:
        raise Refused(f"data/models.json: profile {profile_id!r} needs a non-empty model — the "
                      "wire name the request carries")
    out = {}
    for key, expected in PROFILE_FIELDS.items():
        if key not in value:
            continue
        got = value[key]
        # `type is`, not isinstance: a bool is an int in Python, and a port of `true` is a port
        # of one, which is not what the number means.
        if type(got) is not expected:
            raise Refused(f"data/models.json: profile {profile_id!r} field {key!r} must be a "
                          f"{expected.__name__}, not a {type(got).__name__}")
        if key == "port" and not 1 <= got <= 65535:
            raise Refused(f"data/models.json: profile {profile_id!r} port {got} is not a port "
                          "(1-65535)")
        if key == "context" and got <= 0:
            raise Refused(f"data/models.json: profile {profile_id!r} context {got} is not a "
                          "window (a positive token count)")
        if key == "think_levels" and any(type(word) is not str for word in got):
            bad = next(word for word in got if type(word) is not str)
            raise Refused(f"data/models.json: profile {profile_id!r} field {key!r} holds {bad!r}, "
                          f"which is not a level word (a string)")
        out[key] = got
    return out


def load_models(root: str | None = None) -> dict | None:
    """The models table, or None when the file is absent: {profile id: profile}.

    The table is a top-level object — id to profile — read whole or refused: a file that is not
    JSON, is not an object, or holds a profile the shape does not name is a Refused naming the
    fault, because a boot that half-reads the table would select a profile whose sibling the
    operator mistyped, and the selection would look deliberate."""
    path = os.path.join(data_dir(root), "models.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    except (OSError, UnicodeDecodeError) as exc:
        raise Refused(f"data/models.json cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Refused(f"data/models.json is not JSON: {exc}") from exc
    if not isinstance(table, dict):
        raise Refused("data/models.json must hold an object: profile id -> profile")
    return {profile_id: parse_profile(profile_id, value)
            for profile_id, value in table.items()}


def prompt_override(name: str, root: str | None = None) -> str | None:
    """The data/<name>.md text, or None when the file is absent: the operator's override of
    the shipped sah/prompts/<name>.md.

    The text is handed back as read: the override runs through the same strip-and-render the
    shipped text does (the caller's pass, after this hand-back), so an owner's data copy
    inherits the comment rule and the insert rule rather than escaping them. A file that
    cannot be read is a refusal, not a crash: the run's prompt is boot data, and a traceback
    at the boot is not an answer to a file the operator left unreadable."""
    path = os.path.join(data_dir(root), f"{name}.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise Refused(f"data/{name}.md cannot be read: {exc}") from exc
