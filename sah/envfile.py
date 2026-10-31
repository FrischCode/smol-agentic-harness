"""sah.envfile — the optional machine-config file a run reads before it starts.

SAH's configuration is the process environment: SAH_HOST, SAH_PORT, SAH_MODEL, SAH_BASE_URL,
SAH_API_KEY, SAH_SEARCH_URL, SEARXNG_BASE_URL, and the rest. A machine-specific set of those —
which port the backend answers on, which SearXNG serves the search — is useful in one place but
is not the tree to ship, so it lives in a file the .gitignore excludes while a committed example
(.env.example) shows the shape.

The loader is tiny and stdlib-only on purpose. It reads KEY=VALUE lines: a blank line or a
whole-line # comment is skipped, an optional `export ` prefix is dropped, the line's first `=`
splits the key from the value, and one pair of matching single or double quotes around the value
is removed. It does not interpolate $, does not execute anything, and does not strip an inline
` # comment`, so a value is everything after the first `=` — which keeps a URL with a fragment
intact when it is quoted.

Precedence is the standard dotenv one. The file is merged with setdefault: a variable the
environment already holds (the operator's shell, a CI, a flag's default) wins over the file. The
file is a floor that fills the gaps the tree does not carry, never a ceiling over a value the
operator set on purpose.
"""
from __future__ import annotations

import os


def parse_env_file(text: str) -> dict[str, str]:
    """Parse the body of a .env file into {key: value}.

    A blank line or a whole-line `#` comment is skipped. Otherwise the line's first `=` splits the
    key from the value; the key is stripped and must be a bare name (no internal whitespace) or the
    line is dropped. The value is stripped of surrounding whitespace and of one pair of matching
    single or double quotes.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or " " in key or "\t" in key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def load_env_file(path: str) -> dict[str, str]:
    """Parse one .env file on disk, or {} when the path is not a readable file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return parse_env_file(handle.read())
    except OSError:
        return {}


def repo_root() -> str:
    """The directory that holds the `sah` package — the tree the .env sits in."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_env_file() -> str | None:
    """Merge the .env into the process environment for the keys it does not already set, and return
    the path it read, or None when there was no file to read.

    The path is $SAH_ENV_FILE when set (absolute, or relative to the caller's working directory),
    else <repo_root>/.env. Only keys the environment lacks are set, so a value the operator put in
    the shell is never overridden by the file.
    """
    path = os.environ.get("SAH_ENV_FILE") or os.path.join(repo_root(), ".env")
    values = load_env_file(path)
    if not values:
        return None
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return path
