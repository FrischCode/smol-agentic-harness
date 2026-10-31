"""sah.template — expand `{{placeholder}}` inserts in a text file, so one static file can carry
facts that change every time it is read.

The problem it solves: a document like `sah/prompts/agents.md` wants to state live facts —
"rendered on 2026-10-24, on win32/AMD64, 128 GiB RAM" — without those going stale in the source
and without splitting the doc into a static half and a generated half. One file, `{{datetime}}`
and `{{sys.os}}` written inline, and the dynamic values poured in at render time. The one
consumer in the core is `sah.prompts.load`, which renders the standing guidance before the model
sees it; a document that is injected verbatim (the contributor notes) carries no inserts, because
a reader of braces gets braces.

The parser is deliberately tiny: a placeholder is `{{name}}` or `{{name:format}}`, `name` is a
dotted word (`sys.cpu`, `datetime`), and `format` (when the value is a date) is a `strftime`
pattern. There is no logic, no loops, no conditionals — substitution only. A placeholder is a
*lookup that fills in text*, and anything more powerful is a templating language, which this is not.

Unknown placeholders are left exactly as written, never blanked: this renders prose and code that
may contain braces of its own, and silently eating `{{that}}` would corrupt text it was never asked
to touch. The cost is a typo in a placeholder name passes quietly — it is left as written, not
flagged. The known keys are therefore enumerated in `facts()`, the one place that defines what a
placeholder may name.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import sys

# {{name}} or {{name:format}}. The name is a dotted identifier; the format is everything up to the
# closing braces, so a strftime pattern with its own colons (`%H:%M`) survives intact.
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*(?::\s*([^}]*?)\s*)?\}\}")


def _total_ram_gib() -> int | None:
    """Total physical memory in GiB, or None when the platform gives no stdlib route. Windows:
    GlobalMemoryStatusEx via ctypes; elsewhere: os.sysconf. A failure is absence, never a guess."""
    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return round(m.ullTotalPhys / 2**30)
        except Exception:
            return None
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, KeyError):
        return None


def _resolved_shell() -> str:
    """The name of the shell a command would actually get. Falls back to the platform's usual answer if
    the tool layer cannot be imported, because rendering a document must not fail over a label."""
    try:
        from sah.tools import shell as _shell
        return _shell.SHELL.name
    except Exception:                                   # a label, never a build failure
        return "cmd.exe" if sys.platform == "win32" else "/bin/sh"


def _ram() -> str:
    """Total RAM as a host fact, or an honest `unknown`. A helper so the probe runs once per prompt load."""
    gib = _total_ram_gib()
    return f"{gib} GiB" if gib is not None else "unknown"


def facts() -> dict:
    """The values a placeholder can name, fresh for this instant.

    Every fact is stdlib-only. `datetime` is a real datetime object so `{{datetime:%Y-%m-%d}}` can
    format it; the rest are already strings. The set mirrors the basic host facts a run would
    otherwise spend turns discovering (os, machine, shell, capacity).

    `sys.shell` is the one fact not read from the platform layer. It comes from the shell resolver in
    `sah.tools.shell`, imported here rather than at module scope: "what shell is there" does not mean "what
    does the OS call a shell", it means "which shell would this run's commands actually be handed to",
    and a template that guesses wrong teaches the model syntax that will fail. Imported lazily so that
    rendering a document does not pull in the tool layer.
    """
    plat = platform.system() or sys.platform
    release = " ".join(x for x in (plat, platform.release(), platform.version()) if x).strip()
    return {
        "datetime": _dt.datetime.now().astimezone(),
        "sys.os": sys.platform,
        "sys.system": plat,
        "sys.release": release,
        "sys.machine": platform.machine(),
        "sys.processor": platform.processor() or "unknown",
        "sys.python": platform.python_version(),
        "sys.cpu": str(os.cpu_count() or "unknown"),
        # Asked once, not twice. The condition and the value were separate calls, which on Windows is two
        # ctypes round trips per prompt load for one number, and a fact consulted twice is a fact that can
        # disagree with itself.
        "sys.ram": _ram(),
        "sys.host": platform.node() or "unknown",
        "sys.shell": _resolved_shell(),
        "sys.newlines": ("CRLF" if sys.platform == "win32" else "LF"),
        "sys.text": "UTF-8",
    }


def render(text: str, context: dict | None = None) -> str:
    """Fill every *known* `{{placeholder}}` in `text`. Unknown ones are left untouched.

    `context` defaults to `facts()`; a caller may pass a dict to override or add keys (a caller that
    wants a frozen timestamp, or one that wants a host fact replaced). A `{{datetime:...}}` format runs
    `strftime`; a format on a non-date value is ignored rather than an error, so a copy-paste typo
    degrades instead of raising."""
    values = facts() if context is None else {**facts(), **context}

    def fill(match: re.Match) -> str:
        key, fmt = match.group(1), match.group(2)
        value = values.get(key)
        if value is None:
            return match.group(0)                      # leave an unknown insert exactly as written
        if isinstance(value, _dt.datetime):
            return value.strftime(fmt) if fmt else value.isoformat()
        return str(value)

    return _PLACEHOLDER.sub(fill, text)


def render_file(path: str) -> str:
    """Read a file and return it with its placeholders expanded."""
    with open(path, encoding="utf-8") as handle:
        return render(handle.read())


if __name__ == "__main__":                            # `python -m sah.template FILE` -> rendered stdout
    for arg in sys.argv[1:]:
        sys.stdout.write(render_file(arg))
