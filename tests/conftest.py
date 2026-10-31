"""Put the repo root on the path so `import sah` resolves to this tree.

And keep the battery hermetic in the same stroke, in both the ways hermeticity can be broken. A
test that calls `cli.build` makes the core load the machine's `.env`, whose values land in
`os.environ` and would make an assertion about a default read this box's endpoints instead of the
shipped ones: pin `SAH_ENV_FILE` at a path that never exists, so the file is missing, the merge is
a no-op, and a default resolves the same on every machine. A test that wants a file points
`SAH_ENV_FILE` at its own (the monkeypatch restores this pin after it).

The second way is ambient, and it is the one a pin cannot reach: a run started by the harness
reads the machine's `.env` into the process environment before anything in this tree runs, and an
ambient value wins over the file the pin names — so a battery launched from inside a run would
read this box's endpoints and its extensions (a measured case: a build run's battery carried ten
failures in the full-run tests, each one the ambient extension reaching a run the test did not
name). The battery's contract is the same on every machine and in every environment, so the
ambient SAH_* values are cleared before the pin lands, and the pin is set after the clear: the
tests are the only voice on what a value is, and a monkeypatch that sets one restores the cleared
state after it.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for name in [key for key in os.environ if key.startswith("SAH_")]:
    del os.environ[name]
os.environ["SAH_ENV_FILE"] = str(ROOT / "tests" / "_no_env_for_the_battery")


# The imports below come after the path insert on purpose: `import sah` cannot resolve until the
# repo root is on sys.path.
import pytest

from sah import ingest


@pytest.fixture
def fits_a_result():
    """The one property every producer that caps its own output owes: what it returns has to pass the
    choke point untouched.

    A producer that caps to `MAX_*` and then appends a notice arrives over the limit, is re-clipped --
    and head-clipping drops the END, where the notice lives -- and is spilled. Asserting a producer
    "looks reasonable" missed that for as long as it existed; the size pair is the whole contract, so
    it is asserted the same way everywhere rather than re-invented per file.
    """
    def check(text):
        lines = len(text.split("\n"))
        assert lines <= ingest.MAX_LINES, f"{lines} lines is over the {ingest.MAX_LINES} result limit"
        nbytes = len(text.encode("utf-8"))
        assert nbytes <= ingest.MAX_BYTES, f"{nbytes} bytes is over the {ingest.MAX_BYTES} result limit"
        return text
    return check


def png_bytes(w=2, h=2, rgb=(0, 255, 0)) -> bytes:
    """A real PNG, built here so no test depends on a fixture file or on a codec.

    Two pixels, one colour: enough for a sniffer to recognise (the magic bytes and an IHDR chunk are
    real) and small enough that a test can assert on its size, or decode it back and compare."""
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


@pytest.fixture
def picture(tmp_path):
    """A picture on disk, at a path a tool can be pointed at."""
    path = tmp_path / "green.png"
    path.write_bytes(png_bytes())
    return path
