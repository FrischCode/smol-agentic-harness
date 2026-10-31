"""sah.envfile — the machine-config loader.

The .env fills the environment's gaps (setdefault), never overrides a value the operator already
set, and is found via $SAH_ENV_FILE or the tree root. The parser is tiny: KEY=VALUE lines,
whole-line # comments, an optional `export ` prefix, and one pair of matching quotes."""
import os

from sah import envfile


def test_parse_reads_key_value_lines():
    got = envfile.parse_env_file("SAH_HOST=127.0.0.1\nSAH_PORT=9090\n")
    assert got == {"SAH_HOST": "127.0.0.1", "SAH_PORT": "9090"}


def test_parse_skips_blanks_and_comments():
    text = "\n# a comment\nSAH_PORT=8080\n   \n# another\nSAH_MODEL=m\n"
    assert envfile.parse_env_file(text) == {"SAH_PORT": "8080", "SAH_MODEL": "m"}


def test_parse_drops_the_export_prefix():
    assert envfile.parse_env_file("export SAH_PORT=9090") == {"SAH_PORT": "9090"}


def test_parse_strips_one_pair_of_matching_quotes():
    assert envfile.parse_env_file('SAH_BASE_URL="http://x/v1"') == {"SAH_BASE_URL": "http://x/v1"}
    assert envfile.parse_env_file("SAH_API_KEY='sk-1'") == {"SAH_API_KEY": "sk-1"}


def test_parse_keeps_value_after_first_equals():
    # no inline-comment stripping: a # stays part of the value, quoted or not
    assert envfile.parse_env_file('SAH_SEARCH_URL="http://x/?a=b#c"') == {"SAH_SEARCH_URL": "http://x/?a=b#c"}
    assert envfile.parse_env_file("SAH_URL=http://x/?a=b#c") == {"SAH_URL": "http://x/?a=b#c"}


def test_parse_drops_lines_without_equals_and_bad_keys():
    text = "not a kv line\nMY KEY=bad\n= novalue\nSAH_OK=1\n"
    assert envfile.parse_env_file(text) == {"SAH_OK": "1"}


def test_load_missing_file_is_empty(tmp_path):
    assert envfile.load_env_file(str(tmp_path / "nope.env")) == {}


def test_load_reads_a_file(tmp_path):
    p = tmp_path / "m.env"
    p.write_text("SAH_PORT=9090\n", encoding="utf-8")
    assert envfile.load_env_file(str(p)) == {"SAH_PORT": "9090"}


def test_apply_fills_gaps_and_returns_path(tmp_path, monkeypatch):
    p = tmp_path / "m.env"
    p.write_text("SAH_PORT=9090\nSAH_MODEL=m\n", encoding="utf-8")
    monkeypatch.setenv("SAH_ENV_FILE", str(p))
    monkeypatch.delenv("SAH_PORT", raising=False)
    monkeypatch.delenv("SAH_MODEL", raising=False)
    read = envfile.apply_env_file()
    assert read == str(p)
    assert os.environ.get("SAH_PORT") == "9090"
    assert os.environ.get("SAH_MODEL") == "m"


def test_apply_never_overrides_an_existing_value(tmp_path, monkeypatch):
    p = tmp_path / "m.env"
    p.write_text("SAH_PORT=9090\n", encoding="utf-8")
    monkeypatch.setenv("SAH_ENV_FILE", str(p))
    monkeypatch.setenv("SAH_PORT", "1234")     # the operator's shell value wins over the file
    envfile.apply_env_file()
    assert os.environ.get("SAH_PORT") == "1234"


def test_apply_no_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("SAH_ENV_FILE", str(tmp_path / "missing.env"))
    assert envfile.apply_env_file() is None
