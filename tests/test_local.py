"""The machine-local layer: the data/ directory, the models table, and the prompt overrides.
The guarantees: a table that does not hold the shape is a named refusal (never a partial
load), a field the shape names is checked for its exact type (a bool is not a port), and an
override that is absent leaves the shipped text in force."""
import json
import pathlib

import pytest

from sah import local
from sah.record import Refused


def write(root: pathlib.Path, name: str, text: str) -> None:
    folder = root / "data"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


# --- the profile's closed field set -------------------------------------------


def test_a_minimal_profile_names_only_the_model():
    profile = local.parse_profile("flash", {"model": "flash-next"})
    assert profile == {"model": "flash-next"}      # nothing defaulted into place


def test_a_full_profile_holds_exactly_the_fields_the_file_held():
    value = {"model": "m", "host": "10.0.0.1", "port": 9090, "base_url": "https://g/v1",
             "api_key": "k", "think": "xhigh", "think_levels": ["low", "high"],
             "context": 131072, "vision": True}
    assert local.parse_profile("full", value) == value


def test_think_levels_that_is_not_a_list_is_refused():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", {"model": "m", "think_levels": "high"})
    assert "think_levels" in str(exc.value) and "list" in str(exc.value)


def test_a_think_levels_word_that_is_not_a_string_is_refused_by_value():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", {"model": "m", "think_levels": ["high", 3]})
    # the refusal names the value it held, the way the port's refusal names its port
    assert "3" in str(exc.value)


def test_a_field_the_shape_does_not_name_is_refused_by_name():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", {"model": "m", "contex": 100})
    assert "contex" in str(exc.value)


def test_a_profile_that_declares_levels_holds_them_as_the_table_held_them():
    profile = local.parse_profile("flash", {"model": "m", "think": "high",
                                            "think_levels": ["low", "medium", "high", "xhigh"]})
    assert profile["think_levels"] == ["low", "medium", "high", "xhigh"]


def test_a_profile_that_is_not_an_object_is_refused():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", ["m"])
    assert "object" in str(exc.value)


def test_a_profile_without_a_model_selects_nothing():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", {"host": "10.0.0.1"})
    assert "model" in str(exc.value)


def test_an_empty_model_name_is_refused():
    with pytest.raises(Refused):
        local.parse_profile("flash", {"model": ""})


def test_a_bool_is_not_a_port_even_though_it_is_an_int_in_python():
    with pytest.raises(Refused) as exc:
        local.parse_profile("flash", {"model": "m", "port": True})
    assert "bool" in str(exc.value)


def test_a_port_outside_the_port_range_is_refused():
    for port in (0, 70000):
        with pytest.raises(Refused) as exc:
            local.parse_profile("flash", {"model": "m", "port": port})
        assert str(port) in str(exc.value)


def test_a_window_that_is_not_positive_is_refused():
    with pytest.raises(Refused):
        local.parse_profile("flash", {"model": "m", "context": 0})


def test_an_empty_id_names_nothing():
    with pytest.raises(Refused):
        local.parse_profile("", {"model": "m"})


# --- the table ----------------------------------------------------------------


def test_a_table_that_is_absent_is_none_not_a_refusal(tmp_path):
    assert local.load_models(root=str(tmp_path)) is None


def test_a_table_holds_id_to_profile_and_nothing_else(tmp_path):
    write(tmp_path, "models.json", json.dumps({"flash": {"model": "flash-next"},
                                               "sopro": {"model": "s", "port": 8100}}))
    table = local.load_models(root=str(tmp_path))
    assert table == {"flash": {"model": "flash-next"}, "sopro": {"model": "s", "port": 8100}}


def test_a_table_that_is_not_json_is_a_refusal(tmp_path):
    write(tmp_path, "models.json", "{ not json")
    with pytest.raises(Refused) as exc:
        local.load_models(root=str(tmp_path))
    assert "JSON" in str(exc.value)


def test_a_table_that_is_not_an_object_is_a_refusal(tmp_path):
    write(tmp_path, "models.json", json.dumps([{"model": "m"}]))
    with pytest.raises(Refused) as exc:
        local.load_models(root=str(tmp_path))
    assert "object" in str(exc.value)


def test_a_sloppy_sibling_profile_does_not_half_load_the_table(tmp_path):
    """A boot that half-read the table would select a good profile whose sibling the operator
    mistyped, and the selection would look deliberate: the whole table is checked, or none of it."""
    write(tmp_path, "models.json", json.dumps({"flash": {"model": "m"},
                                               "bad": {"model": "m", "port": "nope"}}))
    with pytest.raises(Refused) as exc:
        local.load_models(root=str(tmp_path))
    # the refusal names the profile and the field's exact type, not the value it held
    assert "bad" in str(exc.value) and "'port'" in str(exc.value) and "int" in str(exc.value)


# --- the prompt overrides -------------------------------------------------------


def test_an_override_that_is_absent_leaves_nothing(tmp_path):
    assert local.prompt_override("agents", root=str(tmp_path)) is None


def test_an_override_is_the_text_of_the_data_file(tmp_path):
    write(tmp_path, "agents.md", "one rule: be brief\n")
    assert local.prompt_override("agents", root=str(tmp_path)) == "one rule: be brief\n"


def test_the_data_dir_is_the_tree_root_data_by_default():
    assert local.data_dir() == str(pathlib.Path(local.ROOT) / "data")
