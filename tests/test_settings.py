"""The settings files load, and invalid files are rejected with a clear message."""
import pytest
import yaml

from conftest import REPO, write_config
from forcecal.models import Geometry
from forcecal.settings import SettingsError, load_settings


def test_thesis_settings_load(settings):
    assert settings.data.runs == tuple(f"test_5{i}" for i in range(8))
    assert settings.object_name == "instrument"
    assert settings.evaluation.seed == 42
    assert settings.models.mlp.hidden_layers == (64, 64)


def test_geometry_matches_thesis_values(settings):
    # Values used by the original evaluator (Geometry.from_object), in m and Pa.
    geo = Geometry.from_config(settings.object_name, settings.object_geometry)
    assert geo.sensor_to_tip_length == 0.310
    assert geo.bending_length == 0.290
    assert geo.exposed_length == 0.330
    assert geo.shaft_od == 0.004
    assert geo.shaft_wall == 0.0003
    assert geo.E == 200e9
    assert not geo.section()["solid"]


def test_screwdriver_is_solid():
    objects = yaml.safe_load((REPO / "configs" / "objects.yaml").read_text())
    from forcecal.settings import ObjectGeometry
    geo = Geometry.from_config("screwdriver", ObjectGeometry(**objects["screwdriver"]))
    assert geo.section()["solid"]
    assert geo.sensor_to_tip_length == 0.050


def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, **{"evaluation.n_bootstrap": 100})
    with pytest.raises(SettingsError, match="unknown key"):
        load_settings(path)


def test_missing_key_is_rejected(tmp_path):
    path = write_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    del data["dwell"]["settle_frac"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SettingsError, match="missing key"):
        load_settings(path)


def test_out_of_range_value_is_rejected(tmp_path):
    path = write_config(tmp_path, **{"dwell.settle_frac": 1.5})
    with pytest.raises(SettingsError, match="settle_frac"):
        load_settings(path)


def test_unknown_object_is_rejected(tmp_path):
    path = write_config(tmp_path, **{"data.object": "drill"})
    with pytest.raises(SettingsError, match="not defined"):
        load_settings(path)
