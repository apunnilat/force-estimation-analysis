"""Shared test fixtures. Makes the repository folder importable."""
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from forcecal import plots  # noqa: E402
from forcecal.settings import load_settings  # noqa: E402

plots.configure_figures(100)


@pytest.fixture(scope="session")
def settings():
    """The thesis settings from configs/analysis.yaml."""
    return load_settings(REPO / "configs" / "analysis.yaml")


def write_config(folder: Path, **changes) -> Path:
    """Copy configs/ into ``folder`` and apply ``changes`` ({"section.key": value})."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "configs" / "objects.yaml", folder / "objects.yaml")
    data = yaml.safe_load((REPO / "configs" / "analysis.yaml").read_text())
    for dotted, value in changes.items():
        node = data
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    path = folder / "analysis.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
