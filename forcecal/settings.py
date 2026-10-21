"""Read and check the two settings files.

``configs/analysis.yaml`` holds the analysis settings; ``configs/objects.yaml``
(in the same folder) holds the geometry and material of the test objects.
Both commands call :func:`load_settings` once and pass the parts they need to
the analysis functions.

Every key must be present and spelled correctly. An unknown or missing key
stops the program with an error message, so a typo cannot fall back to a
hidden default value. Relative paths in ``analysis.yaml`` are resolved against
the repository folder (the parent of ``configs/``).
"""
from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


class SettingsError(ValueError):
    """Raised when a settings file is missing a key, has an unknown key or an invalid value."""


# ---------------------------------------------------------------------------
# Structure of analysis.yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSettings:
    root: Path                      # folder containing the run folders
    runs: Tuple[str, ...]           # run folder names, e.g. test_50 ... test_57
    object: str                     # key in objects.yaml


@dataclass(frozen=True)
class DwellSettings:
    settle_frac: float              # share of each dwell discarded at the start


@dataclass(frozen=True)
class EvaluationSettings:
    val_frac: float                 # share of training units held out for validation
    seed: int                       # validation split and MLP initialisation
    n_boot: int                     # run-level bootstrap resamples (paired model contrasts)
    n_boot_transfer: int            # bootstrap resamples for the transfer summary


@dataclass(frozen=True)
class MLPSettings:
    hidden_layers: Tuple[int, ...]
    max_iter: int


@dataclass(frozen=True)
class CompliantLeverSettings:
    max_iter: int
    tol_N: float
    relaxation: float


@dataclass(frozen=True)
class ModelSettings:
    use_mlp: bool
    ridge_alpha: float
    mlp: MLPSettings
    compliant_lever: CompliantLeverSettings


@dataclass(frozen=True)
class Thresholds:
    accuracy_N: float               # Test A RMSE requirement
    robustness_factor: float        # Test B RMSE / Test A RMSE requirement
    moment_overflow_Nmm: float      # firmware limit, data-integrity check


@dataclass(frozen=True)
class NominalAmplitudes:
    accuracy: float                 # maximum training load, Test A [N]
    robustness: float               # Test B load [N]


@dataclass(frozen=True)
class ReferenceValues:
    datasheet_accuracy_N: float
    known_fy_gain: float
    nominal_amplitude_N: NominalAmplitudes


@dataclass(frozen=True)
class FigureSettings:
    dpi: int


@dataclass(frozen=True)
class AnalysisFile:
    data: DataSettings
    dwell: DwellSettings
    evaluation: EvaluationSettings
    models: ModelSettings
    thresholds: Thresholds
    reference_values: ReferenceValues
    figures: FigureSettings


# ---------------------------------------------------------------------------
# Structure of one entry in objects.yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectGeometry:
    sensor_to_tip_length_mm: float  # lever arm L in M = r x F
    bending_length_mm: float        # unsupported length, clamp to tip
    exposed_length_mm: float
    shaft_outer_diameter_mm: float
    shaft_wall_mm: float            # equal to the radius for a solid shaft
    youngs_modulus_GPa: float


# ---------------------------------------------------------------------------
# Combined settings passed to the analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    data: DataSettings
    dwell: DwellSettings
    evaluation: EvaluationSettings
    models: ModelSettings
    thresholds: Thresholds
    reference_values: ReferenceValues
    figures: FigureSettings
    object_name: str
    object_geometry: ObjectGeometry
    analysis_file: Path
    objects_file: Path
    raw: Dict[str, Any]             # file contents as read, written to run_info.json


def load_settings(analysis_file: str | Path) -> Settings:
    """Read ``analysis.yaml`` and the ``objects.yaml`` next to it, check them, return Settings."""
    analysis_file = Path(analysis_file).resolve()
    objects_file = analysis_file.parent / "objects.yaml"
    raw_analysis = _read_yaml(analysis_file)
    raw_objects = _read_yaml(objects_file)

    a = _build(AnalysisFile, raw_analysis, analysis_file.name)
    repo = analysis_file.parent.parent
    root = a.data.root if a.data.root.is_absolute() else (repo / a.data.root)
    a = dataclasses.replace(a, data=dataclasses.replace(a.data, root=root))

    if a.data.object not in raw_objects:
        raise SettingsError(f"{analysis_file.name}: data.object '{a.data.object}' is not defined in "
                            f"{objects_file.name} (defined: {', '.join(raw_objects)})")
    geometry = _build(ObjectGeometry, raw_objects[a.data.object],
                      f"{objects_file.name}: {a.data.object}")

    settings = Settings(
        data=a.data, dwell=a.dwell, evaluation=a.evaluation, models=a.models,
        thresholds=a.thresholds, reference_values=a.reference_values, figures=a.figures,
        object_name=a.data.object, object_geometry=geometry,
        analysis_file=analysis_file, objects_file=objects_file,
        raw={"analysis.yaml": copy.deepcopy(raw_analysis),
             "objects.yaml": copy.deepcopy(raw_objects)},
    )
    _check_ranges(settings)
    return settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SettingsError(f"settings file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SettingsError(f"{path.name}: expected a mapping at the top level")
    return data


def _build(cls, data: Any, where: str):
    """Create dataclass ``cls`` from a dict; reject unknown and missing keys."""
    if not isinstance(data, dict):
        raise SettingsError(f"{where}: expected a section with keys, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise SettingsError(f"{where}: unknown key(s) {unknown}; allowed keys are {names}")
    if missing:
        raise SettingsError(f"{where}: missing key(s) {missing}")
    values = {n: _convert(hints[n], data[n], f"{where}.{n}") for n in names}
    return cls(**values)


def _convert(tp, value, where: str):
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    origin = typing.get_origin(tp)
    if origin is tuple:
        item_tp = typing.get_args(tp)[0]
        if not isinstance(value, (list, tuple)) or not value:
            raise SettingsError(f"{where}: expected a non-empty list, got {value!r}")
        return tuple(_convert(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))
    if tp is Path:
        if not isinstance(value, str):
            raise SettingsError(f"{where}: expected a path, got {value!r}")
        return Path(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise SettingsError(f"{where}: expected true or false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise SettingsError(f"{where}: expected text, got {value!r}")
        return value
    raise SettingsError(f"{where}: unsupported setting type {tp}")


def _check_ranges(s: Settings) -> None:
    checks = [
        (0.0 <= s.dwell.settle_frac < 1.0, "dwell.settle_frac must be in [0, 1)"),
        (0.0 <= s.evaluation.val_frac < 1.0, "evaluation.val_frac must be in [0, 1)"),
        (s.evaluation.n_boot >= 1, "evaluation.n_boot must be at least 1"),
        (s.evaluation.n_boot_transfer >= 1, "evaluation.n_boot_transfer must be at least 1"),
        (s.models.ridge_alpha >= 0.0, "models.ridge_alpha must be >= 0"),
        (all(n >= 1 for n in s.models.mlp.hidden_layers), "models.mlp.hidden_layers must be positive"),
        (s.models.mlp.max_iter >= 1, "models.mlp.max_iter must be at least 1"),
        (s.models.compliant_lever.max_iter >= 1, "models.compliant_lever.max_iter must be at least 1"),
        (s.models.compliant_lever.tol_N > 0.0, "models.compliant_lever.tol_N must be > 0"),
        (0.0 < s.models.compliant_lever.relaxation <= 1.0,
         "models.compliant_lever.relaxation must be in (0, 1]"),
        (s.thresholds.accuracy_N > 0.0, "thresholds.accuracy_N must be > 0"),
        (s.thresholds.robustness_factor > 0.0, "thresholds.robustness_factor must be > 0"),
        (s.thresholds.moment_overflow_Nmm > 0.0, "thresholds.moment_overflow_Nmm must be > 0"),
        (s.reference_values.nominal_amplitude_N.accuracy > 0.0
         and s.reference_values.nominal_amplitude_N.robustness > 0.0,
         "reference_values.nominal_amplitude_N values must be > 0"),
        (s.figures.dpi > 0, "figures.dpi must be > 0"),
    ]
    g = s.object_geometry
    checks += [
        (min(g.sensor_to_tip_length_mm, g.bending_length_mm, g.exposed_length_mm,
             g.shaft_outer_diameter_mm, g.shaft_wall_mm, g.youngs_modulus_GPa) > 0.0,
         f"objects.yaml: {s.object_name}: all values must be > 0"),
        (g.sensor_to_tip_length_mm >= g.bending_length_mm,
         f"objects.yaml: {s.object_name}: sensor_to_tip_length_mm must be >= bending_length_mm"),
        (g.shaft_wall_mm <= g.shaft_outer_diameter_mm / 2.0,
         f"objects.yaml: {s.object_name}: shaft_wall_mm cannot exceed the shaft radius"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise SettingsError("invalid settings:\n  " + "\n  ".join(errors))
