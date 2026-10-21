"""End-to-end: synthetic runs through both commands, without real data.

The runs are short (1 s dwells) and use few bootstrap resamples, so the test
takes about a minute. The MLP (D2) is checked separately in test_one_run.
"""
import json

import numpy as np
import pytest

import diagnostics_report
import results_report
from conftest import write_config
from forcecal.constants import MODEL_ORDER
from forcecal.data import audit_run, load_run, reduce_dwells
from forcecal.evaluation import channel_metrics_from_record, run_model_comparison
from forcecal.models import Geometry
from synthetic import synthetic_run

N_RUNS = 3


def make_run(path, seed, settings):
    geo = Geometry.from_config(settings.object_name, settings.object_geometry)
    synthetic_run(str(path), seed=seed, mounting_offset=0.01 * seed,
                  sensor_hz=50.0, log_hz=100.0, dwell_s=1.0,
                  r_nominal=(-geo.sensor_to_tip_length, 0.0, 0.0),
                  L_free=geo.bending_length, EI_true=geo.section()["EI_Nm2"])


@pytest.fixture(scope="module")
def campaign(tmp_path_factory, settings):
    root = tmp_path_factory.mktemp("campaign")
    runs = [f"test_5{i}" for i in range(N_RUNS)]
    for i, name in enumerate(runs):
        (root / name).mkdir()
        make_run(root / name / "ground_truth.csv", i, settings)
    return root, runs


def test_one_run(campaign, settings):
    root, runs = campaign
    df, meta = load_run(str(root / runs[0] / "ground_truth.csv"))
    audit = audit_run(df, meta)
    assert (audit.loc[audit["check"] == "schema_version", "verdict"] == "OK").all()
    dw = reduce_dwells(df, meta, settle_frac=settings.dwell.settle_frac)
    assert len(dw) == df["pattern_index"].nunique()

    geo = Geometry.from_config(settings.object_name, settings.object_geometry)
    res = run_model_comparison(dw, geo, settings.models,
                               val_frac=settings.evaluation.val_frac, seed=settings.evaluation.seed,
                               nominal_amplitude_N=settings.reference_values.nominal_amplitude_N)
    assert sorted(res["models"]) == sorted(MODEL_ORDER)

    cm = channel_metrics_from_record({"run_id": runs[0], "model_res": res})
    pooled = cm[(cm.force_channel == "Pooled") & (cm.evaluation == "accuracy")]
    # The synthetic data has small offsets and noise only, so the affine model must fit well.
    pa0 = float(pooled.loc[pooled.model == "PA0", "rmse_N"].iloc[0])
    assert pa0 < 0.1
    assert np.isfinite(pooled["rmse_N"]).all()


def quick_config(tmp_path, root, runs):
    return write_config(tmp_path / "configs", **{
        "data.root": str(root), "data.runs": runs,
        "models.use_mlp": False, "evaluation.n_boot": 200, "evaluation.n_boot_transfer": 200,
        "figures.dpi": 60})


def test_results_command(campaign, tmp_path):
    root, runs = campaign
    out = tmp_path / "results"
    assert results_report.main(["--config", str(quick_config(tmp_path, root, runs)),
                                "--out", str(out)]) == 0
    assert (out / "results_skeleton.md").is_file()
    assert (out / "tables" / "model_summary.csv").is_file()
    assert any((out / "figures").glob("*.png"))
    info = json.loads((out / "run_info.json").read_text())
    assert info["settings"]["analysis.yaml"]["models"]["use_mlp"] is False
    assert len(info["input_files"]) == N_RUNS


def test_diagnostics_command(campaign, tmp_path):
    root, runs = campaign
    out = tmp_path / "diagnostics"
    assert diagnostics_report.main(["--config", str(quick_config(tmp_path, root, runs)),
                                    "--out", str(out)]) == 0
    assert (out / "diagnostics_report.md").is_file()
    assert (out / "tables" / "force_transmission_matrix_summary.csv").is_file()
