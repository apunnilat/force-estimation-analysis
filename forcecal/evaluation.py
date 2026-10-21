"""Train/test splits, model fitting and scoring, and the dead-weight check.

The sampling unit is the dwell, not the CSV row: every metric is computed over
dwell means. Metrics are reported per split (train, validation, Test A, Test B)
and never pooled across them. Test A (interpolation) and Test B (1.5 x the
training load, extrapolation) are compared to measure robustness.

Errors are reported as RMSE together with the mean residual (bias) and the
residual standard deviation (scatter). A channel 300 mN off with 50 mN scatter
and a centred channel with 300 mN scatter have similar RMSE but different
causes: the first is a systematic calibration error, the second is not
removable by a static mapping.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    FORCE_MEAS_S,
    model_family,
    model_name,
    S_AXIAL_INDEX,
    S_LATERAL_INDICES,
    SENSOR_COLS,
    TARGET_COLS,
    TARGET_COLS_S,
    WEIGHT_SENSOR,
)
from .metrics import direction_error, nanmean_or_nan, r2
from .models import build_models, Geometry, geometry_diagnostics

if TYPE_CHECKING:
    from .settings import ModelSettings, NominalAmplitudes


def _matrix(dwells: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    missing = [c for c in cols if f"{c}_mean" not in dwells.columns]
    if missing:
        raise ValueError(f"dwell table is missing reduced columns for: {missing}")
    return dwells[[f"{c}_mean" for c in cols]].to_numpy(dtype=float)


def make_splits(dwells: pd.DataFrame, target: str, val_frac: float, seed: int
                ) -> Tuple[Dict[str, tuple], pd.DataFrame]:
    """Split the dwell table into train / val / accuracy / robustness.

    The validation carve-out is by UNIT, not by dwell: a whole staircase series
    goes to one side or the other. Dwells within a series share a loading
    history and a mounting, so splitting inside one leaks.
    """
    if target not in TARGET_COLS:
        raise ValueError(f"target must be one of {list(TARGET_COLS)}")

    d = dwells[(dwells["quality"] == "OK") &
               (dwells["split"] != "preload")].copy()

    d["unit"] = np.where(d["series_id"] >= 0,
                         "series_" + d["series_id"].astype(str),
                         "dwell_" + d["pattern_index"].astype(str))

    tr = d[d["split"] == "train"].copy()
    units = np.asarray(tr["unit"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = min(max(1, int(round(val_frac * len(units)))), max(1, len(units) - 1))
    val_units = set(units[:n_val]) if val_frac > 0 else set()

    frames = {
        "train": tr[~tr["unit"].isin(val_units)],
        "val": tr[tr["unit"].isin(val_units)],
        "accuracy_test_A": d[d["split"] == "test_A"],
        "robustness_test_B": d[d["split"] == "test_B"],
        # Repeated reference dwells are retained as a descriptive repeatability
        # condition. They may also be part of the calibration sequence, so this
        # split is NOT interpreted as an independent accuracy validation set.
        "repeatability_reference": d[d["family"] == "reference"],
        "anchor": d[d["split"] == "anchor"],
    }

    dth_cols = ["dth_yaw_rad", "dth_pitch_rad", "dth_roll_rad"]
    have_dth = all(f"{c}_mean" in d.columns for c in dth_cols)

    out = {}
    for k, g in frames.items():
        if not len(g):
            out[k] = (np.empty((0, 6)), np.empty((0, 6)), None)
            continue
        out[k] = (_matrix(g, SENSOR_COLS),
                  _matrix(g, TARGET_COLS_S[target]),
                  _matrix(g, dth_cols) if have_dth else None)

    counts = pd.DataFrame([{"split": k, "n_dwells": len(g),
                            "n_units": g["unit"].nunique() if len(g) else 0}
                           for k, g in frames.items()])
    families = {k: g["family"].to_numpy() for k, g in frames.items() if len(g)}
    counts.attrs["families"] = families
    return out, counts


def run_model_comparison(dwells: pd.DataFrame, geo: Geometry, cfg: "ModelSettings", *,
                         val_frac: float, seed: int,
                         nominal_amplitude_N: "NominalAmplitudes",
                         target: str = "delivered") -> Dict[str, object]:
    """Fit the eight force models on the training dwells and score them on held-out splits.

    ``nominal_amplitude_N`` is stored in the result and later used to normalise
    the RMSE (see channel_metrics_from_record).
    """
    splits, counts = make_splits(dwells, target, val_frac, seed)
    Xtr, ytr, dtr = splits["train"]
    if len(Xtr) < 8:
        raise ValueError(f"only {len(Xtr)} usable training dwells")

    caveats: List[str] = [
        "All model-ranking claims in this evaluator concern force reconstruction. "
        "The HEX-32 measured moment channels are interpreted as Nmm despite the legacy "
        "CSV suffix '_Nm'. The moment channels are predictors for the moment-based models, "
        "but distal moments are not reconstructed or scored as an outcome.",
        "P1/P2 and their affine descendants use a mixed physical reconstruction: "
        "the shaft-axial Fx_EE component is copied directly from the measured EE force "
        "channel because bending moment cannot observe it, while Fx_EE/Fy_EE are reconstructed "
        "from Ty_EE/Tz_EE using the fixed (P1) or compliant 3D (P2) lever. Tx_EE is torsion.",
        "The compliant variants do not fit EI or length. Geometry is fixed by the "
        f"object '{geo.object_name}' in configs/objects.yaml; handbook EI and the declared clamp/sensor "
        "dimensions are used only to impose a simple load-dependent 3D deformed-lever assumption.",
    ]
    caveats.extend(geo.warnings())

    models = build_models(Xtr, ytr, geo, cfg, seed, dtr)
    expected = 8 if cfg.use_mlp else 7
    if len(models) != expected:
        caveats.append(
            f"Expected {expected} models but built {len(models)}; this usually means "
            "scikit-learn is unavailable and the MLP could not be created.")

    scored = {k: v for k, v in splits.items()
              if k in ("train", "val", "accuracy_test_A", "robustness_test_B")}
    table = compare_models(models, scored, TARGET_COLS_S[target])

    # Moment metrics are not part of the present model comparison. Keep the
    # columns for backward-compatible CSV schemas, but mark them explicitly NA.
    for c in ("moment_rmse_Nmm", "moment_max_abs_Nmm", "moment_r2_mean"):
        if c in table.columns:
            table[c] = np.nan

    return {
        "counts": counts,
        "comparison": table,
        "caveats": caveats,
        "split_families": counts.attrs.get("families", {}),
        "geometry_summary": geo.assumption_table(),
        "diagnostics": geometry_diagnostics(geo),
        "models": models,
        "splits": splits,
        "nominal_amplitude_N": {"accuracy": nominal_amplitude_N.accuracy,
                                "robustness": nominal_amplitude_N.robustness},
    }


def compare_models(models: Dict[str, dict],
                   splits: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
                   channel_names: Sequence[str]) -> pd.DataFrame:
    """Score every model on force only.

    The measured moment channels are predictors for P1/P2 and for every PA/D model,
    not distal moment outputs. PA0/PA1/PA2 all contain the same three measured
    sensor-moment channels; only their force representation differs. Moment metric
    columns are retained as NaN only so older
    downstream table code can read the CSV schema without special cases.

    Every model's ``predict`` returns a sensor-frame (S) force, so ``ef`` below
    is an S-frame residual: axial/lateral split on S_AXIAL_INDEX/S_LATERAL_INDICES
    (shaft axis = z_S), not the EE-frame indices used inside the physics functions.
    """
    rows: List[dict] = []
    for name, mdl in models.items():
        for split_name, (X, y, dth) in splits.items():
            if not len(X):
                continue
            p = mdl["predict"](X, dth)
            ef = p[:, :3] - y[:, :3]
            de = direction_error(y[:, :3], p[:, :3])
            rows.append({
                "model": name, "model_name": model_name(name),
                "model_family": model_family(name),
                "split": split_name, "n": len(X),
                "force_rmse_N": float(np.sqrt(np.mean(ef ** 2))),
                "force_vec_rmse_N": float(np.sqrt(np.mean(np.sum(ef ** 2, axis=1)))),
                "axial_rmse_N": float(np.sqrt(np.mean(ef[:, S_AXIAL_INDEX] ** 2))),
                "lateral_rmse_N": float(np.sqrt(np.mean(ef[:, list(S_LATERAL_INDICES)] ** 2))),
                "lateral_vec_rmse_N": float(np.sqrt(np.mean(np.sum(ef[:, list(S_LATERAL_INDICES)] ** 2, axis=1)))),
                "force_max_abs_N": float(np.max(np.abs(ef))),
                "force_bias_N": float(np.mean(ef)),
                "moment_rmse_Nmm": np.nan,
                "moment_max_abs_Nmm": np.nan,
                "force_r2_mean": nanmean_or_nan(
                    [r2(y[:, j], p[:, j]) for j in range(3)]),
                "moment_r2_mean": np.nan,
                "dir_err_median_deg": float(de["median_deg"].iloc[0]) if len(de) else np.nan,
                "dir_err_p95_deg": float(de["p95_deg"].iloc[0]) if len(de) else np.nan,
                "note": mdl.get("note", ""),
            })
    return pd.DataFrame(rows)


def _model_force_channel_metrics(models: Dict[str, dict],
                                 split_tuple: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
                                 evaluation: str, group: str, run_id: str,
                                 nominal_amplitude_N: Dict[str, float]) -> pd.DataFrame:
    """Per-model force metrics for one evaluation condition in one run.

    Normalised RMSE is expressed relative to the nominal maximum load of the
    evaluation condition (``nominal_amplitude_N`` in configs/analysis.yaml:
    accuracy/Test A and robustness/Test B). Evaluations without a nominal range use the
    largest reference force-vector magnitude in that condition instead. Using
    one vector-magnitude scale for Fx/Fy/Fz avoids undefined percentages on
    channels whose commanded component is zero while retaining a direct
    "percent of applied load" interpretation.

    ``standard_deviation_N`` is the sample SD of the residuals within the run.
    Run-to-run variation is kept separate and is shown by the aggregate
    boxplots and their SD annotations.
    """
    X, y, dth = split_tuple
    if X is None or y is None or not len(X):
        return pd.DataFrame()
    y3 = np.asarray(y, dtype=float)[:, :3]
    ref_norm = np.linalg.norm(y3, axis=1)
    if evaluation in nominal_amplitude_N:
        normalisation_N = float(nominal_amplitude_N[evaluation])
    else:  # repeatability: no nominal range, keep the measured maximum
        normalisation_N = float(np.nanmax(ref_norm)) if np.isfinite(ref_norm).any() else np.nan
    if not np.isfinite(normalisation_N) or normalisation_N <= 0:
        normalisation_N = np.nan

    rows: List[dict] = []
    channel_labels = ["Fx", "Fy", "Fz"]
    for model, mdl in models.items():
        try:
            pred = np.asarray(mdl["predict"](X, dth), dtype=float)[:, :3]
        except Exception:
            continue
        e = pred - y3
        for j, ch in enumerate(channel_labels):
            ej = e[:, j]
            rmse = float(np.sqrt(np.mean(ej ** 2)))
            rows.append({
                "group": group, "run_id": run_id, "evaluation": evaluation,
                "model": model, "model_name": model_name(model),
                "model_family": model_family(model),
                "force_channel": ch, "n_dwells": int(len(ej)),
                "normalisation_force_N": normalisation_N,
                "rmse_N": rmse,
                "normalized_rmse_pct": (100.0 * rmse / normalisation_N
                                         if np.isfinite(normalisation_N) else np.nan),
                "standard_deviation_N": (float(np.std(ej, ddof=1))
                                           if len(ej) > 1 else np.nan),
                "r2": r2(y3[:, j], pred[:, j]),
            })

        # Pooled channel: flatten Fx/Fy/Fz so this is the same pooled component
        # RMSE convention used by force_rmse_N in the model-comparison table.
        ep = e.reshape(-1)
        yp = y3.reshape(-1)
        pp = pred.reshape(-1)
        rmse_p = float(np.sqrt(np.mean(ep ** 2)))
        rows.append({
            "group": group, "run_id": run_id, "evaluation": evaluation,
            "model": model, "model_name": model_name(model),
            "model_family": model_family(model),
            "force_channel": "Pooled", "n_dwells": int(len(y3)),
            "normalisation_force_N": normalisation_N,
            "rmse_N": rmse_p,
            "normalized_rmse_pct": (100.0 * rmse_p / normalisation_N
                                     if np.isfinite(normalisation_N) else np.nan),
            "standard_deviation_N": (float(np.std(ep, ddof=1))
                                       if len(ep) > 1 else np.nan),
            "r2": r2(yp, pp),
        })
    return pd.DataFrame(rows)


def channel_metrics_from_record(record: dict) -> pd.DataFrame:
    """Collect accuracy, robustness and repeated-reference metrics for a run."""
    mr = record.get("model_res")
    if mr is None:
        return pd.DataFrame()
    mapping = {
        "accuracy": "accuracy_test_A",
        "robustness": "robustness_test_B",
        "repeatability": "repeatability_reference",
    }
    rows = []
    for evaluation, split_name in mapping.items():
        split_tuple = mr.get("splits", {}).get(split_name)
        if split_tuple is None or not len(split_tuple[0]):
            continue
        q = _model_force_channel_metrics(
            mr["models"], split_tuple, evaluation,
            str(record.get("group", "all_runs")), str(record.get("run_id", "run")),
            mr["nominal_amplitude_N"])
        if len(q):
            rows.append(q)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def weight_validation(df: pd.DataFrame, meta: Dict[str, str], *,
                      settle_frac: float,
                      geo: Optional[Geometry] = None,
                      models: Optional[Dict[str, dict]] = None) -> Dict[str, object]:
    """Compare a model's predicted tip force against a traceable mass.

    Magnitude is the frame-independent check: |F| = m*g holds whatever the tip
    frame orientation, so it is valid even where the Euler convention is
    unverified. Direction is reported separately and is only meaningful once
    that convention is confirmed.

    The two baselines bracket the loaded segment, so their difference is a
    measured bound on drift and handling hysteresis rather than an assumption.
    """
    mass = float(meta.get("mass_kg", np.nan))
    g = float(meta.get("gravity_m_s2", 9.80665))
    expected = mass * g

    chans = [c for c in WEIGHT_SENSOR if c in df.columns]
    chans_s = [c for c in FORCE_MEAS_S if c in df.columns]
    segs: Dict[str, np.ndarray] = {}
    stats = []
    for stage in ("baseline_pre", "loaded", "baseline_post"):
        g_ = df[df["stage"] == stage]
        if not len(g_):
            continue
        t = pd.to_numeric(g_["t_rel_s"], errors="coerce").to_numpy()
        keep = g_[t >= t[0] + settle_frac * (t[-1] - t[0])]
        v = keep[chans].to_numpy(float)
        segs[stage] = v.mean(axis=0)
        rec = {"stage": stage, "n": len(keep),
              **{c: float(v[:, i].mean()) for i, c in enumerate(chans)},
              **{f"{c}_sd": float(v[:, i].std(ddof=1)) for i, c in enumerate(chans)}}
        if chans_s:
            vs = keep[chans_s].to_numpy(float)
            rec.update({c: float(vs[:, i].mean()) for i, c in enumerate(chans_s)})
            rec.update({f"{c}_sd": float(vs[:, i].std(ddof=1)) for i, c in enumerate(chans_s)})
        stats.append(rec)

    seg_df = pd.DataFrame(stats)
    out: Dict[str, object] = {"segments": seg_df, "mass_kg": mass,
                              "expected_force_N": expected}

    if "baseline_pre" not in segs or "loaded" not in segs:
        out["error"] = "missing baseline_pre or loaded segment"
        return out

    baseline = segs["baseline_pre"]
    if "baseline_post" in segs:
        drift = segs["baseline_post"] - segs["baseline_pre"]
        out["baseline_drift"] = dict(zip(chans, drift))
        out["baseline_drift_force_N"] = float(np.linalg.norm(drift[:3]))
        # Average the two baselines: the load sat between them, so this is the
        # best estimate of the unloaded state during the loaded segment.
        baseline = 0.5 * (segs["baseline_pre"] + segs["baseline_post"])

    delta = segs["loaded"] - baseline
    out["delta_sensor"] = dict(zip(chans, delta))

    # What the sensor alone says, before any model.
    raw_mag = float(np.linalg.norm(delta[:3]))
    out["raw"] = {"magnitude_N": raw_mag, "abs_error_N": raw_mag - expected,
                  "absolute_error_N": abs(raw_mag - expected),
                  "rel_error_pct": 100.0 * (raw_mag - expected) / expected
                  if expected else np.nan}

    # Apply every fitted reconstruction model to the independent measurement.
    # The result is pred(loaded) - pred(baseline), so affine intercepts cancel
    # correctly and nonlinear models are evaluated at the actual sensor states.
    # A zero pose increment is supplied for pose-aware models because the
    # weight-validation CSV does not contain the excitation-run dth channels.
    if models is not None and len(delta) >= 6:
        X_loaded = segs["loaded"].reshape(1, 6)
        X_base = baseline.reshape(1, 6)
        dth0 = np.zeros((1, 3), dtype=float)
        mrows = []
        for name, mdl in models.items():
            try:
                pred_loaded = np.asarray(mdl["predict"](X_loaded, dth0), dtype=float)[0]
                pred_base = np.asarray(mdl["predict"](X_base, dth0), dtype=float)[0]
                pred = pred_loaded - pred_base
            except Exception:
                continue
            mag = float(np.linalg.norm(pred[:3]))
            signed = mag - expected
            mrows.append({
                "model": name, "magnitude_N": mag,
                "expected_force_N": expected,
                "signed_error_N": signed,
                "absolute_error_N": abs(signed),
                "rel_error_pct": 100.0 * signed / expected if expected else np.nan,
                "pred_fx_N": float(pred[0]), "pred_fy_N": float(pred[1]),
                "pred_fz_N": float(pred[2]),
            })
        out["model_results"] = pd.DataFrame(mrows)

    notes = []
    if "baseline_drift_force_N" in out:
        d = out["baseline_drift_force_N"]
        notes.append(
            f"Baseline drift between the two unloaded segments: {d:.4f} N "
            f"({100 * d / expected:.1f}% of the applied load). Any agreement "
            f"tighter than this is not resolvable by this measurement.")
    notes.append(
        f"Sensor alone: {raw_mag:.4f} N against {expected:.4f} N expected "
        f"({out['raw']['rel_error_pct']:+.2f}%). This is the traceable check -- "
        f"it does not depend on the haptic device in any way, so it is the only "
        f"number here that can detect a systematic error in the reference the "
        f"model was trained against.")
    out["notes"] = notes
    return out
