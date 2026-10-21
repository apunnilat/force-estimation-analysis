"""Diagnostic analysis of the repeated remountings.

Complements the Results report with six diagnostics, each reported in the
sensor frame S:

1. hysteresis of the bidirectional staircases;
2. force transmission (cross-axis coupling) matrix and force-to-moment coupling;
3. within-session drift of the anchor dwells;
4. model residual versus delivered load;
5. apparent EI from the measured end-effector displacement, compared with the
   handbook EI;
6. delivered-versus-predicted parity for all models.

Interpretation limits
---------------------
* The coupling matrix is a SYSTEM transmission matrix. It combines sensor gain
  and crosstalk, frame rotation, mounting and load path; it is not an intrinsic
  transducer calibration matrix. It is computed in EE and rotated to S.
* The apparent EI uses the Lambda.7 end-effector displacement, which also
  contains adapter and device compliance. It is a consistency check, not a
  traceable material measurement.
* The models are fitted per run with the same settings as the Results report.
  No Test A or Test B dwell is used for fitting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    ee_axis_to_s_label,
    FORCE_MEAS_S,
    rotate_ee_to_s,
    rotate_ee_to_s_matrix,
    FORCE_LABELS,
    FORCE_MEAS,
    FORCE_REF,
    model_name,
    MODEL_ORDER,
    MOMENT_LABELS,
    MOMENT_MEAS,
)
from .data import discover_campaign, load_run, reduce_dwells
from .evaluation import run_model_comparison
from .metrics import iso376_metrics, r2_plain
from .models import Geometry

if TYPE_CHECKING:
    from .settings import Settings


@dataclass
class RunRecord:
    run_id: str
    folder: Path
    ground_truth: Path
    alignment_log: Optional[Path]
    weight_validation: Optional[Path]
    df: pd.DataFrame
    meta: Dict[str, str]
    dwells: pd.DataFrame
    model_result: dict


def load_campaign(settings: "Settings") -> Tuple[List[RunRecord], pd.DataFrame]:
    """Load every run, reduce it to dwells and fit the eight models (same settings as the Results)."""
    discovered = [row for row in discover_campaign(settings.data.root, settings.data.runs) if row["exists"]]
    if not discovered:
        raise FileNotFoundError(
            f"No requested run folders found below {settings.data.root}. "
            f"Expected: {', '.join(settings.data.runs)}.")

    geo = Geometry.from_config(settings.object_name, settings.object_geometry)
    records: List[RunRecord] = []
    manifest_rows = []
    for row in discovered:
        run_id, folder = row["run_id"], Path(row["folder"])
        gt = Path(row["ground_truth"]) if row["ground_truth"] else None
        align = Path(row["alignment"]) if row["alignment"] else None
        weight = Path(row["weight"]) if row["weight"] else None
        if gt is None:
            manifest_rows.append({
                "run_id": run_id, "folder": str(folder), "status": "NO_GROUND_TRUTH",
                "ground_truth": "", "alignment_log": str(align or ""),
                "weight_validation": str(weight or "")})
            continue

        print(f"[{run_id}] loading {gt}")
        df, meta = load_run(str(gt))
        dw = reduce_dwells(df, meta, settle_frac=settings.dwell.settle_frac)
        model_result = run_model_comparison(
            dw, geo, settings.models,
            val_frac=settings.evaluation.val_frac, seed=settings.evaluation.seed,
            nominal_amplitude_N=settings.reference_values.nominal_amplitude_N)
        records.append(RunRecord(run_id, folder, gt, align, weight,
                                 df, meta, dw, model_result))
        manifest_rows.append({
            "run_id": run_id,
            "folder": str(folder),
            "status": "OK",
            "ground_truth": str(gt),
            "alignment_log": str(align or ""),
            "weight_validation": str(weight or ""),
            "pattern_seed": meta.get("pattern_seed", ""),
            "skipped_phases": meta.get("skipped_phases", ""),
            "n_rows": len(df),
            "n_dwells": len(dw),
        })
    if not records:
        raise RuntimeError("Run folders were found, but none contained a usable ground_truth CSV.")
    return records, pd.DataFrame(manifest_rows)


def _mean_cols(dw: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return dw[[f"{c}_mean" for c in cols]].to_numpy(dtype=float)


def _ols_matrix(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Y = X B^T + intercept. Return B [nout,nin], intercept, residuals."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    B = coef[:-1, :].T
    b = coef[-1, :]
    residual = Y - (X @ B.T + b)
    return B, b, residual


def transmission_matrices(rec: RunRecord) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = rec.dwells.copy()
    d = d[(d["quality"] == "OK") & (d["split"] == "train")]
    # Repeated references would overweight one +Fz point; omit them here.
    d = d[~d["family"].isin(["reference", "anchor", "preload"])]
    if len(d) < 12:
        return pd.DataFrame(), pd.DataFrame(), {}

    Fref = _mean_cols(d, FORCE_REF)
    Fmeas = _mean_cols(d, FORCE_MEAS)
    Mmeas = _mean_cols(d, MOMENT_MEAS)  # Nmm despite legacy suffix

    BF, bF, rF = _ols_matrix(Fref, Fmeas)
    BM, bM, rM = _ols_matrix(Fref, Mmeas)
    # Both blocks and intercepts are EE-ordered; convert to sensor frame once,
    # here, so every downstream table/figure (generic, position-indexed via
    # FORCE_LABELS/MOMENT_LABELS) is correct with no further changes.
    BF, bF = rotate_ee_to_s_matrix(BF), rotate_ee_to_s(bF)
    BM, bM = rotate_ee_to_s_matrix(BM), rotate_ee_to_s(bM)

    force_rows = []
    for i, out_ch in enumerate(FORCE_LABELS):
        for j, in_ch in enumerate(FORCE_LABELS):
            force_rows.append({
                "run_id": rec.run_id, "output_channel": out_ch,
                "input_force": in_ch, "coefficient": BF[i, j],
                "unit": "N/N"})
        force_rows.append({
            "run_id": rec.run_id, "output_channel": out_ch,
            "input_force": "intercept", "coefficient": bF[i], "unit": "N"})

    moment_rows = []
    for i, out_ch in enumerate(MOMENT_LABELS):
        for j, in_ch in enumerate(FORCE_LABELS):
            moment_rows.append({
                "run_id": rec.run_id, "output_channel": out_ch,
                "input_force": in_ch, "coefficient": BM[i, j],
                "unit": "Nmm/N"})
        moment_rows.append({
            "run_id": rec.run_id, "output_channel": out_ch,
            "input_force": "intercept", "coefficient": bM[i], "unit": "Nmm"})

    # Polar decomposition of the 3x3 force block.  This is a descriptive
    # decomposition of the forward system matrix, not an intrinsic sensor-only
    # rotation/gain calibration.
    U, s, Vt = np.linalg.svd(BF)
    R = U @ Vt
    detR = float(np.linalg.det(R))
    # Angle is meaningful for a proper rotation. If the SVD yields a reflection,
    # report it and use the nearest proper rotation only as a diagnostic.
    Rproper = R.copy()
    if detR < 0:
        U2 = U.copy()
        U2[:, -1] *= -1.0
        Rproper = U2 @ Vt
    cosang = float(np.clip((np.trace(Rproper) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    H = Rproper.T @ BF
    polar = {
        "run_id": rec.run_id,
        "rotation_angle_deg": angle_deg,
        "det_force_matrix": float(np.linalg.det(BF)),
        "det_rotation_raw": detR,
        "singular_1": float(s[0]),
        "singular_2": float(s[1]),
        "singular_3": float(s[2]),
        "gain_like_x": float(H[0, 0]),
        "gain_like_y": float(H[1, 1]),
        "gain_like_z": float(H[2, 2]),
        "force_matrix_residual_pooled_rmse_N": float(np.sqrt(np.mean(rF ** 2))),
        "moment_matrix_residual_pooled_rmse_Nmm": float(np.sqrt(np.mean(rM ** 2))),
        "n_dwells": int(len(d)),
    }
    return pd.DataFrame(force_rows), pd.DataFrame(moment_rows), polar


def hysteresis_table(rec: RunRecord) -> pd.DataFrame:
    # Hysteresis is computed on the sensor-frame (S) dwell means, so the
    # channel labels are already "Fx_meas_S_N", "Fy_meas_S_N", "Fz_meas_S_N".
    iso = iso376_metrics(rec.dwells, rec.meta, channels=FORCE_MEAS_S)
    h = iso[iso["metric"] == "hysteresis_v"].copy()
    if not len(h):
        return h
    h["run_id"] = rec.run_id
    h["abs_value_N"] = h["value"].abs()
    # pattern_label carries the EE-convention commanded axis (delivered/test
    # metadata, e.g. "Fx@..."); convert it to the same sensor-frame letter
    # used for output_channel below, so the two stay comparable.
    ee_to_s_axis = {lab: ee_axis_to_s_label(lab) for lab in FORCE_LABELS}
    series_axis = {}
    for sid, g in rec.dwells[rec.dwells["family"] == "stair_bidir"].groupby("series_id"):
        lab = str(g["pattern_label"].iloc[0])
        m = re.match(r"(Fx|Fy|Fz)", lab)
        series_axis[f"series_{int(sid)}"] = ee_to_s_axis[m.group(1)] if m else f"series_{int(sid)}"
    h["applied_axis"] = h["group"].map(series_axis).fillna(h["group"])
    # output_channel: iso's "channel" is already sensor-labeled full column
    # names ("Fx_meas_S_N"); take the short "Fx"/"Fy"/"Fz" prefix directly.
    h["output_channel"] = h["channel"].astype(str).str.split("_").str[0]
    return h


def anchor_table(rec: RunRecord) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = rec.dwells[(rec.dwells["family"] == "anchor") &
                   (rec.dwells["quality"] == "OK")].sort_values("t_mid_s").copy()
    if not len(a):
        return pd.DataFrame(), pd.DataFrame()
    t0 = float(a["t_mid_s"].iloc[0])
    rows = []
    summary = []
    # Sensor-frame channel k is read from the S-frame dwell mean FORCE_MEAS_S[k]
    # and labeled FORCE_LABELS[k].
    for k in range(3):
        c = FORCE_MEAS_S[k]
        vals = a[f"{c}_mean"].to_numpy(dtype=float)
        times = a["t_mid_s"].to_numpy(dtype=float)
        delta = vals - vals[0]
        elapsed_min = (times - t0) / 60.0
        for i in range(len(a)):
            rows.append({
                "run_id": rec.run_id, "anchor_index": i + 1,
                "channel": FORCE_LABELS[k], "elapsed_min": elapsed_min[i],
                "measured_N": vals[i], "delta_from_first_N": delta[i],
                "pattern_label": a["pattern_label"].iloc[i],
            })
        slope_h = np.nan
        if len(a) >= 2 and np.ptp(times) > 0:
            slope_h = np.polyfit((times - t0) / 3600.0, vals, 1)[0]
        summary.append({
            "run_id": rec.run_id, "channel": FORCE_LABELS[k],
            "anchor_slope_N_per_h": slope_h,
            "anchor_span_N": float(np.ptp(vals)),
            "first_to_last_N": float(vals[-1] - vals[0]),
            "n_anchors": len(a),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def _pose_dwell_means(rec: RunRecord, settle_frac: float) -> pd.DataFrame:
    """Reduce p_relative EE pose channels to dwell means from the raw log.

    The main evaluator intentionally does not need these channels for model
    fitting, so they are reduced here independently.  Position is a logger-side
    device state rather than a sensor message; therefore rows are not de-duplicated
    on sensor_msg_count for this pose-only reduction.
    """
    cols = ["p_relative_x_EE_m", "p_relative_y_EE_m", "p_relative_z_EE_m"]
    if "ramp_state" not in rec.df.columns or not all(c in rec.df.columns for c in cols):
        return pd.DataFrame()
    d = rec.df[rec.df["ramp_state"] == 1].copy()
    time_col = "t_rel_s" if "t_rel_s" in d.columns else "time_s"
    rows = []
    for idx, g in d.groupby("pattern_index", sort=True):
        g = g.sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(dtype=float)
        if len(t) == 0 or not np.isfinite(t).any():
            continue
        t0, t1 = np.nanmin(t), np.nanmax(t)
        cut = t0 + settle_frac * (t1 - t0)
        w = g[pd.to_numeric(g[time_col], errors="coerce") >= cut]
        row = {"pattern_index": int(idx)}
        for c in cols:
            v = pd.to_numeric(w[c], errors="coerce").to_numpy(dtype=float)
            row[f"{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def apparent_ei(rec: RunRecord, geo: Geometry, settle_frac: float) -> pd.DataFrame:
    """Estimate apparent EI from transverse displacement vs delivered load,
    reported in the sensor frame.

    Internally the shaft axis is x_EE (tip on the -x_EE side; see module
    docstring) and the two bending directions are y_EE and z_EE -- unchanged.
    In sensor-frame report terms, x_EE is "Fz" (the excluded/shaft axis) and
    the two bending directions are "Fy" (= y_EE, unchanged) and "Fx" (= z_EE).
    Use isolated Fx/Fy staircase dwells and fit

        p_relative_axis = c * F_delivered_axis + b.

    For a free cantilever, apparent EI = L_b^3 / (3 |c|); since it only uses
    |slope|, no sign correction is needed for this particular quantity.
    """
    d = rec.dwells[(rec.dwells["quality"] == "OK") &
                   (rec.dwells["family"].isin(["stair_bidir", "stair_asc"]))].copy()
    pose = _pose_dwell_means(rec, settle_frac)
    if len(pose):
        d = d.merge(pose, on="pattern_index", how="left")
    needed = ["p_relative_y_EE_m_mean", "p_relative_z_EE_m_mean"]
    if not all(c in d.columns for c in needed):
        return pd.DataFrame()

    handbook = float(geo.section()["EI_Nm2"])
    L = float(geo.bending_length)
    rows = []
    # Sensor-frame labels: "Fy" is unchanged (y_EE=y_S); the other bending
    # axis is z_EE, which is sensor-frame "Fx" (not "Fz" -- that's now the
    # shaft/excluded axis). The underlying command/reference/pose columns
    # for that axis are still the original z_EE ones, just relabeled here.
    configs = [
        ("Fy", "fy_cmd_N_mean", f"{FORCE_REF[1]}_mean", "p_relative_y_EE_m_mean"),
        ("Fx", "fz_cmd_N_mean", f"{FORCE_REF[2]}_mean", "p_relative_z_EE_m_mean"),
    ]
    for axis, cmd_col, force_col, disp_col in configs:
        if cmd_col not in d.columns or force_col not in d.columns:
            continue
        if axis == "Fy":
            sel = d[(d["fx_cmd_N_mean"].abs() < 1e-6) &
                    (d["fz_cmd_N_mean"].abs() < 1e-6) &
                    (d["fy_cmd_N_mean"].abs() > 0.1)]
        else:
            sel = d[(d["fx_cmd_N_mean"].abs() < 1e-6) &
                    (d["fy_cmd_N_mean"].abs() < 1e-6) &
                    (d["fz_cmd_N_mean"].abs() > 0.1)]
        x = pd.to_numeric(sel[force_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sel[disp_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 5 or np.ptp(x) < 1e-9:
            continue
        A = np.column_stack([x, np.ones(len(x))])
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = slope * x + intercept
        compliance = abs(float(slope))
        EI = np.nan if compliance <= 1e-12 else L ** 3 / (3.0 * compliance)
        rows.append({
            "run_id": rec.run_id,
            "axis": axis,
            "n_dwells": len(x),
            "compliance_m_per_N": float(slope),
            "intercept_m": float(intercept),
            "fit_r2": r2_plain(y, pred),
            "EI_apparent_Nm2": EI,
            "EI_handbook_Nm2": handbook,
            "EI_ratio_apparent_to_handbook": EI / handbook if np.isfinite(EI) else np.nan,
            "bending_length_m": L,
            "note": "apparent EI from transverse displacement; includes all compliance in measured load path",
        })
    return pd.DataFrame(rows)


def _split_frame(rec: RunRecord, split_name: str) -> pd.DataFrame:
    split_value = {"accuracy_test_A": "test_A", "robustness_test_B": "test_B"}[split_name]
    d = rec.dwells[(rec.dwells["quality"] == "OK") &
                   (rec.dwells["split"] == split_value)].copy()
    return d.reset_index(drop=True)


def prediction_rows(rec: RunRecord) -> pd.DataFrame:
    rows = []
    models = rec.model_result["models"]
    splits = rec.model_result["splits"]
    for split_name in ("accuracy_test_A", "robustness_test_B"):
        X, y, dth = splits[split_name]
        frame = _split_frame(rec, split_name)
        if not len(X):
            continue
        # split construction preserves dwell order for Test A/B.
        n = min(len(X), len(frame))
        X, y = X[:n], y[:n]
        dth2 = None if dth is None else dth[:n]
        frame = frame.iloc[:n].reset_index(drop=True)
        yF = y[:, :3]
        load = np.linalg.norm(yF, axis=1)
        for model in MODEL_ORDER:
            mdl = models.get(model)
            if mdl is None:
                continue
            pred = np.asarray(mdl["predict"](X, dth2), float)[:, :3]
            err = pred - yF
            dwell_rmse = np.sqrt(np.mean(err ** 2, axis=1))
            vec_err = np.linalg.norm(err, axis=1)
            for i in range(n):
                rec0 = {
                    "run_id": rec.run_id,
                    "evaluation": "Test A" if split_name == "accuracy_test_A" else "Test B",
                    "model": model,
                    "model_name": model_name(model),
                    "pattern_index": int(frame.loc[i, "pattern_index"]),
                    "pattern_label": frame.loc[i, "pattern_label"],
                    "load_magnitude_N": float(load[i]),
                    "dwell_pooled_rmse_N": float(dwell_rmse[i]),
                    "vector_error_N": float(vec_err[i]),
                    "normalized_dwell_rmse_pct": float(100.0 * dwell_rmse[i] / load[i]) if load[i] > 1e-12 else np.nan,
                }
                # Targets and predictions are already in the sensor frame S
                # (see build_models), so no further rotation is applied here.
                for j, ch in enumerate(FORCE_LABELS):
                    rec0[f"delivered_{ch}_N"] = float(yF[i, j])
                    rec0[f"predicted_{ch}_N"] = float(pred[i, j])
                    rec0[f"residual_{ch}_N"] = float(err[i, j])
                rows.append(rec0)
    return pd.DataFrame(rows)


def residual_load_fit(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(pred_rows):
        return pd.DataFrame()
    for (run_id, model), g in pred_rows.groupby(["run_id", "model"]):
        x = g["load_magnitude_N"].to_numpy(dtype=float)
        y = g["dwell_pooled_rmse_N"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or np.ptp(x) <= 1e-12:
            continue
        A = np.column_stack([x, np.ones(len(x))])
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        p = slope * x + intercept
        rows.append({
            "run_id": run_id, "model": model,
            "slope_N_error_per_N_load": float(slope),
            "intercept_N": float(intercept),
            "r2": r2_plain(y, p), "n_dwells": len(x),
        })
    return pd.DataFrame(rows)


def aggregate_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame()
    g = df[df["input_force"] != "intercept"]
    return g.groupby(["output_channel", "input_force", "unit"])["coefficient"].agg(
        mean="mean", median="median", sd="std", min="min", max="max", n="count").reset_index()


def aggregate_hysteresis(h: pd.DataFrame) -> pd.DataFrame:
    if not len(h):
        return pd.DataFrame()
    return h.groupby(["applied_axis", "output_channel"])["abs_value_N"].agg(
        median_abs_N="median", mean_abs_N="mean", sd_abs_N="std",
        max_abs_N="max", n_pairs="count").reset_index()


def aggregate_anchor(a: pd.DataFrame) -> pd.DataFrame:
    if not len(a):
        return pd.DataFrame()
    return a.groupby("channel").agg(
        median_abs_slope_N_per_h=("anchor_slope_N_per_h", lambda x: np.nanmedian(np.abs(x))),
        sd_slope_N_per_h=("anchor_slope_N_per_h", "std"),
        median_span_N=("anchor_span_N", "median"),
        max_span_N=("anchor_span_N", "max"),
        n_runs=("run_id", "nunique"),
    ).reset_index()


def aggregate_residual_fit(fit: pd.DataFrame) -> pd.DataFrame:
    if not len(fit):
        return pd.DataFrame()
    return fit.groupby("model").agg(
        median_intercept_N=("intercept_N", "median"),
        sd_intercept_N=("intercept_N", "std"),
        median_slope_N_per_N=("slope_N_error_per_N_load", "median"),
        sd_slope_N_per_N=("slope_N_error_per_N_load", "std"),
        median_r2=("r2", "median"),
        n_runs=("run_id", "nunique"),
    ).reset_index()


def aggregate_ei(ei: pd.DataFrame) -> pd.DataFrame:
    if not len(ei):
        return pd.DataFrame()
    return ei.groupby("axis").agg(
        median_EI_apparent_Nm2=("EI_apparent_Nm2", "median"),
        sd_EI_apparent_Nm2=("EI_apparent_Nm2", "std"),
        median_ratio_to_handbook=("EI_ratio_apparent_to_handbook", "median"),
        median_fit_r2=("fit_r2", "median"),
        n_runs=("run_id", "nunique"),
    ).reset_index()


def parity_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(pred):
        return pd.DataFrame()
    for (model, ev), g in pred.groupby(["model", "evaluation"]):
        y = np.column_stack([g[f"delivered_{c}_N"].to_numpy(float) for c in FORCE_LABELS])
        p = np.column_stack([g[f"predicted_{c}_N"].to_numpy(float) for c in FORCE_LABELS])
        e = p - y
        rows.append({
            "model": model, "evaluation": ev,
            "pooled_rmse_N": float(np.sqrt(np.mean(e ** 2))),
            "residual_sd_N": float(np.std(e.ravel(), ddof=1)) if e.size > 1 else np.nan,
            "pooled_r2": r2_plain(y.ravel(), p.ravel()),
            "n_run_dwells": int(len(g)),
            "n_runs": int(g["run_id"].nunique()),
        })
    return pd.DataFrame(rows)
