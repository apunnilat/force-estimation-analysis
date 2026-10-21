"""Statistics across the repeated remountings (one fitted record per run).

Every run is analysed on its own first; this module then combines the runs:
model metrics per run, paired model contrasts with bootstrap confidence
intervals, between-remount spread, transfer of a calibration from one run to
another, worst-case dwells, error versus load, and the PA0 matrix decomposition.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import model_family, model_name
from .evaluation import channel_metrics_from_record
from .metrics import reproducibility, run_level_stats


def _between_run_reproducibility(records: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, summaries = [], []
    groups = sorted({r["group"] for r in records})
    for group in groups:
        rr = [r for r in records if r["group"] == group]
        if len(rr) < 2:
            continue
        out = reproducibility([r["dwells"] for r in rr], [r["run_id"] for r in rr])
        if not len(out):
            continue
        x = out.copy()
        # pandas propagates DataFrame.attrs on copy; the reproducibility output
        # stores a DataFrame in attrs["summary"], which cannot be compared
        # during concat across multiple groups. The summary is appended
        # explicitly below, so clear attrs on the row table.
        x.attrs = {}
        x.insert(0, "group", group)
        rows.append(x)
        sm = out.attrs.get("summary", pd.DataFrame()).copy()
        if len(sm):
            sm.insert(0, "group", group)
            summaries.append(sm)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame())


def transfer_from_records(records: Sequence[dict], n_boot: int
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-run calibration transfer: models fitted on one run, scored on the others.

    ``n_boot`` is the number of bootstrap resamples for the run-level summary
    (``evaluation.n_boot_transfer`` in configs/analysis.yaml).
    """
    rows = []
    for group in sorted({r["group"] for r in records}):
        rr = [r for r in records if r["group"] == group and r.get("model_res") is not None]
        if len(rr) < 2:
            continue
        for src in rr:
            models = src["model_res"]["models"]
            for dst in rr:
                X, y, dth = dst["model_res"]["splits"].get(
                    "accuracy_test_A", (np.empty((0, 6)), np.empty((0, 6)), None))
                if not len(X):
                    continue
                for name, mdl in models.items():
                    try:
                        p = mdl["predict"](X, dth)
                    except Exception:
                        continue
                    e = p[:, :3] - y[:, :3]
                    rows.append({
                        "group": group, "model": name,
                        "fit_on": src["run_id"], "tested_on": dst["run_id"],
                        "same_run": src["run_id"] == dst["run_id"],
                        "n_dwells": len(X),
                        "force_rmse_N": float(np.sqrt(np.mean(e ** 2))),
                        "force_vec_rmse_N": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                    })
    tr = pd.DataFrame(rows)
    if not len(tr):
        return tr, pd.DataFrame(), pd.DataFrame()

    # First reduce the many destination runs to one cross-run value per source
    # run.  This prevents a source run with many destinations from becoming a
    # pseudo-replicated observation in the final across-run summary.
    source_rows = []
    for (group, model, fit_on), g in tr.groupby(["group", "model", "fit_on"]):
        same = g[g["same_run"]]
        cross = g[~g["same_run"]]
        if not len(same) or not len(cross):
            continue
        same_rmse = float(same["force_rmse_N"].iloc[0])
        cross_med = float(cross["force_rmse_N"].median())
        source_rows.append({
            "group": group, "model": model, "fit_on": fit_on,
            "same_run_rmse_N": same_rmse,
            "cross_run_rmse_median_N": cross_med,
            "cross_run_rmse_min_N": float(cross["force_rmse_N"].min()),
            "cross_run_rmse_max_N": float(cross["force_rmse_N"].max()),
            "transfer_penalty_x": cross_med / same_rmse if same_rmse > 0 else np.nan,
            "n_other_runs": int(cross["tested_on"].nunique()),
        })
    src = pd.DataFrame(source_rows)

    summary_rows = []
    seed = 4000
    for (group, model), g in src.groupby(["group", "model"]):
        s_same = run_level_stats(g["same_run_rmse_N"], n_boot=n_boot, seed=seed); seed += 1
        s_cross = run_level_stats(g["cross_run_rmse_median_N"], n_boot=n_boot, seed=seed); seed += 1
        s_pen = run_level_stats(g["transfer_penalty_x"], n_boot=n_boot, seed=seed); seed += 1
        summary_rows.append({
            "group": group, "model": model, "n_source_runs": s_pen["n_runs"],
            "same_run_rmse_median_N": s_same["median"],
            "cross_run_rmse_median_N": s_cross["median"],
            "transfer_penalty_median_x": s_pen["median"],
            "transfer_penalty_q25_x": s_pen["q25"],
            "transfer_penalty_q75_x": s_pen["q75"],
            "transfer_penalty_ci95_median_lo_x": s_pen["ci95_median_lo"],
            "transfer_penalty_ci95_median_hi_x": s_pen["ci95_median_hi"],
        })
    return tr, src, pd.DataFrame(summary_rows)


def _extract_affine_dict(model: dict) -> Optional[dict]:
    fn = model.get("predict") if isinstance(model, dict) else None
    if fn is None:
        return None
    candidates: List[object] = []
    if getattr(fn, "__defaults__", None):
        candidates.extend(fn.__defaults__)
    if getattr(fn, "__closure__", None):
        for c in fn.__closure__:
            try:
                candidates.append(c.cell_contents)
            except Exception:
                pass
    for x in candidates:
        if isinstance(x, dict) and {"W", "sd", "mu_a", "mu_y"}.issubset(x.keys()):
            return x
    return None


def _raw_affine_coefficients(aff: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert standardized row-vector ridge parameters to y = A x + b.

    _ridge_predict implements y_row = x_row @ B + b, hence the conventional
    column-vector matrix is A = B.T.
    """
    W = np.asarray(aff["W"], dtype=float)
    sd = np.asarray(aff["sd"], dtype=float)
    mu_a = np.asarray(aff["mu_a"], dtype=float)
    mu_y = np.asarray(aff["mu_y"], dtype=float)
    B = W / sd[:, None]
    b = mu_y - (mu_a / sd) @ W
    return B.T, b


def _polar_force_block(A3: np.ndarray) -> dict:
    """SVD polar decomposition A = R S of a 3x3 force block."""
    A3 = np.asarray(A3, dtype=float)
    U, s, Vt = np.linalg.svd(A3)
    R = U @ Vt
    S = Vt.T @ np.diag(s) @ Vt
    detR = float(np.linalg.det(R))
    if detR > 0:
        c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))
    else:
        ang = np.nan
    return {
        "R": R, "S": S, "singular_values": s,
        "det_R": detR, "rotation_angle_deg": ang,
        "gain_x": float(S[0, 0]), "gain_y": float(S[1, 1]), "gain_z": float(S[2, 2]),
        "offdiag_force_fro": float(np.linalg.norm(A3 - np.diag(np.diag(A3)))),
    }


def pa0_decomposition(run_id: str, res: dict, known_fy_gain: float) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    aff = _extract_affine_dict(res["models"].get("PA0", {}))
    if aff is None:
        return {}, pd.DataFrame(), pd.DataFrame()
    A, b = _raw_affine_coefficients(aff)  # 3 x 6
    Af = A[:, :3]
    Am = A[:, 3:6]
    pol = _polar_force_block(Af)
    summary = {
        "run_id": run_id,
        "rotation_angle_deg": pol["rotation_angle_deg"],
        "det_rotation": pol["det_R"],
        "gain_x": pol["gain_x"],
        "gain_y": pol["gain_y"],
        "gain_z": pol["gain_z"],
        "gain_y_minus_known": pol["gain_y"] - known_fy_gain,
        "force_block_offdiag_fro": pol["offdiag_force_fro"],
        "moment_block_fro_N_per_Nmm": float(np.linalg.norm(Am)),
        "intercept_norm_N": float(np.linalg.norm(b)),
    }
    force_rows = []
    for i, outch in enumerate(("Fx_out", "Fy_out", "Fz_out")):
        for j, inch in enumerate(("Fx_in", "Fy_in", "Fz_in")):
            force_rows.append({"run_id": run_id, "output": outch, "input": inch,
                               "coefficient": float(Af[i, j])})
    moment_rows = []
    for i, outch in enumerate(("Fx_out", "Fy_out", "Fz_out")):
        for j, inch in enumerate(("Mx_in", "My_in", "Mz_in")):
            moment_rows.append({"run_id": run_id, "output": outch, "input": inch,
                                "coefficient_N_per_Nmm": float(Am[i, j])})
    return summary, pd.DataFrame(force_rows), pd.DataFrame(moment_rows)


def _evaluation_frame_from_dwells(dw: pd.DataFrame, split: str) -> pd.DataFrame:
    return dw[(dw["quality"] == "OK") & (dw["split"] == split)].copy().reset_index(drop=True)


def worst_case_rows(record: dict) -> pd.DataFrame:
    mr = record["model_res"]
    out: List[dict] = []
    for split_name, eval_name in (("accuracy_test_A", "Test A"), ("robustness_test_B", "Test B")):
        X, y, dth = mr["splits"][split_name]
        split_key = "test_A" if split_name == "accuracy_test_A" else "test_B"
        dd = _evaluation_frame_from_dwells(record["dwells"], split_key)
        # The evaluator preserves original order for the direct test splits.
        if len(dd) != len(X):
            dd = pd.DataFrame(index=np.arange(len(X)))
        for name, mdl in mr["models"].items():
            p = np.asarray(mdl["predict"](X, dth), dtype=float)[:, :3]
            e = p - y[:, :3]
            verr = np.linalg.norm(e, axis=1)
            cmax = np.max(np.abs(e), axis=1)
            for i in range(len(X)):
                rr = {
                    "run_id": record["run_id"], "model": name, "evaluation": eval_name,
                    "dwell_index": i,
                    "vector_error_N": float(verr[i]),
                    "max_component_error_N": float(cmax[i]),
                    "err_Fx_N": float(e[i, 0]), "err_Fy_N": float(e[i, 1]), "err_Fz_N": float(e[i, 2]),
                    "ref_Fx_N": float(y[i, 0]), "ref_Fy_N": float(y[i, 1]), "ref_Fz_N": float(y[i, 2]),
                    "pred_Fx_N": float(p[i, 0]), "pred_Fy_N": float(p[i, 1]), "pred_Fz_N": float(p[i, 2]),
                }
                if len(dd) == len(X):
                    for c in ("pattern_index", "pattern_label", "family"):
                        if c in dd.columns:
                            rr[c] = dd.loc[i, c]
                out.append(rr)
    return pd.DataFrame(out)


def error_vs_load_rows(record: dict, models: Sequence[str] = ("PA0", "D2", "PA1")) -> pd.DataFrame:
    mr = record["model_res"]
    rows = []
    for split_name, evaluation in (("accuracy_test_A", "Test A"), ("robustness_test_B", "Test B")):
        X, y, dth = mr["splits"][split_name]
        load = np.linalg.norm(y[:, :3], axis=1)
        for name in models:
            if name not in mr["models"]:
                continue
            p = np.asarray(mr["models"][name]["predict"](X, dth), dtype=float)[:, :3]
            e = p - y[:, :3]
            # Per-dwell pooled-component RMS: directly comparable with pooled RMSE
            # after RMS aggregation over dwells.
            er = np.sqrt(np.mean(e * e, axis=1))
            for i in range(len(load)):
                rows.append({"run_id": record["run_id"], "model": name,
                             "evaluation": evaluation, "load_N": float(load[i]),
                             "dwell_component_rms_error_N": float(er[i])})
    return pd.DataFrame(rows)


def model_metric_rows(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        cm = channel_metrics_from_record(r)
        if len(cm):
            rows.append(cm)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def model_summary_from_channels(cm: pd.DataFrame) -> pd.DataFrame:
    if not len(cm):
        return pd.DataFrame()
    d = cm[cm["force_channel"] == "Pooled"].copy()
    out = []
    for (ev, model), g in d.groupby(["evaluation", "model"]):
        vals = g["rmse_N"].to_numpy(dtype=float)
        nr = g["normalized_rmse_pct"].to_numpy(dtype=float)
        sd = g["standard_deviation_N"].to_numpy(dtype=float)
        out.append({
            "evaluation": ev, "model": model, "model_name": model_name(model),
            "family": model_family(model), "n_runs": len(g),
            "rmse_median_N": float(np.nanmedian(vals)),
            "rmse_run_sd_N": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "rmse_min_N": float(np.nanmin(vals)), "rmse_max_N": float(np.nanmax(vals)),
            "normalized_rmse_median_pct": float(np.nanmedian(nr)),
            "residual_sd_median_N": float(np.nanmedian(sd)),
            "r2_median": float(np.nanmedian(g["r2"].to_numpy(dtype=float))),
        })
    return pd.DataFrame(out)


def paired_contrast(cm: pd.DataFrame, evaluation: str, a: str, b: str, n_boot: int) -> dict:
    """Paired comparison of two models across runs: median RMSE difference a - b with a bootstrap 95 % CI."""
    d = cm[(cm["evaluation"] == evaluation) & (cm["force_channel"] == "Pooled")]
    aa = d[d["model"] == a][["run_id", "rmse_N"]].rename(columns={"rmse_N": "a"})
    bb = d[d["model"] == b][["run_id", "rmse_N"]].rename(columns={"rmse_N": "b"})
    m = aa.merge(bb, on="run_id")
    if not len(m):
        return {"evaluation": evaluation, "model_a": a, "model_b": b, "n_runs": 0}
    delta = (m["a"] - m["b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(20260911 + len(m) + sum(map(ord, a+b)))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.median(rng.choice(delta, size=len(delta), replace=True))
    return {
        "evaluation": evaluation, "model_a": a, "model_b": b,
        "n_runs": len(m), "median_a_minus_b_N": float(np.median(delta)),
        "mean_a_minus_b_N": float(np.mean(delta)),
        "sd_delta_N": float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan,
        "ci95_median_lo_N": float(np.quantile(boots, 0.025)),
        "ci95_median_hi_N": float(np.quantile(boots, 0.975)),
        "a_better_count": int(np.sum(delta < 0)),
        "b_better_count": int(np.sum(delta > 0)),
        "ties": int(np.sum(np.isclose(delta, 0.0, atol=1e-12))),
    }


def count_best_models(cm: pd.DataFrame, evaluation: str, models: Sequence[str]) -> pd.DataFrame:
    d = cm[(cm["evaluation"] == evaluation) & (cm["force_channel"] == "Pooled") & (cm["model"].isin(models))]
    rows = []
    for run_id, g in d.groupby("run_id"):
        if not len(g):
            continue
        mn = g["rmse_N"].min()
        winners = g[np.isclose(g["rmse_N"], mn, rtol=1e-10, atol=1e-12)]["model"].tolist()
        for w in winners:
            rows.append({"run_id": run_id, "evaluation": evaluation, "winner": w, "rmse_N": mn})
    if not rows:
        return pd.DataFrame()
    rr = pd.DataFrame(rows)
    return rr.groupby(["evaluation", "winner"], as_index=False).agg(
        best_count=("run_id", "nunique"), n_winner_rows=("run_id", "size"))


def repeatability_raw_summary(records: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in records:
        iso = r["iso"]
        q = iso[(iso["metric"] == "repeatability_b") & (iso["unit"] == "N")]
        for _, z in q.iterrows():
            ch = str(z["channel"]).split("_")[0]
            rows.append({"run_id": r["run_id"], "force_channel": ch, "repeatability_sd_N": float(z["value"])})
    raw = pd.DataFrame(rows)
    if not len(raw):
        return raw, pd.DataFrame()
    sm = raw.groupby("force_channel")["repeatability_sd_N"].agg(
        n_runs="count", median_N="median", mean_N="mean", run_sd_N="std", min_N="min", max_N="max").reset_index()
    return raw, sm


def between_remount_spread(records: Sequence[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Use the generic matched-pattern implementation and retain only force channels.
    detail, summary = _between_run_reproducibility(records)
    if len(detail) and "channel" in detail.columns:
        detail = detail[detail["channel"].astype(str).str.startswith(("Fx", "Fy", "Fz"))].copy()
    if len(summary) and "channel" in summary.columns:
        summary = summary[summary["channel"].astype(str).str.startswith(("Fx", "Fy", "Fz"))].copy()
    return detail, summary


def descriptive_error_load_fit(err: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(err):
        return pd.DataFrame()
    for (run_id, model), g in err.groupby(["run_id", "model"]):
        x = g["load_N"].to_numpy(dtype=float)
        y = g["dwell_component_rms_error_N"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if np.sum(ok) < 3 or np.ptp(x[ok]) < 1e-12:
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        rows.append({"run_id": run_id, "model": model,
                     "error_intercept_N": float(intercept),
                     "error_slope_N_per_N": float(slope),
                     "n_dwells": int(np.sum(ok))})
    return pd.DataFrame(rows)
