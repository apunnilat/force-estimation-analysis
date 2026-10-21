"""Model-free quantities and error metrics, all computed on dwell means.

* ``noise_ceiling``: within-dwell scatter, the lowest error any model can reach.
* ``iso376_metrics``: ISO 376-style repeatability, hysteresis and zero return.
* ``zero_drift`` and ``reproducibility``: drift within a run and spread between runs.
* ``direction_error``: angle between predicted and reference force vectors.
* ``run_level_stats``: median, spread and bootstrap CI over runs.
* small numeric helpers (``r2``, ``r2_plain``, ``safe_float``, ``format_number``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import channel_unit, FORCE_CMD, FORCE_MEAS_S, MOMENT_MEAS


def noise_ceiling(dwells: pd.DataFrame,
                  channels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Descriptive variability scale from two independent routes.

    sensor_floor     median within-dwell SD -- the measurement noise at the
                     operating point, per channel.
    reference_floor  SD across repeats of the SAME commanded wrench -- how
                     reproducibly the haptic device delivers it. This includes
                     the sensor noise, so it is normally the larger of the two
                     and is the one that binds.

    This is intentionally *not* treated as a traceable uncertainty bound. The
    repeated-reference term mixes reference variation, mechanics, drift and sensor
    response, while the within-dwell SD describes the filtered measurement process.
    The returned ``floor`` field is kept for backward compatibility with existing
    notebooks and should be interpreted as a variability scale.
    """
    channels = list(channels or [c for c in FORCE_MEAS_S + MOMENT_MEAS
                                 if f"{c}_mean" in dwells.columns])
    ok = dwells[dwells["quality"] == "OK"]
    ref = ok[ok["family"] == "reference"]

    rows = []
    for c in channels:
        sd = ok[f"{c}_sd"].dropna()
        sensor = float(sd.median()) if len(sd) else np.nan
        reference = (float(ok[f"{c}_mean"].loc[ref.index].std(ddof=1))
                     if len(ref) >= 3 else np.nan)
        floor = np.nanmax([sensor, reference])
        rows.append({
            "channel": c,
            "unit": channel_unit(c),
            "sensor_floor": sensor,
            "n_dwells_sensor": int(len(sd)),
            "reference_floor": reference,
            "n_repeats_reference": int(len(ref)),
            "floor": floor,
            "binding": ("reference" if np.isfinite(reference) and reference >= (sensor or 0)
                        else "sensor"),
        })
    out = pd.DataFrame(rows)

    # Vector descriptive scale over the three force channels, for comparison
    # against the force-vector RMSE the model comparison reports. The legacy
    # attribute name is retained for compatibility with existing notebooks.
    f_rows = out[out["unit"] == "N"]
    if len(f_rows) == 3:
        out.attrs["force_vector_floor"] = float(
            np.sqrt(np.sum(f_rows["floor"].to_numpy() ** 2)))
    return out


def iso376_metrics(dwells: pd.DataFrame,
                   meta: Optional[Dict[str, str]] = None,
                   channels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Hysteresis v, repeatability b, zero drift f0 and anchor drift.

    These are model-free descriptors of the rig. They quantify hysteresis,
    within-session repeatability, within-dwell scatter and drift, but they are
    not a complete traceable uncertainty budget and are not treated as proven
    lower bounds on achievable reconstruction error. The ISO 376-style naming
    is used for clearly defined descriptive quantities; this multiaxial protocol
    is not asserted to be a standards-compliant force-proving calibration.
    """
    channels = list(channels or [c for c in FORCE_MEAS_S if f"{c}_mean" in dwells.columns])
    rows: List[dict] = []
    ok = dwells[dwells["quality"] == "OK"]

    # --- hysteresis v -----------------------------------------------------
    # Bidirectional staircases revisit the same load levels on the way down.
    # The ascending/descending gap at a matched level is friction and stiction
    # in the collet and the shaft-clamp interface.
    stair = ok[ok["family"] == "stair_bidir"]
    for sid, g in stair.groupby("series_id"):
        g = g.sort_values("step_index")
        peak_step = int(g["step_index"].max())
        n_up = (peak_step // 2) + 1
        up = g[g["step_index"] < n_up]
        down = g[g["step_index"] >= n_up]
        for _, u in up.iterrows():
            match = down[np.isclose(down["cmd_resultant_N"], u["cmd_resultant_N"],
                                    rtol=1e-6, atol=1e-9)]
            if not len(match):
                continue
            d = match.iloc[0]
            for c in channels:
                rows.append({
                    "metric": "hysteresis_v",
                    "group": f"series_{int(sid)}",
                    "channel": c,
                    "level_N": float(u["cmd_resultant_N"]),
                    "value": float(d[f"{c}_mean"] - u[f"{c}_mean"]),
                    "unit": channel_unit(c),
                    "n": 2,
                    "note": "descending minus ascending at matched load",
                })

    # --- repeatability b --------------------------------------------------
    # Spread across repeats of one wrench WITHOUT remounting.
    ref = ok[ok["family"] == "reference"]
    if len(ref) >= 2:
        for c in channels:
            v = ref[f"{c}_mean"].to_numpy()
            rows.append({
                "metric": "repeatability_b", "group": "reference_wrench",
                "channel": c, "level_N": float(ref["cmd_resultant_N"].mean()),
                "value": float(np.std(v, ddof=1)), "unit": channel_unit(c),
                "n": int(len(v)), "note": "SD across within-session repeats",
            })
            rows.append({
                "metric": "repeatability_b_range", "group": "reference_wrench",
                "channel": c, "level_N": float(ref["cmd_resultant_N"].mean()),
                "value": float(np.ptp(v)), "unit": channel_unit(c),
                "n": int(len(v)), "note": "max - min across within-session repeats",
            })

    # --- within-dwell noise floor ----------------------------------------
    # Measured, not taken from the datasheet. At a ~1.5 N operating point on a
    # +-125 N sensor the datasheet allowance is nearly as large as the signal,
    # so the empirical figure is worth having.
    for c in channels:
        sd = ok[f"{c}_sd"].dropna()
        if len(sd):
            rows.append({
                "metric": "within_dwell_sd_median", "group": "all_ok",
                "channel": c, "level_N": np.nan, "value": float(sd.median()),
                "unit": channel_unit(c), "n": int(len(sd)),
                "note": "empirical within-dwell variability at the operating point",
            })

    # --- anchor drift -----------------------------------------------------
    anc = dwells[dwells["family"] == "anchor"].sort_values("t_mid_s")
    if len(anc) >= 2:
        t = anc["t_mid_s"].to_numpy() - anc["t_mid_s"].to_numpy()[0]
        for c in channels:
            v = anc[f"{c}_mean"].to_numpy()
            slope = np.polyfit(t, v, 1)[0] if len(t) >= 2 else np.nan
            rows.append({
                "metric": "anchor_drift_slope", "group": "anchor",
                "channel": c, "level_N": float(anc["cmd_resultant_N"].mean()),
                "value": float(slope * 3600.0),
                "unit": ("N/h" if channel_unit(c) == "N" else "Nmm/h"),
                "n": int(len(v)),
                "note": "three anchors give a slope; two only give an endpoint difference",
            })
            rows.append({
                "metric": "anchor_span", "group": "anchor", "channel": c,
                "level_N": float(anc["cmd_resultant_N"].mean()),
                "value": float(np.ptp(v)), "unit": channel_unit(c),
                "n": int(len(v)), "note": "first-to-last spread",
            })

    return pd.DataFrame(rows)


def zero_drift(df: pd.DataFrame, tail_frac: float = 0.5) -> pd.DataFrame:
    """ISO 376 f0: the reading at zero commanded load, over the session.

    Zero-load samples are the settled tails of each RAMP_DOWN. Blocks are
    contiguous runs of rows with all force commands at zero.
    """
    # The controller snaps the feedforward to exactly zero, but rows logged
    # during the final approach carry values of order ramp_rate * dt. 1e-3 N is
    # far below the lowest commanded level (0.2 * 4 N = 0.8 N).
    cmd = df[FORCE_CMD].abs().max(axis=1)
    mask = (cmd < 1e-3).to_numpy()
    if not mask.any():
        return pd.DataFrame()

    block = np.cumsum(~mask)
    time_col = "t_rel_s" if "t_rel_s" in df.columns else "time_s"
    channels = [c for c in FORCE_MEAS_S + MOMENT_MEAS if c in df.columns]

    out: List[dict] = []
    for b, g in df[mask].groupby(block[mask]):
        if len(g) < 5:
            continue
        g = g.sort_values(time_col)
        tail = g.iloc[int((1 - tail_frac) * len(g)):]
        rec = {"block": int(b), "t_mid_s": float(pd.to_numeric(tail[time_col]).mean()),
               "n": int(len(tail))}
        for c in channels:
            rec[c] = float(pd.to_numeric(tail[c], errors="coerce").mean())
        out.append(rec)

    z = pd.DataFrame(out)
    if len(z) >= 2:
        t = z["t_mid_s"].to_numpy() - z["t_mid_s"].to_numpy()[0]
        summary = {"n_blocks": len(z)}
        for c in channels:
            summary[f"{c}_first"] = z[c].iloc[0]
            summary[f"{c}_last"] = z[c].iloc[-1]
            summary[f"{c}_drift_per_h"] = np.polyfit(t, z[c].to_numpy(), 1)[0] * 3600.0
        z.attrs["summary"] = summary
    return z


def r2(y, yhat, min_rel_var: float = 1e-6):
    """R^2, returning NaN when the reference has essentially no variance.

    Repeated identical wrenches (anchors, reference dwells) give ss_tot near
    zero, which makes R^2 explode to enormous negative values that look like
    catastrophic model failure and are actually an artefact of the denominator.
    """
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    scale = float(np.sum(y ** 2))
    if ss_tot <= 0 or (scale > 0 and ss_tot / scale < min_rel_var):
        return np.nan
    return 1.0 - ss_res / ss_tot


def direction_error(y_true: np.ndarray, y_pred: np.ndarray,
                    split_labels: Optional[Sequence[str]] = None,
                    min_norm: float = 1e-6) -> pd.DataFrame:
    """Angle between true and predicted force vectors, per dwell.

    Clinically the more legible statistic: a 90 deg median direction error says
    more than R^2 = -4466, and it is the quantity the Schaefer et al. JND
    thresholds are expressed in.
    """
    y_true = np.asarray(y_true, dtype=float)[:, :3]
    y_pred = np.asarray(y_pred, dtype=float)[:, :3]
    nt = np.linalg.norm(y_true, axis=1)
    npd = np.linalg.norm(y_pred, axis=1)
    valid = (nt > min_norm) & (npd > min_norm)
    cos = np.full(len(y_true), np.nan)
    cos[valid] = np.sum(y_true[valid] * y_pred[valid], axis=1) / (nt[valid] * npd[valid])
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    splits = np.asarray(split_labels if split_labels is not None else ["all"] * len(ang))
    rows = []
    for sp in pd.unique(splits):
        a = ang[(splits == sp) & np.isfinite(ang)]
        if not len(a):
            continue
        rows.append({
            "split": sp, "n": int(len(a)),
            "median_deg": float(np.median(a)),
            "p95_deg": float(np.percentile(a, 95)),
            "max_deg": float(np.max(a)),
        })
    return pd.DataFrame(rows)


def reproducibility(dwell_frames: Sequence[pd.DataFrame],
                    labels: Sequence[str],
                    channels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """ISO 376-style b': spread across matched patterns in repeated runs.

    Within-session reference repeats describe repeatability without remounting.
    When the acquisition design changes only the mounting between otherwise
    comparable runs, the extra between-run spread is evidence for a mounting
    contribution. It is not treated as a standards-compliant ISO 376 estimate
    or as a uniquely isolated mounting uncertainty. Patterns are matched by
    label, so comparable runs should share the same excitation design.
    """
    channels = list(channels or FORCE_MEAS_S)
    merged: List[pd.DataFrame] = []
    for lab, d in zip(labels, dwell_frames):
        sub = d[d["quality"] == "OK"][["pattern_label", "family", "split"] +
                                      [f"{c}_mean" for c in channels]].copy()
        sub["mounting"] = lab
        merged.append(sub)
    allm = pd.concat(merged, ignore_index=True)

    rows: List[dict] = []
    for label, g in allm.groupby("pattern_label"):
        if g["mounting"].nunique() < 2:
            continue
        for c in channels:
            v = g[f"{c}_mean"].to_numpy()
            rows.append({
                "pattern_label": label,
                "family": g["family"].iloc[0],
                "split": g["split"].iloc[0],
                "channel": c,
                "n_mountings": int(g["mounting"].nunique()),
                "mean": float(np.mean(v)),
                "b_prime_sd": float(np.std(v, ddof=1)),
                "b_prime_range": float(np.ptp(v)),
            })
    out = pd.DataFrame(rows)
    if not len(out):
        return out

    summary = (out.groupby("channel")
               .agg(n_patterns=("pattern_label", "nunique"),
                    b_prime_sd_median=("b_prime_sd", "median"),
                    b_prime_sd_p95=("b_prime_sd", lambda s: float(np.percentile(s, 95))),
                    b_prime_range_max=("b_prime_range", "max"))
               .reset_index())
    out.attrs["summary"] = summary
    return out


def nanmean_or_nan(vals) -> float:
    """Mean ignoring NaN, returning NaN for an all-NaN block without warning.

    An all-NaN block is the normal case for the moment channels under
    force-only excitation, where the target is a constant zero.
    """
    v = np.asarray(vals, dtype=float)
    return float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan


def run_level_stats(values: Sequence[float], n_boot: int = 5000,
                     seed: int = 0) -> Dict[str, float]:
    """Descriptive summary with a bootstrap interval over RUNS, not dwells."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n == 0:
        return {"n_runs": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
                "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan,
                "ci95_median_lo": np.nan, "ci95_median_hi": np.nan}
    out = {
        "n_runs": n,
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if n > 1 else np.nan,
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }
    if n >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        med = np.median(v[idx], axis=1)
        out["ci95_median_lo"] = float(np.percentile(med, 2.5))
        out["ci95_median_hi"] = float(np.percentile(med, 97.5))
    else:
        out["ci95_median_lo"] = np.nan
        out["ci95_median_hi"] = np.nan
    return out


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def format_number(x, nd=3):
    return "n/a" if not np.isfinite(safe_float(x)) else f"{float(x):.{nd}f}"


def r2_plain(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss = np.sum((y - p) ** 2)
    den = np.sum((y - np.mean(y)) ** 2)
    return np.nan if den <= 1e-15 else float(1.0 - ss / den)
