"""Loading and checking the recorded runs.

* ``load_run`` reads one schema-7 CSV (older v2/v3 files are migrated) and adds
  the sensor-frame force columns.
* ``audit_run`` checks the file before any analysis (schema, completeness,
  sensor timing, saturation).
* ``reduce_dwells`` reduces each dwell to one point. The dwell, not the CSV row,
  is the sampling unit: a 5 s dwell at 200 Hz gives ~1000 rows that are strongly
  correlated (EMA filter, repeated sensor messages, constant command). Rows are
  de-duplicated on ``sensor_msg_count`` before averaging, and the effective
  sample count is reported as ``n_eff``.
* ``discover_campaign`` finds the files of each run folder.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_SCHEMA_VERSION,
    FORCE_CMD,
    FORCE_CMD_S,
    FORCE_MEAS,
    FORCE_MEAS_S,
    FORCE_REF,
    FORCE_REF_S,
    MOMENT_CMD,
    MOMENT_MEAS,
    MOMENT_REF,
    POSE_DEV,
    RAMP_DWELL,
    RAW_SENSOR,
    rotate_ee_to_s,
)


def _add_sensor_frame_force_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach sensor-frame (S) counterparts of the EE-frame force triplets.

    Adds Fx/Fy/Fz_meas_S_N, f_delivered_*_S_N and fx/fy/fz_cmd_S_N wherever the
    corresponding EE-frame triplet is fully present, computed in one pass via
    the fixed EE_TO_S rotation. Existing EE columns are left untouched -- the
    physics models keep consuming those. Operates in place and also returns df.
    """
    for ee_cols, s_cols in ((FORCE_MEAS, FORCE_MEAS_S),
                            (FORCE_REF, FORCE_REF_S),
                            (FORCE_CMD, FORCE_CMD_S)):
        if all(c in df.columns for c in ee_cols):
            F_ee = df[ee_cols].to_numpy(dtype=float)
            F_s = rotate_ee_to_s(F_ee)
            for i, c in enumerate(s_cols):
                df[c] = F_s[:, i]
    return df


def load_run(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read a schema-7 CSV plus its '#' provenance lines.

    Provenance appears both as a header block (written at Phase 5 entry) and as
    a trailer (written at close: completed_patterns, aborted_at_pattern,
    skipped_pattern_indices). Both are collected; pandas skips them via
    comment='#'.
    """
    meta: Dict[str, str] = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            body = line[1:].strip()
            if not body:
                continue
            key, _, value = body.partition(",")
            meta[key.strip()] = value.strip()

    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.astype(str).str.strip()

    if is_legacy(df, meta):
        df, meta, notes = migrate_legacy(df, meta)
        print(f"  [legacy] {os.path.basename(path)} is not schema "
              f"{EXPECTED_SCHEMA_VERSION}; migrating:")
        for n in notes:
            print(f"    - {n}")

    df = _add_sensor_frame_force_columns(df)

    meta["_path"] = path
    meta["_n_rows"] = str(len(df))
    return df, meta


def meta_float(meta: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(meta[key])
    except (KeyError, TypeError, ValueError):
        return default


def load_weight_run(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    meta: Dict[str, str] = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(",")
                if k:
                    meta[k.strip()] = v.strip()
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.astype(str).str.strip()
    if "stage" not in df.columns:
        raise ValueError(f"{path} has no 'stage' column -- this reads the "
                         f"weight_validation.csv written by lambda_controller_4, "
                         f"not the excitation log.")
    df = _add_sensor_frame_force_columns(df)
    meta["_path"] = path
    return df, meta


# Migration of older (schema v2/v3) files.
#
# Older runs use different column names and have no pattern_index, so nothing
# downstream can group by dwell. This maps what exists and synthesises what can
# be derived; it cannot invent what was never logged. Columns that are simply
# absent (sensor_age_s, the raw sensor-frame values, velocity, series_id) stay
# absent, and the audit reports those checks as MISSING rather than passing --
# a migrated file must not look as trustworthy as a natively logged one.
LEGACY_RENAMES = {
    "Fx_meas_N": "Fx_meas_EE_N", "Fy_meas_N": "Fy_meas_EE_N",
    "Fz_meas_N": "Fz_meas_EE_N", "Tx_meas_Nm": "Tx_meas_EE_Nm",
    "Ty_meas_Nm": "Ty_meas_EE_Nm", "Tz_meas_Nm": "Tz_meas_EE_Nm",
    "f_delivered_x_EE": "f_delivered_x_EE_N",
    "f_delivered_y_EE": "f_delivered_y_EE_N",
    "f_delivered_z_EE": "f_delivered_z_EE_N",
    "t_delivered_x_EE": "t_delivered_yaw_Nm",
    "t_delivered_y_EE": "t_delivered_pitch_Nm",
    "t_delivered_z_EE": "t_delivered_roll_Nm",
    "th_relative_x_EE": "dth_yaw_rad", "th_relative_y_EE": "dth_pitch_rad",
    "th_relative_z_EE": "dth_roll_rad",
    "p_relative_x_EE": "p_relative_x_EE_m",
    "p_relative_y_EE": "p_relative_y_EE_m",
    "p_relative_z_EE": "p_relative_z_EE_m",
    "phase_name": "family",
}


def is_legacy(df: pd.DataFrame, meta: Dict[str, str]) -> bool:
    if meta.get("schema_version") == str(EXPECTED_SCHEMA_VERSION):
        return False
    return "pattern_index" not in df.columns or "Fx_meas_N" in df.columns


def migrate_legacy(df: pd.DataFrame, meta: Dict[str, str]
                   ) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Bring a v2/v3 file up to the schema-7 column names."""
    notes: List[str] = []
    d = df.copy()
    d.columns = d.columns.astype(str).str.strip()

    renamed = {k: v for k, v in LEGACY_RENAMES.items() if k in d.columns}
    d = d.rename(columns=renamed)
    if renamed:
        notes.append(f"renamed {len(renamed)} legacy columns")

    # pattern_index from contiguous runs of (split, family, pattern_label) --
    # the same block-assignment rule wrench_common.assign_blocks uses.
    if "pattern_index" not in d.columns:
        keys = [c for c in ("split", "family", "pattern_label") if c in d.columns]
        if not keys:
            raise ValueError(
                "cannot migrate: no pattern_index and no split/family/"
                "pattern_label columns to derive dwell boundaries from")
        key = d[keys].astype(str).agg("||".join, axis=1)
        d["pattern_index"] = (key != key.shift()).cumsum() - 1
        notes.append(f"synthesised pattern_index from contiguous {keys} runs "
                     f"({d['pattern_index'].nunique()} dwells)")

    # Staircases did not exist before v4, so every dwell is standalone.
    for col, val in (("series_id", -1), ("step_index", -1), ("skipped", 0)):
        if col not in d.columns:
            d[col] = val
    notes.append("series_id/step_index set to -1: v3 had no staircase series, "
                 "so ISO 376 hysteresis cannot be computed from this file")

    if "t_rel_s" not in d.columns and "time_s" in d.columns:
        t = pd.to_numeric(d["time_s"], errors="coerce")
        d["t_rel_s"] = t - t.min()

    missing = [c for c in ("sensor_age_s", "sensor_msg_count", "v_mag_mps",
                           "Fx_raw_sensor_N", "kp_scale") if c not in d.columns]
    if missing:
        notes.append(
            f"NOT RECOVERABLE from this file: {', '.join(missing)}. Staleness "
            f"cannot be distinguished from a settled dwell, duplicate rows "
            f"cannot be identified, and the quasi-static assumption behind "
            f"f_delivered cannot be checked. Treat the resulting uncertainties "
            f"as lower bounds.")

    meta = dict(meta)
    meta["_migrated_from"] = meta.get("schema_version", "unknown")
    meta["schema_version"] = str(EXPECTED_SCHEMA_VERSION)
    meta["_migration_notes"] = " | ".join(notes)
    return d, meta, notes


def audit_run(df: pd.DataFrame, meta: Dict[str, str]) -> pd.DataFrame:
    """Data-quality checks that must pass before anything is fitted.

    Returns a tidy table of (check, value, verdict, note). Nothing here is
    fatal on its own -- the point is that the failure modes become visible
    instead of being silently absorbed into the results.
    """
    rows: List[dict] = []

    def add(check, value, verdict, note=""):
        rows.append({"check": check, "value": value, "verdict": verdict, "note": note})

    # --- provenance -------------------------------------------------------
    schema = meta.get("schema_version", "MISSING")
    if "_migrated_from" in meta:
        add("schema_version", f"MIGRATED from {meta['_migrated_from']}", "WARN",
            meta.get("_migration_notes", ""))
    else:
        add("schema_version", schema,
            "OK" if schema == str(EXPECTED_SCHEMA_VERSION) else "FAIL",
            f"expected {EXPECTED_SCHEMA_VERSION}")
    add("controller_build", meta.get("controller_build", "MISSING"),
        "OK" if "controller_build" in meta else
        ("MISSING" if "_migrated_from" in meta else "FAIL"),
        "identifies which binary wrote this file")
    add("excitation_type", meta.get("excitation_type", "MISSING"),
        "OK" if "excitation_type" in meta else "WARN")
    legacy = "_migrated_from" in meta
    for k in ("force_bias_N", "torque_bias_Nm", "ema_alpha", "pattern_seed"):
        add(k, meta.get(k, "MISSING"),
            "OK" if k in meta else ("MISSING" if legacy else "FAIL"),
            "baked into every logged value" if "bias" in k or "ema" in k else "")

    # --- completion -------------------------------------------------------
    if "aborted_at_pattern" in meta:
        add("run_completed", f"ABORTED at {meta['aborted_at_pattern']}", "FAIL",
            "operator ended Phase 5 early")
    elif "completed_patterns" in meta:
        done = meta.get("completed_patterns")
        total = meta.get("total_patterns", "?")
        add("run_completed", f"{done}/{total}",
            "OK" if done == total else "WARN")
    else:
        add("run_completed", "NO TRAILER",
            "MISSING" if "_migrated_from" in meta else "FAIL",
            "cannot distinguish an aborted run from a short complete one")

    if "skipped_pattern_indices" in meta:
        n_sk = len(meta["skipped_pattern_indices"].split(";"))
        add("skipped_patterns", n_sk, "WARN", meta["skipped_pattern_indices"])
    if "clamped_patterns" in meta:
        add("clamped_patterns", meta["clamped_patterns"], "FAIL",
            "these did not deliver nominal amplitude")
    if "skipped_phases" in meta:
        add("skipped_alignment_phases", meta["skipped_phases"], "WARN")

    # --- sensor liveness --------------------------------------------------
    # A dead /sensor_data topic produces a perfectly constant reading, and so
    # does a settled dwell. sensor_age_s is what separates them.
    if "sensor_age_s" in df.columns:
        age = pd.to_numeric(df["sensor_age_s"], errors="coerce")
        add("sensor_age_max_s", round(float(np.nanmax(age)), 4),
            "FAIL" if np.nanmax(age) > 0.1 else "OK")
        add("sensor_age_p99_s", round(float(np.nanpercentile(age.dropna(), 99)), 4), "INFO")
        frac_stale = float((age > 0.1).mean())
        add("rows_stale_frac", round(frac_stale, 5),
            "FAIL" if frac_stale > 0.0 else "OK", "sensor_age_s > 0.1 s")
    else:
        add("sensor_age_s", "MISSING COLUMN",
            "MISSING" if "_migrated_from" in meta else "FAIL",
            "staleness cannot be distinguished from a settled dwell")

    # --- effective sensor rate and duplicate rows -------------------------
    if "sensor_msg_count" in df.columns and "t_rel_s" in df.columns:
        cnt = pd.to_numeric(df["sensor_msg_count"], errors="coerce")
        t = pd.to_numeric(df["t_rel_s"], errors="coerce")
        span = float(t.max() - t.min())
        n_msgs = float(cnt.max() - cnt.min())
        rate = n_msgs / span if span > 0 else np.nan
        log_rate = len(df) / span if span > 0 else np.nan
        add("sensor_rate_hz", round(rate, 1), "INFO")
        add("log_rate_hz", round(log_rate, 1), "INFO")
        dup_frac = float((cnt.diff() == 0).mean())
        add("duplicate_row_frac", round(dup_frac, 4),
            "WARN" if dup_frac > 0.05 else "OK",
            "rows repeating one sensor message; de-duplicated before averaging")

    # --- quasi-static assumption -----------------------------------------
    # f_delivered neglects the damping term on the assumption v ~ 0. This is
    # where that assumption is checked rather than asserted.
    dwell = df[df["ramp_state"] == RAMP_DWELL] if "ramp_state" in df.columns else df
    if "v_mag_mps" in df.columns and len(dwell):
        v = pd.to_numeric(dwell["v_mag_mps"], errors="coerce")
        vmax = float(np.nanmax(v))
        add("dwell_velocity_max_mps", round(vmax, 6),
            "WARN" if vmax > 1e-3 else "OK",
            "f_delivered neglects damping; large v invalidates that")
        add("dwell_velocity_p95_mps", round(float(np.nanpercentile(v.dropna(), 95)), 6), "INFO")
    else:
        add("v_mag_mps", "MISSING COLUMN", "WARN",
            "quasi-static assumption cannot be verified")

    # --- stiffness scale during excitation --------------------------------
    if "kp_scale" in dwell.columns and len(dwell):
        kps = pd.to_numeric(dwell["kp_scale"], errors="coerce")
        add("dwell_kp_scale_range", f"{kps.min():.3f}..{kps.max():.3f}", "INFO",
            "at 0, f_cmd equals f_delivered at equilibrium")

    # --- split accounting -------------------------------------------------
    if "split" in df.columns and "pattern_index" in df.columns:
        per = df.groupby("pattern_index")["split"].first()
        counts = per.value_counts().to_dict()
        scored = sum(counts.get(k, 0) for k in ("train", "test_A", "test_B"))
        for k in ("preload", "train", "test_A", "test_B", "anchor"):
            add(f"n_dwells_{k}", counts.get(k, 0), "INFO")
        if scored:
            pct = tuple(round(100.0 * counts.get(k, 0) / scored, 1)
                        for k in ("train", "test_A", "test_B"))
            add("split_pct", f"{pct[0]}/{pct[1]}/{pct[2]}",
                "OK" if abs(pct[0] - 80) <= 1 and abs(pct[1] - 10) <= 1
                and abs(pct[2] - 10) <= 1 else "WARN")

        # Distinct-wrench check: v3's pattern table silently collapsed to half
        # its nominal coverage while still reporting 80/10/10. Deliberate
        # repeats are exempt, and so are staircase revisits -- the ascending
        # and descending legs return to the same load levels on purpose, and
        # their difference IS the hysteresis measurement. Keying on
        # (wrench, series_id, step_index) permits those while still catching
        # duplicated standalone directions, which is the actual failure mode.
        fam = df.groupby("pattern_index")["family"].first()
        # Read the commanded wrench from DWELL rows only. Taking a max over all
        # rows collapses every negative-direction pattern to zero, because the
        # ramp passes through zero and 0 exceeds any negative value -- which
        # made every negative pattern look like a duplicate of every other.
        dw_rows = df[df["ramp_state"] == RAMP_DWELL] if "ramp_state" in df.columns else df
        cmd = dw_rows.groupby("pattern_index")[FORCE_CMD].median().round(6)
        for extra in ("series_id", "step_index"):
            if extra in df.columns:
                cmd[extra] = df.groupby("pattern_index")[extra].first()
        scored_mask = (per.isin(["train", "test_A", "test_B"]) &
                       ~fam.isin(["reference", "anchor", "preload"]))
        sub = cmd[scored_mask.reindex(cmd.index, fill_value=False)]
        n_unique = len(sub.drop_duplicates())
        add("distinct_scored_wrenches", f"{n_unique}/{len(sub)}",
            "OK" if n_unique == len(sub) else "FAIL",
            "duplicates mean reported coverage does not exist")

    return pd.DataFrame(rows)


def reduce_dwells(df: pd.DataFrame,
                  meta: Optional[Dict[str, str]] = None, *,
                  settle_frac: float,
                  min_samples: int = 5,
                  dedupe: bool = True) -> pd.DataFrame:
    """Collapse each dwell to one point: mean, SD and effective n per channel.

    settle_frac  fraction of the dwell window discarded at the start, so the
                 tail of the ramp transient is not averaged in.
    dedupe       keep one row per distinct sensor message. Averaging repeated
                 copies leaves the mean alone but biases the SD down and
                 inflates n.
    """
    if "ramp_state" not in df.columns:
        raise ValueError("no ramp_state column: cannot identify dwell samples")

    d = df[df["ramp_state"] == RAMP_DWELL].copy()
    if not len(d):
        raise ValueError("no rows with ramp_state == 1 (DWELL)")

    time_col = "t_rel_s" if "t_rel_s" in d.columns else "time_s"
    # Everything a model or metric may need, reduced in one pass. Missing
    # columns are skipped rather than raising, so schema-6 files still load.
    wanted = (FORCE_MEAS + FORCE_MEAS_S + MOMENT_MEAS + RAW_SENSOR +
              FORCE_REF + FORCE_REF_S + FORCE_CMD + FORCE_CMD_S +
              MOMENT_REF + MOMENT_CMD + POSE_DEV)
    channels = [c for c in wanted if c in d.columns]
    ref_cols = []

    out: List[dict] = []
    for idx, g in d.groupby("pattern_index", sort=True):
        g = g.sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy()
        t0, t1 = t[0], t[-1]
        cut = t0 + settle_frac * (t1 - t0)
        w = g[t >= cut]
        n_raw = len(w)

        # Without sensor_msg_count (legacy files) duplicate rows cannot be
        # identified, so n_eff overstates the independent sample count and the
        # within-dwell SD is biased low.
        if dedupe and "sensor_msg_count" in w.columns:
            w = w.drop_duplicates(subset="sensor_msg_count", keep="last")
        n_eff = len(w)

        rec: dict = {
            "pattern_index": int(idx),
            "split": g["split"].iloc[0],
            "family": g["family"].iloc[0],
            "pattern_label": g["pattern_label"].iloc[0],
            "series_id": int(g["series_id"].iloc[0]) if "series_id" in g else -1,
            "step_index": int(g["step_index"].iloc[0]) if "step_index" in g else -1,
            "skipped": int(g["skipped"].iloc[0]) if "skipped" in g else 0,
            "t_start_s": float(t0),
            "t_mid_s": float(0.5 * (t0 + t1)),
            "dwell_s": float(t1 - t0),
            "n_raw": int(n_raw),
            "n_eff": int(n_eff),
        }

        if "v_mag_mps" in w.columns and n_eff:
            rec["v_mag_max_mps"] = float(pd.to_numeric(w["v_mag_mps"]).max())
        if "sensor_age_s" in w.columns and n_eff:
            rec["sensor_age_max_s"] = float(pd.to_numeric(w["sensor_age_s"]).max())
        if "kp_scale" in w.columns and n_eff:
            rec["kp_scale"] = float(pd.to_numeric(w["kp_scale"]).mean())

        if n_eff < min_samples:
            rec["quality"] = "SHORT"
        elif rec.get("sensor_age_max_s", 0.0) > 0.1:
            rec["quality"] = "STALE"
        elif rec["skipped"]:
            rec["quality"] = "SKIPPED"
        else:
            rec["quality"] = "OK"

        for c in channels + ref_cols:
            v = pd.to_numeric(w[c], errors="coerce").to_numpy() if n_eff else np.array([])
            rec[f"{c}_mean"] = float(np.nanmean(v)) if v.size else np.nan
            rec[f"{c}_sd"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else np.nan

        out.append(rec)

    dwells = pd.DataFrame(out).sort_values("pattern_index").reset_index(drop=True)

    # Commanded resultant, useful for staircase level matching.
    if all(f"{c}_mean" in dwells.columns for c in FORCE_CMD):
        cmd = dwells[[f"{c}_mean" for c in FORCE_CMD]].to_numpy()
        dwells["cmd_resultant_N"] = np.linalg.norm(cmd, axis=1)
        amp_f = meta_float(meta or {}, "amp_f_N", np.nan)
        dwells["cmd_level"] = dwells["cmd_resultant_N"] / amp_f if amp_f else np.nan

    return dwells


def safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_.")
    return name or "run"


def _first_match(folder: Path, patterns: Sequence[str]) -> Optional[str]:
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(sorted(folder.glob(pat)))
    # Prefer the shortest/base filename when Windows has added "(2)" etc.
    matches = sorted(set(matches), key=lambda p: (len(p.name), p.name.lower()))
    return str(matches[0]) if matches else None


def discover_campaign(root: str | Path, run_names: Sequence[str]) -> List[dict]:
    """Find the ground-truth, alignment and dead-weight files of each run folder."""
    rootp = Path(root)
    rows: List[dict] = []
    for name in run_names:
        d = rootp / name
        row = {"run_id": name, "folder": str(d), "exists": d.is_dir()}
        if d.is_dir():
            row["ground_truth"] = _first_match(d, ["ground_truth.csv", "ground_truth*.csv", "*ground*truth*.csv"])
            row["alignment"] = _first_match(d, ["alignment_log.csv", "alignment_log*.csv", "*alignment*.csv"])
            row["weight"] = _first_match(d, ["weight_validation.csv", "weight_validation*.csv", "*weight*validation*.csv"])
        else:
            row.update({"ground_truth": None, "alignment": None, "weight": None})
        rows.append(row)
    return rows


def scan_text_status(path: Optional[str]) -> dict:
    out = {
        "alignment_file_present": bool(path and os.path.isfile(path)),
        "alignment_failed": False,
        "failure_hits": "",
    }
    if not out["alignment_file_present"]:
        return out
    hits: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                low = line.lower()
                if "alignment_failed" in low or "alignment failed" in low:
                    out["alignment_failed"] = True
                    if len(hits) < 6:
                        hits.append(f"L{lineno}: {line.strip()[:180]}")
                # A generic event field containing failed is retained as a warning.
                elif ",failed," in low or low.rstrip().endswith(",failed"):
                    out["alignment_failed"] = True
                    if len(hits) < 6:
                        hits.append(f"L{lineno}: {line.strip()[:180]}")
    except OSError:
        pass
    out["failure_hits"] = " | ".join(hits)
    return out


def metadata_contains_failure(meta: Dict[str, str]) -> bool:
    return any("fail" in str(v).lower() for v in meta.values())


def load_signature(dw: pd.DataFrame) -> Tuple[tuple, ...]:
    q = dw[(dw["quality"] == "OK") & (dw["split"].isin(["test_A", "test_B"]))].copy()
    cols = ["split", "pattern_label", "pattern_index"]
    for c in FORCE_REF:
        mc = f"{c}_mean"
        if mc in q.columns:
            cols.append(mc)
    sig = []
    for _, r in q.sort_values(["split", "pattern_index"]).iterrows():
        item = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                item.append(round(float(v), 8))
            else:
                item.append(str(v))
        sig.append(tuple(item))
    return tuple(sig)


def moment_range_audit(df: pd.DataFrame, dw: pd.DataFrame, threshold_Nmm: float) -> dict:
    arr = df[MOMENT_MEAS].to_numpy(dtype=float) if all(c in df.columns for c in MOMENT_MEAS) else np.empty((0, 3))
    if len(arr):
        rowmax = np.nanmax(np.abs(arr), axis=1)
        maxs = np.nanmax(np.abs(arr), axis=0)
        n_rows = int(np.sum(rowmax >= threshold_Nmm))
        frac = float(np.mean(rowmax >= threshold_Nmm))
    else:
        maxs = np.full(3, np.nan); n_rows = 0; frac = np.nan
    dcols = [f"{c}_mean" for c in MOMENT_MEAS]
    if all(c in dw.columns for c in dcols) and len(dw):
        dm = dw[dcols].to_numpy(dtype=float)
        dmax = np.nanmax(np.abs(dm), axis=1)
        n_dw = int(np.sum(dmax >= threshold_Nmm))
        max_dwell = float(np.nanmax(dmax))
    else:
        n_dw = 0; max_dwell = np.nan
    return {
        "moment_max_Tx_Nmm": float(maxs[0]),
        "moment_max_Ty_Nmm": float(maxs[1]),
        "moment_max_Tz_Nmm": float(maxs[2]),
        "moment_max_any_Nmm": float(np.nanmax(maxs)),
        "rows_over_moment_threshold": n_rows,
        "row_fraction_over_moment_threshold": frac,
        "dwells_over_moment_threshold": n_dw,
        "max_dwell_moment_Nmm": max_dwell,
        "moment_threshold_Nmm": float(threshold_Nmm),
        "moment_threshold_crossed": bool(n_rows > 0),
    }
