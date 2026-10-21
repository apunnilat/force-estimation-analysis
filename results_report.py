#!/usr/bin/env python3
"""Results report: tables, figures and text for the Results chapter.

Usage
-----
    python results_report.py
    python results_report.py --config configs/analysis.yaml --out results/thesis_results

All settings are read from the config file (default: configs/analysis.yaml).

What it does
------------
For each run listed in ``data.runs``:
  1. load the recorded CSV and check it (schema, completeness, sensor timing);
  2. reduce every dwell to one point (dwell mean);
  3. fit the eight models on the training dwells and score them on
     Test A (accuracy) and Test B (robustness, 1.5 x training load);
  4. apply the fitted models to the dead-weight file, if present.
The runs are then combined (paired model contrasts, between-remount spread,
transfer between runs, error versus load).

Output (in --out)
-----------------
    results_skeleton.md, results_skeleton.tex   report text with all numbers
    tables/*.csv                                all result tables
    figures/*.png, figures/*.pdf                all figures
    run_info.json                               settings, input files, versions
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from forcecal import plots
from forcecal.campaign import (
    between_remount_spread,
    count_best_models,
    descriptive_error_load_fit,
    error_vs_load_rows,
    model_metric_rows,
    model_summary_from_channels,
    pa0_decomposition,
    paired_contrast,
    repeatability_raw_summary,
    transfer_from_records,
    worst_case_rows,
)
from forcecal.constants import MODEL_ORDER
from forcecal.data import (
    audit_run,
    discover_campaign,
    load_run,
    load_signature,
    load_weight_run,
    meta_float,
    metadata_contains_failure,
    moment_range_audit,
    reduce_dwells,
    scan_text_status,
)
from forcecal.evaluation import channel_metrics_from_record, run_model_comparison, weight_validation
from forcecal.metrics import iso376_metrics, noise_ceiling, zero_drift
from forcecal.models import Geometry
from forcecal.report import write_results_report, write_run_info
from forcecal.settings import Settings, load_settings

REPO = Path(__file__).resolve().parent

# Paired model comparisons reported in the Results (model a minus model b).
MODEL_CONTRASTS = (("PA0", "P0"), ("D2", "PA0"), ("P2", "P1"), ("PA2", "PA1"), ("PA1", "PA0"))


def analyse_run(item: pd.Series, s: Settings, geo: Geometry) -> dict:
    """Analyse one run. Returns its record and the rows it adds to the campaign tables."""
    run_id, gt = str(item.run_id), str(item.ground_truth)
    print(f"\n=== {run_id}: {gt} ===")
    df, meta = load_run(gt)
    audit = audit_run(df, meta)
    dw = reduce_dwells(df, meta, settle_frac=s.dwell.settle_frac, dedupe=True)
    res = run_model_comparison(
        dw, geo, s.models,
        val_frac=s.evaluation.val_frac, seed=s.evaluation.seed,
        nominal_amplitude_N=s.reference_values.nominal_amplitude_N)

    status = scan_text_status(item.alignment if isinstance(item.alignment, str) else None)
    alignment_failed = bool(status["alignment_failed"] or metadata_contains_failure(meta))
    moment_audit = moment_range_audit(df, dw, s.thresholds.moment_overflow_Nmm)
    skipped_phases = str(meta.get("skipped_phases", ""))

    rec = {"group": "instrument", "run_id": run_id, "path": gt, "meta": meta, "audit": audit,
           "dwells": dw, "iso": iso376_metrics(dw, meta), "zero": zero_drift(df),
           "floors": noise_ceiling(dw), "model_res": res, "weight": None,
           "test_signature": load_signature(dw)}
    out = {"record": rec, "weight_rows": [], "polar": None,
           "force_block": pd.DataFrame(), "moment_block": pd.DataFrame()}

    # Independent dead-weight check (optional file in the run folder).
    wpath = item.weight if isinstance(item.weight, str) else None
    if wpath and os.path.isfile(wpath):
        try:
            wdf, wmeta = load_weight_run(wpath)
            wres = weight_validation(wdf, wmeta, settle_frac=s.dwell.settle_frac,
                                     geo=geo, models=res["models"])
            rec["weight"] = wres
            raw = wres.get("raw", {})
            if raw:
                out["weight_rows"].append({
                    "run_id": run_id, "model": "sensor_raw",
                    "expected_force_N": wres.get("expected_force_N", np.nan),
                    "magnitude_N": raw.get("magnitude_N", np.nan),
                    "absolute_error_N": raw.get("absolute_error_N", np.nan),
                    "rel_error_pct": raw.get("rel_error_pct", np.nan)})
            mm = wres.get("model_results")
            if isinstance(mm, pd.DataFrame) and len(mm):
                for _, z in mm.iterrows():
                    row = z.to_dict()
                    row["run_id"] = run_id
                    out["weight_rows"].append(row)
        except Exception as exc:
            print(f"  dead-weight validation skipped: {exc}")

    # Decomposition of the fitted PA0 force matrix.
    polar, force_block, moment_block = pa0_decomposition(run_id, res, s.reference_values.known_fy_gain)
    out.update(polar=polar or None, force_block=force_block, moment_block=moment_block)
    out["worst"] = worst_case_rows(rec)
    out["err_load"] = error_vs_load_rows(rec)

    # Per-run numbers for the integrity table.
    cm_run = channel_metrics_from_record(rec)

    def pooled_rmse(model: str, evaluation: str) -> float:
        q = cm_run[(cm_run.model == model) & (cm_run.evaluation == evaluation)
                   & (cm_run.force_channel == "Pooled")]
        return float(q.iloc[0]["rmse_N"]) if len(q) else np.nan

    pa0_a, pa0_b = pooled_rmse("PA0", "accuracy"), pooled_rmse("PA0", "robustness")
    out["audit_row"] = {
        "run_id": run_id, "pattern_seed": meta.get("pattern_seed", ""),
        "same_test_signature": True,  # set after all runs are known
        "skipped_rotation": "ROTATION" in skipped_phases,
        "alignment_failed": alignment_failed, "alignment_failure_hits": status["failure_hits"],
        **moment_audit,
        "deadweight_present": bool(wpath and os.path.isfile(wpath)),
        "testA_PA0_rmse_N": pa0_a, "testA_D2_rmse_N": pooled_rmse("D2", "accuracy"),
        "testA_PA1_rmse_N": pooled_rmse("PA1", "accuracy"),
        "testB_PA0_rmse_N": pa0_b, "testB_D2_rmse_N": pooled_rmse("D2", "robustness"),
        "testB_PA1_rmse_N": pooled_rmse("PA1", "robustness"),
        "pa0_degradation_x": pa0_b / pa0_a if pa0_a > 0 else np.nan,
        "testA_PA0_pass": bool(np.isfinite(pa0_a) and pa0_a <= s.thresholds.accuracy_N),
        "pa0_degradation_pass": bool(np.isfinite(pa0_a) and pa0_a > 0
                                     and (pa0_b / pa0_a) <= s.thresholds.robustness_factor),
        "robust_scale_f": meta_float(meta, "robust_scale_f", np.nan),
    }
    del df  # release the large raw table before the next run
    return out


def concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Results report from the repeated runs.")
    ap.add_argument("--config", default=str(REPO / "configs" / "analysis.yaml"),
                    help="settings file (default: configs/analysis.yaml)")
    ap.add_argument("--out", default=str(REPO / "results" / "thesis_results"),
                    help="output folder (default: results/thesis_results)")
    args = ap.parse_args(argv)

    s = load_settings(args.config)
    plots.configure_figures(s.figures.dpi)

    disc = pd.DataFrame(discover_campaign(s.data.root, s.data.runs))
    missing = disc[(~disc.exists) | (disc.ground_truth.isna())]
    if len(missing):
        print("WARNING: missing run folders or ground_truth files:")
        print(missing[["run_id", "folder", "exists", "ground_truth"]].to_string(index=False))
    valid = disc[disc.ground_truth.notna()].copy()
    if not len(valid):
        print(f"ERROR: no ground_truth CSV found below {s.data.root}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    geo = Geometry.from_config(s.object_name, s.object_geometry)
    runs = [analyse_run(item, s, geo) for _, item in valid.iterrows()]
    records = [r["record"] for r in runs]

    # Integrity table; flag runs whose Test A/B loads differ from the first run.
    audit_table = pd.DataFrame([r["audit_row"] for r in runs])
    reference_signature = records[0]["test_signature"]
    for r in records:
        audit_table.loc[audit_table.run_id == r["run_id"], "same_test_signature"] = (
            r["test_signature"] == reference_signature)

    cm = model_metric_rows(records)
    model_summary = model_summary_from_channels(cm)
    err_load = concat([r["err_load"] for r in runs])
    paired = pd.DataFrame([paired_contrast(cm, ev, a, b, s.evaluation.n_boot)
                           for ev in ("accuracy", "robustness") for a, b in MODEL_CONTRASTS])
    best = pd.concat([count_best_models(cm, "accuracy", MODEL_ORDER),
                      count_best_models(cm, "robustness", MODEL_ORDER)], ignore_index=True)
    repeat_raw, repeat_summary = repeatability_raw_summary(records)
    remount_detail, remount_summary = between_remount_spread(records)
    transfer, transfer_by_source, transfer_summary = transfer_from_records(
        records, s.evaluation.n_boot_transfer)

    report = write_results_report(
        args.out, records, disc, audit_table, cm, model_summary,
        pd.DataFrame([r["polar"] for r in runs if r["polar"]]),
        concat([r["force_block"] for r in runs]), concat([r["moment_block"] for r in runs]),
        paired, best, repeat_raw, repeat_summary, remount_detail, remount_summary,
        transfer, transfer_by_source, transfer_summary,
        concat([r["worst"] for r in runs]), err_load, descriptive_error_load_fit(err_load),
        pd.DataFrame([w for r in runs for w in r["weight_rows"]]), geo,
        s.thresholds.accuracy_N, s.thresholds.robustness_factor,
        s.reference_values.datasheet_accuracy_N, s.reference_values.known_fy_gain,
        s.thresholds.moment_overflow_Nmm)

    inputs = [p for col in ("ground_truth", "alignment", "weight")
              for p in valid[col].tolist() if isinstance(p, str)]
    info = write_run_info(args.out, "results_report.py", s, inputs)

    print(f"\nResults report: {report}")
    print(f"LaTeX version:  {os.path.join(args.out, 'results_skeleton.tex')}")
    print(f"Figures:        {os.path.join(args.out, 'figures')}")
    print(f"Tables:         {os.path.join(args.out, 'tables')}")
    print(f"Run info:       {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
