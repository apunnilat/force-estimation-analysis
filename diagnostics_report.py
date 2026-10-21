#!/usr/bin/env python3
"""Diagnostics report: hysteresis, cross-axis coupling, drift and compliance.

Usage
-----
    python diagnostics_report.py
    python diagnostics_report.py --config configs/analysis.yaml --out results/thesis_diagnostics

Uses the same settings file as results_report.py, so the models are fitted
with the same settings as in the Results.

What it does
------------
For each run listed in ``data.runs`` it fits the eight models (as in the
Results) and computes six diagnostics: hysteresis of the staircases, the force
transmission (coupling) matrix, force-to-moment coupling, anchor drift,
apparent EI from the measured displacement, and residual versus load /
delivered-versus-predicted parity for every model. See forcecal/diagnostics.py
for the interpretation limits.

Output (in --out)
-----------------
    diagnostics_report.md     report text with all numbers
    tables/*.csv              all diagnostic tables
    figures/*.png             all figures
    run_info.json             settings, input files, versions, caveats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from forcecal import plots
from forcecal.diagnostics import (
    aggregate_anchor,
    aggregate_ei,
    aggregate_hysteresis,
    aggregate_matrix_table,
    aggregate_residual_fit,
    anchor_table,
    apparent_ei,
    hysteresis_table,
    load_campaign,
    parity_summary,
    prediction_rows,
    residual_load_fit,
    transmission_matrices,
)
from forcecal.models import Geometry
from forcecal.report import write_diagnostics_report, write_run_info
from forcecal.settings import load_settings

REPO = Path(__file__).resolve().parent

CAVEATS = [
    "force crosstalk matrix is a system transmission matrix, not an intrinsic sensor-only calibration",
    "apparent EI uses Lambda.7 EE displacement and may include adapter/device/load-path compliance",
    "predictive models are fitted separately per run; Test A/B are not used for fitting",
    "sensor moments are interpreted as Nmm",
]


def concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the diagnostics report from the repeated runs.")
    ap.add_argument("--config", default=str(REPO / "configs" / "analysis.yaml"),
                    help="settings file (default: configs/analysis.yaml)")
    ap.add_argument("--out", default=str(REPO / "results" / "thesis_diagnostics"),
                    help="output folder (default: results/thesis_diagnostics)")
    args = ap.parse_args(argv)

    s = load_settings(args.config)
    plots.configure_figures(s.figures.dpi)
    try:
        records, manifest = load_campaign(s)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    fig_dir, table_dir = out / "figures", out / "tables"
    for d in (out, fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)
    geo = Geometry.from_config(s.object_name, s.object_geometry)

    # Per-run diagnostics.
    force, moment, polar, hyst, anchor, anchor_by_run, ei, pred = ([] for _ in range(8))
    for rec in records:
        fmat, mmat, pol = transmission_matrices(rec)
        force.append(fmat)
        moment.append(mmat)
        if pol:
            polar.append(pol)
        hyst.append(hysteresis_table(rec))
        a, a_sum = anchor_table(rec)
        anchor.append(a)
        anchor_by_run.append(a_sum)
        ei.append(apparent_ei(rec, geo, s.dwell.settle_frac))
        pred.append(prediction_rows(rec))

    force, moment, hyst = concat(force), concat(moment), concat(hyst)
    anchor, anchor_by_run, ei, pred = concat(anchor), concat(anchor_by_run), concat(ei), concat(pred)
    polar = pd.DataFrame(polar)
    rfit = residual_load_fit(pred)

    # Summaries across runs.
    force_agg = aggregate_matrix_table(force)
    moment_agg = aggregate_matrix_table(moment)
    hyst_agg = aggregate_hysteresis(hyst)
    anchor_agg = aggregate_anchor(anchor_by_run)
    residual_agg = aggregate_residual_fit(rfit)
    ei_agg = aggregate_ei(ei)
    parity = parity_summary(pred)

    tables: Dict[str, pd.DataFrame] = {
        "campaign_manifest.csv": manifest,
        "force_transmission_matrix_by_run.csv": force,
        "force_transmission_matrix_summary.csv": force_agg,
        "force_to_moment_coupling_by_run.csv": moment,
        "force_to_moment_coupling_summary.csv": moment_agg,
        "force_matrix_polar_decomposition.csv": polar,
        "hysteresis_matched_levels.csv": hyst,
        "hysteresis_summary.csv": hyst_agg,
        "anchor_drift_points.csv": anchor,
        "anchor_drift_by_run.csv": anchor_by_run,
        "anchor_drift_summary.csv": anchor_agg,
        "apparent_EI_by_run.csv": ei,
        "apparent_EI_summary.csv": ei_agg,
        "test_dwell_predictions.csv": pred,
        "residual_vs_load_fit_by_run.csv": rfit,
        "residual_vs_load_summary.csv": residual_agg,
        "delivered_vs_predicted_summary.csv": parity,
    }
    for name, df in tables.items():
        df.to_csv(table_dir / name, index=False)

    figures: Dict[str, object] = {
        "force_matrix": plots.plot_force_matrix(force, fig_dir),
        "moment_matrix": plots.plot_moment_matrix(moment, fig_dir),
        "hysteresis": plots.plot_hysteresis(hyst, fig_dir),
        "anchor": plots.plot_anchor_drift(anchor, fig_dir),
        "residual": plots.plot_residual_vs_load(pred, fig_dir),
        "ei": plots.plot_ei(ei, fig_dir),
        "parity": plots.plot_parity_per_model(pred, fig_dir),
    }

    report = write_diagnostics_report(out, records, manifest, force_agg, moment_agg, polar,
                                      hyst_agg, anchor_agg, residual_agg, ei_agg, parity, figures)

    inputs = [str(p) for r in records for p in (r.ground_truth, r.alignment_log, r.weight_validation) if p]
    info = write_run_info(out, "diagnostics_report.py", s, inputs, extra={
        "runs": [r.run_id for r in records],
        "report": str(report),
        "figure_files": figures,
        "table_files": list(tables),
        "caveats": CAVEATS,
    })

    print(f"\nDiagnostics report: {report}")
    print(f"Figures:  {fig_dir}")
    print(f"Tables:   {table_dir}")
    print(f"Run info: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
