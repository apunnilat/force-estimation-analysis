"""Writing the reports: tables, Markdown and LaTeX text, and run_info.json.

* ``write_results_report`` writes the Results report (results_skeleton.md/.tex),
  its tables and figures.
* ``write_diagnostics_report`` writes diagnostics_report.md.
* ``write_run_info`` records what produced an output folder: settings, input
  files with checksums, git commit and package versions.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import model_name, MODEL_ORDER
from .diagnostics import RunRecord
from .metrics import format_number
from .models import Geometry
from .plots import (
    plot_abs_vs_normalized,
    plot_all_models_accuracy,
    plot_channel_observability,
    plot_compliance_delta,
    plot_deadweight,
    plot_degradation,
    plot_error_vs_load,
    plot_family_accuracy,
    plot_paired_effect,
    plot_polar,
    plot_repeatability,
    plot_rmse_across_runs,
    plot_transfer_heatmap,
)

if TYPE_CHECKING:
    from .settings import Settings


def _markdown_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, digits: int = 3) -> str:
    if df is None or not len(df): return "_(no data)_\n"
    d=df[list(cols)].copy() if cols else df.copy()
    def cv(v):
        if isinstance(v,(float,np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.{digits}f}"
        return str(v)
    lines=["| "+" | ".join(map(str,d.columns))+" |","|"+"|".join("---" for _ in d.columns)+"|"]
    for _,r in d.iterrows(): lines.append("| "+" | ".join(cv(v) for v in r)+" |")
    return "\n".join(lines)+"\n"


def _write_csv(df: pd.DataFrame, table_dir: str, name: str) -> Optional[str]:
    if df is None or not len(df): return None
    os.makedirs(table_dir,exist_ok=True)
    p=os.path.join(table_dir,name+".csv"); df.to_csv(p,index=False); return p


def _relfig(name: str) -> str:
    return f"figures/{name}.png"


def _tex_escape(s: object) -> str:
    t=str(s)
    repl={"\\":"\\textbackslash{}","_":"\\_","%":"\\%","&":"\\&","#":"\\#"}
    for a,b in repl.items(): t=t.replace(a,b)
    return t


def _tex_figure(name: str, caption: str, label: str) -> str:
    return ("\\begin{figure}[tb]\n\\centering\n"
            f"\\includegraphics[width=0.82\\linewidth]{{figures/{name}.pdf}}\n"
            f"\\caption{{{_tex_escape(caption)}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n")


def write_results_report(out_dir: str, records: Sequence[dict], discovery: pd.DataFrame,
                             audit_table: pd.DataFrame, cm: pd.DataFrame,
                             model_summary: pd.DataFrame, polar: pd.DataFrame,
                             force_coeff: pd.DataFrame, moment_coeff: pd.DataFrame,
                             paired: pd.DataFrame, best_counts: pd.DataFrame,
                             repeat_raw: pd.DataFrame, repeat_summary: pd.DataFrame,
                             remount_detail: pd.DataFrame, remount_summary: pd.DataFrame,
                             tr: pd.DataFrame, trsrc: pd.DataFrame, trsum: pd.DataFrame,
                             worst: pd.DataFrame, err_load: pd.DataFrame, err_fit: pd.DataFrame,
                             weight_rows: pd.DataFrame, geometry: Geometry,
                             accuracy_threshold_N: float, robustness_threshold_x: float,
                             datasheet_accuracy_N: float, known_fy_gain: float,
                             moment_threshold_Nmm: float) -> str:
    os.makedirs(out_dir,exist_ok=True)
    fig_dir=os.path.join(out_dir,"figures"); table_dir=os.path.join(out_dir,"tables")
    os.makedirs(fig_dir,exist_ok=True); os.makedirs(table_dir,exist_ok=True)

    # ---- Figures ---------------------------------------------------------
    acc_figs=plot_family_accuracy(cm,"accuracy",fig_dir,"4_2_accuracy")
    robust_figs=plot_family_accuracy(cm,"robustness",fig_dir,"4_5_robustness")
    all_acc_fig=plot_all_models_accuracy(cm,fig_dir,"4_2_accuracy_all_models")
    runs_fig=plot_rmse_across_runs(cm,fig_dir,"4_2_rmse_across_runs")
    plot_paired_effect(cm,"accuracy",["P0","PA0","PA1","D2"],fig_dir,"4_2_key_model_accuracy",
                        "Test A — baseline, affine force, affine moment and MLP")
    plot_polar(polar,fig_dir)
    plot_compliance_delta(cm,fig_dir)
    plot_channel_observability(cm,fig_dir)
    plot_repeatability(repeat_raw,fig_dir)
    for model in ("PA0","D2","PA1"):
        plot_transfer_heatmap(tr,model,fig_dir)
    plot_abs_vs_normalized(cm,fig_dir)
    plot_degradation(cm,fig_dir)
    plot_error_vs_load(err_load,fig_dir)
    plot_deadweight(weight_rows,fig_dir)

    # ---- Tables ----------------------------------------------------------
    _write_csv(discovery,table_dir,"run_discovery")
    _write_csv(audit_table,table_dir,"run_integrity_checks")
    _write_csv(cm,table_dir,"all_force_channel_metrics")
    # Thesis-facing force-channel tables: one CSV per model and evaluation, plus
    # a pooled summary for direct model comparison.
    if len(cm):
        chroot=os.path.join(table_dir,"force_channel_tables")
        for ev in ("accuracy","robustness","repeatability"):
            evd=cm[cm.evaluation==ev]
            if not len(evd): continue
            evdir=os.path.join(chroot,ev); os.makedirs(evdir,exist_ok=True)
            for model in sorted(evd.model.unique(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999):
                q=evd[evd.model==model].copy()
                # Equal-run medians per force channel for the thesis table.
                tab=q.groupby("force_channel",as_index=False).agg(
                    rmse_N=("rmse_N","median"),
                    normalized_rmse_pct=("normalized_rmse_pct","median"),
                    standard_deviation_N=("standard_deviation_N","median"),
                    r2=("r2","median"),
                    n_runs=("run_id","nunique"))
                tab.to_csv(os.path.join(evdir,f"{model}_force_channels.csv"),index=False)
            pooled=evd[evd.force_channel=="Pooled"].groupby(["model","model_name","model_family"],as_index=False).agg(
                rmse_N=("rmse_N","median"),
                normalized_rmse_pct=("normalized_rmse_pct","median"),
                standard_deviation_N=("standard_deviation_N","median"),
                r2=("r2","median"),
                n_runs=("run_id","nunique"))
            pooled.to_csv(os.path.join(evdir,"all_models_pooled_channel.csv"),index=False)
    _write_csv(model_summary,table_dir,"model_summary")
    _write_csv(polar,table_dir,"pa0_polar_decomposition")
    _write_csv(force_coeff,table_dir,"pa0_force_block_coefficients")
    _write_csv(moment_coeff,table_dir,"pa0_moment_block_coefficients")
    _write_csv(paired,table_dir,"paired_model_contrasts")
    _write_csv(best_counts,table_dir,"best_model_counts")
    _write_csv(repeat_raw,table_dir,"repeatability_by_run")
    _write_csv(repeat_summary,table_dir,"repeatability_summary")
    _write_csv(remount_detail,table_dir,"between_remounting_detail")
    _write_csv(remount_summary,table_dir,"between_remounting_summary")
    _write_csv(tr,table_dir,"transfer_matrix_long")
    _write_csv(trsrc,table_dir,"transfer_by_source_run")
    _write_csv(trsum,table_dir,"transfer_summary")
    _write_csv(worst,table_dir,"worst_case_per_dwell")
    _write_csv(err_load,table_dir,"error_vs_load_per_dwell")
    _write_csv(err_fit,table_dir,"error_vs_load_linear_fit")
    _write_csv(weight_rows,table_dir,"deadweight_validation")

    # Convenience lookups.
    pooled=cm[cm.force_channel=="Pooled"] if len(cm) else pd.DataFrame()
    def med(model,ev,col="rmse_N"):
        q=pooled[(pooled.model==model)&(pooled.evaluation==ev)]
        return float(np.nanmedian(q[col])) if len(q) else np.nan
    def runs_better(a,b,ev):
        q=paired[(paired.evaluation==ev)&(paired.model_a==a)&(paired.model_b==b)]
        if len(q) and np.isfinite(q.iloc[0].get("a_better_count", np.nan)) and np.isfinite(q.iloc[0].get("n_runs", np.nan)):
            return int(q.iloc[0].a_better_count), int(q.iloc[0].n_runs)
        q=paired[(paired.evaluation==ev)&(paired.model_a==b)&(paired.model_b==a)]
        if len(q) and np.isfinite(q.iloc[0].get("b_better_count", np.nan)) and np.isfinite(q.iloc[0].get("n_runs", np.nan)):
            return int(q.iloc[0].b_better_count), int(q.iloc[0].n_runs)
        return 0,0
    def degr(model):
        a=pooled[(pooled.model==model)&(pooled.evaluation=="accuracy")][["run_id","rmse_N"]].rename(columns={"rmse_N":"A"})
        b=pooled[(pooled.model==model)&(pooled.evaluation=="robustness")][["run_id","rmse_N"]].rename(columns={"rmse_N":"B"})
        z=a.merge(b,on="run_id"); v=(z.B/z.A).to_numpy(float) if len(z) else np.array([])
        return float(np.nanmedian(v)) if len(v) else np.nan
    def fmt(x,n=3): return "n/a" if not np.isfinite(x) else f"{x:.{n}f}"

    n_runs=len(records)
    seeds=[str(r["meta"].get("pattern_seed","")) for r in records]
    same_seed=len(set(seeds))==1 and n_runs>0
    sigs=[r.get("test_signature") for r in records]
    same_loads=bool(sigs) and all(s==sigs[0] for s in sigs[1:])
    skipped_rotation=sum("ROTATION" in str(r["meta"].get("skipped_phases","")) for r in records)
    robust_scales=pd.to_numeric(audit_table.get("robust_scale_f",pd.Series(dtype=float)),errors="coerce").dropna().to_numpy(float)
    same_robust_scale=bool(len(robust_scales) and np.allclose(robust_scales,robust_scales[0]))
    robust_scale_value=float(robust_scales[0]) if same_robust_scale else np.nan
    failed_runs=audit_table[audit_table.alignment_failed==True].run_id.tolist() if len(audit_table) else []
    overflow_runs=audit_table[audit_table.moment_threshold_crossed==True].run_id.tolist() if len(audit_table) else []

    p0A,pa0A,pa1A,d2A=[med(m,"accuracy") for m in ("P0","PA0","PA1","D2")]
    pa0B,pa1B,d2B=[med(m,"robustness") for m in ("PA0","PA1","D2")]
    p0sd=med("P0","accuracy","standard_deviation_N")
    pa0nA,pa0nB=med("PA0","accuracy","normalized_rmse_pct"),med("PA0","robustness","normalized_rmse_pct")
    d2nA,d2nB=med("D2","accuracy","normalized_rmse_pct"),med("D2","robustness","normalized_rmse_pct")
    pa0_better=runs_better("PA0","P0","accuracy")
    d2_better_pa0=runs_better("D2","PA0","accuracy")
    d2_better_B=runs_better("D2","PA0","robustness")
    # Median paired differences, read from the paired-contrast table.
    def contrast(evaluation, a, b):
        if not len(paired):
            return {}
        q = paired[(paired.evaluation == evaluation) & (paired.model_a == a) & (paired.model_b == b)]
        return q.iloc[0].to_dict() if len(q) else {}
    p2p1A=contrast("accuracy","P2","P1")
    p2p1B=contrast("robustness","P2","P1")
    pa2pa1A=contrast("accuracy","PA2","PA1")
    pa2pa1B=contrast("robustness","PA2","PA1")
    r6=geometry.deformed_lever_vector(np.array([0.0,6.0,0.0]))
    r0=geometry.sensor_to_tip_length

    # Transfer lookups.
    def transfer_model(model):
        q=trsum[trsum.model==model] if len(trsum) else pd.DataFrame()
        return q.iloc[0].to_dict() if len(q) else {}
    tr_pa0=transfer_model("PA0")

    # Worst cases.
    worst_summary=pd.DataFrame()
    if len(worst):
        idx=worst.groupby(["model","evaluation"])["vector_error_N"].idxmax()
        worst_summary=worst.loc[idx].sort_values(["evaluation","vector_error_N"],ascending=[True,False])
        _write_csv(worst_summary,table_dir,"worst_case_summary")

    # ---- Markdown skeleton ---------------------------------------------
    md=[]
    md.append("# Thesis Results Skeleton — Eight Instrument Remountings\n")
    md.append("> Automatically generated from the raw run folders. Numerical statements in Sections 4.1–4.5 are data-derived. Interpretive language is intentionally kept in the Discussion hand-off at the end.\n")
    md.append(f"**Campaign:** {n_runs} discovered runs. Main accuracy metric: pooled per-axis RMSE, first computed within a run and then summarized across runs. Absolute RMSE is primary; R² is secondary because channel variance differs strongly between test sets.\n")

    md.append("## 4.1 Dataset and integrity checks\n")
    md.append(f"All runs used the same pattern seed: **{same_seed}** ({', '.join(sorted(set(seeds))) if seeds else 'n/a'}). The complete Test A + Test B command signature was identical across runs: **{same_loads}**. Therefore, between-run variation is not caused by different selected test loads.\n")
    md.append(f"The ROTATION alignment stage is listed as skipped in **{skipped_rotation}/{n_runs}** runs. Explicit ALIGNMENT_FAILED markers were detected in: **{', '.join(failed_runs) if failed_runs else 'none'}**.\n")
    md.append(f"The robustness-scale metadata were identical across runs: **{same_robust_scale}**" + (f", with robust_scale_f = **{robust_scale_value:.2f}×**." if np.isfinite(robust_scale_value) else ".") + "\n")
    theoretical_moment_Nmm=6.0*geometry.sensor_to_tip_length_mm
    md.append(f"The firmware-risk moment threshold was set to **{moment_threshold_Nmm:.0f} Nmm**. A 6 N transverse load acting through the nominal {geometry.sensor_to_tip_length_mm:.0f} mm lever corresponds to **{theoretical_moment_Nmm:.0f} Nmm**. The recorded threshold was crossed in **{len(overflow_runs)}/{n_runs}** runs ({', '.join(overflow_runs) if overflow_runs else 'none'}). A threshold crossing is reported as a data-integrity warning; this script does not infer a sign flip merely from exceeding the threshold.\n")
    md.append("\n**Run-level audit table**\n")
    md.append(_markdown_table(audit_table,[c for c in ["run_id","pattern_seed","same_test_signature","alignment_failed","skipped_rotation","moment_max_any_Nmm","rows_over_moment_threshold","deadweight_present","testA_PA0_rmse_N","testA_PA0_pass","testA_D2_rmse_N","testB_PA0_rmse_N","testB_D2_rmse_N","pa0_degradation_x","pa0_degradation_pass"] if c in audit_table],3))
    if "test_53" in set(audit_table.run_id):
        r53=audit_table[audit_table.run_id=="test_53"].iloc[0]
        md.append(f"**Specific check — test_53.** ALIGNMENT_FAILED = **{bool(r53.alignment_failed)}**; Test A PA0 RMSE = **{fmt(float(r53.testA_PA0_rmse_N))} N** (0.5 N pass = **{bool(r53.testA_PA0_pass)}**); Test B/A PA0 degradation = **{fmt(float(r53.pa0_degradation_x),2)}×** (2× pass = **{bool(r53.pa0_degradation_pass)}**). This directly tests whether the failed alignment run is also the requirement outlier.\n")

    md.append("## 4.2 Force reconstruction and model comparison\n")
    md.append("### 4.2.1 Overall model ranking\n")
    d2pa0=paired[(paired.evaluation=="accuracy")&(paired.model_a=="D2")&(paired.model_b=="PA0")]
    if len(d2pa0):
        zz=d2pa0.iloc[0]; d2delta=float(zz.median_a_minus_b_N); d2lo=float(zz.ci95_median_lo_N); d2hi=float(zz.ci95_median_hi_N)
    else:
        d2delta=d2lo=d2hi=np.nan
    md.append(f"Across the {n_runs} remountings, the median Test A pooled RMSE was **{fmt(p0A)} N** for P0, **{fmt(pa0A)} N** for PA0, **{fmt(pa1A)} N** for PA1, and **{fmt(d2A)} N** for D2. PA0 outperformed P0 in **{pa0_better[0]}/{pa0_better[1]}** paired runs. D2 outperformed PA0 in **{d2_better_pa0[0]}/{d2_better_pa0[1]}** paired runs on Test A; the median paired D2−PA0 difference was **{fmt(d2delta)} N** (run-level bootstrap 95% interval **{fmt(d2lo)} to {fmt(d2hi)} N**).\n")
    if len(best_counts):
        md.append("\n**Best-model count across the compared candidates**\n")
        md.append(_markdown_table(best_counts,None,3))
    md.append(f"For P0, the median Test A residual SD was **{fmt(p0sd)} N**, compared with an RMSE of **{fmt(p0A)} N**. The nominal HEX-32 operating-point accuracy supplied for comparison is **{datasheet_accuracy_N:.2f} N**; cite the corresponding datasheet source in the thesis before using this comparison.\n")
    for nm in acc_figs: md.append(f"![Accuracy]({_relfig(nm)})\n")
    if all_acc_fig: md.append(f"![All models]({_relfig(all_acc_fig)})\n")
    if runs_fig: md.append(f"![RMSE across runs]({_relfig(runs_fig)})\n")
    md.append(f"![Key models]({_relfig('4_2_key_model_accuracy')})\n")
    md.append("\n**All-model Test A summary**\n")
    msA=model_summary[model_summary.evaluation=="accuracy"].sort_values("rmse_median_N") if len(model_summary) else pd.DataFrame()
    md.append(_markdown_table(msA,[c for c in ["family","model","model_name","n_runs","rmse_median_N","rmse_run_sd_N","normalized_rmse_median_pct","residual_sd_median_N","r2_median"] if c in msA],3))

    md.append("### 4.2.2 Affine correction and PA0 matrix decomposition\n")
    if len(polar):
        md.append(f"The 3×3 force-input block of PA0 was converted back to physical (unstandardized) coefficients and decomposed as A = R S. The median residual rotation angle was **{fmt(float(np.nanmedian(polar.rotation_angle_deg)),2)}°**. Median stretch/gain terms were Fx **{fmt(float(np.nanmedian(polar.gain_x)),3)}**, Fy **{fmt(float(np.nanmedian(polar.gain_y)),3)}**, and Fz **{fmt(float(np.nanmedian(polar.gain_z)),3)}**. The independently observed Fy gain supplied for comparison is **{known_fy_gain:.2f}**. Because PA0 also includes the three moment channels, this polar decomposition applies only to its force-input sub-block with moment features held fixed; it is a diagnostic, not a unique decomposition of the full 3×6 map.\n")
        md.append(f"![Rotation]({_relfig('4_2_pa0_rotation_angle')})\n")
        md.append(f"![Gains]({_relfig('4_2_pa0_axis_gains')})\n")
        md.append(_markdown_table(polar,[c for c in ["run_id","rotation_angle_deg","det_rotation","gain_x","gain_y","gain_z","gain_y_minus_known","force_block_offdiag_fro","moment_block_fro_N_per_Nmm","intercept_norm_N"] if c in polar],3))
    md.append("**TODO for the thesis:** connect this diagnostic to the independent Fy gain / Tx cross-talk characterization and verify the controller-side `a_ee_to_sensor` implementation from source code. The CSV itself cannot prove that a source-code matrix was an identity placeholder.\n")

    md.append("### 4.2.3 Effect of shaft compliance\n")
    md.append(f"At a 6 N transverse load, the magnitude of the assumed deformed lever x-component changes from **{1000*r0:.1f} mm** to **{1000*abs(r6[0]):.1f} mm** under the handbook-EI model. The paired compliant-minus-fixed Test A RMSE difference is **{fmt(float(p2p1A.get('median_a_minus_b_N',np.nan))*1000,1)} mN** for P2−P1 and **{fmt(float(pa2pa1A.get('median_a_minus_b_N',np.nan))*1000,1)} mN** for PA2−PA1. On Test B the corresponding differences are **{fmt(float(p2p1B.get('median_a_minus_b_N',np.nan))*1000,1)} mN** and **{fmt(float(pa2pa1B.get('median_a_minus_b_N',np.nan))*1000,1)} mN**. Positive values mean the compliant variant is worse.\n")
    md.append(f"![Compliance effect]({_relfig('4_2_compliance_effect')})\n")

    md.append("### 4.2.4 Axial observability and direct-Fx moment models\n")
    qobs=cm[(cm.evaluation=="accuracy")&(cm.model.isin(["PA1","PA2"]))&(cm.force_channel.isin(["Fx","Fy","Fz"]))]
    obs_summary=qobs.groupby(["model","force_channel"])[["rmse_N","r2"]].median().reset_index() if len(qobs) else pd.DataFrame()
    md.append("Bending moments cannot observe force parallel to the shaft because r ∥ F produces no bending moment. In these recordings the sensor-to-tip lever is along -x_EE, so P1 and P2 take Fx_EE directly from the sensor force channel by construction; Fy_EE is reconstructed from -Tz_EE/L and Fz_EE from +Ty_EE/L. PA1 and PA2 apply the same six-input affine calibration to these physically completed force estimates plus the measured sensor moments.\n")
    md.append(_markdown_table(obs_summary,[c for c in ["model","force_channel","rmse_N","r2"] if c in obs_summary],3))
    md.append(f"![Channel RMSE]({_relfig('4_2_channel_rmse')})\n")
    md.append(f"![Channel R2]({_relfig('4_2_channel_r2')})\n")

    md.append("### 4.2.5 Worst-case held-out dwell errors\n")
    if len(worst_summary):
        md.append("The table below reports the single largest force-vector error per model and held-out test set, preserving the run and pattern identifier. This fills the worst-case-per-dwell item that was missing from the previous Results draft.\n")
        md.append(_markdown_table(worst_summary,[c for c in ["model","evaluation","run_id","pattern_index","pattern_label","vector_error_N","max_component_error_N","err_Fx_N","err_Fy_N","err_Fz_N"] if c in worst_summary],3))

    md.append("## 4.3 Independent dead-weight validation\n")
    if len(weight_rows):
        md.append(f"Independent dead-weight files were found for **{weight_rows.run_id.nunique()}/{n_runs}** runs. These measurements do not use the Lambda.7 reference and are therefore reported separately from Test A/Test B.\n")
        md.append(_markdown_table(weight_rows,[c for c in ["run_id","model","expected_force_N","magnitude_N","absolute_error_N","rel_error_pct"] if c in weight_rows],3))
        md.append(f"![Dead weight]({_relfig('4_3_deadweight_validation')})\n")
    else:
        md.append("**Missing:** no usable dead-weight validation file was found in the discovered run folders. Do not describe Lambda.7-referenced Test A/B numbers as traceable absolute accuracy.\n")

    md.append("## 4.4 Repeatability and remounting\n")
    md.append("### 4.4.1 Within-session repeated-dwell scatter\n")
    md.append(_markdown_table(repeat_summary,[c for c in ["force_channel","n_runs","median_N","run_sd_N","min_N","max_N"] if c in repeat_summary],3))
    if len(repeat_raw):
        piv=repeat_raw.pivot(index="run_id",columns="force_channel",values="repeatability_sd_N")
        use=[c for c in ["Fx","Fy","Fz"] if c in piv.columns]
        pooled_rep=np.sqrt(np.mean(np.square(piv[use].to_numpy(float)),axis=1)) if use else np.array([])
        pooled_rep_med=float(np.nanmedian(pooled_rep)) if len(pooled_rep) else np.nan
        smap={r.force_channel:r.median_N for _,r in repeat_summary.iterrows()}
        fx=float(smap.get("Fx",np.nan)); fy=float(smap.get("Fy",np.nan)); fz=float(smap.get("Fz",np.nan))
        md.append(f"The median pooled component repeatability scale (root-mean-square of the three channel SDs within each run) is **{fmt(pooled_rep_med)} N**. The axial Fx SD is **{fmt(fx)} N**, versus transverse Fy **{fmt(fy)} N** and Fz **{fmt(fz)} N**.\n")
    md.append(f"![Repeatability]({_relfig('4_4_repeatability_channels')})\n")
    md.append("### 4.4.2 Between-remounting spread\n")
    if len(remount_summary): md.append(_markdown_table(remount_summary,None,3))
    md.append("### 4.4.3 Calibration transfer without refitting\n")
    if len(trsum):
        md.append(_markdown_table(trsum[trsum.model.isin(["PA0","PA1","D1","D2"])].sort_values("transfer_penalty_median_x"),None,3))
        for m in ("PA0","D2","PA1"):
            if len(tr[tr.model==m]): md.append(f"![Transfer {m}]({_relfig('4_4_transfer_'+m)})\n")
        if tr_pa0:
            md.append(f"For PA0, the median same-run Test A RMSE is **{fmt(float(tr_pa0.get('same_run_rmse_median_N',np.nan)))} N** and the median cross-remounting RMSE is **{fmt(float(tr_pa0.get('cross_run_rmse_median_N',np.nan)))} N**, corresponding to a transfer penalty of **{fmt(float(tr_pa0.get('transfer_penalty_median_x',np.nan)),2)}×**. Relative to the declared **{accuracy_threshold_N:.2f} N** requirement, the cross-remounting result is {'above' if float(tr_pa0.get('cross_run_rmse_median_N',0))>accuracy_threshold_N else 'below'} the threshold.\n")

    md.append("## 4.5 Robustness and extrapolation\n")
    scale_txt=f"{robust_scale_value:.2f}×" if np.isfinite(robust_scale_value) else "not identical across runs"
    md.append(f"Test B uses a robustness scale of **{scale_txt}** relative to the nominal training amplitude. Median Test B pooled RMSE is PA0 **{fmt(pa0B)} N**, D2 **{fmt(d2B)} N**, and PA1 **{fmt(pa1B)} N**. D2 outperformed PA0 in **{d2_better_B[0]}/{d2_better_B[1]}** paired Test B runs.\n")
    for nm in robust_figs: md.append(f"![Robustness]({_relfig(nm)})\n")
    rbch=cm[(cm.evaluation=="robustness")&(cm.model.isin(["PA0","PA1","D2"]))&(cm.force_channel.isin(["Fx","Fy","Fz"]))]
    if len(rbch):
        rbtab=rbch.groupby(["model","force_channel"])[["rmse_N","normalized_rmse_pct","standard_deviation_N","r2"]].median().reset_index()
        md.append("\n**Test B channel metrics for the key models**\n")
        md.append(_markdown_table(rbtab,None,3))
    md.append(f"Median B/A degradation is PA0 **{fmt(degr('PA0'),2)}×**, D2 **{fmt(degr('D2'),2)}×**, D1 **{fmt(degr('D1'),2)}×**, and PA1 **{fmt(degr('PA1'),2)}×** (ratio of absolute RMSE). The degradation figure plots the ratio of normalized RMSE, with 4.0 N (Test A) and 6.0 N (Test B) as normalization loads; each plotted value therefore equals the absolute RMSE ratio divided by 1.5.\n")
    md.append(f"![Degradation]({_relfig('4_5_degradation')})\n")
    md.append("### 4.5.1 Absolute versus normalized error\n")
    md.append(f"PA0 normalized RMSE changes from **{fmt(pa0nA,1)}%** on Test A to **{fmt(pa0nB,1)}%** on Test B; D2 changes from **{fmt(d2nA,1)}%** to **{fmt(d2nB,1)}%**. A decrease in normalized error while absolute RMSE increases is consistent with a non-scaling error contribution, but does not by itself identify its physical source.\n")
    md.append(f"![Absolute error]({_relfig('4_5_absolute_error')})\n")
    md.append(f"![Normalized error]({_relfig('4_5_normalized_error')})\n")
    if len(err_fit):
        ef=err_fit.groupby("model")[["error_intercept_N","error_slope_N_per_N"]].median().reset_index()
        md.append("A descriptive linear fit of per-dwell pooled-component RMS error versus load magnitude gives the following intercepts. These are descriptive intercepts, not metrological uncertainty floors.\n")
        md.append(_markdown_table(ef,None,3))
        md.append(f"![Error vs load]({_relfig('4_5_error_vs_load')})\n")

    md.append("## Discussion hand-off — do not paste into Results unchanged\n")
    md.append("The following items are the interpretation targets indicated by the feedback:\n")
    md.append("- **§5.1 / §5.3:** compare the affine and MLP rankings. If D2 remains best across runs, describe PA0 as an interpretability/complexity trade-off rather than the performance winner. D2 uses scikit-learn's default **ReLU** activation, so the trained mapping is piecewise affine; Test B is only a mild amplitude extrapolation.\n")
    md.append("- **§5.2:** use the PA0 force-block rotation/gain decomposition together with the independent Fy gain/cross-talk evidence and the alignment audit. Keep the decomposition caveat: the full PA0 map is 3×6 because moments are also predictors.\n")
    md.append("- **§5.2 / §5.4:** if P1≈P2 and PA1≈PA2 across runs, state that the tested shaft-compliance correction is second order for force reconstruction at these loads. Do not extend that claim to distal moment reconstruction, which was not independently excited.\n")
    md.append("- **§5.3:** explain that bending moments cannot observe axial force, so P1/P2 deliberately pass Fx_EE through directly while reconstructing Fy_EE=-Tz_EE/L and Fz_EE=+Ty_EE/L; PA1/PA2 then calibrate that physically completed representation.\n")
    md.append("- **§5.5 / §6.2:** use repeatability and cross-remounting transfer to justify whether recalibration is required after remounting.\n")
    md.append("- **Threshold citations:** the script treats 0.5 N and 2× as declared engineering requirements only. Add the actual source/rationale in §2.4 before presenting them as literature-backed thresholds.\n")
    md.append("- **Moment overflow:** if any run crosses the configured firmware-risk threshold, resolve that issue before relying on moment-derived model results in the thesis.\n")

    md_path=os.path.join(out_dir,"results_skeleton.md")
    with open(md_path,"w",encoding="utf-8") as f: f.write("\n".join(md))

    # ---- Minimal LaTeX skeleton with figure hooks ----------------------
    tex=[]
    tex.append("% Auto-generated thesis Results skeleton. Review wording before submission.\n")
    tex.append("\\section{Results}\n")
    tex.append("\\subsection{Dataset and integrity checks}\n")
    tex.append(f"The analysis comprised {n_runs} repeated instrument remountings. All runs used the same pattern seed ({_tex_escape(', '.join(sorted(set(seeds))))}) and identical held-out Test A and Test B load signatures: {_tex_escape(str(same_loads))}.\\par\n")
    tex.append("% Insert Table: tables/run_integrity_checks.csv\n")
    tex.append("\\subsection{Force reconstruction and model comparison}\n")
    tex.append(f"Across remountings, the median pooled Test A RMSE was {fmt(p0A)}~N for P0, {fmt(pa0A)}~N for PA0 and {fmt(d2A)}~N for D2. PA0 outperformed P0 in {pa0_better[0]}/{pa0_better[1]} paired runs.\\par\n")
    for nm in acc_figs:
        tex.append(_tex_figure(nm,"Test A pooled force RMSE across remountings.",nm))
    tex.append(_tex_figure("4_2_pa0_rotation_angle","Residual rotation angle from polar decomposition of the PA0 force-input block.","pa0_rotation"))
    tex.append(_tex_figure("4_2_pa0_axis_gains","Axis stretch terms from polar decomposition of the PA0 force-input block.","pa0_gains"))
    tex.append(_tex_figure("4_2_compliance_effect","Paired effect of the compliant lever assumption on pooled RMSE.","compliance_effect"))
    tex.append(_tex_figure("4_2_channel_rmse","Test A force-channel RMSE for the moment, hybrid and benchmark models.","channel_rmse"))
    tex.append("\\subsection{Independent dead-weight validation}\n")
    if len(weight_rows): tex.append(_tex_figure("4_3_deadweight_validation","Independent dead-weight validation.","deadweight"))
    else: tex.append("% TODO: independent dead-weight validation not available.\n")
    tex.append("\\subsection{Repeatability and remounting}\n")
    tex.append(_tex_figure("4_4_repeatability_channels","Within-session repeated-dwell standard deviation by force channel.","repeatability"))
    if len(tr) and "model" in tr.columns and len(tr[tr.model=="PA0"]):
        tex.append(_tex_figure("4_4_transfer_PA0","Cross-remounting transfer matrix for PA0.","transfer_pa0"))
    tex.append("\\subsection{Robustness outside the calibration amplitude}\n")
    for nm in robust_figs: tex.append(_tex_figure(nm,"Test B pooled force RMSE across remountings.",nm))
    tex.append(_tex_figure("4_5_degradation","Ratio of normalized RMSE, test set B / test set A, across remountings (normalization loads 4.0 N and 6.0 N).","degradation"))
    tex.append(_tex_figure("4_5_normalized_error","Normalized Test A and Test B force error.","normalized_error"))
    tex_path=os.path.join(out_dir,"results_skeleton.tex")
    with open(tex_path,"w",encoding="utf-8") as f: f.write("\n".join(tex))
    return md_path


def _markdown_table_sig(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df is None or not len(df):
        return "_No data available._"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    # Keep Markdown readable without requiring tabulate.
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if not np.isfinite(v) else f"{v:.4g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_diagnostics_report(out: Path, records: Sequence[RunRecord], manifest: pd.DataFrame,
                 force_agg: pd.DataFrame, moment_agg: pd.DataFrame,
                 polar: pd.DataFrame, hyst_agg: pd.DataFrame,
                 anchor_agg: pd.DataFrame, residual_agg: pd.DataFrame,
                 ei_agg: pd.DataFrame, parity: pd.DataFrame,
                 figures: Dict[str, object]) -> Path:
    n = len(records)
    seeds = sorted(set(str(r.meta.get("pattern_seed", "")) for r in records))
    handbook_ei = np.nan
    if len(ei_agg) and records:
        handbook_ei = records[0].model_result["geometry_summary"].loc[
            records[0].model_result["geometry_summary"]["quantity"] == "EI_handbook", "value"
        ].astype(float).iloc[0]

    lines: List[str] = []
    lines += [
        "# Diagnostic results report: hysteresis, cross-axis coupling, drift and compliance",
        "",
        f"Campaign analysed: **{n} run(s)** ({', '.join(r.run_id for r in records)}).",
        "Analysis code: `forcecal` package; the git commit and settings are recorded in `run_info.json`.",
        f"Pattern seed(s): **{', '.join(seeds)}**.",
        "",
        "> This report is designed as a Results-section skeleton. It reports measured/descriptive findings first and keeps causal attribution as a Discussion question. The forward matrix below is a system transmission matrix, not an intrinsic sensor-only crosstalk calibration.",
        "",
        "## 1. Dataset and provenance",
        "",
        _markdown_table_sig(manifest),
        "",
        "**Suggested thesis placement:** dataset integrity / beginning of the Results diagnostics subsection.",
        "",
        "## 2. Hysteresis",
        "",
        "Hysteresis is evaluated only where the same positive-axis load level was visited on the ascending and descending branches. Values are descending minus ascending at matched delivered/commanded levels. The table below aggregates absolute hysteresis across runs and matched levels.",
        "",
        _markdown_table_sig(hyst_agg),
        "",
    ]
    if figures.get("hysteresis"):
        lines += [f"![Loaded-channel hysteresis](figures/{figures['hysteresis']})", ""]
    lines += [
        "**Result-writing prompt:** state which loaded axis has the largest median/max hysteresis and whether off-axis outputs show comparable history dependence. Do not attribute the effect uniquely to collet friction; the matched-load difference contains the complete assembly/reference response.",
        "",
        "## 3. Force transmission and cross-axis coupling",
        "",
        "The forward force matrix is fitted on training dwells as `F_measured = B F_delivered + b`, excluding repeated reference dwells so that one direction is not overweighted. The fit itself runs in the EE frame (schema-7 measured channels are already mapped from S), but all quantities below are reported in the raw sensor frame (S); the shaft axis is therefore the \"Fz\"/\"Tz\" channel (x_EE, tip on the -x_EE side). An ideal correctly framed, unit-gain force path would have a force block close to the identity. Off-diagonal terms quantify cross-axis coupling of the complete setup.",
        "",
        _markdown_table_sig(force_agg),
        "",
    ]
    if figures.get("force_matrix"):
        lines += [f"![Force transmission matrix](figures/{figures['force_matrix']})", ""]
    if len(polar):
        polar_show = polar[["run_id", "rotation_angle_deg", "gain_like_x", "gain_like_y", "gain_like_z",
                            "force_matrix_residual_pooled_rmse_N"]].copy()
        lines += [
            "### 3.1 Descriptive polar decomposition of the force block",
            "",
            "The 3x3 force block is decomposed into a nearest rotation-like part and a symmetric gain/shear-like part. This is useful for testing whether a simple frame/gain error is plausible, but it does **not** separate sensor intrinsic crosstalk from mounting/frame effects.",
            "",
            _markdown_table_sig(polar_show),
            "",
        ]
    lines += [
        "### 3.2 Force-to-moment coupling",
        "",
        "Because the experiment is force-only, measured moments can still respond strongly through the lever arm. These coefficients are therefore reported as force-to-moment coupling [Nmm/N], not as moment-channel crosstalk in the metrological sense.",
        "",
        _markdown_table_sig(moment_agg),
        "",
    ]
    if figures.get("moment_matrix"):
        lines += [f"![Force-to-moment coupling](figures/{figures['moment_matrix']})", ""]
    lines += [
        "**Suggested thesis placement:** Results §4.2; Discussion §5.2 for interpretation as gain/crosstalk/frame rotation.",
        "",
        "## 4. Within-session anchor drift",
        "",
        "Anchor drift is evaluated from the three fixed reference anchors within each session. The first anchor is used as the zero for the figure; the table summarises componentwise slopes and spans across runs.",
        "",
        _markdown_table_sig(anchor_agg),
        "",
    ]
    if figures.get("anchor"):
        lines += [f"![Anchor drift](figures/{figures['anchor']})", ""]
    lines += [
        "**Suggested thesis placement:** Results §4.4. Report the span as a directly observed within-session change; treat the three-point slope only as a coarse descriptor.",
        "",
        "## 5. Residual versus load magnitude",
        "",
        "For each Test A/B dwell, the residual shown is the RMS of the three force-component residuals for that dwell. Linear fits are first calculated separately per run and then summarised across runs; this keeps remounting as the independent replication level.",
        "",
        _markdown_table_sig(residual_agg),
        "",
    ]
    if figures.get("residual"):
        lines += [f"![Residual versus load](figures/{figures['residual']})", ""]
    lines += [
        "**Result-writing prompt:** a positive intercept is descriptive evidence for a load-independent error component; a positive slope indicates an additional load-dependent component. Do not interpret the intercept as a unique physical mechanism without the drift/hysteresis evidence.",
        "",
        "## 6. Apparent EI versus handbook EI",
        "",
        f"Handbook EI used by the compliant model: **{format_number(handbook_ei, 4)} N m²**." if np.isfinite(handbook_ei) else "Handbook EI was not available.",
        "",
        "Apparent EI is calculated independently from the reconstruction models by fitting end-effector lateral displacement against delivered lateral load on isolated Fx/Fy staircase dwells and applying the free-cantilever relation `delta = F L^3/(3 EI)` with the declared unsupported length. This does **not** fit EI for P2/PA2.",
        "",
        _markdown_table_sig(ei_agg),
        "",
    ]
    if figures.get("ei"):
        lines += [f"![Apparent EI](figures/{figures['ei']})", ""]
    lines += [
        "> **Interpretation limit:** `p_relative_*_EE` (reported above under its sensor-frame axis labels) is Lambda.7 end-effector displacement. The inferred compliance can include the distal adapter, coupling, haptic-device behaviour, frame error and shaft bending. Therefore this is an apparent system EI / consistency check, not a material-property measurement.",
        "",
        "**Suggested thesis placement:** Results §4.2 or an appendix diagnostic; Discussion §5.4 for the compliance hypothesis.",
        "",
        "## 7. Dwell-mean delivered versus predicted forces",
        "",
        "The following pooled table uses all Test A/B run-dwells for visualization. The models themselves were fitted independently within each run on training data only.",
        "",
        _markdown_table_sig(parity),
        "",
    ]
    for fn in figures.get("parity", []) or []:
        model = Path(fn).stem.split("_")[-1]
        lines += [f"### {model}: {model_name(model)}", "",
                  f"![{model} parity](figures/{fn})", ""]
    lines += [
        "## 8. Thesis-facing checklist",
        "",
        "Use this diagnostic report to support the following factual statements only after checking the generated tables:",
        "",
        "- whether the force transmission matrix is predominantly linear and whether off-diagonal terms are stable across remountings;",
        "- whether a rotation/gain-like decomposition is consistent across runs;",
        "- which axes show the largest matched-load hysteresis and anchor drift;",
        "- whether PA0/D2 residuals contain a substantial non-zero intercept as load approaches zero;",
        "- whether the apparent EI estimate is remotely consistent with handbook EI, while preserving the caveat that the displacement is system-level;",
        "- whether delivered-versus-predicted parity reveals channel-specific compression, offset, saturation or extrapolation error.",
        "",
        "These diagnostics do not establish traceable absolute accuracy, distal-moment accuracy, or trocar-condition performance.",
    ]

    report = out / "diagnostics_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

PACKAGES = ("numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML")


def write_run_info(out_dir: str | Path, command: str, settings: "Settings",
                   input_files: Sequence[str | Path], extra: Optional[dict] = None) -> Path:
    """Write <out_dir>/run_info.json: what was run, with which settings, on which files."""
    repo = Path(__file__).resolve().parent.parent  # the code's repository
    info = {
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(repo),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "settings_files": {"analysis": str(settings.analysis_file), "objects": str(settings.objects_file)},
        "settings": settings.raw,
        "input_files": [_file_info(Path(f)) for f in input_files],
    }
    if extra:
        info.update(extra)
    path = Path(out_dir) / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def _git_state(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
    try:
        head = git("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "note": "not a git repository"}
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(),
                "uncommitted_changes": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "note": "git not available"}


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _file_info(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
