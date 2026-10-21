"""All figures of the Results and diagnostics reports.

Functions here only draw and save; they do not compute results. Each function
takes a finished table and writes PNG (and for the Results report also PDF)
files. Call ``configure_figures`` once before plotting.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import FORCE_LABELS, model_name, MODEL_ORDER, MOMENT_LABELS
from .data import safe_name
from .metrics import format_number, r2_plain


# PNG resolution, set once per command from figures.dpi in configs/analysis.yaml.
_FIGURE_DPI: Optional[int] = None


def configure_figures(dpi: int) -> None:
    """Set the resolution of all saved PNG figures."""
    global _FIGURE_DPI
    _FIGURE_DPI = int(dpi)


def _dpi() -> int:
    if _FIGURE_DPI is None:
        raise RuntimeError("call plots.configure_figures(settings.figures.dpi) before plotting")
    return _FIGURE_DPI


# Top fraction of the axes kept free of data so the mean/SD labels fit inside.
_BOX_HEADER_FRAC = 0.22


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def _boxplot_ylim_with_header(ax, extents, include=(), keep_zero=True,
                              header_frac: float = _BOX_HEADER_FRAC) -> None:
    """Set y-limits so the drawn boxes/whiskers fill the lower part of the axes.

    ``extents`` are the y-values that must stay visible (whisker end points).
    The top ``header_frac`` of the axes stays empty for the mean/SD labels.
    ``include`` adds reference values (e.g. 0 or 1) that must stay visible.
    """
    parts = [_finite(a) for a in extents]
    parts = [p for p in parts if len(p)]
    if not parts:
        return
    v = np.concatenate(parts + [_finite(include)])
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-6) * 0.2
    bottom = 0.0 if (keep_zero and lo >= 0.0) else lo - 0.06 * span
    data_top = hi + 0.04 * span
    top = bottom + (data_top - bottom) / (1.0 - header_frac)
    ax.set_ylim(bottom, top)


def _annotate_mean_sd(ax, arrays, positions, fmt: str = ".3f",
                      fontsize: float = 7.2) -> None:
    """Write mean and sample SD above each box, inside the axes.

    x is in data coordinates (box position), y in axes coordinates, so the
    label always sits at the top edge inside the plotting area.
    """
    trans = ax.get_xaxis_transform()
    for x, a in zip(positions, arrays):
        a = _finite(a)
        if not len(a):
            continue
        mean = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
        sd_txt = f"SD {sd:{fmt}}" if np.isfinite(sd) else "SD n/a"
        ax.text(x, 0.975, f"mean {mean:{fmt}}\n{sd_txt}", transform=trans,
                ha="center", va="top", fontsize=fontsize, linespacing=1.1)


def _draw_boxplot(ax, arrays, positions=None, labels=None, fmt: str = ".3f",
                  include=(), keep_zero: bool = True, widths: float = 0.55,
                  fontsize: float = 7.2, annotate: bool = True) -> np.ndarray:
    """Boxplot without individual points or fliers, with optional mean/SD labels."""
    arrays = [_finite(a) for a in arrays]
    if positions is None:
        positions = np.arange(1, len(arrays) + 1, dtype=float)
    positions = np.asarray(positions, dtype=float)
    bp = ax.boxplot(arrays, positions=positions, widths=widths, showfliers=False,
                    patch_artist=True, medianprops={"linewidth": 2.0},
                    boxprops={"alpha": 0.55}, whiskerprops={"linewidth": 1.1},
                    capprops={"linewidth": 1.1})
    # Fliers are hidden, so scale the axis to the whiskers, not to the data extremes.
    extents = [np.asarray(w.get_ydata(), dtype=float) for w in bp["whiskers"]]
    # No annotations means no header space needs to be reserved above the boxes.
    header_frac = _BOX_HEADER_FRAC if annotate else 0.0
    _boxplot_ylim_with_header(ax, extents, include=include, keep_zero=keep_zero,
                              header_frac=header_frac)
    if annotate:
        _annotate_mean_sd(ax, arrays, positions, fmt=fmt, fontsize=fontsize)
    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    return positions


def _save_figure(fig, fig_dir: str, name: str) -> Tuple[str, str]:
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, name + ".png")
    pdf = os.path.join(fig_dir, name + ".pdf")
    fig.savefig(png, dpi=_dpi(), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return png, pdf


def _boxplot_metric(df: pd.DataFrame, models: Sequence[str], value: str,
                    title: str, ylabel: str, fig_dir: str, name: str,
                    fmt: str = ".3f", include=(), keep_zero: bool = True,
                    annotate: bool = True) -> Optional[str]:
    import matplotlib.pyplot as plt
    order = [m for m in models
             if m in set(df["model"]) and df.loc[df["model"] == m, value].notna().any()]
    if not order:
        return None
    arrays = [df.loc[df["model"] == m, value].to_numpy(dtype=float) for m in order]
    fig, ax = plt.subplots(figsize=(max(5.6, 1.15 * len(order)), 4.4))
    _draw_boxplot(ax, arrays, labels=order, fmt=fmt, include=include, keep_zero=keep_zero,
                  annotate=annotate)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, fig_dir, name)
    return name


def plot_family_accuracy(cm: pd.DataFrame, evaluation: str, fig_dir: str, prefix: str) -> List[str]:
    pooled = cm[(cm["evaluation"] == evaluation) & (cm["force_channel"] == "Pooled")].copy()
    out = []
    families = [
        ("Physical Models", ["P0", "P1", "P2"]),
        ("Linear Error Models", ["PA0", "PA1", "PA2"]),
        ("Data-Driven", ["D1", "D2"]),
    ]
    for family, mods in families:
        q = pooled[pooled["model"].isin(mods)]
        nm = f"{prefix}_{safe_name(family)}"
        made = _boxplot_metric(q, mods, "rmse_N",
                               f"{'Calibration Accuracy' if evaluation == 'accuracy' else evaluation.capitalize()} — {family}",
                               "Pooled per-axis RMSE [N]", fig_dir, nm)
        if made:
            out.append(made)
    return out


def plot_all_models_accuracy(cm: pd.DataFrame, fig_dir: str, name: str) -> Optional[str]:
    """Single Test-A accuracy boxplot spanning every model, unannotated and
    scaled tightly to the data range instead of a fixed zero baseline."""
    pooled = cm[(cm["evaluation"] == "accuracy") & (cm["force_channel"] == "Pooled")]
    return _boxplot_metric(pooled, MODEL_ORDER, "rmse_N", "Calibration Accuracy-All Models",
                           "Pooled per-axis RMSE [N]", fig_dir, name,
                           keep_zero=False, annotate=False)


def plot_rmse_across_runs(cm: pd.DataFrame, fig_dir: str,
                           name: str = "4_2_rmse_across_runs") -> Optional[str]:
    """Pooled RMSE of every model for each run (remounting), Test A and Test B."""
    import matplotlib.pyplot as plt
    pooled = cm[cm["force_channel"] == "Pooled"]
    if not len(pooled):
        return None
    models = [m for m in MODEL_ORDER if m in set(pooled["model"])]
    runs = sorted(pooled["run_id"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.3 * len(runs) + 5), 4.6))
    cmap = plt.get_cmap("tab10")
    for ax, (ev, label) in zip(axes, (("accuracy", "Test A"), ("robustness", "Test B"))):
        q = pooled[pooled["evaluation"] == ev]
        for i, m in enumerate(models):
            g = q[q["model"] == m]
            by_run = dict(zip(g["run_id"].astype(str), g["rmse_N"].astype(float)))
            ax.plot(range(len(runs)), [by_run.get(r, np.nan) for r in runs],
                    marker="o", ms=4, lw=1.1, color=cmap(i % 10), label=m)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha="right")
        ax.set_ylabel("Pooled per-axis RMSE [N]")
        ax.set_title(f"{label} — RMSE across runs")
        ax.grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=7, title="model", title_fontsize=7)
    fig.tight_layout()
    _save_figure(fig, fig_dir, name)
    return name


def plot_paired_effect(cm: pd.DataFrame, evaluation: str, models: Sequence[str],
                        fig_dir: str, name: str, title: str) -> Optional[str]:
    q = cm[(cm["evaluation"] == evaluation) & (cm["force_channel"] == "Pooled") & (cm["model"].isin(models))]
    return _boxplot_metric(q, models, "rmse_N", title, "Pooled per-axis RMSE [N]", fig_dir, name)


def plot_polar(polar: pd.DataFrame, fig_dir: str) -> List[str]:
    import matplotlib.pyplot as plt
    out = []
    if not len(polar):
        return out
    vals = polar["rotation_angle_deg"].dropna().to_numpy(dtype=float)
    if len(vals):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        _draw_boxplot(ax, [vals], labels=["PA0 force block"], fmt=".2f")
        ax.set_ylabel("Residual rotation angle [deg]")
        ax.set_title("Polar decomposition of PA0 force block")
        ax.grid(axis="y", alpha=0.25)
        _save_figure(fig, fig_dir, "4_2_pa0_rotation_angle")
        out.append("4_2_pa0_rotation_angle")
    arrs = [polar[c].dropna().to_numpy(dtype=float) for c in ("gain_x", "gain_y", "gain_z")]
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    _draw_boxplot(ax, arrs, labels=["Fx", "Fy", "Fz"], fmt=".3f", keep_zero=False)
    ax.set_ylabel("Stretch / gain from polar factor")
    ax.set_title("PA0 force-block gains across remountings")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, fig_dir, "4_2_pa0_axis_gains")
    out.append("4_2_pa0_axis_gains")
    return out


def plot_compliance_delta(cm: pd.DataFrame, fig_dir: str) -> Optional[str]:
    import matplotlib.pyplot as plt
    rows = []
    for ev in ("accuracy", "robustness"):
        q = cm[(cm["evaluation"] == ev) & (cm["force_channel"] == "Pooled")]
        for a, b, label in (("P2", "P1", "P2 − P1"), ("PA2", "PA1", "PA2 − PA1")):
            aa = q[q["model"] == a][["run_id", "rmse_N"]].rename(columns={"rmse_N": "a"})
            bb = q[q["model"] == b][["run_id", "rmse_N"]].rename(columns={"rmse_N": "b"})
            m = aa.merge(bb, on="run_id")
            for _, z in m.iterrows():
                rows.append({"evaluation": ev, "contrast": label, "delta_N": z["a"] - z["b"]})
    d = pd.DataFrame(rows)
    if not len(d):
        return None
    ev_label = {"accuracy": "Test Set A", "robustness": "Test Set B"}
    labels = []; arrays = []
    for ev in ("accuracy", "robustness"):
        for contrast in ("P2 − P1", "PA2 − PA1"):
            a = d[(d.evaluation == ev) & (d.contrast == contrast)]["delta_N"].to_numpy(float)
            if len(a):
                labels.append(f"{ev_label[ev]}\n{contrast}"); arrays.append(a)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _draw_boxplot(ax, arrays, labels=labels, fmt=".4f", include=(0.0,), keep_zero=False)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("RMSE difference [N] (compliant − fixed)")
    ax.set_title("Effect of the compliant lever assumption")
    ax.grid(axis="y", alpha=.25)
    _save_figure(fig, fig_dir, "4_2_compliance_effect")
    return "4_2_compliance_effect"


def plot_channel_observability(cm: pd.DataFrame, fig_dir: str) -> List[str]:
    out = []
    q = cm[(cm["evaluation"] == "accuracy") & (cm["model"].isin(["PA0", "PA1", "PA2", "D2"])) &
           (cm["force_channel"].isin(["Fx", "Fy", "Fz"]))]
    if not len(q):
        return out
    import matplotlib.pyplot as plt
    for metric, ylabel, name, keep_zero in (("rmse_N", "RMSE [N]", "4_2_channel_rmse", True),
                                            ("r2", "R²", "4_2_channel_r2", False)):
        labels = []; arrays = []
        for m in ["PA0", "PA1", "PA2", "D2"]:
            for ch in ["Fx", "Fy", "Fz"]:
                a = q[(q.model == m) & (q.force_channel == ch)][metric].dropna().to_numpy(float)
                if len(a):
                    labels.append(f"{m}\n{ch}"); arrays.append(a)
        if arrays:
            fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(arrays)), 4.6))
            _draw_boxplot(ax, arrays, labels=labels, fmt=".3f",
                          keep_zero=keep_zero, fontsize=6.5)
            ax.set_ylabel(ylabel); ax.set_title("Test A channel observability")
            ax.grid(axis="y", alpha=.25)
            _save_figure(fig, fig_dir, name); out.append(name)
    return out


def plot_repeatability(raw: pd.DataFrame, fig_dir: str) -> Optional[str]:
    if not len(raw):
        return None
    import matplotlib.pyplot as plt
    arrs = [raw[raw.force_channel == ch]["repeatability_sd_N"].to_numpy(float)
            for ch in ("Fx", "Fy", "Fz")]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    _draw_boxplot(ax, arrs, labels=["Fx", "Fy", "Fz (axial)"], fmt=".3f")
    ax.set_ylabel("Repeated-dwell SD [N]")
    ax.set_title("Within-session repeatability across remountings")
    ax.grid(axis="y", alpha=.25)
    _save_figure(fig, fig_dir, "4_4_repeatability_channels")
    return "4_4_repeatability_channels"


def plot_transfer_heatmap(tr: pd.DataFrame, model: str, fig_dir: str) -> Optional[str]:
    if tr is None or not len(tr) or "model" not in tr.columns:
        return None
    q=tr[tr.model==model].copy()
    if not len(q): return None
    import matplotlib.pyplot as plt
    runs=sorted(set(q.fit_on)|set(q.tested_on))
    piv=q.pivot_table(index="fit_on",columns="tested_on",values="force_rmse_N",aggfunc="mean").reindex(index=runs,columns=runs)
    fig,ax=plt.subplots(figsize=(6.2,5.2))
    im=ax.imshow(piv.to_numpy(float),aspect="auto")
    ax.set_xticks(np.arange(len(runs))); ax.set_xticklabels(runs,rotation=45,ha="right")
    ax.set_yticks(np.arange(len(runs))); ax.set_yticklabels(runs)
    ax.set_xlabel("Tested on"); ax.set_ylabel("Calibration fitted on")
    ax.set_title(f"Cross-remounting calibration transfer — {model}")
    cbar=fig.colorbar(im,ax=ax); cbar.set_label("Test A pooled RMSE [N]")
    vals=piv.to_numpy(float)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i,j]): ax.text(j,i,f"{vals[i,j]:.2f}",ha="center",va="center",fontsize=7)
    _save_figure(fig,fig_dir,f"4_4_transfer_{model}")
    return f"4_4_transfer_{model}"


def plot_abs_vs_normalized(cm: pd.DataFrame, fig_dir: str) -> List[str]:
    out = []
    q = cm[(cm.force_channel == "Pooled") & (cm.model.isin(["PA0", "D2", "PA1"])) &
           (cm.evaluation.isin(["accuracy", "robustness"]))].copy()
    if not len(q):
        return out
    import matplotlib.pyplot as plt
    for metric, ylabel, name, fmt in (("rmse_N", "Pooled per-axis RMSE [N]", "4_5_absolute_error", ".3f"),
                                      ("normalized_rmse_pct", "Normalized RMSE [%]", "4_5_normalized_error", ".1f")):
        labels = []; arrs = []
        for m in ["PA0", "D2", "PA1"]:
            for ev in ["accuracy", "robustness"]:
                a = q[(q.model == m) & (q.evaluation == ev)][metric].dropna().to_numpy(float)
                if len(a):
                    labels.append(f"{m}\n{'A' if ev == 'accuracy' else 'B'}"); arrs.append(a)
        if not arrs:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(arrs)), 4.4))
        _draw_boxplot(ax, arrs, labels=labels, fmt=fmt)
        ax.set_ylabel(ylabel); ax.set_title("Absolute and relative extrapolation error")
        ax.grid(axis="y", alpha=.25)
        _save_figure(fig, fig_dir, name); out.append(name)
    return out


def plot_degradation(cm: pd.DataFrame, fig_dir: str) -> Optional[str]:
    """Paired per-run degradation, pooled across force channels only.

    Uses normalized RMSE with the nominal normalization loads of
    nominal amplitudes (reference_values.nominal_amplitude_N in
    configs/analysis.yaml; see evaluation._model_force_channel_metrics), so each ratio equals the absolute RMSE ratio
    divided by 1.5 and matches Eq. (8) and Table III of the paper. Deliberately
    restricted to the pooled force channel -- no per-axis (Fx/Fy/Fz) breakdown.
    No in-figure title is drawn; the figure caption in the paper carries it.
    """
    d=cm[(cm.force_channel=="Pooled") & (cm.evaluation.isin(["accuracy","robustness"]))]
    rows=[]
    for m in [x for x in MODEL_ORDER if x in set(d.model)]:
        a=d[(d.model==m)&(d.evaluation=="accuracy")][["run_id","normalized_rmse_pct"]].rename(columns={"normalized_rmse_pct":"A"})
        b=d[(d.model==m)&(d.evaluation=="robustness")][["run_id","normalized_rmse_pct"]].rename(columns={"normalized_rmse_pct":"B"})
        z=a.merge(b,on="run_id");
        for _,r in z.iterrows(): rows.append({"model":m,"degradation_x":r.B/r.A if r.A>0 else np.nan})
    dd=pd.DataFrame(rows)
    if not len(dd): return None
    return _boxplot_metric(dd,[m for m in MODEL_ORDER if m in set(dd.model)],"degradation_x",
                           "", "NRMSE ratio, test set B / test set A [–]",
                           fig_dir,"4_5_degradation", fmt=".2f", include=(1.0,))


def plot_error_vs_load(err: pd.DataFrame, fig_dir: str) -> Optional[str]:
    q=err[err.model.isin(["PA0","D2","PA1"])].copy()
    if not len(q): return None
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,4.6))
    markers={"PA0":"o","D2":"s","PA1":"^"}
    for m in ["PA0","D2","PA1"]:
        z=q[q.model==m]
        if not len(z): continue
        ax.scatter(z.load_N,z.dwell_component_rms_error_N,s=22,alpha=.55,marker=markers[m],label=m)
        if np.ptp(z.load_N)>1e-12:
            slope,intercept=np.polyfit(z.load_N,z.dwell_component_rms_error_N,1)
            xx=np.linspace(z.load_N.min(),z.load_N.max(),60)
            ax.plot(xx,intercept+slope*xx,label=f"{m} fit: intercept {intercept:.2f} N")
    ax.set_xlabel("Reference force magnitude [N]")
    ax.set_ylabel("Per-dwell pooled-component RMS error [N]")
    ax.set_title("Residual error versus applied load")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    _save_figure(fig,fig_dir,"4_5_error_vs_load")
    return "4_5_error_vs_load"


def plot_deadweight(weight_rows: pd.DataFrame, fig_dir: str) -> Optional[str]:
    if not len(weight_rows): return None
    q=weight_rows[np.isfinite(weight_rows["absolute_error_N"])].copy()
    if not len(q): return None
    models=[m for m in ["sensor_raw","PA0","PA1","D2"] if m in set(q.model)]
    return _boxplot_metric(q,models,"absolute_error_N","Independent dead-weight validation",
                           "Absolute magnitude error [N]",fig_dir,"4_3_deadweight_validation")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_force_matrix(force_matrix: pd.DataFrame, fig_dir: Path) -> Optional[str]:
    if not len(force_matrix):
        return None
    plt = _mpl()
    g = force_matrix[force_matrix["input_force"] != "intercept"].copy()
    summary = g.groupby(["output_channel", "input_force"])["coefficient"].agg(["mean", "std"]).reset_index()
    M = np.full((3, 3), np.nan)
    S = np.full((3, 3), np.nan)
    for i, out in enumerate(FORCE_LABELS):
        for j, inn in enumerate(FORCE_LABELS):
            r = summary[(summary.output_channel == out) & (summary.input_force == inn)]
            if len(r):
                M[i, j] = r["mean"].iloc[0]
                S[i, j] = r["std"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(M, aspect="equal")
    ax.set_xticks(range(3), FORCE_LABELS)
    ax.set_yticks(range(3), FORCE_LABELS)
    ax.set_xlabel("Delivered force input")
    ax.set_ylabel("Measured force output")
    ax.set_title("Force transmission / cross-axis matrix across runs")
    for i in range(3):
        for j in range(3):
            if np.isfinite(M[i, j]):
                sd = S[i, j]
                txt = f"{M[i,j]:.3f}" + (f"\nSD {sd:.3f}" if np.isfinite(sd) else "")
                ax.text(j, i, txt, ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Coefficient [N/N]")
    fig.tight_layout()
    out = fig_dir / "01_force_transmission_crosstalk_matrix.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_moment_matrix(moment_matrix: pd.DataFrame, fig_dir: Path) -> Optional[str]:
    if not len(moment_matrix):
        return None
    plt = _mpl()
    g = moment_matrix[moment_matrix["input_force"] != "intercept"].copy()
    summary = g.groupby(["output_channel", "input_force"])["coefficient"].agg(["mean", "std"]).reset_index()
    M = np.full((3, 3), np.nan)
    S = np.full((3, 3), np.nan)
    for i, out in enumerate(MOMENT_LABELS):
        for j, inn in enumerate(FORCE_LABELS):
            r = summary[(summary.output_channel == out) & (summary.input_force == inn)]
            if len(r):
                M[i, j] = r["mean"].iloc[0]
                S[i, j] = r["std"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(M, aspect="equal")
    ax.set_xticks(range(3), FORCE_LABELS)
    ax.set_yticks(range(3), MOMENT_LABELS)
    ax.set_xlabel("Delivered force input")
    ax.set_ylabel("Measured sensor moment")
    ax.set_title("Force-to-moment coupling across runs")
    for i in range(3):
        for j in range(3):
            if np.isfinite(M[i, j]):
                sd = S[i, j]
                txt = f"{M[i,j]:.1f}" + (f"\nSD {sd:.1f}" if np.isfinite(sd) else "")
                ax.text(j, i, txt, ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Coefficient [Nmm/N]")
    fig.tight_layout()
    out = fig_dir / "02_force_to_moment_coupling_matrix.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_hysteresis(h: pd.DataFrame, fig_dir: Path) -> Optional[str]:
    if not len(h):
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, applied in zip(axes, FORCE_LABELS):
        g = h[(h["applied_axis"] == applied) & (h["output_channel"] == applied)]
        for run_id, r in g.groupby("run_id"):
            r = r.sort_values("level_N")
            ax.plot(r["level_N"], r["value"], marker="o", alpha=0.55, label=run_id)
        if len(g):
            med = g.groupby("level_N")["value"].median().sort_index()
            ax.plot(med.index, med.values, marker="o", linewidth=2.5, label="median")
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"Applied {applied}")
        ax.set_xlabel("Matched load magnitude [N]")
        ax.set_ylabel("Descending - ascending [N]")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Loaded-channel hysteresis across remountings")
    fig.tight_layout()
    out = fig_dir / "03_hysteresis_loaded_channel.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_anchor_drift(a: pd.DataFrame, fig_dir: Path) -> Optional[str]:
    if not len(a):
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=False)
    for ax, ch in zip(axes, FORCE_LABELS):
        g = a[a["channel"] == ch]
        for run_id, r in g.groupby("run_id"):
            r = r.sort_values("elapsed_min")
            ax.plot(r["elapsed_min"], r["delta_from_first_N"], marker="o", alpha=0.65, label=run_id)
        ax.axhline(0.0, linewidth=1)
        ax.set_title(ch)
        ax.set_xlabel("Elapsed time from first anchor [min]")
        ax.set_ylabel("Anchor change from first [N]")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Within-session anchor drift")
    fig.tight_layout()
    out = fig_dir / "04_anchor_drift.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_residual_vs_load(pred: pd.DataFrame, fig_dir: Path,
                          models: Sequence[str] = ("P0", "PA0", "PA1", "PA2", "D2")) -> Optional[str]:
    g = pred[pred["model"].isin(models)].copy()
    if not len(g):
        return None
    plt = _mpl()
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.3), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        m = g[g["model"] == model]
        for ev, marker in [("Test A", "o"), ("Test B", "s")]:
            e = m[m["evaluation"] == ev]
            ax.scatter(e["load_magnitude_N"], e["dwell_pooled_rmse_N"],
                       s=24, alpha=0.42, marker=marker, label=ev)
        x = m["load_magnitude_N"].to_numpy(float)
        y = m["dwell_pooled_rmse_N"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(np.min(x[ok]), np.max(x[ok]), 100)
            ax.plot(xs, slope * xs + intercept, linewidth=2)
            ax.text(0.03, 0.97, f"pooled fit\nintercept {intercept:.2f} N\nslope {slope:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(model)
        ax.set_xlabel("Delivered load magnitude [N]")
        ax.set_ylabel("Dwell pooled-component residual [N]")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Residual versus load magnitude")
    fig.tight_layout()
    out = fig_dir / "05_residual_vs_load.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_ei(ei: pd.DataFrame, fig_dir: Path) -> Optional[str]:
    if not len(ei):
        return None
    plt = _mpl()
    axes_names = [x for x in ("Fx", "Fy") if x in set(ei["axis"])]
    data = [ei.loc[ei["axis"] == a, "EI_apparent_Nm2"].dropna().to_numpy() for a in axes_names]
    if not any(len(x) for x in data):
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.boxplot(data, tick_labels=axes_names, showfliers=True)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(i + jitter, vals, s=30, alpha=0.75)
    handbook = ei["EI_handbook_Nm2"].dropna()
    if len(handbook):
        ax.axhline(float(handbook.iloc[0]), linestyle="--", linewidth=1.5,
                   label=f"handbook EI = {handbook.iloc[0]:.3f} N m²")
        ax.legend()
    ax.set_ylabel("Apparent EI [N m²]")
    ax.set_title("Apparent EI from lateral displacement vs handbook EI")
    fig.tight_layout()
    out = fig_dir / "06_apparent_EI_vs_handbook.png"
    fig.savefig(out, dpi=_dpi(), bbox_inches="tight")
    plt.close(fig)
    return out.name


def plot_parity_per_model(pred: pd.DataFrame, fig_dir: Path) -> List[str]:
    if not len(pred):
        return []
    plt = _mpl()
    files = []
    for model in MODEL_ORDER:
        g = pred[pred["model"] == model]
        if not len(g):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
        vals_all = []
        for ch in FORCE_LABELS:
            vals_all.extend(g[f"delivered_{ch}_N"].tolist())
            vals_all.extend(g[f"predicted_{ch}_N"].tolist())
        vals_all = np.asarray(vals_all, float)
        vals_all = vals_all[np.isfinite(vals_all)]
        lo = float(np.min(vals_all)) if len(vals_all) else -1.0
        hi = float(np.max(vals_all)) if len(vals_all) else 1.0
        margin = max(0.2, 0.05 * (hi - lo))
        lo, hi = lo - margin, hi + margin
        for ax, ch in zip(axes, FORCE_LABELS):
            for ev, marker in [("Test A", "o"), ("Test B", "s")]:
                e = g[g["evaluation"] == ev]
                ax.scatter(e[f"delivered_{ch}_N"], e[f"predicted_{ch}_N"],
                           s=24, alpha=0.45, marker=marker, label=ev)
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            y = g[f"delivered_{ch}_N"].to_numpy(float)
            p = g[f"predicted_{ch}_N"].to_numpy(float)
            rmse = np.sqrt(np.mean((p - y) ** 2))
            ax.text(0.03, 0.97, f"RMSE {rmse:.3f} N\nR² {format_number(r2_plain(y,p),2)}",
                    transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(ch)
            ax.set_xlabel("Delivered dwell mean [N]")
            ax.set_ylabel("Predicted dwell mean [N]")
        axes[-1].legend(fontsize=8, loc="best")
        fig.suptitle(f"{model}: {model_name(model)}")
        fig.tight_layout()
        fn = f"07_parity_{model}.png"
        fig.savefig(fig_dir / fn, dpi=_dpi(), bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
    return files
