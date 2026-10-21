"""Synthetic schema-7 calibration runs with a known ground truth.

Used by the tests only. The sensor reading is generated forwards from a known
tip force: the tip carries zero moment, bending moves the tip by
delta = L^3 / (3 EI) * f_perp, and the sensor reads m = (r + delta) x f.
The pattern set mirrors the real excitation table (staircases, diagonals,
octants, references, Test A, Test B and anchors).
"""
from __future__ import annotations


import numpy as np

from forcecal.constants import EXPECTED_SCHEMA_VERSION


def synthetic_run(path: str, seed: int = 0, mounting_offset: float = 0.0,
                   sensor_hz: float = 100.0, log_hz: float = 200.0,
                   dwell_s: float = 5.0, EI_true: float = 2.51,
                   L_free: float = 0.330,
                   r_nominal=(-0.30, 0.0, 0.0)) -> None:
    """Write a structurally faithful schema-7 file with a KNOWN compliance.

    The sensor reading is generated forwards from a true tip wrench, so the
    round trip is a real test: the tip carries zero moment (force-only
    excitation), bending moves the tip to r + delta with
    delta = L^3/(3 EI) * f_perp, and the sensor therefore reads
    m_sensor = (r + delta) x f. A correct compliance fit must recover EI_true.

    The pattern set deliberately includes planar diagonals and octants. Pure
    single-axis loads deflect the tip PARALLEL to the load and leave the moment
    arm unchanged to first order, so a run containing only single-axis
    staircases cannot identify EI at all -- injecting a known value and
    checking it comes back is what makes that visible.
    """
    rng = np.random.default_rng(seed)
    amp = 4.0
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    r_nom = np.asarray(r_nominal, dtype=float)
    axis = r_nom / np.linalg.norm(r_nom)
    c_ei = L_free ** 3 / (3.0 * EI_true)
    s2, s3 = 1 / np.sqrt(2), 1 / np.sqrt(3)

    patterns, sid = [], 0
    for axl, vec in (("Fx+", (1, 0, 0)), ("Fy+", (0, 1, 0)), ("Fz+", (0, 0, 1))):
        for k, lv in enumerate(levels + levels[-2::-1]):
            patterns.append(dict(f=tuple(v * amp * lv for v in vec),
                                 label=f"{axl}@{lv:.2f}{'v' if k >= len(levels) else '^'}",
                                 family="stair_bidir", split="train",
                                 sid=sid, step=k, desc=k >= len(levels)))
        sid += 1
    for pl, (a, b) in (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))):
        for qa, qb in ((1, 1), (-1, 1), (1, -1)):
            for lv in (0.6, 1.0):
                v = [0.0, 0.0, 0.0]
                v[a], v[b] = qa * s2 * amp * lv, qb * s2 * amp * lv
                patterns.append(dict(f=tuple(v), label=f"diag_{pl}_{qa:+d}{qb:+d}@{lv:.2f}",
                                     family="planar_diagonal", split="train",
                                     sid=-1, step=-1, desc=False))
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                patterns.append(dict(f=(sx * s3 * amp, sy * s3 * amp, sz * s3 * amp),
                                     label=f"oct_{sx:+d}{sy:+d}{sz:+d}@1.00",
                                     family="octant", split="train",
                                     sid=-1, step=-1, desc=False))
    for i in range(6):
        patterns.append(dict(f=(0, 0, 0.6 * amp), label=f"ref_{i+1}", family="reference",
                             split="train", sid=-1, step=-1, desc=False))
    c22, s22 = np.cos(np.radians(22.5)), np.sin(np.radians(22.5))
    for pl, (a, b) in (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))):
        for sgn in (1, -1):
            v = [0.0, 0.0, 0.0]
            v[a], v[b] = sgn * c22 * amp * 0.8, sgn * s22 * amp * 0.8
            patterns.append(dict(f=tuple(v), label=f"offgrid_{pl}_{sgn:+d}@0.80",
                                 family="offgrid_diagonal", split="test_A",
                                 sid=-1, step=-1, desc=False))
    for lbl, v in (("Fz+@1.50_robust", (0, 0, 1.5 * amp)),
                   ("Fx-@1.50_robust", (-1.5 * amp, 0, 0)),
                   ("diag_XY@1.50_robust", (1.5 * amp * s2, 1.5 * amp * s2, 0)),
                   ("oct_+++@1.50_robust", (1.5 * amp * s3,) * 3)):
        patterns.append(dict(f=v, label=lbl, family="robust", split="test_B",
                             sid=-1, step=-1, desc=False))
    for lbl in ("anchor_1_pre", "anchor_2_mid", "anchor_3_post"):
        patterns.append(dict(f=(0, 0, 0.4 * amp), label=lbl, family="anchor",
                             split="anchor", sid=-1, step=-1, desc=False))

    dt, rows, t, msg, last_sensor_t = 1.0 / log_hz, [], 0.0, 0, 0.0
    for pidx, p in enumerate(patterns):
        for phase, n in (("ramp", int(0.5 * log_hz)), ("dwell", int(dwell_s * log_hz)),
                         ("down", int(0.5 * log_hz))):
            rs = {"ramp": 0, "dwell": 1, "down": 2}[phase]
            for i in range(n):
                t += dt
                if t - last_sensor_t >= 1.0 / sensor_hz:
                    last_sensor_t, msg = t, msg + 1
                if phase == "down" and i > 0.7 * n:
                    scale = 0.0
                else:
                    scale = 1.0 if phase == "dwell" else (
                        i / n if phase == "ramp" else 1 - i / n)
                f_tip = np.array(p["f"]) * scale

                # Forward model: tip carries zero moment; bending shifts the
                # lever arm, so the sensor reads (r + delta) x f.
                f_perp = f_tip - (f_tip @ axis) * axis
                r_eff = r_nom + c_ei * f_perp
                m_sensor_Nm = np.cross(r_eff, f_tip)
                m_sensor_Nmm = 1e3 * m_sensor_Nm
                f_sensor = f_tip + np.array([0.02, -0.05, 0.01]) + mounting_offset
                if p["desc"]:
                    f_sensor = f_sensor + 0.03 * np.sign(f_tip)
                f_sensor = f_sensor + rng.normal(0, 0.006, 3)
                m_sensor_Nmm = m_sensor_Nmm + rng.normal(0, 0.2, 3)

                rows.append([
                    1.7e9 + t, t, last_sensor_t, t - last_sensor_t, msg,
                    p["split"], p["family"], p["label"], pidx, p["sid"], p["step"], 0, rs,
                    *f_tip, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, abs(rng.normal(0, 5e-5)),
                    *(f_sensor[[2, 1, 0]] * np.array([1, 1, -1])),
                    *(m_sensor_Nmm[[2, 1, 0]] * np.array([1, 1, -1])),
                    *f_sensor, *m_sensor_Nmm,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    *f_tip, 0.0, 0.0, 0.0, 0.0,
                ])

    cols = ("time_s,t_rel_s,sensor_time_s,sensor_age_s,sensor_msg_count,"
            "split,family,pattern_label,pattern_index,series_id,step_index,skipped,ramp_state,"
            "fx_cmd_N,fy_cmd_N,fz_cmd_N,tx_cmd_Nm,ty_cmd_Nm,tz_cmd_Nm,"
            "px_m,py_m,pz_m,oa_rad,ob_rad,og_rad,vx_mps,vy_mps,vz_mps,v_mag_mps,"
            "Fx_raw_sensor_N,Fy_raw_sensor_N,Fz_raw_sensor_N,"
            "Tx_raw_sensor_Nm,Ty_raw_sensor_Nm,Tz_raw_sensor_Nm,"
            "Fx_meas_EE_N,Fy_meas_EE_N,Fz_meas_EE_N,Tx_meas_EE_Nm,Ty_meas_EE_Nm,Tz_meas_EE_Nm,"
            "p_relative_x_EE_m,p_relative_y_EE_m,p_relative_z_EE_m,"
            "dth_yaw_rad,dth_pitch_rad,dth_roll_rad,"
            "f_delivered_x_EE_N,f_delivered_y_EE_N,f_delivered_z_EE_N,"
            "t_delivered_yaw_Nm,t_delivered_pitch_Nm,t_delivered_roll_Nm,kp_scale").split(",")

    with open(path, "w") as fh:
        fh.write(f"# schema_version,{EXPECTED_SCHEMA_VERSION}\n")
        fh.write("# controller_build,SYNTHETIC\n# excitation_type,force_only\n")
        fh.write("# amp_f_N,4.0\n# amp_t_Nm,0.0\n")
        fh.write(f"# synthetic_EI_true_Nm2,{EI_true}\n")
        fh.write("# force_bias_N,0;0;0\n# torque_bias_Nm,0;0;0\n# ema_alpha,0.15\n")
        fh.write("# pattern_seed,20260907\n")
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{x:.9f}" if isinstance(x, (int, float, np.floating))
                              else str(x) for x in r) + "\n")
        fh.write(f"# completed_patterns,{len(patterns)}\n")
        fh.write(f"# total_patterns,{len(patterns)}\n")
