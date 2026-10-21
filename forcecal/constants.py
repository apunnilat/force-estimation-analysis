"""Fixed definitions shared by all modules.

Everything in this file is fixed by the recorded data or by the hardware:
the CSV column names (schema 7, written by the Lambda.7 controller), the unit
convention of the sensor moments, the rotation between the two coordinate
frames and the list of reconstruction models. Values that a user may change
for an analysis are in ``configs/analysis.yaml`` and ``configs/objects.yaml``.

Coordinate frames
-----------------
S   HEX-32 transducer frame. All reported force results use this frame.
    The shaft axis is +z_S, so the axial force channel is Fz_S.
EE  Lambda.7 end-effector frame. The logged ``*_meas_EE_*`` channels are
    already in this frame, and the physical models are formulated in it.

    x_EE = -z_S,   y_EE = y_S,   z_EE = x_S

The sensor-to-tip lever points along -x_EE (= +z_S). Fx_EE is therefore the
shaft-axial force, and the transverse forces follow from the bending moments:

    r = [-L, 0, 0]  =>  M = r x F = [0, +L Fz, -L Fy]
    Fy = -Tz / L,   Fz = +Ty / L

Tx_EE is torsion about the shaft and is not used to reconstruct forces.

Unit convention
---------------
The sensor moment columns keep their historical names ending in ``_Nm`` because
the names are part of the recorded schema, but their values are in Nmm.
M [Nmm] / L [mm] therefore gives F [N] directly. The controller's torque command
and reference columns are in Nm; they are not model inputs.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Recorded CSV schema
# ---------------------------------------------------------------------------

EXPECTED_SCHEMA_VERSION = 7

# Measured sensor channels, EE frame (moments in Nmm, see module docstring).
FORCE_MEAS = ["Fx_meas_EE_N", "Fy_meas_EE_N", "Fz_meas_EE_N"]
MOMENT_MEAS = ["Tx_meas_EE_Nm", "Ty_meas_EE_Nm", "Tz_meas_EE_Nm"]

# Reference wrench delivered by the Lambda.7 and the commanded wrench, EE frame.
FORCE_REF = ["f_delivered_x_EE_N", "f_delivered_y_EE_N", "f_delivered_z_EE_N"]
MOMENT_REF = ["t_delivered_yaw_Nm", "t_delivered_pitch_Nm", "t_delivered_roll_Nm"]
FORCE_CMD = ["fx_cmd_N", "fy_cmd_N", "fz_cmd_N"]
MOMENT_CMD = ["tx_cmd_Nm", "ty_cmd_Nm", "tz_cmd_Nm"]
POSE_DEV = ["dth_yaw_rad", "dth_pitch_rad", "dth_roll_rad"]

# Raw transducer channels, S frame.
RAW_SENSOR = [
    "Fx_raw_sensor_N", "Fy_raw_sensor_N", "Fz_raw_sensor_N",
    "Tx_raw_sensor_Nm", "Ty_raw_sensor_Nm", "Tz_raw_sensor_Nm",
]

# Sensor-frame copies of the three EE force triplets. They are computed once
# when a file is loaded (data.add_sensor_frame_force_columns) and used for all
# reported force results. The EE columns stay unchanged and feed the models.
FORCE_MEAS_S = ["Fx_meas_S_N", "Fy_meas_S_N", "Fz_meas_S_N"]
FORCE_REF_S = ["f_delivered_x_S_N", "f_delivered_y_S_N", "f_delivered_z_S_N"]
FORCE_CMD_S = ["fx_cmd_S_N", "fy_cmd_S_N", "fz_cmd_S_N"]

# Model inputs (six sensor channels) and reference targets.
SENSOR_COLS = FORCE_MEAS + MOMENT_MEAS
TARGET_COLS = {
    "delivered": FORCE_REF + MOMENT_REF,
    "commanded": FORCE_CMD + MOMENT_CMD,
}
# Scoring targets: forces in the sensor frame S. Moments are not reconstructed
# or scored, so their columns are unchanged.
TARGET_COLS_S = {
    "delivered": FORCE_REF_S + MOMENT_REF,
    "commanded": FORCE_CMD_S + MOMENT_CMD,
}

# Channels read from the dead-weight validation file.
WEIGHT_SENSOR = FORCE_MEAS + MOMENT_MEAS

# Value of the ``ramp_state`` column during a dwell (0 ramp up, 1 dwell, 2 ramp down).
RAMP_DWELL = 1

SENSOR_MOMENT_UNIT = "Nmm"

FORCE_LABELS = ["Fx", "Fy", "Fz"]
MOMENT_LABELS = ["Tx", "Ty", "Tz"]

# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------

S_TO_EE = np.array([[0.0, 0.0, -1.0],
                    [0.0, 1.0,  0.0],
                    [1.0, 0.0,  0.0]], dtype=float)
EE_TO_S = S_TO_EE.T

# Axial and transverse components. EE indices are used inside the physical
# models; S indices are used on reported (S-frame) results. Do not mix them.
EE_AXIAL_INDEX = 0
EE_LATERAL_INDICES = (1, 2)
EE_SHAFT_AXIS = np.array([-1.0, 0.0, 0.0], dtype=float)
S_AXIAL_INDEX = 2
S_LATERAL_INDICES = (0, 1)


def rotate_ee_to_s(v: np.ndarray) -> np.ndarray:
    """Map a force or moment triplet, shape (3,) or (N, 3), from EE to S."""
    a = np.asarray(v, dtype=float)
    if a.ndim == 1:
        return EE_TO_S @ a
    return a @ EE_TO_S.T


def rotate_ee_to_s_matrix(M: np.ndarray) -> np.ndarray:
    """Express a 3x3 map between EE-frame triplets in the sensor frame S.

    Rows are the output axes and columns the input axes, both in EE order.
    The result is EE_TO_S @ M @ EE_TO_S.T.
    """
    return EE_TO_S @ np.asarray(M, dtype=float) @ EE_TO_S.T


def ee_axis_to_s_label(ee_label: str) -> str:
    """Return the sensor-frame label ("Fx", "Fy", "Fz") of an EE-frame axis label."""
    p = FORCE_LABELS.index(ee_label)
    return FORCE_LABELS[int(np.argmax(np.abs(EE_TO_S[:, p])))]


def channel_unit(channel: str) -> str:
    """Unit of a recorded channel. Sensor moments are Nmm despite the ``_Nm`` suffix."""
    if channel in MOMENT_MEAS or channel in RAW_SENSOR[3:]:
        return SENSOR_MOMENT_UNIT
    if channel.endswith("_N"):
        return "N"
    if channel.endswith("_Nm"):
        return "Nm"
    return ""

# ---------------------------------------------------------------------------
# Reconstruction models
# ---------------------------------------------------------------------------

# Short IDs are used in tables and figures; the equations are in models.py.
MODEL_ORDER = ["P0", "P1", "P2", "PA0", "PA1", "PA2", "D1", "D2"]
MODEL_INFO = {
    "P0":  {"name": "Direct-force equilibrium", "family": "Physical Models"},
    "P1":  {"name": "Fixed-lever moment model", "family": "Physical Models"},
    "P2":  {"name": "Compliant-lever moment model", "family": "Physical Models"},
    "PA0": {"name": "Affine direct-force model", "family": "Linear Error Models"},
    "PA1": {"name": "Affine fixed-lever model", "family": "Linear Error Models"},
    "PA2": {"name": "Affine compliant-lever model", "family": "Linear Error Models"},
    "D1":  {"name": "Quadratic regression", "family": "Data-Driven"},
    "D2":  {"name": "Multilayer perceptron", "family": "Data-Driven"},
}


def model_name(model: str) -> str:
    return MODEL_INFO.get(model, {}).get("name", model)


def model_family(model: str) -> str:
    return MODEL_INFO.get(model, {}).get("family", "Other")
