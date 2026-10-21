# Conventions

This page defines the coordinate frames, units, CSV columns and data splits used
by the analysis. The definitions in code are in
[`forcecal/constants.py`](../forcecal/constants.py).

## Coordinate frames

| Frame | Definition | Used for |
|---|---|---|
| **S** | Resense HEX-32 transducer frame. The shaft axis is +z_S. | All reported force results (tables, figures, reports) |
| **EE** | Force Dimension Lambda.7 end-effector frame | Logged `*_meas_EE_*` channels; equations of the physical models |

```
x_EE = -z_S      y_EE = y_S      z_EE = x_S
```

The sensor-to-tip lever points along -x_EE (= +z_S). Consequences:

- The **axial force** is Fx in EE and **Fz in S**.
- The transverse forces follow from the bending moments (EE frame, lever length L):

  ```
  r = [-L, 0, 0]   =>   M = r x F = [0, +L Fz, -L Fy]
  Fy = -Tz / L,   Fz = +Ty / L
  ```

- Tx_EE is torsion about the shaft and is not used to reconstruct forces.
- Bending moments cannot observe the axial force, so the moment-based models
  (P1, P2) take the axial component directly from the measured force.

The physical models compute in EE. Every result is rotated to S with the fixed
matrix `EE_TO_S` before it is scored or reported. The fitted models
(PA0–PA2, D1, D2) are trained directly on S-frame targets.

## Units

| Quantity | Unit | Note |
|---|---|---|
| Forces | N | |
| **Sensor moments** (`Tx/Ty/Tz_meas_EE_Nm`, `Tx/Ty/Tz_raw_sensor_Nm`) | **Nmm** | The column names end in `_Nm` for historical reasons; the values are Nmm. |
| Controller torque command and reference (`t*_cmd_Nm`, `t_delivered_*_Nm`) | Nm | Not used as model inputs |
| Lengths in `objects.yaml` | mm | Converted to m in `Geometry.from_config` |
| Young's modulus in `objects.yaml` | GPa | Converted to Pa |

With moments in Nmm and lengths in mm, M / L gives the force in N directly.

## CSV columns used (schema 7)

| Columns | Content |
|---|---|
| `Fx/Fy/Fz_meas_EE_N`, `Tx/Ty/Tz_meas_EE_Nm` | Measured sensor wrench in EE: the six **model inputs** |
| `f_delivered_x/y/z_EE_N` | Force delivered by the Lambda.7: the **reference** the models are scored against |
| `fx/fy/fz_cmd_N` | Commanded force |
| `ramp_state` | 0 ramp up, 1 dwell, 2 ramp down; only dwell rows (1) are analysed |
| `pattern_index`, `pattern_label`, `series_id`, `step_index` | Identify each dwell and staircase |
| `split`, `family` | Data split and pattern type (see below) |
| `sensor_msg_count` | Sensor message counter, used to remove repeated rows |
| `p_relative_x/y/z_EE_m` | End-effector displacement, used for the apparent EI |
| `t_rel_s` | Time since the start of the run |

At load time the code adds sensor-frame copies of the force columns
(`*_S_N`). Older files (schema v2/v3) are migrated by `data.migrate_legacy`;
the audit table marks checks that cannot be done on migrated files as MISSING.

## Sampling unit: the dwell

Each commanded load is held for a dwell (a few seconds at ~200 Hz). The rows of
one dwell are not independent: the EMA filter correlates them, the sensor
publishes slower than the logger writes, and the command is constant. Each
dwell is therefore reduced to one point:

1. drop the first `dwell.settle_frac` of the dwell (settling);
2. remove repeated rows of the same sensor message;
3. keep the mean, the standard deviation and the effective sample count `n_eff`.

Treating rows as independent would shrink every standard error by about a
factor of 17 here. All metrics use dwell means.

## Data splits

| `split` value | Use |
|---|---|
| `train` | Model fitting; a share `evaluation.val_frac` of the training units is held out for validation. A staircase is held out as a whole, never split. |
| `test_A` | **Accuracy**: held-out loads inside the training range (interpolation) |
| `test_B` | **Robustness**: loads at 1.5 × the training amplitude (extrapolation) |
| `anchor` | Repeated identical loads at the start, middle and end of a run: drift |
| `preload` | Excluded from the analysis |

Pattern families used by the diagnostics: `stair_bidir` (ascending and
descending staircase, hysteresis), `stair_asc`, `reference` (repeated identical
load, repeatability), `anchor`.

Metrics are always reported per split and never pooled across splits. The
comparison of Test A and Test B is itself a result.

## Models

| ID | Name | Family | Inputs | Fitted parameters |
|---|---|---|---|---|
| P0 | Direct-force equilibrium | Physical Models | measured force | none |
| P1 | Fixed-lever moment model | Physical Models | axial force + bending moments, fixed L | none |
| P2 | Compliant-lever moment model | Physical Models | as P1, lever deformed with handbook EI | none |
| PA0 | Affine direct-force model | Linear Error Models | [P0 estimate, 3 sensor moments] | A (3×6), b (3) |
| PA1 | Affine fixed-lever model | Linear Error Models | [P1 estimate, 3 sensor moments] | A (3×6), b (3) |
| PA2 | Affine compliant-lever model | Linear Error Models | [P2 estimate, 3 sensor moments] | A (3×6), b (3) |
| D1 | Quadratic regression | Data-Driven | 6 sensor channels, degree-2 terms | 27 coefficients per axis + intercept |
| D2 | Multilayer perceptron | Data-Driven | 6 sensor channels | network weights (2 × 64 hidden units) |

PA0–PA2 and D1 use ridge regression on standardised features
(`models.ridge_alpha`). Geometry and EI are never fitted; they come from
`configs/objects.yaml`. Every model is fitted per run.
