"""Object geometry and the eight force-reconstruction models.

Models are fitted on dwell means, not on CSV rows. There are three families:

  Physical Models (no fitted parameters)
    P0   direct-force equilibrium, F_tip = F_sensor
    P1   fixed-lever free-cantilever moment model, F_perp = (M [Nmm] x a) / L [mm]
    P2   compliant-lever moment model: handbook EI and a load-dependent
         3D deformed lever vector r_def(F)

  Linear Error Models (affine A, b on six inputs)
    PA0  [P0 force estimate, measured sensor moments]
    PA1  [P1 force estimate, measured sensor moments]
    PA2  [P2 force estimate, measured sensor moments]

  Data-Driven (six measured sensor channels as inputs)
    D1   degree-2 polynomial regression
    D2   multilayer perceptron

Bending moments cannot observe the shaft-axial force. The sensor-to-tip lever
is -x_EE (= +z_S), so Fx_EE is axial: P1 and P2 take it directly from the
measured force and use Ty_EE / Tz_EE only for the transverse components.
EI and the lever lengths are never fitted; they come from configs/objects.yaml.
All model predictions are returned in the sensor frame S.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

from .constants import EE_AXIAL_INDEX, EE_LATERAL_INDICES, EE_SHAFT_AXIS, rotate_ee_to_s

if TYPE_CHECKING:
    from .settings import ModelSettings, ObjectGeometry


@dataclass
class Geometry:
    """Fixed object geometry used by the force-reconstruction hypotheses.

    The three lengths are deliberately kept separate:
      * ``exposed_length``: physical shaft length outside the bare object [m]
      * ``sensor_to_tip_length``: lever arm from the F/T sensor origin to the
        distal load point [m]; this is the L used in M = r x F
      * ``bending_length``: unsupported length from the clamp to the distal
        load point [m]; this is the L used only for the compliance assumption

    EI and all lengths are fixed from geometry/material assumptions. They are
    never identified from calibration data.
    """
    object_name: str
    exposed_length: float
    sensor_to_tip_length: float
    bending_length: float
    shaft_od: float
    shaft_wall: float
    E: float
    opposite_reaction_signs: bool = False

    def __post_init__(self):
        self.axis = EE_SHAFT_AXIS.copy()
        if min(self.exposed_length, self.sensor_to_tip_length,
               self.bending_length, self.shaft_od, self.shaft_wall, self.E) <= 0:
            raise ValueError("geometry/material values must be positive")
        if self.sensor_to_tip_length < self.bending_length:
            raise ValueError("sensor_to_tip_length must be >= bending_length")

    @classmethod
    def from_config(cls, object_name: str, geometry: "ObjectGeometry") -> "Geometry":
        """Create the geometry of one object from its entry in configs/objects.yaml.

        The file uses mm and GPa; this class uses m and Pa.
        """
        return cls(object_name=object_name,
                   exposed_length=geometry.exposed_length_mm / 1000.0,
                   sensor_to_tip_length=geometry.sensor_to_tip_length_mm / 1000.0,
                   bending_length=geometry.bending_length_mm / 1000.0,
                   shaft_od=geometry.shaft_outer_diameter_mm / 1000.0,
                   shaft_wall=geometry.shaft_wall_mm / 1000.0,
                   E=geometry.youngs_modulus_GPa * 1e9)

    @property
    def sensor_to_tip_length_mm(self) -> float:
        return 1e3 * self.sensor_to_tip_length

    @property
    def bending_length_mm(self) -> float:
        return 1e3 * self.bending_length

    @property
    def rigid_offset(self) -> float:
        """Sensor-origin to clamp contribution that is treated as rigid [m]."""
        return self.sensor_to_tip_length - self.bending_length

    def section(self) -> Dict[str, float]:
        d_i = max(0.0, self.shaft_od - 2.0 * self.shaft_wall)
        I = np.pi * (self.shaft_od ** 4 - d_i ** 4) / 64.0
        return {
            "outer_diameter_m": self.shaft_od,
            "inner_diameter_m": d_i,
            "I_m4": I,
            "EI_Nm2": self.E * I,
            "solid": bool(d_i <= 1e-15),
        }

    def assumption_table(self) -> pd.DataFrame:
        """Geometry/material quantities used only as declared assumptions.

        Nothing in this table is fitted from the calibration runs. The
        3D-compliant model uses the handbook EI together with the fixed
        unsupported bending length to compute a load-dependent geometric
        shortening of the deformed 3D moment arm for a free cantilever.
        """
        sec = self.section()
        rows = [
            {"quantity": "object", "value": self.object_name, "unit": ""},
            {"quantity": "exposed_length", "value": self.exposed_length, "unit": "m"},
            {"quantity": "sensor_to_tip_length", "value": self.sensor_to_tip_length, "unit": "m"},
            {"quantity": "bending_length", "value": self.bending_length, "unit": "m"},
            {"quantity": "rigid_offset", "value": self.rigid_offset, "unit": "m"},
            {"quantity": "outer_diameter", "value": self.shaft_od, "unit": "m"},
            {"quantity": "inner_diameter", "value": sec["inner_diameter_m"], "unit": "m"},
            {"quantity": "Youngs_modulus_assumed", "value": self.E, "unit": "Pa"},
            {"quantity": "I", "value": sec["I_m4"], "unit": "m^4"},
            {"quantity": "EI_handbook", "value": sec["EI_Nm2"], "unit": "N m^2"},
        ]
        for F in (1.0, 3.0, 4.0, 6.0):
            f_vec = np.array([0.0, F, 0.0])
            r_def = self.deformed_lever_vector(f_vec)
            delta = self.tip_deflection(F)
            rows.extend([
                {"quantity": f"tip_deflection_cantilever_{F:g}N", "value": delta, "unit": "m"},
                {"quantity": f"r_def_x_cantilever_{F:g}N", "value": r_def[0], "unit": "m"},
                {"quantity": f"r_def_y_cantilever_{F:g}N", "value": r_def[1], "unit": "m"},
                {"quantity": f"r_def_z_cantilever_{F:g}N", "value": r_def[2], "unit": "m"},
                {"quantity": f"r_def_norm_cantilever_{F:g}N", "value": np.linalg.norm(r_def), "unit": "m"},
            ])
        return pd.DataFrame(rows)

    def tip_deflection(self, force_N: float) -> float:
        """Small-deflection Euler-Bernoulli tip deflection of a free cantilever [m].

        delta = F L^3 / (3 EI)
        """
        EI = self.section()["EI_Nm2"]
        return abs(float(force_N)) * self.bending_length ** 3 / (3.0 * EI)

    def deformed_lever_vector(self, force_N: np.ndarray) -> np.ndarray:
        """Approximate the full 3D sensor-to-tip lever vector after bending.

        The free cantilever is assumed to bend in the direction of the force
        component transverse to the nominal shaft axis ``a``.  With circular
        section geometry, the Euler-Bernoulli deflection vector is

            delta_vec = F_perp * L_bend^3 / (3 EI).

        To keep a simple geometric large-displacement correction without fitting
        any additional parameter, the unsupported segment is treated as an
        inextensible chord.  Its axial projection is therefore

            L_proj = sqrt(L_bend^2 - ||delta_vec||^2).

        The complete sensor-to-tip vector is then

            r_def = (L_rigid + L_proj) a + delta_vec.

        This vector, not a scalar corrected length, is used by P2/PA2 in the
        cross-product equation M = r_def x F.  EI and all lengths remain fixed
        handbook/geometry assumptions; none is identified from calibration data.
        """
        f = np.asarray(force_N, dtype=float).reshape(3)
        a = self.axis
        f_perp = f - np.dot(f, a) * a
        EI = self.section()["EI_Nm2"]
        delta_vec = f_perp * (self.bending_length ** 3 / (3.0 * EI))

        Lb = self.bending_length
        delta_mag = float(np.linalg.norm(delta_vec))
        # The linear beam formula ceases to be credible when the deflection is
        # comparable with L.  Keep the numerical model finite and flag such
        # cases through warnings/diagnostics rather than fitting around them.
        max_delta = 0.95 * Lb
        if delta_mag > max_delta and delta_mag > 0:
            delta_vec = delta_vec * (max_delta / delta_mag)
            delta_mag = max_delta

        projected = float(np.sqrt(max(Lb * Lb - delta_mag * delta_mag,
                                      (0.05 * Lb) ** 2)))
        return (self.rigid_offset + projected) * a + delta_vec

    def adjusted_sensor_to_tip_length(self, force_N: float) -> float:
        """Deprecated scalar diagnostic: x-projection for a +y transverse load.

        The reconstruction model no longer uses a scalar effective length.  This
        helper is retained only for backward-compatible diagnostics.
        """
        r = self.deformed_lever_vector(np.array([0.0, float(force_N), 0.0]))
        return float(np.dot(r, self.axis))

    def warnings(self) -> List[str]:
        sec = self.section()
        out = [
            f"Object '{self.object_name}': measured sensor moments are interpreted in Nmm, "
            f"so moment conversion uses M[Nmm]/L[mm] with fixed sensor-origin-to-tip "
            f"length {1e3*self.sensor_to_tip_length:.1f} mm; "
            f"beam compliance uses the unsupported clamped length "
            f"{1e3*self.bending_length:.1f} mm.",
            f"Handbook bending stiffness is an assumption, not an identified "
            f"parameter: E={self.E/1e9:.1f} GPa, I={sec['I_m4']:.4e} m^4, "
            f"EI={sec['EI_Nm2']:.4g} N m^2.",
            "The sensor-to-tip lever is -x_EE (= +z_S), so Fx_EE is the shaft-axial "
            "force channel. P1 and P2 pass Fx directly from the measured EE force. "
            "P1 reconstructs Fy=-Tz/L and Fz=+Ty/L; P2 reconstructs Fy/Fz from "
            "the full deformed 3D lever while keeping Fx equal to the direct force "
            "measurement. Tx_EE is torsion and is not used to infer force. All "
            "force results reported by this package (P0-D2, measured/delivered/"
            "commanded diagnostics) are then rotated into the sensor frame S, "
            "where the shaft-axial component is Fz, not Fx.",
        ]
        d6 = self.tip_deflection(6.0)
        if d6 / self.bending_length > 0.10:
            out.append(
                f"At 6 N the free-cantilever small-deflection estimate gives "
                f"delta/L={d6/self.bending_length:.2f}; treat the compliant "
                "3D deformation correction as a sensitivity hypothesis, not a validated "
                "large-deflection beam solution.")
        return out


def direct_force(X: np.ndarray, geo: Geometry) -> np.ndarray:
    """P0: direct-force equilibrium in the canonical EE frame.

    X[:,0:3] are the logged ``Fx/Fy/Fz_meas_EE_N`` channels. They already obey
    x_EE=-z_S, y_EE=y_S, z_EE=x_S, so P0 MUST NOT permute them again.
    The shaft-axial channel used by the mechanics model is Fx_EE.
    """
    F = np.asarray(X, dtype=float)[:, :3].copy()
    if geo.opposite_reaction_signs:
        F = -F
    return F


def _moment_force_once(M_Nmm: np.ndarray, geo: Geometry, factor: float,
                       length_mm: np.ndarray) -> np.ndarray:
    """Invert bending moment into transverse EE-frame force using Nmm and mm.

    The sensor-to-tip lever direction is -x_EE. For r=[-L,0,0],

        M = r x F = [0, +L Fz, -L Fy].

    Therefore

        Fx is unobservable from bending moment,
        Fy = -Mz / L,
        Fz = +My / L.

    The compact vector expression ``F_perp = (M x a)/L`` with a=-x_EE gives
    exactly this mapping.  Tx_EE is torsion about the shaft and contributes zero
    to the bending-force inversion.
    """
    M_Nmm = np.asarray(M_Nmm, dtype=float)
    L_mm = np.asarray(length_mm, dtype=float).reshape(-1)
    if np.any(L_mm <= 0):
        raise ValueError("moment-model lever length must be positive")
    a = geo.axis
    F = factor * np.cross(M_Nmm, a[None, :]) / L_mm[:, None]
    if geo.opposite_reaction_signs:
        F = -F
    return F


def moment_force(X: np.ndarray, geo: Geometry) -> np.ndarray:
    """P1: fixed-lever free-cantilever force reconstruction in EE.

    The sensor-to-tip lever is r=[-L,0,0] in EE, hence Fx is axial and cannot be
    inferred from bending moment.  It is passed through directly from the force
    channel.  Only the two transverse components are reconstructed from moments:

        Fx_EE =  Fx_sensor,EE
        Fy_EE = -Tz_EE / L
        Fz_EE = +Ty_EE / L

    Tx_EE is torsion about the shaft and is intentionally ignored by P1.
    Sensor moments are Nmm and L is mm.
    """
    X = np.asarray(X, dtype=float)
    M_Nmm = X[:, 3:6]
    L_mm = np.full(len(M_Nmm), geo.sensor_to_tip_length_mm, dtype=float)
    F = _moment_force_once(M_Nmm, geo, 1.0, L_mm)
    F[:, EE_AXIAL_INDEX] = direct_force(X, geo)[:, EE_AXIAL_INDEX]
    return F


def _transverse_force_from_moment_known_axial(M_Nmm: np.ndarray, r_mm: np.ndarray,
                                                Faxial: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Solve M = r x F for the two transverse components with axial Fx fixed.

    Coordinates are EE.  The known axial component is Fx; the unknowns are Fy and
    Fz.  For the nominal r=[-L,0,0], this reduces exactly to

        Fy=-Tz/L,  Fz=+Ty/L.

    The full three moment equations are used in least squares for P2 when the
    compliant lever has transverse components.
    """
    M_Nmm = np.asarray(M_Nmm, dtype=float)
    r_mm = np.asarray(r_mm, dtype=float)
    Faxial = np.asarray(Faxial, dtype=float).reshape(-1)
    if len(M_Nmm) != len(r_mm) or len(Faxial) != len(r_mm):
        raise ValueError("M, r and Faxial must have the same sample count")

    out = np.zeros((len(r_mm), 3), dtype=float)
    out[:, EE_AXIAL_INDEX] = Faxial
    M_eff = float(sign) * M_Nmm
    e_ax = np.eye(3)[EE_AXIAL_INDEX]
    e1 = np.eye(3)[EE_LATERAL_INDICES[0]]
    e2 = np.eye(3)[EE_LATERAL_INDICES[1]]
    for i, r in enumerate(r_mm):
        rhs = M_eff[i] - np.cross(r, Faxial[i] * e_ax)
        A = np.column_stack((np.cross(r, e1), np.cross(r, e2)))
        lat, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        out[i, EE_LATERAL_INDICES[0]] = lat[0]
        out[i, EE_LATERAL_INDICES[1]] = lat[1]
    return out


def moment_force_compliant(X: np.ndarray, geo: Geometry, *,
                           max_iter: int, tol_N: float, relaxation: float) -> np.ndarray:
    """P2: compliant 3D moment reconstruction with direct axial Fx.

    Every quantity is expressed in EE. The nominal sensor-to-tip lever points
    along -x_EE. Fx is always copied from the direct EE force channel; only Fy/Fz
    are inferred from moment. The full deformed lever ``r_def(F)`` uses the fixed
    free-cantilever geometry and handbook EI. Since r_def depends on transverse
    force, the inversion is solved by fixed-point iteration. Tx_EE is torsion
    about the shaft; any Tx contribution in the overdetermined solve is treated
    only as a residual from nonideal geometry/coupling.
    """
    X = np.asarray(X, dtype=float)
    M_Nmm = X[:, 3:6]
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")

    Faxial_direct = direct_force(X, geo)[:, EE_AXIAL_INDEX]
    F = moment_force(X, geo)
    sign = -1.0 if geo.opposite_reaction_signs else 1.0

    for _ in range(max_iter):
        r_mm = np.vstack([geo.deformed_lever_vector(f) for f in F]) * 1e3
        F_new = _transverse_force_from_moment_known_axial(
            M_Nmm, r_mm, Faxial_direct, sign)
        step_lat = F_new[:, EE_LATERAL_INDICES] - F[:, EE_LATERAL_INDICES]
        F[:, EE_LATERAL_INDICES] = F[:, EE_LATERAL_INDICES] + relaxation * step_lat
        F[:, EE_AXIAL_INDEX] = Faxial_direct
        if len(F) == 0 or float(np.max(np.linalg.norm(step_lat, axis=1))) < tol_N:
            break

    return F


def _as_wrench(F: np.ndarray) -> np.ndarray:
    """Keep legacy reporting code compatible while modelling force only."""
    F = np.asarray(F, dtype=float)
    return np.hstack([F, np.zeros_like(F)])


def _ridge_fit(A: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge regression with intercept and training-set standardisation."""
    mu_a, mu_y = A.mean(0), y.mean(0)
    Ac, yc = A - mu_a, y - mu_y
    sd = Ac.std(0)
    sd[sd < 1e-12] = 1.0
    As = Ac / sd
    G = As.T @ As + alpha * np.eye(As.shape[1])
    W = np.linalg.solve(G, As.T @ yc)
    return {"W": W, "sd": sd, "mu_a": mu_a, "mu_y": mu_y}


def _ridge_predict(model, A: np.ndarray) -> np.ndarray:
    return ((A - model["mu_a"]) / model["sd"]) @ model["W"] + model["mu_y"]


def _poly2(A: np.ndarray) -> np.ndarray:
    """Degree-2 expansion with interactions."""
    cols = [A]
    n = A.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((A[:, i] * A[:, j])[:, None])
    return np.hstack(cols)


def build_models(Xtr, ytr, geo: Geometry, cfg: "ModelSettings", seed: int,
                 dth_tr=None) -> Dict[str, dict]:
    """Build the eight force-reconstruction hypotheses requested for the thesis.

    Model count with MLP enabled:
      P0 + P1 + P2 + PA0 + PA1 + PA2 + D1 + D2 = 8.

    P0/P1/P2 are Physical Models hypotheses. PA0/PA1/PA2 use the corresponding
    physics force estimate as the three force features and ALWAYS append the
    three measured sensor-moment channels. Thus every PA model is a six-input
    affine calibration; only the force representation differs between PA0,
    PA1 and PA2. D1 and D2 are Data-Driven models using the raw six measured
    sensor channels.
    """
    # ytr's three force columns are already sensor-frame (TARGET_COLS_S), so
    # every fitted model below (PA0/PA1/PA2/D1/D2) learns a direct EE-features
    # -> S-frame-target map and its raw prediction needs no further rotation.
    # The three Physical Models models (P0/P1/P2) are pure functions with nothing
    # to fit, so their EE-frame estimate is rotated to S explicitly below.
    yF = np.asarray(ytr, dtype=float)[:, :3]
    Xtr = np.asarray(Xtr, dtype=float)
    alpha = cfg.ridge_alpha
    # P2 with the iteration settings from configs/analysis.yaml
    compliant_force = partial(moment_force_compliant,
                              max_iter=cfg.compliant_lever.max_iter,
                              tol_N=cfg.compliant_lever.tol_N,
                              relaxation=cfg.compliant_lever.relaxation)
    models: Dict[str, dict] = {}

    # P0 -- direct-force equilibrium baseline.
    models["P0"] = {
        "predict": lambda X, dth=None: _as_wrench(rotate_ee_to_s(direct_force(X, geo))),
        "note": "baseline: F_tip = F_sensor; no fitted parameters; reported in sensor frame S"}

    # P1 -- free cantilever with fixed lever length.
    models["P1"] = {
        "predict": lambda X, dth=None: _as_wrench(rotate_ee_to_s(moment_force(X, geo))),
        "note": "free cantilever, fixed L: shaft-axial component (Fz_S) taken directly from "
                "the direct force channel, transverse Fx_S/Fy_S solved from the measured moments; "
                "reported in sensor frame S"}

    # P2 -- free cantilever with a load-dependent full 3D lever vector.
    models["P2"] = {
        "predict": lambda X, dth=None: _as_wrench(rotate_ee_to_s(compliant_force(X, geo))),
        "note": "free cantilever with fixed handbook-EI full 3D deformed lever; "
                "shaft-axial component (Fz_S) taken directly from the direct force channel, "
                "Fx_S/Fy_S solved from M=r x F; reported in sensor frame S"}

    # PA0/PA1/PA2 -- Linear Error Models affine models.
    #
    # All three affine models see the SAME three measured sensor moment channels.
    # The only difference is how the three force-like input channels are formed:
    #   PA0: direct measured force          = P0(X)
    #   PA1: fixed-lever moment force       = P1(X)
    #   PA2: compliant-lever moment force   = P2(X)
    #
    # Hence every PA model has six affine input features:
    #       z = [F_physics_x, F_physics_y, F_physics_z,
    #            Mx_sensor, My_sensor, Mz_sensor]
    # and fits
    #       F_tip = A z + b.
    # F_physics is rotated to sensor frame S before the fit, so both sides of
    # this affine map share one frame and the fitted A stays physically
    # interpretable (see _pa0_decomposition): a well-calibrated PA0 should be
    # close to diagonal with the shaft-axial gain on its (z, z) entry.
    #
    # The sensor moments are recorded in Nmm. _ridge_fit standardises every
    # feature on the training set, so mixing N and Nmm does not numerically
    # privilege the moment channels because of their raw scale.
    def pa_features(X, force_fn):
        X = np.asarray(X, dtype=float)
        F_phys = rotate_ee_to_s(force_fn(X, geo))
        M_sensor_Nmm = X[:, 3:6]
        return np.hstack([F_phys, M_sensor_Nmm])

    # PA0 -- direct-force representation + all three sensor moments.
    Z_pa0 = pa_features(Xtr, direct_force)
    aff_pa0 = _ridge_fit(Z_pa0, yF, alpha)
    models["PA0"] = {
        "predict": lambda X, dth=None, aff=aff_pa0: _as_wrench(
            _ridge_predict(aff, pa_features(X, direct_force))),
        "note": "affine on [P0 direct-force estimate, M_sensor]; six inputs total"}

    # PA1 -- fixed-lever moment-force representation + all three sensor moments.
    Z_pa1 = pa_features(Xtr, moment_force)
    aff_pa1 = _ridge_fit(Z_pa1, yF, alpha)
    models["PA1"] = {
        "predict": lambda X, dth=None, aff=aff_pa1: _as_wrench(
            _ridge_predict(aff, pa_features(X, moment_force))),
        "note": "affine on [P1 fixed-lever force estimate, M_sensor]; six inputs total"}

    # PA2 -- compliant 3D moment-force representation + all three sensor moments.
    Z_pa2 = pa_features(Xtr, compliant_force)
    aff_pa2 = _ridge_fit(Z_pa2, yF, alpha)
    models["PA2"] = {
        "predict": lambda X, dth=None, aff=aff_pa2: _as_wrench(
            _ridge_predict(aff, pa_features(X, compliant_force))),
        "note": "affine on [P2 compliant-lever force estimate, M_sensor]; six inputs total"}

    # D1 -- degree-2 polynomial on the complete measured wrench.
    poly = _ridge_fit(_poly2(Xtr), yF, alpha)
    models["D1"] = {
        "predict": lambda X, dth=None: _as_wrench(
            _ridge_predict(poly, _poly2(np.asarray(X, dtype=float)))),
        "note": "degree-2 polynomial regression from forces [N] and sensor moments [Nmm]"}

    # D2 -- bounded-capacity MLP on the same six measured wrench channels.
    if cfg.use_mlp:
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=tuple(cfg.mlp.hidden_layers),
                             max_iter=cfg.mlp.max_iter,
                             random_state=seed, early_stopping=False))
            mlp.fit(Xtr, yF)
            models["D2"] = {
                "predict": lambda X, dth=None: _as_wrench(
                    mlp.predict(np.asarray(X, dtype=float))),
                "note": "MLP regression from forces [N] and sensor moments [Nmm]"}
        except ImportError:
            pass

    return models


def geometry_diagnostics(geo: Geometry) -> Dict[str, object]:
    sec = geo.section()
    out: Dict[str, object] = {
        "object": geo.object_name,
        "exposed_length_mm": 1e3 * geo.exposed_length,
        "sensor_to_tip_length_mm": 1e3 * geo.sensor_to_tip_length,
        "bending_length_mm": 1e3 * geo.bending_length,
        "outer_diameter_mm": 1e3 * geo.shaft_od,
        "inner_diameter_mm": 1e3 * sec["inner_diameter_m"],
        "EI_handbook_Nm2": sec["EI_Nm2"],
        "EI_fitted": False,
        "length_fitted": False,
    }
    for F in (3.0, 4.0, 6.0):
        r = geo.deformed_lever_vector(np.array([0.0, F, 0.0]))
        out[f"deflection_cantilever_{F:g}N_mm"] = 1e3 * geo.tip_deflection(F)
        out[f"r_def_x_cantilever_{F:g}N_mm"] = 1e3 * r[0]
        out[f"r_def_y_cantilever_{F:g}N_mm"] = 1e3 * r[1]
        out[f"r_def_z_cantilever_{F:g}N_mm"] = 1e3 * r[2]
        out[f"r_def_norm_cantilever_{F:g}N_mm"] = 1e3 * np.linalg.norm(r)
    return out
