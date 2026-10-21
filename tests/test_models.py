"""Physical models recover known forces; all eight models can be built."""
import numpy as np

from forcecal.models import Geometry, direct_force, moment_force, moment_force_compliant


def geometry(settings):
    return Geometry.from_config(settings.object_name, settings.object_geometry)


def compliant_kwargs(settings):
    c = settings.models.compliant_lever
    return dict(max_iter=c.max_iter, tol_N=c.tol_N, relaxation=c.relaxation)


def test_p0_passes_measured_force_through(settings):
    X = np.zeros((1, 6))
    X[0, :3] = [1.2, -0.7, 2.5]
    assert np.allclose(direct_force(X, geometry(settings))[0], [1.2, -0.7, 2.5])


def test_p1_recovers_force_from_moment(settings):
    geo = geometry(settings)
    F = np.array([1.25, 4.0, -1.5])
    L = geo.sensor_to_tip_length_mm
    X = np.zeros((1, 6))
    X[0, 0] = F[0]            # axial force is taken directly
    X[0, 3] = 999.0           # torsion must not change the force
    X[0, 4] = L * F[2]        # Ty = +L Fz
    X[0, 5] = -L * F[1]       # Tz = -L Fy
    assert np.allclose(moment_force(X, geo)[0], F, atol=1e-12)


def test_deformed_lever_shortens_under_transverse_load(settings):
    geo = geometry(settings)
    r = geo.deformed_lever_vector(np.array([0.0, 6.0, 0.0]))
    assert abs(r[0]) < geo.sensor_to_tip_length
    assert r[1] > 0.0


def test_p2_recovers_force_from_deformed_lever(settings):
    geo = geometry(settings)
    F = np.array([1.25, 4.0, -1.5])
    r_mm = 1e3 * geo.deformed_lever_vector(F)
    X = np.zeros((1, 6))
    X[0, 0] = F[0]
    X[0, 3:6] = np.cross(r_mm, F)
    est = moment_force_compliant(X, geo, **compliant_kwargs(settings))[0]
    assert np.isclose(est[0], F[0], atol=1e-12)
    assert np.allclose(est[1:], F[1:], atol=1e-5)
