"""Frame conventions: x_EE = -z_S, y_EE = y_S, z_EE = x_S."""
import numpy as np

from forcecal.constants import (EE_TO_S, S_TO_EE, ee_axis_to_s_label, rotate_ee_to_s,
                                rotate_ee_to_s_matrix)


def test_documented_axis_mapping():
    v_s = np.array([1.0, 2.0, 3.0])
    assert np.allclose(S_TO_EE @ v_s, [-3.0, 2.0, 1.0])


def test_rotation_round_trip():
    v = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(rotate_ee_to_s(v) @ S_TO_EE.T, v)
    assert np.allclose(EE_TO_S @ S_TO_EE, np.eye(3))


def test_shaft_axis_is_fz_in_sensor_frame():
    assert ee_axis_to_s_label("Fx") == "Fz"
    assert ee_axis_to_s_label("Fy") == "Fy"
    assert ee_axis_to_s_label("Fz") == "Fx"


def test_matrix_rotation_matches_vector_rotation():
    M = np.random.default_rng(1).normal(size=(3, 3))
    f_ee = np.array([0.3, -1.2, 2.0])
    assert np.allclose(rotate_ee_to_s_matrix(M) @ rotate_ee_to_s(f_ee), rotate_ee_to_s(M @ f_ee))
