import numpy as np
import pytest

from quadrotor_nmpc import CRAZYFLIE
from quadrotor_nmpc.model import NX, quadrotor_dynamics
from quadrotor_nmpc.rotations import quat_to_euler, quat_to_rotmat, rotmat_to_quat

f = quadrotor_dynamics()["f"]


def xdot(x, u, d=(0.0, 0.0, 0.0)):
    return np.array(f(x, u, d)).ravel()


def hover_state():
    x = np.zeros(NX)
    x[6] = 1.0
    return x


def test_mixer_is_invertible_and_symmetric():
    M = CRAZYFLIE.mixer()
    assert np.allclose(M @ CRAZYFLIE.inverse_mixer(), np.eye(4))
    f = CRAZYFLIE.inverse_mixer() @ np.array([CRAZYFLIE.hover_thrust, 0, 0, 0])
    assert np.allclose(f, CRAZYFLIE.hover_thrust / 4)


def test_torque_limits_are_achievable():
    p = CRAZYFLIE
    f = p.inverse_mixer() @ np.array([2 * p.max_motor_thrust, p.max_roll_pitch_torque, 0, 0])
    assert np.all(f >= -1e-12) and np.all(f <= p.max_motor_thrust + 1e-12)


def test_hover_is_equilibrium():
    assert np.allclose(xdot(hover_state(), [CRAZYFLIE.hover_thrust, 0, 0, 0]), 0.0)


def test_free_fall_and_disturbance():
    acc = xdot(hover_state(), [0, 0, 0, 0], d=[0.5, -0.2, 0.0])[3:6]
    assert np.allclose(acc, [0.5, -0.2, -CRAZYFLIE.gravity])


def test_positive_torque_gives_positive_body_rate():
    for axis in range(3):
        tau = np.zeros(3)
        tau[axis] = 1e-4
        wdot = xdot(hover_state(), [CRAZYFLIE.hover_thrust, *tau])[10:13]
        assert wdot[axis] > 0 and np.allclose(np.delete(wdot, axis), 0.0)


def test_quaternion_kinematics_preserve_norm():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = hover_state()
        q = rng.normal(size=4)
        x[6:10] = q / np.linalg.norm(q)
        x[10:13] = rng.normal(size=3)
        qdot = xdot(x, [0.3, 0, 0, 0])[6:10]
        assert abs(np.dot(x[6:10], qdot)) < 1e-12


def test_tilted_thrust_points_along_body_z():
    x = hover_state()
    x[6:10] = rotmat_to_quat(_rot_y(np.radians(10)))  # pitch forward
    acc = xdot(x, [CRAZYFLIE.hover_thrust, 0, 0, 0])[3:6]
    assert acc[0] > 0 and acc[2] < 0


def _rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


@pytest.mark.parametrize("seed", range(5))
def test_rotation_round_trip(seed):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    q *= np.sign(q[0])
    assert np.allclose(rotmat_to_quat(quat_to_rotmat(q)), q)


def test_euler_of_pitch():
    rpy = quat_to_euler(rotmat_to_quat(_rot_y(0.3)))
    assert np.allclose(rpy, [0.0, 0.3, 0.0])
