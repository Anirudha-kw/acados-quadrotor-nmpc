import numpy as np
import pytest

from quadrotor_nmpc import CRAZYFLIE
from quadrotor_nmpc import trajectory as tr
from quadrotor_nmpc.scenarios import SCENARIOS

DT = 0.02


def test_quintic_boundary_conditions():
    t = tr.quintic((0, 0, 0), (1, 2, 3), 4.0, DT)
    assert np.allclose(t.pos[0], 0) and np.allclose(t.pos[-1], [1, 2, 3])
    assert np.allclose(t.vel[[0, -1]], 0) and np.allclose(t.acc[[0, -1]], 0, atol=1e-9)
    assert np.isclose(t.duration, 4.0)


@pytest.mark.parametrize(
    "traj",
    [
        tr.quintic((0, 0, 0), (1, -1, 2), 3.0, DT, v1=(0.5, 0, 0)),
        tr.circle((0, 0, 1), 1.0, 4.0, 2, DT),
        tr.figure_eight((0, 0, 1), (1.5, 0.6, 0.3), 6.0, 2, DT),
    ],
    ids=["quintic", "circle", "figure_eight"],
)
def test_derivatives_match_finite_differences(traj):
    vel_fd = np.gradient(traj.pos, DT, axis=0)[1:-1]
    acc_fd = np.gradient(traj.vel, DT, axis=0)[1:-1]
    assert np.max(np.abs(vel_fd - traj.vel[1:-1])) < 5e-3
    assert np.max(np.abs(acc_fd - traj.acc[1:-1])) < 2e-2


@pytest.mark.parametrize("make", [tr.circle, tr.figure_eight])
def test_periodic_paths_start_and_end_at_rest(make):
    traj = make((0, 0, 1), 1.0, 4.0, 2, DT) if make is tr.circle else make((0, 0, 1), period=4.0, dt=DT)
    assert np.allclose(traj.vel[[0, -1]], 0, atol=1e-9)
    assert np.allclose(traj.acc[[0, -1]], 0, atol=1e-9)
    assert np.allclose(traj.pos[0], traj.pos[-1], atol=1e-9)


def test_then_rejects_discontinuities():
    a = tr.hold((0, 0, 1), 1.0, DT)
    with pytest.raises(ValueError, match="discontinuous pos"):
        a.then(tr.hold((1, 0, 1), 1.0, DT))
    # The original circle demo jumped from (0, 0, 1) to (1, 0, 1); this must now be rejected.
    with pytest.raises(ValueError):
        a.then(tr.circle((0, 0, 1), 1.0, 4.0, 1, DT))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_continuous_and_feasible(name):
    traj = SCENARIOS[name](DT).trajectory
    assert np.max(np.abs(np.diff(traj.pos, axis=0))) < 0.05  # < 2.5 m/s at 50 Hz
    x_ref, u_ref = tr.flat_to_state(traj)
    assert np.all(u_ref[:, 0] < 0.8 * CRAZYFLIE.max_total_thrust)
    assert np.allclose(np.linalg.norm(x_ref[:, 6:10], axis=1), 1.0)


def test_flatness_at_hover_and_under_acceleration():
    x_ref, u_ref = tr.flat_to_state(tr.hold((0, 0, 1), 0.1, DT))
    assert np.allclose(x_ref[:, 6:10], [1, 0, 0, 0]) and np.allclose(u_ref[:, 0], CRAZYFLIE.hover_thrust)

    acc = tr.Trajectory(np.arange(3) * DT, np.zeros((3, 3)), np.zeros((3, 3)), np.tile([2.0, 0, 0], (3, 1)))
    x_ref, u_ref = tr.flat_to_state(acc)
    # Accelerating along +x needs a positive pitch, i.e. qy > 0.
    assert np.all(x_ref[:, 8] > 0) and np.allclose(x_ref[:, [7, 9]], 0)
    assert np.allclose(u_ref[:, 0], CRAZYFLIE.mass * np.hypot(2.0, CRAZYFLIE.gravity))
