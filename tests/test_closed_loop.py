"""Integration tests that need PyBullet and/or a working acados installation."""

import numpy as np
import pytest

from quadrotor_nmpc import CRAZYFLIE
from quadrotor_nmpc.model import quadrotor_dynamics

pb = pytest.importorskip("pybullet")


def test_pybullet_plant_matches_model():
    from quadrotor_nmpc.simulator import CrazyflieSim

    f = quadrotor_dynamics()["f"]
    u = np.array([1.2 * CRAZYFLIE.hover_thrust, 1e-4, -1e-4, 5e-5])
    with CrazyflieSim() as sim:
        x = x0 = sim.reset((0, 0, 1.0))
        for _ in range(10):
            x = sim.step(u, 0.02)
    # Reference: RK4 on the CasADi model with a fine step.
    xm, h = x0.copy(), 1e-3
    for _ in range(200):
        k1 = np.array(f(xm, u, np.zeros(3))).ravel()
        k2 = np.array(f(xm + h / 2 * k1, u, np.zeros(3))).ravel()
        k3 = np.array(f(xm + h / 2 * k2, u, np.zeros(3))).ravel()
        k4 = np.array(f(xm + h * k3, u, np.zeros(3))).ravel()
        xm += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(x[:3], xm[:3], atol=1e-3)
    assert np.allclose(x[10:], xm[10:], atol=1e-2)


def test_vehicle_rests_on_ground():
    from quadrotor_nmpc.simulator import REST_HEIGHT, CrazyflieSim

    with CrazyflieSim() as sim:
        x = sim.reset()
        for _ in range(25):
            x = sim.step(np.zeros(4), 0.02)
    assert abs(x[2] - REST_HEIGHT) < 1e-3


@pytest.mark.slow
def test_nmpc_point_to_point(tmp_path):
    pytest.importorskip("acados_template")
    from quadrotor_nmpc.controller import NMPCConfig
    from quadrotor_nmpc.scenarios import point_to_point, run

    cfg = NMPCConfig(build_dir=str(tmp_path))
    log = run(point_to_point(cfg.dt), cfg, verbose=False)
    m = log.metrics()
    assert m["nonzero_status"] == 0
    assert m["rmse_m"] < 0.02
    assert m["final_error_m"] < 0.005
    assert m["solve_max_ms"] < 1e3 * cfg.dt
