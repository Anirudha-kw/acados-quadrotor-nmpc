"""Benchmark scenarios and the closed-loop simulation runner."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import trajectory as tr
from .controller import NMPCConfig, QuadrotorNMPC
from .params import CRAZYFLIE, REST_HEIGHT
from .wind import WindConfig, WindField

if TYPE_CHECKING:
    from .simulator import CrazyflieSim

START = (0.0, 0.0, REST_HEIGHT)


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    trajectory: tr.Trajectory
    wind: WindConfig | None = None


def point_to_point(dt: float) -> Scenario:
    goal = (1.0, 1.0, 1.0)
    traj = tr.quintic(START, goal, 6.0, dt).then(tr.hold(goal, 3.0, dt))
    return Scenario("point_to_point", "Point-to-point", "Quintic take-off from the ground to (1, 1, 1) m", traj)


def circle(dt: float) -> Scenario:
    entry = (1.0, 0.0, 1.0)
    traj = (
        tr.quintic(START, entry, 4.0, dt)
        .then(tr.circle(center=(0.0, 0.0, 1.0), radius=1.0, period=4.0, n_loops=3, dt=dt, ramp_time=2.0))
        .then(tr.hold(entry, 2.0, dt))
    )
    return Scenario("circle", "Circle", "Three 1 m-radius loops at 1.6 m/s (4 s period)", traj)


def figure_eight(dt: float) -> Scenario:
    center = (0.0, 0.0, 1.0)
    traj = (
        tr.quintic(START, center, 4.0, dt)
        .then(tr.figure_eight(center, amplitude=(1.5, 0.6, 0.3), period=6.0, n_loops=2, dt=dt, ramp_time=2.0))
        .then(tr.hold(center, 2.0, dt))
    )
    return Scenario("figure_eight", "Figure-eight", "3D lemniscate, 3 m x 1.2 m, peaking at 2 m/s", traj)


def wind_hover(dt: float) -> Scenario:
    target = (0.0, 0.0, 1.0)
    traj = tr.quintic(START, target, 4.0, dt).then(tr.hold(target, 16.0, dt))
    return Scenario(
        "wind",
        "Wind rejection",
        "Hold position in a 0.8 m/s base wind plus gusts and turbulence (peaks of 1.7 m/s)",
        traj,
        wind=WindConfig(),
    )


SCENARIOS: dict[str, Callable[[float], Scenario]] = {
    "point_to_point": point_to_point,
    "circle": circle,
    "figure_eight": figure_eight,
    "wind": wind_hover,
}


@dataclass
class SimLog:
    """Closed-loop time series, one row per control step."""

    scenario: str
    dt: float
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    pos_ref: np.ndarray
    rotor_thrust: np.ndarray
    solve_time: np.ndarray
    status: np.ndarray
    disturbance: np.ndarray
    wind: np.ndarray
    ext_accel: np.ndarray  # true unmodelled acceleration (drag / m) over the preceding step
    extra: dict = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        return np.linalg.norm(self.x[:, 0:3] - self.pos_ref, axis=1)

    def metrics(self) -> dict[str, float]:
        e = self.error
        st = self.solve_time * 1e3
        return {
            "rmse_m": float(np.sqrt(np.mean(e**2))),
            "mean_error_m": float(np.mean(e)),
            "max_error_m": float(np.max(e)),
            "final_error_m": float(e[-1]),
            "solve_mean_ms": float(np.mean(st)),
            "solve_p99_ms": float(np.percentile(st, 99)),
            "solve_max_ms": float(np.max(st)),
            "nonzero_status": int(np.count_nonzero(self.status)),
            "steps": int(len(self.t)),
        }

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            scenario=self.scenario,
            dt=self.dt,
            t=self.t,
            x=self.x,
            u=self.u,
            pos_ref=self.pos_ref,
            rotor_thrust=self.rotor_thrust,
            solve_time=self.solve_time,
            status=self.status,
            disturbance=self.disturbance,
            wind=self.wind,
            ext_accel=self.ext_accel,
        )

    @classmethod
    def load(cls, path: str) -> SimLog:
        d = np.load(path)
        return cls(
            scenario=str(d["scenario"]),
            dt=float(d["dt"]),
            t=d["t"],
            x=d["x"],
            u=d["u"],
            pos_ref=d["pos_ref"],
            rotor_thrust=d["rotor_thrust"],
            solve_time=d["solve_time"],
            status=d["status"],
            disturbance=d["disturbance"],
            wind=d["wind"],
            ext_accel=d["ext_accel"],
        )


StepCallback = Callable[[int, "CrazyflieSim", QuadrotorNMPC, "SimLog"], None]


def run(
    scenario: Scenario,
    config: NMPCConfig | None = None,
    gui: bool = False,
    realtime: bool = False,
    on_step: StepCallback | None = None,
    controller: QuadrotorNMPC | None = None,
    verbose: bool = True,
) -> SimLog:
    """Fly ``scenario`` in closed loop and return the logged time series."""
    from .simulator import CrazyflieSim

    config = config or NMPCConfig()
    traj = scenario.trajectory
    if not np.isclose(traj.dt, config.dt):
        raise ValueError(f"trajectory dt {traj.dt} != controller dt {config.dt}")

    ctrl = controller or QuadrotorNMPC(CRAZYFLIE, config)
    wind = WindField(scenario.wind) if scenario.wind is not None else None
    n = len(traj)
    log = SimLog(
        scenario=scenario.name,
        dt=config.dt,
        t=traj.t.copy(),
        x=np.zeros((n, 13)),
        u=np.zeros((n, 4)),
        pos_ref=traj.pos.copy(),
        rotor_thrust=np.zeros((n, 4)),
        solve_time=np.zeros(n),
        status=np.zeros(n, dtype=int),
        disturbance=np.zeros((n, 3)),
        wind=np.zeros((n, 3)),
        ext_accel=np.zeros((n, 3)),
    )

    with CrazyflieSim(CRAZYFLIE, gui=gui, wind=wind, start_position=traj.pos[0]) as sim:
        x = sim.reset(traj.pos[0])
        ctrl.reset(x)
        wall0 = time.perf_counter()
        for k in range(n):
            u = ctrl.control(x, traj, k)
            info = ctrl.info
            log.x[k], log.u[k] = x, u
            log.rotor_thrust[k] = sim.rotor_forces(u)
            log.solve_time[k], log.status[k], log.disturbance[k] = info.solve_time, info.status, info.disturbance
            log.wind[k] = wind.velocity if wind is not None else 0.0
            if on_step is not None:
                on_step(k, sim, ctrl, log)
            if verbose and k % int(round(2.0 / config.dt)) == 0:
                print(
                    f"  t={traj.t[k]:5.2f}s  pos=[{x[0]:+.3f} {x[1]:+.3f} {x[2]:+.3f}]  "
                    f"err={log.error[k] * 1e3:6.1f} mm  solve={info.solve_time * 1e3:.2f} ms"
                )
            x = sim.step(u, config.dt)
            if k + 1 < n:
                log.ext_accel[k + 1] = sim.external_force / CRAZYFLIE.mass
            if gui:
                sim.follow_camera(x[:3])
                if realtime:
                    lag = wall0 + (k + 1) * config.dt - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
    return log
