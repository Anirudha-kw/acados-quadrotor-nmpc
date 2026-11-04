"""Nonlinear MPC for quadrotor trajectory tracking, built on acados.

Optimal control problem solved every ``dt`` seconds (real-time iteration)::

    min   sum_k ||x_k - x_ref,k||^2_Q + ||u_k - u_ref,k||^2_R  +  ||x_N - x_ref,N||^2_P
    s.t.  x_{k+1} = F_RK4(x_k, u_k; d_hat)            (quaternion rigid-body model)
          0 <= M^-1 u_k <= f_max                     (per-rotor thrust limits)
          |omega_k| <= omega_max                     (soft, L1 + L2 penalty)
          |u_0 - u_prev| <= du_max * dt              (actuator slew-rate limits)
          x_0 = x_measured

``x_ref``/``u_ref`` come from differential flatness, so the attitude and
thrust references are dynamically consistent with the position reference.
``d_hat`` is an external-acceleration estimate from a disturbance observer,
which gives offset-free tracking under wind and model mismatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .model import NP, NU, NX, export_acados_model
from .params import CRAZYFLIE, QuadrotorParams
from .rotations import quat_to_rotmat
from .trajectory import Trajectory, flat_to_state


@dataclass
class NMPCConfig:
    dt: float = 0.02  # control period [s]
    horizon: int = 25  # number of shooting intervals (0.5 s look-ahead)

    # Stage cost weights.
    w_pos: tuple[float, float, float] = (400.0, 400.0, 600.0)
    w_vel: tuple[float, float, float] = (20.0, 20.0, 30.0)
    w_quat: tuple[float, float, float, float] = (0.0, 30.0, 30.0, 30.0)
    w_rate: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Input weights act on inputs normalised by hover thrust / max torque.
    w_thrust: float = 1.0
    w_torque: tuple[float, float, float] = (1.0, 1.0, 1.0)
    terminal_scale: float = 5.0  # terminal weight = terminal_scale * stage weight

    max_body_rate: float = 6.0  # soft bound [rad/s]
    slack_l1: float = 100.0
    slack_l2: float = 100.0
    max_thrust_rate: float | None = 20.0  # [N/s]
    max_torque_rate: float | None = 0.5  # [N m/s]

    use_observer: bool = True
    observer_time_constant: float = 0.15  # [s]
    observer_limit: float = 5.0  # [m/s^2]

    build_dir: str = "build"
    name: str = "crazyflie_nmpc"


class DisturbanceObserver:
    """Low-pass estimate of unmodelled acceleration from velocity measurements.

    ``d_raw = (v_k - v_{k-1}) / dt - (R(q) e_z T / m - g e_z)`` is filtered with a
    first-order low-pass of time constant ``tau``.
    """

    def __init__(self, params: QuadrotorParams, dt: float, time_constant: float, limit: float):
        self.params = params
        self.dt = dt
        self.alpha = dt / (time_constant + dt)
        self.limit = limit
        self.reset()

    def reset(self) -> None:
        self.estimate = np.zeros(3)

    def update(self, x_prev: np.ndarray, u_prev: np.ndarray, x_now: np.ndarray) -> np.ndarray:
        p = self.params
        a_meas = (x_now[3:6] - x_prev[3:6]) / self.dt
        z_axis = 0.5 * (quat_to_rotmat(x_prev[6:10])[:, 2] + quat_to_rotmat(x_now[6:10])[:, 2])
        a_model = z_axis * u_prev[0] / p.mass - np.array([0.0, 0.0, p.gravity])
        self.estimate += self.alpha * (a_meas - a_model - self.estimate)
        self.estimate = np.clip(self.estimate, -self.limit, self.limit)
        return self.estimate


@dataclass
class SolveInfo:
    status: int = 0
    solve_time: float = 0.0  # wall-clock time of the acados solve call (time_tot) [s]
    disturbance: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_ref: np.ndarray = field(default_factory=lambda: np.zeros(NX))
    predicted_positions: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))


class QuadrotorNMPC:
    """Real-time-iteration NMPC with optional disturbance observer."""

    def __init__(self, params: QuadrotorParams = CRAZYFLIE, config: NMPCConfig | None = None, verbose: bool = False):
        from acados_template import AcadosOcpSolver

        self.params = params
        self.cfg = config or NMPCConfig()
        self.ocp = self._build_ocp()
        os.makedirs(self.cfg.build_dir, exist_ok=True)
        json_file = os.path.join(self.cfg.build_dir, f"{self.cfg.name}.json")
        self.solver = AcadosOcpSolver(self.ocp, json_file=json_file, verbose=verbose)
        self.observer = DisturbanceObserver(
            params, self.cfg.dt, self.cfg.observer_time_constant, self.cfg.observer_limit
        )
        self.reset()

    # ------------------------------------------------------------------ setup
    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        c, p = self.cfg, self.params
        Q = np.diag(np.concatenate([c.w_pos, c.w_vel, c.w_quat, c.w_rate]))
        scale = np.array([p.hover_thrust, p.max_roll_pitch_torque, p.max_roll_pitch_torque, p.max_yaw_torque])
        R = np.diag(np.array([c.w_thrust, *c.w_torque]) / scale**2)
        return Q, R

    def _build_ocp(self):
        from acados_template import AcadosOcp

        c, p = self.cfg, self.params
        ocp = AcadosOcp()
        ocp.model = export_acados_model(p, name=c.name)
        ocp.solver_options.N_horizon = c.horizon
        ocp.solver_options.tf = c.horizon * c.dt

        # Linear least-squares tracking cost on y = [x; u].
        Q, R = self._weights()
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([[Q, np.zeros((NX, NU))], [np.zeros((NU, NX)), R]])
        ocp.cost.W_e = c.terminal_scale * Q
        ocp.cost.Vx = np.vstack([np.eye(NX), np.zeros((NU, NX))])
        ocp.cost.Vu = np.vstack([np.zeros((NX, NU)), np.eye(NU)])
        ocp.cost.Vx_e = np.eye(NX)
        ocp.cost.yref = np.zeros(NX + NU)
        ocp.cost.yref_e = np.zeros(NX)

        # Input box (collective thrust and torques) ...
        self.lbu = np.array([0.0, -p.max_roll_pitch_torque, -p.max_roll_pitch_torque, -p.max_yaw_torque])
        self.ubu = np.array([p.max_total_thrust, p.max_roll_pitch_torque, p.max_roll_pitch_torque, p.max_yaw_torque])
        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = self.lbu.copy()
        ocp.constraints.ubu = self.ubu.copy()
        # ... intersected with the exact per-rotor thrust limits.
        ocp.constraints.C = np.zeros((4, NX))
        ocp.constraints.D = p.inverse_mixer()
        ocp.constraints.lg = np.zeros(4)
        ocp.constraints.ug = np.full(4, p.max_motor_thrust)

        # Soft body-rate limits keep the QP feasible after strong gusts.
        ocp.constraints.idxbx = np.array([10, 11, 12])
        ocp.constraints.lbx = -np.full(3, c.max_body_rate)
        ocp.constraints.ubx = np.full(3, c.max_body_rate)
        ocp.constraints.idxsbx = np.arange(3)
        ocp.cost.zl = ocp.cost.zu = np.full(3, c.slack_l1)
        ocp.cost.Zl = ocp.cost.Zu = np.full(3, c.slack_l2)

        ocp.constraints.x0 = hover_state()
        ocp.parameter_values = np.zeros(NP)

        opts = ocp.solver_options
        opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        opts.hessian_approx = "GAUSS_NEWTON"
        opts.integrator_type = "ERK"
        opts.sim_method_num_stages = 4
        opts.sim_method_num_steps = 1
        opts.nlp_solver_type = "SQP_RTI"
        opts.qp_solver_cond_N = c.horizon // 5
        opts.print_level = 0

        ocp.code_export_directory = os.path.join(c.build_dir, f"c_generated_code_{c.name}")
        return ocp

    # -------------------------------------------------------------- run time
    def reset(self, x0: np.ndarray | None = None) -> None:
        """Clear the observer and warm-start the solver at hover in ``x0``."""
        x0 = hover_state() if x0 is None else np.asarray(x0, dtype=float)
        u_hover = np.array([self.params.hover_thrust, 0.0, 0.0, 0.0])
        for k in range(self.cfg.horizon):
            self.solver.set(k, "x", x0)
            self.solver.set(k, "u", u_hover)
        self.solver.set(self.cfg.horizon, "x", x0)
        self.observer.reset()
        self.u_prev = u_hover
        self.x_prev: np.ndarray | None = None
        self.info = SolveInfo()

    def reference_window(self, traj: Trajectory, k: int, disturbance: np.ndarray | None = None):
        """References for stages ``k .. k+N`` (held at the final sample past the end)."""
        idx = np.minimum(np.arange(k, k + self.cfg.horizon + 1), len(traj) - 1)
        window = Trajectory(
            t=traj.t[0] + np.arange(len(idx)) * traj.dt,
            pos=traj.pos[idx],
            vel=traj.vel[idx],
            acc=traj.acc[idx],
            yaw=traj.yaw[idx],
        )
        if disturbance is not None:
            # Tilt so that thrust also cancels the estimated disturbance.
            window.acc = window.acc - disturbance
        return flat_to_state(window, self.params)

    def control(self, x: np.ndarray, traj: Trajectory, k: int) -> np.ndarray:
        """Compute the input to apply at sample ``k`` given the measured state ``x``."""
        c = self.cfg
        x = np.asarray(x, dtype=float).copy()
        if x[6] < 0.0:  # q and -q are the same attitude; keep the reference hemisphere
            x[6:10] = -x[6:10]

        d_hat = np.zeros(3)
        if c.use_observer:
            if self.x_prev is not None:
                self.observer.update(self.x_prev, self.u_prev, x)
            d_hat = self.observer.estimate.copy()

        x_ref, u_ref = self.reference_window(traj, k, d_hat if c.use_observer else None)

        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)
        lbu, ubu = self.lbu.copy(), self.ubu.copy()
        if c.max_thrust_rate is not None:
            lbu[0] = max(lbu[0], self.u_prev[0] - c.max_thrust_rate * c.dt)
            ubu[0] = min(ubu[0], self.u_prev[0] + c.max_thrust_rate * c.dt)
        if c.max_torque_rate is not None:
            lbu[1:] = np.maximum(lbu[1:], self.u_prev[1:] - c.max_torque_rate * c.dt)
            ubu[1:] = np.minimum(ubu[1:], self.u_prev[1:] + c.max_torque_rate * c.dt)
        self.solver.constraints_set(0, "lbu", lbu)
        self.solver.constraints_set(0, "ubu", ubu)

        for j in range(c.horizon):
            self.solver.set(j, "yref", np.concatenate([x_ref[j], u_ref[j]]))
            self.solver.set(j, "p", d_hat)
        self.solver.set(c.horizon, "yref", x_ref[c.horizon])
        self.solver.set(c.horizon, "p", d_hat)

        status = self.solver.solve()
        u = self.solver.get(0, "u")

        self.info = SolveInfo(
            status=int(status),
            solve_time=float(self.solver.get_stats("time_tot")),
            disturbance=d_hat,
            x_ref=x_ref[0],
            predicted_positions=np.array([self.solver.get(j, "x")[:3] for j in range(c.horizon + 1)]),
        )
        self.x_prev = x
        self.u_prev = u.copy()
        return u


def hover_state(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    x = np.zeros(NX)
    x[0:3] = position
    x[6] = 1.0
    return x
