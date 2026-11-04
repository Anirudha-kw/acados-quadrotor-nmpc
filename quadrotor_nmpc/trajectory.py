"""Smooth reference trajectories and differential-flatness feed-forward.

Every generator returns a :class:`Trajectory` sampled at a fixed ``dt`` whose
position is C2-continuous (continuous acceleration). Segments are chained
with :meth:`Trajectory.then`, which checks that the junction is continuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import CRAZYFLIE, QuadrotorParams
from .rotations import rotmat_to_quat


@dataclass
class Trajectory:
    """Time-sampled flat outputs: position, its derivatives, and yaw."""

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        n = len(self.t)
        self.pos = np.asarray(self.pos, dtype=float).reshape(n, 3)
        self.vel = np.asarray(self.vel, dtype=float).reshape(n, 3)
        self.acc = np.asarray(self.acc, dtype=float).reshape(n, 3)
        self.yaw = np.zeros(n) if self.yaw is None else np.asarray(self.yaw, dtype=float).reshape(n)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def then(self, other: Trajectory, tol: float = 1e-6) -> Trajectory:
        """Append ``other`` (whose first sample must equal this one's last)."""
        if not np.isclose(self.dt, other.dt):
            raise ValueError(f"sample times differ: {self.dt} vs {other.dt}")
        for name in ("pos", "vel", "acc"):
            gap = np.max(np.abs(getattr(self, name)[-1] - getattr(other, name)[0]))
            if gap > tol:
                raise ValueError(f"discontinuous {name} at junction (gap {gap:.3g})")
        return Trajectory(
            t=np.concatenate([self.t, self.t[-1] + other.t[1:] - other.t[0]]),
            pos=np.vstack([self.pos, other.pos[1:]]),
            vel=np.vstack([self.vel, other.vel[1:]]),
            acc=np.vstack([self.acc, other.acc[1:]]),
            yaw=np.concatenate([self.yaw, other.yaw[1:]]),
        )


def _time_grid(duration: float, dt: float) -> np.ndarray:
    n = int(round(duration / dt))
    return np.arange(n + 1) * dt


def quintic(p0, p1, duration: float, dt: float, v0=0.0, v1=0.0, a0=0.0, a1=0.0) -> Trajectory:
    """Minimum-jerk style quintic polynomial between two boundary states."""
    p0, p1 = np.broadcast_to(p0, 3).astype(float), np.broadcast_to(p1, 3).astype(float)
    v0, v1 = np.broadcast_to(v0, 3).astype(float), np.broadcast_to(v1, 3).astype(float)
    a0, a1 = np.broadcast_to(a0, 3).astype(float), np.broadcast_to(a1, 3).astype(float)
    T = float(duration)
    A = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
            [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
        ]
    )
    coeffs = np.linalg.solve(A, np.vstack([p0, v0, a0, p1, v1, a1]))  # (6, 3)
    t = _time_grid(T, dt)
    powers = np.vander(t, 6, increasing=True)  # t^0 .. t^5
    d1 = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    d2 = np.array([0, 0, 2, 6, 12, 20], dtype=float)
    pos = powers @ coeffs
    vel = np.hstack([np.zeros((len(t), 1)), powers[:, :5]]) * d1 @ coeffs
    acc = np.hstack([np.zeros((len(t), 2)), powers[:, :4]]) * d2 @ coeffs
    return Trajectory(t, pos, vel, acc)


def hold(p, duration: float, dt: float) -> Trajectory:
    """Stay at a fixed position."""
    t = _time_grid(duration, dt)
    pos = np.tile(np.broadcast_to(p, 3).astype(float), (len(t), 1))
    zeros = np.zeros_like(pos)
    return Trajectory(t, pos, zeros, zeros.copy())


def _smooth_phase(t: np.ndarray, omega: float, t_ramp: float, t_end: float):
    """Phase ``theta(t)`` whose rate ramps 0 -> omega -> 0 with C2 smoothstep blends.

    Returns ``(theta, theta_dot, theta_ddot)``. The total swept angle is
    ``omega * (t_end - t_ramp)``.
    """

    def s(x):  # quintic smoothstep
        return 10 * x**3 - 15 * x**4 + 6 * x**5

    def s_int(x):  # integral of s from 0 to x
        return 2.5 * x**4 - 3 * x**5 + x**6

    def s_dot(x):
        return 30 * x**2 - 60 * x**3 + 30 * x**4

    th = np.empty_like(t)
    thd = np.empty_like(t)
    thdd = np.empty_like(t)
    total = omega * (t_end - t_ramp)

    up = t < t_ramp
    down = t > t_end - t_ramp
    mid = ~(up | down)

    x = t[up] / t_ramp
    th[up] = omega * t_ramp * s_int(x)
    thd[up] = omega * s(x)
    thdd[up] = omega * s_dot(x) / t_ramp

    th[mid] = omega * t_ramp / 2 + omega * (t[mid] - t_ramp)
    thd[mid] = omega
    thdd[mid] = 0.0

    y = np.clip((t_end - t[down]) / t_ramp, 0.0, 1.0)
    th[down] = total - omega * t_ramp * s_int(y)
    thd[down] = omega * s(y)
    thdd[down] = -omega * s_dot(y) / t_ramp
    return th, thd, thdd


def _periodic(dt: float, period: float, n_loops: float, ramp_time: float):
    if ramp_time <= 0:
        raise ValueError("ramp_time must be positive")
    t_end = n_loops * period + ramp_time
    omega = 2 * np.pi * n_loops / (t_end - ramp_time)
    t = _time_grid(t_end, dt)
    return t, _smooth_phase(t, omega, ramp_time, t_end)


def circle(center, radius: float, period: float, n_loops: float, dt: float, ramp_time: float = 2.0) -> Trajectory:
    """Horizontal circle starting (and ending) at rest at ``center + [radius, 0, 0]``."""
    c = np.broadcast_to(center, 3).astype(float)
    t, (th, thd, thdd) = _periodic(dt, period, n_loops, ramp_time)
    cos, sin = np.cos(th), np.sin(th)
    zeros = np.zeros_like(t)
    pos = c + radius * np.stack([cos, sin, zeros], axis=1)
    vel = radius * thd[:, None] * np.stack([-sin, cos, zeros], axis=1)
    acc = radius * (
        thdd[:, None] * np.stack([-sin, cos, zeros], axis=1) - (thd**2)[:, None] * np.stack([cos, sin, zeros], axis=1)
    )
    return Trajectory(t, pos, vel, acc)


def figure_eight(
    center,
    amplitude=(1.5, 0.6, 0.25),
    period: float = 8.0,
    n_loops: float = 2,
    dt: float = 0.02,
    ramp_time: float = 2.0,
) -> Trajectory:
    """3D lemniscate ``(ax sin th, ay sin 2th, az sin th)`` starting at rest at ``center``."""
    c = np.broadcast_to(center, 3).astype(float)
    ax, ay, az = amplitude
    t, (th, thd, thdd) = _periodic(dt, period, n_loops, ramp_time)
    s1, c1, s2, c2 = np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)
    pos = c + np.stack([ax * s1, ay * s2, az * s1], axis=1)
    dpos = np.stack([ax * c1, 2 * ay * c2, az * c1], axis=1)  # d pos / d theta
    ddpos = np.stack([-ax * s1, -4 * ay * s2, -az * s1], axis=1)
    vel = dpos * thd[:, None]
    acc = ddpos * (thd**2)[:, None] + dpos * thdd[:, None]
    return Trajectory(t, pos, vel, acc)


def flat_to_state(traj: Trajectory, params: QuadrotorParams = CRAZYFLIE) -> tuple[np.ndarray, np.ndarray]:
    """Map flat outputs to full reference states and inputs (differential flatness).

    The thrust direction must align with ``acc + g e_z``; together with the
    reference yaw this fixes the attitude. Body rates are recovered from the
    finite-difference derivative of the attitude. Returns ``x_ref (N, 13)``
    and ``u_ref (N, 4)`` where ``u_ref = [T_ff, 0, 0, 0]``.
    """
    n = len(traj)
    f = traj.acc + np.array([0.0, 0.0, params.gravity])
    f_norm = np.linalg.norm(f, axis=1)
    z_b = f / f_norm[:, None]
    x_c = np.stack([np.cos(traj.yaw), np.sin(traj.yaw), np.zeros(n)], axis=1)
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b, axis=1)[:, None]
    x_b = np.cross(y_b, z_b)
    R = np.stack([x_b, y_b, z_b], axis=2)  # columns are body axes in world frame

    quat = np.array([rotmat_to_quat(Rk) for Rk in R])
    if n > 1:
        R_dot = np.gradient(R, traj.dt, axis=0)
        W = np.einsum("nji,njk->nik", R, R_dot)  # R^T R_dot = [omega]_x
        omega = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    else:
        omega = np.zeros((n, 3))

    x_ref = np.hstack([traj.pos, traj.vel, quat, omega])
    u_ref = np.zeros((n, 4))
    u_ref[:, 0] = params.mass * f_norm
    return x_ref, u_ref
