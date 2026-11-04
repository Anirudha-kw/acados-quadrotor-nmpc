"""Quaternion-based rigid-body quadrotor model for acados.

State   ``x = [p (3), v (3), q (4), omega (3)]`` - world position and velocity,
        scalar-first body-to-world quaternion, body angular rates.
Input   ``u = [T, tau_x, tau_y, tau_z]`` - collective thrust and body torques.
Param.  ``d = [d_x, d_y, d_z]`` - external acceleration in the world frame
        (supplied online by the disturbance observer; zero for the nominal model).

Dynamics::

    p_dot     = v
    v_dot     = R(q) e_z T / m - g e_z + d
    q_dot     = 1/2 q (x) [0, omega]
    omega_dot = J^-1 (tau - omega x J omega)
"""

from __future__ import annotations

import casadi as ca

from .params import CRAZYFLIE, QuadrotorParams

NX = 13
NU = 4
NP = 3

STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "p", "q", "r")
INPUT_NAMES = ("T", "tau_x", "tau_y", "tau_z")


def quadrotor_dynamics(params: QuadrotorParams = CRAZYFLIE) -> dict:
    """Build the continuous-time dynamics symbolically with CasADi.

    Returns a dict with the symbols ``x``, ``u``, ``d`` and the explicit
    right-hand side ``f_expl`` as well as a callable ``f(x, u, d)``.
    """
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    d = ca.SX.sym("d", NP)

    v = x[3:6]
    qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    p, q, r = x[10], x[11], x[12]
    thrust = u[0]
    tau = u[1:4]

    m, g = params.mass, params.gravity
    J1, J2, J3 = params.inertia

    # Third column of R(q): direction of the thrust vector in the world frame.
    z_body = ca.vertcat(
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy),
    )
    acc = z_body * thrust / m - ca.vertcat(0, 0, g) + d

    q_dot = 0.5 * ca.vertcat(
        -p * qx - q * qy - r * qz,
        p * qw + r * qy - q * qz,
        q * qw - r * qx + p * qz,
        r * qw + q * qx - p * qy,
    )

    omega_dot = ca.vertcat(
        (tau[0] - (J3 - J2) * q * r) / J1,
        (tau[1] - (J1 - J3) * p * r) / J2,
        (tau[2] - (J2 - J1) * p * q) / J3,
    )

    f_expl = ca.vertcat(v, acc, q_dot, omega_dot)
    return {
        "x": x,
        "u": u,
        "d": d,
        "f_expl": f_expl,
        "f": ca.Function("f", [x, u, d], [f_expl], ["x", "u", "d"], ["x_dot"]),
    }


def export_acados_model(params: QuadrotorParams = CRAZYFLIE, name: str = "crazyflie"):
    """Wrap :func:`quadrotor_dynamics` in an :class:`acados_template.AcadosModel`."""
    from acados_template import AcadosModel

    dyn = quadrotor_dynamics(params)
    xdot = ca.SX.sym("xdot", NX)

    model = AcadosModel()
    model.name = name
    model.x = dyn["x"]
    model.u = dyn["u"]
    model.p = dyn["d"]
    model.xdot = xdot
    model.f_expl_expr = dyn["f_expl"]
    model.f_impl_expr = xdot - dyn["f_expl"]
    model.x_labels = list(STATE_NAMES)
    model.u_labels = list(INPUT_NAMES)
    return model
