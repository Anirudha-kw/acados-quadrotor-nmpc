"""Nonlinear MPC for a Crazyflie nano-quadrotor with acados and PyBullet."""

from .params import CRAZYFLIE, QuadrotorParams
from .trajectory import Trajectory, circle, figure_eight, flat_to_state, hold, quintic
from .wind import WindConfig, WindField

__version__ = "1.0.0"

__all__ = [
    "CRAZYFLIE",
    "QuadrotorParams",
    "Trajectory",
    "WindConfig",
    "WindField",
    "circle",
    "figure_eight",
    "flat_to_state",
    "hold",
    "quintic",
]
