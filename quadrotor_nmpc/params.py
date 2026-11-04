"""Physical parameters of the Bitcraze Crazyflie 2.x nano-quadrotor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadrotorParams:
    """Rigid-body and actuator parameters.

    Defaults correspond to a Crazyflie 2.x in the "X" configuration
    (Förster, 2015; Landry, 2015). Rotor numbering and body axes follow the
    convention used throughout this package (top view)::

                  +x (front)
             m0         m1
               \\       /
                 [body]          +y points left, +z points up
               /       \\
             m2         m3

    Motor ``i`` sits at body position ``(sx_i * d, sy_i * d, 0)`` with
    ``d = arm_length / sqrt(2)`` and produces a yaw reaction torque
    ``sz_i * (km / kf) * f_i``; ``(sx_i, sy_i, sz_i)`` is ``motor_signs[i]``.
    """

    mass: float = 0.027  # [kg]
    gravity: float = 9.81  # [m/s^2]
    inertia: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)  # [kg m^2]
    arm_length: float = 0.0397  # hub-to-motor distance [m]
    kf: float = 3.16e-10  # thrust coefficient [N / rpm^2]
    km: float = 7.94e-12  # drag-torque coefficient [N m / rpm^2]
    thrust_to_weight: float = 2.25  # maximum total thrust / weight [-]

    # Body-frame motor positions (sign of x, sign of y) and yaw-torque sign.
    motor_signs: tuple[tuple[int, int, int], ...] = (
        (+1, +1, +1),
        (+1, -1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
    )

    @property
    def hover_thrust(self) -> float:
        """Total thrust needed to hover [N]."""
        return self.mass * self.gravity

    @property
    def max_motor_thrust(self) -> float:
        """Maximum thrust a single rotor can produce [N]."""
        return self.thrust_to_weight * self.hover_thrust / 4.0

    @property
    def max_total_thrust(self) -> float:
        return 4.0 * self.max_motor_thrust

    @property
    def motor_offset(self) -> float:
        """Distance of each rotor from the body x/y axes [m]."""
        return self.arm_length / math.sqrt(2.0)

    @property
    def yaw_coefficient(self) -> float:
        """Ratio between rotor drag torque and rotor thrust [m]."""
        return self.km / self.kf

    @property
    def max_roll_pitch_torque(self) -> float:
        """Largest achievable roll/pitch torque (two rotors at full thrust) [N m]."""
        return 2.0 * self.motor_offset * self.max_motor_thrust

    @property
    def max_yaw_torque(self) -> float:
        return 2.0 * self.yaw_coefficient * self.max_motor_thrust

    @property
    def motor_positions(self) -> np.ndarray:
        """Body-frame rotor positions, shape ``(4, 3)``."""
        d = self.motor_offset
        return np.array([[sx * d, sy * d, 0.0] for sx, sy, _ in self.motor_signs])

    def mixer(self) -> np.ndarray:
        """Allocation matrix ``M`` with ``[T, tau_x, tau_y, tau_z] = M @ f``.

        ``f`` holds the four rotor thrusts. Roll torque is ``sum(y_i f_i)``,
        pitch torque ``-sum(x_i f_i)`` and yaw torque ``c * sum(s_i f_i)``.
        """
        d = self.motor_offset
        c = self.yaw_coefficient
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [sy * d for _, sy, _ in self.motor_signs],
                [-sx * d for sx, _, _ in self.motor_signs],
                [sz * c for _, _, sz in self.motor_signs],
            ]
        )

    def inverse_mixer(self) -> np.ndarray:
        """``M^-1``: maps ``[T, tau]`` to individual rotor thrusts."""
        return np.linalg.inv(self.mixer())


CRAZYFLIE = QuadrotorParams()

# Height of the body origin when the landing feet touch the ground (see assets/crazyflie.urdf) [m].
REST_HEIGHT = 0.014
