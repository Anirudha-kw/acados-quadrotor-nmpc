"""PyBullet plant: a Crazyflie driven by four rotor forces plus optional wind."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

from .model import NX
from .params import CRAZYFLIE, REST_HEIGHT, QuadrotorParams
from .rotations import quat_to_rotmat, xyzw_to_wxyz
from .wind import WindField

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_URDF = ASSET_DIR / "crazyflie.urdf"
GROUND_TEXTURE = ASSET_DIR / "ground.png"

__all__ = ["CrazyflieSim", "REST_HEIGHT", "DEFAULT_URDF"]


class CrazyflieSim:
    """Rigid-body simulation in PyBullet.

    The controller commands ``u = [T, tau_x, tau_y, tau_z]``. The inverse mixer
    maps it to rotor thrusts, which are clipped to the physical range and then
    applied as forces at the rotor hubs along the body z-axis at every physics
    sub-step (the yaw torque is applied as a pure body torque).
    """

    def __init__(
        self,
        params: QuadrotorParams = CRAZYFLIE,
        gui: bool = False,
        physics_hz: float = 500.0,
        wind: WindField | None = None,
        start_position=(0.0, 0.0, REST_HEIGHT),
        urdf_path: str | Path = DEFAULT_URDF,
    ):
        self.params = params
        self.gui = gui
        self.physics_dt = 1.0 / physics_hz
        self.wind = wind
        options = "--background_color_red=0.05 --background_color_green=0.06 --background_color_blue=0.08"
        self.client = pb.connect(pb.GUI if gui else pb.DIRECT, options=options if gui else "")
        if gui:
            for flag in (
                pb.COV_ENABLE_GUI,
                pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                pb.COV_ENABLE_RGB_BUFFER_PREVIEW,
            ):
                pb.configureDebugVisualizer(flag, 0, physicsClientId=self.client)
            pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0, 0, -params.gravity, physicsClientId=self.client)
        pb.setTimeStep(self.physics_dt, physicsClientId=self.client)
        pb.setRealTimeSimulation(0, physicsClientId=self.client)

        self.plane = pb.loadURDF("plane.urdf", physicsClientId=self.client)
        if GROUND_TEXTURE.exists():
            tex = pb.loadTexture(str(GROUND_TEXTURE), physicsClientId=self.client)
            pb.changeVisualShape(
                self.plane, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1], physicsClientId=self.client
            )
        self.body = pb.loadURDF(
            str(urdf_path),
            list(start_position),
            [0, 0, 0, 1],
            flags=pb.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )
        # No artificial damping: drag comes only from the wind model.
        pb.changeDynamics(self.body, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=self.client)

        self._inv_mixer = params.inverse_mixer()
        self._rotor_pos = params.motor_positions
        self._yaw_signs = np.array([s for _, _, s in params.motor_signs], dtype=float)
        self.time = 0.0
        self.rotor_thrusts = np.full(4, params.hover_thrust / 4)
        self.external_force = np.zeros(3)  # mean aerodynamic force over the last step [N]

    # ---------------------------------------------------------------- state
    def state(self) -> np.ndarray:
        """``[p, v, q (w, x, y, z), omega_body]``."""
        pos, orn = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        vel, ang_vel = pb.getBaseVelocity(self.body, physicsClientId=self.client)
        q = xyzw_to_wxyz(orn)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        omega_body = quat_to_rotmat(q).T @ np.asarray(ang_vel)
        x = np.empty(NX)
        x[0:3], x[3:6], x[6:10], x[10:13] = pos, vel, q, omega_body
        return x

    def reset(self, position=(0.0, 0.0, REST_HEIGHT)) -> np.ndarray:
        pb.resetBasePositionAndOrientation(self.body, list(position), [0, 0, 0, 1], physicsClientId=self.client)
        pb.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        if self.wind is not None:
            self.wind.reset()
        self.time = 0.0
        return self.state()

    # ------------------------------------------------------------- stepping
    def rotor_forces(self, u: np.ndarray) -> np.ndarray:
        return np.clip(self._inv_mixer @ np.asarray(u, dtype=float), 0.0, self.params.max_motor_thrust)

    def step(self, u: np.ndarray, duration: float) -> np.ndarray:
        """Hold ``u`` (zero-order hold) for ``duration`` seconds and return the new state."""
        f = self.rotor_forces(u)
        self.rotor_thrusts = f
        yaw_torque = self.params.yaw_coefficient * float(self._yaw_signs @ f)
        n_sub = max(1, int(round(duration / self.physics_dt)))
        self.external_force = np.zeros(3)
        for _ in range(n_sub):
            pos, orn = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
            R = quat_to_rotmat(xyzw_to_wxyz(orn))
            z_axis = R[:, 2]
            pos = np.asarray(pos)
            for i in range(4):
                pb.applyExternalForce(
                    self.body,
                    -1,
                    (f[i] * z_axis).tolist(),
                    (pos + R @ self._rotor_pos[i]).tolist(),
                    pb.WORLD_FRAME,
                    physicsClientId=self.client,
                )
            pb.applyExternalTorque(
                self.body, -1, (yaw_torque * z_axis).tolist(), pb.WORLD_FRAME, physicsClientId=self.client
            )
            if self.wind is not None:
                vel, _ = pb.getBaseVelocity(self.body, physicsClientId=self.client)
                drag = self.wind.drag_force(np.asarray(vel))
                pb.applyExternalForce(
                    self.body, -1, drag.tolist(), pos.tolist(), pb.WORLD_FRAME, physicsClientId=self.client
                )
                self.external_force += drag / n_sub
                self.wind.step(self.physics_dt)
            pb.stepSimulation(physicsClientId=self.client)
            self.time += self.physics_dt
        return self.state()

    def follow_camera(self, target, distance: float = 1.6, yaw: float = 45.0, pitch: float = -25.0) -> None:
        """Move the GUI camera (no-op in DIRECT mode)."""
        if self.gui:
            pb.resetDebugVisualizerCamera(distance, yaw, pitch, list(target), physicsClientId=self.client)

    def close(self) -> None:
        if pb.isConnected(self.client):
            pb.disconnect(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
