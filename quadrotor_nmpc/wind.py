"""Stochastic wind field and quadratic aerodynamic drag.

The wind velocity is the sum of four components:

* a constant mean wind,
* a slow sinusoidal variation,
* discrete "1 - cos" gusts at random times (MIL-F-8785C gust shape),
* Dryden-like turbulence modelled as a first-order Gauss-Markov process
  per axis with time constant ``L / V`` and standard deviation ``sigma``.

The field is advanced exactly once per simulation step via :meth:`step`;
queries in between (e.g. for logging) do not change its state. All
randomness comes from a single seeded generator, so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindConfig:
    mean: tuple[float, float, float] = (0.8, 0.0, 0.0)  # [m/s]
    sine_amplitude: tuple[float, float, float] = (0.4, 0.2, 0.0)  # [m/s]
    sine_frequency: float = 0.2  # [Hz]
    gust_amplitude: tuple[float, float, float] = (1.5, 1.0, 0.8)  # peak [m/s]
    gust_duration: float = 1.0  # [s]
    gust_interval: tuple[float, float] = (2.0, 4.0)  # pause between gusts [s]
    turbulence_sigma: tuple[float, float, float] = (0.2, 0.2, 0.1)  # [m/s]
    turbulence_length: tuple[float, float, float] = (25.0, 25.0, 15.0)  # [m]
    airspeed: float = 2.0  # reference airspeed for the turbulence time constant [m/s]
    air_density: float = 1.225  # [kg/m^3]
    drag_coefficient: float = 1.2  # [-]
    reference_area: float = 0.010  # [m^2]
    seed: int = 7


class WindField:
    """Time-varying, spatially uniform wind with quadratic drag on the vehicle."""

    def __init__(self, config: WindConfig | None = None):
        self.cfg = config or WindConfig()
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self._turb = np.zeros(3)
        self._gust_start = self.rng.uniform(*cfg.gust_interval)
        self._gust_peak = self._sample_gust()
        self.velocity = self._evaluate()

    def _sample_gust(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, 3) * np.asarray(self.cfg.gust_amplitude)

    def _gust(self, t: float) -> np.ndarray:
        cfg = self.cfg
        while t > self._gust_start + cfg.gust_duration:
            self._gust_start += cfg.gust_duration + self.rng.uniform(*cfg.gust_interval)
            self._gust_peak = self._sample_gust()
        tau = t - self._gust_start
        if tau < 0.0:
            return np.zeros(3)
        return 0.5 * self._gust_peak * (1.0 - np.cos(2.0 * np.pi * tau / cfg.gust_duration))

    def _evaluate(self) -> np.ndarray:
        cfg = self.cfg
        sine = np.asarray(cfg.sine_amplitude) * np.sin(2.0 * np.pi * cfg.sine_frequency * self.t)
        return np.asarray(cfg.mean, dtype=float) + sine + self._gust(self.t) + self._turb

    def step(self, dt: float) -> np.ndarray:
        """Advance the field by ``dt`` seconds and return the new wind velocity."""
        cfg = self.cfg
        self.t += dt
        tau = np.asarray(cfg.turbulence_length) / max(cfg.airspeed, 1e-3)
        phi = np.exp(-dt / tau)
        noise = self.rng.standard_normal(3)
        self._turb = phi * self._turb + np.asarray(cfg.turbulence_sigma) * np.sqrt(1.0 - phi**2) * noise
        self.velocity = self._evaluate()
        return self.velocity

    def drag_force(self, vehicle_velocity: np.ndarray) -> np.ndarray:
        """Aerodynamic force on the vehicle in the world frame [N]."""
        cfg = self.cfg
        v_rel = self.velocity - np.asarray(vehicle_velocity, dtype=float)
        k = 0.5 * cfg.air_density * cfg.drag_coefficient * cfg.reference_area
        return k * np.linalg.norm(v_rel) * v_rel
