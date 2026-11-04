import numpy as np

from quadrotor_nmpc.wind import WindConfig, WindField


def sample(field, n=2000, dt=0.01):
    return np.array([field.step(dt) for _ in range(n)])


def test_wind_is_reproducible_and_seeded():
    a = sample(WindField(WindConfig(seed=3)))
    b = sample(WindField(WindConfig(seed=3)))
    c = sample(WindField(WindConfig(seed=4)))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reset_restarts_the_realisation():
    field = WindField()
    first = sample(field, 300)
    field.reset()
    assert np.array_equal(first, sample(field, 300))


def test_queries_do_not_advance_state():
    field = WindField()
    field.step(0.01)
    v = field.velocity.copy()
    for _ in range(5):
        field.drag_force(np.zeros(3))
    assert np.array_equal(v, field.velocity)


def test_constant_wind_and_drag_direction():
    cfg = WindConfig(
        mean=(1.0, 0.0, 0.0), sine_amplitude=(0, 0, 0), gust_amplitude=(0, 0, 0), turbulence_sigma=(0, 0, 0)
    )
    field = WindField(cfg)
    assert np.allclose(sample(field, 50), [1.0, 0.0, 0.0])
    drag = field.drag_force(np.zeros(3))
    assert drag[0] > 0 and np.allclose(drag[1:], 0)
    assert np.allclose(field.drag_force([1.0, 0.0, 0.0]), 0.0)  # moving with the air
    k = 0.5 * cfg.air_density * cfg.drag_coefficient * cfg.reference_area
    assert np.isclose(drag[0], k)


def test_turbulence_statistics():
    cfg = WindConfig(
        mean=(0, 0, 0),
        sine_amplitude=(0, 0, 0),
        gust_amplitude=(0, 0, 0),
        turbulence_sigma=(0.3, 0.3, 0.3),
        turbulence_length=(1.0, 1.0, 1.0),
        airspeed=5.0,
    )
    v = sample(WindField(cfg), 40000, dt=0.01)
    assert np.allclose(v.std(axis=0), 0.3, rtol=0.15)
