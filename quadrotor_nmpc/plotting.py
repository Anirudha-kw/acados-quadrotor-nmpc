"""Static result figures (PNG, light and dark variants)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .params import CRAZYFLIE  # noqa: E402
from .scenarios import SimLog  # noqa: E402
from .style import themed  # noqa: E402

AXES = ("x", "y", "z")


def _stat_title(ax, th, text: str) -> None:
    ax.set_title(text, loc="right", fontsize=9, fontweight="normal", color=th["ink2"])


def _style_3d(ax, th) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(th["surface"]))
        axis._axinfo["grid"].update(color=th["grid"], linewidth=0.8)
        axis.line.set_color(th["axis"])
        axis.label.set_color(th["ink2"])
    ax.tick_params(colors=th["ink2"], labelsize=8)
    ax.set_facecolor(th["page"])


def _equal_3d(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, half = (lo + hi) / 2, max((hi - lo).max() / 2, 0.3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.0, center[2] - half), center[2] + half)


def plot_summary(log: SimLog, title: str, theme: str = "light"):
    """Four-panel overview of one closed-loop run."""
    m = log.metrics()
    with themed(theme) as th:
        fig = plt.figure(figsize=(13, 7.2), layout="constrained")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

        ax3 = fig.add_subplot(gs[:, 0], projection="3d")
        _style_3d(ax3, th)
        ax3.plot(*log.pos_ref.T, color=th["ink2"], lw=1.2, ls=(0, (4, 3)), label="Reference")
        ax3.plot(*log.x[:, :3].T, color=th["series"][0], lw=2.0, label="Flown")
        ax3.scatter(*log.x[0, :3], color=th["series"][2], s=36, depthshade=False, label="Start")
        _equal_3d(ax3, np.vstack([log.pos_ref, log.x[:, :3]]))
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")
        ax3.set_zlabel("z [m]")
        ax3.view_init(elev=24, azim=-58)
        ax3.set_box_aspect(None, zoom=0.88)  # keep the z label inside its own column
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_major_locator(MaxNLocator(5))
        ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
        ax3.set_title("3D path", loc="left")

        ax = fig.add_subplot(gs[0, 1])
        for i, name in enumerate(AXES):
            ax.plot(log.t, log.pos_ref[:, i], color=th["series"][i], lw=1.1, ls=(0, (4, 3)), alpha=0.9)
            ax.plot(log.t, log.x[:, i], color=th["series"][i], label=name)
        handles = [Line2D([], [], color=th["series"][i], label=n) for i, n in enumerate(AXES)]
        handles.append(Line2D([], [], color=th["ink2"], ls=(0, (4, 3)), lw=1.1, label="reference"))
        ax.legend(
            handles=handles,
            ncol=4,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.2,
            handlelength=1.6,
            columnspacing=1.0,
        )
        ax.set_title("Position")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position [m]")
        ax.margins(x=0)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(log.t, log.error * 1e3, color=th["series"][0])
        ax.set_title("Tracking error")
        _stat_title(ax, th, f"RMSE {m['rmse_m'] * 1e3:.2f} mm · max {m['max_error_m'] * 1e3:.2f} mm")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("error [mm]")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)

        ax = fig.add_subplot(gs[1, 1])
        for i in range(4):
            ax.plot(log.t, log.rotor_thrust[:, i] * 1e3 / 9.81, color=th["series"][i], lw=1.3, label=f"rotor {i}")
        fmax_g = CRAZYFLIE.max_motor_thrust * 1e3 / 9.81
        ax.axhline(fmax_g, color=th["ink2"], lw=1.0, ls=(0, (4, 3)))
        ax.text(log.t[-1], fmax_g, "rotor limit ", ha="right", va="bottom", color=th["ink2"], fontsize=8.5)
        ax.set_ylim(0, fmax_g * 1.15)
        ax.set_title("Rotor thrust")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("thrust [gf]")
        ax.legend(ncol=4, loc="lower right", handlelength=1.4, columnspacing=0.9)
        ax.margins(x=0)

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(log.t, log.solve_time * 1e3, color=th["series"][0], lw=1.2)
        ax.set_title("acados solve time")
        _stat_title(ax, th, f"mean {m['solve_mean_ms']:.2f} ms · max {m['solve_max_ms']:.2f} ms")
        ax.set_xlabel(f"time [s]   (real-time budget: {log.dt * 1e3:.0f} ms per step)")
        ax.set_ylabel("solve time [ms]")
        ax.set_ylim(0, 1.25 * np.max(log.solve_time * 1e3))
        ax.margins(x=0)

        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="semibold", color=th["ink"])
    return fig


def plot_wind_comparison(with_obs: SimLog, without_obs: SimLog, theme: str = "light"):
    """Observer vs. no observer under the same wind realisation."""
    m_on, m_off = with_obs.metrics(), without_obs.metrics()
    with themed(theme) as th:
        fig, axs = plt.subplots(2, 2, figsize=(13, 6.6), layout="constrained")
        t = with_obs.t

        ax = axs[0, 0]
        for i, name in enumerate(AXES):
            ax.plot(t, with_obs.wind[:, i], color=th["series"][i], lw=1.3, label=f"wind {name}")
        ax.set_title("Wind velocity (mean + sine + gusts + turbulence)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [m/s]")
        ax.legend(ncol=3, loc="upper left")
        ax.margins(x=0)

        ax = axs[0, 1]
        for i, name in enumerate(AXES[:2]):
            ax.plot(t, with_obs.ext_accel[:, i], color=th["series"][i], lw=1.0, alpha=0.45)
            ax.plot(t, with_obs.disturbance[:, i], color=th["series"][i], lw=1.6, label=f"{name}")
        handles = [Line2D([], [], color=th["series"][i], label=f"estimate {n}") for i, n in enumerate(AXES[:2])]
        handles.append(Line2D([], [], color=th["series"][0], alpha=0.45, lw=1.0, label="true drag / m (faded)"))
        ax.legend(handles=handles, ncol=3, loc="upper left")
        ax.set_title("Disturbance observer")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("acceleration [m/s²]")
        ax.margins(x=0)

        ax = axs[1, 0]
        ax.plot(t, without_obs.error * 1e3, color=th["series"][1], label="NMPC")
        ax.plot(t, with_obs.error * 1e3, color=th["series"][0], label="NMPC + disturbance observer")
        ax.set_title("Position error")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("error [mm]")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.margins(x=0)

        ax = axs[1, 1]
        hold = t >= 5.0
        for log, color, label, _mm in (
            (without_obs, th["series"][1], "NMPC", m_off),
            (with_obs, th["series"][0], "NMPC + observer", m_on),
        ):
            e = (log.x[hold, :2] - log.pos_ref[hold, :2]) * 1e3
            ax.plot(
                e[:, 0],
                e[:, 1],
                color=color,
                lw=1.2,
                alpha=0.9,
                label=f"{label}: RMSE {np.sqrt(np.mean(np.sum(e**2, axis=1))):.1f} mm",
            )
        ax.scatter([0], [0], color=th["ink"], s=30, zorder=5, marker="+")
        lim = 1.15 * max(
            np.max(np.abs((lg.x[hold, :2] - lg.pos_ref[hold, :2]) * 1e3)) for lg in (with_obs, without_obs)
        )
        # Equal mm scales on both axes; x is widened to the panel's ~2.4:1 shape, y has legend head-room.
        ax.set_ylim(-lim, 1.6 * lim)
        ax.set_xlim(-3.1 * lim, 3.1 * lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Horizontal deviation while holding (t ≥ 5 s)")
        ax.set_xlabel("x error [mm]")
        ax.set_ylabel("y error [mm]")
        ax.legend(loc="upper left")

        fig.suptitle(
            f"Wind rejection - RMSE {m_off['rmse_m'] * 1e3:.1f} mm → {m_on['rmse_m'] * 1e3:.1f} mm with the observer",
            x=0.01,
            ha="left",
            fontsize=14,
            fontweight="semibold",
            color=th["ink"],
        )
    return fig


def plot_solver_timing(logs: Sequence[tuple[str, SimLog]], theme: str = "light"):
    """Distribution of acados solve time per control step for each scenario."""
    with themed(theme) as th:
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(logs) + 1.6), layout="constrained")
        data = [log.solve_time * 1e3 for _, log in logs]
        labels = [name for name, _ in logs]
        bp = ax.boxplot(
            data,
            vert=False,
            tick_labels=labels,
            widths=0.45,
            showfliers=True,
            patch_artist=True,
            medianprops=dict(color=th["ink"], lw=1.6),
            boxprops=dict(facecolor=th["series"][0], edgecolor=th["series"][0], alpha=0.85),
            whiskerprops=dict(color=th["series"][0], lw=1.2),
            capprops=dict(color=th["series"][0], lw=1.2),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=th["muted"], markeredgecolor="none"),
        )
        del bp
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("acados solve time per control step [ms]  (real-time budget: 20 ms)")
        ax.set_title("Solver timing (SQP-RTI, N = 25)")
    return fig


def save_both_themes(make_fig, path_stem: str, dpi: int = 150) -> list[str]:
    """Render ``make_fig(theme)`` in light and dark and save ``<stem>.png`` / ``<stem>_dark.png``."""
    out = []
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        fig = make_fig(theme)
        path = f"{path_stem}{suffix}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        out.append(path)
    return out
