"""Render closed-loop runs to MP4/GIF: the native PyBullet view with clean path and telemetry overlays."""

from __future__ import annotations

import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from .controller import NMPCConfig, QuadrotorNMPC  # noqa: E402
from .scenarios import Scenario, SimLog  # noqa: E402
from .simulator import CrazyflieSim  # noqa: E402
from .style import FONT_FAMILY, THEMES  # noqa: E402


class SceneCamera:
    """Offscreen PyBullet camera that can also project world points to pixels."""

    def __init__(self, client: int, width: int, height: int, fov: float = 40.0, near: float = 0.02, far: float = 40.0):
        self.client, self.width, self.height = client, width, height
        self.proj = pb.computeProjectionMatrixFOV(fov, width / height, near, far)
        self.P = np.array(self.proj).reshape(4, 4).T
        gui = pb.getConnectionInfo(client)["connectionMethod"] == pb.GUI
        self.renderer = pb.ER_BULLET_HARDWARE_OPENGL if gui else pb.ER_TINY_RENDERER
        self.look(target=(0, 0, 0.5), distance=3.0, yaw=30.0, pitch=-25.0)

    def look(self, target, distance: float, yaw: float, pitch: float) -> None:
        self.view = pb.computeViewMatrixFromYawPitchRoll(list(target), distance, yaw, pitch, 0.0, 2)
        self.V = np.array(self.view).reshape(4, 4).T

    def render(self) -> np.ndarray:
        _, _, rgba, depth, _ = pb.getCameraImage(
            self.width,
            self.height,
            self.view,
            self.proj,
            renderer=self.renderer,
            shadow=1,
            lightDirection=[0.4, -0.6, 1.0],
            flags=pb.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client,
        )
        img = np.reshape(np.asarray(rgba, dtype=np.uint8), (self.height, self.width, 4))[:, :, :3].copy()
        if self.renderer == pb.ER_TINY_RENDERER:
            # TinyRenderer has a white sky; paint it to match the OpenGL background.
            sky = np.reshape(np.asarray(depth), (self.height, self.width)) >= 0.99999
            img[sky] = (13, 15, 20)
        return img

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel coordinates ``(N, 2)``; points behind the camera become NaN."""
        pts = np.atleast_2d(points)
        clip = (self.P @ self.V @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        w = clip[:, 3:4]
        ndc = clip[:, :3] / np.where(np.abs(w) < 1e-9, 1e-9, w)
        px = np.empty((len(pts), 2))
        px[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * self.width
        px[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * self.height
        px[w[:, 0] <= 0] = np.nan
        return px


def _orbit_for(traj_pos: np.ndarray, fov: float = 40.0) -> tuple[np.ndarray, float]:
    lo, hi = traj_pos.min(axis=0), traj_pos.max(axis=0)
    center = (lo + hi) / 2
    center[2] = max(lo[2] + 0.65 * (hi[2] - lo[2]), 0.45)  # bias toward flight altitude
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    distance = float(np.clip(1.45 * span * np.tan(np.radians(20.0)) / np.tan(np.radians(fov / 2)), 2.2, 6.0))
    return center, distance


class Dashboard:
    """Matplotlib compositor for the native PyBullet render plus clean overlays.

    ``layout="native"`` shows the PyBullet view full-frame with minimal overlays;
    ``layout="dashboard"`` adds live telemetry panels on the right.
    """

    def __init__(
        self,
        scenario: Scenario,
        config: NMPCConfig,
        width: int = 1280,
        height: int = 720,
        theme: str = "light",
        layout: str = "native",
    ):
        if layout not in ("native", "dashboard"):
            raise ValueError(f"unknown layout {layout!r}")
        self.scenario, self.cfg, self.layout = scenario, config, layout
        self.W, self.H = width, height
        self.scene_w = width if layout == "native" else int(round(width * 0.66))
        self.inset_w, self.inset_h = 288, 216
        th = self.th = THEMES[theme]
        traj = scenario.trajectory
        self.t_end = traj.t[-1]

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = FONT_FAMILY
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=th["page"])

        # ---------------- scene
        self.ax_scene = self.fig.add_axes((0, 0, self.scene_w / width, 1))
        self.ax_scene.set_axis_off()
        self.im_scene = self.ax_scene.imshow(
            np.zeros((height, self.scene_w, 3), np.uint8), extent=(0, self.scene_w, height, 0), interpolation="bilinear"
        )
        self.ax_scene.set_xlim(0, self.scene_w)
        self.ax_scene.set_ylim(height, 0)
        s = th["series"]
        (self.ln_ref,) = self.ax_scene.plot([], [], color=th["ink2"], lw=1.4, ls=(0, (4, 3)), alpha=0.9)
        (self.ln_trail,) = self.ax_scene.plot([], [], color=s[0], lw=2.4)
        (self.ln_pred,) = self.ax_scene.plot([], [], color=s[1], lw=2.6)
        (self.pt_pred,) = self.ax_scene.plot([], [], "o", color=s[1], ms=5, mec=th["surface"], mew=1.5)
        (self.pt_drone,) = self.ax_scene.plot([], [], "o", ms=22, mfc="none", mec=th["ink"], mew=1.4, alpha=0.75)
        self.wind_arrow = None

        # Title block and legend sit on translucent cards so they stay legible over the checkerboard.
        x_txt = 34
        title_block = [
            self.ax_scene.text(
                x_txt, 34, scenario.title, color=th["ink"], fontsize=21, fontweight="semibold", va="top", zorder=4
            ),
            self.ax_scene.text(x_txt, 72, scenario.description, color=th["ink2"], fontsize=11.5, va="top", zorder=4),
            self.ax_scene.text(
                x_txt,
                94,
                f"acados SQP-RTI NMPC · {1 / config.dt:.0f} Hz · N = {config.horizon}",
                color=th["ink2"],
                fontsize=10.5,
                va="top",
                zorder=4,
            ),
        ]
        legend = []
        legend_y = 40
        x0 = self.scene_w - 190
        for label, color, ls in (
            ("flown", s[0], "-"),
            ("reference", th["ink2"], (0, (4, 3))),
            ("MPC prediction", s[1], "-"),
        ):
            legend += self.ax_scene.plot([x0, x0 + 26], [legend_y, legend_y], color=color, lw=2.4, ls=ls, zorder=4)
            legend.append(
                self.ax_scene.text(x0 + 36, legend_y, label, color=th["ink"], fontsize=10.5, va="center", zorder=4)
            )
            legend_y += 22
        self.fig.canvas.draw()
        self._card(title_block)
        self._card(legend)
        self.hud = self.ax_scene.text(
            x_txt,
            height - 30,
            "",
            color=th["ink"],
            fontsize=11.5,
            va="bottom",
            family="DejaVu Sans Mono",
            linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.75,rounding_size=0.8", fc=th["surface"], ec=th["axis"], alpha=0.88),
            zorder=4,
        )

        # ---------------- chase-cam inset
        m = 18
        self.ax_inset = self.fig.add_axes(
            (
                (self.scene_w - m - self.inset_w) / width,
                m / height,
                self.inset_w / width,
                self.inset_h / height,
            )
        )
        self.im_inset = self.ax_inset.imshow(np.zeros((self.inset_h, self.inset_w, 3), np.uint8))
        self.ax_inset.set_xticks([])
        self.ax_inset.set_yticks([])
        for sp in self.ax_inset.spines.values():
            sp.set_color(th["ink2"])
            sp.set_linewidth(1.2)
        self.ax_inset.text(
            10,
            12,
            "chase camera",
            color=th["ink"],
            fontsize=9.5,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc=th["surface"], ec="none", alpha=0.8),
        )

        self.panel_axes = []
        self.top_trail = None
        if layout == "dashboard":
            self._build_panels(scenario, width, height)

    def _build_panels(self, scenario: Scenario, width: int, height: int) -> None:
        th, s, traj = self.th, self.th["series"], scenario.trajectory
        left = (self.scene_w + 58) / width
        pw = (width - self.scene_w - 58 - 22) / width
        heights = [0.33, 0.22, 0.22]
        gaps = 0.075
        tops = [0.95]
        for h in heights[:-1]:
            tops.append(tops[-1] - h - gaps)
        for top, h in zip(tops, heights, strict=True):
            ax = self.fig.add_axes((left, top - h, pw, h), facecolor=th["surface"])
            ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=9, length=0)
            ax.grid(True, color=th["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for name, sp in ax.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_color(th["axis"])
            self.panel_axes.append(ax)

        ax_top, ax_err, ax_solve = self.panel_axes
        self.is_hover = np.ptp(traj.pos[:, 0]) < 0.2 and np.ptp(traj.pos[:, 1]) < 0.2
        if self.is_hover and scenario.wind is not None:
            ax_top.set_title("Wind velocity [m/s]", loc="left", color=th["ink"], fontsize=11, fontweight="semibold")
            self.wind_lines = [ax_top.plot([], [], color=s[i], lw=1.6, label=n)[0] for i, n in enumerate("xyz")]
            ax_top.set_xlim(0, self.t_end)
            ax_top.set_ylim(-2.0, 3.0)
            ax_top.legend(loc="upper left", ncol=3, frameon=False, labelcolor=th["ink2"], fontsize=9)
            self.top_trail = None
        else:
            ax_top.set_title("Top view [m]", loc="left", color=th["ink"], fontsize=11, fontweight="semibold")
            ax_top.plot(traj.pos[:, 0], traj.pos[:, 1], color=th["ink2"], lw=1.1, ls=(0, (5, 4)), alpha=0.8)
            (self.top_trail,) = ax_top.plot([], [], color=s[0], lw=2.0)
            (self.top_dot,) = ax_top.plot([], [], "o", color=s[0], ms=7, mec=th["surface"], mew=2)
            # Equal metric scaling without letting matplotlib fight fixed limits.
            lo, hi = traj.pos[:, :2].min(axis=0) - 0.25, traj.pos[:, :2].max(axis=0) + 0.25
            box = ax_top.get_position()
            w_px, h_px = box.width * width, box.height * height
            scale = max((hi[0] - lo[0]) / w_px, (hi[1] - lo[1]) / h_px)
            cx, cy = (lo + hi) / 2
            ax_top.set_xlim(cx - scale * w_px / 2, cx + scale * w_px / 2)
            ax_top.set_ylim(cy - scale * h_px / 2, cy + scale * h_px / 2)

        ax_err.set_title("Position error [mm]", loc="left", color=th["ink"], fontsize=11, fontweight="semibold")
        (self.err_line,) = ax_err.plot([], [], color=s[0], lw=1.6)
        ax_err.set_xlim(0, self.t_end)

        ax_solve.set_title("acados solve time [ms]", loc="left", color=th["ink"], fontsize=11, fontweight="semibold")
        (self.solve_line,) = ax_solve.plot([], [], color=s[0], lw=1.3)
        ax_solve.set_xlim(0, self.t_end)
        ax_solve.set_xlabel("time [s]", color=th["ink2"], fontsize=9.5)

    def _card(self, artists, pad: float = 12.0) -> None:
        """Rounded translucent panel behind ``artists`` (scene pixel coordinates)."""
        renderer = self.fig.canvas.get_renderer()
        bb = Bbox.union([a.get_window_extent(renderer) for a in artists])
        (xa, ya), (xb, yb) = self.ax_scene.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        left, right = min(xa, xb) - pad, max(xa, xb) + pad
        top, bottom = min(ya, yb) - pad, max(ya, yb) + pad
        th = self.th
        self.ax_scene.add_patch(
            FancyBboxPatch(
                (left, top),
                right - left,
                bottom - top,
                boxstyle="round,pad=0,rounding_size=9",
                fc=th["surface"],
                ec=th["axis"],
                lw=1.0,
                alpha=0.88,
                zorder=3,
            )
        )

    def compose(
        self,
        k: int,
        log: SimLog,
        scene: np.ndarray,
        inset: np.ndarray,
        cam: SceneCamera,
        predicted: np.ndarray,
        err_ylim: float,
        solve_ylim: float,
    ) -> np.ndarray:
        th = self.th
        traj = self.scenario.trajectory
        self.im_scene.set_data(scene)
        self.im_inset.set_data(inset)

        ref_px = cam.project(traj.pos)
        self.ln_ref.set_data(ref_px[:, 0], ref_px[:, 1])
        trail_px = cam.project(log.x[: k + 1, :3])
        self.ln_trail.set_data(trail_px[:, 0], trail_px[:, 1])
        pred_px = cam.project(predicted)
        self.ln_pred.set_data(pred_px[:, 0], pred_px[:, 1])
        self.pt_pred.set_data(pred_px[-1:, 0], pred_px[-1:, 1])
        self.pt_drone.set_data(trail_px[-1:, 0], trail_px[-1:, 1])

        if self.wind_arrow is not None:
            self.wind_arrow.remove()
            self.wind_arrow = None
        wind = log.wind[k]
        if np.linalg.norm(wind) > 1e-3:
            p0 = log.x[k, :3]
            a, b = cam.project(np.vstack([p0, p0 + 0.35 * wind]))
            if np.all(np.isfinite([a, b])):
                self.wind_arrow = self.ax_scene.annotate(
                    "",
                    xy=b,
                    xytext=a,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.9", color=th["series"][2], lw=2.4),
                )

        t = log.t[: k + 1]
        err = log.error[: k + 1] * 1e3
        speed = np.linalg.norm(log.x[k, 3:6])
        lines = [
            f"t      {log.t[k]:6.2f} s",
            f"error  {err[-1]:6.1f} mm",
            f"speed  {speed:6.2f} m/s",
            f"solve  {log.solve_time[k] * 1e3:6.2f} ms",
        ]
        if self.scenario.wind is not None:
            lines.append(f"wind   {np.linalg.norm(wind):6.2f} m/s")
        self.hud.set_text("\n".join(lines))

        if self.panel_axes:
            if self.top_trail is not None:
                self.top_trail.set_data(log.x[: k + 1, 0], log.x[: k + 1, 1])
                self.top_dot.set_data([log.x[k, 0]], [log.x[k, 1]])
            else:
                for i, ln in enumerate(self.wind_lines):
                    ln.set_data(t, log.wind[: k + 1, i])
            self.err_line.set_data(t, err)
            self.panel_axes[1].set_ylim(0, err_ylim)
            self.solve_line.set_data(t, log.solve_time[: k + 1] * 1e3)
            self.panel_axes[2].set_ylim(0, solve_ylim)

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self) -> None:
        plt.close(self.fig)


class VideoRecorder:
    """``on_step`` callback for :func:`quadrotor_nmpc.scenarios.run` that renders frames.

    Frames are buffered as raw scene renders + state and composed after the
    run, so the optional telemetry panels can use the final error/solve-time ranges.
    """

    def __init__(
        self,
        scenario: Scenario,
        config: NMPCConfig,
        fps: int = 25,
        width: int = 1280,
        height: int = 720,
        yaw: float = 35.0,
        yaw_sweep: float = 0.0,
    ):
        self.scenario, self.cfg, self.fps = scenario, config, fps
        self.W, self.H = width, height
        self.every = max(1, int(round(1.0 / (fps * config.dt))))
        self.dashboard = Dashboard(scenario, config, width, height)
        self.center, self.distance = _orbit_for(scenario.trajectory.pos)
        self.yaw0, self.yaw_sweep = yaw, yaw_sweep  # a fixed camera keeps GIFs small
        self.frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.cam = self.chase = None

    def __call__(self, k: int, sim: CrazyflieSim, ctrl: QuadrotorNMPC, log: SimLog) -> None:
        if k % self.every:
            return
        if self.cam is None:
            self.cam = SceneCamera(sim.client, self.dashboard.scene_w, self.H, fov=40.0)
            self.chase = SceneCamera(sim.client, self.dashboard.inset_w, self.dashboard.inset_h, fov=50.0)
        frac = log.t[k] / max(log.t[-1], 1e-9)
        yaw = self.yaw0 + self.yaw_sweep * frac
        self.cam.look(self.center, self.distance, yaw, -24.0)
        pos = log.x[k, :3]
        self.chase.look(pos, 0.30, yaw - 20.0, -34.0)
        self.frames.append(
            (
                k,
                self.cam.render(),
                self.chase.render(),
                ctrl.info.predicted_positions.copy(),
                self.cam.V.copy(),
                np.asarray(self.cam.view),
            )
        )

    def write(self, log: SimLog, path: str) -> None:
        import imageio.v2 as imageio

        err_ylim = max(5.0, 1.15 * np.percentile(log.error * 1e3, 99.5))
        solve_ylim = max(1.0, 1.2 * np.max(log.solve_time * 1e3))
        writer = imageio.get_writer(
            path,
            fps=self.fps,
            codec="libx264",
            quality=None,
            macro_block_size=8,
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "22", "-preset", "slow", "-movflags", "+faststart"],
        )
        try:
            for k, scene, inset, predicted, V, view in self.frames:
                self.cam.V, self.cam.view = V, view
                frame = self.dashboard.compose(k, log, scene, inset, self.cam, predicted, err_ylim, solve_ylim)
                writer.append_data(frame)
        finally:
            writer.close()
            self.dashboard.close()


def mp4_to_gif(src: str, dst: str, width: int = 720, fps: int = 12, speed: float = 1.0, colors: int = 128) -> None:
    """High-quality GIF via ffmpeg's two-pass palette workflow."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-loop", "0", dst], check=True)
