"""Regenerate every result figure, video and GIF in ``media/``.

Each scenario is flown twice: a clean headless run provides all metrics, plots and video telemetry,
and an identical recording run (asserted to match) provides the rendered 3D views.

Usage (headless machines need a virtual display for the OpenGL renderer)::

    xvfb-run -a -s "-screen 0 1920x1080x24" python scripts/make_media.py

Without a display the software TinyRenderer is used instead (slower, aliased).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_nmpc.controller import NMPCConfig, QuadrotorNMPC  # noqa: E402
from quadrotor_nmpc.plotting import (  # noqa: E402
    plot_solver_timing,
    plot_summary,
    plot_wind_comparison,
    save_both_themes,
)
from quadrotor_nmpc.scenarios import SCENARIOS, SimLog, run  # noqa: E402
from quadrotor_nmpc.video import VideoRecorder, mp4_to_gif  # noqa: E402

MEDIA = ROOT / "media"
RUNS = ROOT / "runs"
ORDER = ["figure_eight", "circle", "point_to_point", "wind"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--no-video", action="store_true", help="only re-run and re-plot")
    ap.add_argument("--replot", action="store_true", help="re-plot from runs/*.npz without simulating")
    ap.add_argument(
        "--reuse-logs", action="store_true", help="take metrics/telemetry from runs/*.npz instead of re-flying them"
    )
    args = ap.parse_args()
    if args.replot:
        replot()
        return

    MEDIA.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    gui = bool(os.environ.get("DISPLAY"))
    config = NMPCConfig(build_dir=str(ROOT / "build"))
    controller = QuadrotorNMPC(config=config)  # compile once, reuse for every scenario

    logs: dict[str, SimLog] = {}
    for name in args.scenarios:
        scenario = SCENARIOS[name](config.dt)
        print(f"\n=== {scenario.title}: {scenario.description}")

        # 1) Clean headless run: metrics, plots and the telemetry shown in the videos.
        saved = RUNS / f"{name}.npz"
        if args.reuse_logs and saved.exists():
            log = SimLog.load(str(saved))
            print(f"    reusing {saved.name}: {fmt(log.metrics())}")
        else:
            t0 = time.perf_counter()
            log = run(scenario, config, controller=controller)
            log.save(str(saved))
            print(f"    simulated in {time.perf_counter() - t0:.0f} s: {fmt(log.metrics())}")
        logs[name] = log
        save_both_themes(lambda th, lg=log, sc=scenario: plot_summary(lg, sc.title, th), str(MEDIA / f"{name}"))

        if args.no_video:
            continue
        # 2) Re-fly the identical (deterministic) flight with the renderer attached. The GUI's render
        #    thread competes with the solver for CPU, so solve times are taken from the clean run.
        recorder = VideoRecorder(scenario, config)
        video_log = run(scenario, config, gui=gui, on_step=recorder, controller=controller, verbose=False)
        drift = float(np.max(np.abs(video_log.x - log.x)))
        if drift > 1e-6:
            raise RuntimeError(f"recording run diverged from the clean run (max state difference {drift:.2e})")
        mp4 = str(MEDIA / f"{name}.mp4")
        recorder.write(log, mp4)
        mp4_to_gif(mp4, str(MEDIA / f"{name}.gif"), width=560, fps=10, speed=1.5, colors=96)
        if name == "figure_eight":
            mp4_to_gif(mp4, str(MEDIA / "hero.gif"), width=800, fps=12, speed=1.25, colors=128)
        print(f"    wrote {mp4} and GIF(s) (recording run matched to {drift:.1e})")

    if "wind" in logs:
        print("\n=== Wind rejection without the disturbance observer")
        # Same OCP; with the observer off the disturbance parameter simply stays at zero.
        saved = RUNS / "wind_no_observer.npz"
        if args.reuse_logs and saved.exists():
            nominal = SimLog.load(str(saved))
        else:
            controller.cfg.use_observer = False
            nominal = run(SCENARIOS["wind"](config.dt), controller.cfg, controller=controller)
            controller.cfg.use_observer = True
            nominal.save(str(saved))
        logs["wind_no_observer"] = nominal
        print(f"    {fmt(nominal.metrics())}")
        save_both_themes(lambda th: plot_wind_comparison(logs["wind"], nominal, th), str(MEDIA / "wind_comparison"))

    titles = {n: SCENARIOS[n](config.dt).title for n in ORDER}
    timing = [(titles[n], logs[n]) for n in ORDER if n in logs]
    if timing:
        save_both_themes(lambda th: plot_solver_timing(timing, th), str(MEDIA / "solver_timing"))

    results = {name: log.metrics() for name, log in logs.items()}
    results_path = MEDIA / "results.json"
    if results_path.exists():
        results = {**json.loads(results_path.read_text()), **results}
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print("\n" + markdown_table(results))


def replot() -> None:
    dt = NMPCConfig().dt
    logs = {p.stem: SimLog.load(str(p)) for p in sorted(RUNS.glob("*.npz"))}
    for name in ORDER:
        if name in logs:
            title = SCENARIOS[name](dt).title
            save_both_themes(lambda th, lg=logs[name], t=title: plot_summary(lg, t, th), str(MEDIA / name))
    if "wind" in logs and "wind_no_observer" in logs:
        save_both_themes(
            lambda th: plot_wind_comparison(logs["wind"], logs["wind_no_observer"], th), str(MEDIA / "wind_comparison")
        )
    timing = [(SCENARIOS[n](dt).title, logs[n]) for n in ORDER if n in logs]
    save_both_themes(lambda th: plot_solver_timing(timing, th), str(MEDIA / "solver_timing"))
    results = {name: log.metrics() for name, log in logs.items()}
    (MEDIA / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(markdown_table(results))


def fmt(m: dict) -> str:
    return (
        f"RMSE {m['rmse_m'] * 1e3:.1f} mm, max {m['max_error_m'] * 1e3:.1f} mm, "
        f"solve {m['solve_mean_ms']:.2f}/{m['solve_max_ms']:.2f} ms (mean/max), status!=0: {m['nonzero_status']}"
    )


def markdown_table(results: dict) -> str:
    rows = [
        "| Scenario | RMSE | Max error | Final error | Solve mean | Solve p99 | Solve max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['rmse_m'] * 1e3:.1f} mm | {m['max_error_m'] * 1e3:.1f} mm | "
            f"{m['final_error_m'] * 1e3:.1f} mm | {m['solve_mean_ms']:.2f} ms | {m['solve_p99_ms']:.2f} ms | "
            f"{m['solve_max_ms']:.2f} ms |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
