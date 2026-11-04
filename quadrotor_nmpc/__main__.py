"""Command-line entry point: ``python -m quadrotor_nmpc --scenario figure_eight --gui``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    from .scenarios import SCENARIOS

    parser = argparse.ArgumentParser(
        prog="python -m quadrotor_nmpc",
        description="Fly a Crazyflie with acados NMPC in PyBullet.",
    )
    parser.add_argument("-s", "--scenario", choices=sorted(SCENARIOS), default="figure_eight")
    parser.add_argument("--gui", action="store_true", help="open the PyBullet GUI")
    parser.add_argument("--realtime", action="store_true", help="pace the GUI at wall-clock speed")
    parser.add_argument("--no-observer", action="store_true", help="disable the disturbance observer")
    parser.add_argument("--plot", metavar="PNG", help="save a summary figure")
    parser.add_argument("--video", metavar="MP4", help="render a dashboard video (headless is fine)")
    parser.add_argument("--save-log", metavar="NPZ", help="save the logged time series")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from .controller import NMPCConfig
    from .scenarios import run

    config = NMPCConfig(use_observer=not args.no_observer)
    scenario = SCENARIOS[args.scenario](config.dt)
    print(f"Scenario: {scenario.title} - {scenario.description} ({scenario.trajectory.duration:.1f} s)")

    recorder = None
    if args.video:
        from .video import VideoRecorder

        recorder = VideoRecorder(scenario, config)

    log = run(scenario, config, gui=args.gui, realtime=args.realtime, on_step=recorder, verbose=not args.quiet)

    metrics = log.metrics()
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))

    if args.save_log:
        log.save(args.save_log)
    if args.plot:
        from .plotting import plot_summary

        plot_summary(log, scenario.title).savefig(args.plot, dpi=150)
        print(f"Saved {args.plot}")
    if recorder is not None:
        recorder.write(log, args.video)
        print(f"Saved {args.video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
