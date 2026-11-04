"""Shared light/dark chart themes (CVD-validated categorical palette)."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
from matplotlib import font_manager

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        # blue, orange, aqua, yellow - fixed order
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

_PREFERRED_FONTS = ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
FONT_FAMILY = [next((f for f in _PREFERRED_FONTS if f in _installed), "DejaVu Sans")]


@contextmanager
def themed(name: str = "light"):
    """Matplotlib rc context for the given theme."""
    th = THEMES[name]
    rc = {
        "figure.facecolor": th["page"],
        "axes.facecolor": th["surface"],
        "savefig.facecolor": th["page"],
        "axes.edgecolor": th["axis"],
        "axes.labelcolor": th["ink2"],
        "axes.titlecolor": th["ink"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=th["series"]),
        "grid.color": th["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": th["muted"],
        "ytick.color": th["muted"],
        "xtick.labelcolor": th["ink2"],
        "ytick.labelcolor": th["ink2"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": th["ink2"],
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # Concrete family list (not "sans-serif") so fonts resolve at creation time, even if the
        # figure is saved after this context exits.
        "font.family": FONT_FAMILY,
        "text.color": th["ink"],
    }
    with mpl.rc_context(rc):
        yield th
