"""Colours and matplotlib defaults shared by the figures (light and dark themes).

The categorical slots (blue, orange) and the blue sequential ramp are steps of one
documented, pre-validated palette; nothing here is hand-picked. In dark mode the
sequential ramp is flipped so that "zero" recedes into the dark surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str  # primary text
    ink_secondary: str
    muted: str
    grid: str  # hairline gridlines
    axis: str  # baselines and axes
    blue: str  # categorical slot 1
    orange: str  # categorical slot 2
    blue_low: str  # de-emphasised shade of blue ("before")
    blue_high: str  # emphasised shade of blue ("after")
    ramp: Tuple[str, ...]  # sequential ramp, from "zero" to "maximum"

    def cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(f"blue_{self.name}", list(self.ramp))


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    blue="#2a78d6",
    orange="#eb6834",
    blue_low="#86b6ef",
    blue_high="#1c5cab",
    ramp=("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    blue="#3987e5",
    orange="#d95926",
    blue_low="#184f95",
    blue_high="#86b6ef",
    ramp=("#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"),
)

THEMES = (LIGHT, DARK)


def apply(theme: Theme) -> None:
    """Set matplotlib's rcParams for ``theme``: recessive hairline chrome, thin marks, no top/right spines."""
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink_secondary,
            "axes.edgecolor": theme.axis,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": theme.ink_secondary,
            "ytick.color": theme.ink_secondary,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "savefig.dpi": 200,
        }
    )
