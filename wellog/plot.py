"""Multi-track log display.

Track layout follows the usual petrophysical convention:

  1. GR (+ caliper)            5. Volumetrics (Vsh / matrix / PHIE)
  2. Deep / shallow resistivity 6. Water saturation
  3. Density-neutron overlay    7. Reservoir / pay flags
  4. Sonic (measured + predicted)

Tracks are drawn only if the curves they need are present, so the same
function works for a raw LAS file and for a fully interpreted well.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .las_io import WellLog

# categorical slots (fixed order) and neutral inks
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SHALE = "#9c9a92"
MATRIX = "#f2d98a"
LW = 1.0


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.grid(True, which="minor", axis="x", color=GRID, linewidth=0.3)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=9, color=INK, pad=50, loc="left", fontweight="bold")


def _legend(ax, handles):
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=1 if len(handles) < 3 else 2,
        columnspacing=0.8,
        fontsize=7,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.6,
        labelcolor=INK_2,
    )


def _line(label, color, ls="-"):
    return Line2D([0], [0], color=color, lw=1.6, ls=ls, label=label)


def plot_well(
    well: WellLog,
    results: Optional[pd.DataFrame] = None,
    zones: Optional[pd.DataFrame] = None,
    depth_range: Optional[tuple] = None,
    out: Optional[str] = None,
    title: Optional[str] = None,
):
    """Draw the log display. Returns the matplotlib Figure.

    ``results`` is the output of interpret()/flag_pay(); it may also contain
    a DT_PRED column from the ML predictor. ``zones`` (from pay_zones()) are
    shaded across every track.
    """
    df = well.data.copy()
    if results is not None:
        df = df.join(results, how="left", rsuffix="_R")
    if depth_range is not None:
        df = df.loc[depth_range[0] : depth_range[1]]
    z = df.index.to_numpy()

    def has(*cols):
        return all(c in df.columns and df[c].notna().any() for c in cols)

    tracks = []
    if has("GR"):
        tracks.append("gr")
    if has("RT"):
        tracks.append("res")
    if has("RHOB", "NPHI"):
        tracks.append("nd")
    if has("DT") or has("DT_PRED"):
        tracks.append("dt")
    if has("VSH", "PHIE"):
        tracks.append("vol")
    if has("SW"):
        tracks.append("sw")
    if has("PAY_FLAG"):
        tracks.append("flag")
    if not tracks:
        raise ValueError("nothing to plot: no recognised curves")

    widths = {"gr": 1.2, "res": 1.2, "nd": 1.4, "dt": 1.1, "vol": 1.2, "sw": 1.0, "flag": 0.45}
    fig, axes = plt.subplots(
        1,
        len(tracks),
        sharey=True,
        figsize=(sum(widths[t] for t in tracks) * 1.9, 12),
        gridspec_kw={"width_ratios": [widths[t] for t in tracks], "wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, track in zip(axes, tracks):
        if track == "gr":
            _style(ax, "Gamma ray")
            ax.plot(df["GR"], z, color=C["green"], lw=LW)
            ax.set_xlim(0, 150)
            ax.set_xticks([30, 60, 90, 120])
            handles = [_line("GR (gAPI) 0-150", C["green"])]
            if has("CALI"):
                cal = ax.twiny()
                cal.plot(df["CALI"], z, color=INK_2, lw=0.8, ls="--")
                cal.set_xlim(6, 16)
                cal.set_xticks([])
                for spine in cal.spines.values():
                    spine.set_color(GRID)
                handles.append(_line("CALI (in) 6-16", INK_2, "--"))
            _legend(ax, handles)

        elif track == "res":
            _style(ax, "Resistivity")
            ax.set_xscale("log")
            ax.set_xlim(0.2, 2000)
            ax.plot(df["RT"], z, color=C["red"], lw=LW)
            handles = [_line("RT (ohm.m)", C["red"])]
            if has("RXO"):
                ax.plot(df["RXO"], z, color=C["blue"], lw=LW, ls="--")
                handles.append(_line("RXO (ohm.m)", C["blue"], "--"))
            _legend(ax, handles)

        elif track == "nd":
            _style(ax, "Density / neutron")
            rho_lo, rho_hi = 1.95, 2.95
            n_lo, n_hi = 0.45, -0.15
            ax.plot(df["RHOB"], z, color=C["red"], lw=LW)
            ax.set_xlim(rho_lo, rho_hi)
            ax.set_xticks([2.2, 2.45, 2.7])
            # put neutron on the density scale so the crossover can be shaded
            nphi_on_rho = rho_lo + (df["NPHI"] - n_lo) / (n_hi - n_lo) * (rho_hi - rho_lo)
            ax.plot(nphi_on_rho, z, color=C["blue"], lw=LW, ls="--")
            ax.fill_betweenx(z, df["RHOB"], nphi_on_rho, where=df["RHOB"] < nphi_on_rho,
                             color=C["yellow"], alpha=0.45, lw=0, interpolate=True)
            ax.fill_betweenx(z, df["RHOB"], nphi_on_rho, where=df["RHOB"] >= nphi_on_rho,
                             color=SHALE, alpha=0.25, lw=0, interpolate=True)
            _legend(ax, [
                _line("RHOB (g/cc) 1.95-2.95", C["red"]),
                _line("NPHI (v/v) 0.45 to -0.15", C["blue"], "--"),
                Patch(color=C["yellow"], alpha=0.45, label="crossover"),
                Patch(color=SHALE, alpha=0.25, label="separation"),
            ])

        elif track == "dt":
            _style(ax, "Sonic")
            handles = []
            if has("DT_PRED"):
                ax.plot(df["DT_PRED"], z, color=C["orange"], lw=LW, ls="--")
                handles.append(_line("DT predicted", C["orange"], "--"))
            if has("DT"):
                ax.plot(df["DT"], z, color=C["violet"], lw=LW)
                handles.insert(0, _line("DT (us/ft) 140-40", C["violet"]))
            ax.set_xlim(140, 40)
            ax.set_xticks([120, 90, 60])
            _legend(ax, handles)

        elif track == "vol":
            _style(ax, "Volumetrics")
            vsh = df["VSH"].clip(0, 1)
            phie = df["PHIE"].clip(0, 1)
            ax.fill_betweenx(z, 0, vsh, color=SHALE, lw=0)
            ax.fill_betweenx(z, vsh, 1 - phie, where=(1 - phie) > vsh, color=MATRIX, lw=0)
            ax.fill_betweenx(z, 1 - phie, 1, color=C["blue"], alpha=0.75, lw=0)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.25, 0.5, 0.75])
            _legend(ax, [
                Patch(color=SHALE, label="Vsh"),
                Patch(color=MATRIX, label="Matrix"),
                Patch(color=C["blue"], alpha=0.75, label="PHIE"),
            ])

        elif track == "sw":
            _style(ax, "Saturation")
            sw = df["SW"].clip(0, 1)
            ax.plot(sw, z, color=C["blue"], lw=LW)
            # axis runs 1 -> 0, so water fills from SW to the right and
            # hydrocarbon (1 - SW) from SW to the left edge
            ax.fill_betweenx(z, sw, 1, color=C["aqua"], alpha=0.5, lw=0)
            ax.set_xlim(1, 0)
            ax.set_xticks([0.75, 0.5, 0.25])
            _legend(ax, [_line("SW (v/v) 1-0", C["blue"]), Patch(color=C["aqua"], alpha=0.5, label="Hydrocarbon")])

        elif track == "flag":
            _style(ax, "Flags")
            ax.fill_betweenx(z, 0, 0.5, where=df["RES_FLAG"] == 1, color=C["yellow"], lw=0, step="mid")
            ax.fill_betweenx(z, 0.5, 1, where=df["PAY_FLAG"] == 1, color=C["red"], lw=0, step="mid")
            ax.set_xlim(0, 1)
            ax.set_xticks([0.25, 0.75])
            ax.set_xticklabels(["Res", "Pay"])
            _legend(ax, [Patch(color=C["yellow"], label="Res"), Patch(color=C["red"], label="Pay")])

    if zones is not None and len(zones):
        for ax in axes:
            for _, zone in zones.iterrows():
                ax.axhspan(zone["top"], zone["base"], color=C["red"], alpha=0.07, lw=0, zorder=0)

    axes[0].set_ylim(z.max(), z.min())
    axes[0].set_ylabel(f"Depth ({well.depth_unit.lower()})", color=INK_2, fontsize=8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)

    fig.suptitle(title or well.name, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.02)
    if out:
        fig.savefig(out, dpi=150, facecolor="white")
        plt.close(fig)
    return fig
