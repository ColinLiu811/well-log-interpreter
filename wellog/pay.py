"""Cutoff-based reservoir and pay flagging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cutoffs:
    vsh_max: float = 0.40
    phi_min: float = 0.08
    sw_max: float = 0.50
    min_thickness: float = 0.5  # in depth units; thinner pay intervals are dropped


def flag_pay(results: pd.DataFrame, cutoffs: Cutoffs = Cutoffs()) -> pd.DataFrame:
    """Add RES_FLAG (net reservoir) and PAY_FLAG (net pay) columns.

    Reservoir: Vsh <= vsh_max and PHIE >= phi_min.
    Pay: reservoir and Sw <= sw_max, in intervals at least min_thickness thick.
    """
    out = results.copy()
    vsh, phi, sw = out["VSH"], out["PHIE"], out["SW"]
    res = (vsh <= cutoffs.vsh_max) & (phi >= cutoffs.phi_min)
    pay = res & (sw <= cutoffs.sw_max)

    if cutoffs.min_thickness > 0:
        depth = out.index.to_numpy()
        pay_arr = pay.to_numpy().copy()
        for top_i, base_i in _runs(pay_arr):
            if _thickness(depth, top_i, base_i) < cutoffs.min_thickness:
                pay_arr[top_i : base_i + 1] = False
        pay = pd.Series(pay_arr, index=out.index)

    out["RES_FLAG"] = res.astype(int)
    out["PAY_FLAG"] = pay.astype(int)
    return out


def _runs(mask: np.ndarray):
    """Yield (start, end) index pairs (inclusive) of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    yield from zip(starts, ends)


def _thickness(depth: np.ndarray, i0: int, i1: int) -> float:
    step = np.median(np.diff(depth)) if depth.size > 1 else 0.0
    return float(depth[i1] - depth[i0] + step)


def pay_zones(results: pd.DataFrame, merge_gap: float = 0.0) -> pd.DataFrame:
    """Summarise contiguous pay intervals.

    ``merge_gap`` joins pay intervals separated by non-pay thinner than this
    (e.g. thin shale streaks). Gross is the interval thickness, net counts
    only pay-flagged samples. Averages are thickness-weighted over pay.
    """
    if "PAY_FLAG" not in results:
        raise ValueError("run flag_pay() first")
    depth = results.index.to_numpy()
    step = float(np.median(np.diff(depth))) if depth.size > 1 else 0.0
    flag = results["PAY_FLAG"].to_numpy().astype(bool)

    runs = list(_runs(flag))
    merged = []
    for start, end in runs:
        if merged and depth[start] - depth[merged[-1][1]] - step <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    rows = []
    for i, (start, end) in enumerate(merged, 1):
        seg = results.iloc[start : end + 1]
        pay = seg[seg["PAY_FLAG"] == 1]
        net = len(pay) * step
        rows.append(
            {
                "zone": i,
                "top": depth[start],
                "base": depth[end] + step,
                "gross": _thickness(depth, start, end),
                "net": net,
                "ntg": net / _thickness(depth, start, end),
                "phie_avg": pay["PHIE"].mean(),
                "sw_avg": pay["SW"].mean(),
                "vsh_avg": pay["VSH"].mean(),
                # hydrocarbon pore thickness: sum(phi * (1 - Sw) * h)
                "hcpt": float((pay["PHIE"] * (1 - pay["SW"])).sum() * step),
            }
        )
    cols = ["zone", "top", "base", "gross", "net", "ntg", "phie_avg", "sw_avg", "vsh_avg", "hcpt"]
    return pd.DataFrame(rows, columns=cols).set_index("zone")


def well_summary(results: pd.DataFrame) -> dict:
    """Whole-well net reservoir / net pay totals."""
    depth = results.index.to_numpy()
    step = float(np.median(np.diff(depth))) if depth.size > 1 else 0.0
    res = results["RES_FLAG"] == 1
    pay = results["PAY_FLAG"] == 1
    return {
        "interval": float(depth[-1] - depth[0] + step),
        "net_reservoir": float(res.sum() * step),
        "net_pay": float(pay.sum() * step),
        "phie_avg_pay": float(results.loc[pay, "PHIE"].mean()) if pay.any() else float("nan"),
        "sw_avg_pay": float(results.loc[pay, "SW"].mean()) if pay.any() else float("nan"),
        "hcpt": float((results.loc[pay, "PHIE"] * (1 - results.loc[pay, "SW"])).sum() * step),
    }
