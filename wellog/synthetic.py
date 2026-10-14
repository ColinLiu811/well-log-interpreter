"""Generate synthetic but physically consistent LAS files.

Real LAS data is often proprietary, so the repo ships with a generator that
builds a layered earth model (shale / sand / shaly sand / limestone with an
oil column above an oil-water contact) and forward-models the logs from it.
Resistivity is forward-modelled with the Simandoux equation and density,
neutron and sonic with simple volumetric mixing laws, so the interpretation
code should recover the model reasonably well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import lasio
import numpy as np

# matrix / fluid end points used by the model
RHO = {"sand": 2.65, "lime": 2.71, "shale": 2.55, "water": 1.05, "oil": 0.80}
DT = {"sand": 55.5, "lime": 47.5, "shale": 100.0, "water": 189.0, "oil": 230.0}
NPHI_SHALE = 0.38
NPHI_LIME_OFFSET = -0.03  # sandstone-scaled neutron reading low in limestone
GR_CLEAN, GR_SHALE = 25.0, 130.0
RW, RSH, A, M, N = 0.05, 2.0, 1.0, 2.0, 2.0


def _layers(rng: np.random.Generator, top: float, base: float):
    """Random sequence of (top, base, lithology) layers."""
    layers = []
    z = top
    choices = ["shale", "sand", "shaly_sand", "lime"]
    probs = [0.42, 0.30, 0.18, 0.10]
    while z < base:
        lith = rng.choice(choices, p=probs)
        thick = rng.uniform(4, 25) if lith == "shale" else rng.uniform(2, 15)
        layers.append((z, min(z + thick, base), lith))
        z += thick
    return layers


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Gaussian smoothing to mimic the vertical response of a logging tool."""
    if width <= 1:
        return x
    k = np.exp(-0.5 * (np.arange(-2 * width, 2 * width + 1) / width) ** 2)
    k /= k.sum()
    padded = np.pad(x, 2 * width, mode="edge")
    return np.convolve(padded, k, mode="same")[2 * width : -2 * width]


def make_model(
    seed: int = 0,
    top: float = 1500.0,
    base: float = 1900.0,
    step: float = 0.1524,
    owc: Optional[float] = None,
):
    """Build the 'true' earth model sampled on a regular depth grid."""
    rng = np.random.default_rng(seed)
    depth = np.round(np.arange(top, base, step), 4)
    owc = owc if owc is not None else rng.uniform(top + 0.45 * (base - top), top + 0.75 * (base - top))

    vsh = np.zeros_like(depth)
    phi = np.zeros_like(depth)
    lime = np.zeros_like(depth)
    for ztop, zbase, lith in _layers(rng, top, base):
        idx = (depth >= ztop) & (depth < zbase)
        compaction = 1.0 - 0.25 * (ztop - top) / (base - top)
        if lith == "shale":
            vsh[idx] = rng.uniform(0.75, 0.95)
            phi[idx] = 0.0
        elif lith == "sand":
            vsh[idx] = rng.uniform(0.02, 0.12)
            phi[idx] = rng.uniform(0.18, 0.30) * compaction
        elif lith == "shaly_sand":
            vsh[idx] = rng.uniform(0.25, 0.45)
            phi[idx] = rng.uniform(0.10, 0.18) * compaction
        else:
            vsh[idx] = rng.uniform(0.0, 0.08)
            phi[idx] = rng.uniform(0.03, 0.12)
            lime[idx] = 1.0

    # small in-layer heterogeneity
    vsh = np.clip(vsh + rng.normal(0, 0.02, depth.size), 0, 1)
    phi = np.clip(phi + rng.normal(0, 0.008, depth.size) * (phi > 0), 0, 0.35)
    phi = np.minimum(phi, 1 - vsh)

    # saturation: water below OWC; above it Sw falls with porosity and
    # height above the contact (crude capillary-pressure behaviour)
    sw = np.ones_like(depth)
    above = (depth < owc) & (phi > 0.08) & (lime == 0)
    height = np.clip(owc - depth, 0, None)
    swirr = np.clip(0.05 / np.maximum(phi, 1e-3) * 0.6, 0.12, 0.6)
    sw_hc = swirr + (1 - swirr) * np.exp(-height / 4.0)
    sw[above] = sw_hc[above]
    sw = np.clip(sw, 0.1, 1.0)

    return {"depth": depth, "vsh": vsh, "phi": phi, "sw": sw, "lime": lime, "owc": owc}


def forward_model(model: dict, seed: int = 0, noise: float = 1.0) -> dict:
    """Compute log responses for an earth model."""
    rng = np.random.default_rng(seed + 1000)
    vsh, phi, sw, lime = model["vsh"], model["phi"], model["sw"], model["lime"]
    vma = np.clip(1 - vsh - phi, 0, 1)

    rho_ma = np.where(lime > 0, RHO["lime"], RHO["sand"])
    dt_ma = np.where(lime > 0, DT["lime"], DT["sand"])
    rho_fl = sw * RHO["water"] + (1 - sw) * RHO["oil"]
    dt_fl = sw * DT["water"] + (1 - sw) * DT["oil"]

    gr = GR_CLEAN + vsh * (GR_SHALE - GR_CLEAN)
    rhob = vma * rho_ma + vsh * RHO["shale"] + phi * rho_fl
    nphi = phi * (sw + 0.95 * (1 - sw)) + vsh * NPHI_SHALE + lime * NPHI_LIME_OFFSET
    dt = vma * dt_ma + vsh * DT["shale"] + phi * dt_fl

    # Simandoux forward model: 1/Rt = phi^m Sw^n / (a Rw) + Vsh Sw / Rsh
    cond = (np.maximum(phi, 1e-4) ** M) * sw**N / (A * RW) + vsh * sw / RSH
    rt = 1.0 / np.maximum(cond, 1e-4)
    rt = np.clip(rt, 0.2, 2000)

    w = 4
    gr = _smooth(gr, w) + rng.normal(0, 4.0 * noise, gr.size)
    rhob = _smooth(rhob, w) + rng.normal(0, 0.015 * noise, rhob.size)
    nphi = _smooth(nphi, w) + rng.normal(0, 0.012 * noise, nphi.size)
    dt = _smooth(dt, w) + rng.normal(0, 1.5 * noise, dt.size)
    rt = 10 ** (_smooth(np.log10(rt), w + 2) + rng.normal(0, 0.03 * noise, rt.size))

    # caliper: washouts in shale
    cali = 8.5 + np.where(vsh > 0.6, rng.uniform(0.0, 1.5), 0.0) * _smooth((vsh > 0.6).astype(float), 6)
    cali = cali + rng.normal(0, 0.05, cali.size)

    return {"GR": gr, "RHOB": rhob, "NPHI": nphi, "RT": rt, "DT": dt, "CALI": cali}


_UNITS = {"GR": "GAPI", "RHOB": "G/C3", "NPHI": "V/V", "RT": "OHMM", "DT": "US/F", "CALI": "IN"}


def write_synthetic_las(
    path: Union[str, Path],
    well_name: str = "SYN-1",
    seed: int = 0,
    drop_dt: Optional[tuple] = None,
    mnemonics: Optional[dict] = None,
    **model_kwargs,
) -> Path:
    """Write a synthetic well to ``path``.

    ``drop_dt`` is an optional (top, base) depth interval where the sonic is
    set to null, to simulate a missing log section. ``mnemonics`` lets you
    rename curves (e.g. {"RT": "ILD"}) to exercise alias handling.
    """
    model = make_model(seed=seed, **model_kwargs)
    logs = forward_model(model, seed=seed)
    depth = model["depth"]

    if drop_dt is not None:
        if drop_dt == "all":
            logs.pop("DT")
        else:
            gap = (depth >= drop_dt[0]) & (depth <= drop_dt[1])
            logs["DT"] = np.where(gap, np.nan, logs["DT"])

    mnemonics = mnemonics or {}
    las = lasio.LASFile()
    las.well["WELL"].value = well_name
    las.well["COMP"].value = "SYNTHETIC"
    las.well["FLD"].value = "DEMO FIELD"
    las.well["NULL"].value = -999.25
    las.params["OWC"] = lasio.HeaderItem("OWC", unit="M", value=round(float(model["owc"]), 2), descr="True oil-water contact")
    las.params["RW"] = lasio.HeaderItem("RW", unit="OHMM", value=RW, descr="Formation water resistivity")
    las.append_curve("DEPT", depth, unit="M")
    for name, values in logs.items():
        las.append_curve(mnemonics.get(name, name), values, unit=_UNITS[name])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        las.write(fh, version=2.0)
    return path


def write_demo_dataset(out_dir: Union[str, Path] = "data") -> list:
    """The example wells shipped in ``data/``."""
    out_dir = Path(out_dir)
    return [
        write_synthetic_las(out_dir / "SYN-1.las", "SYN-1", seed=1),
        write_synthetic_las(out_dir / "SYN-2.las", "SYN-2", seed=2, mnemonics={"RT": "ILD", "RHOB": "RHOZ", "NPHI": "TNPH"}),
        write_synthetic_las(out_dir / "SYN-3.las", "SYN-3", seed=3),
        write_synthetic_las(out_dir / "SYN-4.las", "SYN-4", seed=4, drop_dt=(1620.0, 1760.0)),
    ]


if __name__ == "__main__":
    for p in write_demo_dataset():
        print("wrote", p)
