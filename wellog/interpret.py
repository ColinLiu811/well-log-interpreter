"""End-to-end interpretation of a single well."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import petro
from .las_io import WellLog


@dataclass
class InterpretationParams:
    # shale volume
    vsh_method: str = "linear"
    gr_clean: Optional[float] = None  # picked from percentiles when None
    gr_shale: Optional[float] = None
    # porosity
    porosity_method: str = "neutron_density"  # or "density", "sonic"
    rho_ma: float = 2.65
    rho_fl: float = 1.0
    rho_sh: float = 2.55
    nphi_sh: float = 0.38
    dt_ma: float = 55.5
    dt_fl: float = 189.0
    dt_sh: float = 100.0
    # saturation
    sw_method: str = "simandoux"  # or "archie"
    rw: Optional[float] = None  # estimated from Rwa when None
    rsh: Optional[float] = None  # estimated from shales when None
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


def interpret(well: WellLog, params: Optional[InterpretationParams] = None) -> pd.DataFrame:
    """Compute VSH, PHIT/PHIE, SW and BVW for a well.

    Returns a DataFrame on the well's depth index. Parameters that were
    estimated from the data (GR endpoints, Rw, Rsh) are written back onto
    ``params`` so the caller can report them.
    """
    p = params or InterpretationParams()
    df = well.data
    out = pd.DataFrame(index=df.index)

    if not well.has("GR"):
        raise ValueError(f"{well.name}: a GR curve is required for shale volume")
    if p.gr_clean is None or p.gr_shale is None:
        lo, hi = petro.pick_gr_endpoints(df["GR"])
        p.gr_clean = lo if p.gr_clean is None else p.gr_clean
        p.gr_shale = hi if p.gr_shale is None else p.gr_shale
    igr = petro.gamma_ray_index(df["GR"], p.gr_clean, p.gr_shale)
    out["VSH"] = petro.shale_volume(igr, p.vsh_method)
    vsh = out["VSH"].to_numpy()

    if p.porosity_method == "neutron_density":
        if not well.has("RHOB", "NPHI"):
            raise ValueError(f"{well.name}: neutron-density porosity needs RHOB and NPHI")
        phid = petro.density_porosity(df["RHOB"], p.rho_ma, p.rho_fl)
        out["PHIT"] = np.clip(petro.neutron_density_porosity(df["NPHI"], phid), 0, 0.5)
        out["PHIE"] = petro.shale_corrected_nd_porosity(
            df["RHOB"], df["NPHI"], vsh, p.rho_ma, p.rho_fl, p.rho_sh, p.nphi_sh
        )
    elif p.porosity_method == "density":
        if not well.has("RHOB"):
            raise ValueError(f"{well.name}: density porosity needs RHOB")
        phit = np.clip(petro.density_porosity(df["RHOB"], p.rho_ma, p.rho_fl), 0, 0.5)
        phid_sh = (p.rho_ma - p.rho_sh) / (p.rho_ma - p.rho_fl)
        out["PHIT"] = phit
        out["PHIE"] = petro.effective_porosity(phit, vsh, phid_sh)
    elif p.porosity_method == "sonic":
        if not well.has("DT"):
            raise ValueError(f"{well.name}: sonic porosity needs DT")
        phit = np.clip(petro.sonic_porosity(df["DT"], p.dt_ma, p.dt_fl), 0, 0.5)
        out["PHIT"] = phit
        phis_sh = petro.sonic_porosity(p.dt_sh, p.dt_ma, p.dt_fl)
        out["PHIE"] = petro.effective_porosity(phit, vsh, phis_sh)
    else:
        raise ValueError(f"unknown porosity method {p.porosity_method!r}")
    phie = out["PHIE"].to_numpy()

    if not well.has("RT"):
        raise ValueError(f"{well.name}: a deep resistivity curve (RT) is required for saturation")
    rt = df["RT"].to_numpy()
    if p.rw is None:
        p.rw = petro.estimate_rw(rt, phie, vsh, p.a, p.m)
    if p.sw_method == "archie":
        out["SW"] = petro.archie_sw(rt, phie, p.rw, p.a, p.m, p.n)
    elif p.sw_method == "simandoux":
        if p.rsh is None:
            p.rsh = petro.estimate_rsh(rt, vsh)
        out["SW"] = petro.simandoux_sw(rt, phie, vsh, p.rw, p.rsh, p.a, p.m, p.n)
    else:
        raise ValueError(f"unknown saturation method {p.sw_method!r}")

    out["BVW"] = petro.bulk_volume_water(phie, out["SW"])
    out["SH"] = 1.0 - out["SW"]
    return out
