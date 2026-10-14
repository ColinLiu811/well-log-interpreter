"""Petrophysical calculations.

All functions are vectorised over numpy arrays / pandas Series and return
fractions (v/v), not percentages. NaNs propagate.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray

# --------------------------------------------------------------------------
# Shale volume
# --------------------------------------------------------------------------

VSH_METHODS = ("linear", "larionov_tertiary", "larionov_older", "clavier", "steiber")


def gamma_ray_index(gr: ArrayLike, gr_clean: float, gr_shale: float) -> ArrayLike:
    """IGR = (GR - GRclean) / (GRshale - GRclean), clipped to [0, 1]."""
    if gr_shale <= gr_clean:
        raise ValueError("gr_shale must be greater than gr_clean")
    igr = (np.asarray(gr, dtype=float) - gr_clean) / (gr_shale - gr_clean)
    return np.clip(igr, 0.0, 1.0)


def pick_gr_endpoints(gr: ArrayLike, low_pct: float = 5, high_pct: float = 95) -> tuple:
    """Pick clean-sand and shale GR values from percentiles of the curve."""
    gr = np.asarray(gr, dtype=float)
    gr = gr[np.isfinite(gr)]
    return float(np.percentile(gr, low_pct)), float(np.percentile(gr, high_pct))


def shale_volume(igr: ArrayLike, method: str = "linear") -> ArrayLike:
    """Convert gamma-ray index to shale volume.

    * ``linear`` - Vsh = IGR (upper bound, pessimistic)
    * ``larionov_tertiary`` - unconsolidated / Tertiary rocks
    * ``larionov_older`` - consolidated / older rocks
    * ``clavier`` - Clavier et al. (1971)
    * ``steiber`` - Steiber (1970)
    """
    igr = np.asarray(igr, dtype=float)
    if method == "linear":
        vsh = igr
    elif method == "larionov_tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "larionov_older":
        vsh = 0.33 * (2 ** (2 * igr) - 1)
    elif method == "clavier":
        vsh = 1.7 - np.sqrt(np.clip(3.38 - (igr + 0.7) ** 2, 0, None))
    elif method == "steiber":
        vsh = igr / (3 - 2 * igr)
    else:
        raise ValueError(f"unknown Vsh method {method!r}, choose from {VSH_METHODS}")
    return np.clip(vsh, 0.0, 1.0)


# --------------------------------------------------------------------------
# Porosity
# --------------------------------------------------------------------------


def density_porosity(rhob: ArrayLike, rho_ma: float = 2.65, rho_fl: float = 1.0) -> ArrayLike:
    """PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)."""
    return (rho_ma - np.asarray(rhob, dtype=float)) / (rho_ma - rho_fl)


def sonic_porosity(dt: ArrayLike, dt_ma: float = 55.5, dt_fl: float = 189.0, method: str = "wyllie") -> ArrayLike:
    """Sonic porosity, Wyllie time-average or Raymer-Hunt-Gardner."""
    dt = np.asarray(dt, dtype=float)
    if method == "wyllie":
        return (dt - dt_ma) / (dt_fl - dt_ma)
    if method == "rhg":
        return 0.625 * (dt - dt_ma) / dt
    raise ValueError("method must be 'wyllie' or 'rhg'")


def neutron_density_porosity(phin: ArrayLike, phid: ArrayLike, gas_correction: bool = True) -> ArrayLike:
    """Combine neutron and density porosity.

    Arithmetic mean normally. Where the density reads higher than the
    neutron (the gas crossover) the root-mean-square form is used, which
    corrects for the gas effect on both tools.
    """
    phin = np.asarray(phin, dtype=float)
    phid = np.asarray(phid, dtype=float)
    mean = 0.5 * (phin + phid)
    if not gas_correction:
        return mean
    rms = np.sqrt(0.5 * (phin**2 + phid**2))
    return np.where(phid > phin, rms, mean)


def effective_porosity(phit: ArrayLike, vsh: ArrayLike, phi_shale: float = 0.0) -> ArrayLike:
    """Remove the shale-bound porosity: PHIE = PHIT - Vsh * PHIT_shale.

    With ``phi_shale=0`` this reduces to clipping PHIT; pass the total
    porosity read in a nearby shale to correct shaly sands.
    """
    phit = np.asarray(phit, dtype=float)
    phie = phit - np.asarray(vsh, dtype=float) * phi_shale
    return np.clip(phie, 0.0, np.clip(phit, 0.0, 1.0))


def shale_corrected_nd_porosity(
    rhob: ArrayLike,
    nphi: ArrayLike,
    vsh: ArrayLike,
    rho_ma: float = 2.65,
    rho_fl: float = 1.0,
    rho_sh: float = 2.55,
    nphi_sh: float = 0.38,
) -> ArrayLike:
    """Neutron-density porosity with each tool corrected for shale first.

    PHID_corr = PHID - Vsh * PHID_shale, PHIN_corr = PHIN - Vsh * PHIN_shale,
    then the two are combined. This is the usual workflow for shaly sands.
    """
    vsh = np.asarray(vsh, dtype=float)
    phid = density_porosity(rhob, rho_ma, rho_fl)
    phid_sh = (rho_ma - rho_sh) / (rho_ma - rho_fl)
    phid_c = phid - vsh * phid_sh
    phin_c = np.asarray(nphi, dtype=float) - vsh * nphi_sh
    phi = neutron_density_porosity(phin_c, phid_c)
    return np.clip(phi, 0.0, 0.5)


# --------------------------------------------------------------------------
# Water saturation
# --------------------------------------------------------------------------


def archie_sw(rt: ArrayLike, phi: ArrayLike, rw: float, a: float = 1.0, m: float = 2.0, n: float = 2.0) -> ArrayLike:
    """Archie (1942): Sw = (a * Rw / (phi^m * Rt)) ^ (1/n).

    Valid for clean (shale-free) formations. Returned values are clipped to
    [0, 1]; zero porosity gives Sw = 1.
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = (a * rw / (phi**m * rt)) ** (1.0 / n)
    sw = np.where(phi <= 0, 1.0, sw)
    return np.where(np.isnan(rt) | np.isnan(phi), np.nan, np.clip(sw, 0.0, 1.0))


def simandoux_sw(
    rt: ArrayLike,
    phi: ArrayLike,
    vsh: ArrayLike,
    rw: float,
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> ArrayLike:
    """Simandoux (1963) shaly-sand saturation.

    Solves  1/Rt = phi^m * Sw^n / (a*Rw) + Vsh * Sw / Rsh  for Sw.

    For n = 2 the closed-form root of the quadratic is used. For other n the
    left-hand side is monotonic in Sw, so it is solved by bisection.
    With Vsh = 0 this reduces to Archie.
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.clip(np.asarray(phi, dtype=float), 0, None)
    vsh = np.clip(np.asarray(vsh, dtype=float), 0, 1)
    c = phi**m / (a * rw)  # clean-sand conductance term
    b = vsh / rsh  # shale conductance term

    with np.errstate(divide="ignore", invalid="ignore"):
        if n == 2:
            sw = (-b + np.sqrt(b**2 + 4 * c / rt)) / (2 * c)
        else:
            lo = np.zeros_like(rt)
            hi = np.ones_like(rt)
            target = 1.0 / rt
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                f = c * mid**n + b * mid - target
                lo = np.where(f < 0, mid, lo)
                hi = np.where(f >= 0, mid, hi)
            sw = 0.5 * (lo + hi)

    sw = np.where(c <= 0, 1.0, sw)
    return np.where(np.isnan(rt) | np.isnan(phi), np.nan, np.clip(sw, 0.0, 1.0))


def apparent_rw(rt: ArrayLike, phi: ArrayLike, a: float = 1.0, m: float = 2.0) -> ArrayLike:
    """Rwa = Rt * phi^m / a. Equals Rw in clean water-bearing rock."""
    return np.asarray(rt, dtype=float) * np.asarray(phi, dtype=float) ** m / a


def estimate_rw(
    rt: ArrayLike,
    phi: ArrayLike,
    vsh: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    vsh_max: float = 0.2,
    phi_min: float = 0.12,
    percentile: float = 10,
) -> float:
    """Estimate Rw from the Rwa curve in clean, porous rock.

    Hydrocarbons can only raise Rwa, so a low percentile of Rwa over the
    clean porous intervals approximates the water-zone value.
    """
    rwa = apparent_rw(rt, phi, a, m)
    mask = (np.asarray(vsh) <= vsh_max) & (np.asarray(phi) >= phi_min) & np.isfinite(rwa)
    if mask.sum() < 10:
        raise ValueError("not enough clean porous samples to estimate Rw; pass rw explicitly")
    return float(np.percentile(rwa[mask], percentile))


def estimate_rsh(rt: ArrayLike, vsh: ArrayLike, vsh_min: float = 0.8) -> float:
    """Shale resistivity: median Rt where Vsh is high."""
    rt = np.asarray(rt, dtype=float)
    mask = (np.asarray(vsh) >= vsh_min) & np.isfinite(rt)
    if mask.sum() < 10:
        raise ValueError("not enough shale samples to estimate Rsh; pass rsh explicitly")
    return float(np.median(rt[mask]))


def bulk_volume_water(phi: ArrayLike, sw: ArrayLike) -> ArrayLike:
    return np.asarray(phi, dtype=float) * np.asarray(sw, dtype=float)
