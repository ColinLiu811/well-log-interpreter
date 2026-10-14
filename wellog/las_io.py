"""Reading and writing LAS files.

Vendors use many different mnemonics for the same measurement (ILD, RT,
AT90, ...). On load every curve we recognise is renamed to a canonical
mnemonic so the rest of the package only has to deal with one name per
measurement. Curves we don't recognise are kept under their original name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import lasio
import numpy as np
import pandas as pd

# canonical mnemonic -> known aliases (upper case, checked in order)
CURVE_ALIASES: Dict[str, list] = {
    "GR": ["GR", "GRC", "SGR", "CGR", "GR_EDTC", "HGR", "GAMMA"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "ZDEN", "HDEN", "DENS"],
    "NPHI": ["NPHI", "TNPH", "NPOR", "CNC", "PHIN", "NPHI_LS", "CNCF"],
    "RT": ["RT", "ILD", "RD", "LLD", "AT90", "RILD", "HDRS", "RES_DEEP", "AHT90"],
    "RXO": ["RXO", "MSFL", "RXOZ", "SFL", "LLS", "RS"],
    "DT": ["DT", "DTC", "DTCO", "AC", "SONIC", "DT4P"],
    "DTS": ["DTS", "DTSM", "DT4S", "DTSH"],
    "CALI": ["CALI", "CAL", "HCAL", "C1", "CALS"],
    "PEF": ["PEF", "PE", "PEFZ"],
    "SP": ["SP", "SSP"],
}

DEFAULT_UNITS = {
    "DEPT": "M",
    "GR": "GAPI",
    "RHOB": "G/C3",
    "NPHI": "V/V",
    "RT": "OHMM",
    "RXO": "OHMM",
    "DT": "US/F",
    "DTS": "US/F",
    "CALI": "IN",
    "PEF": "B/E",
    "SP": "MV",
}

_ALIAS_LOOKUP = {alias: canon for canon, aliases in CURVE_ALIASES.items() for alias in aliases}


def canonical_name(mnemonic: str) -> str:
    """Map a vendor mnemonic to its canonical name (or return it unchanged)."""
    key = mnemonic.strip().upper()
    # lasio de-duplicates repeated mnemonics as NAME:1, NAME:2
    base = key.split(":")[0]
    return _ALIAS_LOOKUP.get(base, key)


@dataclass
class WellLog:
    """A single well's curves plus the header information we care about."""

    name: str
    data: pd.DataFrame  # indexed by depth
    units: Dict[str, str] = field(default_factory=dict)
    header: Dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None

    @property
    def depth(self) -> np.ndarray:
        return self.data.index.to_numpy()

    @property
    def depth_unit(self) -> str:
        return self.units.get("DEPT", "M")

    def has(self, *curves: str) -> bool:
        return all(c in self.data.columns and self.data[c].notna().any() for c in curves)

    def __getitem__(self, curve: str) -> pd.Series:
        return self.data[curve]

    def summary(self) -> pd.DataFrame:
        rows = []
        for col in self.data.columns:
            s = self.data[col]
            rows.append(
                {
                    "curve": col,
                    "unit": self.units.get(col, ""),
                    "coverage": s.notna().mean(),
                    "min": s.min(),
                    "max": s.max(),
                    "mean": s.mean(),
                }
            )
        return pd.DataFrame(rows).set_index("curve")


def load_las(path: Union[str, Path], rename: bool = True) -> WellLog:
    """Load a LAS (1.2 / 2.0) file into a :class:`WellLog`.

    Null values are converted to NaN by lasio. When ``rename`` is true,
    curves are renamed to canonical mnemonics; if two curves map to the same
    canonical name the first one wins and the other keeps its original name.
    """
    path = Path(path)
    las = lasio.read(str(path))
    df = las.df()
    df.index.name = "DEPT"

    units = {"DEPT": (las.curves[0].unit or "M").upper()}
    columns = {}
    for curve in las.curves[1:]:
        original = curve.mnemonic
        new = canonical_name(original) if rename else original
        if new in columns.values():
            new = original
        columns[original] = new
        units[new] = curve.unit
    df = df.rename(columns=columns)
    df = df.sort_index()
    df = df.astype(float)

    header = {}
    for item in las.well:
        if item.value not in (None, ""):
            header[item.mnemonic] = str(item.value)
    name = header.get("WELL") or path.stem

    return WellLog(name=name, data=df, units=units, header=header, source=str(path))


def save_las(well: WellLog, path: Union[str, Path], extra: Optional[pd.DataFrame] = None) -> Path:
    """Write a WellLog (optionally with extra computed curves) to LAS 2.0."""
    path = Path(path)
    data = well.data if extra is None else well.data.join(extra, rsuffix="_CALC")

    las = lasio.LASFile()
    for key, value in well.header.items():
        if key in ("STRT", "STOP", "STEP", "NULL"):
            continue
        las.well[key] = lasio.HeaderItem(key, value=value)
    las.well["WELL"] = lasio.HeaderItem("WELL", value=well.name)
    las.well["NULL"].value = -999.25

    las.append_curve("DEPT", data.index.to_numpy(), unit=well.depth_unit)
    for col in data.columns:
        unit = well.units.get(col, DEFAULT_UNITS.get(col, ""))
        las.append_curve(col, data[col].to_numpy(dtype=float), unit=unit)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        las.write(fh, version=2.0)
    return path
