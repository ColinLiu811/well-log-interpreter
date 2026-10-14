import numpy as np

from wellog import load_las, save_las
from wellog.las_io import canonical_name


def test_canonical_name():
    assert canonical_name("ild") == "RT"
    assert canonical_name("RHOZ") == "RHOB"
    assert canonical_name("TNPH") == "NPHI"
    assert canonical_name("DTCO") == "DT"
    assert canonical_name("GR:1") == "GR"
    assert canonical_name("FOO") == "FOO"


def test_aliases_are_renamed(las_dir):
    w = load_las(las_dir / "B.las")
    assert {"GR", "RHOB", "NPHI", "RT", "DT"} <= set(w.data.columns)
    assert w.name == "B"
    assert w.units["RT"] == "OHMM"


def test_nulls_become_nan(las_dir):
    w = load_las(las_dir / "C.las")
    dt = w.data["DT"]
    assert dt.isna().any()
    assert not (dt == -999.25).any()
    assert dt.loc[1600:1700].isna().all()


def test_roundtrip(tmp_path, well_a):
    extra = well_a.data[["GR"]].rename(columns={"GR": "VSH"}) / 150
    out = save_las(well_a, tmp_path / "out.las", extra=extra)
    w2 = load_las(out)
    assert "VSH" in w2.data
    np.testing.assert_allclose(w2.data["GR"], well_a.data["GR"], rtol=1e-4)
