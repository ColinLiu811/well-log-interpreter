import numpy as np
import pandas as pd
import pytest

from wellog.interpret import InterpretationParams, interpret
from wellog.pay import Cutoffs, flag_pay, pay_zones, well_summary
from wellog.synthetic import make_model


@pytest.mark.parametrize("sw_method", ["archie", "simandoux"])
def test_interpret_recovers_model(well_a, sw_method):
    model = make_model(seed=11)
    params = InterpretationParams(sw_method=sw_method)
    res = interpret(well_a, params)
    assert {"VSH", "PHIT", "PHIE", "SW", "BVW"} <= set(res.columns)
    assert 0.02 < params.rw < 0.08

    reservoir = (model["phi"] > 0.12) & (model["vsh"] < 0.5)
    assert np.nanmean(np.abs(res["PHIE"].to_numpy() - model["phi"])[reservoir]) < 0.04
    assert np.nanmean(np.abs(res["SW"].to_numpy() - model["sw"])[reservoir]) < 0.12


def test_simandoux_beats_archie_in_shaly_sand(well_a):
    model = make_model(seed=11)
    shaly = (model["phi"] > 0.08) & (model["vsh"] > 0.2) & (model["vsh"] < 0.5)
    err = {}
    for method in ["archie", "simandoux"]:
        sw = interpret(well_a, InterpretationParams(sw_method=method, rw=0.05, rsh=2.0))["SW"].to_numpy()
        err[method] = np.nanmean(np.abs(sw - model["sw"])[shaly])
    assert err["simandoux"] < err["archie"]


def test_pay_is_above_contact(well_a):
    model = make_model(seed=11)
    res = flag_pay(interpret(well_a))
    zones = pay_zones(res, merge_gap=1.0)
    assert len(zones) > 0
    assert zones["base"].max() <= model["owc"] + 2.0
    assert (zones["net"] <= zones["gross"] + 1e-9).all()


def test_flag_pay_cutoffs_and_min_thickness():
    idx = pd.Index(np.round(np.arange(100, 110, 0.5), 2), name="DEPT")
    df = pd.DataFrame({"VSH": 0.1, "PHIE": 0.2, "SW": 0.9}, index=idx)
    df.loc[101.0:103.0, "SW"] = 0.3  # 2.5 m pay
    df.loc[106.0, "SW"] = 0.3  # 0.5 m pay, dropped by 1 m min thickness
    flagged = flag_pay(df, Cutoffs(min_thickness=1.0))
    assert flagged["RES_FLAG"].all()
    assert flagged["PAY_FLAG"].sum() == 5
    zones = pay_zones(flagged)
    assert len(zones) == 1
    assert zones.iloc[0]["top"] == 101.0
    assert zones.iloc[0]["gross"] == pytest.approx(2.5)
    assert well_summary(flagged)["net_pay"] == pytest.approx(2.5)


def test_pay_zones_merge_gap():
    idx = pd.Index(np.round(np.arange(0, 10, 0.5), 2))
    df = pd.DataFrame({"VSH": 0.1, "PHIE": 0.2, "SW": 0.3, "RES_FLAG": 1, "PAY_FLAG": 1}, index=idx)
    df.loc[4.0, "PAY_FLAG"] = 0
    assert len(pay_zones(df)) == 2
    merged = pay_zones(df, merge_gap=0.5)
    assert len(merged) == 1
    assert merged.iloc[0]["net"] == pytest.approx(merged.iloc[0]["gross"] - 0.5)


def test_missing_curve_raises(well_a):
    from wellog.las_io import WellLog

    w = WellLog(name="x", data=well_a.data.drop(columns=["RT"]))
    with pytest.raises(ValueError, match="RT"):
        interpret(w)


def test_plot_writes_file(tmp_path, well_a):
    from wellog.plot import plot_well

    res = flag_pay(interpret(well_a))
    out = tmp_path / "log.png"
    plot_well(well_a, res, pay_zones(res), out=str(out))
    assert out.stat().st_size > 10_000
