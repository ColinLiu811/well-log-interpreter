import numpy as np

from wellog import load_las
from wellog.ml import LogPredictor, leave_one_well_out
from wellog.synthetic import forward_model, make_model


def test_fill_gap_close_to_truth(las_dir, tmp_path):
    train = [load_las(las_dir / "A.las"), load_las(las_dir / "B.las")]
    target = load_las(las_dir / "C.las")
    predictor = LogPredictor().fit(train)

    filled = predictor.fill(target)
    assert filled["DT_FILLED"].notna().all()
    gap = target.data["DT"].isna().to_numpy()
    truth = forward_model(make_model(seed=13), seed=13)["DT"]
    rmse = np.sqrt(np.mean((filled["DT_PRED"].to_numpy()[gap] - truth[gap]) ** 2))
    assert rmse < 4.0  # us/ft; synthetic noise is 1.5

    # measured samples are kept as-is
    keep = ~gap
    np.testing.assert_array_equal(filled["DT_FILLED"].to_numpy()[keep], target.data["DT"].to_numpy()[keep])

    path = predictor.save(tmp_path / "m.joblib")
    np.testing.assert_allclose(LogPredictor.load(path).predict(target), filled["DT_PRED"])


def test_leave_one_well_out(las_dir):
    wells = [load_las(las_dir / f"{n}.las") for n in "ABC"]
    cv = leave_one_well_out(wells)
    assert list(cv.index) == ["A", "B", "C"]
    assert (cv["r2"] > 0.85).all()
    assert (cv["rmse"] < cv["rmse_linear"]).mean() >= 2 / 3
