import pytest

from wellog import load_las
from wellog.synthetic import write_synthetic_las


@pytest.fixture(scope="session")
def las_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("las")
    write_synthetic_las(d / "A.las", "A", seed=11)
    write_synthetic_las(d / "B.las", "B", seed=12, mnemonics={"RT": "ILD", "RHOB": "RHOZ", "NPHI": "TNPH"})
    write_synthetic_las(d / "C.las", "C", seed=13, drop_dt=(1600.0, 1700.0))
    return d


@pytest.fixture(scope="session")
def well_a(las_dir):
    return load_las(las_dir / "A.las")
