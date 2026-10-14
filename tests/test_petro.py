import numpy as np
import pytest

from wellog import petro


def test_gamma_ray_index_bounds():
    igr = petro.gamma_ray_index(np.array([0, 20, 70, 120, 200]), 20, 120)
    np.testing.assert_allclose(igr, [0, 0, 0.5, 1, 1])


@pytest.mark.parametrize("method", petro.VSH_METHODS)
def test_vsh_methods_monotonic_and_bounded(method):
    igr = np.linspace(0, 1, 50)
    vsh = petro.shale_volume(igr, method)
    assert vsh[0] == pytest.approx(0, abs=1e-6)
    assert vsh[-1] == pytest.approx(1, abs=0.02)
    assert np.all(np.diff(vsh) >= -1e-12)
    # non-linear transforms are optimistic relative to linear
    assert np.all(vsh <= igr + 1e-9)


def test_density_porosity():
    assert petro.density_porosity(2.65) == pytest.approx(0.0)
    assert petro.density_porosity(1.0) == pytest.approx(1.0)
    assert petro.density_porosity(2.32) == pytest.approx(0.2)


def test_sonic_porosity_wyllie():
    assert petro.sonic_porosity(55.5) == pytest.approx(0)
    assert petro.sonic_porosity(55.5 + 0.2 * (189 - 55.5)) == pytest.approx(0.2)


def test_neutron_density_gas_correction():
    # liquid-filled: average; gas crossover (phid > phin): rms
    assert petro.neutron_density_porosity(0.25, 0.21) == pytest.approx(0.23)
    assert petro.neutron_density_porosity(0.10, 0.30) == pytest.approx(np.sqrt((0.01 + 0.09) / 2))


def test_archie_recovers_known_sw():
    phi, rw, sw_true = 0.2, 0.05, 0.3
    rt = rw / (phi**2 * sw_true**2)
    assert petro.archie_sw(rt, phi, rw) == pytest.approx(sw_true)


def test_archie_water_zone_and_zero_porosity():
    assert petro.archie_sw(0.05 / 0.04, 0.2, 0.05) == pytest.approx(1.0)
    assert petro.archie_sw(10.0, 0.0, 0.05) == 1.0
    assert np.isnan(petro.archie_sw(np.nan, 0.2, 0.05))


@pytest.mark.parametrize("n", [2.0, 1.8, 2.4])
def test_simandoux_inverts_forward_model(n):
    rng = np.random.default_rng(0)
    phi = rng.uniform(0.08, 0.3, 200)
    vsh = rng.uniform(0, 0.5, 200)
    sw = rng.uniform(0.15, 1.0, 200)
    rw, rsh, a, m = 0.04, 2.5, 1.0, 2.0
    rt = 1 / (phi**m * sw**n / (a * rw) + vsh * sw / rsh)
    np.testing.assert_allclose(petro.simandoux_sw(rt, phi, vsh, rw, rsh, a, m, n), sw, atol=1e-6)


def test_simandoux_reduces_to_archie_without_shale():
    rt = np.array([1.25, 5.0, 20.0])
    phi = np.array([0.2, 0.2, 0.2])
    np.testing.assert_allclose(
        petro.simandoux_sw(rt, phi, np.zeros(3), 0.05, 2.0),
        petro.archie_sw(rt, phi, 0.05),
    )


def test_shale_lowers_simandoux_sw():
    rt, phi, rw, rsh = 8.0, 0.18, 0.05, 2.0
    assert petro.simandoux_sw(rt, phi, 0.3, rw, rsh) < petro.archie_sw(rt, phi, rw)


def test_estimate_rw_from_water_sand():
    rng = np.random.default_rng(1)
    phi = rng.uniform(0.15, 0.3, 500)
    rt = 0.05 / phi**2
    vsh = np.full(500, 0.05)
    assert petro.estimate_rw(rt, phi, vsh) == pytest.approx(0.05, rel=1e-6)
