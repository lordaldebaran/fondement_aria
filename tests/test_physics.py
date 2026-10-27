"""Tests de validation du module physics.py."""
import numpy as np
import pytest
from scipy import stats
from onn_arquS.config import K_B, LAMBDA_399
from onn_arquS.physics import (
    sigma_thermal, sigma_psf_gaussian, sample_positions_gaussian,
    simulate_langevin, tau_trap, motion_blur_factor, kappa_from_omega
)


def test_sigma_scales_as_sqrt_T():
    """sigma_thermal doit varier comme sqrt(T)."""
    omega = 2 * np.pi * 100e3
    s1 = sigma_thermal(1e-6, omega)
    s4 = sigma_thermal(4e-6, omega)
    assert abs(s4 / s1 - 2.0) < 1e-6


def test_sigma_matches_equipartition():
    """sigma^2 doit valoir k_B T / kappa."""
    omega = 2 * np.pi * 100e3
    T = 10e-6
    s = sigma_thermal(T, omega)
    kappa = kappa_from_omega(omega)
    assert abs(s ** 2 - K_B * T / kappa) / (s ** 2) < 1e-12


def test_gaussian_sampling_ks():
    """Test KS : les échantillons suivent bien N(0, sigma^2)."""
    omega = 2 * np.pi * 100e3
    T = 10e-6
    samples = sample_positions_gaussian(T, omega, n=10000)
    sigma = sigma_thermal(T, omega)
    _, p = stats.kstest(samples / sigma, "norm")
    assert p > 0.01, f"Test KS échoué : p = {p}"


def test_langevin_matches_gaussian():
    """La simulation de Langevin doit converger vers la même variance."""
    omega = 2 * np.pi * 100e3
    T = 10e-6
    gamma = 1e5
    tau = tau_trap(omega)
    dt = tau / 100
    n_steps = 200_000
    positions = simulate_langevin(T, omega, gamma, n_steps, dt)
    sigma_sim = positions.std()
    sigma_th = sigma_thermal(T, omega)
    assert abs(sigma_sim - sigma_th) / sigma_th < 0.2, \
        f"sigma_sim = {sigma_sim:.3e}, sigma_th = {sigma_th:.3e}"


def test_psf_order_of_magnitude():
    """La PSF doit être de l'ordre de grandeur de lambda/NA."""
    s = sigma_psf_gaussian(wavelength_m=LAMBDA_399, NA=0.7)
    # ordre de grandeur : 100-200 nm
    assert 1e-7 < s < 2e-7, f"sigma_psf = {s*1e9:.1f} nm hors plage attendue"


def test_thermal_dominated_by_psf():
    """Validation centrale : la variance thermique est noyée dans la PSF."""
    omega = 2 * np.pi * 100e3
    s_thermal = sigma_thermal(10e-6, omega)
    s_psf = sigma_psf_gaussian()
    ratio_var = (s_thermal / s_psf) ** 2
    assert ratio_var < 0.1, \
        f"ratio_var = {ratio_var:.3f}, attendu < 0.1"


def test_motion_blur_decreases_in_validity_range():
    """Le facteur de Lindballe DÉCROÎT pour alpha << 1.

    Physiquement, le flou de mouvement moyenne la position sur la durée
    d'exposition, ce qui réduit la variance de la position mesurée.
    La formule 1 - alpha/3 + alpha^2/12 décroît jusqu'à alpha=2 puis
    remonte (comportement non physique pour alpha > 1, d'où la limite
    de validité alpha << 1).
    """
    tau = tau_trap(2 * np.pi * 100e3)
    f_01 = motion_blur_factor(0.1 * tau, tau)
    f_02 = motion_blur_factor(0.2 * tau, tau)
    f_05 = motion_blur_factor(0.5 * tau, tau)
    # Décroissance stricte dans la plage de validité
    assert f_01 > f_02 > f_05, \
        f"Attendu f_01 > f_02 > f_05, obtenu {f_01:.4f}, {f_02:.4f}, {f_05:.4f}"
    # Toutes les valeurs doivent rester < 1 (réduction de variance)
    assert f_01 < 1.0 and f_02 < 1.0 and f_05 < 1.0


def test_motion_blur_known_values():
    """Vérification numérique de la formule à des points connus."""
    tau = 1.0  # normalisé
    # alpha = 0 -> facteur = 1
    assert abs(motion_blur_factor(0.0, tau) - 1.0) < 1e-12
    # alpha = 0.3 -> 1 - 0.1 + 0.0075 = 0.9075
    assert abs(motion_blur_factor(0.3, tau) - 0.9075) < 1e-10
    # alpha = 0.6 -> 1 - 0.2 + 0.03 = 0.83
    assert abs(motion_blur_factor(0.6, tau) - 0.83) < 1e-10


def test_motion_blur_minimum_at_alpha_2():
    """Le minimum de la formule est atteint à alpha = 2 (extrapolation)."""
    tau = 1.0
    f_2 = motion_blur_factor(2.0, tau)
    # 1 - 2/3 + 4/12 = 1 - 0.6667 + 0.3333 = 0.6667
    assert abs(f_2 - 2/3) < 1e-10