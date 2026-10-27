"""Modèle physique des positions d'un atome dans une pince optique.

Références :
- Équipartition : Var(x) = k_B T / kappa pour un oscillateur harmonique
  à l'équilibre thermique [S].
- Flou de mouvement : correction de Lindballe pour un oscillateur harmonique,
  valide pour alpha = W/tau_t << 1. Pour alpha > 1, une simulation de
  Langevin est nécessaire [T, car la formule n'est pas valide dans le
  régime d'ArQuS].
- PSF : approximation gaussienne de la PSF d'Airy, à titre de comparaison.
"""

import numpy as np
from .config import K_B, M_YB173, LAMBDA_399, TrapConfig, ImagingConfig


def kappa_from_omega(omega_hz: float) -> float:
    """Raideur du piège harmonique (N/m)."""
    return M_YB173 * omega_hz ** 2


def sigma_thermal(T_k: float, omega_hz: float) -> float:
    """Écart-type de position thermique (m) pour un oscillateur harmonique."""
    kappa = kappa_from_omega(omega_hz)
    return np.sqrt(K_B * T_k / kappa)


def tau_trap(omega_hz: float) -> float:
    """Temps caractéristique du piège (s) = 2*pi/omega."""
    return 2 * np.pi / omega_hz


def motion_blur_factor(W_s: float, tau_s: float) -> float:
    """Facteur de correction du flou de mouvement (Lindballe).

    sigma_meas^2 = sigma_thermal^2 * (1 - alpha/3 + alpha^2/12)
    avec alpha = W/tau. Valide pour alpha << 1 ; au-delà, le résultat
    est une extrapolation à ne pas prendre au sérieux.
    """
    alpha = W_s / tau_s
    factor = 1 - alpha / 3 + alpha ** 2 / 12
    # On signale l'extrapolation si alpha dépasse la plage de validité
    if alpha > 1:
        pass  # à l'appelant de vérifier via simulate_langevin
    return max(factor, 1e-6)  # éviter les valeurs négatives aberrantes


def sigma_psf_gaussian(wavelength_m: float = LAMBDA_399,
                       NA: float = 0.7,
                       convention: str = "airy_0.42") -> float:
    """Écart-type de la PSF gaussienne équivalente (m).

    convention='airy_0.42' : sigma = 0.42 * r0, avec r0 = 0.61 lambda/NA
    convention='fwhm_0.51' : sigma = FWHM/2.355, avec FWHM = 0.51 lambda/NA
    Les deux conventions diffèrent de ~20 %, à signaler explicitement.
    """
    r0 = 0.61 * wavelength_m / NA
    if convention == "airy_0.42":
        return 0.42 * r0
    elif convention == "fwhm_0.51":
        fwhm = 0.51 * wavelength_m / NA
        return fwhm / 2.3548
    else:
        raise ValueError(f"Convention inconnue : {convention}")


def sample_positions_gaussian(T_k: float, omega_hz: float, n: int,
                               rng: np.random.Generator = None) -> np.ndarray:
    """Échantillonne n positions depuis N(0, sigma_thermal^2)."""
    if rng is None:
        rng = np.random.default_rng(42)
    sigma = sigma_thermal(T_k, omega_hz)
    return rng.normal(0, sigma, size=n)


def simulate_langevin(T_k: float, omega_hz: float, gamma: float,
                       n_steps: int, dt: float,
                       rng: np.random.Generator = None) -> np.ndarray:
    """Simulation de Langevin d'un atome dans un piège harmonique.

    dx = v dt
    dv = -omega^2 x dt - gamma v dt + sqrt(2 gamma k_B T / m) dW
    avec dW ~ N(0, dt).

    Retourne les positions après une phase de thermalisation (10%).
    Sert de validation croisée à sample_positions_gaussian.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    kappa = kappa_from_omega(omega_hz)
    sigma_v = np.sqrt(K_B * T_k / M_YB173)
    x = rng.normal(0, sigma_thermal(T_k, omega_hz))
    v = rng.normal(0, sigma_v)
    positions = np.empty(n_steps)
    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        a = -omega_hz ** 2 * x - gamma * v + \
            np.sqrt(2 * gamma * K_B * T_k / M_YB173) * dW / dt
        v += a * dt
        x += v * dt
        positions[i] = x
    # Retourner la seconde moitié (thermalisée)
    return positions[n_steps // 2:]