"""Paramètres physiques et de simulation.

Chaque paramètre est annoté par son statut épistémique, conformément au
document LaTeX :
- [S] = sourcé (vérifiable dans la littérature)
- [T] = transposé (résultat d'un autre système physique, non validé ici)
- [S?]= supposé (aucune source, valeur à balayer en sensibilité)

Les plages de balayage sont indiquées pour les paramètres supposés.
"""

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Constantes physiques (sourcées)
# ---------------------------------------------------------------------------
K_B = 1.380649e-23            # J/K, constante de Boltzmann [S]
M_YB173 = 173 * 1.66053906660e-27  # kg, masse de l'ytterbium-173 [S]
LAMBDA_399 = 399e-9           # m, transition 1S0 -> 1P1 [S]
LAMBDA_556 = 556e-9           # m, transition 1S0 -> 3P1 [S]


@dataclass
class TrapConfig:
    """Configuration du piège optique."""
    omega_hz: float = 2 * 3.141592653589793 * 100e3  # rad/s, [S?] à balayer 2π·50-500 kHz
    # tau_t = 2π/omega est la période d'oscillation, ~10 µs pour 100 kHz
    damping_gamma: float = 1e3    # s^-1, amortissement (refroidissement) [S?] à balayer
    temperature_k: float = 10e-6  # K, température effective [S?] à balayer 1-100 µK


@dataclass
class ImagingConfig:
    """Configuration de l'imagerie."""
    wavelength_m: float = LAMBDA_399  # [S]
    numerical_aperture: float = 0.7   # [S?] dépend de l'objectif ArQuS
    pixel_size_m: float = 6.5e-6      # [S?] typique sCMOS
    magnification: float = 100.0      # [S?] typique objectif microscope
    exposure_s: float = 1e-6          # s, [S?] pulse d'imagerie ArQuS ~µs
    photons_per_atom: float = 1e4     # [S?] à balayer 1e3-1e5
    background_per_pixel: float = 10.0  # [S?] à balayer 0-100
    readout_std_e: float = 2.0        # [S?] à balayer 1-5
    dark_current_e_per_s: float = 0.1 # [S?] typique sCMOS


@dataclass
class SimulationConfig:
    """Configuration de la simulation numérique."""
    grid_size: int = 128              # pixels, [S?] 64-256
    field_size_m: float = 128 * 6.5e-6 / 100  # m, taille du champ objet
    n_samples: int = 1000             # nombre d'échantillons par classe
    n_frames: int = 50                # frames par séquence
    seed: int = 42


def print_parameter_table():
    """Affiche un tableau récapitulatif des paramètres et de leur statut."""
    rows = [
        ("Constante de Boltzmann", f"{K_B:.4e}", "J/K", "[S]"),
        ("Masse Yb-173", f"{M_YB173:.4e}", "kg", "[S]"),
        ("Longueur d'onde", f"{LAMBDA_399*1e9:.0f}", "nm", "[S]"),
        ("Fréquence de piège", f"{TrapConfig().omega_hz/(2*3.14159)/1e3:.0f}", "kHz", "[S?] balayer 50-500"),
        ("Température", f"{TrapConfig().temperature_k*1e6:.0f}", "µK", "[S?] balayer 1-100"),
        ("Ouverture numérique", f"{ImagingConfig().numerical_aperture:.1f}", "-", "[S?] dépend de l'objectif"),
        ("Taille pixel", f"{ImagingConfig().pixel_size_m*1e6:.1f}", "µm", "[S?] typique sCMOS"),
        ("Grandissement", f"{ImagingConfig().magnification:.0f}", "x", "[S?] typique"),
        ("Temps d'exposition", f"{ImagingConfig().exposure_s*1e6:.1f}", "µs", "[S?] pulse ArQuS"),
        ("Photons par atome", f"{ImagingConfig().photons_per_atom:.0e}", "-", "[S?] balayer 1e3-1e5"),
        ("Bruit de lecture", f"{ImagingConfig().readout_std_e:.1f}", "e-", "[S?] balayer 1-5"),
    ]
    print(f"{'Paramètre':<25} {'Valeur':>12} {'Unité':>6} {'Statut':<20}")
    print("-" * 70)
    for name, val, unit, status in rows:
        print(f"{name:<25} {val:>12} {unit:>6} {status:<20}")