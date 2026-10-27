"""Tests de validation du module imaging.py."""
import numpy as np
import pytest
from onn_arquS.imaging import (
    airy_psf, gaussian_psf, zernike, aberrated_psf,
    render_intensity, add_scmos_noise
)
from onn_arquS.config import LAMBDA_399
from onn_arquS.physics import sigma_psf_gaussian


GRID = 64
PIXEL = 6.5e-6 / 100  # pixel objet : 65 nm


def test_airy_normalized():
    """L'intégrale 2D de la PSF d'Airy doit valoir 1."""
    psf = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.7)
    integral = psf.sum() * PIXEL ** 2
    assert abs(integral - 1.0) < 1e-6, f"integrale = {integral}"


def test_airy_width_matches_theory():
    """La largeur de la PSF d'Airy doit être cohérente avec lambda/NA."""
    psf = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.7)
    # Calcul de la largeur à mi-hauteur par interpolation
    center = (GRID // 2, GRID // 2)
    profile = psf[center[0], :]
    half_max = profile.max() / 2
    # Indices où le profil croise la mi-hauteur
    above = np.where(profile >= half_max)[0]
    fwhm_pixels = above[-1] - above[0]
    fwhm_m = fwhm_pixels * PIXEL
    # Théorie : FWHM ≈ 0.51 lambda / NA = 0.51 * 399e-9 / 0.7 ≈ 291 nm
    fwhm_theory = 0.51 * LAMBDA_399 / 0.7
    # Tolérance de 50% car l'échantillonnage est grossier
    assert 0.5 < fwhm_m / fwhm_theory < 2.0, \
        f"FWHM = {fwhm_m*1e9:.0f} nm, théorie = {fwhm_theory*1e9:.0f} nm"


def test_gaussian_matches_airy_for_low_NA():
    """Pour NA faible, la gaussienne doit approximer l'Airy."""
    sigma = sigma_psf_gaussian(LAMBDA_399, NA=0.3, convention="airy_0.42")
    psf_airy = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.3)
    psf_gauss = gaussian_psf(GRID, PIXEL, sigma)
    # Corrélation élevée
    corr = np.corrcoef(psf_airy.ravel(), psf_gauss.ravel())[0, 1]
    assert corr > 0.9, f"Corrélation = {corr:.3f}"


def test_zernike_orthogonality_spot_check():
    """Zernike(0,0) doit valoir 1 partout dans la pupille."""
    rho = np.linspace(0, 1, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, T = np.meshgrid(rho, theta, indexing="ij")
    z = zernike(0, 0, R, T)
    assert np.allclose(z, 1.0)


def test_aberration_broadens_psf():
    """L'aberration doit élargir la PSF (réduction du pic central)."""
    psf_ideal = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.7)
    psf_coma = aberrated_psf(GRID, PIXEL, LAMBDA_399, NA=0.7,
                             zernike_coeffs={(3, 1): 0.2})
    # Le pic central de la PSF aberrée est plus bas
    center = (GRID // 2, GRID // 2)
    assert psf_coma[center] < psf_ideal[center] * 0.9, \
        f"pic coma = {psf_coma[center]:.4e}, pic idéal = {psf_ideal[center]:.4e}"


def test_render_single_atom_centered():
    """Un seul atome centré doit donner une image centrée sur la PSF."""
    psf = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.7)
    positions = np.array([[0.0, 0.0]])  # centre
    image = render_intensity(positions, psf, PIXEL,
                              photons_per_atom=1e4,
                              background_per_pixel=0.0)
    # Maximum au centre
    center = (GRID // 2, GRID // 2)
    assert np.argmax(image) == center[0] * GRID + center[1], \
        "Le maximum n'est pas au centre"


def test_render_preserves_total_flux():
    """L'intensité totale doit valoir photons_per_atom."""
    psf = airy_psf(GRID, PIXEL, LAMBDA_399, NA=0.7)
    positions = np.array([[0.0, 0.0]])
    photons = 1e4
    image = render_intensity(positions, psf, PIXEL,
                              photons_per_atom=photons,
                              background_per_pixel=0.0)
    # La PSF discrète n'est pas exactement normalisée à cause du roll,
    # mais l'écart doit être < 1%
    total = image.sum()
    assert abs(total - photons) / photons < 0.01, \
        f"total = {total:.0f}, attendu {photons:.0f}"


def test_scmos_noise_statistics():
    """Le bruit sCMOS doit préserver la moyenne et augmenter la variance."""
    image = np.full((32, 32), 100.0)  # 100 photons partout
    noisy = add_scmos_noise(image, readout_std_map=np.full((32,32), 2.0),
                            gain_map=np.ones((32,32)),
                            exposure_s=1.0, rng=np.random.default_rng(42))
    # Moyenne à ±5%
    assert abs(noisy.mean() - 100.0) / 100.0 < 0.05
    # Variance > variance Poisson seule (100)
    assert noisy.var() > 100


def test_aberration_preserves_normalization():
    """La PSF aberrée doit rester normalisée à 1."""
    psf = aberrated_psf(GRID, PIXEL, LAMBDA_399, NA=0.7,
                        zernike_coeffs={(3, 1): 0.1, (2, 2): 0.1})
    integral = psf.sum() * PIXEL ** 2
    assert abs(integral - 1.0) < 1e-6, f"integrale = {integral}"