"""Formation de l'image de fluorescence d'un atome dans une pince optique."""

import numpy as np
from math import factorial
from scipy.special import j1
from scipy.ndimage import zoom as ndzoom


def airy_psf(grid_size: int, pixel_size_m: float,
             wavelength_m: float, NA: float) -> np.ndarray:
    """PSF d'Airy normalisée : sum(psf) * pixel_size_m^2 = 1."""
    half = grid_size // 2
    coords = (np.arange(grid_size) - half) * pixel_size_m
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(xx ** 2 + yy ** 2)
    x = 2 * np.pi * NA * r / wavelength_m
    with np.errstate(divide="ignore", invalid="ignore"):
        psf = (2 * j1(x) / x) ** 2
    psf[r == 0] = 1.0
    psf /= psf.sum() * pixel_size_m ** 2
    return psf


def gaussian_psf(grid_size: int, pixel_size_m: float,
                 sigma_m: float) -> np.ndarray:
    """PSF gaussienne normalisée : sum(psf) * pixel_size_m^2 = 1."""
    half = grid_size // 2
    coords = (np.arange(grid_size) - half) * pixel_size_m
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    r2 = xx ** 2 + yy ** 2
    psf = np.exp(-r2 / (2 * sigma_m ** 2))
    psf /= psf.sum() * pixel_size_m ** 2
    return psf


def zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Polynôme de Zernike Z_n^m(rho, theta), convention de Noll."""
    m_abs = abs(m)
    if m == 0:
        norm = np.sqrt(n + 1)
        angular = np.ones_like(theta)
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        angular = np.cos(m_abs * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        angular = np.sin(m_abs * theta)
    radial = np.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        c = ((-1) ** k * factorial(n - k) /
             (factorial(k) *
              factorial((n + m_abs) // 2 - k) *
              factorial((n - m_abs) // 2 - k)))
        radial += c * rho ** (n - 2 * k)
    return norm * radial * angular


def aberrated_psf(grid_size: int, pixel_size_m: float,
                  wavelength_m: float, NA: float,
                  zernike_coeffs: dict,
                  max_oversampling: int = 8) -> np.ndarray:
    """PSF avec aberrations de Zernike, sur la même grille que airy_psf.

    Zero-padding de la pupille pour obtenir un pixel PSF natif plus fin
    que lambda/(2*NA), puis interpolation sur la grille image.
    """
    pixel_psf_natif = wavelength_m / (2 * NA)
    oversampling = max(1, min(max_oversampling,
                              int(np.ceil(pixel_psf_natif / pixel_size_m))))
    N_pupil = grid_size * oversampling

    half = N_pupil // 2
    coords = (np.arange(N_pupil) - half) / (grid_size / 2)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    pupil = (rho <= 1).astype(float)

    phase = np.zeros_like(rho)
    for (n, m), coeff in zernike_coeffs.items():
        phase += 2 * np.pi * coeff * zernike(n, m, rho, theta) * pupil

    field = pupil * np.exp(1j * phase)
    psf_full = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2

    pixel_psf_eff = pixel_psf_natif / oversampling
    region_size = int(round(grid_size * pixel_size_m / pixel_psf_eff))
    region_size = max(1, min(region_size, N_pupil))

    if region_size >= N_pupil:
        psf_region = psf_full
    else:
        start = (N_pupil - region_size) // 2
        psf_region = psf_full[start:start + region_size,
                              start:start + region_size]

    if psf_region.shape[0] != grid_size:
        psf = ndzoom(psf_region, grid_size / psf_region.shape[0], order=1)
    else:
        psf = psf_region.copy()

    psf /= psf.sum() * pixel_size_m ** 2
    return psf


def shift_image_fft(image: np.ndarray, shift_y: float,
                    shift_x: float) -> np.ndarray:
    """Décale une image par le théorème du décalage (sous-pixel exact)."""
    H, W = image.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (fy * shift_y + fx * shift_x))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * phase))


def render_intensity(positions_m: np.ndarray,
                     psf: np.ndarray,
                     pixel_size_m: float,
                     photons_per_atom: float,
                     background_per_pixel: float = 0.0) -> np.ndarray:
    """Rend une image d'intensité en photons à partir des positions."""
    psf_norm = psf / psf.sum()
    H, W = psf_norm.shape
    image = np.zeros((H, W), dtype=np.float64)
    for pos in positions_m:
        image += shift_image_fft(psf_norm,
                                 pos[1] / pixel_size_m,
                                 pos[0] / pixel_size_m)
    image *= photons_per_atom
    image += background_per_pixel
    return image


def add_scmos_noise(image: np.ndarray,
                    readout_std_map: np.ndarray = None,
                    dark_current: float = 0.0,
                    gain_map: np.ndarray = None,
                    exposure_s: float = 1.0,
                    rng: np.random.Generator = None) -> np.ndarray:
    """Bruit Poisson + Gaussien dépendant du pixel (sCMOS)."""
    if rng is None:
        rng = np.random.default_rng(42)
    H, W = image.shape
    if readout_std_map is None:
        readout_std_map = 2.0 * np.ones((H, W))
    if gain_map is None:
        gain_map = 1.0 * np.ones((H, W))
    signal = image + dark_current * exposure_s
    noisy = rng.poisson(np.clip(signal, 0, None)).astype(float)
    noisy += rng.normal(0, readout_std_map)
    adu = noisy / gain_map
    return np.clip(adu, 0, None).astype(np.float64)