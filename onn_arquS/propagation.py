"""Propagation 4f différentiable (NumPy et PyTorch).

Le montage 4f réalise une convolution : une lentille calcule la transformée
de Fourier de l'entrée, un masque H(u,v) est appliqué dans le plan de
Fourier, une seconde lentille effectue la transformée inverse. Le champ de
sortie est donc :

    U_out = IFFT[ H(u,v) * FFT[U_in] ]

En choisissant H(u,v) = FFT[kernel], on obtient U_out = U_in * kernel.

Conventions :
- Les FFT sont non normalisées (comme np.fft.fft2 et torch.fft.fft2).
- Le facteur d'échelle 1/(H*W) est appliqué par l'IFFT.
- Pour que la convolution soit exacte, le noyau doit être placé au centre
  du plan de Fourier (fftshift avant multiplication, ifftshift après).
"""

import numpy as np


def propagate_4f_numpy(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Propagation 4f en NumPy.

    field : champ complexe d'entrée (H, W).
    mask : masque dans le plan de Fourier (H, W), complexe.
    Retourne le champ de sortie (H, W), complexe.
    """
    field_f = np.fft.fft2(field)
    field_f_masked = field_f * mask
    return np.fft.ifft2(field_f_masked)


def intensity(field: np.ndarray) -> np.ndarray:
    """Intensité |U|^2."""
    return np.abs(field) ** 2


def phase_only_mask(phase: np.ndarray) -> np.ndarray:
    """Masque de phase pure : exp(i*phase), module unitaire."""
    return np.exp(1j * phase)