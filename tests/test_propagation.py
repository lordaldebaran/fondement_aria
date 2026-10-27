"""Tests de validation de la propagation 4f et des SLM."""
import numpy as np
import torch
import pytest
from onn_arquS.propagation import propagate_4f_numpy, intensity, phase_only_mask
from onn_arquS.slm import PhaseOnlySLM, AmplitudeSLM, ComplexSLM


GRID = 64


def test_4f_identity_mask():
    """Avec un masque identité, la sortie doit être égale à l'entrée."""
    field = np.random.randn(GRID, GRID) + 1j * np.random.randn(GRID, GRID)
    mask = np.ones((GRID, GRID), dtype=complex)
    out = propagate_4f_numpy(field, mask)
    assert np.allclose(out, field)


def test_4f_convolution_with_delta():
    """Un delta en (0,0) doit donner le noyau centré en sortie."""
    field = np.zeros((GRID, GRID), dtype=complex)
    field[0, 0] = 1.0

    half = GRID // 2
    y, x = np.mgrid[-half:half, -half:half]
    kernel = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
    kernel /= kernel.sum()
    kernel_centered = np.roll(kernel, (half, half), axis=(0, 1))
    mask = np.fft.fft2(kernel_centered)

    out = propagate_4f_numpy(field, mask)
    ecart = np.max(np.abs(out - kernel_centered))
    assert ecart < 1e-10, f"Max ecart = {ecart:.2e}"


def test_4f_convolution_with_delta_kernel():
    """Un noyau delta en (0,0) doit redonner l'identité."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((GRID, GRID)) + 1j * rng.standard_normal((GRID, GRID))
    kernel = np.zeros((GRID, GRID))
    kernel[0, 0] = 1.0
    mask = np.fft.fft2(kernel)
    out = propagate_4f_numpy(field, mask)
    assert np.allclose(out, field, atol=1e-10)


def test_4f_convolution_broadens_gaussian():
    """Convolution de deux gaussiennes : variance additive.

    Si le champ d'entrée a une variance spatiale sigma_in² et le noyau
    une variance sigma_k², la sortie a une variance sigma_in² + sigma_k².
    La variance est mesurée sur le MODULE du champ |out|, pas sur
    l'intensité |out|², car |out|² est une gaussienne de variance
    divisée par 2 (propriété du carré d'une gaussienne).
    """
    half = GRID // 2
    y, x = np.mgrid[-half:half, -half:half]

    sigma_in = 3.0
    sigma_k = 2.0
    field = np.exp(-(x**2 + y**2) / (2 * sigma_in**2)).astype(complex)

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma_k**2))
    kernel /= kernel.sum()
    kernel_centered = np.roll(kernel, (half, half), axis=(0, 1))
    mask = np.fft.fft2(kernel_centered)

    out = propagate_4f_numpy(field, mask)
    # Mesure sur le module du champ, pas sur l'intensité
    mag = np.abs(out)
    total = mag.sum()
    var_x = (mag * x**2).sum() / total
    var_y = (mag * y**2).sum() / total
    sigma_out = np.sqrt((var_x + var_y) / 2)
    sigma_expected = np.sqrt(sigma_in**2 + sigma_k**2)

    ecart_rel = abs(sigma_out - sigma_expected) / sigma_expected
    assert ecart_rel < 0.05, \
        f"sigma_out = {sigma_out:.3f}, attendu {sigma_expected:.3f}, ecart = {ecart_rel:.3f}"


def test_4f_intensity_broadens_squared():
    """Vérification croisée : l'intensité |out|² doit avoir une variance
    égale à la moitié de celle du champ, par la propriété du carré
    d'une gaussienne.

    Ce test complète le précédent : il documente que la mesure sur
    l'intensité (ce que la caméra observe) donne une largeur divisée par
    sqrt(2) par rapport à la largeur du champ.
    """
    half = GRID // 2
    y, x = np.mgrid[-half:half, -half:half]
    sigma_in, sigma_k = 3.0, 2.0
    field = np.exp(-(x**2 + y**2) / (2 * sigma_in**2)).astype(complex)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma_k**2))
    kernel /= kernel.sum()
    kernel_centered = np.roll(kernel, (half, half), axis=(0, 1))
    mask = np.fft.fft2(kernel_centered)
    out = propagate_4f_numpy(field, mask)
    I = np.abs(out)**2
    total = I.sum()
    var_x = (I * x**2).sum() / total
    var_y = (I * y**2).sum() / total
    sigma_I = np.sqrt((var_x + var_y) / 2)
    sigma_champ = np.sqrt(sigma_in**2 + sigma_k**2)
    # sigma_I devrait valoir sigma_champ / sqrt(2)
    sigma_expected = sigma_champ / np.sqrt(2)
    ecart_rel = abs(sigma_I - sigma_expected) / sigma_expected
    assert ecart_rel < 0.05, \
        f"sigma_I = {sigma_I:.3f}, attendu {sigma_expected:.3f}"


def test_phase_only_mask_has_unit_module():
    """Le masque de phase pure doit avoir |H| = 1 partout."""
    phase = np.random.uniform(-np.pi, np.pi, (GRID, GRID))
    mask = phase_only_mask(phase)
    assert np.allclose(np.abs(mask), 1.0)


def test_intensity_is_positive():
    """L'intensité doit être réelle et positive."""
    field = np.random.randn(GRID, GRID) + 1j * np.random.randn(GRID, GRID)
    I = intensity(field)
    assert np.all(I >= 0)
    assert I.dtype == np.float64


def test_phase_only_slm_pytorch_unit_module():
    """Le SLM de phase PyTorch doit produire un module unitaire."""
    slm = PhaseOnlySLM(GRID, GRID)
    H = slm()
    assert torch.allclose(torch.abs(H), torch.ones_like(torch.abs(H)))


def test_amplitude_slm_pytorch_bounded():
    """Le SLM d'amplitude doit produire des valeurs dans [0, 1]."""
    slm = AmplitudeSLM(GRID, GRID)
    with torch.no_grad():
        slm.logit_a.normal_(0, 5)
    H = slm()
    assert torch.all(H >= 0) and torch.all(H <= 1)


def test_complex_slm_gradient_flow():
    """Le SLM complexe doit être différentiable."""
    slm = ComplexSLM(GRID, GRID)
    H = slm()
    loss = torch.abs(H).sum()
    loss.backward()
    assert slm.phase.grad is not None
    assert slm.logit_a.grad is not None


def test_4f_gradient_through_pytorch():
    """Le gradient doit passer à travers la propagation 4f PyTorch."""
    torch.manual_seed(0)
    field = torch.randn(GRID, GRID, dtype=torch.complex64)
    slm = PhaseOnlySLM(GRID, GRID)
    H = slm()
    F = torch.fft.fft2(field)
    out = torch.fft.ifft2(F * H)
    loss = torch.abs(out).pow(2).sum()
    loss.backward()
    assert slm.phase.grad is not None
    assert slm.phase.grad.shape == (GRID, GRID)