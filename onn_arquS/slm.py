"""Modèles de SLM pour PyTorch.

Trois familles :
- PhaseOnlySLM : module |H|=1 par construction (SLM LCoS de phase).
- AmplitudeSLM : module dans [0, 1], phase fixe (DMD ou SLM d'amplitude).
- ComplexSLM : module et phase indépendants (deux SLM en cascade).

Tous les modules sont différentiables et compatibles avec torch.fft.
"""

import torch
import torch.nn as nn


class PhaseOnlySLM(nn.Module):
    """SLM de phase pure : H = exp(i*phi)."""

    def __init__(self, height: int, width: int, init_scale: float = 0.1):
        super().__init__()
        self.phase = nn.Parameter(init_scale * torch.randn(height, width))

    def forward(self) -> torch.Tensor:
        return torch.exp(1j * self.phase)

    def amplitude_response(self) -> torch.Tensor:
        """Module de la réponse (doit valoir 1 partout)."""
        return torch.abs(self.forward())


class AmplitudeSLM(nn.Module):
    """SLM d'amplitude : H = a, avec a dans [0, 1]."""

    def __init__(self, height: int, width: int):
        super().__init__()
        # Paramètre logit pour garantir a dans [0,1] via sigmoïde
        self.logit_a = nn.Parameter(torch.zeros(height, width))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_a)


class ComplexSLM(nn.Module):
    """SLM complexe : H = a * exp(i*phi), avec a dans [0,1]."""

    def __init__(self, height: int, width: int, init_scale: float = 0.1):
        super().__init__()
        self.logit_a = nn.Parameter(torch.zeros(height, width))
        self.phase = nn.Parameter(init_scale * torch.randn(height, width))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_a) * torch.exp(1j * self.phase)