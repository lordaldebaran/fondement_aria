"""Architectures optique et électronique comparables.

Corrections par rapport à la v1 :
- Transformation log sur l'entrée pour comprimer la plage dynamique.
- Normalisation par frame entre chaque couche optique.
- Extraction de features par adaptive max pooling (4x4) au lieu de
  mean/std globaux, ce qui préserve l'information spatiale locale.
- Classifieur avec dropout pour prévenir l'effondrement sur une classe.
- SLM initialisé avec une phase plus grande (init_scale=1.0) pour que
  le gradient soit non nul dès l'initialisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from onn_arquS.slm import PhaseOnlySLM


class OpticalConvLayer(nn.Module):
    """Couche 4f : SLM de phase + intensité |.|^2."""
    def __init__(self, size, init_scale=1.0):
        super().__init__()
        self.slm = PhaseOnlySLM(size, size, init_scale=init_scale)

    def forward(self, x):
        F_field = torch.fft.fft2(x.to(torch.complex64))
        out = torch.fft.ifft2(F_field * self.slm())
        return torch.abs(out) ** 2


class OpticalCNN(nn.Module):
    """CNN optique à n_layers couches 4f phase-only."""
    def __init__(self, input_size=32, n_layers=2, n_classes=3, pool_size=4):
        super().__init__()
        self.layers = nn.ModuleList([
            OpticalConvLayer(input_size, init_scale=1.0)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.pool_size = pool_size
        per_layer_dim = pool_size * pool_size
        feature_dim = n_layers * per_layer_dim * 2  # mean + std temporels
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        B, T, H, W = x.shape
        x = torch.log1p(x.clamp(min=0))
        frames = x.reshape(B * T, H, W)
        all_feats = []
        for layer in self.layers:
            frames = layer(frames)
            m = frames.mean(dim=(1, 2), keepdim=True)
            s = frames.std(dim=(1, 2), keepdim=True) + 1e-6
            frames = (frames - m) / s
            pool = F.adaptive_max_pool2d(
                frames.unsqueeze(1), (self.pool_size, self.pool_size)
            ).squeeze(1)  # (B*T, pool, pool)
            pool = pool.reshape(B, T, -1)  # (B, T, pool*pool)
            all_feats.append(pool.mean(dim=1))
            all_feats.append(pool.std(dim=1))
        feats = torch.cat(all_feats, dim=1)
        return self.classifier(feats)


class ElectronicCNN(nn.Module):
    """Baseline électronique de capacité comparable."""
    def __init__(self, n_layers=2, n_channels=8, n_classes=3, pool_size=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            in_ch = 1 if i == 0 else n_channels
            convs.append(nn.Conv2d(in_ch, n_channels, 5, padding=2))
        self.convs = nn.ModuleList(convs)
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pool_size = pool_size
        per_layer_dim = n_channels * pool_size * pool_size
        feature_dim = n_layers * per_layer_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        B, T, H, W = x.shape
        x = torch.log1p(x.clamp(min=0))
        frames = x.reshape(B * T, 1, H, W)
        all_feats = []
        for conv, act in zip(self.convs, self.acts):
            frames = act(conv(frames))
            pool = F.adaptive_max_pool2d(
                frames, (self.pool_size, self.pool_size)
            )  # (B*T, C, pool, pool)
            pool = pool.reshape(B, T, -1)
            all_feats.append(pool.mean(dim=1))
            all_feats.append(pool.std(dim=1))
        feats = torch.cat(all_feats, dim=1)
        return self.classifier(feats)