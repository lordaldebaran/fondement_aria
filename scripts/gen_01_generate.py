"""Génère 01_generate_data.ipynb : génération des séquences étiquetées."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# Phase 3 — Génération des séquences étiquetées

Ce notebook génère trois classes de séquences :

- `nominal` : atome unique, aucune anomalie.
- `atom_loss` : l'atome disparaît à un instant `t_loss`.
- `aberration_drift` : les coefficients de Zernike croissent linéairement.

Chaque séquence contient `n_frames` images de fluorescence.
Les séquences sont sauvegardées dans `data/<classe>/`.

**Important** : les performances sur ces données synthétiques ne sont
pas une preuve de performance sur les données réelles d'ArQuS.
"""),
    nbf.v4.new_code_cell("""import sys, os, time
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from onn_arquS.config import (
    LAMBDA_399, TrapConfig, ImagingConfig, SimulationConfig
)
from onn_arquS.physics import sigma_thermal, sample_positions_gaussian
from onn_arquS.imaging import (
    airy_psf, aberrated_psf, render_intensity, add_scmos_noise
)

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['figure.dpi'] = 100

cfg_trap = TrapConfig()
cfg_img = ImagingConfig()
cfg_sim = SimulationConfig()

pixel_object_m = cfg_img.pixel_size_m / cfg_img.magnification
print(f'Pixel objet : {pixel_object_m*1e9:.1f} nm')
print(f'Grille : {cfg_sim.grid_size} x {cfg_sim.grid_size}')
print(f'Champ objet : {cfg_sim.grid_size * pixel_object_m*1e6:.2f} um')"""),
    nbf.v4.new_markdown_cell("""## 1. PSF de référence"""),
    nbf.v4.new_code_cell("""psf_ideal = airy_psf(cfg_sim.grid_size, pixel_object_m,
                      LAMBDA_399, cfg_img.numerical_aperture)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(psf_ideal, cmap='hot')
axes[0].set_title('PSF Airy (intensite)')
axes[1].plot(psf_ideal[cfg_sim.grid_size//2, :] / psf_ideal.max())
axes[1].set_xlabel('Pixel'); axes[1].set_ylabel('Intensite normalisee')
axes[1].set_title('Profil horizontal')
axes[2].imshow(np.log10(psf_ideal + 1e-30), cmap='hot')
axes[2].set_title('Log10(intensite)')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 2. Fonctions de génération"""),
    nbf.v4.new_code_cell("""def generate_nominal(n_frames, n_atoms=1, photons=1e4,
                     readout_std=2.0, background=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    frames = []
    for _ in range(n_frames):
        positions = np.zeros((n_atoms, 2))
        for i in range(n_atoms):
            x = sample_positions_gaussian(cfg_trap.temperature_k,
                                          cfg_trap.omega_hz, 1, rng)[0]
            y = sample_positions_gaussian(cfg_trap.temperature_k,
                                          cfg_trap.omega_hz, 1, rng)[0]
            positions[i] = [x, y]
        img = render_intensity(positions, psf_ideal, pixel_object_m,
                               photons, background)
        img = add_scmos_noise(img, readout_std_map=np.full_like(img, readout_std),
                              gain_map=np.ones_like(img), rng=rng)
        frames.append(img)
    return np.array(frames), {'class': 'nominal', 'n_frames': n_frames,
                              'n_atoms': n_atoms}


def generate_atom_loss(n_frames, t_loss, n_atoms=1, photons=1e4,
                     readout_std=2.0, background=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    frames = []
    for f in range(n_frames):
        if f >= t_loss:
            img = np.full((cfg_sim.grid_size, cfg_sim.grid_size), background)
            img = add_scmos_noise(img, readout_std_map=np.full_like(img, readout_std),
                                  gain_map=np.ones_like(img), rng=rng)
        else:
            positions = np.zeros((n_atoms, 2))
            for i in range(n_atoms):
                x = sample_positions_gaussian(cfg_trap.temperature_k,
                                              cfg_trap.omega_hz, 1, rng)[0]
                y = sample_positions_gaussian(cfg_trap.temperature_k,
                                              cfg_trap.omega_hz, 1, rng)[0]
                positions[i] = [x, y]
            img = render_intensity(positions, psf_ideal, pixel_object_m,
                                   photons, background)
            img = add_scmos_noise(img, readout_std_map=np.full_like(img, readout_std),
                                  gain_map=np.ones_like(img), rng=rng)
        frames.append(img)
    return np.array(frames), {'class': 'atom_loss', 'n_frames': n_frames,
                              't_loss': t_loss}


def generate_aberration_drift(n_frames, t_start, rate, n_atoms=1, photons=1e4,
                               readout_std=2.0, background=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    frames = []
    for f in range(n_frames):
        coeff = (f - t_start) * rate if f >= t_start else 0.0
        psf_drift = aberrated_psf(cfg_sim.grid_size, pixel_object_m,
                                   LAMBDA_399, cfg_img.numerical_aperture,
                                   zernike_coeffs={(3, 1): coeff})
        positions = np.zeros((n_atoms, 2))
        for i in range(n_atoms):
            x = sample_positions_gaussian(cfg_trap.temperature_k,
                                          cfg_trap.omega_hz, 1, rng)[0]
            y = sample_positions_gaussian(cfg_trap.temperature_k,
                                          cfg_trap.omega_hz, 1, rng)[0]
            positions[i] = [x, y]
        img = render_intensity(positions, psf_drift, pixel_object_m,
                               photons, background)
        img = add_scmos_noise(img, readout_std_map=np.full_like(img, readout_std),
                              gain_map=np.ones_like(img), rng=rng)
        frames.append(img)
    return np.array(frames), {'class': 'aberration_drift', 'n_frames': n_frames,
                              't_start': t_start, 'rate': rate}"""),
    nbf.v4.new_markdown_cell("""## 3. Visualisation : une séquence par classe"""),
    nbf.v4.new_code_cell("""n_show = 6
rng = np.random.default_rng(0)

seq_nom, meta_nom = generate_nominal(n_show, rng=rng)
seq_loss, meta_loss = generate_atom_loss(n_show, t_loss=3, rng=rng)
seq_drift, meta_drift = generate_aberration_drift(n_show, t_start=2, rate=0.1, rng=rng)

fig, axes = plt.subplots(3, n_show, figsize=(2*n_show, 6))
for j in range(n_show):
    axes[0, j].imshow(seq_nom[j], cmap='hot'); axes[0, j].set_title(f'f{j}'); axes[0, j].axis('off')
    axes[1, j].imshow(seq_loss[j], cmap='hot'); axes[1, j].set_title(f'f{j}'); axes[1, j].axis('off')
    axes[2, j].imshow(seq_drift[j], cmap='hot'); axes[2, j].set_title(f'f{j}'); axes[2, j].axis('off')
axes[0, 0].set_ylabel('nominal'); axes[1, 0].set_ylabel('atom_loss'); axes[2, 0].set_ylabel('drift')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 4. Test de séparabilité triviale"""),
    nbf.v4.new_code_cell("""n_seq_per_class = 30
n_frames = 50

t0 = time.time()
all_seqs = {'nominal': [], 'atom_loss': [], 'aberration_drift': []}
all_meta = {'nominal': [], 'atom_loss': [], 'aberration_drift': []}
rng = np.random.default_rng(1)

for _ in range(n_seq_per_class):
    seq, meta = generate_nominal(n_frames, rng=rng)
    all_seqs['nominal'].append(seq); all_meta['nominal'].append(meta)

    t_loss = rng.integers(10, 40)
    seq, meta = generate_atom_loss(n_frames, t_loss=int(t_loss), rng=rng)
    all_seqs['atom_loss'].append(seq); all_meta['atom_loss'].append(meta)

    t_start = rng.integers(5, 25)
    rate = rng.uniform(0.02, 0.15)
    seq, meta = generate_aberration_drift(n_frames, t_start=int(t_start),
                                            rate=float(rate), rng=rng)
    all_seqs['aberration_drift'].append(seq); all_meta['aberration_drift'].append(meta)

print(f'Generation : {time.time()-t0:.1f} s')

features = {}
for cls in all_seqs:
    feats = np.array([seq.sum(axis=(1, 2)).mean() for seq in all_seqs[cls]])
    features[cls] = feats

fig, ax = plt.subplots()
for cls, feats in features.items():
    ax.hist(feats, bins=15, alpha=0.5, label=cls)
ax.set_xlabel('Moyenne temporelle de l intensite integree')
ax.set_ylabel('Nombre de sequences')
ax.legend(); plt.tight_layout(); plt.show()

nom = features['nominal']; loss = features['atom_loss']
threshold = (nom.mean() + loss.mean()) / 2
acc_trivial = (np.mean(nom > threshold) + np.mean(loss < threshold)) / 2
print(f'Accuracy triviale (nominal vs atom_loss) : {acc_trivial:.2%}')"""),
    nbf.v4.new_markdown_cell("""## 5. Sauvegarde"""),
    nbf.v4.new_code_cell("""data_root = Path('../data')
for cls in all_seqs:
    cls_dir = data_root / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i, (seq, meta) in enumerate(zip(all_seqs[cls], all_meta[cls])):
        np.savez_compressed(cls_dir / f'seq_{i:03d}.npz',
                           frames=seq, meta=meta)
    print(f'{cls} : {len(all_seqs[cls])} sequences sauvegardees dans {cls_dir}')

test_file = data_root / 'nominal' / 'seq_000.npz'
loaded = np.load(test_file, allow_pickle=True)
print(f'Rechargement OK : frames {loaded[\"frames\"].shape}')"""),
    nbf.v4.new_markdown_cell("""## 6. Conclusion

Voir la sortie de la cellule de séparabilité. Suivant l'accuracy
obtenue, la suite du projet s'oriente vers une évaluation sur la
classe `aberration_drift` (si `atom_loss` est triviale) ou vers une
évaluation conjointe (si les deux classes sont non triviales).
"""),
]

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python (onn_arquS)', 'language': 'python', 'name': 'onn_arquS'},
    'language_info': {'name': 'python', 'version': '3.14'},
}

# Correction : chemin relatif à la racine du projet, création du dossier
out = Path('notebooks') / '01_generate_data.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f'Notebook ecrit : {out.resolve()}')