"""Génère notebooks/diag_03.ipynb : entraînement OpticalCNN vs ElectronicCNN."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# Diagnostic — Entraînement OpticalCNN vs ElectronicCNN

Ce notebook entraîne les deux architectures sur les séquences générées
dans `01_generate_data.ipynb` et compare :

- la performance brute (accuracy) ;
- le nombre de paramètres (comparaison équitable) ;
- le temps d'entraînement et d'inférence ;
- la robustesse au bruit gaussien ajouté aux images de test.

Le notebook imprime les résultats sans graphique, pour faciliter
l'extraction des chiffres clés vers un rapport.
"""),
    nbf.v4.new_code_cell("""import sys, os, time
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from onn_arquS.network import OpticalCNN, ElectronicCNN
from onn_arquS.training import train_model
from onn_arquS.metrics import accuracy, confusion_matrix, delta_acc, count_parameters

torch.manual_seed(42)
np.random.seed(42)

# Parametres de simulation
INPUT_SIZE = 32
N_FRAMES = 10
N_EPOCHS = 15
BATCH_SIZE = 4
N_CLASSES = 3
CLASSES = ['nominal', 'atom_loss', 'aberration_drift']

print("=" * 70)
print("ENTRAINEMENT — OpticalCNN vs ElectronicCNN")
print("=" * 70)
print(f"Parametres : input={INPUT_SIZE}, frames={N_FRAMES}, epochs={N_EPOCHS}, batch={BATCH_SIZE}")"""),
    nbf.v4.new_markdown_cell("""## 1. Chargement et prétraitement des données

Les séquences font `(50, 128, 128)`. On les sous-échantillonne à
`(10, 32, 32)` pour réduire le coût de calcul sur CPU :
- temporellement : 10 frames régulièrement espacées ;
- spatialement : moyenne de blocs 4×4.

La normalisation par statistiques globales évite que le classifieur
apprenne à partir de l'échelle absolue (qui varie avec les paramètres
de génération)."""),
    nbf.v4.new_code_cell("""print("\\n[1] Chargement des donnees")
t0 = time.time()

data_root = Path('../data')
X_list, y_list = [], []
for ci, cls in enumerate(CLASSES):
    for f in sorted((data_root / cls).glob('seq_*.npz')):
        frames = np.load(f, allow_pickle=True)['frames']
        idx_t = np.linspace(0, frames.shape[0] - 1, N_FRAMES).astype(int)
        frames = frames[idx_t].astype(np.float32)
        H, W = frames.shape[1], frames.shape[2]
        bh, bw = H // INPUT_SIZE, W // INPUT_SIZE
        frames = frames.reshape(N_FRAMES, INPUT_SIZE, bh, INPUT_SIZE, bw).mean(axis=(2, 4))
        X_list.append(frames)
        y_list.append(ci)

X = torch.tensor(np.stack(X_list), dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.long)
print(f"    X : {tuple(X.shape)}, y : {tuple(y.shape)}")
print(f"    Temps de chargement : {time.time()-t0:.1f} s")

X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std
print(f"    Normalisation : mean={X_mean:.2f}, std={X_std:.2f}")"""),
    nbf.v4.new_markdown_cell("""## 2. Split stratifié 70/15/15"""),
    nbf.v4.new_code_cell("""print("\\n[2] Split stratifie")
idx_train, idx_val, idx_test = [], [], []
rng = np.random.default_rng(42)
for ci in range(N_CLASSES):
    idx_c = np.where(y.numpy() == ci)[0]
    rng.shuffle(idx_c)
    n = len(idx_c)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    idx_train.extend(idx_c[:n_tr])
    idx_val.extend(idx_c[n_tr:n_tr + n_va])
    idx_test.extend(idx_c[n_tr + n_va:])

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]
print(f"    Train : {len(idx_train)} | Val : {len(idx_val)} | Test : {len(idx_test)}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)"""),
    nbf.v4.new_markdown_cell("""## 3. Architecture des modèles"""),
    nbf.v4.new_code_cell("""print("\\n[3] Architecture")
opt_model = OpticalCNN(input_size=INPUT_SIZE, n_layers=2, n_classes=N_CLASSES)
elec_model = ElectronicCNN(n_layers=2, n_channels=8, n_classes=N_CLASSES)
print(f"    OpticalCNN    : {count_parameters(opt_model):,} parametres")
print(f"    ElectronicCNN : {count_parameters(elec_model):,} parametres")"""),
    nbf.v4.new_markdown_cell("""## 4. Entraînement OpticalCNN"""),
    nbf.v4.new_code_cell("""print("\\n[4] Entrainement OpticalCNN")
hist_opt = train_model(opt_model, train_loader, val_loader,
                       n_epochs=N_EPOCHS, lr=1e-3, patience=3, verbose=True)
print(f"    Temps total : {hist_opt['train_time']:.1f} s")"""),
    nbf.v4.new_markdown_cell("""## 5. Entraînement ElectronicCNN"""),
    nbf.v4.new_code_cell("""print("\\n[5] Entrainement ElectronicCNN")
hist_elec = train_model(elec_model, train_loader, val_loader,
                        n_epochs=N_EPOCHS, lr=1e-3, patience=3, verbose=True)
print(f"    Temps total : {hist_elec['train_time']:.1f} s")"""),
    nbf.v4.new_markdown_cell("""## 6. Évaluation sur le test"""),
    nbf.v4.new_code_cell("""def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            preds.extend(out.argmax(1).numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels), time.time() - t0

print("\\n" + "=" * 70)
print("EVALUATION SUR TEST")
print("=" * 70)

pred_opt, lab, t_inf_opt = evaluate(opt_model, test_loader)
pred_elec, _, t_inf_elec = evaluate(elec_model, test_loader)

acc_opt = accuracy(lab, pred_opt)
acc_elec = accuracy(lab, pred_elec)
print(f"Accuracy OpticalCNN    : {acc_opt:.3f}")
print(f"Accuracy ElectronicCNN : {acc_elec:.3f}")
print(f"Temps d'inference (test) : Optical {t_inf_opt*1000:.0f} ms | Elec {t_inf_elec*1000:.0f} ms")

print("\\nMatrice de confusion OpticalCNN :")
print(f"  {confusion_matrix(lab, pred_opt, N_CLASSES)}")
print("Matrice de confusion ElectronicCNN :")
print(f"  {confusion_matrix(lab, pred_elec, N_CLASSES)}")"""),
    nbf.v4.new_markdown_cell("""## 7. Robustesse au bruit

On ajoute un bruit gaussien d'écart-type variable aux images de test
et on mesure la dégradation de l'accuracy pour chaque modèle."""),
    nbf.v4.new_code_cell("""print("\\n" + "=" * 70)
print("ROBUSTESSE AU BRUIT")
print("=" * 70)
print(f"{'sigma (ADU)':>12} | {'acc Optical':>12} | {'acc Elec':>10} | "
      f"{'delta_opt':>10} | {'delta_elec':>11}")
print("-" * 70)

for sigma in [0, 2, 5, 10, 20]:
    X_noisy = X_test + sigma * torch.randn_like(X_test)
    loader_noisy = DataLoader(TensorDataset(X_noisy, y_test), batch_size=BATCH_SIZE)
    pred_opt_n, _, _ = evaluate(opt_model, loader_noisy)
    pred_elec_n, _, _ = evaluate(elec_model, loader_noisy)
    acc_opt_n = accuracy(y_test.numpy(), pred_opt_n)
    acc_elec_n = accuracy(y_test.numpy(), pred_elec_n)
    d_opt = delta_acc(acc_opt, acc_opt_n)
    d_elec = delta_acc(acc_elec, acc_elec_n)
    print(f"{sigma:>12.1f} | {acc_opt_n:>12.3f} | {acc_elec_n:>10.3f} | "
          f"{d_opt:>10.3f} | {d_elec:>11.3f}")"""),
    nbf.v4.new_markdown_cell("""## 8. Verdict et interprétation"""),
    nbf.v4.new_code_cell("""print("\\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

gap = acc_elec - acc_opt
if abs(gap) < 0.05:
    print("Parite en accuracy : les deux architectures sont equivalentes.")
    print("Le reseau photonique justifie son interet sur latence/energie.")
elif gap > 0:
    print(f"Baseline electronique superieure de {gap*100:.1f} points.")
    print("Le reseau photonique a 2 couches ne suffit pas.")
    print("Piste : tester 3 couches phase-only ou 1 couche complexe.")
else:
    print(f"OpticalCNN superieur de {-gap*100:.1f} points.")
    print("Verifier l'absence d'artefact (tache triviale).")

print(f"\\nTache triviale ? acc_opt={acc_opt:.3f}, acc_elec={acc_elec:.3f}")
if min(acc_opt, acc_elec) > 0.95:
    print("-> OUI. La robustesse au bruit est le vrai discriminant.")
    print("   Pour rendre la tache non triviale, reduire photons_per_atom")
    print("   ou augmenter le niveau de bruit dans la generation.")"""),
    nbf.v4.new_markdown_cell("""## 9. Conclusion

### Ce que ce notebook a mesuré
- La performance des deux architectures à paramètres comparables.
- Le coût en temps d'entraînement et d'inférence.
- La robustesse au bruit gaussien additif.

### Ce que ce notebook ne prouve pas
- Que la tâche reflète une difficulté expérimentale réelle.
- Que la robustesse au bruit gaussien est représentative du bruit
  de photons ou des aberrations.
- Que l'entraînement sur 90 séquences est suffisant (overfitting
  possible avec seulement 63 séquences d'entraînement).

### Prochaine étape
Selon le verdict :
- Parité : construire `04_evaluate_robustness.ipynb` avec un sweep
  multi-paramètres (photons, aberrations, désalignement) et un front
  de Pareto.
- Baseline supérieure : tester 3 couches ou une couche complexe.
- Tâche triviale : régénérer les données avec moins de photons par
  atome pour créer une difficulté réelle.
"""),
]

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python (onn_arquS)', 'language': 'python', 'name': 'onn_arquS'},
    'language_info': {'name': 'python', 'version': '3.14'},
}

out = Path('notebooks') / 'diag_03.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f'Notebook ecrit : {out.resolve()}')