"""Boucle d'entraînement avec early stopping et weight decay."""
import torch
import torch.nn as nn
import time


def train_model(model, train_loader, val_loader, n_epochs=50, lr=1e-3,
                patience=5, weight_decay=1e-3, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        tl = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_loader), 1)
        model.eval()
        vl = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                vl += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        vl /= max(len(val_loader), 1)
        va = correct / max(total, 1)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        if verbose:
            print(f'  epoch {epoch+1:2d}/{n_epochs} | train {tl:.4f} | '
                  f'val {vl:.4f} | acc {va:.3f} | {time.time()-t0:.1f}s')
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f'  Early stopping a l\'epoque {epoch+1}')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history['train_time'] = time.time() - t0
    return history