"""Métriques de classification."""
import numpy as np


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def delta_acc(acc_ideal, acc_pert):
    if acc_ideal < 1e-3:
        return acc_pert - acc_ideal
    return (acc_ideal - acc_pert) / acc_ideal


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)