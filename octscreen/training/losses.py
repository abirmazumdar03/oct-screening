# octscreen/training/losses.py
import numpy as np
import torch
import torch.nn.functional as F


def class_weights(labels, num_classes, mode):
    """none | inverse | inverse_sqrt: counters class imbalance (CNV has ~4x more scans than DRUSEN)."""
    if mode in (None, "none"):
        return None
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(np.float64), 1.0)
    w = counts.sum() / counts
    if mode == "inverse_sqrt":
        w = np.sqrt(w)
    elif mode != "inverse":
        raise ValueError(f"Unknown class_weighting: {mode}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def kd_loss(student_logits, teacher_logits, targets, ce_fn, temperature, alpha):
    """(1-α)·CE(student, labels) + α·T²·KL(teacher_T || student_T)   (Hinton et al., 2015)."""
    T = float(temperature)
    ce = ce_fn(student_logits, targets)
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                  F.softmax(teacher_logits / T, dim=1),
                  reduction="batchmean") * (T * T)
    return (1 - alpha) * ce + alpha * kl, ce.detach(), kl.detach()
