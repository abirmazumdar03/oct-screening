import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, class_names, urgent_classes=()):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    urgent = [class_names.index(c) for c in urgent_classes if c in class_names]
    if urgent:
        t, pr = np.isin(y_true, urgent), np.isin(y_pred, urgent)
        out["urgent_sensitivity"] = float((t & pr).sum() / max(int(t.sum()), 1))     # of truly urgent, % flagged urgent
        out["urgent_specificity"] = float((~t & ~pr).sum() / max(int((~t).sum()), 1))
    return out
