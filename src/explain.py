"""Per-prediction and global explanations (SHAP, with a built-in fallback)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FEATURES

try:  # optional dependency
    import shap  # type: ignore
    HAVE_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAVE_SHAP = False


def make_predict_fn(model):
    def f(arr):
        arr = np.atleast_2d(arr)
        return model.predict_proba(pd.DataFrame(arr, columns=FEATURES))[:, 1]
    return f


def sampled_shapley(f, X, background, n_perm=80, seed=0):
    """Monte-Carlo Shapley values (permutation sampling, interventional
    baseline).  Same quantity SHAP estimates; used if `shap` is unavailable."""
    rng = np.random.RandomState(seed)
    X = np.atleast_2d(X)
    n, d = X.shape
    vals = np.zeros((n, d))
    for i, x in enumerate(X):
        seqs, orders = [], []
        for _ in range(n_perm):
            order = rng.permutation(d)
            cur = background[rng.randint(len(background))].copy()
            seq = [cur.copy()]
            for j in order:
                cur[j] = x[j]
                seq.append(cur.copy())
            seqs.append(np.array(seq))
            orders.append(order)
        P = f(np.vstack(seqs)).reshape(n_perm, d + 1)
        diffs = np.diff(P, axis=1)
        for p in range(n_perm):
            vals[i, orders[p]] += diffs[p]
        vals[i] /= n_perm
    return vals, float(f(background).mean())


def shap_values(model, X, background):
    """Return (values[n, d], base_value, method_name)."""
    f = make_predict_fn(model)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    background = np.asarray(background, dtype=float)
    if HAVE_SHAP:
        try:
            masker = shap.maskers.Independent(background, max_samples=len(background))
            sv = shap.Explainer(f, masker)(X)
            return np.asarray(sv.values), float(np.mean(sv.base_values)), "SHAP"
        except Exception as e:  # API differences between shap versions
            print(f"  (shap failed: {e!r}; using built-in Shapley sampling)")
    vals, base = sampled_shapley(f, X, background)
    return vals, base, "Shapley sampling"
