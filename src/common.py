"""Shared helpers: paths, dataset loading/downloading, SMOTE, metrics."""
from __future__ import annotations

import io
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
MODEL_PATH = OUT / "model.joblib"
RESULTS_PATH = OUT / "results.json"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
# In the Pima data a 0 in these columns is physiologically impossible,
# i.e. it really means "not measured".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DATA_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
]
OPENML_MAP = {"preg": "Pregnancies", "plas": "Glucose", "pres": "BloodPressure",
              "skin": "SkinThickness", "insu": "Insulin", "mass": "BMI",
              "pedi": "DiabetesPedigreeFunction", "age": "Age", "class": TARGET}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: OPENML_MAP.get(str(c).strip(), str(c).strip()))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[FEATURES + [TARGET]].copy()
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].astype(str).str.contains("pos|1", case=False).astype(int)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if len(df) < 500 or df[TARGET].nunique() != 2:
        raise ValueError("File does not look like the Pima diabetes dataset.")
    return df.reset_index(drop=True)


def _read_text(text: str) -> pd.DataFrame:
    first = text.lstrip().split("\n", 1)[0].split(",")[0].strip()
    try:
        float(first)
        df = pd.read_csv(io.StringIO(text), header=None, names=FEATURES + [TARGET])
    except ValueError:
        df = pd.read_csv(io.StringIO(text))
    return _normalize(df)


def download_data(verbose: bool = True) -> Path:
    """Download the real Pima dataset to data/diabetes.csv (tries mirrors)."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    for url in DATA_URLS:
        try:
            if verbose:
                print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=25, context=ctx) as r:
                df = _read_text(r.read().decode("utf-8", "replace"))
            df.to_csv(DATA_PATH, index=False)
            return DATA_PATH
        except Exception as e:  # try the next mirror
            if verbose:
                print(f"    failed: {e}")
    try:  # last resort: OpenML (via scikit-learn)
        from sklearn.datasets import fetch_openml
        if verbose:
            print("  trying OpenML ...")
        d = fetch_openml("diabetes", version=1, as_frame=True).frame
        _normalize(d).to_csv(DATA_PATH, index=False)
        return DATA_PATH
    except Exception as e:
        if verbose:
            print(f"    failed: {e}")
    raise RuntimeError(
        "Could not download the Pima Indians Diabetes dataset (no internet?).\n"
        "Download diabetes.csv manually (Kaggle: 'Pima Indians Diabetes Database') "
        f"and save it as:\n  {DATA_PATH}")


def load_data(path: str | Path | None = None, auto_download: bool = True) -> pd.DataFrame:
    """Load the dataset; zeros that mean 'not measured' become NaN."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        if not auto_download or path != DATA_PATH:
            raise FileNotFoundError(path)
        download_data()
    df = _read_text(path.read_text())
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


# --------------------------------------------------------------------------
# SMOTE (Chawla et al. 2002) implemented with scikit-learn only
# --------------------------------------------------------------------------
def smote_resample(X, y, k_neighbors=5, random_state=42):
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        if n == target or n < 2:
            continue
        Xc = X[y == c]
        need = target - n
        k = min(k_neighbors, len(Xc) - 1)
        idx = NearestNeighbors(n_neighbors=k + 1).fit(Xc).kneighbors(
            Xc, return_distance=False)[:, 1:]
        base = rng.randint(0, len(Xc), need)
        nbr = idx[base, rng.randint(0, k, need)]
        gap = rng.rand(need, 1)
        Xs.append(Xc[base] + gap * (Xc[nbr] - Xc[base]))
        ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


class SmoteClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a classifier; optionally oversamples the *training fold only*
    with SMOTE inside fit(), so cross-validation never leaks synthetic rows."""

    def __init__(self, estimator=None, smote=False, k_neighbors=5, random_state=42):
        self.estimator = estimator
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.smote:
            X, y = smote_resample(X, y, self.k_neighbors, self.random_state)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))


def make_pipeline(estimator, smote=False, seed=42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SmoteClassifier(estimator, smote=smote, random_state=seed)),
    ])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, len(y)),
        "precision": prec,
        "recall": rec,
        "f1": div(2 * prec * rec, prec + rec),
        "roc_auc": float(roc_auc_score(y, proba)),
        "specificity": div(tn, tn + fp),
        "avg_precision": float(average_precision_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, proba


def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=to_jsonable))
