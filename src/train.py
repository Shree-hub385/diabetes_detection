"""Training / evaluation pipeline for the Diabetes Detection capstone.

  1. Re-run the base paper's 5 classifiers (identical stratified 80:20 split)
  2. Imbalance handling: class weighting and SMOTE (ablation + searched)
  3. Hyper-parameter tuning of Random Forest and XGBoost (randomized search)
  4. Stacked ensemble (RF + XGBoost -> Logistic Regression meta-learner)
  5. Metrics: accuracy, precision, recall, F1, ROC-AUC, specificity, confusion matrix
  6. Plots: confusion matrices, ROC, PR, feature importance / SHAP, bar chart vs paper
  7. Save the best model + results for the dashboard

Usage:  python -m src.train [--quick] [--data path/to/diabetes.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.common import (FEATURES, MODEL_PATH, OUT, PLOTS, RESULTS_PATH, TARGET,
                        ZERO_AS_MISSING, evaluate, load_data, make_pipeline, save_json)
from src.explain import HAVE_SHAP, make_predict_fn, shap_values

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    XGBClassifier = None
    HAVE_XGB = False

SEED = 42
PAPER_ACC = {"Logistic Regression": 80.4, "Random Forest": 80.4,
             "Support Vector Machine": 79.1, "XGBoost": 77.8, "Decision Tree": 75.8}
XGB_NAME = "XGBoost" if HAVE_XGB else "XGBoost (HistGradientBoosting fallback)"
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed",
          "#0891b2", "#db2777", "#4b5563", "#ea580c"]


def log(msg):
    print(msg, flush=True)


def make_xgb(**kw):
    if HAVE_XGB:
        return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1, **kw)
    return HistGradientBoostingClassifier(random_state=SEED)


def baseline_models():
    """The base paper's five classifiers at (essentially) default settings."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=1),
        "Support Vector Machine": SVC(probability=True, random_state=SEED),
        XGB_NAME: make_xgb(),
    }


def tune(name, estimator, space, X, y, n_iter, cv):
    log(f"  tuning {name} ({n_iter} candidates x {cv.get_n_splits()} folds) ...")
    t = time.time()
    search = RandomizedSearchCV(make_pipeline(estimator), space, n_iter=n_iter,
                                scoring="f1", cv=cv, random_state=SEED,
                                n_jobs=-1, refit=True)
    search.fit(X, y)
    log(f"    best CV F1 = {search.best_score_:.3f}  ({time.time() - t:.0f}s)")
    params = {k.replace("clf__estimator__", "").replace("clf__", ""): v
              for k, v in search.best_params_.items()}
    return search.best_estimator_, params, float(search.best_score_)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 130})


def plot_confusions(results, path):
    names = list(results)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.7 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, n in zip(np.ravel(axes), names):
        r = results[n]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.axis("on"); ax.grid(False)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "#111")
        ax.set_xticks([0, 1], ["Non-diabetic", "Diabetic"])
        ax.set_yticks([0, 1], ["Non-diabetic", "Diabetic"], rotation=90, va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{n}\nrecall {r['recall']:.2f} · spec. {r['specificity']:.2f}", fontsize=9.5)
    fig.suptitle("Confusion matrices (held-out 20% test set)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_curves(kind, base, improved, y, probas, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, group, title in zip(axes, (base, improved),
                                ("Base-paper models (re-run)", "Capstone models (improved)")):
        for k, n in enumerate(group):
            if kind == "roc":
                x_, y_, _ = roc_curve(y, probas[n])
                lab = f"{n} (AUC {base.get(n, improved.get(n))['roc_auc']:.3f})"
            else:
                y_, x_, _ = precision_recall_curve(y, probas[n])
                lab = f"{n} (AP {base.get(n, improved.get(n))['avg_precision']:.3f})"
            ax.plot(x_, y_, lw=2, color=COLORS[k % len(COLORS)], label=lab)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "--", color="#9ca3af")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        else:
            ax.axhline(y.mean(), ls="--", color="#9ca3af")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(title); ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=8)
    fig.suptitle("ROC curves" if kind == "roc" else "Precision-Recall curves", fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_bars(base, improved, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = list(PAPER_ACC)
    ours = [base[n if n in base else XGB_NAME]["accuracy"] * 100 for n in names]
    x = np.arange(len(names)); w = 0.38
    b1 = ax.bar(x - w / 2, [PAPER_ACC[n] for n in names], w, label="Base paper", color="#9ca3af")
    b2 = ax.bar(x + w / 2, ours, w, label="Our re-run (same 80:20 protocol)", color="#2563eb")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .3, f"{b.get_height():.1f}",
                ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylim(50, 92); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: base paper vs our re-run"); ax.legend(loc="upper right", fontsize=8, ncol=1)

    ax = axes[1]
    allr = {**{f"{k}\n(baseline)": v for k, v in base.items()},
            **{f"{k}\n(improved)": v for k, v in improved.items()}}
    names = list(allr)
    y = np.arange(len(names))
    ax.barh(y - .2, [allr[n]["f1"] for n in names], .4, label="F1", color="#16a34a")
    ax.barh(y + .2, [allr[n]["recall"] for n in names], .4, label="Recall", color="#f59e0b")
    ax.set_yticks(y, [n.replace("\n", " ") for n in names], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_title("F1 and Recall, all models"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_importance(rf_pipe, perm, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    imp = rf_pipe.named_steps["clf"].estimator_.feature_importances_
    o = np.argsort(imp)
    axes[0].barh(np.array(FEATURES)[o], imp[o], color="#2563eb")
    axes[0].set_title("Tuned Random Forest - impurity importance")
    m, s = perm
    o = np.argsort(m)
    axes[1].barh(np.array(FEATURES)[o], m[o], xerr=s[o], color="#dc2626")
    axes[1].set_title("Final model - permutation importance (drop in ROC-AUC)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_shap_summary(values, Xvals, method, path):
    """Beeswarm-style summary: x = SHAP value, colour = feature value."""
    order = np.argsort(np.abs(values).mean(0))
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for rank, j in enumerate(order):
        col = Xvals[:, j]
        rngc = np.ptp(col) or 1
        c = (col - col.min()) / rngc
        ax.scatter(values[:, j], rank + rng.uniform(-.25, .25, len(col)), c=c,
                   cmap="coolwarm", s=16, alpha=.85, edgecolors="none")
    ax.set_yticks(range(len(order)), np.array(FEATURES)[order])
    ax.axvline(0, color="#6b7280", lw=1)
    ax.set_xlabel("Impact on predicted diabetes probability")
    ax.set_title(f"{method} summary (test patients)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller search (fast demo)")
    ap.add_argument("--data", default=None, help="path to diabetes.csv")
    args = ap.parse_args()
    n_iter = 8 if args.quick else 40
    OUT.mkdir(exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    style(); t0 = time.time()

    log("[1/7] Loading data")
    df = load_data(args.data)
    X, y = df[FEATURES], df[TARGET]
    log(f"  {len(df)} patients, {y.mean():.1%} diabetic")
    log(f"  XGBoost: {'yes' if HAVE_XGB else 'NOT installed -> HistGradientBoosting fallback'};"
        f"  SHAP: {'yes' if HAVE_SHAP else 'NOT installed -> built-in Shapley sampling'}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    log("[2/7] Base-paper benchmark (5 classifiers, default settings, stratified 80:20)")
    base, probas, base_models = {}, {}, {}
    for name, est in baseline_models().items():
        m = make_pipeline(est).fit(Xtr, ytr)
        base[name], probas[name] = evaluate(m, Xte, yte)
        base_models[name] = m
        log(f"  {name:<42} acc {base[name]['accuracy']:.3f}  rec {base[name]['recall']:.3f}  auc {base[name]['roc_auc']:.3f}")

    log("[3/7] Imbalance-handling ablation (Random Forest, default hyper-parameters)")
    ablation = {}
    for label, est, sm in [
        ("No correction", RandomForestClassifier(random_state=SEED, n_jobs=1), False),
        ("Class weighting", RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight="balanced"), False),
        ("SMOTE", RandomForestClassifier(random_state=SEED, n_jobs=1), True),
    ]:
        ablation[label], _ = evaluate(make_pipeline(est, smote=sm).fit(Xtr, ytr), Xte, yte)
        a = ablation[label]
        log(f"  {label:<16} acc {a['accuracy']:.3f}  rec {a['recall']:.3f}  f1 {a['f1']:.3f}  spec {a['specificity']:.3f}")

    log("[4/7] Hyper-parameter tuning (imbalance strategy is part of the search)")
    rf_space = {
        "clf__estimator__n_estimators": [200, 300, 500],
        "clf__estimator__max_depth": [None, 4, 6, 8, 12],
        "clf__estimator__min_samples_leaf": [1, 2, 4, 8],
        "clf__estimator__max_features": ["sqrt", "log2", 0.5],
        "clf__estimator__class_weight": [None, "balanced"],
        "clf__smote": [True, False],
    }
    if HAVE_XGB:
        xgb_space = {
            "clf__estimator__n_estimators": [100, 200, 300, 500],
            "clf__estimator__max_depth": [2, 3, 4, 5, 6],
            "clf__estimator__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__estimator__subsample": [0.6, 0.8, 1.0],
            "clf__estimator__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__estimator__min_child_weight": [1, 3, 5],
            "clf__estimator__reg_lambda": [1, 5, 10],
            "clf__estimator__scale_pos_weight": [1.0, spw],
            "clf__smote": [True, False],
        }
    else:
        xgb_space = {
            "clf__estimator__max_iter": [100, 200, 300],
            "clf__estimator__max_depth": [2, 3, 4, None],
            "clf__estimator__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "clf__estimator__min_samples_leaf": [10, 20, 40],
            "clf__estimator__l2_regularization": [0.0, 1.0, 5.0],
            "clf__estimator__class_weight": [None, "balanced"],
            "clf__smote": [True, False],
        }
    rf_best, rf_params, rf_cv = tune("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=1),
                                     rf_space, Xtr, ytr, n_iter, cv)
    xgb_best, xgb_params, xgb_cv = tune(XGB_NAME, make_xgb(), xgb_space, Xtr, ytr, n_iter, cv)

    log("[5/7] Improved candidates + stacked ensemble")
    lr_bal = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
    stack = StackingClassifier(
        estimators=[("rf", rf_best), ("xgb", xgb_best)],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=5, n_jobs=1).fit(Xtr, ytr)
    improved_models = {
        "Logistic Regression (balanced)": lr_bal,
        "Random Forest (tuned)": rf_best,
        f"{XGB_NAME} (tuned)": xgb_best,
        "Stacked Ensemble (RF+XGB -> LR)": stack,
    }
    improved, cv_auc = {}, {}
    for name, m in improved_models.items():
        improved[name], probas[name] = evaluate(m, Xte, yte)
        cv_auc[name] = float(cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        r = improved[name]
        log(f"  {name:<42} acc {r['accuracy']:.3f}  rec {r['recall']:.3f}  f1 {r['f1']:.3f}  auc {r['roc_auc']:.3f}  (CV auc {cv_auc[name]:.3f})")
    # Model selection uses training-set cross-validation only (never the test set).
    best_name = max(cv_auc, key=cv_auc.get)
    best = improved_models[best_name]
    log(f"  -> final model (best CV ROC-AUC): {best_name}")

    log("[6/7] Explainability")
    medians = Xtr.median()
    Xtr_f, Xte_f = Xtr.fillna(medians), Xte.fillna(medians)
    background = Xtr_f.sample(min(50, len(Xtr_f)), random_state=SEED).values
    sub = Xte_f.sample(min(80, len(Xte_f)), random_state=SEED)
    vals, base_val, method = shap_values(best, sub.values, background)
    mean_abs = np.abs(vals).mean(0)
    shap_rank = sorted(zip(FEATURES, mean_abs.tolist()), key=lambda t: -t[1])
    log(f"  {method}: top features = {[f for f, _ in shap_rank[:3]]}")
    pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)

    log("[7/7] Plots + saving")
    all_res = {**base, **improved}
    plot_confusions(all_res, PLOTS / "confusion_matrices.png")
    plot_curves("roc", base, improved, yte.values, probas, PLOTS / "roc_curves.png")
    plot_curves("pr", base, improved, yte.values, probas, PLOTS / "pr_curves.png")
    plot_bars(base, improved, PLOTS / "comparison_bars.png")
    plot_importance(rf_best, (pi.importances_mean, pi.importances_std), PLOTS / "feature_importance.png")
    plot_shap_summary(vals, sub.values, method, PLOTS / "shap_summary.png")

    bundle = {
        "model": best, "name": best_name, "features": FEATURES,
        "medians": medians.to_dict(), "background": background,
        "ranges": {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES},
        "explain_method": method,
    }
    joblib.dump(bundle, MODEL_PATH)
    save_json({
        "dataset": {
            "n": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "positive_rate": float(y.mean()),
            "missing_zeros": {c: int(df[c].isna().sum()) for c in ZERO_AS_MISSING},
        },
        "paper_accuracy": PAPER_ACC,
        "baseline": base, "improved": improved, "ablation": ablation,
        "cv_roc_auc": cv_auc, "best_model": best_name,
        "tuning": {"Random Forest": {"cv_f1": rf_cv, "params": rf_params},
                   XGB_NAME: {"cv_f1": xgb_cv, "params": xgb_params}},
        "feature_ranking": shap_rank, "explain_method": method,
        "have_xgboost": HAVE_XGB, "have_shap": HAVE_SHAP,
        "trained_seconds": round(time.time() - t0),
    }, RESULTS_PATH)
    log(f"Done in {time.time() - t0:.0f}s. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
