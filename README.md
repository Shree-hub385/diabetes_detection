# Diabetes Detection System — Capstone Project

Implements the plan from the capstone deck *"Evaluating a Machine-Learning Base Paper and
Applying Its Lessons to an Early-Diabetes-Risk Prediction Capstone"*
(Prajwal Joshi, Shreerang Kulkarni, Janhavi Kaysath, Kunal Salunke).

## Run it (one click)

| Your system | Do this |
|---|---|
| **Windows** | double-click **`run.bat`** |
| **macOS** | double-click **`run.command`** (first time: right-click → Open) |
| **Linux** | `./run.sh` |
| Any | `python run.py` |

Requires **Python 3.9+** and internet on the first run. The launcher automatically:
1. creates a private `.venv` and installs the packages,
2. downloads the Pima Indians Diabetes dataset into `data/diabetes.csv`
   (if you're offline, put your own copy of `diabetes.csv` there),
3. trains and evaluates all models (first run only, ~2–5 min) and saves plots/metrics,
4. starts the dashboard and opens **http://127.0.0.1:5000** in your browser.

Options: `python run.py --retrain` · `--quick` (fast demo) · `--no-browser` · `--port 8000`

## Deck requirement → where it lives

| Slide | Requirement | Implementation |
|---|---|---|
| 2 | Pima data, 5 base-paper classifiers, stratified 80:20 | `src/train.py` step 2 (LR, DT, RF, SVM, XGBoost; `random_state=42`) |
| 3–4 | Accuracy, Precision, Recall, F1, ROC-AUC | `evaluate()` in `src/common.py` |
| 5 | Accuracy table vs. paper | dashboard → *Model performance* → "Base paper vs. our re-run" |
| 6 | ROC curves, feature importance | `outputs/plots/roc_curves.png`, `feature_importance.png` |
| 8 | Adopt RF + XGBoost, LR baseline | `train.py` steps 2 & 4 |
| 8 | **SMOTE / class weighting** | `SmoteClassifier` (in-fold SMOTE) + class weights; ablation table + part of the tuning search |
| 8 | **Stacked ensemble** (RF+XGB → LR meta-learner) | `StackingClassifier` in `train.py` step 5 |
| 9 | Specificity + confusion matrix | `evaluate()`; `confusion_matrices.png` |
| 9 | Visuals: confusion matrices, overlaid ROC, PR curve, SHAP summary, bar chart vs. paper | `outputs/plots/*.png` (also shown in the dashboard) |
| 10 | Same-split comparison strategy | all models share one stratified split; final model chosen by CV ROC-AUC on *training* data only |
| 10 | **Hyper-parameter tuning** of RF & XGBoost | `RandomizedSearchCV` (5-fold, F1) |
| 10 | **Explainability (SHAP)** | per-patient SHAP bars in the dashboard + global summary plot |
| 10 | **Dashboard** for a clinician | `src/app.py` + `templates/index.html` (risk score, bands, threshold slider, explanation, metrics) |

## Project layout
```
run.py / run.bat / run.sh / run.command   one-click launchers
src/common.py    data download/cleaning, SMOTE, metrics
src/train.py     benchmark, imbalance ablation, tuning, stacking, plots, saves model
src/explain.py   SHAP (or built-in Shapley sampling if `shap` isn't installed)
src/app.py       Flask dashboard + JSON API (/api/predict, /api/metrics)
templates/       dashboard page          outputs/   model, results.json, plots
```

## Notes
* Preprocessing: impossible zeros (glucose, BP, skin-fold, insulin, BMI) → missing → median-imputed
  inside the pipeline (fit on training folds only). The base paper may have preprocessed differently,
  so small differences from its numbers are expected — report *your* re-run numbers as the benchmark.
* If `xgboost` / `shap` can't be installed on your machine, the code transparently falls back to
  scikit-learn `HistGradientBoostingClassifier` / a built-in Shapley estimator, and the dashboard says so.
* This is an educational screening aid, **not** a medical device or diagnosis.
