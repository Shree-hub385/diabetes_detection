"""Flask dashboard: risk score + SHAP explanation + model metrics/visuals."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from src.common import FEATURES, MODEL_PATH, PLOTS, RESULTS_PATH
from src.explain import shap_values

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / "templates"))

# field -> (label, unit, min allowed, max allowed, required)
FIELDS = {
    "Pregnancies": ("Pregnancies", "count", 0, 20, True),
    "Glucose": ("Plasma glucose (2h OGTT)", "mg/dL", 40, 300, True),
    "BloodPressure": ("Diastolic blood pressure", "mm Hg", 30, 160, False),
    "SkinThickness": ("Triceps skin-fold thickness", "mm", 5, 100, False),
    "Insulin": ("2-hour serum insulin", "µU/mL", 10, 900, False),
    "BMI": ("Body-mass index", "kg/m²", 10, 70, True),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", "score", 0.05, 3.0, True),
    "Age": ("Age", "years", 18, 100, True),
}

app.json.sort_keys = False
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not trained yet - run:  python -m src.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def band(p):
    return ("Low", "low") if p < 0.35 else ("Moderate", "moderate") if p < 0.65 else ("High", "high")


@app.get("/")
def index():
    ready = MODEL_PATH.exists() and RESULTS_PATH.exists()
    return render_template("index.html", ready=ready, fields=FIELDS)


@app.get("/api/meta")
def meta():
    b = bundle()
    return jsonify({
        "model": b["name"], "explain_method": b["explain_method"],
        "medians": b["medians"],
        "fields": {k: {"label": v[0], "unit": v[1], "min": v[2], "max": v[3], "required": v[4]}
                   for k, v in FIELDS.items()},
    })


@app.get("/api/metrics")
def metrics():
    if not RESULTS_PATH.exists():
        abort(404)
    return jsonify(json.loads(RESULTS_PATH.read_text()))


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/plots/<path:name>")
def plots(name):
    return send_from_directory(PLOTS, name)


@app.post("/api/predict")
def predict():
    b = bundle()
    data = request.get_json(silent=True) or {}
    row, assumed, errors = {}, [], {}
    for k, (label, unit, lo, hi, required) in FIELDS.items():
        raw = data.get(k)
        if raw in (None, ""):
            if required:
                errors[k] = f"{label} is required."
            else:
                row[k] = b["medians"][k]
                assumed.append(k)
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[k] = f"{label} must be a number."
            continue
        if not (lo <= v <= hi) or not np.isfinite(v):
            errors[k] = f"{label} should be between {lo} and {hi} {unit}."
            continue
        row[k] = v
    if errors:
        return jsonify({"errors": errors}), 400

    x = pd.DataFrame([row])[FEATURES]
    prob = float(b["model"].predict_proba(x)[0, 1])
    vals, base, method = shap_values(b["model"], x.values, b["background"])
    contrib = [{"feature": f, "label": FIELDS[f][0], "value": float(row[f]),
                "unit": FIELDS[f][1], "shap": float(vals[0, i]), "assumed": f in assumed}
               for i, f in enumerate(FEATURES)]
    contrib.sort(key=lambda c: -abs(c["shap"]))
    label, css = band(prob)
    return jsonify({"probability": prob, "band": label, "band_class": css,
                    "base_value": base, "method": method, "contributions": contrib,
                    "assumed": assumed, "model": b["name"]})


def run(host="127.0.0.1", port=5000):
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run()
