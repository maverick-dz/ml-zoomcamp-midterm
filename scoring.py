"""
scoring.py - single source of truth for the feature schema and for turning a
feature record into a calibrated risk probability.

Used by:
  * train.py            (feature order when building the training matrix)
  * predict.py          (FastAPI request validation + inference)
  * aws/lambda_function.py (optional AWS Lambda handler)

Keeping this in one place means the API, the Lambda handler and the training
script can never disagree about which features exist, their order, or their
valid ranges.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# name -> (kind, min, max, description, example)
# kind: "bool" (0/1), "int" (whole number) or "float".
FEATURE_SPECS: Dict[str, tuple] = {
    "highbp": ("bool", 0, 1, "High blood pressure: 0=no, 1=yes", 1),
    "highchol": ("bool", 0, 1, "High cholesterol: 0=no, 1=yes", 1),
    "cholcheck": ("bool", 0, 1, "Cholesterol checked in past 5 years: 0/1", 1),
    "bmi": ("float", 10, 100, "Body Mass Index (10-100)", 28.0),
    "smoker": ("bool", 0, 1, "Smoked at least 100 cigarettes in lifetime: 0/1", 1),
    "stroke": ("bool", 0, 1, "Ever told you had a stroke: 0/1", 0),
    "heartdiseaseorattack": ("bool", 0, 1,
                             "Coronary heart disease or myocardial infarction: 0/1", 0),
    "physactivity": ("bool", 0, 1,
                     "Physical activity in past 30 days (not job): 0/1", 0),
    "fruits": ("bool", 0, 1, "Consumes fruit at least once a day: 0/1", 0),
    "veggies": ("bool", 0, 1, "Consumes vegetables at least once a day: 0/1", 1),
    "hvyalcoholconsump": ("bool", 0, 1,
                          "Heavy drinker (men >14 drinks/week, women >7): 0/1", 0),
    "anyhealthcare": ("bool", 0, 1, "Has any kind of health care coverage: 0/1", 1),
    "nodocbccost": ("bool", 0, 1,
                    "Could not see a doctor because of cost in past 12 months: 0/1", 0),
    "genhlth": ("int", 1, 5, "Self-rated general health: 1=excellent ... 5=poor", 3),
    "menthlth": ("int", 0, 30, "Days of poor mental health in past 30 days", 0),
    "physhlth": ("int", 0, 30, "Days of poor physical health in past 30 days", 0),
    "diffwalk": ("bool", 0, 1,
                 "Serious difficulty walking or climbing stairs: 0/1", 0),
    "sex": ("bool", 0, 1, "0=female, 1=male", 0),
    "age": ("int", 1, 13,
            "Age bucket in 5-year steps: 1=18-24, 2=25-29, ... 13=80 or older", 9),
    "education": ("int", 1, 6,
                  "Education level: 1=never attended/kindergarten only ... "
                  "6=college graduate", 4),
    "income": ("int", 1, 8,
               "Income bucket: 1=under $10k ... 8=$75k or more", 3),
}

FEATURES = list(FEATURE_SPECS)  # canonical column order (21 features)


def validate_record(record: Dict[str, Any]) -> Dict[str, float]:
    """Validate a raw dict (used by the Lambda handler; FastAPI uses Pydantic).

    Raises ValueError with a readable message on missing/unknown/out-of-range
    fields. Returns a clean {feature: float} dict.
    """
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    unknown = [k for k in record if k not in FEATURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}")
    clean: Dict[str, float] = {}
    for name, (kind, lo, hi, _desc, _ex) in FEATURE_SPECS.items():
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if kind != "float" and float(value) != int(value):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}, got {value!r}")
        clean[name] = float(value)
    return clean


def load_bundle(path: Path | str) -> Dict[str, Any]:
    """Load model.bin and warn if library versions differ from training."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)  # only ever load files you produced yourself
    trained = bundle.get("versions", {})
    try:
        import sklearn
        import xgboost
        now = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
               "numpy": np.__version__}
    except ImportError:  # pragma: no cover
        now = {}
    for lib, ver in trained.items():
        if lib in now and now[lib] != ver:
            warnings.warn(
                f"model.bin was trained with {lib}=={ver} but {lib}=={now[lib]} "
                "is installed; predictions may differ or loading may fail. "
                "Install the pinned requirements or re-run train.py.")
    return bundle


def predict_one(bundle: Dict[str, Any], record: Dict[str, float],
                threshold: Optional[float] = None) -> Dict[str, Any]:
    """Score one record -> calibrated probability + 0/1 decision."""
    x = np.array([[record[f] for f in bundle["features"]]], dtype=float)
    raw = float(bundle["model"].predict_proba(x)[0, 1])
    calibrator = bundle.get("calibrator")
    proba = float(calibrator.predict([raw])[0]) if calibrator is not None else raw
    thr = float(bundle["threshold"] if threshold is None else threshold)
    return {
        "diabetes_probability": proba,
        "diabetes_prediction": int(proba >= thr),
        "threshold": thr,
        "model": bundle.get("model_name", "unknown"),
    }
