"""
predict.py - FastAPI service that serves calibrated diabetes-risk predictions.

Endpoints
  GET  /health    -> {"status": "ok"}
  GET  /          -> service + model info
  POST /predict   -> {"diabetes_probability", "diabetes_prediction", "threshold", "model"}
                     optional query parameter ?threshold=0.2 overrides the default
                     decision threshold stored in model.bin

Run locally:
    uvicorn predict:app --host 0.0.0.0 --port 9696
Swagger UI: http://localhost:9696/docs

Environment variables
    MODEL_PATH       path to model.bin           (default: ./model.bin)
    ALLOWED_ORIGINS  comma-separated CORS origins (default: "*"). Set this to
                     your Cloudflare Pages URL in production.

NOT a medical device: this is a screening-style demo trained on self-reported
survey data (CDC BRFSS 2015). It does not diagnose diabetes.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model

from scoring import FEATURE_SPECS, load_bundle, predict_one

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent / "model.bin"))
BUNDLE = load_bundle(MODEL_PATH)  # loaded once at startup
MODEL_NAME = BUNDLE.get("model_name", "unknown")

DISCLAIMER = ("Educational demo trained on self-reported CDC BRFSS 2015 survey "
              "data. It estimates risk for screening purposes only and does not "
              "diagnose diabetes. Consult a healthcare professional.")

# ---- Request schema, generated from the shared FEATURE_SPECS ---------------
_fields = {}
for _name, (_kind, _lo, _hi, _desc, _example) in FEATURE_SPECS.items():
    _type = float if _kind == "float" else int
    _fields[_name] = (
        _type,
        Field(..., ge=_lo, le=_hi, description=_desc,
              json_schema_extra={"example": _example}),
    )

PatientFeatures = create_model(
    "PatientFeatures",
    __config__=ConfigDict(extra="forbid"),  # reject unknown fields
    __doc__="One respondent: all 21 BRFSS features (see /docs for ranges).",
    **_fields,
)


class PredictionOut(BaseModel):
    diabetes_probability: float = Field(
        description="Calibrated probability of diabetes / pre-diabetes (0-1)")
    diabetes_prediction: int = Field(
        description="1 if probability >= threshold else 0")
    threshold: float = Field(description="Decision threshold that was applied")
    model: str


# ---- App --------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Health Indicators Predictor",
    description=("Estimates the probability that a respondent has diabetes or "
                 "pre-diabetes from CDC BRFSS lifestyle/health indicators.\n\n"
                 + DISCLAIMER),
    version="2.0.0",
)

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
            if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/")
def root():
    return {
        "service": "diabetes-health-indicators-predictor",
        "model": MODEL_NAME,
        "default_threshold": BUNDLE["threshold"],
        "target_recall": BUNDLE.get("target_recall"),
        "feature_count": len(BUNDLE["features"]),
        "endpoints": ["/health", "/predict", "/docs"],
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(
    features: PatientFeatures,  # type: ignore[valid-type]
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Override the default decision threshold (0-1)."),
):
    return PredictionOut(**predict_one(BUNDLE, features.model_dump(), threshold))
