"""
AWS Lambda handler (API Gateway REST proxy integration) - OPTIONAL deployment.

Uses the same scoring.py / model.bin as the FastAPI service, packaged as a
Lambda *container image* (see aws/Dockerfile) so the model and libraries are not
squeezed into the 250 MB zip-package limit.
"""
import json
import os
from pathlib import Path

from scoring import load_bundle, predict_one, validate_record

BUNDLE = load_bundle(Path(os.environ.get("MODEL_PATH", "model.bin")))

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": os.environ.get("ALLOWED_ORIGINS", "*"),
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}


def _response(status, payload):
    return {"statusCode": status, "headers": HEADERS, "body": json.dumps(payload)}


def lambda_handler(event, context):
    path = event.get("rawPath") or event.get("path") or ""
    method = (event.get("httpMethod")
              or event.get("requestContext", {}).get("http", {}).get("method", ""))

    if method == "OPTIONS":
        return _response(200, {})
    if path.endswith("/health"):
        return _response(200, {"status": "ok"})

    try:
        body = event.get("body")
        record = json.loads(body) if isinstance(body, str) else (body or {})
        threshold = (event.get("queryStringParameters") or {}).get("threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1")
        clean = validate_record(record)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return _response(422, {"error": str(exc)})

    return _response(200, predict_one(BUNDLE, clean, threshold))
