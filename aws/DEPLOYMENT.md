# Optional: AWS Lambda deployment (container image)

> **Status: not deployed or tested by the author.** The primary deployment of
> this project is **Render (API) + Cloudflare Pages (UI)** - see the main
> README. This folder is an alternative that has not been run end to end.

Why a container image? The XGBoost + scikit-learn + SciPy + NumPy stack is too
large for Lambda's 250 MB zip-package limit; container images allow up to 10 GB.

## Files
- `lambda_function.py` - handler; reuses `../scoring.py` and `../model.bin`
- `Dockerfile` - Lambda base image + slim requirements (build from repo root)
- `template.yaml` - SAM template (Lambda + API Gateway)
- `requirements.txt` - Lambda-only dependencies (keep in sync with `../requirements-serve.txt`)

## Prerequisites
AWS account, AWS credentials (`aws configure`), Docker running, SAM CLI.

## Deploy (run from the repository root)
```bash
sam build -t aws/template.yaml
sam deploy --guided        # accept the defaults; allow SAM to create ECR repos
```
SAM prints an endpoint like `https://abc123.execute-api.us-east-1.amazonaws.com/Prod`.

## Test
```bash
export API_URL="https://abc123.execute-api.us-east-1.amazonaws.com/Prod"
curl $API_URL/health
curl -X POST $API_URL/predict -H "Content-Type: application/json" -d @sample_request.json
```

## Tear down (avoid charges)
```bash
sam delete
```

## Notes
- If you build on an ARM machine (Apple Silicon) the template pins `x86_64`;
  Docker will cross-build the image (slower, but correct).
- Cold starts for this image are typically a few seconds; warm calls are fast.
