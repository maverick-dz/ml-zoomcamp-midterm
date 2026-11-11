# Lambda container image. Build from the REPOSITORY ROOT so model.bin and
# scoring.py are inside the build context:
#
#     docker build -f aws/Dockerfile -t diabetes-lambda .
#
# (aws/template.yaml already points SAM at the right context.)
FROM public.ecr.aws/lambda/python:3.12

COPY aws/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY scoring.py model.bin ${LAMBDA_TASK_ROOT}/
COPY aws/lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]
