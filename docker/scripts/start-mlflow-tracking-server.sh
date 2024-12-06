#!/bin/bash

set -euxo pipefail

mlflow server \
    --app-name basic-auth \
    --host 0.0.0.0 \
    --port 8080 \
    --backend-store-uri "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_HOST}:5432/${POSTGRES_DB}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_STORE}"
   