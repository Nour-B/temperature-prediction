#!/bin/bash

set -euxo pipefail

mlflow server \
    --host 0.0.0.0 \
    --port 8080 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_STORE}"
   