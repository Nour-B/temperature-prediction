#!/bin/bash

poetry run mlflow server \
    --host 0.0.0.0 \
    --port 8080 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE}