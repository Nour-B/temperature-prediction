#!/bin/bash
set -euxo pipefail

/start-mlflow-tracking-server.sh &
tail -F anything