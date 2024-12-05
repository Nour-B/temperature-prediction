# Make all targets .PHONY
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

include .envs/.postgres
include .envs/.mlflow
include .envs/.mlflow_auth
COMPOSE_DOCKER_CLI_BUILD=1
export


USER_NAME=$(shell whoami)
USER_ID=$(shell id -u)

DOCKER_IMAGE_NAME = fastapi-server
GCP_DOCKER_IMAGE_NAME = europe-west4-docker.pkg.dev/temperature-predictor-441809/temperature-prediction/mlflow-job
GCP_DOCKER_IMAGE_TAG = latest

DIRS_TO_VALIDATE= app tests web-app

# Run the pipeline
run_pipeline:
	@cd ./app && poetry run python3 main.py
	
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

exec-in-mlflow:
	docker exec -it local-mlflow-tracking-server bash 

exec-in-fastapi:
	docker exec -it fastapi-server bash
logs:
	docker compose logs $$1

## build image 
build-image: 
	docker build -t mlflow-job --build-arg=USER_ID=1000 --build-arg=USER_NAME=mlflow -f docker/dockerfile_mlflow_cloud_run .


## Push docker image to GCP Artifact Registry
push: build-image
	gcloud auth configure-docker europe-west4-docker.pkg.dev
	docker tag mlflow-job:latest "$(GCP_DOCKER_IMAGE_NAME):$(GCP_DOCKER_IMAGE_TAG)"
	docker push "$(GCP_DOCKER_IMAGE_NAME):$(GCP_DOCKER_IMAGE_TAG)"

buildx:
	@gcloud auth configure-docker eu.gcr.io --quiet
	@docker buildx build \
	--builder container \
	--build-arg USER_NAME=mlflow \
	--build-arg USER_ID=1000 \
	--file ./docker/dockerfile_mlflow_cloud_run \
	--platform linux/amd64,linux/arm64 \
	--progress auto \
	--tag "$(GCP_DOCKER_IMAGE_NAME):$(GCP_DOCKER_IMAGE_TAG)" \
	--push .

## Run unit tests	
unit-test:
	PYTHONPATH=. poetry run pytest

## Lint code using flake8
lint:
	$(foreach dir, $(DIRS_TO_VALIDATE), flake8 $(dir);)

## Deploy the model training function to GCP using cloud Functions
gcp-cloud-functions:
	cd ./app && gcloud functions deploy mlflow-client-test \
	--gen2 \
	--runtime=python311 \
	--region=europe-west4 \
	--source=. \
	--entry-point=run \
	--trigger-bucket=temperature-prediction-data \
	--memory=1GiB \
	--set-env-vars TRACKING_URI="http://34.147.71.252:8080" \
	--set-secrets  'MLFLOW_TRACKING_USERNAME=MLFLOW_TRACKING_USERNAME:1,MLFLOW_TRACKING_PASSWORD=MLFLOW_TRACKING_PASSWORD:2'
	
