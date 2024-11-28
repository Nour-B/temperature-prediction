# Make all targets .PHONY
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

include .envs/.postgres
include .envs/.mlflow
COMPOSE_DOCKER_CLI_BUILD=1
export


USER_NAME=$(shell whoami)
USER_ID=$(shell id -u)

DOCKER_IMAGE_NAME = fastapi-server
GCP_DOCKER_IMAGE_NAME = europe-west4-docker.pkg.dev/temperature-predictor-441809/temperature-prediction/temperature-prediction-fastapiserver
GCP_DOCKER_IMAGE_TAG := $(strip $(shell uuidgen))

DIRS_TO_VALIDATE= app tests web-app

# Run the pipeline
run_pipeline:
	@python3 ./app/main.py
	
build:
		docker compose build

up:
	docker compose up 

down:
	docker compose down

exec-in-mlflow:
	docker exec -it local-mlflow-tracking-server bash 

exec-in-fastapi:
	docker exec -it fastapi-server bash
logs:
	docker compose logs $$1

## Push docker image to GCP Artifact Registry
push: build
	gcloud auth configure-docker europe-west4-docker.pkg.dev
	docker tag "$(DOCKER_IMAGE_NAME)":latest "$(GCP_DOCKER_IMAGE_NAME):$(GCP_DOCKER_IMAGE_TAG)"
	docker push "$(GCP_DOCKER_IMAGE_NAME):$(GCP_DOCKER_IMAGE_TAG)"

## Run unit tests	
unit-test:
	PYTHONPATH=. poetry run pytest

## Lint code using flake8
lint:
	$(foreach dir, $(DIRS_TO_VALIDATE), flake8 $(dir);)
