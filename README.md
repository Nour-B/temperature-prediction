# Predicting Temperature

## About

This is an end-to-end machine learning project that predicts the mean temperature in degrees Celsius using supervised learning techniques. The idea of this project comes from DataCamp platform [(link)](https://app.datacamp.com/learn/projects/predicting_temperature_in_london).
The project includes data preprocessing, model training, evaluation and model deployment in Production. It uses tools for managing dependencies, automation, testing, linting, containerization, and cloud deployment.


## Table of Contents
- [Architecture](#architecture)
- [Setup](#setup)
- [Development workflow](#development-workflow)
- [Model Tracking with MLflow](#model-tracking-with-mlflow)
- [CI/CD with GitHub Actions](#ci/cd-with-github-actions)
- [Google cloud Functions](#google-cloud-functions)
- [Future Enhancements](#future-enhancements)


## Architecture
Below is the architecture diagram that illustrates the workflow of the project from data cleaning to model deployment in production

![Image](docs/project_architecture.png)

## Setup

### Technologies Used
- Poetry
- Make
- Units Tests: pytest
- Flake8
- Docker/ Docker compose
- Kubernetes
- DVC
- GitHub Actions
- GCP Cloud (GKE, Cloud Run Functions, Cloud storage, Secret manager, Artifact Registry)
- FastAPI
- MLflow

### Prerequisites
Before getting started, ensure you have the following prerequisites:
- Python 3.11+
- Poetry
- Docker
- kubectl
- gcloud CLI

Additionally, make sure you have the following accounts and services:
- GCP account
- GitHub account
- GKE cluster set up in GCP
- Ingress controller (e.g., Nginx Ingress Controller)


### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Nour-B/temperature-prediction.git
cd temperature-prediction
```

2. Install dependencies:
```bash
poetry install
```
3. Activate the poetry virtual environment:
```bash
poetry shell
```

## Development workflow
This project uses a Makefile to train, test, deploy the model and run the docker containers locally.

### Unit  Tests with pytest

Unit tests are in the tests/ directory. To run the tests:
```bash
make unit-test
```

### Linting with Flake8
You can lint your code by running the following command:
```bash
make lint
```

### Running the project Locally:

Ensure the `.envs` directory is configured by replacing the following `Environment variables` with you values:
- `MLFLOW_ARTIFACT_STORE`: GCS Bucket where models will be stored.
- `MLFLOW_TRACKING_USERNAME`: User used for MLflow authentication
- `MLFLOW_TRACKING_PASSWORD`: Password used for MLflow authentication
- `TRACKING_URI`: MLflow server
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: user used for Postgres authentication
- `POSTGRES_PASSWORD`: password used for Postgres authentication
- `DB_HOST`: Host where the database is running

To run the project locally, you can use docker compose to run the services (MLflow tracking sevrer, Postgres and FastAPI app):

```bash
make build
make up
```
### FastAPI
Now you can interact with the API via FastAPI using this URL: http://localhost:8000.

To predict the mean tempreature, send a POST request to the /DecisionTreeRegressor, /LinearRegression or /RandomForestRegressor endpoints with the appropriate data.

Example:
```bash
{
   "month":1,
    "cloud_cover": 2.0,
    "sunshine": 7.0,
    "precipitation": 0.4,
    "pressure": 101900.0,
    "global_radiation": 52.0
}
```

## Model Tracking with MLflow
This project uses a remote MLflow tracking server with the following configuration:
- GCS Bucket to store the artifacts (models)
- PostgreSQL Database to store the metadata (parameters, metrics)

## CI/CD with GitHub Actions

For the CI/CD, GitHub Actions is used. The workflow `.github/workflows/ci-cd.yaml` includes the following steps:

- Lint
- Run the Unit Tests
- Build the Docker images and push them to Google Artifact Registry
- Deploy the application to GKE

The .github/workflows/ci-cd.yml file is configured to automatically run these steps whenever code is pushed to the repository.

## Google cloud Functions
You can trigger the model training using Google cloud functions by pushing the raw data to GCS Bucket. The function code is stored in the `app` directory. To deploy the function use the following command. Make sure to replace the variables with your values:

```bash
make gcp-cloud-functions
```

## Future Enhancements
- Configure a reverse proxy with SSL certificates to enable HTTPS for the FAST API endpoint and the MLflow.
- Use Helm to simplify the Kubernetes deployment