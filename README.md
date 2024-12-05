# Predicting Temperature

## About

This is an end-to-end machine learning project that predicts the mean temperature in degrees Celsius using supervised learning techniques. The idea of this project comes from DataCamp platform [(link)](https://app.datacamp.com/learn/projects/predicting_temperature_in_london).
The project includes data preprocessing, model training, evaluation and model deployment in Production.


## Table of Contents
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Makefile](#makefile)
- [Testing and Linting](#testing-and-linting)
- [Model Tracking with MLflow](#model-tracking-with-mlflow)
- [Deploying on Google Cloud Functions](#deploying-on-google-cloud-functions)
- [FastAPI for Model Servings](#fastapi-for-model-serving)
- [Running the project Locally](#running-the-project-locally)
- [Future Enhancements](#future-enhancements)


## Architecture
Below is the architecture diagram that illustrates the workflow of the project from data cleaning to model deployment in production
![Image](docs/project_architecture.png)

## Technologies Used
- Poetry
- Make
- Units Tests: pytest
- Flake8
- Docker
- Kubernetes
- DVC
- GitHub Actions
- GCP Cloud (GKE, Cloud Run Functions, Cloud storage, Secret manager, Artifact Registry)
- FastAPI
- MLflow

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Nour-B/temperature-prediction.git
cd temperature-prediction
```

2. Install dependencies:
```bash
poetry install
```

## Makefile
The commands used to train, test, deploy the model and run the docker containers locally are listed in the Makefile.

## Testing and Linting

### Unit  Tests with pytest

Unit tests are in the tests/ directory. To run the tests, use the following command:
```bash
make unit-test
```

### Linting with Flake8
```bash
make lint
```

## Model Tracking with MLflow

## Deploying on Google Cloud Functions:

```bash
make gcp-cloud-functions
```

## FastAPI for Model Serving
Once the application is deployed, you can interact with the API via FastAPI.

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

## Running the project Localy:
To run the project locally, you can start the docker containers with docker compose:
```bash
make build
make up
```
Then you can Access the FastAPI app by opening the browser using http://localhost:8000 URL.

## Future Enhancements
- Configure a reverse proxy with SSL certificates to enable HTTPS for the FAST API endpoint and the MLflow.
- Implement a Kubernetes Ingress for externl Access instead of the service.
