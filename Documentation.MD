# Crop Price Forecast API Documentation

This documentation provides instructions on setting up and using the Crop Price Forecast API.

## Setup

You can set up the API using Docker in two ways:

### Option 1: Build and Run Locally

1.  Ensure you have Docker installed.
2.  Navigate to the root directory of the codebase.
3.  Build the Docker image using the provided `Dockerfile`:
    ```bash
    docker build -t crop-price-api .
    ```
4.  Run the Docker container, mapping port 8080:
    ```bash
    docker run -p 8080:8080 crop-price-api
    ```
5.  The API should now be running locally. You can access the OpenAPI documentation at `http://localhost:8080/docs`.

### Option 2: Pull from Docker Hub

1.  Ensure you have Docker installed.
2.  Pull the latest image from Docker Hub:
    ```bash
    docker pull sllakshancc/crop-price-api:latest
    ```
3.  Run the Docker container, mapping port 8080:
    ```bash
    docker run -p 8080:8080 sllakshancc/crop-price-api:latest
    ```
4.  The API should now be running locally. You can access the OpenAPI documentation at `http://localhost:8080/docs`.

### Hosted App

Alternatively, you can use the hosted version of the app at:
[https://data-crunch-app-1086781479342.us-central1.run.app/docs](https://data-crunch-app-1086781479342.us-central1.run.app/docs)

## How to Use the API

**Important Note:** Before using any prediction or evaluation endpoints, you **must** first invoke the `/api/train_scratch` endpoint to train and initialize the models from scratch.

### Endpoints

Here's a description of the available endpoints:

* **`POST /api/train_scratch`**
    * Initiates the training of all models from scratch using historical data. This must be called before using other endpoints for the first time.
* **`POST /api/data/weather`**
    * Ingests new weather data records.
    * Triggers incremental training of models based on the new weather data.
    * Evaluates models and performs a full retraining from scratch if the NRMSE exceeds a threshold (1.55).
* **`GET /api/data/weather`**
    * Retrieves the latest weather data records for a specified region.
* **`POST /api/data/prices`**
    * Ingests new price data records.
    * Triggers incremental training for the models related to the commodities present in the new data.
    * Evaluates the models for the affected commodities and performs a full retraining from scratch if any of their NRMSE exceeds a threshold (1.55).
* **`GET /api/data/price`**
    * Retrieves the latest price data records for a specified commodity and region.
* **`POST /api/predict`**
    * Provides price predictions for a specified crop and region for the next 4 weeks.
* **`POST /api/evaluate`**
    * Evaluates all models and returns the overall RMSE and NRMSE, as well as individual evaluations per commodity.
* **`POST /api/evaluate_model`**
    * Evaluates the model for a specific commodity and returns its RMSE and NRMSE.

## Data Ingress and Prediction Approach

The application employs a hybrid approach to data ingress and prediction, balancing the trade-off between latency and accuracy. When new data (weather or price) is ingested, the relevant models are updated through **online learning** (incremental training). After the incremental training, the models are evaluated. If the prediction error (specifically, the Normalized Root Mean Squared Error - NRMSE) for new data is higher than a defined threshold (1.55), the system triggers a full **training from scratch** process using all available historical and newly ingested data. This ensures that the models are quickly updated with new data for most cases, while a full retraining occurs when necessary to maintain accuracy when the model's performance significantly degrades on new data.