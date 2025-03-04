Movie Recommender System based on Movielens Dataset
==============================

This project is for the DataScientest course.
We are building a Movie Recommendation System based on a subset of the 32m movielens dataset.([https://grouplens.org/datasets/movielens/32m/](https://grouplens.org/datasets/movielens/32m/))

For this particular project, which is more focused around a functioning API and deployment of a model, we are using a basic Surprise ([https://surpriselib.com/](https://surpriselib.com/)) model.
The type of model can be configured using the config.json in /config. THe available models are:

- SVD
- SVDpp
- SlopeOne
- KNNWithMeans
- KNNBaseline

The project is preconfigured for SVD, for the fastest performance.


Data Preparation
==============================
For the raw data for our model, any MovieLens .csv ratings file can be used (or any other .csv that contains explicit ratings, userIDs and movieIDs).
The base model (constituting our reference data to compare to for data drift/performance) is based on the ml-latest-small MovieLens set.

The data is loaded from data/raw into Surprise and converted to the proprietary data format for Surprise, then saved in data/preprocessed as pickled files.
Data is tracked via DVC. Current DVC setup is for googledrive.

Folder Structure:
------------

    ├── LICENSE
    ├── README.md          <- This readme
    ├── config
    │   ├── config.json          <- config file used to select model
    │   ├── param_grid.json      <- config file to set the parameter grid for different models
    │   ├── users.json          <- sample json for a very basic user database - CHANGE IN DEPLOYMENT
    │   └── secrets.json      <- sample json for setting JWT secret key - CHANGE IN DEPLOYMENT
    ├── data               <- tracked with DVC
    │   ├── processed      <- Processed Surprise Datasets
    │   └── raw            <- Contains at least ratings.csv and movies.csv from MovieLens
    │
    ├── metrics            <- Location to save metric data 
    |
    ├── models             <- Contains the trained and serialized model, as well as the best_params.pkl used to train the model 
    │
    ├── notebooks          <- Contains Jupyter Notebooks from previous project for reference
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment
    │
    ├── src                <- Source code for use in this project. Top folder contains API and testing files
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data       <- Scripts to turn raw data into features for modeling
    │   │   └── data_split.py <- turns data into correct format for surprise and splits into test- and trainset
    │   │   └── create_prediction_data.py <- creates a prediction sample for 15 users (10 fixed, 5 random) to compare for model drift
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model_grid_search.py <- script to perform a grid search to find the best parameters for currently selected model
    │   │   ├── model_evaluate.py <- script to create evaluaion metrics to be saved in /metrics
    │   │   └── train_model.py <- Training the actual model using best_params created earlier
    │   │
    │   ├── evidently_service.py  <- creates evidently data and model drift reports
    │   │   
    │   ├── service.py  <- BentoML API service, started with bentoml serve service.py
    |   |
    │   ├── test_service.py  <- pytest file to test service.py

Architecture
==============================
![JAN25MLOPS-FILM-RECO Pipeline](https://github.com/user-attachments/assets/54dce1e1-ce2b-4810-99d3-9319ff8f5be9)

                            Figure 1. Pipeline for the project architecture


DagsHub Integrations
==============================
This project has a corresponding DagsHub repository: [https://dagshub.com/bihen/JAN25MLOPS-FILM-RECO](https://dagshub.com/bihen/JAN25MLOPS-FILM-RECO)

MLFlow
==============================
This project has experiment tracking with MLFlow, hosted on DagsHub:  [https://dagshub.com/bihen/JAN25MLOPS-FILM-RECO/experiments](https://dagshub.com/bihen/JAN25MLOPS-FILM-RECO/experiments)

Deployment
==============================
Requirements outside python:
* Docker
* Docker-compose
* DVC

1. Setup DVC

Configure .dvc/config.local with the necessary secret data for gdrive (including the secret json file), or setup your own DVC (editing .dvc/config necessary)

2. Pull data from DVC

Use ```dvc pull``` to pull the current data from DVC

3. Setup other Secrets

Setup a .env file to configure DagsHub. 
The .env file should contain: DAGSHUB_USER_TOKEN=[YOUR TOKEN HERE].
DagsHub user tokens can be found in settings -> Tokens.
Setup config/secrets.json and config/users.json to a secret and userbase respectively.

4. Build containers with Docker-Compose

Use ```docker-compose build``` to build all necessary containers

5. Start docker containers

Use ```docker-compose up -d``` to start all containers in detached state. This might take a while, as the model will need to do a grid search and then be trained.

Services
==============================
After following these steps the following services will be available:
- Port 3000: BentoML API
  * ENDPOINT /healthz: checks if API is running and model is loaded
  * ENDPOINT /login: logs in user based on config/users.json and returns a valid Bearer token
  * ENDPOINT /v1/models/movierecommender/predict: route for actual API prediction. Requires a valid bearer token in the header and a JSON just containing the ID of the user you want to predict. E.g.: { "userID" : 1 }
- Port 3001: Grafana
  * Contains dashboards vizualizing data from Prometheus
- Port 8000: Evidently UI
  * Contains Evidently reports on data drift and model drift
- Port 9090: Prometheus
  * Contains different metrics collected on the API

Testing
==============================
6. Run API tests

Use ```pytest src/test_service.py``` to test functionality for the API

Shut down and clean up
==============================
7. Shut down the API and clean up docker-compose

Use ```docker-compose down``` to stop all running containers and clean up the process

Development Team
==============================

Felix Oey 
https://github.com/lyxoey

Hassan Haddouchin 
https://github.com/hhadd

Bianca Edelweiss van Hemert
https://github.com/bihen

--------


