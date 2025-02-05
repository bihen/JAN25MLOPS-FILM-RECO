import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from surprise import accuracy
import joblib
import os
import mlflow
import dagshub

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
MODEL_FOLDER = BASE_DIR / "models"
METRICS_FOLDER = BASE_DIR / "metrics"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

# DagsHub integration for MLflow
dagshub.init(repo_owner='bihen', repo_name='JAN25MLOPS-FILM-RECO', mlflow=True)

# Main function
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading the trained model...")
    model = load_trained_model()

    logger.info("Loading the test data...")
    testset = load_test_data()

    logger.info("Evaluating the model...")
    metrics = evaluate_model(model, testset)

    logger.info(f"Model evaluation metrics: {metrics}")
    with mlflow.start_run():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_param("model_evaluate", "complete")
        print("Metrics logged to MLflow via DagsHub.")
        
        
# Evaluate the model
def evaluate_model(model, testset):
    """
    Evaluate the model using test data and calculate various metrics.
    """
    # Make predictions on the test set
    predictions = model.test(testset)

    # Calculate MSE, R^2, and optionally more metrics
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    fcp = accuracy.fcp(predictions)

    # Store metrics in a dictionary
    metrics = {
        "rmse": rmse,
        "fcp": fcp,
        "mae": mae,
    }
    
    # Create folder if necessary 
    if check_existing_folder(METRICS_FOLDER) :
        os.makedirs(METRICS_FOLDER)

    #--Saving the best params in .pkl file
    
    output_filepath = os.path.join(METRICS_FOLDER, 'metrics.json')

    with open(output_filepath, "w", encoding="utf-8") as output_filepath:
        json.dump(metrics, output_filepath, indent=4, ensure_ascii=False)

    return metrics


# Load the trained model from a joblib file
def load_trained_model():
    """
    Load the trained model from the saved file.
    """
    model_path = MODEL_FOLDER / "trained_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

# Load test data
def load_test_data():
    """
    Load test dataset.
    """
    testset = joblib.load(os.path.join(INPUT_FOLDER, "testset.data"))
    return testset

   
       
def check_existing_folder(folder_path):
    '''Check if a folder already exists.'''
    if os.path.exists(folder_path) == False :
        return True
    else:
        return False
if __name__ == '__main__':
    main()
