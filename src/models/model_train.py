import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "models" 
MODEL_FOLDER = BASE_DIR / "models"
CONFIG_FOLDER = BASE_DIR / "config"

# Mapping for model names to classes
MODEL_MAPPING = {
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "lasso": Lasso,
    "randomforestregressor": RandomForestRegressor,
    "gradientboostingregressor": GradientBoostingRegressor
}

def load_config():
    """
    Load the model selection config (config.json).
    """
    with open(os.path.join(CONFIG_FOLDER, "config.json")) as f:
        config = json.load(f)
    return config

def load_best_params():
    """
    Load the parameter grid config (best_params.pkl).
    """
    path = os.path.join(MODEL_FOLDER, "best_params.pkl")
    param_grid = joblib.load(path)
    return param_grid

def main():
    """ Trains the model using the best parameters determined by GridSearch.
    """
    config = load_config()
    best_params = load_best_params()
    model_name = config.get("model_name")
    model_class = MODEL_MAPPING.get(model_name)
    
    logger = logging.getLogger(__name__)
    logger.info(f'Training model {model_name}')
    
    input_filepath_test_x = os.path.join(INPUT_FOLDER, "X_test_scaled.csv")
    input_filepath_train_x = os.path.join(INPUT_FOLDER, "X_train_scaled.csv")
    input_filepath_test_y = os.path.join(INPUT_FOLDER, "y_test.csv")
    inpút_filepath_train_y = os.path.join(INPUT_FOLDER, "y_train.csv")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    train_model(input_filepath_test_x, input_filepath_train_x, 
                 input_filepath_test_y, inpút_filepath_train_y,
                 output_folderpath,
                 model_class, best_params)

def train_model(input_filepath_test_x, input_filepath_train_x, 
                 input_filepath_test_y, inpút_filepath_train_y, 
                 output_folderpath,
                 model, best_params):
 
    #--Importing dataset
    X_train = pd.read_csv(input_filepath_train_x, sep=",")
    X_test = pd.read_csv(input_filepath_test_x, sep=",")
    y_train = pd.read_csv(inpút_filepath_train_y, sep=",")
    y_test = pd.read_csv(input_filepath_test_y, sep=",")
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    trained_model = model()
    trained_model.set_params(**best_params)
    trained_model.fit(X_train, y_train)

    print(f"Training complete with parameters: {best_params}")
    
    
    
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the best params in .pkl file
    for file, filename in zip([trained_model], ['trained_model']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.pkl')
        if check_existing_file(output_filepath):
            joblib.dump(trained_model, output_filepath)
            

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()