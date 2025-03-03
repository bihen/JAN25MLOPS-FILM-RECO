import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from surprise import SVD, SVDpp, SlopeOne, KNNWithMeans, KNNBaseline
from surprise import accuracy
from surprise.model_selection import cross_validate, GridSearchCV
import joblib
import os
import bentoml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "models" 
MODEL_FOLDER = BASE_DIR / "models"
CONFIG_FOLDER = BASE_DIR / "config"

# Mapping for model names to classes
MODEL_MAPPING = {
    "svd": SVD,
    "svdpp": SVDpp,
    "slopeone": SlopeOne,
    "knnwithmeans": KNNWithMeans,
    "knnbaseline": KNNBaseline
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
    projectname = "mlops_movie_recommender"
    
    logger = logging.getLogger(__name__)
    logger.info(f'Training model {model_name}')
    
    input_filepath_train = os.path.join(INPUT_FOLDER, "trainset.data")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    model = train_model(input_filepath_train, 
                 output_folderpath,
                 model_class, best_params)
    model_ref = bentoml.picklable_model.save_model(projectname+"_"+model_name, model)
    print(f"Model saved as: {model_ref}")

def train_model(input_filepath_train, 
                 output_folderpath,
                 model, best_params):
 
    #--Importing dataset  
    trainset = joblib.load(input_filepath_train)
    
    trained_model = model(**best_params)
    trained_model.fit(trainset)

    print(f"Training complete with parameters: {best_params}")
    
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the best params in .pkl file
    output_filepath = os.path.join(output_folderpath, 'trained_model.pkl')
    joblib.dump(trained_model, output_filepath)
    return trained_model
       
def check_existing_folder(folder_path):
    '''Check if a folder already exists.'''
    if os.path.exists(folder_path) == False :
        return True
    else:
        return False
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()