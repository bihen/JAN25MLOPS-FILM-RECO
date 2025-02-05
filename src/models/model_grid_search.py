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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER = BASE_DIR / "models" 
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

def load_param_grid():
    """
    Load the parameter grid config (param_grid.json).
    """
    with open(os.path.join(CONFIG_FOLDER, "param_grid.json")) as f:
        param_grid = json.load(f)
    return param_grid

def main():
    """ Runs a GridSearchCV with the selected model from config.json and saves the best parameters in models/best_params.skl
    """
    config = load_config()
    param_grid = load_param_grid()
    model_name = config.get("model_name")
    
    logger = logging.getLogger(__name__)
    logger.info(f'finding best parameters using GridSearch with model {model_name}')
    
    input_filepath_data = os.path.join(INPUT_FOLDER, "data.data")
    output_folderpath = OUTPUT_FOLDER
    
    try:
        # Get the model and parameter grid based on config file
        model, param_grid = get_model_and_params(model_name, param_grid)
    except ValueError as e:
        # Handle invalid model names
        print(e)
        return
    
    # Call the main data processing function with the provided file paths
    find_best_params(input_filepath_data, 
                 output_folderpath,
                 model, param_grid)

def find_best_params(input_filepath_data, 
                 output_folderpath,
                 model, param_grid):
 
    #--Importing dataset
    data = joblib.load(input_filepath_data)
    
    # Creating model of class model
    selected_model = model
    
    # Perform Grid Search Cross Validation
    grid_search = GridSearchCV(selected_model, param_grid=param_grid, measures=['rmse'], cv=3, n_jobs=-1)

    # Fit the grid search
    grid_search.fit(data)

    # Get the best score and best parameters
    best_params = grid_search.best_params['rmse']
    best_score = grid_search.best_score
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    
    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the best params in .pkl file
    for file, filename in zip([best_params], ['best_params']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.pkl')
        joblib.dump(best_params, output_filepath)
            
            
def get_model_and_params(model_name, param_grid):
    """
    Returns the model and parameter grid for a given model name.
    """
    model_name = model_name.lower()
    if model_name in param_grid:
        model_config = param_grid[model_name]
        model_class = MODEL_MAPPING.get(model_name)
        
        if not model_class:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAPPING.")
        
        param_grid = model_config['param_grid']  # Extract the parameter grid
        
        return model_class, param_grid
    else:
        raise ValueError(f"Model '{model_name}' not found in param_grid.")
    
    
            
       
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