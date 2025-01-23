import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging
import os
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "raw"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

def main():
    """ Performs Surprise Train/test split on ratings
    """
    
    logger = logging.getLogger(__name__)
    logger.info('Creating a Surprise Dataset and splitting into Test/Train')
    
    ratings_filepath = os.path.join(INPUT_FOLDER, "ratings.csv")
    output_folderpath = OUTPUT_FOLDER
    
    # Call the main data processing function with the provided file paths
    create_datasets(ratings_filepath, output_folderpath)

def create_datasets(ratings_filepath, output_folderpath):
    
    #--Importing dataset
    ratings = pd.read_csv(ratings_filepath, sep=",")
    
    # Drop Timestamp to fit into Surprise Dataset format
    ratings = ratings.drop('timestamp', axis = 1)

    # Define the reader format for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))

    # Load the ratings dataset into Surprise's Dataset format
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Splitting the data into training and testing (e.g., 80% training, 20% testing)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print("Data has been prepared for collaborative filtering.")

    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([trainset, testset, data], ['trainset', 'testset', 'data']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.data')
        if check_existing_file(output_filepath):
            joblib.dump(file, output_filepath)
            
def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True
    
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if os.path.exists(folder_path) == False :
        while True:
            response = input(f"{os.path.basename(folder_path)} doesn't exists. Do you want to create it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return False
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()
