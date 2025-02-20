import dagshub.auth
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from surprise import accuracy
import joblib
import os
import random

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "raw"
MODEL_FOLDER = BASE_DIR / "models"
METRICS_FOLDER = BASE_DIR / "metrics"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

# Main function
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading the trained model...")
    model = load_trained_model()

    logger.info("Creating predictions...")
    create_predictions(model)
    logger.info(f"Saved predictions.csv")

        
# Evaluate the model
def create_predictions(model):
    """
    Create a prediction dataset for drift analysis
    """
    movies_df, ratings, user_ids = load_data()
    movies_df = pd.DataFrame(movies_df)
    predicted_df = pd.DataFrame()
    users = user_ids[:10]
    additional_users = random.sample(user_ids[10:], 5)
    users.extend(additional_users) 
    for user in users:
        user_data = ratings[ratings['userId'] == user]  
        movies_df["user"] = user
        movies_df["predicted_score"] = 0.0
        predicted_ratings = [model.predict(user, movie_id).est for movie_id in movies_df["movieId"]]
        movies_df["predicted_score"] = predicted_ratings
        for index, movie in movies_df.iterrows():
            predicted_rating = model.predict(user, movie["movieId"]).est
            movies_df.at[index, "predicted_score"] = predicted_rating
        predicted_df = pd.concat([predicted_df, movies_df])
    predicted_df = predicted_df[["user", "movieId", "predicted_score"]]
    output_filepath = os.path.join(OUTPUT_FOLDER, 'predictions.csv')

    with open(output_filepath, "w", encoding="utf-8") as output_filepath:
        predicted_df.to_csv(output_filepath, index= False)
       
def load_data():
    """
    Load neccesary data
    """
    movies_df = pd.read_csv(os.path.join(INPUT_FOLDER, "movies.csv"))
    ratings = pd.read_csv(os.path.join(INPUT_FOLDER, "ratings.csv"))
    
    user_ids = ratings["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = ratings["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    ratings["user"] = ratings["userId"].map(user2user_encoded)
    ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)
    movies_df["movie"] = movies_df["movieId"].map(movie2movie_encoded)
    movies_df["genres"] = movies_df["genres"].str.split('|')
    return movies_df, ratings, user_ids


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
          
def check_existing_folder(folder_path):
    '''Check if a folder already exists.'''
    if os.path.exists(folder_path) == False :
        return True
    else:
        return False
if __name__ == '__main__':
    main()
