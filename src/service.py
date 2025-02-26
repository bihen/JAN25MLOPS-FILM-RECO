import numpy as np
import pandas as pd
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from bentoml import Runnable
from surprise import SVD, SVDpp, SlopeOne, KNNWithMeans, KNNBaseline
import jwt
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import joblib
import warnings
import warnings
import logging
import pickle
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
from starlette.responses import StreamingResponse
import asyncio
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter("ignore", category=DeprecationWarning)
logging.getLogger("bentoml").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
DATA_FOLDER = BASE_DIR / "data" / "raw"
OUTPUT_FOLDER = BASE_DIR / "models" 
CONFIG_FOLDER = BASE_DIR / "config"
MODEL_FOLDER = BASE_DIR / "models"

# Create Prometheus metrics
API_ACCESS_COUNTER = Counter(
    "api_access_count", "Number of times the API has been accessed"
)
TOTAL_RATINGS_COUNTER = Counter(
    'total_ratings_sum', 
    'Cumulative sum of predicted ratings across all API accesses',  
)
RESPONSE_TIME_HISTOGRAM = Histogram(
    "api_response_duration_seconds",
    "API response duration in seconds",
    buckets=[1, 5, 10, 20, 30, 60]
)
AVERAGE_RATING_GAUGE = Gauge("average_rating", "Average rating returned by the API")


# Mapping for model names to . classes
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

def load_json_config(json_file):
    """
    Load the json config
    """
    with open(os.path.join(CONFIG_FOLDER, json_file)) as f:
        config = json.load(f)
    return config

SECRETS_FILE = "secrets.json"
USERS_FILE = "users.json"

SECRETS = load_json_config(SECRETS_FILE)
USERS = load_json_config(USERS_FILE)

JWT_SECRET_KEY = SECRETS["JWT_SECRET_KEY"]
JWT_ALGORITHM = SECRETS["JWT_ALGORITHM"]

# Custom Runner to load Surprise models
class SurpriseRunner(Runnable):
    SUPPORTED_RESOURCES = ("cpu",)  
    SUPPORTS_BATCH = False  
    SUPPORTS_CPU_MULTI_THREADING = True
    
    model_name = load_config().get("model_name")
    model_name = model_name.lower()

    # Get the model from the Model Store
    def __init__(self):
        model_ref = bentoml.picklable_model.get(f"mlops_movie_recommender_{self.model_name}:latest")
        model_path = model_ref.path  
        # Load the Surprise model from the pickle file
        with open(os.path.join(model_path, "saved_model.pkl"), "rb") as f:
            self.model = pickle.load(f)  

    @bentoml.Runnable.method(batchable=False)
    def run(self, user_id: int, movie_id: int) -> float:
        """Runs prediction using Surprise's `predict` method."""
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est  
    # Add a 'predict' method for convenience
    def predict(self, user_id: int, movie_id: int) -> float:
        return self.run(user_id, movie_id)
    
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/movierecommender/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return {"detail": "Invalid credentials"}, 401
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class InputModel(BaseModel):
    UserID: int
    
# Define your login model
class LoginModel(BaseModel):
    username: str
    password: str
    
# Load neccessary DFs
def load_data():
    movies_df = pd.read_csv(os.path.join(DATA_FOLDER, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_FOLDER, "ratings.csv"))
    
    user_ids = ratings["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = ratings["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    ratings["user"] = ratings["userId"].map(user2user_encoded)
    ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)
    movies_df["movie"] = movies_df["movieId"].map(movie2movie_encoded)
    movies_df["genres"] = movies_df["genres"].str.split('|')
    return movies_df, ratings


# Get current model type from config
model_name = load_config().get("model_name")
model_name = model_name.lower()

# Create a runner with previously created Runner Class
movie_runner = bentoml.Runner(SurpriseRunner)

# Create a service API
movie_service = bentoml.Service("mlops_movie_recommender", runners=[movie_runner])

# Add the JWTAuthMiddleware to the service
movie_service.add_asgi_middleware(JWTAuthMiddleware)

#Load neccessary data
movies_df, ratings = load_data()

@movie_service.api(route="/healthz", input=JSON(), output=JSON())
async def health_check(ctx: bentoml.Context) -> dict:
    """
    Health check endpoint to verify that the API is running and the model is loaded.
    """
    try:
        # Ensure the model is loaded
        if movie_runner is None or movie_runner.model is None:
            return {"status": "error", "message": "Model is not loaded"}, 500
        
        return {"status": "ok", "message": "Service is running"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# Create an API endpoint for the service (login)
@movie_service.api(input=JSON(pydantic_model=LoginModel), output=JSON())
def login(credentials: LoginModel, ctx: bentoml.Context) -> dict:
    username = credentials.username
    password = credentials.password

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        ctx.response.status_code = 401
        return {"detail": "Invalid credentials"}

    
# Create an API endpoint for the service
@movie_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='v1/models/movierecommender/predict'
)
async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    start_time = time.time()  # Start measuring time
    # Track number of API accesses
    API_ACCESS_COUNTER.inc()
    
    print(f"Received input: {input_data}")
    request = ctx.request
    user_id = input_data.UserID
    
    global ratings, movies_df
    
    user = request.state.user if hasattr(request.state, 'user') else None
    
    # Filter ratings for the given User ID
    user_data = ratings[ratings['userId'] == user_id]  
    top_movies = user_data.sort_values(by="rating", ascending=False)
    top_movies = top_movies.merge(movies_df, on = "movieId")
    
    # Get the list of movies not yet watched by the user
    movies_not_watched = movies_df[~movies_df["movieId"].isin(user_data.movieId.values)]
    movies_not_watched["predicted_score"] = 0.0

    # Gather all movie ids and make predictions in batches
    movie_ids_to_predict = movies_not_watched["movieId"].values.tolist()
    
    # Perform batch prediction (make async requests to run predictions for all movie IDs)
    predictions = await asyncio.gather(*[movie_runner.async_run(int(user_id), int(movie_id)) for movie_id in movie_ids_to_predict])

    # Assign the predicted ratings to the DataFrame
    movies_not_watched["predicted_score"] = predictions
    
    # Sort and get top 10 recommendations
    recommendations = movies_not_watched.sort_values(by="predicted_score", ascending=False).head(10)
    
    # Calculate average rating
    average_rating = recommendations["predicted_score"].mean()
    TOTAL_RATINGS_COUNTER.inc(average_rating)
    overall_average = TOTAL_RATINGS_COUNTER._value.get() / API_ACCESS_COUNTER._value.get()

    AVERAGE_RATING_GAUGE.set(overall_average)
    
    response_duration = time.time() - start_time
    RESPONSE_TIME_HISTOGRAM.observe(response_duration)
    
    return {
        "prediction": recommendations["title"].tolist(),  # Returning list of movie titles
        "userId": user_id,
        "user": user
    }    
    
    
# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token