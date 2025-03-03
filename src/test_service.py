import requests
import pytest
import requests
import jwt
import datetime
import os
from pathlib import Path
import json
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

# The URL of the login and prediction endpoints
LOGIN_URL = "http://127.0.0.1:3000/login"
PREDICT_URL = "http://127.0.0.1:3000/v1/models/movierecommender/predict"
HEALTHCHECK_URL = "http://127.0.0.1:3000/healthz"

def load_json_config(json_file):
    """
    Load the json config
    """
    with open(os.path.join(CONFIG_FOLDER, json_file)) as f:
        config = json.load(f)
    return config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FOLDER = BASE_DIR / "config"

USERS_FILE = "users.json"
USERS = load_json_config(USERS_FILE)

test_user = list(USERS.items())[0]

VALID_CREDENTIALS = {
    "username": test_user[0],  # username
    "password": test_user[1]   # password
}

INVALID_CREDENTIALS = {
    "username": "user123",
    "password": "bad_password"
}

# Sample input data for prediction
VALID_DATA = {
    "UserID" : 1
}
INVALID_DATA = {
    "UserID" : "String"
}



@pytest.fixture
def get_valid_token():
    """Get a valid JWT token using login API"""
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS)
    assert response.status_code == 200
    return response.json().get("token")

def test_healthcheck():
    """Verify that the health check works"""
    response = requests.get(HEALTHCHECK_URL)
    assert response.status_code == 200
 
def test_missing_jwt():
    """Verify that authentication fails if the JWT token is missing"""
    response = requests.get(PREDICT_URL)
    assert response.status_code == 401


def test_invalid_jwt():
    """Verify that authentication fails if the JWT token is invalid"""
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.get(PREDICT_URL, headers=headers)
    assert response.status_code == 401


def test_expired_jwt():
    """Verify that authentication fails if the JWT token has expired"""
    expired_token = jwt.encode(
        {"exp": datetime.datetime.utcnow() - datetime.timedelta(seconds=10)},
        "your_secret_key",  # Replace with actual secret key
        algorithm="HS256"
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = requests.get(PREDICT_URL, headers=headers)
    assert response.status_code == 401


def test_valid_jwt(get_valid_token):
    """Verify that authentication succeeds with a valid JWT token"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}
    response = requests.get(PREDICT_URL, headers=headers)
    assert response.status_code != 401  # Should not be unauthorized


def test_login_success():
    """Verify that the API returns a valid JWT token for correct user credentials"""
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS)
    assert response.status_code == 200
    assert "token" in response.json()

    
def test_login_failure():
    """Verify that the API returns a 401 error for incorrect user credentials"""
    response = requests.post(LOGIN_URL, json=INVALID_CREDENTIALS)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    assert response.status_code == 401


def test_prediction_missing_jwt():
    """Verify that the API returns a 401 error if the JWT token is missing"""
    response = requests.post(PREDICT_URL, json=VALID_DATA)
    assert response.status_code == 401

def test_prediction_invalid_input(get_valid_token):
    """Verify that the API returns an error for invalid input data"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}
    response = requests.post(PREDICT_URL, json=INVALID_DATA, headers=headers)
    assert response.status_code >= 400
    
def test_prediction_valid_input(get_valid_token):
    """Verify that the API returns a valid prediction for correct input data"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}
    response = requests.post(PREDICT_URL, json=VALID_DATA, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()
    
def test_parallel_predictions_2(get_valid_token):
    """Test multiple API inputs in parallel"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}

    test_cases = [
        ({"UserID": 1}, 200),  
        ({"UserID": 2}, 200),  
    ]

    def send_request(input_data, expected_status):
        response = requests.post(PREDICT_URL, json=input_data, headers=headers)
        assert response.status_code == expected_status
        if expected_status == 200:
            assert "prediction" in response.json()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request, data, status) for data, status in test_cases]
        concurrent.futures.wait(futures)  # Ensure all finish before proceeding
        
def test_parallel_predictions_4(get_valid_token):
    """Test multiple API inputs in parallel"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}

    test_cases = [
        ({"UserID": 1}, 200),  
        ({"UserID": 2}, 200),  
        ({"UserID": 5}, 200),  
        ({"UserID": 6}, 200),  
    ]

    def send_request(input_data, expected_status):
        response = requests.post(PREDICT_URL, json=input_data, headers=headers)
        assert response.status_code == expected_status
        if expected_status == 200:
            assert "prediction" in response.json()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request, data, status) for data, status in test_cases]
        concurrent.futures.wait(futures)  # Ensure all finish before proceeding