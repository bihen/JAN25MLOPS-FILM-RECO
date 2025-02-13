import requests
import pytest
import requests
import jwt
import datetime

# The URL of the login and prediction endpoints
LOGIN_URL = "http://127.0.0.1:3000/login"
PREDICT_URL = "http://127.0.0.1:3000/v1/models/admission/predict"

# Données de connexion
VALID_CREDENTIALS = {
    "username": "user123",
    "password": "password123"
}

INVALID_CREDENTIALS = {
    "username": "user123",
    "password": "password2345"
}

# Sample input data for prediction
VALID_DATA = {
    "UserID" : 1
}
INVALID_DATA = {
    "UserID" : "test"
}

@pytest.fixture
def get_valid_token():
    """Get a valid JWT token using login API"""
    response = requests.post(LOGIN_URL, json=VALID_CREDENTIALS)
    assert response.status_code == 200
    return response.json().get("token")


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


def test_prediction_valid_input(get_valid_token):
    """Verify that the API returns a valid prediction for correct input data"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}
    response = requests.post(PREDICT_URL, json=VALID_DATA, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_prediction_invalid_input(get_valid_token):
    """Verify that the API returns an error for invalid input data"""
    headers = {"Authorization": f"Bearer {get_valid_token}"}
    response = requests.post(PREDICT_URL, json=INVALID_DATA, headers=headers)
    assert response.status_code >= 400