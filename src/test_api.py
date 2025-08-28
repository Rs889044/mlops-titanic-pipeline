# src/test_api.py

import requests
import json

# The URL of our deployed API endpoint
URL = "http://127.0.0.1:8000/predict"

# Sample passenger data (e.g., Rose from the movie - she survived)
# This data structure must match our Pydantic model in src/app.py
rose_data = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 17.0,
    "SibSp": 1,
    "Parch": 2,
    "Fare": 100.0,
    "Embarked": "S",
    "IsAlone": 0
}

# Send a POST request with the sample data [cite: 213]
print("Sending sample request to the API...")
response = requests.post(URL, json=rose_data)

# Check the response [cite: 214]
if response.status_code == 200:
    print("API call successful!")
    print("Response from server:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"API call failed with status code: {response.status_code}")
    print("Response from server:")
    print(response.text)