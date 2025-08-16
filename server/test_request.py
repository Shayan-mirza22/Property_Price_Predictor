import requests

url = "http://127.0.0.1:5000/predict"

data = {
    'total_sqft': 1000,
    'location': '1st phase jp nagar',
    'bath': 2,
    'bhk': 3
}

response = requests.post(url, data=data)  # <-- sending as form-data
print("Server response:", response.json())
