import streamlit as st
import requests
from util import get_meta_data
import pandas as pd

st.title("Property Price Predictor")
st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="centered")

st.markdown('------------------------------------')

st.write("Enter Property Details to get an Estimated Price")

@st.cache_data
def get_location_names():
    url = "http://127.0.0.1:5000/get_loc_names"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['locations']
    else:
        return []
    
locations = get_location_names()

location = st.selectbox("Select Location", locations)
total_sqft = st.number_input("Square Foot Area", min_value=500, max_value=10000, step=50)
bhk = st.slider("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
bath = st.slider("Number of Bathrooms", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    url = "http://127.0.0.1:5000/predict"
    data = {
        "location": location,
        "total_sqft": total_sqft,
        "bhk": bhk,
        "bath": bath
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"🏠 Estimated Price: {result['estimated_price']:,.2f} Million")
    else:
        st.error("Error: Could not fetch prediction from backend")