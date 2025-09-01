import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and feature names
model, feature_columns = joblib.load("house_price_pipeline4.pkl")

# Automatically detect location columns (one-hot encoded)
locations = [col for col in feature_columns if col not in ['total_sqft', 'BHK', 'bath']]

st.title("🏠 House Price Prediction")

# User inputs
area = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=50)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bedrooms = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
location = st.selectbox("Select Location", locations)

# Prepare input
x = np.zeros(len(feature_columns))
x[feature_columns.index("total_sqft")] = area
x[feature_columns.index("BHK")] = bedrooms
x[feature_columns.index("bath")] = bathrooms

if location in feature_columns:
    x[feature_columns.index(location)] = 1

# Prediction
if st.button("Predict Price"):
    predicted_price = model.predict([x])[0]
    st.success(f"Estimated Price: ₹ {predicted_price:,.2f}")

