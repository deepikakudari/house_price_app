import streamlit as st
import joblib
import numpy as np

# Load your model and column structure
model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")

# Streamlit App Title
st.title("🏠 Bengaluru House Price Predictor")

# Input Widgets
location = st.selectbox("Select Location", sorted([col for col in columns if col not in ['total_sqft', 'bath', 'bhk']]))
sqft = st.number_input("Enter Total Square Feet", min_value=300, max_value=10000, step=50)
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5])

# Predict Button
if st.button("Predict Price"):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1
    predicted_price = model.predict([x])[0]
    st.success(f"Estimated House Price: ₹ {predicted_price:.2f} Lakhs")