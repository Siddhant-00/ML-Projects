import streamlit as st
import joblib
import numpy as np 

# Load the model
model = joblib.load("model.pkl")

# App title
st.title("House Price Prediction App")

# Description
st.write("This app uses ML for predicting house prices based on the given features of the house. Enter the inputs below and click the **Predict** button.")

st.divider()

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=2)
livingarea = st.number_input("Living Area (sq. ft.)", min_value=0, value=2000)
condition = st.number_input("Condition (1-5)", min_value=0, max_value=5, value=3)
numberofschools = st.number_input("Number of Schools Nearby", min_value=0, value=2)

st.divider()

# Collect inputs into a feature vector
X = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]

predict_button = st.button("Predict")

if predict_button:
    X_array = np.array(X)
    prediction = model.predict(X_array)
    st.write(f"Price Prediction: ${prediction[0]:,.2f}")
else:
    st.write("Please use the Predict button after entering values.")
