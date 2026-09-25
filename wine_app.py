
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("wine_quality_scaler.pkl")

st.title("Wine Quality Prediction")

st.write("Enter the wine properties to predict whether the wine is Good or Bad.")

fixed_acidity = st.number_input("Fixed Acidity", value=8.3)
volatile_acidity = st.number_input("Volatile Acidity", value=0.53)
citric_acid = st.number_input("Citric Acid", value=0.27)
residual_sugar = st.number_input("Residual Sugar", value=2.5)
chlorides = st.number_input("Chlorides", value=0.087)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.9)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=46.5)
density = st.number_input("Density", value=0.9967)
pH = st.number_input("pH", value=3.31)
sulphates = st.number_input("Sulphates", value=0.66)
alcohol = st.number_input("Alcohol", value=10.4)

if st.button("Predict Quality"):

    input_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [pH],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("The wine is predicted to be GOOD.")
    else:
        st.error("The wine is predicted to be BAD.")
