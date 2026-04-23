import os
import pickle
import streamlit as st
import pandas as pd

# Ensure the model path is correct
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")

# Load the trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is saved correctly.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Harris Titanic App")
st.write("Predict if a passenger would survive the Titanic disaster.")

# Input fields for the features
st.sidebar.header("Passenger Features")

pclass = st.sidebar.selectbox("Pclass (Passenger Class)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=30.0)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[pclass, sex, age, fare]],
                          columns=["Pclass", "Sex", "Age", "Fare"])

# Make prediction
if st.sidebar.button("Predict Survival"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1] # Probability of survival

        st.subheader("Prediction")
        if prediction[0] == 1:
            st.success(f"The passenger is predicted to **survive**!")
            st.write(f"Probability of survival: **{prediction_proba[0]:.2f}**")
        else:
            st.error(f"The passenger is predicted to **not survive**.")
            st.write(f"Probability of survival: **{prediction_proba[0]:.2f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
