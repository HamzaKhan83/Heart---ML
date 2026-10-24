import pickle
import os

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# ---------- Label encoders (same values as the notebook, Step 5.1) ----------
SEX = ["Male", "Female"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
MAJOR_VESSELS = ["Zero", "One", "Two", "Three"]
EXERCISE_ANGINA = ["Yes", "No"]

sex_encoder = LabelEncoder().fit(SEX)
chest_pain_encoder = LabelEncoder().fit(CHEST_PAIN)
major_vessels_encoder = LabelEncoder().fit(MAJOR_VESSELS)
exercise_angina_encoder = LabelEncoder().fit(EXERCISE_ANGINA)


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "svc_heart_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- UI ----------
st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning Fundamentals (AIC354) – Assignment 2, Task 2")
st.caption("Model: Support Vector Classifier trained on the UCI Cleveland Heart Disease dataset.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file `svc_heart_model.pkl` not found. Put it in the same folder as app.py.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN)
with col2:
    major_vessels = st.selectbox("Major Vessels", MAJOR_VESSELS)
    exercise_angina = st.selectbox("Exercise Induced Angina", EXERCISE_ANGINA)

if st.button("Predict", type="primary"):
    patient = pd.DataFrame({
        "Sex": [sex_encoder.transform([sex])[0]],
        "ChestPain": [chest_pain_encoder.transform([chest_pain])[0]],
        "MajorVessels": [major_vessels_encoder.transform([major_vessels])[0]],
        "ExerciseAngina": [exercise_angina_encoder.transform([exercise_angina])[0]],
    })

    prediction = model.predict(patient)[0]

    if prediction == 1:
        st.error("Prediction: **HEART DISEASE LIKELY**")
    else:
        st.success("Prediction: **NO HEART DISEASE**")

st.warning("This is a student project and is **not** intended for real medical use.")
