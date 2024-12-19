import streamlit as st
import pickle
from streamlit_option_menu import option_menu

# Configure app
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🩺")

# Load models
diabetes_model = pickle.load(open("saved_models/diabetes.pkl", "rb"))
heart_model = pickle.load(open("saved_models\heart_disease.pkl", "rb"))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes", "Heart Disease"],
        icons=["activity", "heart"],
        default_index=0,
    )

# Diabetes Prediction
if selected == "Diabetes":
    st.title("Diabetes Prediction")
    pregnancies = st.text_input("Number of Pregnancies")
    glucose = st.text_input("Glucose Level")
    blood_pressure = st.text_input("Blood Pressure")
    skin_thickness = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin Level")
    bmi = st.text_input("BMI")
    pedigree_function = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")
    
    if st.button("Predict Diabetes"):
        try:
            input_data = [[float(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness),
                           float(insulin), float(bmi), float(pedigree_function), float(age)]]
            prediction = diabetes_model.predict(input_data)
            st.success("Diabetic" if prediction[0] == 1 else "Non-Diabetic")
        except ValueError:
            st.error("Please enter valid numeric values")

# Heart Disease Prediction
if selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    age = st.text_input("Age")
    sex = st.text_input("Sex (1 = Male, 0 = Female)")
    chest_pain_type = st.text_input("Chest Pain Type (0-3)")
    resting_bp = st.text_input("Resting Blood Pressure")
    cholesterol = st.text_input("Serum Cholesterol")
    fasting_bs = st.text_input("Fasting Blood Sugar > 120 (1 = True, 0 = False)")
    resting_ecg = st.text_input("Resting ECG Results (0-2)")
    max_hr = st.text_input("Max Heart Rate Achieved")
    exercise_angina = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
    old_peak = st.text_input("ST Depression")
    slope = st.text_input("Slope of ST Segment (0-2)")
    num_major_vessels = st.text_input("Major Vessels (0-3)")
    thalassemia = st.text_input("Thalassemia (1-3)")
    
    if st.button("Predict Heart Disease"):
        try:
            input_data = [[float(age), float(sex), float(chest_pain_type), float(resting_bp), float(cholesterol), 
                           float(fasting_bs), float(resting_ecg), float(max_hr), float(exercise_angina), 
                           float(old_peak), float(slope), float(num_major_vessels), float(thalassemia)]]
            prediction = heart_model.predict(input_data)
            st.success("Heart Disease" if prediction[0] == 1 else "No Heart Disease")
        except ValueError:
            st.error("Please enter valid numeric values")
