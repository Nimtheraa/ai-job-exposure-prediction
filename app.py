
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing steps
bundle = joblib.load("ai_job_exposure_model.pkl")
model = bundle['model']
preprocessor = bundle['preprocessor']
features = bundle['features']

# Load dataset to get input options and default values
df = pd.read_csv("ai_job_exposure.csv")

st.title("AI-Based Job Exposure Level Prediction")
st.write("Predict the AI exposure level of an occupation using machine learning.")

st.subheader("Enter Occupation Details")

user_input = {}

for feature in features:
    if pd.api.types.is_numeric_dtype(df[feature]):
        default_value = float(df[feature].median())
        min_value = float(df[feature].min())
        max_value = float(df[feature].max())

        user_input[feature] = st.number_input(
            feature.replace('_', ' ').title(),
            min_value=min_value,
            max_value=max_value,
            value=default_value
        )
    else:
        options = sorted(df[feature].dropna().unique().tolist())
        user_input[feature] = st.selectbox(
            feature.replace('_', ' ').title(), options
        )

if st.button("Predict AI Exposure Level"):
    input_df = pd.DataFrame([user_input], columns=features)
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)

    st.success(f"Predicted AI Exposure Level: {prediction[0]}")
