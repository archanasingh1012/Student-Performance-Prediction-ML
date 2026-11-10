
import streamlit as st
import pandas as pd
import joblib

from config import MODEL_DIR, TARGET, DROP_FROM_FEATURES
from data_preprocessing import load_dataset

# Page settings
st.set_page_config(
    page_title="Student Academic Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# Application heading
st.title("🎓 Student Academic Risk Prediction")
st.write(
    "An Explainable Machine Learning Application "
    "for Identifying Students at Academic Risk."
)

st.divider()

# Load dataset and trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model.joblib")


@st.cache_data
def load_data():
    return load_dataset()


try:
    df = load_data()
    model = load_model()
    model_name = (MODEL_DIR / "best_model_name.txt").read_text().strip()

except Exception as e:
    st.error(f"Unable to load dataset or model: {e}")
    st.stop()


# Prepare feature columns
drop_existing = [c for c in DROP_FROM_FEATURES if c in df.columns]
feature_data = df.drop(columns=drop_existing)

st.subheader("Enter Student Information")

st.write("Enter the student's details below to predict academic risk.")

student_input = {}

with st.form("student_prediction_form"):

    for column in feature_data.columns:

        if pd.api.types.is_numeric_dtype(feature_data[column]):

            default_value = feature_data[column].median()

            if pd.isna(default_value):
                default_value = 0.0

            student_input[column] = st.number_input(
                column.replace("_", " ").title(),
                value=float(default_value)
            )

        else:

            options = (
                feature_data[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            options = sorted(options)

            if options:
                student_input[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options
                )
            else:
                student_input[column] = st.text_input(
                    column.replace("_", " ").title()
                )

    submitted = st.form_submit_button("Predict Student Risk")

# Prediction
if submitted:

    input_df = pd.DataFrame(
        [student_input],
        columns=feature_data.columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        if "high" in str(prediction).lower():
            st.error(f"⚠️ Predicted Risk Level: {prediction}")
        else:
            st.success(f"✅ Predicted Risk Level: {prediction}")

        st.write(f"**Selected Model:** {model_name}")

        # Prediction probabilities
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            if hasattr(model, "named_steps"):
                classes = model.named_steps["classifier"].classes_
            else:
                classes = model.classes_

            probability_df = pd.DataFrame({
                "Risk Category": classes,
                "Probability": probabilities
            })

            st.subheader("Prediction Confidence")

            st.bar_chart(
                probability_df.set_index("Risk Category")
            )

            st.dataframe(
                probability_df.style.format(
                    {"Probability": "{:.2%}"}
                ),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("M.Tech Data Science and Analytics | Student Risk Prediction Project")
