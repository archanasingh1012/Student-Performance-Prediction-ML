
import streamlit as st
import pandas as pd
import joblib

from config import MODEL_DIR, TARGET, DROP_FROM_FEATURES
from data_preprocessing import load_dataset


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Academic Risk Analytics",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

h1 {
    color: #1E3A8A;
}

h2, h3 {
    color: #1E40AF;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Load Dataset
# --------------------------------------------------

@st.cache_data
def get_data():

    result = load_dataset()

    if isinstance(result, tuple):
        df = next(
            item for item in result
            if isinstance(item, pd.DataFrame)
        )
    else:
        df = result

    return df


try:
    df = get_data()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# --------------------------------------------------
# 4. Load Trained Model
# --------------------------------------------------

@st.cache_resource
def get_model():

    model_path = MODEL_DIR / "best_model.joblib"

    return joblib.load(model_path)


try:
    model = get_model()

except Exception as e:
    st.error(f"Error loading trained model: {e}")
    st.stop()


# --------------------------------------------------
# 5. Application Header
# --------------------------------------------------

st.markdown("""
<div style="text-align:center; padding:15px;">

<h1>🎓 Student Academic Risk Analytics</h1>

<p style="font-size:18px; color:gray;">
An intelligent machine learning application for
predicting student academic risk and supporting
data-driven educational decision-making.
</p>

<p style="font-size:15px;">
<strong>M.Tech Data Science and Analytics</strong>
</p>

</div>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# 6. Project Information
# --------------------------------------------------

st.subheader("📌 Machine Learning Research Application")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Domain:** Educational Data Mining")

with col2:
    st.info("**Application:** Academic Risk Prediction")

with col3:
    st.info("**Model:** Random Forest")


st.markdown("""
**Research Objective**

To develop a machine learning-based system that identifies
students who may be at academic risk using their academic
and related attributes.
""")


# --------------------------------------------------
# 7. Dataset Overview
# --------------------------------------------------

st.divider()

st.subheader("📊 Dataset Overview")

total_records = len(df)
total_columns = len(df.columns)
missing_values = int(df.isnull().sum().sum())

excluded = set(DROP_FROM_FEATURES)

input_features = [
    col for col in df.columns
    if col not in excluded
]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{total_records:,}")
c2.metric("Input Features", len(input_features))
c3.metric("Dataset Columns", total_columns)
c4.metric("Missing Values", missing_values)

with st.expander("View Dataset Information"):

    st.write("Dataset Preview")

    st.dataframe(df.head(10), width="stretch")

    st.write("Dataset Columns")

    st.write(list(df.columns))

    st.write("Dataset Statistics")

    st.dataframe(df.describe(include="all"), width="stretch")


# --------------------------------------------------
# 8. Student Risk Assessment
# --------------------------------------------------

st.divider()

st.subheader("🎯 Student Risk Assessment")

st.write(
    "Enter the student's information below to generate "
    "an academic risk prediction."
)

st.markdown("### Student Profile")


# Use the model's original feature order when available
if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = input_features

# Keep only features available in the dataset
feature_columns = [
    col for col in feature_columns
    if col in df.columns
]

student_data = {}

with st.form("student_assessment_form"):

    columns = st.columns(2)

    for i, feature in enumerate(feature_columns):

        with columns[i % 2]:

            values = df[feature].dropna()
            label = feature.replace("_", " ").title()

            if len(values) == 0:
                student_data[feature] = None
                continue

            if pd.api.types.is_numeric_dtype(df[feature]):

                minimum = float(values.min())
                maximum = float(values.max())
                default = float(values.median())

                if minimum == maximum:
                    student_data[feature] = minimum

                elif minimum >= 0 and maximum <= 1:

                    student_data[feature] = st.slider(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.01
                    )

                elif pd.api.types.is_integer_dtype(df[feature]):

                    student_data[feature] = st.number_input(
                        label,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1
                    )

                else:

                    student_data[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.1
                    )

            else:

                options = sorted(
                    values.astype(str).unique().tolist()
                )

                student_data[feature] = st.selectbox(
                    label,
                    options
                )

    predict_button = st.form_submit_button(
        "🔍 Predict Student Risk",
        width="stretch"
    )


# --------------------------------------------------
# 9. Prediction
# --------------------------------------------------

if predict_button:

    try:

        input_df = pd.DataFrame(
            [student_data],
            columns=feature_columns
        )

        prediction = model.predict(input_df)[0]

        st.divider()

        st.subheader("📋 Prediction Results")

        st.write("Predicted Academic Risk Level")

        risk = str(prediction).strip().lower()

        if "high" in risk:

            st.error(
                f"⚠️ High Academic Risk: {prediction}"
            )

            st.warning(
                "This student may require additional academic "
                "support and monitoring."
            )

        elif "medium" in risk or "moderate" in risk:

            st.warning(
                f"⚠️ Moderate Academic Risk: {prediction}"
            )

            st.info(
                "Regular monitoring and academic guidance "
                "are recommended."
            )

        elif "low" in risk:

            st.success(
                f"✅ Low Academic Risk: {prediction}"
            )

            st.info(
                "The student currently shows a lower "
                "predicted academic risk."
            )

        else:

            st.info(f"Predicted Risk Level: {prediction}")

        # Prediction probabilities
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            classes = model.classes_

            st.markdown("### Prediction Confidence")

            probability_df = pd.DataFrame({
                "Risk Level": classes,
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            })

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True
            )

            st.bar_chart(
                probability_df.set_index("Risk Level")
            )

        with st.expander("View Student Input Information"):

            st.dataframe(input_df, width="stretch")

    except Exception as e:

        st.error(f"Prediction error: {e}")


# --------------------------------------------------
# 10. Footer
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">

<h3>Student Academic Risk Analytics</h3>

<p>
Machine Learning | Educational Data Mining | Predictive Analytics
</p>

<p>
M.Tech Data Science and Analytics
</p>

</div>
""", unsafe_allow_html=True)
