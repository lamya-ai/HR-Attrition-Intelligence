
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "HR_capstone_dataset.csv"
MODEL_IMAGES = ROOT / "assets" / "model_results"
EDA_IMAGES = ROOT / "assets" / "exploration"

# Local API by default; cloud deployment can override this.
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

NAVY = "#17324D"
TEAL = "#275D62"
MUTED = "#64748B"
BACKGROUND = "#F7F9FC"


st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. DESIGN
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    h1, h2, h3 {
        color: #17324D;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] {
        background-color: #17324D;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #275D62;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 22px;
        font-weight: bold;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #17324D;
        color: white;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
    }

    .section-note {
        color: #64748B;
        font-size: 13px;
    }

    .result-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.title("HR INTELLIGENCE")

st.sidebar.caption("PEOPLE ANALYTICS PLATFORM")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Project Overview",
        "Data Exploration",
        "Model Performance",
        "Interactive Testing"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("""
**PROJECT INFORMATION**

Employee Attrition Prediction

Machine Learning: Random Forest

Backend: FastAPI

Frontend: Streamlit
""")

st.sidebar.divider()

st.sidebar.caption("HR Attrition Intelligence | 2026")


# =========================================================
# 5. HEADER
# =========================================================

st.title("HR Attrition Intelligence")

st.markdown(
    '<p class="subtitle">'
    'An AI-powered analytics platform for understanding '
    'and predicting employee attrition.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PAGE 1: PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.subheader("Project Overview")

    st.write(
        "This project applies data analytics and machine learning "
        "to identify workforce attrition patterns and estimate "
        "employee attrition risk."
    )

    total = len(df)

    left = int((df["left"] == 1).sum())
    stayed = int((df["left"] == 0).sum())

    attrition_rate = left / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TOTAL EMPLOYEES", f"{total:,}")
    c2.metric("EMPLOYEES STAYED", f"{stayed:,}")
    c3.metric("EMPLOYEES LEFT", f"{left:,}")
    c4.metric("ATTRITION RATE", f"{attrition_rate:.2f}%")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("Business Problem")

        st.write(
            "Employee attrition can affect organizational "
            "continuity, productivity, recruitment costs, "
            "and workforce planning."
        )

        st.subheader("Project Objectives")

        st.markdown("""
        - Explore employee workforce data.
        - Identify patterns associated with attrition.
        - Apply a Random Forest classification model.
        - Estimate attrition probability for new employee records.
        - Support evidence-based HR decisions.
        """)

    with right_col:

        st.subheader("Dataset Information")

        st.write(f"**Records:** {total:,}")
        st.write(f"**Features:** {df.shape[1]}")
        st.write("**Target:** `left`")
        st.write("**0:** Employee stayed")
        st.write("**1:** Employee left")

        st.subheader("Technology Stack")

        st.markdown("""
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Random Forest
        - FastAPI
        - Streamlit
        - Plotly
        """)

    st.divider()

    st.subheader("Target Distribution")

    target_counts = (
        df["left"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "Stayed", 1: "Left"})
        .reset_index()
    )

    target_counts.columns = ["Employee Status", "Employees"]

    fig = px.pie(
        target_counts,
        names="Employee Status",
        values="Employees",
        hole=0.55,
        color="Employee Status",
        color_discrete_map={
            "Stayed": "#8FA9A3",
            "Left": "#275D62"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE 2: DATA EXPLORATION
# =========================================================

elif page == "Data Exploration":

    st.subheader("Data Exploration")

    st.write(
        "Explore workforce characteristics and their relationship "
        "with employee attrition."
    )

    with st.expander("View Dataset Preview"):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    st.divider()

    # Attrition by department

    st.subheader("Attrition Rate by Department")

    dept_rate = (
        df.groupby("Department")["left"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
        .sort_values("Attrition Rate", ascending=True)
    )

    fig = px.bar(
        dept_rate,
        x="Attrition Rate",
        y="Department",
        orientation="h",
        color="Attrition Rate",
        color_continuous_scale=["#8FA9A3", "#275D62"],
        text_auto=".1f"
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Attrition by salary

    st.subheader("Attrition by Salary Level")

    salary_rate = (
        df.groupby("salary")["left"]
        .mean()
        .mul(100)
        .reset_index(name="Attrition Rate")
    )

    salary_order = ["low", "medium", "high"]

    salary_rate["salary"] = pd.Categorical(
        salary_rate["salary"],
        categories=salary_order,
        ordered=True
    )

    salary_rate = salary_rate.sort_values("salary")

    fig = px.bar(
        salary_rate,
        x="salary",
        y="Attrition Rate",
        color="salary",
        text_auto=".1f",
        color_discrete_sequence=[
            "#8FA9A3", "#527D7D", "#275D62"
        ]
    )

    fig.update_layout(
        height=350,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Satisfaction comparison

    st.subheader("Satisfaction Level vs Employee Status")

    fig = px.histogram(
        df,
        x="satisfaction_level",
        color="left",
        barmode="overlay",
        histnorm="percent",
        opacity=0.7,
        color_discrete_map={
            0: "#8FA9A3",
            1: "#275D62"
        },
        labels={
            "left": "Employee Status",
            "satisfaction_level": "Satisfaction Level"
        }
    )

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Saved exploration images

    image_files = []

    if EDA_IMAGES.exists():
        image_files.extend(
            sorted(EDA_IMAGES.glob("*.png"))
        )

    if MODEL_IMAGES.exists():
        image_files.extend([
            p for p in sorted(MODEL_IMAGES.glob("*.png"))
            if "distribution" in p.name.lower()
        ])

    if image_files:

        st.divider()
        st.subheader("Exploration Visualizations")

        for start in range(0, len(image_files), 2):

            cols = st.columns(2)

            for col, image_path in zip(
                cols, image_files[start:start + 2]
            ):

                with col:
                    st.image(
                        str(image_path),
                        caption=image_path.stem.replace("_", " ").title(),
                        use_container_width=True
                    )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader("Random Forest Model Performance")

    st.write(
        "The selected Random Forest model was evaluated "
        "using a held-out test dataset."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("ACCURACY", "97.96%")
    c2.metric("F1 SCORE", "93.76%")
    c3.metric("ROC-AUC", "98.23%")

    st.caption(
        "Evaluation results from the local training run. "
        "Training data: 9,592 records | Test data: 2,399 records."
    )

    st.divider()

    st.subheader("Model Evaluation Visualizations")

    if MODEL_IMAGES.exists():

        image_files = sorted(MODEL_IMAGES.glob("*.png"))

        if image_files:

            for start in range(0, len(image_files), 2):

                cols = st.columns(2)

                for col, image_path in zip(
                    cols, image_files[start:start + 2]
                ):

                    with col:

                        st.image(
                            str(image_path),
                            caption=image_path.stem.replace(
                                "_", " "
                            ).title(),
                            use_container_width=True
                        )

        else:
            st.info(
                "No model evaluation images were found "
                "in assets/model_results."
            )

    else:
        st.info("Model results folder was not found.")

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Number of estimators:** 100")
    st.write("**Maximum depth:** 12")
    st.write("**Class weighting:** Balanced")
    st.write("**Validation strategy:** Stratified train-test split")
    st.write("**Duplicate handling:** Exact duplicates removed")

    st.info(
        "Model performance is measured on the held-out test set. "
        "It does not guarantee the same performance on future data."
    )


# =========================================================
# PAGE 4: INTERACTIVE TESTING
# =========================================================

elif page == "Interactive Testing":

    st.subheader("Interactive Employee Attrition Prediction")

    st.write(
        "Enter employee characteristics below to estimate "
        "the likelihood of employee attrition."
    )

    st.info(
        "This prediction is intended for workforce analysis "
        "and decision support, not as the sole basis for HR decisions."
    )

    st.divider()

    st.subheader("Employee Information")

    with st.form("employee_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            satisfaction = st.slider(
                "Satisfaction Level",
                0.0, 1.0, 0.50, 0.01
            )

            evaluation = st.slider(
                "Last Evaluation",
                0.0, 1.0, 0.70, 0.01
            )

            projects = st.number_input(
                "Number of Projects",
                min_value=1,
                max_value=20,
                value=3
            )

            monthly_hours = st.number_input(
                "Average Monthly Hours",
                min_value=0,
                max_value=500,
                value=160
            )

            years = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=50,
                value=3
            )

        with col2:

            departments = sorted(
                df["Department"].dropna().astype(str).unique()
            )

            salaries = ["low", "medium", "high"]

            department = st.selectbox(
                "Department",
                departments
            )

            salary = st.selectbox(
                "Salary Level",
                salaries
            )

            accident = st.selectbox(
                "Work Accident",
                [0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                )
            )

            promotion = st.selectbox(
                "Promotion in Last 5 Years",
                [0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                )
            )

        submitted = st.form_submit_button(
            "Predict Employee Attrition",
            use_container_width=True
        )

    if submitted:

        employee = {
            "satisfaction_level": satisfaction,
            "last_evaluation": evaluation,
            "number_project": int(projects),
            "average_montly_hours": int(monthly_hours),
            "time_spend_company": int(years),
            "Work_accident": int(accident),
            "promotion_last_5years": int(promotion),
            "Department": department,
            "salary": salary
        }

        with st.spinner("Analyzing employee data..."):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    json=employee,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                prediction = result["prediction"]
                probability = result["attrition_probability"]

                st.divider()
                st.subheader("Prediction Results")

                result_col, probability_col = st.columns(2)

                with result_col:

                    if prediction == 1:

                        st.error("LIKELY TO LEAVE")

                    else:

                        st.success("LIKELY TO STAY")

                    st.write(
                        result.get(
                            "prediction_label",
                            "Prediction completed"
                        )
                    )

                with probability_col:

                    st.metric(
                        "ATTRITION PROBABILITY",
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        max(0.0, min(1.0, probability))
                    )

                st.caption(
                    f"Model: {result.get('model', 'Random Forest')}"
                )

                st.success(
                    "Prediction completed successfully."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "The prediction API is unavailable. "
                    "Check the API_URL configuration and ensure "
                    "the FastAPI service is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The prediction request timed out. "
                    "The backend may be starting up. Please try again."
                )

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"The API returned an error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:15px;">
        <b>HR ATTRITION INTELLIGENCE</b><br>
        People Analytics | Machine Learning | Decision Support<br>
        <small>Predictions should support, not replace,
        professional HR judgment.</small>
    </div>
    """,
    unsafe_allow_html=True
)
