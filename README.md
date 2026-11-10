
# HR Attrition Intelligence

An interactive HR analytics and machine learning application designed to analyze employee attrition patterns and support data-informed workforce decisions.

## 1. Project Overview

Employee attrition can create significant operational and financial challenges for organizations.

This project combines exploratory data analysis, machine learning, and an interactive web application to analyze workforce patterns and estimate employee attrition risk.

The solution integrates a Random Forest classification pipeline, a FastAPI backend, and a Streamlit frontend.

## 2. Key Features

- **Project Overview:** Workforce KPIs and employee attrition distribution.
- **Data Exploration:** Analysis of employee characteristics, departments, salary levels, and satisfaction.
- **Model Performance:** Evaluation metrics and model visualizations.
- **Interactive Testing:** Enter employee information and receive a prediction with an estimated attrition probability.
- **REST API:** FastAPI service for model predictions.

## 3. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, Random Forest |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook, VS Code, Git |

## 4. Model Performance

The Random Forest model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 97.96% |
| F1 Score | 93.76% |
| ROC-AUC | 98.23% |

These metrics represent performance on the evaluation dataset and do not guarantee performance on future employee populations.

## 5. Project Architecture

```text
HR-Attrition-Intelligence/
├── app/
│   ├── backend/
│   │   └── main.py
│   └── frontend/
│       └── streamlit_app.py
├── assets/
│   └── model_results/
├── data/
│   └── HR_capstone_dataset.csv
├── models/
│   └── random_forest_pipeline.joblib
├── HR_Insights.ipynb
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Getting Started

### Prerequisites

- Python
- pip
- Git

### Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd HR-Attrition-Intelligence
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Backend

Open a terminal in the project root:

```bash
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### Run the Frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run app/frontend/streamlit_app.py
```

Open the local URL displayed in your terminal.

## 7. Machine Learning Workflow

1. Data loading and validation.
2. Duplicate-record removal.
3. Feature and target separation.
4. Stratified train-test splitting.
5. Numerical and categorical preprocessing.
6. Random Forest model training.
7. Model evaluation.
8. Saving the complete preprocessing and prediction pipeline using Joblib.

## 8. Responsible AI and HR Use

This project is designed for educational, analytical, and decision-support purposes.

Employee attrition predictions are statistical estimates, not facts about an individual's intentions or future behavior.

Predictions should never be used as the sole basis for hiring, termination, promotion, compensation, or other employment decisions.

Any real-world implementation should include appropriate data protection, bias assessment, model validation, transparency, and meaningful human oversight.

## 9. Dataset

The project uses an HR dataset stored in:

`data/HR_capstone_dataset.csv`

Before redistributing the dataset publicly, verify the original source's license and usage conditions. Do not publish confidential or personally identifiable employee information.

## 10. Project Status

- [x] Exploratory Data Analysis
- [x] Random Forest model training
- [x] Model evaluation
- [x] Model serialization
- [x] FastAPI backend
- [x] Streamlit frontend
- [ ] GitHub public repository
- [ ] Cloud deployment
