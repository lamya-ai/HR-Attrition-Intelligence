
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. LOAD TRAINED MODEL
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "random_forest_pipeline.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="HR Attrition Intelligence API",
    description="Employee attrition prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# 3. DEFINE EMPLOYEE INPUT
# --------------------------------------------------

class EmployeeData(BaseModel):

    satisfaction_level: float = Field(ge=0, le=1)
    last_evaluation: float = Field(ge=0, le=1)

    number_project: int = Field(ge=1)
    average_montly_hours: int = Field(ge=0)
    time_spend_company: int = Field(ge=0)

    Work_accident: int = Field(ge=0, le=1)
    promotion_last_5years: int = Field(ge=0, le=1)

    Department: str
    salary: str


# --------------------------------------------------
# 4. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "HR Attrition Intelligence API is running",
        "model": "Random Forest"
    }


# --------------------------------------------------
# 5. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# 6. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(employee: EmployeeData):

    try:

        employee_data = pd.DataFrame(
            [employee.model_dump()]
        )

        prediction = int(model.predict(employee_data)[0])

        probabilities = model.predict_proba(employee_data)[0]

        classes = list(model.classes_)
        left_index = classes.index(1)

        attrition_probability = float(
            probabilities[left_index]
        )

        return {
            "prediction": prediction,
            "prediction_label": (
                "Likely to Leave"
                if prediction == 1
                else "Likely to Stay"
            ),
            "attrition_probability": round(
                attrition_probability, 4
            ),
            "model": "Random Forest"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
