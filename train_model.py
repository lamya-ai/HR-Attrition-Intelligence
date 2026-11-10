
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# 1. Project paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "HR_capstone_dataset.csv"
MODEL_PATH = ROOT / "models" / "random_forest_pipeline.joblib"

# 2. Load and clean data
df = pd.read_csv(DATA_PATH)
df_clean = df.drop_duplicates().copy()

X = df_clean.drop(columns=["left"])
y = df_clean["left"]

# 3. Identify feature types
binary_features = [
    "Work_accident",
    "promotion_last_5years"
]

ordinal_features = ["salary"]

nominal_features = [
    col for col in X.select_dtypes(
        include=["object", "category"]
    ).columns
    if col not in ordinal_features
]

continuous_features = [
    col for col in X.select_dtypes(
        include="number"
    ).columns
    if col not in binary_features
]

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "scaling",
            StandardScaler(),
            continuous_features
        ),
        (
            "nominal_encoding",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            nominal_features
        ),
        (
            "ordinal_encoding",
            OrdinalEncoder(
                categories=[["low", "medium", "high"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ordinal_features
        )
    ],
    remainder="passthrough"
)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Build the original Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 7. Combine preprocessing and model
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# 8. Train
pipeline.fit(X_train, y_train)

# 9. Evaluate
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print("MODEL TRAINING COMPLETED")
print("=" * 40)
print("Original records:", len(df))
print("Records after duplicate removal:", len(df_clean))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

# 10. Save complete pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("\nSaved model:", MODEL_PATH)
