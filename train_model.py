
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project paths
DATA_PATH = Path("Housing.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

def train_model():
    print("Loading Housing dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "price" not in df.columns:
        raise ValueError("Target column 'price' is missing.")

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Separate target and features
    X = df.drop(columns=["price"])
    y = pd.to_numeric(df["price"], errors="coerce")

    # Remove invalid target values
    valid = y.notna() & (y > 0)
    X = X.loc[valid].copy()
    y = y.loc[valid]

    # Identify feature types from the actual dataset
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = [
        ("numeric", numeric_pipeline, numeric_columns)
    ]

    if categorical_columns:
        from sklearn.preprocessing import OneHotEncoder

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Complete ML pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training Random Forest Regressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save model and metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "RandomForestRegressor",
        "dataset_rows": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train_model()
