
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("models/house_price_model.pkl")

FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus_semi-furnished",
    "furnishingstatus_unfurnished",
    "total_rooms",
    "area_per_bedroom",
    "facility_count"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained housing model not found")

    return joblib.load(MODEL_PATH)


@app.get("/")
@app.get("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "housing-price-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        model = load_model()
        predicted_price = float(model.predict(sample)[0])

        return jsonify({
            "predicted_price": round(predicted_price, 2),
            "currency": "dataset_units"
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
