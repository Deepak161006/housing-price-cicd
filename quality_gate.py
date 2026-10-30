
import json
from pathlib import Path

METRICS_PATH = Path("models/metrics.json")
MIN_R2_SCORE = 0.50

def check_model_quality():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Model metrics file not found")

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    r2 = metrics["r2_score"]
    mae = metrics["mae"]
    rmse = metrics["rmse"]

    print("MODEL QUALITY CHECK")
    print("-------------------")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    if r2 < MIN_R2_SCORE:
        raise ValueError(
            f"Model rejected: R2 score {r2:.4f} "
            f"is below threshold {MIN_R2_SCORE}"
        )

    print("QUALITY GATE PASSED")

if __name__ == "__main__":
    check_model_quality()
    