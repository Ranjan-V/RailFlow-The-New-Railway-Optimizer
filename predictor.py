"""
predictor.py
ML-based delay predictor using GradientBoosting.
Generates synthetic training data from your schedule JSON.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from pathlib import Path

MODEL_PATH = Path("delay_model.pkl")


def _generate_training_data(corridor_trains: list, n_samples: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    train_types = ["Raj", "Shatabdi", "SF", "Exp", "Pass", "MEMU", "DEMU"]
    type_recovery = {
        "Raj": 0.5, "Shatabdi": 0.5, "SF": 0.6,
        "Exp": 0.75, "Pass": 0.9, "MEMU": 0.95, "DEMU": 0.95
    }

    for _ in range(n_samples):
        if corridor_trains:
            t = rng.choice(corridor_trains)
            train_type = t.get("type", "Exp")
            n_stops = max(2, len(t["stops"]))
            distance = t.get("distance", 300) or 300
        else:
            train_type = rng.choice(train_types)
            n_stops = int(rng.integers(3, 25))
            distance = int(rng.integers(50, 800))

        hour_of_day = int(rng.integers(0, 24))
        current_delay = int(rng.integers(0, 120))
        stops_remaining = int(rng.integers(1, max(2, n_stops)))
        distance_remaining = int(rng.integers(20, max(21, distance)))
        junction_count = int(rng.integers(0, 5))
        is_peak = 1 if (7 <= hour_of_day <= 10 or 17 <= hour_of_day <= 21) else 0

        recovery = type_recovery.get(train_type, 0.75)
        future_delay = max(0,
            current_delay * recovery
            + junction_count * float(rng.integers(0, 5))
            + is_peak * float(rng.integers(0, 15))
            + float(rng.normal(0, 8))
        )

        rows.append({
            "train_type": train_type,
            "hour_of_day": hour_of_day,
            "current_delay_min": current_delay,
            "stops_remaining": stops_remaining,
            "distance_remaining_km": distance_remaining,
            "junction_count": junction_count,
            "is_peak_hour": is_peak,
            "future_delay_min": round(future_delay, 1),
        })

    return pd.DataFrame(rows)


def train_model(corridor_trains: list):
    print("[predictor] Generating training data...")
    df = _generate_training_data(corridor_trains)
    le = LabelEncoder()
    df["train_type_enc"] = le.fit_transform(df["train_type"])

    features = [
        "train_type_enc", "hour_of_day", "current_delay_min",
        "stops_remaining", "distance_remaining_km", "junction_count", "is_peak_hour"
    ]
    X, y = df[features], df["future_delay_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    print("[predictor] Training model...")
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"[predictor] MAE: {mae:.1f} minutes")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": features}, f)
    print(f"[predictor] Saved to {MODEL_PATH}")
    return model, le, features


def load_or_train(corridor_trains: list):
    if MODEL_PATH.exists():
        print("[predictor] Loading saved model...")
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["label_encoder"], data["features"]
    return train_model(corridor_trains)


def predict_delay(model, label_encoder, features, train_type, hour_of_day,
                  current_delay_min, stops_remaining, distance_remaining_km, junction_count=2):
    is_peak = 1 if (7 <= hour_of_day <= 10 or 17 <= hour_of_day <= 21) else 0
    try:
        type_enc = label_encoder.transform([train_type])[0]
    except ValueError:
        type_enc = label_encoder.transform(["Exp"])[0]

    X = pd.DataFrame([{
        "train_type_enc": type_enc,
        "hour_of_day": hour_of_day,
        "current_delay_min": current_delay_min,
        "stops_remaining": stops_remaining,
        "distance_remaining_km": distance_remaining_km,
        "junction_count": junction_count,
        "is_peak_hour": is_peak,
    }])

    predicted = max(0, round(float(model.predict(X)[0]), 1))
    reasons = []
    if current_delay_min > 20: reasons.append(f"already {current_delay_min} min late")
    if is_peak: reasons.append("peak hour")
    if junction_count >= 3: reasons.append(f"{junction_count} junctions ahead")

    return {
        "predicted_delay_min": predicted,
        "confidence_low": max(0, round(predicted * 0.8)),
        "confidence_high": round(predicted * 1.2),
        "explanation": f"Predicted {predicted} min delay at destination" + (f" ({', '.join(reasons)})" if reasons else ""),
        "is_peak_hour": bool(is_peak),
    }


if __name__ == "__main__":
    from data_loader import build_corridor_data
    _, _, _, _, corridor_trains = build_corridor_data()
    model, le, features = load_or_train(corridor_trains)
    result = predict_delay(model, le, features, "Exp", 8, 30, 5, 200, 2)
    print("\nSample prediction:", result)