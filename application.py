"""Flask interface for the Algerian forest fire weather-index model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]


def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact not found: {path}")
    with path.open("rb") as artifact:
        return pickle.load(artifact)


ridge_model = load_artifact("ridge.pkl")
standard_scaler = load_artifact("scaler.pkl")
application = Flask(__name__)
app = application  # Compatible with `flask --app application run` and WSGI hosts.


@application.get("/")
def index():
    return render_template("home.html", features=FEATURES, values={})


@application.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", features=FEATURES, values={})

    values = {}
    errors = []
    for feature in FEATURES:
        raw_value = request.form.get(feature, "").strip()
        if not raw_value:
            errors.append(f"{feature} is required.")
            continue
        try:
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError
            values[feature] = value
        except ValueError:
            errors.append(f"{feature} must be a finite number.")

    if errors:
        return render_template("home.html", features=FEATURES, values=request.form, errors=errors), 400

    input_frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    scaled_input = standard_scaler.transform(input_frame)
    prediction = float(ridge_model.predict(scaled_input)[0])
    if not np.isfinite(prediction):
        application.logger.error("Model returned a non-finite prediction")
        return render_template("home.html", features=FEATURES, values=values,
                               errors=["The model could not produce a valid prediction. Please try again."]), 500

    # The training target is the Fire Weather Index (FWI).
    prediction = max(0.0, prediction)
    if prediction < 5:
        risk, risk_class = "Low", "low"
    elif prediction < 15:
        risk, risk_class = "Moderate", "moderate"
    elif prediction < 30:
        risk, risk_class = "High", "high"
    else:
        risk, risk_class = "Very high", "very-high"

    if request.accept_mimetypes.best == "application/json":
        return jsonify(fwi=round(prediction, 2), risk=risk)
    return render_template("home.html", features=FEATURES, values=values,
                           prediction=f"{prediction:.2f}", risk=risk, risk_class=risk_class)


@application.get("/health")
def health():
    return jsonify(status="ok", model="ridge", features=len(FEATURES))


if __name__ == "__main__":
    application.run(host="127.0.0.1", port=5000, debug=False)
