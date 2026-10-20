# Algerian Forest Fire FWI Predictor

A small Flask app that estimates the Fire Weather Index (FWI) from nine weather and dataset fields using the saved Ridge regression model and StandardScaler.

## Run locally

From this folder, install dependencies and start the development server:

```powershell
python -m pip install -r requirements.txt
python application.py
```

Open <http://127.0.0.1:5000>. The model artifacts are loaded relative to `application.py`, so the command also works when launched from another working directory.

## Deploy

Deploy this folder as the repository root on a Python web-service host. The included `Procfile` starts the WSGI application with Gunicorn. Install dependencies with `pip install -r requirements.txt`; the host should use Python 3.11 or 3.12 with scikit-learn 1.9.1 for the saved model artifacts.

For Flask CLI use `python -m flask --app application run`. A JSON prediction can be requested by posting the same nine form fields to `/predictdata` with `Accept: application/json`. `GET /health` reports whether the app loaded successfully.

## Inputs

Temperature (°C), relative humidity (%), wind speed (km/h), rain (mm), FFMC, DMC, ISI, fire class (`0` not fire / `1` fire), and region (`0` Bejaia / `1` Sidi-Bel Abbes). FWI is returned on the model's original scale, rounded to two decimal places, along with a broad risk band.

## Model compatibility

The included pickle files are the fitted training artifacts. Pickle files must only be loaded from trusted sources. They were saved with scikit-learn 1.9.1; matching that version is recommended when running the app.
