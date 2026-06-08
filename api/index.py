import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Configure Flask to use the public/static folder and templates at the repo root
app = Flask(__name__, static_folder="../public/static", template_folder="../templates")

# Load trained models, encoders, and scalers
rf_model = joblib.load("model.pkl")
gb_model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")

protocol_encoder = joblib.load("protocol_encoder.pkl")
encrypt_encoder = joblib.load("encrypt_encoder.pkl")
browser_encoder = joblib.load("browser_encoder.pkl")

# Load pre‑generated EDA metrics and model metadata (stored in the static folder)
metadata_path = os.path.join(app.static_folder, "metadata.json")
try:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
except Exception as e:
    metadata = {}
    print(f"Error loading metadata.json: {e}")


@app.route("/")
def home():
    return render_template("index.html", page="predict")


@app.route("/dataset")
def dataset_info():
    return render_template("dataset.html", page="dataset", metadata=metadata)


@app.route("/eda")
def eda():
    return render_template("eda.html", page="eda", metadata=metadata)


@app.route("/models")
def models():
    return render_template("models.html", page="models", metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        network_packet_size = float(request.form["network_packet_size"])
        protocol_type = request.form["protocol_type"]
        login_attempts = int(request.form["login_attempts"])
        session_duration = float(request.form["session_duration"])
        encryption_used = request.form["encryption_used"]
        ip_reputation_score = float(request.form["ip_reputation_score"])
        failed_logins = int(request.form["failed_logins"])
        browser_type = request.form["browser_type"]
        unusual_time_access = int(request.form["unusual_time_access"])
        model_choice = request.form["model"]
    except KeyError as e:
        return render_template(
            "result.html",
            result=f"Input error: Missing field {e}",
            inputs={},
            model_used="rf",
            page="predict"
        )

    # Store inputs to render in the results view
    inputs = {
        "network_packet_size": network_packet_size,
        "protocol_type": protocol_type,
        "login_attempts": login_attempts,
        "session_duration": session_duration,
        "encryption_used": encryption_used,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": failed_logins,
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access
    }

    try:
        encoded_protocol = protocol_encoder.transform([protocol_type])[0]
        encoded_encryption = encrypt_encoder.transform([encryption_used])[0]
        encoded_browser = browser_encoder.transform([browser_type])[0]
    except ValueError as exc:
        return render_template(
            "result.html",
            result=f"Transformation error: {exc}",
            inputs=inputs,
            model_used=model_choice,
            page="predict"
        )

    # Re‑calculate engineered features
    failure_ratio = failed_logins / (login_attempts + 1)
    login_risk_score = login_attempts * failed_logins

    # Assemble data vector (order must match training script)
    data = pd.DataFrame([[
        network_packet_size,
        encoded_protocol,
        login_attempts,
        session_duration,
        encoded_encryption,
        ip_reputation_score,
        failed_logins,
        encoded_browser,
        unusual_time_access,
        failure_ratio,
        login_risk_score
    ]])

    # Scale the features
    data_scaled = scaler.transform(data)

    # Choose classifier
    if model_choice == "rf":
        prediction = rf_model.predict(data_scaled)[0]
    else:
        prediction = gb_model.predict(data_scaled)[0]

    result = (
        "Cyber Attack Detected"
        if prediction == 1
        else "Normal Session"
    )

    return render_template(
        "result.html",
        result=result,
        inputs=inputs,
        model_used=model_choice,
        page="predict"
    )

# When deployed on Vercel the function is imported, so we only run the dev server locally.
if __name__ == "__main__":
    app.run(debug=True)
