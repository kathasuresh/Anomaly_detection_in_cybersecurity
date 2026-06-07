import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# PROJECT ROOT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ==========================================
# DATASET PATH
# ==========================================

excel_path = os.path.join(
    BASE_DIR,
    "dataset",
    "cybersecurity_intrusion_data.xls"
)

print("Loading Dataset...")
print(excel_path)

# ==========================================
# LOAD DATASET
# ==========================================

try:
    data = pd.read_excel(
        excel_path,
        engine="xlrd"
    )
except Exception:
    data = pd.read_csv(excel_path)

# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)

print("\nColumns:")
print(data.columns.tolist())

print("\nShape:")
print(data.shape)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

if data["encryption_used"].isnull().sum() > 0:

    mode_value = (
        data["encryption_used"]
        .mode()
        .iloc[0]
    )

    data["encryption_used"] = (
        data["encryption_used"]
        .fillna(mode_value)
    )

# ==========================================
# LABEL ENCODING
# ==========================================

protocol_encoder = LabelEncoder()
encrypt_encoder = LabelEncoder()
browser_encoder = LabelEncoder()

data["protocol_type"] = (
    protocol_encoder.fit_transform(
        data["protocol_type"]
    )
)

data["encryption_used"] = (
    encrypt_encoder.fit_transform(
        data["encryption_used"]
    )
)

data["browser_type"] = (
    browser_encoder.fit_transform(
        data["browser_type"]
    )
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

data["failure_ratio"] = (
    data["failed_logins"] /
    (data["login_attempts"] + 1)
)

data["login_risk_score"] = (
    data["login_attempts"] *
    data["failed_logins"]
)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = data.drop(
    ["session_id", "attack_detected"],
    axis=1
)

y = data["attack_detected"]

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    x_train,
    y_train
)

# ==========================================
# GRADIENT BOOSTING MODEL
# ==========================================

gb = GradientBoostingClassifier()

gb.fit(
    x_train,
    y_train
)

# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    rf,
    os.path.join(BASE_DIR, "model.pkl")
)

joblib.dump(
    gb,
    os.path.join(BASE_DIR, "gb_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(BASE_DIR, "scaler.pkl")
)

joblib.dump(
    protocol_encoder,
    os.path.join(BASE_DIR,
                 "protocol_encoder.pkl")
)

joblib.dump(
    encrypt_encoder,
    os.path.join(BASE_DIR,
                 "encrypt_encoder.pkl")
)

joblib.dump(
    browser_encoder,
    os.path.join(BASE_DIR,
                 "browser_encoder.pkl")
)

print("\n===================================")
print("Training Completed Successfully")
print("===================================")
print("Files Generated:")
print("model.pkl")
print("gb_model.pkl")
print("scaler.pkl")
print("protocol_encoder.pkl")
print("encrypt_encoder.pkl")
print("browser_encoder.pkl")
print("===================================")