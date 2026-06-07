import joblib
import pandas as pd

rf = joblib.load("../model.pkl")

def predict_attack(data):

    pred = rf.predict(data)

    return pred