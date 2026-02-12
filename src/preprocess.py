# feature engineering and data preprocessing functions
# scaling, test-train split, encoding, etc. saves scalar + preprocessor objects for later use in model training and inference

import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv("data/raw/synthetic_data.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), "data/processed/scaled_data.pkl")
    joblib.dump(scaler, "data/processed/scaler.pkl")

    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
