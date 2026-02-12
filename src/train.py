# loads preprocessed data, trains ML models, evaluates performance, 
# saves trained models and evaluation metrics for later use in inference and reporting

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model():
    X_train, X_test, y_train, y_test = joblib.load("data/processed/scaled_data.pkl")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/malnutrition_model.pkl")

    print("Model trained and saved.")


if __name__ == "__main__":
    train_model()
