import numpy as np
import pandas as pd
import os
import random


def realistic_expected_height(age):
    # Approximate WHO-like linear growth (6–59 months)
    # 6m ≈ 65 cm, 60m ≈ 110 cm
    return 65 + (age - 6) * 0.9


def realistic_expected_weight(age):
    # Rough pediatric scaling
    # 6m ≈ 7 kg, 60m ≈ 18 kg
    return 7 + (age - 6) * 0.2


def generate_data(n=6000):
    data = []

    for _ in range(n):
        age = random.randint(6, 59)
        sex = random.randint(0, 1)

        expected_height = realistic_expected_height(age)
        expected_weight = realistic_expected_weight(age)

        # Add realistic biological variation
        height = np.random.normal(expected_height, 5)
        weight = np.random.normal(expected_weight, 2)
        muac = np.random.normal(135, 15)
        hb = np.random.normal(11.5, 1.5)

        bmi = weight / ((height / 100) ** 2)

        # --- Acute Malnutrition ---
        # Add probabilistic decision boundary
        if muac < 110:
            acute_label = 2
        elif muac < 125:
            acute_label = 1
        else:
            acute_label = 0

        # Inject noise (5%)
        if random.random() < 0.05:
            acute_label = random.choice([0, 1, 2])

        # --- Stunting ---
        height_ratio = height / expected_height
        stunting_flag = 1 if height_ratio < 0.85 else 0

        # Add overlap noise
        if random.random() < 0.05:
            stunting_flag = 1 - stunting_flag

        # --- Anemia ---
        anemia_flag = 1 if hb < 11 else 0

        data.append([
            age, sex, weight, height, muac, hb, bmi,
            acute_label, stunting_flag, anemia_flag
        ])

    columns = [
        "age", "sex", "weight", "height", "muac", "hb", "bmi",
        "acute_label", "stunting_flag", "anemia_flag"
    ]

    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_data()
    df.to_csv("data/raw/synthetic_data.csv", index=False)
    print("Improved realistic synthetic dataset generated.")
