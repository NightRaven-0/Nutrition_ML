# generate synthetic pediatric data for testing purposes

import numpy as np
import pandas as pd
import os
import random

def generate_data(n=5000):
    data = []

    for _ in range(n):
        age = random.randint(6, 59)
        sex = random.randint(0, 1)

        expected_height = (age * 0.5) + 50
        height = np.random.normal(expected_height, 3)

        expected_weight = age * 0.25 + 4
        weight = np.random.normal(expected_weight, 1)

        muac = np.random.normal(135, 10)
        hb = np.random.normal(11.5, 1)

        bmi = weight / ((height/100)**2)

        # Label rules (hidden ground truth)
        if muac < 115:
            label = 2  # SAM
        elif muac < 125:
            label = 1  # MAM
        elif height < 0.9 * expected_height:
            label = 3  # Stunting
        elif hb < 11:
            label = 4  # Anemia
        else:
            label = 0  # Normal

        data.append([age, sex, weight, height, muac, hb, bmi, label])

    columns = ["age", "sex", "weight", "height", "muac", "hb", "bmi", "label"]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_data()
    df.to_csv("data/raw/synthetic_data.csv", index=False)
    print("Synthetic dataset generated.")
