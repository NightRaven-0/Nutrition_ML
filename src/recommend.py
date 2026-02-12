# maps predicted class and makes recommendatiions

def get_recommendation(label):
    mapping = {
        0: "Balanced diet with vegetables, pulses, milk.",
        1: "Increase protein and calorie intake: lentils, banana, oil.",
        2: "Severe risk: high protein + energy dense foods + clinical referral.",
        3: "Focus on long-term protein and micronutrients.",
        4: "Iron-rich foods: spinach, ragi, jaggery + Vitamin C fruits."
    }
    return mapping.get(label, "No recommendation available.")
