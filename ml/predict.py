import joblib
import numpy as np

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("models/model.pkl")


# ==========================================
# BASIC PREDICTION (KEEP THIS)
# ==========================================
def predict(text):
    return model.predict([text])[0]


# ==========================================
# 🔥 TOP-2 + CONFIDENCE (MAIN FUNCTION)
# ==========================================
def predict_with_confidence(text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_

    top2_idx = np.argsort(probs)[-2:][::-1]

    top1_label = labels[top2_idx[0]]
    top1_prob = probs[top2_idx[0]]

    top2_label = labels[top2_idx[1]]
    top2_prob = probs[top2_idx[1]]

    diff = top1_prob - top2_prob

    # Confidence Logic
    if top1_prob > 0.6:
        confidence = "high"
    elif diff < 0.1:
        confidence = "low"
    else:
        confidence = "medium"

    return {
        "primary": (top1_label, float(round(top1_prob, 3))),
        "secondary": (top2_label, float(round(top2_prob, 3))),  # ✅ FIXED
        "confidence": confidence
    }


# ==========================================
# TEST BLOCK (LIKE YOUR CURRENT STYLE)
# ==========================================
if __name__ == "__main__":

    real_test_cases = [
        "I am stressed about exams and my family needs money",
        "I want higher studies but financial pressure is forcing me to work",
        "I feel mentally exhausted and confused about my future",
        "I want a stable job while preparing for masters",
        "Relationship stress is affecting my studies badly",
        "I need income urgently but I also want to continue learning",
        "Depression is affecting my studies and I cannot focus",
        "Family pressure and exam stress are making decisions difficult",
        "I want to move abroad for studies but finances are weak",
        "I feel lost between career growth and emotional stability"
    ]

    print("\n==== TESTING MODEL ====\n")

    for text in real_test_cases:

        # Basic prediction
        pred = predict(text)

        # Top-2 prediction
        top2 = predict_with_confidence(text)

        print(f"Input: {text}")
        print(f"Prediction: {pred}")
        print(f"Top-2: {top2}")
        print("-" * 60)