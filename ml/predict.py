import joblib

model = joblib.load("models/model.pkl")

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

for text in real_test_cases:
    prediction = model.predict([text])[0]

    print("\nInput:", text)
    print("Prediction:", prediction)
    print("-" * 60)