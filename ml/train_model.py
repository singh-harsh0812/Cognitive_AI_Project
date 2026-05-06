import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# CONFIG
# ==========================================
DATASET_PATH = "dataset/hard_realistic_ml_dataset_1400.csv"
MODEL_SAVE_PATH = "models/model.pkl"

# ==========================================
# LOAD DATA
# ==========================================
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["input"]
y = df["decision"]

print(f"Total rows: {len(df)}")
print(f"Total labels: {y.nunique()}")
print("Labels:", sorted(y.unique()))

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# ==========================================
# MODEL PIPELINE
# ==========================================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        lowercase=True,
        strip_accents="unicode"
    )),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        C=0.1
    ))
])

# ==========================================
# TRAIN MODEL
# ==========================================
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete")

# ==========================================
# EVALUATION
# ==========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Test Accuracy: {accuracy:.4f}")
print("==============================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# CROSS VALIDATION
# ==========================================
print("\nRunning Cross Validation...")
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\nModel saved at: {MODEL_SAVE_PATH}")

# ==========================================
# SIMPLE TEST (OPTIONAL CHECK)
# ==========================================
print("\n==== Quick Test ====\n")

test_samples = [
    "I am stressed about exams and my family needs money",
    "I want higher studies but financial pressure is forcing me to work",
    "I feel mentally exhausted and confused about my future"

]

for text in test_samples:
    pred = model.predict([text])[0]
    print(f"Input: {text}")
    print(f"Prediction: {pred}")
    print("-" * 60)