from textblob import TextBlob


# ==========================================
# SENTIMENT DETECTION
# ==========================================
def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"


# ==========================================
# EMOTION DETECTION (ROBUST VERSION)
# ==========================================
def detect_emotion(text):
    text = text.lower()

    emotion_keywords = {
        "Stress": [
            "stress", "stressed", "pressure", "overwhelmed",
            "tired", "exhausted", "burden"
        ],

        "Anxiety": [
            "anxiety", "anxious", "worried", "fear",
            "panic", "nervous", "tension"
        ],

        "Confusion": [
            "confused", "unsure", "lost", "unclear",
            "uncertain", "don't know", "what to do",
            "no idea", "cannot decide", "stuck"
        ],

        "Depression": [
            "depressed", "hopeless", "sad",
            "empty", "burnout"
        ],

        "Motivation": [
            "motivated", "inspired", "focused",
            "determined", "excited"
        ]
    }

    detected = []

    # ==========================================
    # STEP 1: Keyword Detection
    # ==========================================
    for emotion, words in emotion_keywords.items():
        if any(word in text for word in words):
            detected.append(emotion)

    if detected:
        return detected[:2]   # limit to top 2

    # ==========================================
    # STEP 2: Smart Rule-Based Fallback
    # ==========================================
    if any(word in text for word in ["don't know", "lost", "stuck", "confused"]):
        return ["Confusion"]

    # ==========================================
    # STEP 3: Sentiment-Based Fallback
    # ==========================================
    sentiment = detect_sentiment(text)

    if sentiment == "Negative":
        return ["Stress"]

    elif sentiment == "Positive":
        # prevent false motivation
        if any(word in text for word in ["motivated", "excited", "inspired", "focused"]):
            return ["Motivation"]
        return ["Neutral"]

    # ==========================================
    # STEP 4: Final Default
    # ==========================================
    return ["Neutral"]


# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    test_samples = [
        "I feel mentally exhausted and confused about my future.",
        "I am really motivated to improve my coding skills.",
        "I don't know what to do anymore.",
        "Everything feels hopeless and empty.",
        "I am just doing my work normally.",
        "I am worried about my career and feeling nervous.",
        "I feel pressure from my family to succeed.",
        "I am excited to learn new technologies.",
        "I feel stuck and unsure about my next step.",
        "Life feels normal these days."
    ]

    print("\n===== Emotion Detection Test =====\n")

    for text in test_samples:
        emotions = detect_emotion(text)
        sentiment = detect_sentiment(text)

        print(f"Input: {text}")
        print(f"Emotion: {emotions}")
        print(f"Sentiment: {sentiment}")
        print("-" * 60)