
from textblob import TextBlob


def detect_emotion(text):
    text = text.lower()

    emotion_keywords = {
        "Stress": [
            "stress", "stressed", "pressure", "overwhelmed",
            "tired", "burden", "exhausted"
        ],

        "Anxiety": [
            "anxiety", "anxious", "worried", "fear",
            "panic", "nervous", "tension"
        ],

        "Confusion": [
            "confused", "unsure", "lost", "unclear",
            "doubt", "cannot decide", "uncertain"
        ],

        "Depression": [
            "depressed", "depression", "hopeless",
            "low", "sad", "empty", "burnout"
        ],

        "Motivation": [
            "motivated", "inspired", "focused",
            "determined", "goal", "improve"
        ]
    }

    detected_emotions = []

    for emotion, words in emotion_keywords.items():
        for word in words:
            if word in text:
                detected_emotions.append(emotion)
                break

    if not detected_emotions:
        detected_emotions.append("Neutral")

    return detected_emotions


def detect_sentiment(text):
    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


if __name__ == "__main__":
    sample_text = "I feel mentally exhausted and confused about my future."

    emotions = detect_emotion(sample_text)
    sentiment = detect_sentiment(sample_text)

    print("Input Text:")
    print(sample_text)

    print("\nDetected Emotion:")
    print(emotions)

    print("\nSentiment:")
    print(sentiment)