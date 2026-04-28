import spacy
import string

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Process text using spaCy
    doc = nlp(text)

    # Remove stopwords and keep useful words
    tokens = []
    keywords = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_word = token.lemma_.strip()

            if clean_word:
                tokens.append(clean_word)

                # Keep nouns + important words as keywords
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    keywords.append(clean_word)

    cleaned_text = " ".join(tokens)

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "keywords": list(set(keywords))
    }


# Testing
if __name__ == "__main__":
    sample_text = "I am stressed about exams and my family needs money."

    result = preprocess_text(sample_text)

    print("Original Text:")
    print(result["original_text"])

    print("\nCleaned Text:")
    print(result["cleaned_text"])

    print("\nTokens:")
    print(result["tokens"])

    print("\nKeywords:")
    print(result["keywords"])