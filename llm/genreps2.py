from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading Fine-Tuned FLAN Model...")

# Path to your fine-tuned model
model_path = "./llm/flan_model"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Model loaded successfully!")


def generate_ai_response(user_input):
    """
    Generate rich human-like counseling response
    based on student's problem.
    """

    prompt = f"""
You are an expert career counselor, emotional support mentor, and life decision advisor.

A student is facing confusion related to:
- career decisions
- higher studies
- job vs education
- family financial pressure
- family expectations
- emotional stress
- relationship problems
- low confidence
- future uncertainty
- mental burnout

Student Problem:
{user_input}

Write a detailed, natural, human-like response as a supportive mentor.

Explain:
- what the student should do first
- how to handle emotional pressure
- the best short-term action
- the best long-term career direction

IMPORTANT RULES:
- Do NOT write numbered points
- Do NOT write bullet points
- Do NOT write only one sentence
- Write one meaningful helpful paragraph
- Sound like a real human mentor
- Be warm, practical, realistic, and professional

Give a strong personalized response:
"""

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.15,
        early_stopping=True
    )

    # Decode output
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# Direct testing
if __name__ == "__main__":
    sample = "I am confused between job and higher studies because my family needs money and I feel emotionally stressed."

    result = generate_ai_response(sample)

    print("\nAI Response:\n")
    print(result)