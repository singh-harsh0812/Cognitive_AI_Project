from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading FLAN-T5 model... Please wait...")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!\n")

user_input = "I am confused between job and higher studies because my family needs money."

prompt = f"""
You are a helpful career and emotional support assistant.

User Problem:
{user_input}

Give a practical and helpful recommendation:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("FLAN-T5 Response:\n")
print(response)