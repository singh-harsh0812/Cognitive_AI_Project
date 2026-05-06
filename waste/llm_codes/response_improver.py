import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# PHI-3 MINI DATASET RESPONSE IMPROVER
#
# Purpose:
# Keep input_text same
# Regenerate only target_output using fresh Phi-3 Mini
#
# Input CSV:
# input_text,target_output
#
# Output CSV:
# improved_llm_dataset.csv
#
# IMPORTANT:
# First test on 50–100 rows only
# Then run full dataset
# =====================================================


# -----------------------------------------------------
# Settings
# -----------------------------------------------------
INPUT_FILE = "dataset/llm_dataset.csv"
OUTPUT_FILE = "dataset/improved_llm_dataset.csv"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TEST_ROWS = 100   # Start small first (recommended)


# -----------------------------------------------------
# Load Phi-3 Mini
# -----------------------------------------------------
print("Loading Phi-3 Mini tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading Phi-3 Mini model on GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("Phi-3 Mini Loaded Successfully!\n")


# -----------------------------------------------------
# Response Generator
# -----------------------------------------------------
def improve_response(user_input, old_response):
    prompt = f"""
You are an expert career counselor and emotional mentor.

Your task is to rewrite the counselor response naturally.

IMPORTANT RULES:
- Keep the same core advice
- Make it sound human and practical
- Avoid repeated robotic phrases
- Avoid template responses like:
  "Since you are dealing with..."
  "Because of..."
- Response must feel emotionally intelligent
- One paragraph only
- No bullet points
- No labels like Answer:
- Supportive and realistic

Student Problem:
{user_input}

Old Counselor Response:
{old_response}

Better Counselor Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if "Better Counselor Response:" in full_output:
        final_response = full_output.split(
            "Better Counselor Response:"
        )[-1].strip()
    else:
        final_response = full_output.strip()

    if len(final_response) < 30:
        final_response = old_response

    return final_response


# -----------------------------------------------------
# Main Processing
# -----------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows found: {len(df)}")

# Safety test first
df = df.head(TEST_ROWS)

print(f"Processing first {len(df)} rows only...\n")

improved_outputs = []

for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}")

    user_input = str(row["input_text"])
    old_output = str(row["target_output"])

    try:
        better_output = improve_response(
            user_input,
            old_output
        )

        improved_outputs.append({
            "input_text": user_input,
            "target_output": better_output
        })

    except Exception as e:
        print(f"Error on row {idx + 1}: {e}")

        improved_outputs.append({
            "input_text": user_input,
            "target_output": old_output
        })


# -----------------------------------------------------
# Save New Dataset
# -----------------------------------------------------
new_df = pd.DataFrame(improved_outputs)

new_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n====================================")
print("Dataset improvement complete!")
print("====================================")
print(f"Saved file: {OUTPUT_FILE}")
print(f"Total improved rows: {len(new_df)}")
print("\nCheck quality first before full run.")