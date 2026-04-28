from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import streamlit as st

from nlp.emotion_detection import detect_emotion, detect_sentiment
from simulation.simulation import simulate_decision
from simulation.planner import generate_action_plan


# -----------------------------------
# Load Phi-2 Model Once (GPU Version)
# -----------------------------------
@st.cache_resource
def load_phi2_model():
    base_model_name = "microsoft/phi-2"
    adapter_path = "./llm/phi2_model"

    print("Loading Phi-2 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base Phi-2 model on GPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
        offload_folder="offload"
    )

    model.eval()

    print("Phi-2 GPU Model Ready!")

    return tokenizer, model


# -----------------------------------
# Generate AI Response
# -----------------------------------
def generate_ai_response(user_input, prediction):
    tokenizer, model = load_phi2_model()

    # NLP
    emotions = detect_emotion(user_input)
    sentiment = detect_sentiment(user_input)

    # Simulation
    simulation_result = simulate_decision(prediction)

    # Weekly Plan
    action_plan = generate_action_plan(prediction)
    action_plan_text = "\n".join(action_plan)

    prompt = f"""
You are an expert career counselor, emotional mentor, and life decision advisor.

Student Problem:
{user_input}

ML Suggested Decision:
{prediction}

Detected Emotions:
{", ".join(emotions)}

Sentiment:
{sentiment}

Future Outcome Simulation:

Short-Term:
{simulation_result["short_term"]}

Long-Term:
{simulation_result["long_term"]}

Risk:
{simulation_result["risk"]}

Benefit:
{simulation_result["benefit"]}

4-Week Improvement Plan:
{action_plan_text}

Task:
Write one strong, natural, human-like mentor response.

Rules:
- only one paragraph
- no bullet points
- no robotic text
- no labels like Answer:
- supportive and practical
- emotionally intelligent
- realistic advice

Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    # Decode full output
    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    # Extract only final generated answer
    if "Response:" in full_response:
        response = full_response.split("Response:")[-1].strip()
    else:
        response = full_response.strip()

    # Safety fallback
    if len(response) < 30:
        response = (
            "Focus on stability first, reduce immediate pressure, "
            "and take gradual steps toward your long-term goals."
        )

    return response


# -----------------------------------
# Direct Testing
# -----------------------------------
if __name__ == "__main__":
    sample_input = (
        "I am confused between getting a job and pursuing higher studies "
        "because my family needs financial support."
    )

    sample_prediction = "Job First"

    result = generate_ai_response(
        sample_input,
        sample_prediction
    )

    print("\nGenerated Response:\n")
    print(result)