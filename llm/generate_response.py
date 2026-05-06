import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from ml.predict import predict_with_confidence
from nlp.emotion_detection import detect_emotion, detect_sentiment


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "./llm/phi3_model"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------
# GENERATE RESPONSE
# -----------------------------
def generate_response(user_input, context=""):

    intent_data = predict_with_confidence(user_input)
    emotions = detect_emotion(user_input)
    sentiment = detect_sentiment(user_input)

    primary = intent_data["primary"]
    secondary = intent_data["secondary"]
    confidence = intent_data["confidence"]

    # -----------------------------
    # DECISION STRATEGY
    # -----------------------------
    if confidence == "low":
        decision_instruction = f"""
User is confused between:
- {primary[0]}
- {secondary[0]}

Compare both clearly and guide the decision.
"""
    else:
        decision_instruction = f"""
User is clear.

Give direct advice: {primary[0]}
Do not compare.
"""

    # -----------------------------
    # PROMPT (STRICT FORMAT CONTROL)
    # -----------------------------
    prompt = f"""
You are a practical career advisor.

STYLE:
- Simple, direct, human language
- No formal or complex words
- No motivational or generic lines
- Do NOT write "Option name" or any generic label just use the name of option directly there
- Write like you are talking to a friend
- Avoid words like: "consider", "evaluate", "potential"
- Prefer: "go for", "pick", "focus on"

FORMAT (VERY IMPORTANT):
- When comparing, ALWAYS use this format:

• Option Name:
  → short point
  → short point

• Option Name:
  → short point
  → short point

- Max 3 or 4 points per option
- Each line under 20 words
- No long sentences
- Do NOT write paragraphs when comparing

DECISION:
- End with:
  If X → choose A
  If Y → choose B

- Make it practical (money, time, pressure)

{decision_instruction}

Context:
{context}

User Problem:
{user_input}

Emotion: {", ".join(emotions)} | {sentiment}

Response:
"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=420
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,     # tighter control
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -----------------------------
    # CLEAN RESPONSE
    # -----------------------------
    if "Response:" in result:
        result = result.split("Response:")[-1].strip()

    return result