import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

print("Checking GPU...")

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Using CPU (not recommended)")

print("\nLoading dataset...")

# Load CSV
df = pd.read_csv("llm/flan_dataset.csv")

# Keep required columns only
df = df[["input_text", "target_output"]]


# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

model_name = "google/flan-t5-small"

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def preprocess(example):
    inputs = tokenizer(
        example["input_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        example["target_output"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]
    return inputs


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./llm/flan_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_steps=100,
    logging_steps=20,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting fine-tuning...")

trainer.train()

print("Saving fine-tuned model...")

model.save_pretrained("./llm/flan_model")
tokenizer.save_pretrained("./llm/flan_model")

print("Fine-tuning completed successfully!")