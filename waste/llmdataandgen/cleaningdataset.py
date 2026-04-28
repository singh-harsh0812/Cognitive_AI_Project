import pandas as pd

print("Fixing encoding issues...")

file_path = "flan_premium_finalf.csv"   # your current dataset file

df = pd.read_csv(file_path)

def fix_encoding(text):
    if pd.isna(text):
        return text

    text = str(text)

    replacements = {
        "â€”": "—",
        "â€“": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¦": "...",
        "â€": "",
        "Â": "",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # optional safer version:
    text = text.replace("—it", ". It")

    return text.strip()

df["input_text"] = df["input_text"].apply(fix_encoding)
df["target_output"] = df["target_output"].apply(fix_encoding)

output_file = "flan_premium_final_cleaned.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Encoding fixed successfully!")
print("Saved as:", output_file)
print("Rows:", len(df))