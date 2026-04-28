import pandas as pd
import random

# =========================================================
# PERFECT FLAN-T5 PREMIUM DATASET GENERATOR
# Goal:
# Generate HIGH-QUALITY UNIQUE student-life scenarios
# with logical, human-like, context-aware outputs
#
# Output:
# flan_premium_final.csv
#
# Best for:
# FLAN-T5 fine-tuning
# =========================================================


# =========================================================
# 1. REAL STUDENT LIFE ISSUES
# =========================================================

issues = [
    "low CGPA",
    "placement rejection",
    "family financial pressure",
    "relationship breakup",
    "toxic relationship",
    "backlogs in multiple subjects",
    "fear of disappointing parents",
    "government exam pressure from family",
    "internship rejection",
    "scholarship rejection",
    "mental burnout",
    "depression and anxiety",
    "comparison with friends",
    "lack of motivation",
    "coding skill gap",
    "startup failure fear",
    "loan pressure for education",
    "family health problems",
    "study abroad confusion",
    "career confusion after graduation",
    "lack of self-confidence",
    "social media distraction",
    "parental expectations",
    "financial guilt for not earning",
    "peer pressure",
    "toxic college environment",
    "fear of future uncertainty",
    "long-distance relationship stress",
    "family business pressure",
    "confusion between passion and stability"
]


# =========================================================
# 2. OPTIONS WITH SMART CATEGORY TAGS
# =========================================================

options = {
    "higher studies": "academic",
    "study abroad": "academic",
    "MBA preparation": "academic",
    "research career": "academic",

    "stable job": "financial",
    "software job": "financial",
    "government exam preparation": "financial",
    "bank exam preparation": "financial",
    "family business": "financial",
    "freelancing": "financial",
    "remote work": "financial",

    "startup idea": "growth",
    "self-learning and projects": "growth",
    "upskilling first": "growth",
    "internship preparation": "growth"
}


# =========================================================
# 3. NATURAL INPUT STYLES
# =========================================================

input_styles = [
    "I am struggling with {issue} and I cannot decide between {opt1} and {opt2}. Please guide me.",
    "Because of {issue}, I feel stuck choosing between {opt1} and {opt2}. What should I do?",
    "My family is pressuring me because of {issue}. Should I focus on {opt1} or {opt2}?",
    "I feel mentally exhausted due to {issue}. I need help choosing between {opt1} and {opt2}.",
    "Life feels confusing because of {issue}. Is {opt1} better than {opt2} for my future?",
    "I am worried about {issue} and I am confused between {opt1} and {opt2}. Please suggest the best decision.",
    "Due to {issue}, I feel lost and unsure whether I should choose {opt1} or {opt2}.",
    "Everyone around me seems ahead and because of {issue}, I cannot choose between {opt1} and {opt2}.",
    "I want long-term growth but {issue} is making me choose between {opt1} and {opt2}.",
    "I feel pressure because of {issue}. Can you help me decide between {opt1} and {opt2}?"
]


# =========================================================
# 4. SPECIAL ISSUE RESPONSES
# =========================================================

special_issue_responses = {
    "relationship breakup":
        "A breakup can make everything feel unstable, but temporary emotional pain should not permanently damage your academic future. Focus first on emotional recovery, healthy boundaries, and rebuilding confidence. Career stability and self-growth will help you recover faster than emotional overthinking.",

    "toxic relationship":
        "A toxic relationship affects both focus and self-worth. Your priority should be emotional safety and mental peace. Reduce unnecessary emotional chaos, protect your boundaries, and shift your energy toward studies, skills, and long-term personal growth.",

    "depression and anxiety":
        "Major life decisions should not be made when your mind is overwhelmed. Emotional clarity comes first. Speak with a trusted mentor, counselor, or supportive person. Stabilizing your mental health is not a delay—it is a necessary part of building a stronger future.",

    "family health problems":
        "When family health problems exist, emotional and financial pressure becomes very heavy. In such situations, practical decisions matter most. Supporting your family while protecting your long-term career path should be the priority. Stability first, growth next.",

    "fear of disappointing parents":
        "Many students silently carry the fear of disappointing their parents. The solution is not blind sacrifice but honest communication and practical planning. Show responsibility through consistent action and let your progress build trust instead of fear."
}


# =========================================================
# 5. SMART RESPONSE GENERATOR
# =========================================================

def generate_response(issue, opt1, opt2):
    cat1 = options[opt1]
    cat2 = options[opt2]

    if issue in special_issue_responses:
        return special_issue_responses[issue]

    # Financial priority response
    if cat1 == "financial":
        return (
            f"Since you are dealing with {issue}, choosing {opt1} first may provide immediate stability and reduce pressure. "
            f"This helps you manage present responsibilities while keeping {opt2} as a long-term goal. "
            f"Financial security creates confidence, and once stability improves, future academic or personal growth becomes much easier to pursue."
        )

    # Academic priority response
    elif cat1 == "academic":
        return (
            f"Because of {issue}, focusing on {opt1} can strengthen your long-term career growth and professional opportunities. "
            f"Even if {opt2} seems practical right now, strong academic planning often creates better future stability. "
            f"Choose discipline, consistency, and long-term vision over short-term fear."
        )

    # Growth priority response
    elif cat1 == "growth":
        return (
            f"With {issue} affecting your confidence, starting with {opt1} is a smart way to rebuild control and self-belief. "
            f"Skill development creates opportunities that remain valuable regardless of future decisions. "
            f"Once you gain confidence and practical exposure, moving toward {opt2} becomes much easier and less risky."
        )

    # fallback
    return (
        f"A balanced approach is best while dealing with {issue}. "
        f"Focus first on stability, emotional clarity, and realistic planning. "
        f"Choose the option that protects both your present peace and your future growth."
    )


# =========================================================
# 6. GENERATE UNIQUE DATASET
# =========================================================

TARGET_ROWS = 2000

seen = set()
dataset = []

option_list = list(options.keys())

max_attempts = 50000
attempts = 0

while len(dataset) < TARGET_ROWS and attempts < max_attempts:
    attempts += 1

    issue = random.choice(issues)
    opt1, opt2 = random.sample(option_list, 2)

    input_text = random.choice(input_styles).format(
        issue=issue,
        opt1=opt1,
        opt2=opt2
    )

    if input_text in seen:
        continue

    seen.add(input_text)

    target_output = generate_response(issue, opt1, opt2)

    dataset.append((input_text, target_output))


# =========================================================
# 7. SAVE CSV
# =========================================================

df = pd.DataFrame(
    dataset,
    columns=["input_text", "target_output"]
)

df.to_csv("flan_premium_finalf.csv", index=False)

print(f"Dataset generated successfully with {len(df)} UNIQUE rows.")
print("Saved as: flan_premium_final.csv")
print("This is your final premium FLAN fine-tuning dataset 🔥")