import random
import csv

random.seed(42)  # for reproducibility but still high variety

labels = [
    "Job First (Financial Stability)",
    "Higher Studies First",
    "Government Exam Path",
    "Skill Development First",
    "Balanced Job + Study Plan",
    "Emotional Counseling Needed",
    "Career Guidance Needed"
]

cities = ["Pune", "Mumbai", "Nagpur", "Nashik", "Thane", "Aurangabad", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolhapur"]
colleges = ["COEP", "PICT", "VJTI", "IIT Bombay", "IIT Madras", "Symbiosis", "MIT WPU", "SPIT", "Pune University", "Amity", "local college"]
exams = ["UPSC", "MPSC", "SSC CGL", "Banking PO", "Railway", "GATE", "CAT", "GRE"]

def generate_unique_text(label):
    city = random.choice(cities)
    college = random.choice(colleges)
    sem = random.choice(["final year", "3rd year", "4th year", "just graduated", "6th semester", "last semester"])
    
    if label == "Job First (Financial Stability)":
        templates = [
            f"My father got laid off last month from his company in {city}. With hospital bills and EMIs pending, I have to take a job immediately instead of thinking about MBA or MS.",
            f"Family business completely collapsed after pandemic. As the only son, I feel huge pressure to start earning rather than pursuing higher studies.",
            f"My mother needs surgery and we have no savings. I was planning {random.choice(exams)} but now I think job first is the only option.",
            f"Heavy education loan on my father already. I can't put more burden, so I decided to focus on placements right now.",
            f"After dad's sudden death, all responsibility came on me. Higher studies feel like a luxury I can't afford anymore."
        ]
    
    elif label == "Higher Studies First":
        templates = [
            f"I got admission in MBA at IIM {random.choice(['Ahmedabad','Bangalore','Kozhikode'])} but family is worried about money. Still I believe this is the best investment for my future.",
            f"Cleared GRE with good score and got offer from good US university for MS in CS. Even if expensive, I want to go for higher studies.",
            f"I love research and want to do MTech from IIT Bombay. Ready to take education loan if needed because long term growth matters more.",
            f"Despite financial issues, I want to pursue MCA from a reputed college because specialization will help me a lot in career."
        ]
    
    elif label == "Government Exam Path":
        templates = [
            f"My parents strongly believe only government job is safe. They want me to prepare full time for {random.choice(exams)}.",
            f"I have already cleared prelims of MPSC once. Now I am fully dedicated towards government exams for stability and respect.",
            f"Private jobs feel risky and unstable. That's why I am preparing for SSC CGL and banking exams.",
            f"Family pressure is high for government job. I am focusing only on UPSC preparation now."
        ]
    
    elif label == "Skill Development First":
        templates = [
            f"My coding skills are very poor. Failed in 7 campus interviews. I need to focus only on DSA, projects and internships for next 6-8 months.",
            f"Resume is blank - no projects, no internships. I should build strong skills before thinking about placements or higher studies.",
            f"Communication and confidence bahut low hai. First I need to improve soft skills and get practical experience.",
            f"I realized I don't know anything practical. Planning to join coding bootcamp or online courses full time."
        ]
    
    elif label == "Balanced Job + Study Plan":
        templates = [
            f"Got offer from TCS but I also want to prepare for CAT. Will join job and study side by side.",
            f"Planning to work for 1-2 years and then do MBA so I can have both experience and better qualification.",
            f"Will help in family business in morning and continue MCA in evening batch.",
            f"Have a part-time job and will prepare for government exams simultaneously."
        ]
    
    elif label == "Emotional Counseling Needed":
        templates = [
            f"Breakup ke baad main completely broken hoon. Can't focus on studies or placements. Need emotional support.",
            f"Severe anxiety and overthinking about future. Can't sleep, can't study. I think I need counseling first.",
            f"After my mother's death last month, nothing feels important. Career decisions have become impossible.",
            f"Constant family pressure + studies ne mera burnout kar diya hai. I need someone to talk to."
        ]
    
    elif label == "Career Guidance Needed":
        templates = [
            f"Confused between job, MBA, government exams and startup. No clear direction at all.",
            f"Too many options and I don't know what suits my interest and skills. Need proper guidance.",
            f"Parents want government job, I want tech, but I like business also. Completely lost.",
            f"Keep changing my mind every week. Need someone experienced to show me right path."
        ]
    
    base = random.choice(templates)
    
    # Add extra natural variation
    if random.random() < 0.5:
        base += f" I am studying in {college} {city}."
    if random.random() < 0.3:
        base = "Hey, " + base[0].lower() + base[1:]
    
    return base

# ===================== MAIN GENERATION =====================
print("Generating 4900 unique lines... This may take 10-15 seconds.")

data = []
seen = set()

for label in labels:
    count = 0
    while count < 700:
        text = generate_unique_text(label)
        if text not in seen:   # avoid any rare duplicate
            seen.add(text)
            data.append([text, label])
            count += 1

# Shuffle the dataset
random.shuffle(data)

# Save CSV
with open('career_dataset_4900_unique.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['input', 'decision'])
    writer.writerows(data)

print(f"✅ Success! Generated {len(data)} unique lines")
print("File saved: career_dataset_4900_unique.csv")
print("Ab isko directly pandas mein load karke model train kar sakta hai.")