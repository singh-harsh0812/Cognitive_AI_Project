def generate_action_plan(prediction):
    planner_data = {
        "Study First": [
            "Week 1: Create a strong study timetable and remove distractions",
            "Week 2: Focus on exam preparation and assignment completion",
            "Week 3: Improve weak subjects and practice revision",
            "Week 4: Plan future academic goals and higher study options"
        ],

        "Job First": [
            "Week 1: Prepare resume and update professional profiles",
            "Week 2: Apply for internships and entry-level job opportunities",
            "Week 3: Improve interview skills and technical preparation",
            "Week 4: Create a long-term plan for future higher studies"
        ],

        "Higher Studies": [
            "Week 1: Research universities and course options",
            "Week 2: Prepare for entrance exams and required certifications",
            "Week 3: Build academic profile and gather recommendation support",
            "Week 4: Create application roadmap for higher education goals"
        ],

        "Balanced Plan": [
            "Week 1: Create a balanced schedule for work and study",
            "Week 2: Search for part-time work and continue academic preparation",
            "Week 3: Improve time management and maintain consistency",
            "Week 4: Build a practical long-term career and education roadmap"
        ],

        "Counseling Needed": [
            "Week 1: Talk to a mentor, teacher, or trusted counselor",
            "Week 2: Reduce emotional pressure and organize priorities clearly",
            "Week 3: Improve mental wellness and rebuild motivation",
            "Week 4: Make a confident long-term decision with proper clarity"
        ]
    }

    return planner_data.get(prediction, [
        "No action plan available"
    ])


if __name__ == "__main__":
    sample_prediction = "Job First"

    plan = generate_action_plan(sample_prediction)

    print("Prediction:", sample_prediction)
    print("\nWeekly Action Plan:\n")

    for step in plan:
        print(step)