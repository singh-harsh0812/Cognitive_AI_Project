def simulate_decision(prediction):
    simulation_data = {
        "Study First": {
            "short_term": "Better academic focus and improved exam performance",
            "long_term": "Stronger higher education opportunities and career growth",
            "risk": "Financial pressure may continue for some time",
            "benefit": "Improved academic confidence and better long-term options"
        },

        "Job First": {
            "short_term": "Immediate financial support and reduced family pressure",
            "long_term": "Higher studies may get delayed due to work responsibilities",
            "risk": "Academic gap may make future studies harder",
            "benefit": "Financial stability and practical work experience"
        },

        "Higher Studies": {
            "short_term": "More academic pressure and preparation effort required",
            "long_term": "Better specialization and stronger future opportunities",
            "risk": "Short-term financial dependence may continue",
            "benefit": "Long-term career growth and stronger qualifications"
        },

        "Balanced Plan": {
            "short_term": "Manage both work and studies with careful planning",
            "long_term": "Stable progress in both career and education",
            "risk": "Burnout due to excessive workload",
            "benefit": "Financial stability without fully sacrificing education"
        },

        "Counseling Needed": {
            "short_term": "Mental clarity and emotional support improve decision making",
            "long_term": "Better long-term decisions and reduced emotional stress",
            "risk": "Delay in action if guidance is ignored",
            "benefit": "Healthier decisions with stronger confidence"
        }
    }

    return simulation_data.get(prediction, {
        "short_term": "No data available",
        "long_term": "No data available",
        "risk": "No data available",
        "benefit": "No data available"
    })


if __name__ == "__main__":
    sample_prediction = "Job First"

    result = simulate_decision(sample_prediction)

    print("Prediction:", sample_prediction)

    print("\nShort-Term Outcome:")
    print(result["short_term"])

    print("\nLong-Term Outcome:")
    print(result["long_term"])

    print("\nRisk:")
    print(result["risk"])

    print("\nBenefit:")
    print(result["benefit"])