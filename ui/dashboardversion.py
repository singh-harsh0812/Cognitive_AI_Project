import streamlit as st
import joblib

from nlp.preprocess import preprocess_text
from nlp.emotion_detection import detect_emotion, detect_sentiment
from simulation.simulation import simulate_decision
from simulation.planner import generate_action_plan
from llm.generate_response import generate_ai_response


# Load trained ML model
model = joblib.load("models/model.pkl")


# Page settings
st.set_page_config(
    page_title="Cognitive Decision Support System",
    layout="wide"
)

st.title("🧠 Cognitive Decision Support System")
st.subheader("AI-based Career + Study + Life Decision Support")

st.write(
    "Describe your problem below and the system will help with "
    "prediction, emotion analysis, simulation, and AI guidance."
)


# User Input
user_input = st.text_area(
    "Enter your problem here:",
    height=150,
    placeholder="Example: I am confused between higher studies and getting a job. Family pressure is high."
)


if st.button("Analyze Decision"):

    if user_input.strip() == "":
        st.warning("Please enter your problem first.")

    else:

        with st.spinner("Analyzing your situation..."):

            # NLP Preprocessing
            preprocess_result = preprocess_text(user_input)

            cleaned_text = preprocess_result["cleaned_text"]
            keywords = preprocess_result["keywords"]

            # Emotion + Sentiment
            emotions = detect_emotion(user_input)
            sentiment = detect_sentiment(user_input)

            # Greeting / Simple Input Detection
            simple_inputs = [
                "hi", "hello", "hey", "thanks", "thank you",
                "good morning", "good evening", "ok", "okay",
                "yes", "no", "hmm"
            ]

            if user_input.lower().strip() in simple_inputs:
                prediction = "General Conversation"

                ai_response = (
                    "Hello! I'm here to help you with career decisions, "
                    "higher studies, job confusion, emotional stress, and "
                    "life choices. Tell me what’s troubling you today."
                )

                simulation_result = None
                action_plan = []

            else:
                # ML Prediction
                prediction = model.predict([user_input])[0]

                # Simulation
                simulation_result = simulate_decision(prediction)

                # Action Plan
                action_plan = generate_action_plan(prediction)

                # FLAN AI Response
                ai_response = generate_ai_response(
                    user_input,
                    prediction
                )

        # Display Results
        st.success("Analysis Completed Successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧹 NLP Analysis")
            st.write("**Cleaned Text:**", cleaned_text)

            if keywords:
                st.write("**Keywords:**", ", ".join(keywords))
            else:
                st.write("**Keywords:** No major keywords detected")

            st.subheader("😊 Emotion Analysis")
            st.write("**Detected Emotion:**", ", ".join(emotions))
            st.write("**Sentiment:**", sentiment)

        with col2:
            st.subheader("🎯 Decision Prediction")
            st.write("**Recommended Decision:**", prediction)

            if prediction != "General Conversation":
                st.subheader("📈 Future Outcome Simulation")
                st.write("**Short-Term:**", simulation_result["short_term"])
                st.write("**Long-Term:**", simulation_result["long_term"])
                st.write("**Risk:**", simulation_result["risk"])
                st.write("**Benefit:**", simulation_result["benefit"])

        st.subheader("🤖 Personalized AI Guidance")
        st.info(ai_response)

        if prediction != "General Conversation":
            st.subheader("📅 Weekly Action Plan")

            for step in action_plan:
                st.write("•", step)