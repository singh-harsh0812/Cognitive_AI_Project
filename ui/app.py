import streamlit as st
import joblib

from llm.generate_response import generate_ai_response


# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Cognitive Career Counselor",
    layout="wide"
)

st.title("🧠 Cognitive Career Counselor")
st.subheader("AI-based Career + Life Decision Support")

st.write(
    "Describe your problem below and get a personalized AI-based "
    "career counseling response instead of a static dashboard report."
)


# --------------------------------
# Load ML Model
# --------------------------------
@st.cache_resource
def load_ml_model():
    return joblib.load("models/model.pkl")


model = load_ml_model()


# --------------------------------
# User Input
# --------------------------------
user_input = st.text_area(
    "Tell me your problem:",
    height=180,
    placeholder=(
        "Example: I am confused between getting a job and pursuing "
        "higher studies because of family pressure."
    )
)


# --------------------------------
# Ask AI Button
# --------------------------------
if st.button("Ask AI"):

    if not user_input.strip():
        st.warning("Please enter your problem first.")

    else:
        with st.spinner("Analyzing your situation..."):

            # Greeting Filter
            simple_inputs = [
                "hi", "hello", "hey", "thanks",
                "thank you", "ok", "okay",
                "yes", "no", "hmm"
            ]

            if user_input.lower().strip() in simple_inputs:
                prediction = "General Conversation"

                ai_response = (
                    "Hello! I'm here to help with career decisions, "
                    "higher studies, financial pressure, emotional stress, "
                    "and life choices.\n\n"
                    "Tell me what’s troubling you today."
                )

            else:
                # ML Prediction
                prediction = model.predict([user_input])[0]

                # Phi-2 Final Response
                ai_response = generate_ai_response(
                    user_input,
                    prediction
                )

        # --------------------------------
        # Final Conversational Output
        # --------------------------------
        st.markdown("## 💬 Conversation")

        st.markdown("### 👤 You:")
        st.info(user_input)

        st.markdown("### 🤖 AI Counselor:")
        st.success(ai_response)

        if prediction != "General Conversation":
            st.caption(f"Recommended Decision: {prediction}")