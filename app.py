import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Chatbot System", layout="wide")

# -----------------------------
# 🎨 CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}
[data-testid="stSidebar"] {
    background-color: #1e3a8a;
}
[data-testid="stSidebar"] * {
    color: white;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
h1, h2, h3 {
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    return tokenizer, model, sentiment_pipeline

tokenizer, model, sentiment_pipeline = load_models()

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_language(text):
    if re.search(r'[\u0C00-\u0C7F]', text) and re.search(r'[A-Za-z]', text):
        return "Mixed (Telugu + English)"
    elif re.search(r'[\u0C00-\u0C7F]', text):
        return "Telugu"
    elif re.search(r'[\u0900-\u097F]', text):
        return "Hindi"
    elif re.search(r'[A-Za-z]', text):
        return "English"
    else:
        return "Unknown"

def get_sentiment(text):
    result = sentiment_pipeline(text[:256])[0]
    label = result["label"]
    score = round(result["score"], 3)

    if "1 star" in label:
        sentiment = "Very Negative"
    elif "2 star" in label:
        sentiment = "Negative"
    elif "3 star" in label:
        sentiment = "Neutral"
    elif "4 star" in label:
        sentiment = "Positive"
    else:
        sentiment = "Very Positive"

    return sentiment, score

# Remove repetition
def clean_text(text):
    sentences = text.split(". ")
    unique = []
    for s in sentences:
        if s.strip() not in unique:
            unique.append(s.strip())
    return ". ".join(unique)

# One-line answer
import wikipedia

def get_answer(question):
    try:
        # 🔥 Step 1: Get factual content
        summary = wikipedia.summary(question, sentences=2)

        # 🔥 Step 2: Convert to clean one-line
        prompt = f"""
        Convert this into one correct and simple sentence with full meaning:

        {summary}
        """

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.3,
                num_beams=4
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer

    except:
        # 🔥 Fallback if Wikipedia fails
        prompt = f"Give a correct one-line answer: {question}"

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dashboard", "Sentiment Chatbot", "AI Chatbot"]
)

# -----------------------------
# 🏠 HOME
# -----------------------------
if page == "Home":
    st.markdown("# 🤖 AI Chatbot & Sentiment Analysis System")

    st.markdown("### 📌 Project Submission By")
    st.markdown("## 👩‍💻 Lakshmi Tejasree Kurmapu")

    st.markdown("---")

    st.image(
        "https://pckix.com/wp-content/uploads/2025/06/Best-AI-Chatbots.jpg",
        use_container_width=True
    )

    st.markdown("""
## 📌 Project Overview

- AI Chatbot  
- Sentiment Analysis  
- Language Detection  
- Chat History  

👉 Use sidebar to navigate
""")

# -----------------------------
# 📊 DASHBOARD (IMPROVED)
# -----------------------------
elif page == "Dashboard":
    st.title("📊 Project Dashboard")

    st.markdown("## ⚡ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("😊 Sentiment Analysis\n\nDetects Positive, Negative, Neutral emotions.")

    with col2:
        st.info("🌍 Language Detection\n\nIdentifies English, Telugu, Hindi, Mixed text.")

    with col3:
        st.info("🤖 AI Chatbot\n\nAnswers user questions intelligently.")

    st.markdown("---")

    st.markdown("## 🧠 Technologies Used")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.success("Python\nStreamlit")

    with col5:
        st.success("Hugging Face\nTransformers")

    with col6:
        st.success("FLAN-T5\nBERT Model")

    st.markdown("---")

    st.markdown("## 🌍 Real-World Applications")

    st.write("✔ Social Media Sentiment Analysis")
    st.write("✔ Customer Feedback Analysis")
    st.write("✔ YouTube Comment Analysis")
    st.write("✔ Chatbot Systems")

    st.markdown("---")

    st.markdown("## 📈 Project Summary")

    col7, col8, col9 = st.columns(3)

    with col7:
        st.metric("Models Used", "2")

    with col8:
        st.metric("Languages Supported", "3+")

    with col9:
        st.metric("Core Features", "4")

# -----------------------------
# 💬 SENTIMENT CHATBOT
# -----------------------------
elif page == "Sentiment Chatbot":
    st.title("💬 Sentiment Analysis Chatbot")

    user_input = st.text_input("Enter your text:")

    if st.button("Analyze") and user_input.strip():
        answer = get_answer(user_input)
        sentiment, score = get_sentiment(user_input)
        language = detect_language(user_input)

        st.session_state.chat_history.append({
            "text": user_input,
            "answer": answer,
            "sentiment": sentiment,
            "score": score,
            "language": language
        })

        st.markdown("### 🤖 AI Response")
        st.write(answer)

        st.markdown("### 📊 Analysis")
        st.success(f"""
Sentiment: {sentiment}  
Score: {score}  
Language: {language}
""")

    st.subheader("📜 Chat History")
    for chat in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"""
---
**Text:** {chat['text']}  
🤖 {chat['answer']}  
📊 {chat['sentiment']} | {chat['score']}  
🌍 {chat['language']}
""")

# -----------------------------
# 💬 AI CHATBOT
# -----------------------------
elif page == "AI Chatbot":
    st.title("💬 AI Chatbot")

    user_input = st.text_input("Ask anything:")

    if st.button("Ask") and user_input.strip():
        answer = get_answer(user_input)

        st.markdown("### 🤖 Answer")
        st.write(answer)