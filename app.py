import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Chatbot System", layout="wide")

# -----------------------------
# LOAD MODELS (FIXED - NO PIPELINE ERRORS)
# -----------------------------
@st.cache_resource
def load_models():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    return tokenizer, model, sentiment_pipeline


tokenizer, model, sentiment_pipeline = load_models()

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_language(text):

    if re.search(r'[\u0C00-\u0C7F]', text):
        return "Telugu"

    words = re.findall(r"[A-Za-z]+", text)

    if len(words) > 0:
        return "English"

    return "Unknown"


def get_sentiment(text):

    result = sentiment_pipeline(text[:256])[0]

    label = result["label"].lower()
    score = round(result["score"], 4)

    if "positive" in label:
        return "Positive", score
    elif "negative" in label:
        return "Negative", score
    else:
        return "Neutral", score


# -----------------------------
# CHATBOT (FIXED - NO PIPELINE TASK ERROR)
# -----------------------------
def get_answer(question):

    input_text = f"Answer in simple sentence: {question}"

    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Chatbot"])

# -----------------------------
# 🏠 HOME PAGE
# -----------------------------
if page == "Home":

    st.markdown("# 🤖 AI Chatbot & Sentiment Analysis System")

    st.image(
        "https://pckix.com/wp-content/uploads/2025/06/Best-AI-Chatbots.jpg",
        width=650
    )

    st.markdown("""
## 📌 Project Overview

This AI-powered system is built using **Streamlit + Hugging Face Transformers**.

It allows users to:
- Ask questions to AI chatbot
- Get intelligent responses
- Analyze sentiment of text
- Detect language (English / Telugu)
- Store conversation history

This project demonstrates real-world **Natural Language Processing (NLP)** applications.
""")

    st.success("👉 Go to Dashboard or Chatbot")

# -----------------------------
# 📊 DASHBOARD PAGE
# -----------------------------
elif page == "Dashboard":

    st.title("📊 Project Dashboard")

    st.markdown("""
## ⚡ Features
- 🤖 AI Chatbot (FLAN-T5)
- 😊 Sentiment Analysis
- 🌍 Language Detection
- 💬 Chat History Memory

---

## 🧠 Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- FLAN-T5 Small Model
- RoBERTa Sentiment Model

---

## 🌍 Real World Applications
- YouTube comment analysis
- Social media sentiment tracking
- Customer feedback analysis
- AI chatbot systems

---

## 🎯 Why This Project is Useful
- Understands human emotions in text
- Automates question answering
- Real-time NLP application demo
""")

# -----------------------------
# 💬 CHATBOT PAGE
# -----------------------------
elif page == "Chatbot":

    st.title("💬 AI Chatbot")

    user_input = st.text_input("Ask your question:")

    if st.button("Send"):

        if user_input.strip():

            with st.spinner("Thinking..."):

                answer = get_answer(user_input)

            sentiment, score = get_sentiment(user_input)
            language = detect_language(user_input)

            st.session_state.chat_history.append({
                "question": user_input,
                "answer": answer,
                "sentiment": sentiment,
                "score": score,
                "language": language
            })

            st.success("Response Generated")

            st.markdown("## 🤖 Answer")
            st.write(answer)

            st.markdown("## 📊 Analysis")
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Score: {score}")
            st.write(f"Language: {language}")

        else:
            st.warning("Please enter a question")

    # -----------------------------
    # HISTORY
    # -----------------------------
    st.subheader("📜 Chat History")

    if st.session_state.chat_history:

        for chat in reversed(st.session_state.chat_history[-10:]):

            st.markdown(f"""
---
**Q:** {chat['question']}  

🤖 **A:** {chat['answer']}  

📊 Sentiment: {chat['sentiment']} | Score: {chat['score']}  
🌍 Language: {chat['language']}
""")

    else:
        st.info("No chat history yet")