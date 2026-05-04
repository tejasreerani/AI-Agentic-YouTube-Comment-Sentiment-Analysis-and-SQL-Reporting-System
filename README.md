# 🎬 YouTube Comment Sentiment & Language Analysis Project

## 📌 1. Introduction

This project is an AI-powered YouTube analytics system that collects comments from YouTube videos and performs:

🌍 Language detection (English, Telugu, Mixed, Emojis)
😊 Sentiment analysis (Positive / Negative / Neutral)
📊 Data storage using SQLite database
📈 Visualization using Matplotlib & Seaborn
🤖 Interactive chatbot for real-time comment analysis

The system uses Hugging Face Transformer models and YouTube Data API v3 to extract and analyze real-world social media data.

## 🎯 2. Project Objective

The main goal of this project is:
-> To understand public opinion from YouTube comments
-> To detect multilingual content (English + Telugu + mixed text)
-> To classify sentiments using deep learning models
-> To store structured data for analytics
-> To visualize insights for decision making

## ⚙️ 3. Technologies Used

🐍 Python
📡 YouTube Data API v3
🤗 Hugging Face Transformers
🧠 NLP (Natural Language Processing)
🗄️ SQLite Database
📊 Pandas, Matplotlib, Seaborn
🔤 wordfreq (language detection)

## 📦 4. requirements.txt 
-> pandas
-> matplotlib
-> seaborn
-> google-api-python-client
-> transformers
-> huggingface_hub
-> torch
-> wordfreq
-> requests
-> numpy

## 🧠 5. Machine Learning Models Used

### 🌍 Language Detection Model
papluca/xlm-roberta-base-language-detection

### 😊 Sentiment Analysis Model
cardiffnlp/twitter-roberta-base-sentiment-latest

## 🔄 6. Project Workflow

-> Input YouTube Channel URL
-> Extract Channel ID using YouTube API
-> Fetch latest videos
-> Extract comments from each video
-> Clean and preprocess text
-> Detect language (English / Telugu / Mixed)
-> Perform sentiment analysis
-> Store results in SQLite database
-> Generate CSV reports
-> Visualize sentiment & language distribution
-> Run chatbot for real-time prediction

## 🧩 7. Features

📺 YouTube video & comment scraping
🌐 Multilingual language detection
💬 Sentiment classification
🗃️ Database storage (SQLite)
📊 Data visualization
🤖 AI chatbot for instant analysis
📄 Export reports (CSV format)

## 🗄️ 8. Database Schema

The project uses SQLite with 4 main tables:
* channels → stores channel info
* videos → stores video details
* comments → stores comment analysis
* video_summary → sentiment summary per video
* channel_summary → overall channel analysis

## 📊 9. Visualizations

The system generates:
-> Sentiment distribution graph 📈
-> Language distribution graph 🌍
->Video-wise sentiment summary 📊

## 💬 10. Chatbot Feature

The project includes an intelligent chatbot that:
-> Accepts user input comment
-> Detects language
-> Predicts sentiment
-> Returns confidence score

* Example:
Comment: "Super movie bro 🔥"Language: English + TeluguSentiment: PositiveScore: 0.95

## 📁 11. Project Structure

youtube-sentiment-analysis/│├── main.py├── requirements.txt├── README.md│├── data/│   ├── comments_report.csv│   ├── video_summary.csv│   ├── channel_summary.csv│├── database/│   └── youtube_analysis.db│└── assets/    ├── sentiment_plot.png    ├── language_plot.png

## 🚀 12. How to Run the Project

Step 1: Clone repository
git clone https://github.com/yourusername/youtube-sentiment-analysis.gitcd youtube-sentiment-analysis
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Add API keys
Set environment variables:
YOUTUBE_API_KEY=your_api_keyHF_TOKEN=your_huggingface_token
Step 4: Run project
python main.py

## 📦 13. Output Generated

-> Sentiment classification results
-> Language detection results
-> CSV reports
-> SQLite database
-> Graph visualizations
-> Chatbot responses

## 📈 14. Future Improvements

🌐 Streamlit dashboard UI
⚡ Real-time live YouTube monitoring
📱 Mobile-friendly web app
🤖 Advanced transformer fine-tuning
📊 Dashboard analytics panel

## 👨‍💻 15. Author
Name: Lakshmi Tejasree Kurmapu

## 🏁 16. Conclusion
This project successfully demonstrates how Natural Language Processing and Machine Learning can be applied to real-world social media data.
It helps in understanding:
-> Audience reaction
-> Multilingual communication patterns
-> Sentiment trends across YouTube content

This makes it useful for:
-> Content creators
-> Data analysts
-> Marketing teams
-> AI researchers
