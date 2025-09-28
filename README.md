# 🩺 Health Report Analyzer

An AI-powered Streamlit app that analyzes health reports, extracts medical terms, predicts diseases from X-rays, provides summaries with Gemini AI, suggests nearby doctors (Geoapify), and even supports voice-based report reading.  

---

## 🚀 Features
- 📄 OCR & NLP to extract text from PDFs and images  
- 🧠 Gemini AI to summarize health reports & clarify symptoms  
- 🩻 Deep learning X-ray disease prediction (COVID, TB, Pneumonia)  
- 🗺 Nearby doctor suggestions with Geoapify  
- 🔊 Text-to-speech health report reader  
- 🌿 Home remedies & health tips  

---

## 📦 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/AkhilVadla31/Health-Plus-AI-Analyzer.git
cd <repo-name>

2️⃣Create and activate a virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 Setup API Keys

Create a file named .env in the project root.

Add your keys inside it:

GEMINI_API_KEY=your_gemini_api_key_here
GEOAPIFY_API_KEY=your_geoapify_api_key_here

▶️ Run the App
streamlit run app.py

















