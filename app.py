# Keep all your existing imports

import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import pdfplumber
import os
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
import requests
import pyttsx3
import google.generativeai as genai
from geopy.geocoders import Nominatim
import torch
import torchvision.transforms as transforms
from pathlib import Path
import base64
medical_terms = []


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# API Keys
gemini_api_key = "AIzaSyASWgnselcBA5l5uFuNa2m-b9MbZDw97Oo"
geoapify_key = "715c9df9d4fb4a22bc3c842df2e23409"
genai.configure(api_key=gemini_api_key)

# Optional X-ray model
try:
    import torchxrayvision as xrv
    xray_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xray_model.eval()
    xray_classes = xrv.datasets.default_pathologies
except:
    xray_model = None
    xray_classes = []

# Load SciSpaCy
try:
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")
except:
    st.error("SciSpaCy not found. Install it manually.")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="🩺 Health Report Analyzer", layout="wide")
st.title("📄 AI-Powered Health Report Analyzer")
st.markdown("Upload a **photo, PDF, or X-ray image** of your health report. The app will predict your condition using AI & provide remedies or doctors nearby.")

# Background Image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local("pexels-tara-winstead-7723603.jpg")

# Fix for text readability
st.markdown(
    """
    <style>
    .main > div {
        background-color: rgba(255, 255, 255, 0.85);  /* White transparent */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== 🔄 Gemini Chatbot ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def ask_gemini_followup(user_msg):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        convo = model.start_chat(history=st.session_state.chat_history)
        convo.send_message(user_msg)
        st.session_state.chat_history.append({"role": "user", "parts": [user_msg]})
        st.session_state.chat_history.append({"role": "model", "parts": [convo.last.text]})
        return convo.last.text
    except Exception:
        return "❌ Gemini API error or quota exhausted. Try again later."

st.subheader("🩺 Symptom Checker & Doubt Bot")
with st.expander("💬 Ask your health-related doubts or describe symptoms"):
    user_question = st.text_input("👤 You:", key="symptom_chat")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔁 Refresh Conversation"):
            st.session_state.chat_history = []

    if user_question:
        response = ask_gemini_followup(user_question)
        if "symptom_chat_history" not in st.session_state:
         st.session_state.symptom_chat_history = []
        st.session_state.symptom_chat_history.append({"role": "user", "content": user_question})
        st.session_state.symptom_chat_history.append({"role": "bot", "content": response})
        st.markdown(f"🤖 **Bot:** {response}")


    if st.session_state.chat_history:
        with st.expander("🕘 Conversation History", expanded=False):
            for msg in st.session_state.chat_history:
                role = msg["role"].capitalize()
                content = msg["parts"][0]
                st.markdown(f"**{role}:** {content}")

# ========== File Upload ==========
uploaded_file = st.file_uploader("Upload Health Report (Image or PDF)", type=["png", "jpg", "jpeg", "pdf", "webp"])

# ========== Extraction + Prediction ==========
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    else:
        image = Image.open(file)
        return pytesseract.image_to_string(image)

def predict_disease_xray(image):
    try:
        image = image.convert("L").resize((224, 224))
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            output = xray_model(tensor)
            probs = torch.sigmoid(output)[0]
        top = torch.topk(probs, 3).indices.tolist()
        return [(xray_classes[i], round(probs[i].item() * 100, 2)) for i in top]
    except:
        return [("X-ray prediction failed", 0)]

def extract_medical_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def ask_health_status(text):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        prompt = f"""You are a medical expert AI. Based on this report text, summarize the health condition, flag critical issues, and suggest remedies or advice.

Report:
{text}

Reply with:
1. Summary
2. Health Score (0-100)
3. Color Code (Red / Yellow / Green)
4. Remedies
5. Suggest seeing a doctor if needed.
"""
        response = model.generate_content([prompt])
        return response.text.strip()
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return "❌ Gemini API quota exceeded. Please try again tomorrow or upgrade your plan."
        elif "API key not valid" in str(e):
            return "❌ Invalid Gemini API key. Please check your key in the code."
        else:
            return f"❌ Gemini error: {str(e)}"

# ========== Nearby Doctor Suggestion ==========
def find_doctors_nearby(lat, lon, api_key):
    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": "healthcare.hospital",
        "bias": f"proximity:{lon},{lat}",
        "limit": 10,
        "apiKey": api_key
    }
    try:
        res = requests.get(url, params=params)
        features = res.json().get("features", [])
        return [{
            "name": f["properties"].get("name", "Unknown"),
            "address": f["properties"].get("formatted", "N/A"),
            "phone": f["properties"].get("phone", "N/A"),
            "website": f["properties"].get("website", "N/A"),
            "lat": f["geometry"]["coordinates"][1],
            "lon": f["geometry"]["coordinates"][0],
        } for f in features if f["properties"].get("name")]
    except Exception as e:
        st.error(f"Geoapify error: {e}")
        return []

# ========== 🎤 Text-to-Speech ==========
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = pyttsx3.init()

def speak_text(text):
    engine = st.session_state.tts_engine
    engine.stop()  # Stop any previous
    engine.say(text)
    engine.runAndWait()

def stop_speaking():
    engine = st.session_state.tts_engine
    engine.stop()

# ========== Final Workflow ==========
if uploaded_file:
    with st.spinner("🔍 Processing..."):
        extracted_text = extract_text(uploaded_file)
        st.subheader("📝 Extracted Report Text")
        st.text_area("OCR Output", extracted_text, height=200)

        if uploaded_file.type.startswith("image/") and xray_model:
            image = Image.open(uploaded_file)
            st.subheader("🩻 X-ray Disease Prediction")
            for d, p in predict_disease_xray(image):
                st.write(f"**{d}** — {p}%")

        st.subheader("🧬 Medical Terms Detected")
        medical_terms = extract_medical_entities(extracted_text)
        st.session_state["medical_terms"] = medical_terms
        st.write(", ".join(medical_terms) if medical_terms else "No medical terms found.")

        st.subheader("🧐 AI Health Summary")
        ai_summary = ask_health_status(extracted_text)
        st.markdown(ai_summary)

        if "Health Score" in ai_summary:
            try:
                score_line = [line for line in ai_summary.split("\n") if "Health Score" in line][0]
                score = int("".join(filter(str.isdigit, score_line)))
                color = "🟢 Green" if score > 75 else "🟡 Yellow" if score > 50 else "🔴 Red"
                st.subheader("📊 Health Score")
                st.metric(label="Score", value=score)
                st.markdown(f"**Condition:** {color}")
            except:
                st.warning("Couldn't parse health score properly.")

        st.subheader("📍 Nearby Doctors (Geoapify)")
        user_input = st.text_input("🔍 Enter your city or area manually (e.g., Madhapur, Hyderabad):")

        if st.button("📍 Find Doctors in My Area") and user_input:
            try:
                geolocator = Nominatim(user_agent="health_app")
                location = geolocator.geocode(user_input + ", India", language="en")
                if location:
                    lat, lon = location.latitude, location.longitude
                    st.success(f"Using location: {location.address} (Lat: {lat}, Lon: {lon})")
                    with st.spinner("Searching nearby doctors..."):
                        results = find_doctors_nearby(lat, lon, geoapify_key)
                    if results:
                        for doc in results:
                            st.markdown(f"### 🏥 {doc['name']}")
                            st.write(f"📍 {doc['address']}")
                            st.write(f"📞 {doc['phone']}")
                            if doc["website"] != "N/A":
                                st.markdown(f"🌐 [Website]({doc['website']})")
                            st.markdown(f"🗺️ [Google Maps](https://www.google.com/maps/search/?api=1&query={doc['lat']},{doc['lon']})")
                            st.divider()
                    else:
                        st.warning("⚠️ No doctors found nearby.")
                else:
                    st.error("❌ Could not find location.")
            except Exception as e:
                st.error(f"Geo location error: {e}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("🔊 Read My Report"):
                speak_text(ai_summary)
        with colB:
            if st.button("⏹️ Stop Reading"):
                stop_speaking()

if not gemini_api_key or not geoapify_key:
    st.error("❌ API keys not set.")
    st.stop()


# -------------------- 🌿 HOME REMEDIES SECTION --------------------
from streamlit_extras.let_it_rain import rain

# Get extracted terms from report, if any
medical_terms = st.session_state.get("medical_terms", [])

# Get last symptom/doubt from chat history, if any
symptom_chat_history = st.session_state.get("symptom_chat_history", [])
last_user_symptom = ""
for chat in reversed(symptom_chat_history):
    if chat.get("role") == "user" and chat.get("content"):
        last_user_symptom = chat["content"]
        break

# Create the Home Remedies button
show_remedies = st.button("💊 Show Home Remedies", use_container_width=True)

if show_remedies:
    # Case 1: Use extracted medical terms
    if medical_terms:
        symptoms_for_prompt = ", ".join(medical_terms)
    # Case 2: Use symptom from chat
    elif last_user_symptom:
        symptoms_for_prompt = last_user_symptom
    # Case 3: Neither report nor chat
    else:
        st.info("⚠️ Please upload a report or ask a symptom in the doubt bot first.")
        symptoms_for_prompt = None

    if symptoms_for_prompt:
        st.subheader("🌱 AI-Suggested Home Remedies")
        with st.spinner("Generating remedies..."):
            try:
                prompt = f"""
                You are a certified health advisor. Suggest effective home remedies and daily tips for:
                {symptoms_for_prompt}.
                Focus on natural remedies, diet tips, hydration, sleep, and lifestyle practices. Be clear and concise.
                """

                # ✅ Use your active model
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(prompt)
                remedies = response.text

                st.markdown(remedies)

                rain(
                    emoji="🌿",
                    font_size=20,
                    falling_speed=5,
                    animation_length=1.5
                )

            except Exception as e:
                st.error(f"⚠️ Could not get remedies: {str(e)}")



