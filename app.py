import streamlit as st
import tensorflow as tf
import numpy as np
import torch
import speech_recognition as sr
import io
import os
from gtts import gTTS
from PIL import Image
import ollama

# Optional: PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    st.warning("⚠️ reportlab not installed. PDF report feature will be disabled.")
    REPORTLAB_AVAILABLE = False

# ---------------------------
# Load Keras Model
# ---------------------------
@st.cache_resource
def load_keras_model(model_path="plant_disease_vgg16_optimized.keras"):
    if not os.path.exists(model_path):
        st.error(f"❌ Keras model not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Keras model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading Keras model: {str(e)}")
        return None

model = load_keras_model()

# ---------------------------
# Whisper Model Safe Load
# ---------------------------
@st.cache_resource
def load_whisper_safe():
    try:
        import whisper
        model = whisper.load_model("base")
        st.success("✅ Whisper model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Whisper model could not be loaded: {str(e)}")
        return None

whisper_model = load_whisper_safe()

# ---------------------------
# Disease Prediction (Keras)
# ---------------------------
def predict_disease(image):
    if model is None:
        st.warning("⚠️ Model not loaded. Cannot predict.")
        return None, None

    class_names = [
        "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
        "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
        "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
        "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
        "Orange Huanglongbing (Citrus Greening)",
        "Peach Bacterial Spot", "Peach Healthy",
        "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
        "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
        "Raspberry Healthy",
        "Soybean Healthy",
        "Squash Powdery Mildew",
        "Strawberry Leaf Scorch", "Strawberry Healthy",
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
        "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", 
        "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
    ]

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# ---------------------------
# Ollama AI Chat
# ---------------------------
def get_chat_response(user_input):
    prompt = f"""
    You are a plant disease expert. Answer based on scientific agricultural knowledge.
    Question: {user_input}
    """
    try:
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Ollama chat failed: {str(e)}"

# ---------------------------
# Speech Recognition
# ---------------------------
def recognize_speech():
    if whisper_model is None:
        st.warning("⚠️ Voice chat unavailable because Whisper model could not be loaded.")
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak clearly.")
        audio = recognizer.listen(source)
    try:
        audio_data = audio.get_wav_data()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)
        result = whisper_model.transcribe("temp_audio.wav")
        user_text = result["text"]
        if not user_text.strip():
            st.error("❌ No speech detected. Please try again.")
            return None
        st.success(f"🗣️ You said: {user_text}")
        return user_text
    except Exception as e:
        st.error(f"❌ Error in speech recognition: {str(e)}")
        return None

# ---------------------------
# Text-to-Speech
# ---------------------------
def text_to_speech(response_text):
    tts = gTTS(text=response_text, lang="en")
    audio_path = "response.mp3"
    tts.save(audio_path)
    return audio_path

# ---------------------------
# PDF Report
# ---------------------------
def generate_pdf(disease, confidence, ai_suggestions, chat_history):
    if not REPORTLAB_AVAILABLE:
        st.warning("⚠️ PDF report feature unavailable.")
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "🌿 Plant Disease Report")

    y -= 40
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Disease Prediction: {disease}")
    y -= 20
    c.drawString(50, y, f"Confidence: {confidence:.2f}%")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "AI Suggestions:")
    y -= 20
    c.setFont("Helvetica", 12)
    for line in ai_suggestions.split("\n"):
        c.drawString(60, y, line.strip())
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "User Questions & AI Responses:")
    y -= 20
    c.setFont("Helvetica", 12)
    for q, a in chat_history:
        c.drawString(50, y, f"Q: {q}")
        y -= 15
        for line in a.split("\n"):
            c.drawString(70, y, f"A: {line.strip()}")
            y -= 15
        y -= 10

    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌿 Plant Disease Detection & AI Assistant")

chat_history = []
disease, confidence, response = None, None, None

# Upload Image
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("🔍 Analyzing disease..."):
        disease, confidence = predict_disease(image)
        st.subheader("🔍 Prediction Result:")
        st.write(f"**Disease Type:** {disease}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("🩺 AI Suggestions:")
    query = f"What is {disease} and how can I treat it?"
    response = get_chat_response(query)
    st.write(response)

# Text Chat
user_input = st.text_input("💬 Ask a question about plant diseases:")
if user_input:
    ai_response = get_chat_response(user_input)
    st.write(f"**AI Answer:** {ai_response}")
    chat_history.append((user_input, ai_response))

# Voice Chat
if st.button("🎤 Ask with Voice"):
    voice_text = recognize_speech()
    if voice_text:
        ai_response = get_chat_response(voice_text)
        st.write(f"**AI Answer:** {ai_response}")
        chat_history.append((voice_text, ai_response))
        audio_file_path = text_to_speech(ai_response)
        with open(audio_file_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")

# Generate PDF
if st.button("📄 Download PDF Report"):
    if disease and confidence and response:
        pdf_buffer = generate_pdf(disease, confidence, response, chat_history)
        if pdf_buffer:
            st.download_button("⬇️ Download Report", data=pdf_buffer, file_name="plant_disease_report.pdf", mime="application/pdf")
    else:
        st.warning("⚠️ Please upload an image first to generate a report.")

