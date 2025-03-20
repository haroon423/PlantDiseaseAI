import streamlit as st
import tensorflow as tf
import numpy as np
import torch
import speech_recognition as sr
import whisper
from gtts import gTTS
from PIL import Image
import ollama
import io
import os

# **Load Plant Disease Classification Model (VGG16)**
@st.cache_resource
def load_vgg_model():
    return tf.keras.models.load_model("plant_disease_vgg16_optimized.h5")

vgg_model = load_vgg_model()

# **Load Whisper Model**
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy

whisper_model = load_whisper_model()

# **Predict Disease from Image**
def predict_disease(image):
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
    
    image = np.array(image.resize((224, 224))) / 255.0  # Resize & Normalize
    image = np.expand_dims(image, axis=0)
    prediction = vgg_model.predict(image)
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

# **AI Chatbot using Ollama**
def get_chat_response(user_input):
    prompt = f"""
    You are a plant disease expert. Answer based on scientific agricultural knowledge.
    Question: {user_input}
    """
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# **Speech Recognition using Whisper**
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
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

# **Text-to-Speech (TTS) using gTTS**
def text_to_speech(response_text):
    tts = gTTS(text=response_text, lang="en")
    audio_path = "response.mp3"
    tts.save(audio_path)
    return audio_path  # Return path of the saved audio file

# **Streamlit UI**
st.title("🌿 Plant Disease Detection & AI Assistant")

# **Upload an Image**
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # **Predict Disease**
    with st.spinner("Analyzing disease..."):
        disease, confidence = predict_disease(image)
        st.subheader("🔍 Prediction:")
        st.write(f"**Disease Type:** {disease}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    # **AI Recommendations**
    st.subheader("🩺 AI Suggestions:")
    query = f"What is {disease} and how can I treat it?"
    response = get_chat_response(query)
    st.write(response)

# **Text-based Chatbot**
user_input = st.text_input("💬 Type your question:")
if user_input:
    ai_response = get_chat_response(user_input)
    st.write(f"**AI Answer:** {ai_response}")

# **Voice-based Chatbot**
if st.button("🎤 Ask with Voice"):
    voice_text = recognize_speech()
    if voice_text:
        ai_response = get_chat_response(voice_text)
        st.write(f"**AI Answer:** {ai_response}")

        # **Convert AI Response to Speech and Play in UI**
        audio_file_path = text_to_speech(ai_response)
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
