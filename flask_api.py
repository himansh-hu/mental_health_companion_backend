import os
import json
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
import firebase_admin
from firebase_admin import credentials, firestore
import pickle
from dotenv import load_dotenv
load_dotenv()

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

#  flask app initializing
app = Flask(__name__)
CORS(app)

# firebase initialization
FIREBASE_CRED_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "C:/Projects/mental_health_companion/backend/mental-health-companion-firebase-adminsdk-fbsvc-5d827075f3.json"
)

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firestore initialized successfully")
except Exception as e:
    print(f"❌ Firestore initialization failed: {e}")
    db = None

# loading models and tools
MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.pkl" 
LABEL_ENCODER_PATH = "label_encoder.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(" Model, tokenizer, and label encoder loaded successfully")
except Exception as e:
    print(f" Error loading model or tokenizer: {e}")
    model, tokenizer, label_encoder = None, None, None

# Preprocessing
MAX_SEQUENCE_LENGTH = 120

def preprocess_text(text):
    if not tokenizer:
        return None
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH,
                         padding='post', truncating='post')

# gitHub API model setup 
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MODEL_SLUG   = os.getenv("MODEL_SLUG", "meta/Meta-Llama-3.1-8B-Instruct")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set in environment or .env file")

client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)
# -----------------------------------------------

def query_azure_ai(prompt, emotion):
    try:
        response = client.complete(
            messages=[
                UserMessage(
                    f"The user feels {emotion}. Please respond compassionately.\n\n{prompt}"
                )
            ],
            model=MODEL_SLUG,
            temperature=0.8,
            max_tokens=2048,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" Azure AI request failed: {e}")
        return "The AI service is currently unavailable."

# prediction and response
@app.route("/predict", methods=["POST"])
def predict():
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "Model setup incomplete."}), 500

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text input missing."}), 400

    try:
        padded = preprocess_text(text)
        if padded is None:
            return jsonify({"error": "Tokenizer error."}), 500

        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        detected_emotion = label_encoder.inverse_transform([predicted_class])[0]

        ai_response = query_azure_ai(text, detected_emotion)

        return jsonify({
            "emotion": detected_emotion,
            "response": ai_response
        })

    except Exception as e:
        print(f"❌ /predict error: {e}")
        return jsonify({"error": str(e)}), 500

# model test
@app.route("/test_model", methods=["GET"])
def test_model():
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "Model setup incomplete."}), 500

    samples = [
        "I'm feeling very anxious today.",
        "Life feels so beautiful and exciting!",
        "I am tired and bored of everything.",
        "Why does no one care about me?",
        "I'm calm, just thinking about stuff."
    ]
    results = []

    for text in samples:
        processed = preprocess_text(text)
        prediction = model.predict(processed)
        pred_class = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([pred_class])[0]
        results.append({"text": text, "emotion": emotion})

    return jsonify(results)

# app launch point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

