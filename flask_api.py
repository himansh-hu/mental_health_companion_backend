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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Firebase setup - works on both local and Railway
def setup_firebase():
    """
    Set up Firebase connection
    - First tries to use environment variables (for Railway)
    - Then falls back to local JSON file (for development)
    """
    try:
        if not firebase_admin._apps:
            # Check if we have environment variables (Railway deployment)
            if os.environ.get('FIREBASE_PROJECT_ID'):
                print(" Using Firebase environment variables...")
                firebase_config = {
                    "type": "service_account",
                    "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
                    "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
                    "private_key": os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                    "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
                    "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL')
                }
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                print(" Firebase initialized with environment variables")
            else:
                # Use local file path (for development)
                print(" Using local Firebase credentials file...")
                local_path = os.getenv(
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "mental-health-companion-firebase-adminsdk-fbsvc-5d827075f3.json"
                )
                if os.path.exists(local_path):
                    cred = credentials.Certificate(local_path)
                    firebase_admin.initialize_app(cred)
                    print(" Firebase initialized with local file")
                else:
                    print("⚠️ Firebase credentials file not found, continuing without Firebase")
                    return None
                    
        return firestore.client()
    except Exception as e:
        print(f" Firebase initialization failed: {e}")
        return None

# Initialize Firebase database
db = setup_firebase()

# Load AI models and tools
def load_ai_models():
    """
    Load the sentiment analysis model and preprocessing tools
    - Model file: sentiment_model.keras
    - Tokenizer: tokenizer.pkl (converts text to numbers)
    - Label encoder: label_encoder.pkl (converts predictions back to emotions)
    """
    print(" Loading AI models...")
    
    # File paths - these work on both Windows (local) and Linux (Railway)
    model_path = "sentiment_model.keras"
    tokenizer_path = "tokenizer.pkl" 
    label_encoder_path = "label_encoder.pkl"
    
    try:
        # Check if all files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
            
        # Load the main AI model
        model = tf.keras.models.load_model(model_path)
        
        # Load the text preprocessor (tokenizer)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
            
        # Load the emotion label converter
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
            
        print(" All AI models loaded successfully!")
        return model, tokenizer, label_encoder
        
    except Exception as e:
        print(f" Error loading AI models: {e}")
        return None, None, None

# Load the models when app starts
model, tokenizer, label_encoder = load_ai_models()

# Text preprocessing settings
MAX_SEQUENCE_LENGTH = 120  # Maximum length of text we can process

def preprocess_text(text):
    """
    Convert user text into numbers that our AI model can understand
    - Takes raw text input
    - Converts to number sequences
    - Pads/trims to fixed length
    """
    if not tokenizer:
        return None
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH,
                         padding='post', truncating='post')

# Setup GitHub AI model for generating responses
def setup_github_ai():
    """
    Set up connection to GitHub's AI model
    - Used for generating compassionate responses
    - Requires GITHUB_TOKEN environment variable
    """
    github_token = os.getenv("GITHUB_TOKEN")
    model_name = os.getenv("MODEL_SLUG", "meta/Meta-Llama-3.1-8B-Instruct")
    
    if not github_token:
        print(" GITHUB_TOKEN not found - AI responses will be limited")
        return None
        
    try:
        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
        )
        print(" GitHub AI client initialized successfully")
        return client
    except Exception as e:
        print(f" GitHub AI setup failed: {e}")
        return None

# Initialize GitHub AI client
client = setup_github_ai()

def query_azure_ai(prompt, emotion):
    """
    Get a compassionate AI response based on detected emotion
    - Takes user's text and detected emotion
    - Returns helpful, empathetic response
    """
    if not client:
        return "I understand you're feeling this way. While I can't provide detailed responses right now, please know that your feelings are valid."
        
    try:
        response = client.complete(
            messages=[
                UserMessage(
                    f"The user feels {emotion}. Please respond compassionately and helpfully.\n\nUser said: {prompt}"
                )
            ],
            model=os.getenv("MODEL_SLUG", "meta/Meta-Llama-3.1-8B-Instruct"),
            temperature=0.8,
            max_tokens=2048,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" AI response generation failed: {e}")
        return "I understand you're going through something difficult. While I'm having trouble generating a detailed response right now, please know that your feelings matter and it's okay to seek support."

# Health check endpoint - tells us if the app is running properly
@app.route("/", methods=["GET"])
def health_check():
    """
    Simple endpoint to check if our app is working
    Shows status of all components
    """
    status = {
        "status": "healthy",
        "message": "Mental Health Companion Backend is running!",
        "components": {
            "ai_model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "label_encoder_loaded": label_encoder is not None,
            "firebase_connected": db is not None,
            "github_ai_connected": client is not None
        }
    }
    return jsonify(status)

# Main prediction endpoint - analyzes emotions and provides responses
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main endpoint for emotion analysis
    - Receives user text
    - Detects emotion using AI model
    - Generates compassionate response
    """
    # Check if our AI models are loaded
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "AI models not loaded properly. Please try again later."}), 500

    # Get the user's text from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "Please send your message in JSON format."}), 400
        
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please include your message in the 'text' field."}), 400

    try:
        # Step 1: Convert text to numbers for AI processing
        padded = preprocess_text(text)
        if padded is None:
            return jsonify({"error": "Error processing your text. Please try again."}), 500

        # Step 2: Use AI model to predict emotion
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        detected_emotion = label_encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction))

        # Step 3: Generate helpful response based on emotion
        ai_response = query_azure_ai(text, detected_emotion)

        # Step 4: Return results
        return jsonify({
            "emotion": detected_emotion,
            "confidence": round(confidence * 100, 2),  # Convert to percentage
            "response": ai_response,
            "original_text": text
        })

    except Exception as e:
        print(f" Prediction error: {e}")
        return jsonify({"error": "Something went wrong while analyzing your message. Please try again."}), 500

# Test endpoint - shows how the model works with sample texts
@app.route("/test_model", methods=["GET"])
def test_model():
    """
    Test endpoint to see how our AI model performs
    Uses sample texts to show different emotion detections
    """
    # Check if models are loaded
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "AI models not loaded properly."}), 500

    # Sample texts representing different emotions
    samples = [
        "I'm feeling very anxious today.",
        "Life feels so beautiful and exciting!",
        "I am tired and bored of everything.",
        "Why does no one care about me?",
        "I'm calm, just thinking about stuff."
    ]
    
    results = []

    try:
        for text in samples:
            # Process each sample text
            processed = preprocess_text(text)
            prediction = model.predict(processed)
            pred_class = np.argmax(prediction)
            emotion = label_encoder.inverse_transform([pred_class])[0]
            confidence = float(np.max(prediction))
            
            results.append({
                "text": text, 
                "emotion": emotion,
                "confidence": round(confidence * 100, 2)
            })

        return jsonify({
            "message": "Model test completed successfully!",
            "test_results": results
        })
        
    except Exception as e:
        print(f" Model test error: {e}")
        return jsonify({"error": "Error during model testing."}), 500

# App startup configuration
if __name__ == "__main__":
    # Get port from environment (Railway sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    print(f" Starting Mental Health Companion Backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)