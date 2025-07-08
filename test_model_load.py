import tensorflow as tf

MODEL_PATH = "C:/Projects/mental_health_companion/backend/sentiment_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully")
except Exception as e:
    print(f" Model loading failed: {e}")
