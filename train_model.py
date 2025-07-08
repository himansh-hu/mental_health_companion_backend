import google.generativeai as genai
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3zxwEa38FdNNrXT4x4rRIwOEAT0bjXlQ"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import random

#  configure API
genai.configure(api_key=os.getenv("AIzaSyD3zxwEa38FdNNrXT4x4rRIwOEAT0bjXlQ"))

# loading datasets
train_file = "C:/Projects/mental_health_companion/data/train_cleaned.txt"
faq_file = "C:/Projects/mental_health_companion/data/Mental_Health_FAQ.csv"
json_path = "C:/Projects/mental_health_companion/data/intents.json"

data = pd.read_csv(train_file, sep=';', header=None, names=['text', 'emotion'])
faq_data = pd.read_csv(faq_file)

# loading conversational dataset
with open(json_path, "r", encoding="utf-8") as file:
    intents_data = json.load(file)

# extracting patterns from datasets
conv_texts, conv_labels, response_map = [], [], {}
for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        conv_texts.append(pattern)
        conv_labels.append(intent["tag"])
    response_map[intent["tag"]] = intent["responses"]

# merge datasets
faq_df = pd.DataFrame({'text': faq_data['Questions'], 'emotion': ['faq'] * len(faq_data)})
conv_df = pd.DataFrame({'text': conv_texts, 'emotion': conv_labels})
combined_data = pd.concat([data, faq_df, conv_df], ignore_index=True)

# maintaining inconsistencies
combined_data['emotion'] = combined_data['emotion'].replace({'joy': 'happy'})

# encode labels
label_encoder = LabelEncoder()
combined_data['emotion'] = label_encoder.fit_transform(combined_data['emotion'])

# tokenization
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(combined_data['text'])
X_sequences = tokenizer.texts_to_sequences(combined_data['text'])
X_padded = pad_sequences(X_sequences, maxlen=120)
y_encoded = combined_data['emotion'].values

# balance dataset using SMOTE
min_samples = 5
filtered_indices = [i for i, label in enumerate(y_encoded) if Counter(y_encoded)[label] >= min_samples]
X_filtered = X_padded[filtered_indices]
y_filtered = y_encoded[filtered_indices]

# find the smallest class size
min_samples_per_class = min(Counter(y_filtered).values())

# adjusting k_neighbors dynamically
smote_k = max(1, min(min_samples_per_class - 1, 5))

# apply SMOTE with adjusted k_neighbors
smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=smote_k)


X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# model architecture
model = Sequential([
    Embedding(input_dim=15000, output_dim=128, mask_zero=True),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    BatchNormalization(),
    LSTM(64, kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=15, batch_size=32, class_weight=class_weight_dict)

# evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.2f}')
print(f'Test Accuracy: {test_acc:.2f}')

#  API Integration
def get_response(user_input):
    """Generate chatbot response using Gemini API and model predictions."""
    user_seq = tokenizer.texts_to_sequences([user_input])
    user_padded = pad_sequences(user_seq, maxlen=120)
    predicted_class = np.argmax(model.predict(user_padded), axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    if predicted_label in response_map:
        return random.choice(response_map[predicted_label])
    elif predicted_label == "faq":
        return "This seems like a frequently asked question. Please refer to our FAQ section."
    else:
        # API for responses
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        gemini_response = gemini_model.generate_content(user_input)
        return gemini_response.text if gemini_response else "I'm here to help. Can you elaborate on your issue?"

user_input = "I feel so alone and hopeless"
print("Bot Response:", get_response(user_input))

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
#confirmation
print("Tokenizer and Label Encoder saved successfully")


# save model
model.save("sentiment_model.keras")
