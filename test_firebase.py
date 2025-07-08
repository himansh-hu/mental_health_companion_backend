import google.generativeai as genai
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3zxwEa38FdNNrXT4x4rRIwOEAT0bjXlQ"

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro")

    response = model.generate_content("How are you?")
    print(" Gemini Response:")
    print(response.text)
except Exception as e:
    print(" Gemini API Error:", e)
