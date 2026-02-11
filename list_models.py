import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available embedding models:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"Model name: {m.name}")
