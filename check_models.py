import os
import google.generativeai as genai

print("Attempting to list available models...")

try:
    # 1. Configure your API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("-" * 30)
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set the key before running this script:")
    print("(venv) > set GEMINI_API_KEY=your_api_key_here")
    print("-" * 30)
    exit()
except Exception as e:
    print(f"An error occurred during configuration: {e}")
    exit()

print("API key configured.")
print("Fetching models...\n")

# 2. List all models and find the ones that can generate content
try:
    usable_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            usable_models.append(m.name)

    if not usable_models:
        print("No usable models found for 'generateContent'.")
        print("Please check your API key permissions in Google AI Studio.")
    else:
        print("--- Usable Models for 'generateContent' ---")
        for model_name in usable_models:
            print(f"> {model_name}")
        print("---------------------------------------------")
        print("\nSUCCESS: Copy one of the model names above (e.g., 'models/gemini-pro')")
        print("and paste it into your ai_service.py file.")

except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("This may be an API key issue or a network problem.")