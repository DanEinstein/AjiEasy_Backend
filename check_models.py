import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def list_available_models():
    print("🔍 Checking available Gemini models...")
    
    try:
        # List all available models
        models = genai.list_models()
        
        print("\n📋 AVAILABLE MODELS:")
        print("=" * 60)
        
        gemini_models = []
        for model in models:
            if 'gemini' in model.name.lower():
                gemini_models.append(model)
                print(f"🔹 {model.name}")
                print(f"   - Supported Methods: {', '.join(model.supported_generation_methods)}")
                print(f"   - Description: {model.description}")
                print()
        
        print(f"🎯 Total Gemini models found: {len(gemini_models)}")
        
        # Show recommended models for your use case
        print("\n💡 RECOMMENDED MODELS FOR INTERVIEW QUESTIONS:")
        recommended = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest', 
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-thinking-exp-1219',
            'gemini-2.0-pro-exp-02-05'
        ]
        
        for model_name in recommended:
            if any(model_name in model.name for model in gemini_models):
                print(f"✅ {model_name} - AVAILABLE")
            else:
                print(f"❌ {model_name} - NOT AVAILABLE")
                
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    list_available_models()