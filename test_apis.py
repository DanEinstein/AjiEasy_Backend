import asyncio
import aiohttp
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


async def test_deepseek():
    """Test DeepSeek API connection"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in .env")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Say 'TEST OK'"}],
                    "max_tokens": 10
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ DeepSeek API: WORKING")
                    return True
                elif response.status == 402:
                    print("❌ DeepSeek API: PAYMENT REQUIRED - Needs billing setup")
                    return False
                elif response.status == 401:
                    print("❌ DeepSeek API: INVALID API KEY")
                    return False
                elif response.status == 429:
                    print("❌ DeepSeek API: RATE LIMIT EXCEEDED")
                    return False
                else:
                    print(f"❌ DeepSeek API: FAILED (Status {response.status})")
                    return False
    except asyncio.TimeoutError:
        print("❌ DeepSeek API: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ DeepSeek API: ERROR - {e}")
        return False


async def test_openrouter():
    """Test OpenRouter API connection"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://ajieasy.vercel.app",  # Updated for production
                    "X-Title": "AjiEasy Test"
                },
                json={
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [{"role": "user", "content": "Say 'TEST OK'"}],
                    "max_tokens": 10
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ OpenRouter API: WORKING")
                    return True
                elif response.status == 401:
                    print("❌ OpenRouter API: INVALID API KEY")
                    return False
                elif response.status == 429:
                    print("❌ OpenRouter API: RATE LIMIT EXCEEDED")
                    return False
                else:
                    error_text = await response.text()
                    print(f"❌ OpenRouter API: FAILED (Status {response.status}) - {error_text[:100]}")
                    return False
    except asyncio.TimeoutError:
        print("❌ OpenRouter API: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ OpenRouter API: ERROR - {e}")
        return False


async def test_groq():
    """Test Groq API connection (FREE)"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": "Say 'TEST OK'"}],
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"✅ Groq API: WORKING - Response: {content}")
                    return True
                elif response.status == 401:
                    print("❌ Groq API: INVALID API KEY")
                    return False
                elif response.status == 429:
                    print("❌ Groq API: RATE LIMIT EXCEEDED")
                    return False
                else:
                    error_text = await response.text()
                    print(f"❌ Groq API: FAILED (Status {response.status}) - {error_text[:100]}")
                    return False
    except asyncio.TimeoutError:
        print("❌ Groq API: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Groq API: ERROR - {e}")
        return False


async def test_gemini():
    """Test Gemini API connection"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try the model we know works
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content("Say 'TEST OK'")
        
        if response.text and 'TEST OK' in response.text.upper():
            print("✅ Gemini API: WORKING")
            return True
        else:
            print(f"❌ Gemini API: RESPONSE FORMAT ISSUE - Got: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API: ERROR - {e}")
        return False


async def test_backend_health():
    """Test if the backend API is running"""
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{backend_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Backend API: WORKING - {data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Backend API: FAILED (Status {response.status})")
                    return False
    except asyncio.TimeoutError:
        print("❌ Backend API: TIMEOUT - Service may not be running")
        return False
    except Exception as e:
        print(f"❌ Backend API: ERROR - {e}")
        return False


async def check_api_keys():
    """Check which API keys are present in .env"""
    print("\n🔑 API Keys Configuration Check:")
    print("=" * 40)
    
    keys = {
        "Gemini": os.getenv("GEMINI_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API_KEY"), 
        "OpenRouter": os.getenv("OPENROUTER_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY")
    }
    
    for service, key in keys.items():
        status = "✅ PRESENT" if key else "❌ MISSING"
        # Show first few chars of key for verification (but not full key for security)
        key_preview = f" ({key[:10]}...)" if key and len(key) > 10 else ""
        print(f"{service:12} {status}{key_preview}")


async def main():
    print("🔍 Testing All API Connections...")
    print("=" * 50)
    
    # First check what keys are configured
    await check_api_keys()
    
    print("\n🚀 Testing API Connectivity:")
    print("=" * 50)
    
    # Test all APIs
    gemini_ok = await test_gemini()
    deepseek_ok = await test_deepseek()
    openrouter_ok = await test_openrouter()
    groq_ok = await test_groq()
    backend_ok = await test_backend_health()
    
    print("=" * 50)
    
    # Summary
    working_apis = sum([gemini_ok, deepseek_ok, openrouter_ok, groq_ok])
    total_apis = 4
    
    print(f"\n📊 SUMMARY:")
    print(f"AI APIs: {working_apis}/{total_apis} Working")
    print(f"Backend: {'✅ WORKING' if backend_ok else '❌ OFFLINE'}")
    
    if working_apis == total_apis and backend_ok:
        print("🎉 All systems are working perfectly!")
    elif working_apis >= 2 and backend_ok:
        print("✅ Good! Multiple APIs are working and backend is online")
    elif working_apis >= 1 and backend_ok:
        print("⚠️  Limited functionality - some APIs are working")
    else:
        print("❌ Critical issues - check configuration")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not backend_ok:
        print("• Start the backend server: uvicorn main:app --reload")
        print("• Check if backend is running on the correct port")
    
    if not groq_ok:
        print("• Get FREE Groq API key: https://console.groq.com/keys")
    if not deepseek_ok:
        print("• Fix DeepSeek billing: https://platform.deepseek.com/billing")
    if not openrouter_ok:
        print("• Get OpenRouter key: https://openrouter.ai/keys")
    if not gemini_ok:
        print("• Check Gemini API key: https://aistudio.google.com/")


if __name__ == "__main__":
    asyncio.run(main())