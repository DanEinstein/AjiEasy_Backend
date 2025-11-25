import os
import json
import re
import aiohttp
import asyncio
from typing import Dict, Optional, List, Sequence
import google.generativeai as genai
from config import settings

# Lazy Gemini loader (fixes startup failures)
_gemini_model = None
_GEMINI_AVAILABLE = False

def get_gemini_model():
    """Lazy-load Gemini with retries—called per request, not at startup."""
    global _gemini_model, _GEMINI_AVAILABLE

    if _gemini_model is not None:
        return _gemini_model, _GEMINI_AVAILABLE

    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key or not api_key.strip():
        print("⚠️ Gemini API key missing")
        _GEMINI_AVAILABLE = False
        return None, False

    try:
        genai.configure(api_key=api_key.strip())

        # Correct model IDs (no 'models/' prefix; prioritize 2.5 Flash)
        working_models = [
            'gemini-2.5-flash',  # Stable GA—fast, cheap, 1M context<grok-card data-id="05a288" data-type="citation_card"></grok-card>
            'gemini-2.5-flash-preview-10-25',  # Latest preview for image/thinking features<grok-card data-id="fc8882" data-type="citation_card"></grok-card>
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro'
        ]

        for model_name in working_models:
            try:
                print(f"🔄 Testing Gemini model: {model_name}")
                model = genai.GenerativeModel(
                    model_name,
                    generation_config={"temperature": 0.7}
                )
                # Quick test
                response = model.generate_content("Test", stream=False)
                if response.text:
                    print(f"✅ Gemini ready: {model_name}")
                    _gemini_model = model
                    _GEMINI_AVAILABLE = True
                    return model, True
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue

        print("❌ No working Gemini model")
        _GEMINI_AVAILABLE = False
        return None, False

    except Exception as e:
        print(f"❌ Gemini setup failed: {e}")
        _GEMINI_AVAILABLE = False
        return None, False

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _extract_text_from_gemini(response) -> str:
    """
    Safely extract text from Gemini responses regardless of format.
    """
    if not response:
        return ""

    if hasattr(response, "text") and response.text:
        return response.text

    try:
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                part = candidate.content.parts[0]
                if hasattr(part, "text"):
                    return part.text
    except Exception:
        pass

    return ""


async def _groq_completion(
    messages: Sequence[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            GROQ_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.GROQ_API_KEY}"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            error_text = await response.text()
            raise RuntimeError(f"Groq API error {response.status}: {error_text[:200]}")


def _extract_json_array(text: str) -> Optional[List[Dict]]:
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class FreeAIService:
    def __init__(self):
        # API key validation
        self.groq_api_key = getattr(settings, 'GROQ_API_KEY', None)
        self.deepseek_enabled = bool(settings.DEEPSEEK_API_KEY)
        self.openrouter_enabled = bool(settings.OPENROUTER_API_KEY)
<<<<<<< HEAD
        self.groq_enabled = bool(self.groq_api_key and str(self.groq_api_key).strip())

=======
        self.groq_enabled = bool(self.groq_api_key and len(str(self.groq_api_key).strip()) > 0)

        # Debug logging
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        print("=" * 60)
        print("🔍 API KEY VALIDATION:")
        print(f"   Groq Enabled: {self.groq_enabled}")
        print(f"   DeepSeek Enabled: {self.deepseek_enabled}")
        print(f"   OpenRouter Enabled: {self.openrouter_enabled}")
<<<<<<< HEAD
        print(f"   Gemini Available: {GEMINI_ENABLED}")
=======
        print(f"   Gemini Lazy-Load: True")  # Now always attempts
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        print("=" * 60)

    async def generate_topic_recommendations(self) -> List[Dict]:
        """Generate trending interview topics for 2025 using Gemini"""
        fallback_topics = [
            {"topic": "AI Agents & LLMs", "trend": "High Demand", "icon": "fa-robot"},
            {"topic": "Rust Programming", "trend": "Growing Fast", "icon": "fa-cogs"},
            {"topic": "Cloud Security", "trend": "Critical", "icon": "fa-shield-alt"},
            {"topic": "React Server Components", "trend": "Standard", "icon": "fa-code"},
            {"topic": "System Design (Scalability)", "trend": "Evergreen", "icon": "fa-sitemap"}
        ]

<<<<<<< HEAD
        if not GEMINI_ENABLED or not gemini_model:
=======
        model, available = get_gemini_model()
        if not available:
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
            print("⚠️ Gemini unavailable, using fallback recommendations")
            return fallback_topics

        prompt = """Suggest 5 trending software engineering interview topics for 2025.
Focus on emerging tech like AI, Edge Computing, etc.

Return ONLY a valid JSON array with this structure:
[
    {
        "topic": "Topic Name",
        "trend": "Why it's trending (2-3 words)",
        "icon": "FontAwesome class (e.g., fa-robot)"
    }
]"""

        try:
            print("🔮 Generating recommendations with Gemini...")
            response = await asyncio.to_thread(
<<<<<<< HEAD
                gemini_model.generate_content,
=======
                model.generate_content,
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
<<<<<<< HEAD
                    response_mime_type="application/json"
=======
                    response_mime_type="application/json"  # Native JSON for 2.5+
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
                )
            )

            if response.text:
<<<<<<< HEAD
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
=======
                # Try direct JSON first (no regex if mime_type works)
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    # Fallback regex
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', response.text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))

            print("❌ Failed to parse Gemini response")
            return fallback_topics

        except Exception as e:
            print(f"❌ Gemini Recommendation Error: {e}")
            return fallback_topics
<<<<<<< HEAD
        
=======

>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
    async def generate_quiz(
        self,
        topic: str,
        difficulty: str = "medium",
        question_count: int = 10,
        focus_areas: str = ""
    ) -> Dict:
        """Generate quiz using Groq (Primary), Gemini (Fallback)"""
        if not settings.ENABLE_QUIZ:
            return {"error": "Quiz feature is disabled"}

        question_count = min(question_count, settings.MAX_QUIZ_QUESTIONS)
<<<<<<< HEAD
        
        print(f"🎯 Starting quiz generation for: {topic}")
        
        if self.groq_enabled:
            quiz_data = await self._generate_groq_quiz(topic, difficulty, question_count, focus_areas)
            if quiz_data and "error" not in quiz_data:
                return quiz_data
            print(f"❌ Groq quiz generation failed: {quiz_data.get('error', 'Unknown error')}")
        
        # Final fallback - Enhanced local quiz
        print("🔄 Using enhanced local quiz generator...")
=======

        # Try Groq first
        if self.groq_enabled:
            print(f"🔄 Generating Quiz with Groq for: {topic}")
            groq_quiz = await self._generate_groq_quiz(topic, difficulty, question_count, focus_areas)
            if groq_quiz and "error" not in groq_quiz:
                return groq_quiz

        # Fallback to Gemini
        print(f"🔄 Fallback: Generating Quiz with Gemini for: {topic}")
        gemini_quiz = await self._generate_gemini_quiz(topic, difficulty, question_count, focus_areas)
        if gemini_quiz and "error" not in gemini_quiz:
            return gemini_quiz

        # Last resort: local
        print("⚠️ Using local fallback for quiz")
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        return self._generate_enhanced_local_quiz(topic, difficulty, question_count, focus_areas)

    async def _generate_groq_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Optional[Dict]:
        """Generate quiz using Groq API"""
        prompt = f"""Create {question_count} multiple-choice interview questions about "{topic}" at {difficulty} difficulty level.
{f"Focus areas: {focus_areas}" if focus_areas else ""}

Return ONLY valid JSON array with this exact structure:
[
    {{
        "id": 1,
        "question": "Question text?",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": 0,
        "explanation": "Why?",
        "hint": "Hint",
        "type": "technical",
        "difficulty": "{difficulty}"
    }}
]"""

        try:
            response_text = await self._call_groq_api(prompt, max_tokens=4000, json_mode=True)  # Enable JSON mode
            if response_text:
                clean_text = response_text.replace('```json', '').replace('```', '').strip()
                try:
                    questions = json.loads(clean_text)  # Direct parse if JSON mode
                except json.JSONDecodeError:
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON found")

                print(f"✅ Groq generated {len(questions)} quiz questions")
                return {
                    "topic": topic,
                    "difficulty": difficulty,
                    "questions": questions,
                    "totalQuestions": len(questions),
                    "source": "groq"
                }
        except Exception as e:
            print(f"❌ Groq Quiz Gen Error: {e}")
        return None

    async def _generate_gemini_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Optional[Dict]:
        """Generate quiz using Gemini API (Fallback)"""
        model, available = get_gemini_model()
        if not available:
            return None

        prompt = f"""Create {question_count} multiple-choice interview questions about "{topic}" at {difficulty} difficulty level.
{f"Focus areas: {focus_areas}" if focus_areas else ""}

Return ONLY valid JSON array with this exact structure:
[
    {{
        "id": 1,
        "question": "Question text?",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": 0,
        "explanation": "Why?",
        "hint": "Hint",
        "type": "technical",
        "difficulty": "{difficulty}"
    }}
]"""

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                    response_mime_type="application/json"  # Key fix for consistency
                )
            )

            if response.text:
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                try:
                    questions = json.loads(clean_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON found")

                print(f"✅ Gemini generated {len(questions)} quiz questions")
                return {
                    "topic": topic,
                    "difficulty": difficulty,
                    "questions": questions,
                    "totalQuestions": len(questions),
                    "source": "gemini"
                }
        except Exception as e:
            print(f"❌ Gemini Quiz Gen Error: {e}")
        return None

    async def generate_questions(
        self,
        topic: str,
        job_description: str,
        interview_type: str,
        company_nature: str
    ) -> List[Dict]:
        """Generate interview questions using Gemini (Primary)"""
        model, available = get_gemini_model()
        if not available:
            print("⚠️ Gemini unavailable, using local fallback")
            return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

        prompt = f"""Generate 8 professional interview questions for:
TOPIC: {topic}
JOB ROLE: {job_description}
INTERVIEW TYPE: {interview_type}
COMPANY TYPE: {company_nature}

Create a mix of:
- 3 Technical questions
- 3 Behavioral questions
- 2 Situational questions

Return ONLY a valid JSON array with this exact structure:
[
    {{
        "question": "Question text?",
        "type": "technical",
        "difficulty": "medium",
        "explanation": "Brief explanation of what this evaluates"
    }}
]"""

        print(f"🔄 Generating Questions with Gemini for: {topic}")
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                    response_mime_type="application/json"
                )
            )

            print(f"✅ Gemini Response Received")
            if response.text:
                print(f"📝 Raw Response Length: {len(response.text)} chars")
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                try:
                    questions = json.loads(clean_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"No JSON array. First 200 chars: {clean_text[:200]}")

                print(f"✅ Successfully parsed {len(questions)} questions")
                return questions
        except Exception as e:
            print(f"❌ Gemini Question Gen Error: {type(e).__name__}: {e}")

        # Fallback
        print("⚠️ Using local fallback for questions")
        return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

    async def generate_analytics(self, user_data: Dict) -> Dict:
        """Generate analytics insights using Gemini"""
        model, available = get_gemini_model()
        if not available:
            print("⚠️ Gemini unavailable for analytics—using fallback")
            return self._fallback_analytics()

        prompt = f"""Analyze this user's interview preparation performance and provide insights:
User Data: {json.dumps(user_data)}

Return ONLY a valid JSON object with this structure:
{{
    "performance_trend": [65, 70, ...], // 7 data points representing progress
    "topic_mastery": {{"Topic A": 80, "Topic B": 60}},
    "average_score": 75,
    "recommendations": ["Rec 1", "Rec 2", "Rec 3"],
    "strength_areas": ["Area 1", "Area 2"],
    "weakness_areas": ["Area 1", "Area 2"]
}}"""

        print(f"🔄 Generating Analytics with Gemini")
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                try:
                    return json.loads(clean_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
        except Exception as e:
            print(f"❌ Gemini Analytics Error: {e}")

        return self._fallback_analytics()

    def _fallback_analytics(self) -> Dict:
        return {
            "performance_trend": [60, 65, 70, 72, 75, 78, 80],
            "topic_mastery": {"General": 70},
            "average_score": 70,
            "recommendations": ["Keep practicing to improve your score.", "Try different topics."],
            "strength_areas": ["Consistency"],
            "weakness_areas": ["Complex topics"]
        }

    async def send_chat_message(self, topic: str, message: str, history: List[Dict] = []) -> str:
        """Send chat message using Groq (Primary)"""
        system_prompt = f"You are AjiEasy AI, an expert interview coach specialized in {topic}. Provide helpful, concise, and encouraging advice."

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        if self.groq_enabled:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.groq_api_key}"
                        },
                        json={
                            "model": "llama-3.1-70b-versatile",  # Or fallback: "llama3-70b-8192"
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 1000
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"❌ Groq Chat Error: {e}")

        # Fallback to OpenRouter
        if self.openrouter_enabled:
            return await self._generate_openrouter_chat(messages)

        return "I'm currently offline, but keep practicing! You're doing great."

    async def _call_groq_api(self, prompt: str, max_tokens: int = 1000, json_mode: bool = False) -> Optional[str]:
        """Helper to call Groq API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama-3.1-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.groq_api_key}"
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        print(f"❌ Groq API Error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            print(f"❌ Groq API Exception: {e}")
            return None

    async def _generate_deepseek_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """Generate quiz using DeepSeek API (unchanged, but integrated if needed)"""
        # Your existing code here—call from generate_quiz if you want to add as fallback
        prompt = f"""Create {question_count} multiple-choice interview questions about {topic} at {difficulty} difficulty.
{f"Focus on: {focus_areas}" if focus_areas else ""}

Return ONLY valid JSON array with this exact structure:
[
    {{
        "id": 1,
        "question": "Clear question text?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correctAnswer": 0,
        "explanation": "Why this is correct",
        "hint": "Helpful hint",
        "type": "technical",
        "difficulty": "{difficulty}"
    }}
]

Make questions practical for job interviews. Include variety."""

        try:
            print(f"🔍 Attempting DeepSeek quiz generation for: {topic}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 4000
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    print(f"📡 DeepSeek Response Status: {response.status}")

                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        print(f"📝 Raw DeepSeek Response: {content[:200]}...")

                        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                        if json_match:
                            try:
                                questions = json.loads(json_match.group(0))
                                print(f"✅ DeepSeek generated {len(questions)} questions")
                                return {
                                    "topic": topic,
                                    "difficulty": difficulty,
                                    "questions": questions,
                                    "totalQuestions": len(questions),
                                    "source": "deepseek"
                                }
                            except json.JSONDecodeError as e:
                                print(f"❌ DeepSeek JSON parse error: {e}")
                                return {"error": "Failed to parse DeepSeek response"}
                        else:
                            print("❌ No JSON array found in DeepSeek response")
                            return {"error": "No valid JSON in DeepSeek response"}

                    else:
                        error_text = await response.text()
                        print(f"❌ DeepSeek API error {response.status}: {error_text}")
                        return {"error": f"DeepSeek API error: {response.status}"}

        except Exception as e:
            print(f"❌ DeepSeek error: {str(e)}")
            return {"error": f"DeepSeek error: {str(e)}"}

    async def _generate_openrouter_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """Generate quiz using OpenRouter API (unchanged)"""
        # Your existing code—integrate as needed
        prompt = f"""Create {question_count} interview quiz questions about {topic}, difficulty: {difficulty}.
{f'Focus areas: {focus_areas}' if focus_areas else ''}

Return JSON array with: question, options, correctAnswer, explanation, hint, type, difficulty.
Make questions practical for job interviews."""

        try:
            print(f"🔍 Attempting OpenRouter quiz generation for: {topic}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "HTTP-Referer": settings.FRONTEND_URL,
                        "X-Title": "AjiEasy Interview Prep"
                    },
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 3000
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    print(f"📡 OpenRouter Response Status: {response.status}")

                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        print(f"📝 Raw OpenRouter Response: {content[:200]}...")

                        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                        if json_match:
                            try:
                                questions = json.loads(json_match.group(0))
                                print(f"✅ OpenRouter generated {len(questions)} questions")
                                return {
                                    "topic": topic,
                                    "difficulty": difficulty,
                                    "questions": questions,
                                    "totalQuestions": len(questions),
                                    "source": "openrouter"
                                }
                            except json.JSONDecodeError as e:
                                print(f"❌ OpenRouter JSON parse error: {e}")
                                return {"error": "Failed to parse OpenRouter response"}
                        else:
                            print("❌ No JSON array found in OpenRouter response")
                            return {"error": "No valid JSON in OpenRouter response"}

                    else:
                        error_text = await response.text()
                        print(f"❌ OpenRouter API error {response.status}: {error_text}")
                        return {"error": f"OpenRouter API error: {response.status}"}

        except Exception as e:
            print(f"❌ OpenRouter error: {str(e)}")
            return {"error": f"OpenRouter error: {str(e)}"}

    async def _generate_openrouter_chat(self, messages: List[Dict]) -> str:
        """Fallback chat using OpenRouter"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "HTTP-Referer": settings.FRONTEND_URL,
                    },
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": messages,
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter Chat Error: {e}")
        return "I'm having trouble connecting right now."

    def _generate_enhanced_local_quiz(self, topic: str, difficulty: str, question_count: int, focus_areas: str) -> Dict:
        return {
            "topic": topic,
            "difficulty": difficulty,
            "questions": [
                {
                    "id": i,
                    "question": f"Sample question {i} about {topic}?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correctAnswer": 0,
                    "explanation": "This is a placeholder.",
                    "hint": "Try Option A",
                    "type": "technical",
                    "difficulty": difficulty
                } for i in range(1, min(question_count, 5) + 1)  # Scale to count
            ],
            "totalQuestions": min(question_count, 5),
            "source": "local-fallback"
        }

<<<<<<< HEAD
    async def generate_questions(
        self,
        topic: str,
        job_description: str,
        interview_type: str,
        company_nature: str
    ) -> List[Dict]:
        """Generate interview questions using Gemini with local fallback."""
        if not GEMINI_ENABLED or not gemini_model:
            print("⚠️ Gemini unavailable, using local fallback for questions")
            return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

        prompt = f"""Generate 8 professional interview questions for:
TOPIC: {topic}
JOB ROLE: {job_description}
INTERVIEW TYPE: {interview_type}
COMPANY TYPE: {company_nature}

Create a mix of:
- 3 Technical questions
- 3 Behavioral questions
- 2 Situational questions

Return ONLY a valid JSON array with this exact structure:
[
    {{
        "question": "Question text?",
        "type": "technical",
        "difficulty": "medium",
        "explanation": "Brief explanation of what this evaluates"
    }}
]"""

        try:
            print(f"🔄 Generating Questions with Gemini for: {topic}")
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                try:
                    questions = json.loads(clean_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON array found in Gemini response")

                print(f"✅ Successfully parsed {len(questions)} questions")
                return questions
        except Exception as e:
            print(f"❌ Gemini Question Gen Error: {type(e).__name__}: {e}")

        print("⚠️ Using local fallback for questions")
        return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

    async def generate_analytics(self, user_data: Dict) -> Dict:
        """Generate analytics insights using Gemini with local fallback."""
        if not GEMINI_ENABLED or not gemini_model:
            print("⚠️ Gemini unavailable for analytics—using fallback")
            return self._fallback_analytics()

        prompt = f"""Analyze this user's interview preparation performance and provide insights:
User Data: {json.dumps(user_data)}

Return ONLY a valid JSON object with this structure:
{{
    "performance_trend": [65, 70, ...],
    "topic_mastery": {{"Topic A": 80, "Topic B": 60}},
    "average_score": 75,
    "recommendations": ["Rec 1", "Rec 2", "Rec 3"],
    "strength_areas": ["Area 1", "Area 2"],
    "weakness_areas": ["Area 1", "Area 2"]
}}"""

        try:
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                try:
                    return json.loads(clean_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
        except Exception as e:
            print(f"❌ Gemini Analytics Error: {e}")

        return self._fallback_analytics()

    def _fallback_analytics(self) -> Dict:
        return {
            "performance_trend": [60, 65, 70, 72, 75, 78, 80],
            "topic_mastery": {"General": 70},
            "average_score": 70,
            "recommendations": [
                "Keep practicing to improve your score.",
                "Try different topics."
            ],
            "strength_areas": ["Consistency"],
            "weakness_areas": ["Complex topics"]
        }

    async def send_chat_message(self, topic: str, message: str, history: Optional[List[Dict]] = None) -> str:
        """
        Send chat message using Groq with OpenRouter and local fallback options.
        """
        if not settings.ENABLE_CHAT:
            return "Chat feature is currently disabled."

        history = history or []
        system_prompt = f"You are AjiEasy AI, an encouraging interview coach focused on {topic}."
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Message: {message}\n\n"
                "Provide concise, structured advice (<400 words) that includes:\n"
                "- Targeted guidance for the topic\n"
                "- Actionable next steps\n"
                "- Encouraging tone\n"
            )
        })

        if self.groq_enabled:
            try:
                return await _groq_completion(messages, temperature=0.65, max_tokens=800)
            except Exception as exc:
                print(f"Groq chat error: {exc}")

        if self.openrouter_enabled:
            router_response = await self._generate_openrouter_chat(messages)
            if router_response:
                return router_response

        return f"""Hi! Let's work on your {topic} preparation.

Key focus areas:
- Clarify the core problem in: "{message}"
- Rehearse 2-3 stories where you solved a similar challenge.
- Outline a structured answer (context → action → impact).

Next steps:
1. Write a bullet response, then read it aloud.
2. Time yourself for 90 seconds per answer.
3. Note follow-up questions you might receive.

You’ve got this — consistent practice will make the story flow naturally!"""

    async def _generate_openrouter_chat(self, messages: List[Dict]) -> Optional[str]:
        """Fallback chat using OpenRouter."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "HTTP-Referer": settings.FRONTEND_URL,
                    },
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 800
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    error_text = await response.text()
                    print(f"OpenRouter Chat Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"OpenRouter Chat Exception: {e}")
        return None

# Create a global instance
free_ai_service = FreeAIService()

async def generate_questions_async(
    topic: str,
    job_description: str,
    interview_type: str,
    company_nature: str
) -> List[Dict]:
    """Convenience wrapper for the FreeAIService instance."""
    return await free_ai_service.generate_questions(
        topic=topic,
        job_description=job_description,
        interview_type=interview_type,
        company_nature=company_nature
    )
=======
# Global instance
free_ai_service = FreeAIService()

# Wrapper functions (unchanged)
async def generate_questions_async(topic: str, job_description: str, interview_type: str, company_nature: str) -> List[Dict]:
    return await free_ai_service.generate_questions(topic, job_description, interview_type, company_nature)
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961

async def generate_free_quiz(topic: str, difficulty: str = "medium", question_count: int = 10, focus_areas: str = "") -> Dict:
    return await free_ai_service.generate_quiz(topic, difficulty, question_count, focus_areas)

async def send_chat_message(topic: str, message: str, history: List[Dict] = []) -> str:
<<<<<<< HEAD
    return await free_ai_service.send_chat_message(topic, message, history)
=======
    return await free_ai_service.send_chat_message(topic, message, history)

async def generate_local_fallback_questions(topic: str, job_description: str, interview_type: str, company_nature: str) -> List[Dict]:
    return [
        {
            "question": f"Explain the core concepts of {topic}.",
            "type": "technical",
            "difficulty": "medium",
            "explanation": "Tests fundamental knowledge."
        },
        {
            "question": f"Describe a time you used {topic} in a project.",
            "type": "behavioral",
            "difficulty": "medium",
            "explanation": "Tests experience."
        }
    ]
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
