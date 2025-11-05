import os
import json
import re
import aiohttp
import asyncio
from typing import Dict, Optional, List
import google.generativeai as genai
from config import settings

# Initialize Gemini with robust error handling
def initialize_gemini():
    """Initialize Gemini with available models"""
    try:
        if not settings.GEMINI_API_KEY:
            print("⚠️ Gemini API key not found")
            return None, False

        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Model priority based on stability
        working_models = [
            'models/gemini-2.5-flash',        # ✅ Primary replacement
            'models/gemini-2.5-flash-lite',   # Fast & affordable
            'models/gemini-2.5-pro',          # Highest capability
            'models/gemini-2.0-flash',        # Fallback (until Nov 2025)
        ]

        gemini_model = None
        selected_model = None

        for model_name in working_models:
            try:
                print(f"🔄 Testing model: {model_name}")
                model = genai.GenerativeModel(model_name)

                # Enhanced test that handles different response formats
                test_response = model.generate_content(
                    "Respond with exactly: READY",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=10,
                    )
                )

                # Handle different response formats safely
                if hasattr(test_response, 'text') and test_response.text:
                    if 'READY' in test_response.text.upper():
                        gemini_model = model
                        selected_model = model_name
                        print(f"✅ Successfully initialized: {model_name}")
                        break
                else:
                    # Try alternative response access
                    try:
                        if test_response.candidates and len(test_response.candidates) > 0:
                            candidate = test_response.candidates[0]
                            if candidate.content and candidate.content.parts:
                                text = candidate.content.parts[0].text
                                if text and 'READY' in text.upper():
                                    gemini_model = model
                                    selected_model = model_name
                                    print(f"✅ Successfully initialized (via candidate): {model_name}")
                                    break
                    except:
                        continue

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Model {model_name} failed: {error_msg[:80]}...")
                continue

        if gemini_model:
            print(f"🎯 Using Gemini model: {selected_model}")
            return gemini_model, True
        else:
            print("❌ No Gemini models available, using local fallback only")
            return None, False

    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        return None, False

# Initialize Gemini
gemini_model, GEMINI_ENABLED = initialize_gemini()

class FreeAIService:
    def __init__(self):
        self.deepseek_enabled = bool(settings.DEEPSEEK_API_KEY)
        self.openrouter_enabled = bool(settings.OPENROUTER_API_KEY)
        self.groq_enabled = bool(getattr(settings, 'GROQ_API_KEY', None))
        print(f"✅ Free AI Services: DeepSeek={self.deepseek_enabled}, OpenRouter={self.openrouter_enabled}, Groq={self.groq_enabled}")

    async def generate_quiz(
        self,
        topic: str,
        difficulty: str = "medium",
        question_count: int = 10,
        focus_areas: str = ""
    ) -> Dict:
        """
        Generate quiz using multiple API options with better fallbacks
        """
        if not settings.ENABLE_QUIZ:
            return {"error": "Quiz feature is disabled"}

        # Validate question count
        question_count = min(question_count, settings.MAX_QUIZ_QUESTIONS)

        print(f"🎯 Starting quiz generation for: {topic}")

        # Try Groq first (FREE & FAST)
        if self.groq_enabled:
            print("🔄 Trying Groq (Free)...")
            quiz_data = await self._generate_groq_quiz(topic, difficulty, question_count, focus_areas)
            if quiz_data and "error" not in quiz_data:
                print("✅ Quiz generated with Groq")
                return quiz_data
            else:
                print(f"❌ Groq failed: {quiz_data.get('error', 'Unknown error')}")

        # Try OpenRouter second
        if self.openrouter_enabled:
            print("🔄 Trying OpenRouter...")
            quiz_data = await self._generate_openrouter_quiz(topic, difficulty, question_count, focus_areas)
            if quiz_data and "error" not in quiz_data:
                print("✅ Quiz generated with OpenRouter")
                return quiz_data
            else:
                print(f"❌ OpenRouter failed: {quiz_data.get('error', 'Unknown error')}")

        # Try DeepSeek last (might work if billing is set up)
        if self.deepseek_enabled:
            print("🔄 Trying DeepSeek...")
            quiz_data = await self._generate_deepseek_quiz(topic, difficulty, question_count, focus_areas)
            if quiz_data and "error" not in quiz_data:
                print("✅ Quiz generated with DeepSeek")
                return quiz_data
            else:
                print(f"❌ DeepSeek failed: {quiz_data.get('error', 'Unknown error')}")

        # Final fallback - Enhanced local quiz
        print("🔄 Using enhanced local quiz generator...")
        return self._generate_enhanced_local_quiz(topic, difficulty, question_count, focus_areas)

    async def _generate_groq_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """
        Generate quiz using Groq API (FREE & FAST)
        """
        prompt = f"""
        Create {question_count} multiple-choice interview questions about "{topic}" at {difficulty} difficulty level.
        {f"Focus areas: {focus_areas}" if focus_areas else ""}

        Return ONLY valid JSON array with this exact structure:
        [
            {{
                "id": 1,
                "question": "Clear and concise question text?",
                "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
                "correctAnswer": 0,
                "explanation": "Clear explanation why this is correct",
                "hint": "Helpful hint for the question",
                "type": "technical/behavioral/situational",
                "difficulty": "easy/medium/hard"
            }}
        ]

        Make questions practical, interview-focused, and relevant to the topic.
        Include a mix of question types appropriate for the topic.
        """

        try:
            print(f"🔍 Attempting Groq quiz generation for: {topic}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",  # Free and very capable
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 4000
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    print(f"📡 Groq Response Status: {response.status}")

                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        print(f"📝 Raw Groq Response: {content[:200]}...")

                        # Extract JSON from response
                        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                        if json_match:
                            try:
                                questions = json.loads(json_match.group(0))
                                print(f"✅ Groq generated {len(questions)} questions")
                                return {
                                    "topic": topic,
                                    "difficulty": difficulty,
                                    "questions": questions,
                                    "totalQuestions": len(questions),
                                    "source": "groq"
                                }
                            except json.JSONDecodeError as e:
                                print(f"❌ Groq JSON parse error: {e}")
                                return {"error": "Failed to parse Groq response"}
                        else:
                            print("❌ No JSON array found in Groq response")
                            return {"error": "No valid JSON in Groq response"}

                    elif response.status == 401:
                        print("❌ Groq: Invalid API key")
                        return {"error": "Groq API key invalid"}
                    elif response.status == 429:
                        print("❌ Groq: Rate limit exceeded")
                        return {"error": "Groq rate limit exceeded"}
                    else:
                        error_text = await response.text()
                        print(f"❌ Groq API error {response.status}: {error_text}")
                        return {"error": f"Groq API error: {response.status}"}

        except asyncio.TimeoutError:
            print("❌ Groq: Request timeout")
            return {"error": "Groq API timeout"}
        except Exception as e:
            print(f"❌ Groq unexpected error: {e}")
            return {"error": f"Groq error: {str(e)}"}

    async def _generate_deepseek_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """
        Generate quiz using DeepSeek API with enhanced error handling
        """
        prompt = f"""
        Create {question_count} multiple-choice interview questions about {topic} at {difficulty} difficulty.
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
                "type": "technical/behavioral/situational",
                "difficulty": "{difficulty}"
            }}
        ]

        Make questions practical for job interviews. Include variety.
        """

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

                        # Extract JSON from response
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

                    elif response.status == 401:
                        print("❌ DeepSeek: Invalid API key")
                        return {"error": "DeepSeek API key invalid"}
                    elif response.status == 402:
                        print("❌ DeepSeek: Payment required - billing setup needed")
                        return {"error": "DeepSeek billing required"}
                    elif response.status == 429:
                        print("❌ DeepSeek: Rate limit exceeded")
                        return {"error": "DeepSeek rate limit exceeded"}
                    else:
                        error_text = await response.text()
                        print(f"❌ DeepSeek API error {response.status}: {error_text}")
                        return {"error": f"DeepSeek API error: {response.status}"}

        except asyncio.TimeoutError:
            print("❌ DeepSeek: Request timeout")
            return {"error": "DeepSeek API timeout"}
        except Exception as e:
            print(f"❌ DeepSeek unexpected error: {e}")
            return {"error": f"DeepSeek error: {str(e)}"}

    async def _generate_openrouter_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """
        Generate quiz using OpenRouter API with enhanced error handling
        """
        prompt = f"""
        Create {question_count} interview quiz questions about {topic}, difficulty: {difficulty}.
        {f'Focus areas: {focus_areas}' if focus_areas else ''}

        Return JSON array with: question, options, correctAnswer, explanation, hint, type, difficulty.
        Make questions practical for job interviews.
        """

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

                    elif response.status == 401:
                        print("❌ OpenRouter: Invalid API key")
                        return {"error": "OpenRouter API key invalid"}
                    elif response.status == 429:
                        print("❌ OpenRouter: Rate limit exceeded")
                        return {"error": "OpenRouter rate limit exceeded"}
                    else:
                        error_text = await response.text()
                        print(f"❌ OpenRouter API error {response.status}: {error_text}")
                        return {"error": f"OpenRouter API error: {response.status}"}

        except asyncio.TimeoutError:
            print("❌ OpenRouter: Request timeout")
            return {"error": "OpenRouter API timeout"}
        except Exception as e:
            print(f"❌ OpenRouter unexpected error: {e}")
            return {"error": f"OpenRouter error: {str(e)}"}

    def _generate_enhanced_local_quiz(
        self,
        topic: str,
        difficulty: str,
        question_count: int,
        focus_areas: str
    ) -> Dict:
        """
        Enhanced local quiz generator with better questions
        """
        questions = []

        # Difficulty-based question templates
        difficulty_modifiers = {
            "easy": ["basic", "fundamental", "essential", "core"],
            "medium": ["advanced", "complex", "practical", "real-world"],
            "hard": ["expert", "optimization", "scalability", "architecture"]
        }

        mod = difficulty_modifiers.get(difficulty, difficulty_modifiers["medium"])

        question_templates = {
            "technical": [
                f"What is a {mod[0]} concept in {topic} that every professional should know?",
                f"How would you explain {topic} {mod[1]} principles to a junior developer?",
                f"What are the key {mod[2]} considerations when working with {topic}?",
                f"Describe a {mod[3]} {topic} problem you encountered and how you solved it.",
                f"What {topic} {mod[0]} skills are most valuable in production environments?"
            ],
            "behavioral": [
                f"Tell me about a time you used {topic} to solve a challenging problem.",
                f"How do you approach learning new {topic} technologies or frameworks?",
                f"Describe a situation where you had to explain {topic} concepts to non-technical stakeholders.",
                f"What's your process for reviewing and improving {topic} code?",
                f"How do you handle disagreements about {topic} implementation approaches?"
            ],
            "situational": [
                f"If you inherited a poorly documented {topic} system, what would be your first steps?",
                f"How would you handle a critical bug in a {topic} application during peak usage?",
                f"What would you do if business requirements conflicted with {topic} best practices?",
                f"How do you prioritize {topic} technical debt versus new feature development?",
                f"Describe your approach to mentoring someone new to {topic} development."
            ]
        }

        for i in range(1, question_count + 1):
            # Smart distribution of question types
            if i % 3 == 1:
                q_type = "technical"
            elif i % 3 == 2:
                q_type = "behavioral"
            else:
                q_type = "situational"

            templates = question_templates[q_type]
            question_text = templates[i % len(templates)]

            # Context-aware options based on difficulty
            if difficulty == "easy":
                options = [
                    f"Apply fundamental {topic} principles correctly",
                    f"Use a quick workaround that might cause issues later",
                    f"Overcomplicate the solution unnecessarily",
                    f"Avoid the problem entirely"
                ]
            elif difficulty == "hard":
                options = [
                    f"Implement a scalable, maintainable {topic} architecture",
                    f"Choose a quick solution that meets immediate needs only",
                    f"Apply overly complex patterns that reduce readability",
                    f"Delegate the problem to another team member"
                ]
            else:  # medium
                options = [
                    f"Balance {topic} best practices with practical constraints",
                    f"Focus only on immediate functionality",
                    f"Apply academic approaches without considering real-world constraints",
                    f"Wait for more requirements before proceeding"
                ]

            explanations = {
                "technical": f"This assesses your understanding of {topic} {q_type} concepts at {difficulty} level.",
                "behavioral": f"Evaluates your experience and approach to {topic}-related challenges.",
                "situational": f"Tests your problem-solving skills in {topic} scenarios."
            }

            questions.append({
                "id": i,
                "question": question_text,
                "options": options,
                "correctAnswer": 0,  # First option is always best practice
                "explanation": explanations[q_type],
                "hint": f"Consider both {topic} fundamentals and practical application.",
                "type": q_type,
                "difficulty": difficulty
            })

        print(f"✅ Generated {len(questions)} enhanced local quiz questions")
        return {
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
            "totalQuestions": len(questions),
            "source": "local-enhanced"
        }

    async def send_chat_message(self, topic: str, message: str) -> str:
        """
        Send chat message using free APIs with better local fallback
        """
        if not settings.ENABLE_CHAT:
            return "Chat feature is currently disabled."

        prompt = f"""
        You are AjiEasy AI, a professional interview coach and career advisor.

        Context: User wants to discuss: {topic}
        User's message: {message}

        Provide helpful, professional advice about interview preparation, career development, or technical topics.
        Be concise but comprehensive, and focus on practical, actionable advice.
        Keep your response under 500 words.
        """

        # Try DeepSeek first
        if self.deepseek_enabled:
            try:
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
                            "max_tokens": 1000
                        },
                        timeout=aiohttp.ClientTimeout(total=settings.API_TIMEOUT/1000)
                    ) as response:

                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"DeepSeek chat error: {e}")

        # Try OpenRouter as fallback
        if self.openrouter_enabled:
            try:
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
                            "max_tokens": 1000
                        },
                        timeout=aiohttp.ClientTimeout(total=settings.API_TIMEOUT/1000)
                    ) as response:

                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"OpenRouter chat error: {e}")

        # ========== IMPROVED LOCAL FALLBACK ==========
        # More conversational and personalized responses
        if "interview" in message.lower() or "prepare" in message.lower() or "preparation" in message.lower():
            return f"""Hello! I'm excited to help you with your interview preparation for **{topic}**! 🎯

Based on your question about "{message}", here's my advice as your AI interview coach:

### 🚀 **Quick Preparation Framework:**

**1. Research & Understand:**
- Study the company's products, culture, and recent news
- Understand the specific {topic} role requirements
- Identify how your skills match their needs

**2. Practice Key Areas:**
- **Technical {topic} concepts** relevant to the role
- **Behavioral questions** using the STAR method
- **System design** or practical problem-solving

**3. Mock Interviews:**
- Practice explaining {topic} concepts clearly
- Record yourself answering common questions
- Get feedback from peers or mentors

### 💡 **Pro Tips for {topic}:**
- Prepare 2-3 strong projects to discuss
- Be ready to explain complex concepts simply
- Show enthusiasm for continuous learning in {topic}

### 🎯 **Your Action Plan:**
1. Review core {topic} fundamentals this week
2. Practice 3-5 behavioral stories
3. Do at least 2 mock interviews

Remember: Confidence comes from preparation. You've got this! 💪

Would you like me to help you practice specific {topic} questions or review your preparation strategy?"""

        elif "career" in message.lower() or "job" in message.lower() or "role" in message.lower():
            return f"""Hello! Let's talk about building your career in **{topic}**! 🌟

Based on your interest in "{message}", here are some strategic steps:

### 📈 **Career Development Path for {topic}:**

**1. Skill Building:**
- Master core {topic} fundamentals
- Learn relevant tools and technologies
- Build practical projects to demonstrate expertise

**2. Networking & Visibility:**
- Connect with {topic} professionals
- Join relevant communities and forums
- Attend industry events and conferences

**3. Portfolio Development:**
- Create a portfolio with {topic} projects
- Document your {topic} learnings
- Contribute to relevant projects

### 🎯 **Immediate Actions:**
- Update your resume with specific {topic} achievements
- Set up informational interviews with {topic} professionals
- Identify companies doing interesting work in {topic}

### 💼 **Interview Readiness:**
- Prepare your "{topic} story" - why this field interests you
- Research companies that value {topic} expertise
- Practice explaining your {topic} journey clearly

The {topic} field is constantly evolving - your curiosity and adaptability will be your greatest assets!

What specific aspect of your {topic} career journey would you like to explore further?"""

        else:
            return f"""Hello! I'm your AI Interview Coach, and I'd love to help you with **{topic}**! 🤖✨

I understand you're asking about: "{message}"

### Here's how I can assist you with {topic}:

**🎯 Interview Preparation:**
- Practice {topic}-specific technical questions
- Behavioral questions tailored to {topic} roles
- Mock interviews focusing on {topic} concepts

**💡 Career Guidance:**
- {topic} career path options and growth opportunities
- Skill development roadmap for {topic}
- Industry trends and in-demand {topic} skills

**🚀 Practical Strategies:**
- How to showcase your {topic} expertise
- Common {topic} interview patterns and how to prepare
- Real-world {topic} problem-solving approaches

### My Approach:
I believe in practical, actionable advice that you can immediately apply. Whether you're preparing for a {topic} interview, planning your career path, or just exploring this field - I'm here to provide personalized guidance.

What specific aspect of {topic} would you like to dive into? For example:
- "Can we practice some {topic} technical questions?"
- "What's the career outlook for {topic} professionals?"
- "How should I prepare for a {topic} system design interview?"

I'm excited to help you succeed! 🎉"""

# Create a global instance
free_ai_service = FreeAIService()

# ==================== OPTIMIZED GEMINI 2.5 FLASH QUESTION GENERATION ====================
async def generate_questions_async(
    topic: str,
    job_description: str,
    interview_type: str,
    company_nature: str
) -> List[Dict]:
    """
    Generate interview questions using Gemini 2.5 Flash
    """
    try:
        if not GEMINI_ENABLED:
            print("⚠️ Gemini not available, using local fallback")
            return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

        # Optimized prompt for Gemini 2.5 Flash
        prompt = f"""
        Generate 8 professional interview questions for:
        TOPIC: {topic}
        JOB ROLE: {job_description}
        INTERVIEW TYPE: {interview_type}
        COMPANY TYPE: {company_nature}

        Create a mix of:
        - 3 Technical questions (specific {topic} knowledge)
        - 3 Behavioral questions (experience and soft skills)
        - 2 Situational questions (problem-solving scenarios)

        For each question, provide:
        - question: The interview question text
        - type: technical/behavioral/situational
        - difficulty: easy/medium/hard
        - explanation: Brief description of what this assesses

        Return ONLY a valid JSON array with this exact structure:
        [
            {{
                "question": "Question text?",
                "type": "technical",
                "difficulty": "medium",
                "explanation": "What this question evaluates..."
            }}
        ]

        Make questions practical, job-relevant, and tailored to {company_nature} companies.
        """

        print(f"🔮 Generating questions with Gemini 2.5 Flash for: {topic}")

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    max_output_tokens=3000,
                )
            )

            # Enhanced response handling
            response_text = ""
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            else:
                # Alternative response access for different formats
                try:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            response_text = candidate.content.parts[0].text
                except:
                    pass

            if not response_text:
                raise Exception("Empty response from Gemini")

            # Extract JSON from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                print(f"✅ Generated {len(questions)} questions with Gemini 2.5 Flash")
                return questions
            else:
                print("❌ No JSON found in response, using fallback")
                return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

        except Exception as e:
            print(f"❌ Gemini generation error: {e}")
            return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

    except Exception as e:
        print(f"❌ Overall Gemini error: {e}")
        return await generate_local_fallback_questions(topic, job_description, interview_type, company_nature)

async def generate_local_fallback_questions(topic: str, job_description: str, interview_type: str, company_nature: str) -> List[Dict]:
    """Enhanced local fallback question generator"""
    question_templates = {
        "technical": [
            f"Explain the main concepts of {topic} to someone non-technical.",
            f"What are the key advantages of using {topic} in production?",
            f"Describe a challenging {topic} problem you solved and what you learned.",
            f"How would you optimize a slow {topic} application?",
            f"What testing strategies are important for {topic} projects?"
        ],
        "behavioral": [
            f"Tell me about a time you had to learn {topic} quickly for a project.",
            f"Describe how you would mentor someone new to {topic}.",
            f"How do you handle disagreements about {topic} implementation?",
            f"Share an example of a successful {topic} project you led.",
            f"What's your process for staying updated with {topic} trends?"
        ],
        "situational": [
            f"How would you approach a legacy {topic} system with no documentation?",
            f"What would you do if you found a security issue in your {topic} code?",
            f"How do you balance {topic} best practices with tight deadlines?",
            f"Describe your approach to code review for {topic} projects.",
            f"How would you convince stakeholders to adopt a new {topic} approach?"
        ]
    }

    questions = []
    for i in range(8):
        if i < 3:
            q_type = "technical"
        elif i < 6:
            q_type = "behavioral"
        else:
            q_type = "situational"

        templates = question_templates[q_type]
        question_text = templates[i % len(templates)]

        questions.append({
            "question": question_text,
            "type": q_type,
            "difficulty": "medium",
            "explanation": f"This {q_type} question assesses your {topic} knowledge in {job_description.lower()} contexts."
        })

    print(f"✅ Generated {len(questions)} questions using local fallback")
    return questions

# Convenience functions for main.py
async def generate_free_quiz(topic: str, difficulty: str = "medium", question_count: int = 10, focus_areas: str = "") -> Dict:
    return await free_ai_service.generate_quiz(topic, difficulty, question_count, focus_areas)

async def send_chat_message(topic: str, message: str) -> str:
    return await free_ai_service.send_chat_message(topic, message)