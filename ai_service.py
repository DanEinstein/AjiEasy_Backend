import asyncio
from sqlalchemy.orm import Session
import google.generativeai as genai

from auth import get_ai_service_by_name, create_ai_service
from config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

def _generate_description_for_topic(topic_name: str):
    print(f"Generating new description for topic: {topic_name}")
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""
        You are a curriculum designer. A user wants to practice interview questions
        for the topic: "{topic_name}".
        
        Your task is to write a single, concise, one-sentence description
        of what this topic covers in the context of a technical interview.
        
        Example:
        Topic: "React Hooks"
        Description: "Covers useState, useEffect, useContext, and custom hooks for managing state and side effects in React."
        
        Now, generate the description for: "{topic_name}"
        """
        response = model.generate_content(prompt)
        clean_description = response.text.replace("Description: ", "").strip().strip('"')
        return clean_description
    except Exception as e:
        print(f"AI description generation failed: {e}")
        return None

def generate_questions_for_service(db: Session, service_name: str):
    service = get_ai_service_by_name(db, service_name)
    if not service or not service.is_active:
        print(f"Service '{service_name}' not found. Attempting to generate it.")
        new_description = _generate_description_for_topic(service_name)
        if not new_description:
            return {"error": "Failed to generate new service. Please try a different topic."}
        print(f"Saving new service: {service_name}")
        service = create_ai_service(
            db_session=db,
            name=service_name,
            description=new_description
        )
    prompt = f"""
    You are an expert career coach and interviewer for "AjiEasy".
    Your task is to generate 5 high-quality, distinct interview questions
    based on the following topic: {service.description}.
    The questions should be challenging but fair.
    Format the output as a clean, numbered list.
    """
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI question generation failed: {e}")
        return {"error": "An error occurred while generating questions."}

async def generate_questions_for_service_async(db: Session, service_name: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_questions_for_service, db, service_name)
