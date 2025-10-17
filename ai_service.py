# This is the file that will be handling the logic that interacts with the AI services in the database.
# This will be generating the AI interview questions
from sqlalchemy.orm import Session
import google.generativeai as genai

# Import your auth functions and the new settings object
# ADDED 'create_ai_service' to the import
from auth import get_ai_service_by_name, create_ai_service
from config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)


def _generate_description_for_topic(topic_name: str):
    """
    NEW Private Function: Uses AI to create a short description for a new topic.
    """
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
        # Clean up the response text
        clean_description = response.text.replace("Description: ", "").strip().strip('"')
        return clean_description
    except Exception as e:
        print(f"AI description generation failed: {e}")
        return None


def generate_questions_for_service(db: Session, service_name: str):
    """
    MODIFIED Function: Now handles topics that don't exist in the DB.
    """
    service = get_ai_service_by_name(db, service_name)
    
    # --- THIS IS THE NEW LOGIC ---
    if not service or not service.is_active:
        print(f"Service '{service_name}' not found. Attempting to generate it.")
        
        # 1. Use AI to create a new description
        new_description = _generate_description_for_topic(service_name)
        
        if not new_description:
            return {"error": "Failed to generate new service. Please try a different topic."}
        
        # 2. Save this new service to the database for next time
        print(f"Saving new service: {service_name}")
        service = create_ai_service(
            db_session=db,
            name=service_name,
            description=new_description
        )
    # --- END OF NEW LOGIC ---

    # The rest of the function works as before, using the service's description
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


# This test block is still useful. It will add "Python Basics"
# to your 'ajieasy.db' file if it's not already there.
if __name__ == "__main__":
    print("Running ai_service.py directly for testing...")
  
    from database import Database
    from auth import create_ai_service, get_ai_service_by_name
    Database.create_tables()
    db = next(Database.get_db())
    
    TEST_SERVICE_NAME = "Python Basics"
    TEST_SERVICE_DESC = "Basic data types, loops, and functions in Python for a junior role."
    
    test_service = get_ai_service_by_name(db, TEST_SERVICE_NAME)
    if not test_service:
        print(f"Creating test service: '{TEST_SERVICE_NAME}'")
        create_ai_service(
            db_session=db, 
            name=TEST_SERVICE_NAME, 
            description=TEST_SERVICE_DESC
        )
    else:
        print(f"Test service '{TEST_SERVICE_NAME}' already exists.")

    print(f"\nRequesting questions for '{TEST_SERVICE_NAME}'...")
    response = generate_questions_for_service(db, TEST_SERVICE_NAME)
    
    # 4. Print the result
    print("\n--- AI Response ---")
    print(response)
    print("-------------------\n")
    
    # 5. Clean up
    db.close()
    print("Test complete.")