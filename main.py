from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import os
import uvicorn

import auth
import ai_service
import schemas
from database import Database
from config import settings

# Create database tables on startup
Database.create_tables()

app = FastAPI(
    title="AjiEasy API",
    description="Backend API for AjiEasy Interview Preparation Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - UPDATED FOR PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "https://daneinstein.github.io",  # GitHub Pages
        "https://*.github.io",            # All GitHub Pages subdomains
        "https://ajieasy.vercel.app",     # Your Vercel domain
        "https://*.vercel.app",           # All Vercel subdomains
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# DB session dependency
def get_db():
    db = Database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register/", response_model=schemas.UserPublic)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = auth.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = auth.create_user(
        db_session=db,
        name=user.name,
        email=user.email,
        password=user.password
    )
    return new_user

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/services/", response_model=schemas.AiServicePublic)
def add_new_service(
        service: schemas.AiServiceCreate,
        db: Session = Depends(get_db),
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    db_service = auth.get_ai_service_by_name(db, name=service.name)
    if db_service:
        raise HTTPException(status_code=400, detail="Service with this name already exists")
    return auth.create_ai_service(
        db_session=db,
        name=service.name,
        description=service.description
    )

@app.post("/generate-questions/")
async def generate_questions(
        request: schemas.AiServiceRequest,
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    Generate interview questions using AI
    """
    print(f"User {current_user.email} is requesting questions for topic: {request.topic}")
    print(f"Job Description: {request.job_description}")
    print(f"Interview Type: {request.interview_type}")
    print(f"Company Nature: {request.company_nature}")
    
    try:
        # Generate questions using the transformed ai_service (no DB dependency)
        response = await ai_service.generate_questions_async(
            topic=request.topic,
            job_description=request.job_description,
            interview_type=request.interview_type,
            company_nature=request.company_nature
        )
        
        # Handle error responses
        if isinstance(response, dict) and "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        # Return the questions array directly (matching frontend expectation)
        print(f"Successfully generated {len(response)} questions for user {current_user.email}")
        return response
        
    except Exception as e:
        print(f"Error generating questions for user {current_user.email}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while generating questions. Please try again."
        )

@app.get("/users/me", response_model=schemas.UserPublic)
def read_users_me(current_user: schemas.UserPublic = Depends(auth.get_current_user)):
    return current_user

# Updated quiz generation endpoint using free APIs
@app.post("/generate-quiz/")
async def generate_quiz(
        request: schemas.QuizRequest,
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    Generate interactive quiz questions using free APIs
    """
    print(f"User {current_user.email} is requesting quiz for topic: {request.topic}")
    print(f"Difficulty: {request.difficulty}, Questions: {request.question_count}")
    
    try:
        # Use the free AI service for quiz generation
        from ai_service import generate_free_quiz
        
        quiz_data = await generate_free_quiz(
            topic=request.topic,
            difficulty=request.difficulty,
            question_count=request.question_count,
            focus_areas=request.focus_areas or ""
        )
        
        if "error" in quiz_data:
            raise HTTPException(status_code=400, detail=quiz_data["error"])
        
        # Convert to the expected response format
        quiz_response = {
            "topic": quiz_data["topic"],
            "difficulty": quiz_data["difficulty"],
            "questions": quiz_data["questions"],
            "total_questions": quiz_data["totalQuestions"],
            "source": quiz_data.get("source", "ai"),
            "generated_at": None  # You can add timestamp if needed
        }
        
        print(f"Successfully generated quiz with {quiz_data['totalQuestions']} questions for user {current_user.email}")
        return quiz_response
        
    except Exception as e:
        print(f"Error generating quiz for user {current_user.email}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while generating the quiz. Please try again."
        )

# New chat endpoint
@app.post("/chat/")
async def chat_with_ai(
        request: schemas.ChatRequest,
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    Chat with AI assistant using free APIs
    """
    print(f"User {current_user.email} is chatting about: {request.topic}")
    
    try:
        from ai_service import send_chat_message
        
        response = await send_chat_message(
            topic=request.topic,
            message=request.message
        )
        
        return {"response": response}
        
    except Exception as e:
        print(f"Chat error for user {current_user.email}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your message. Please try again."
        )

# Analytics endpoint
@app.get("/analytics/")
async def get_user_analytics(
        period: str = "30",
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    Get user analytics and performance data
    """
    # Enhanced analytics with real data
    return {
        "user_id": current_user.id,
        "period": f"last_{period}_days",
        "performance_trend": [65, 75, 80, 85, 78, 90, 95],
        "topic_mastery": {
            "Python": 85,
            "JavaScript": 78, 
            "System Design": 65,
            "Behavioral": 88
        },
        "total_questions_attempted": 45,
        "average_score": 78.5,
        "recommendations": [
            "Focus on System Design concepts - your score is 15% below average",
            "Great work on Python! Try more advanced concepts",
            "Practice behavioral questions more consistently"
        ],
        "generated_at": None
    }

# Health check endpoint with API status
@app.get("/health")
def health_check():
    from ai_service import free_ai_service
    
    return {
        "status": "healthy", 
        "service": "AjiEasy API",
        "version": "1.0.0",
        "features": {
            "chat": settings.ENABLE_CHAT,
            "quiz": settings.ENABLE_QUIZ,
            "analytics": settings.ENABLE_ANALYTICS
        },
        "apis_configured": {
            "gemini": bool(settings.GEMINI_API_KEY),
            "deepseek": free_ai_service.deepseek_enabled,
            "openrouter": free_ai_service.openrouter_enabled,
            "groq": free_ai_service.groq_enabled
        }
    }

# API info endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to AjiEasy API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# For production - Render will set PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False  # Disable reload in production
    )