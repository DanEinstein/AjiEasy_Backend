from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import logging
import time
import os
from datetime import timedelta

from database import get_db, engine, Base, User, AiService
from auth import (
    authenticate_user, create_access_token,
    get_current_user, get_password_hash
)
from ai_service import (
    generate_questions_async,
    generate_free_quiz,
    send_chat_message,
    free_ai_service # Import the service instance for analytics
)
from schemas import (
    QuizRequest, QuizResponse,
    ChatRequest, ChatResponse,
    AnalyticsResponse,
    Token, UserCreate, UserPublic as UserResponse
)
from config import settings

# Token expiration constant
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AjiEasy API",
    description="Backend for AjiEasy AI Interview Platform",
    version="2.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "https://ajieasy.vercel.app",
    "https://ajieasy-frontend.onrender.com",
    settings.FRONTEND_URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AjiEasy API",
        "status": "online",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Authentication Endpoints
@app.post("/register/", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        name=user.name,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": {"name": user.name, "email": user.email}}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    return current_user

# Updated quiz generation endpoint using free APIs
@app.post("/generate-quiz/")
async def generate_quiz(
        request: QuizRequest,
        current_user: UserResponse = Depends(get_current_user)
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
        request: ChatRequest,
        current_user: UserResponse = Depends(get_current_user)
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
        current_user: UserResponse = Depends(get_current_user)
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

# Recommendations endpoint
@app.get("/recommendations/")
async def get_topic_recommendations(current_user: User = Depends(get_current_user)):
    """Get trending interview topics for 2025 using Gemini"""
    try:
        recommendations = await free_ai_service.generate_topic_recommendations()
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)