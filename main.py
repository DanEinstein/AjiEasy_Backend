from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uvicorn
import logging
import time
from datetime import timedelta

=======
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
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
    QuestionRequest,
    ChatRequest, ChatResponse,
    AnalyticsResponse,
    Token, UserCreate, UserPublic as UserResponse
)
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
from config import settings
from database import get_db, engine, Base, User
from auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
    get_ai_service_by_name,
    create_ai_service
)
from ai_service import (
    generate_questions_async,
    generate_free_quiz,
    send_chat_message,
    free_ai_service
)
from schemas import (
    QuizRequest, QuizResponse,
    AiServiceRequest,
    ChatRequest, ChatResponse,
    Token, UserCreate, UserPublic as UserResponse,
    AiServiceCreate, AiServicePublic
)

# Token expiration constant
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Create database tables at startup
=======
# Create database tables
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AjiEasy API",
    description="Backend for AjiEasy AI Interview Platform",
<<<<<<< HEAD
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback

    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )


# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://ajieasy.vercel.app",
    "https://aji-easy-frontend.vercel.app",
    "https://ajieasy-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://([a-z0-9-]+\.)?vercel\.app$|https://([a-z0-9-]+\.)?github\.io$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


=======
    version="2.0.0"
)

# Global Exception Handler for debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error_details = traceback.format_exc()
    print(f"🔥 CRITICAL ERROR: {str(exc)}")
    print(f"🔥 TRACEBACK:\n{error_details}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"} # Exposing error for debugging
    )

# CORS Configuration - Allow frontend origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://ajieasy.vercel.app",
    "https://aji-easy-frontend.vercel.app",
    "https://ajieasy-frontend.onrender.com",
    "*"  # Temporary: Allow all origins for debugging
]

# Add CORS middleware with permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add middleware to explicitly set CORS headers on every response
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Add explicit OPTIONS handler for preflight requests
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

<<<<<<< HEAD

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AjiEasy API",
        "version": "2.0.0",
        "timestamp": time.time(),
        "features": {
            "chat": settings.ENABLE_CHAT,
            "quiz": settings.ENABLE_QUIZ,
            "analytics": settings.ENABLE_ANALYTICS
        }
    }


@app.post("/register/", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
=======
# Root endpoint is already defined at the bottom, removing this duplicate
# @app.get("/")
# async def root(): ...

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Authentication Endpoints
@app.post("/register/", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
<<<<<<< HEAD
        name=user.name,
        email=user.email,
=======
        email=user.email,
        name=user.name,
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

<<<<<<< HEAD

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
=======
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
<<<<<<< HEAD

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    public_user = UserResponse.model_validate(user)
    return {"access_token": access_token, "token_type": "bearer", "user": public_user}


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    return current_user


@app.post("/services/", response_model=AiServicePublic)
async def add_new_service(
    service: AiServiceCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    existing = get_ai_service_by_name(db, name=service.name)
    if existing:
        raise HTTPException(status_code=400, detail="Service with this name already exists")

    return create_ai_service(
        db_session=db,
        name=service.name,
        description=service.description
=======
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "is_active": user.is_active
        }
    }

<<<<<<< HEAD

@app.post("/generate-questions/")
async def generate_questions(
    request: AiServiceRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    logger.info("User %s requested questions for %s", current_user.email, request.topic)
=======
@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    return current_user

# Question generation endpoint using Gemini
@app.post("/generate-questions/")
async def generate_questions(
        request: QuestionRequest,
        current_user: UserResponse = Depends(get_current_user)
):
    """
    Generate interview questions using Gemini AI
    """
    print(f"User {current_user.email} is requesting questions for topic: {request.topic}")
    print(f"Job: {request.job_description}, Type: {request.interview_type}")

>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
    try:
        questions = await generate_questions_async(
            topic=request.topic,
            job_description=request.job_description,
            interview_type=request.interview_type,
            company_nature=request.company_nature
        )
<<<<<<< HEAD
        logger.info("Generated %s questions for %s", len(questions), current_user.email)
        return questions
    except Exception:
        logger.exception("Question generation failed for %s", current_user.email)
=======

        print(f"Successfully generated {len(questions)} questions for user {current_user.email}")
        return questions

    except Exception as e:
        print(f"Error generating questions for user {current_user.email}: {str(e)}")
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating questions. Please try again."
        )

<<<<<<< HEAD

@app.post("/generate-quiz/", response_model=QuizResponse)
async def generate_quiz(
    request: QuizRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    logger.info(
        "User %s requested quiz topic=%s difficulty=%s count=%s",
        current_user.email,
        request.topic,
        request.difficulty,
        request.question_count
    )

    try:
=======
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

>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        quiz_data = await generate_free_quiz(
            topic=request.topic,
            difficulty=request.difficulty,
            question_count=request.question_count,
            focus_areas=request.focus_areas or ""
        )

<<<<<<< HEAD
        if isinstance(quiz_data, dict) and quiz_data.get("error"):
            raise HTTPException(status_code=400, detail=quiz_data["error"])

        response_payload = QuizResponse(
            topic=quiz_data["topic"],
            difficulty=quiz_data["difficulty"],
            questions=quiz_data["questions"],
            total_questions=quiz_data["totalQuestions"],
            source=quiz_data.get("source", "ai"),
            generated_at=None
        )
        logger.info("Quiz generated for %s with %s questions", current_user.email, response_payload.total_questions)
        return response_payload
    except HTTPException:
        raise
    except Exception:
        logger.exception("Quiz generation failed for %s", current_user.email)
=======
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
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the quiz. Please try again."
        )


@app.post("/chat/", response_model=ChatResponse)
async def chat_with_ai(
<<<<<<< HEAD
    request: ChatRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    logger.info("User %s opened chat for %s", current_user.email, request.topic)
    try:
        response = await send_chat_message(topic=request.topic, message=request.message)
        return ChatResponse(response=response)
    except Exception:
        logger.exception("Chat generation failed for %s", current_user.email)
=======
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
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your message. Please try again."
        )


@app.get("/analytics/")
<<<<<<< HEAD
async def get_user_analytics(current_user: UserResponse = Depends(get_current_user)):
=======
async def get_user_analytics(
        current_user: UserResponse = Depends(get_current_user)
):
    """
    Get user analytics and performance data using Gemini AI
    """
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
    try:
        user_data = {
            "user_id": current_user.id,
            "email": current_user.email,
            "name": current_user.name
<<<<<<< HEAD
=======
        }

        analytics_data = await free_ai_service.generate_analytics(user_data)

        # Add user_id and period to response
        analytics_data["user_id"] = current_user.id
        analytics_data["period"] = "last_30_days"

        return analytics_data

    except Exception as e:
        print(f"Analytics error for user {current_user.email}: {str(e)}")
        # Return fallback data on error
        return {
            "user_id": current_user.id,
            "period": "last_30_days",
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
                "Focus on System Design concepts",
                "Great work on Python!",
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
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        }

        analytics_data = await free_ai_service.generate_analytics(user_data)
        analytics_data["user_id"] = current_user.id
        analytics_data.setdefault("period", "last_30_days")
        return analytics_data
    except Exception as exc:
        logger.exception("Analytics error for %s: %s", current_user.email, exc)
        return {
            "user_id": current_user.id,
            "period": "last_30_days",
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
                "Focus on System Design concepts",
                "Great work on Python!",
                "Practice behavioral questions more consistently"
            ],
            "generated_at": None
        }


@app.get("/recommendations/")
async def get_topic_recommendations(current_user: UserResponse = Depends(get_current_user)):
    try:
        recommendations = await free_ai_service.generate_topic_recommendations()
        return {"recommendations": recommendations}
    except Exception as exc:
        logger.exception("Recommendations error: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to generate recommendations at the moment.")


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to AjiEasy API",
<<<<<<< HEAD
        "version": "2.0.0",
=======
        "version": "1.0.0",
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961
        "docs": "/docs",
        "health": "/health"
    }

<<<<<<< HEAD
=======
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
>>>>>>> 05b47462710753a6b943a79ca5cd5508d4cc6961

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)