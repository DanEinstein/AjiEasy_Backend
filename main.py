from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uvicorn
import logging
import time
from datetime import timedelta

from config import settings
from database import get_db, engine, Base, User, AiService
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

# Create database tables at startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AjiEasy API",
    description="Backend for AjiEasy AI Interview Platform",
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
    settings.FRONTEND_URL,
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


@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}


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
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        name=user.name,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
    )


@app.post("/generate-questions/")
async def generate_questions(
    request: AiServiceRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    logger.info("User %s requested questions for %s", current_user.email, request.topic)
    try:
        questions = await generate_questions_async(
            topic=request.topic,
            job_description=request.job_description,
            interview_type=request.interview_type,
            company_nature=request.company_nature
        )
        logger.info("Generated %s questions for %s", len(questions), current_user.email)
        return questions
    except Exception:
        logger.exception("Question generation failed for %s", current_user.email)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating questions. Please try again."
        )


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
        quiz_data = await generate_free_quiz(
            topic=request.topic,
            difficulty=request.difficulty,
            question_count=request.question_count,
            focus_areas=request.focus_areas or ""
        )

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
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the quiz. Please try again."
        )


@app.post("/chat/", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    logger.info("User %s opened chat for %s", current_user.email, request.topic)
    try:
        response = await send_chat_message(topic=request.topic, message=request.message, history=request.history)
        return ChatResponse(response=response)
    except Exception:
        logger.exception("Chat generation failed for %s", current_user.email)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your message. Please try again."
        )


@app.get("/analytics/")
async def get_user_analytics(current_user: UserResponse = Depends(get_current_user)):
    try:
        user_data = {
            "user_id": current_user.id,
            "email": current_user.email,
            "name": current_user.name
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
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)