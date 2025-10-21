from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # Database - Render will provide DATABASE_URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ajieasy.db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallback-secret-key-change-in-production")
    
    # AI APIs - All optional with fallbacks
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Frontend/Backend URLs - Updated for production
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://ajieasy.vercel.app")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "https://ajieasy-backend.onrender.com")
    
    # API Configuration
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30000"))
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "15"))
    MAX_QUIZ_QUESTIONS: int = int(os.getenv("MAX_QUIZ_QUESTIONS", "20"))
    
    # Feature Flags
    ENABLE_CHAT: bool = os.getenv("ENABLE_CHAT", "true").lower() == "true"
    ENABLE_QUIZ: bool = os.getenv("ENABLE_QUIZ", "true").lower() == "true"
    ENABLE_ANALYTICS: bool = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    
    # JWT configuration
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables
    )

settings = Settings()