# schemas.py
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Union, Optional, List, Dict, Any
from datetime import datetime

# --- User Schemas ---

class UserCreate(BaseModel):
    """Data required to create a new user."""
    name: str = Field(..., min_length=2, max_length=50, description="Full name of the user")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, max_length=100, description="User's password")

class UserPublic(BaseModel):
    """Data you send back to the client (NEVER send the password)."""
    id: int
    name: str
    email: EmailStr
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# --- AI Service Schemas ---

class Question(BaseModel):
    """Individual question schema"""
    question: str = Field(..., description="The interview question text")
    type: str = Field(..., description="Type of question: technical, behavioral, situational, scenario, best_practice")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")
    explanation: Optional[str] = Field(None, description="Explanation for the question")

class AiServiceRequest(BaseModel):
    """Data the client sends to request questions."""
    topic: str = Field(..., min_length=2, max_length=100, description="Main topic for interview questions")
    job_description: str = Field("", max_length=500, description="Description of the job role")
    interview_type: str = Field("", max_length=50, description="Type of interview: Technical, HR, Managerial, etc.")
    company_nature: str = Field("", max_length=50, description="Nature of company: Startup, Corporation, etc.")

class AiServiceResponse(BaseModel):
    """Data you send back with the AI's answer."""
    questions: List[Question]
    summary: Optional[Dict[str, Dict[str, int]]] = None
    generated_at: Optional[str] = None

class AiServiceCreate(BaseModel):
    """Data required to manually create a new service."""
    name: str = Field(..., min_length=2, max_length=100, description="Name of the AI service")
    description: str = Field(..., min_length=10, max_length=500, description="Description of the service")

class AiServicePublic(BaseModel):
    """Data you send back when a service is created or retrieved."""
    id: int
    name: str
    description: str
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# --- Quiz Schemas ---

class QuizQuestion(BaseModel):
    """Quiz question with multiple choice options"""
    id: int
    question: str
    options: List[str]  # List of option texts
    correctAnswer: int = Field(..., ge=0, le=3, description="Index of correct answer (0-3)")
    explanation: str
    hint: str
    type: Optional[str] = None
    difficulty: Optional[str] = None

class QuizRequest(BaseModel):
    """Data for generating a quiz"""
    topic: str = Field(..., min_length=2, max_length=100, description="Topic for the quiz")
    difficulty: str = Field("medium", description="Difficulty level: easy, medium, hard")
    question_count: int = Field(10, ge=5, le=20, description="Number of questions (5-20)")
    focus_areas: Optional[str] = Field(None, max_length=200, description="Specific areas to focus on")

class QuizResponse(BaseModel):
    """Response containing the generated quiz"""
    topic: str
    difficulty: str
    questions: List[QuizQuestion]
    total_questions: int
    generated_at: Optional[str] = None

# --- Question Generation Schemas ---

class QuestionRequest(BaseModel):
    """Request for generating interview questions"""
    topic: str = Field(..., min_length=2, max_length=100, description="Main topic for interview questions")
    job_description: str = Field(..., min_length=2, max_length=500, description="Description of the job role")
    interview_type: str = Field(..., max_length=50, description="Type of interview: Technical, Behavioral, HR, etc.")
    company_nature: str = Field(..., max_length=50, description="Nature of company: Startup, Corporation, Remote, etc.")

# --- Analytics Schemas ---

class AnalyticsRequest(BaseModel):
    """Request for analytics data"""
    period: str = Field("30", description="Time period: 7, 30, 90, 365 days")

class AnalyticsResponse(BaseModel):
    """Analytics data response"""
    user_id: int
    period: str
    performance_trend: List[float] = Field(..., description="Performance scores over time")
    topic_mastery: Dict[str, float] = Field(..., description="Scores by topic")
    total_questions_attempted: int
    average_score: float
    recommendations: List[str]
    generated_at: Optional[str] = None

# --- History & Favorites Schemas ---

class HistoryItem(BaseModel):
    """History item schema"""
    id: int
    topic: str
    job_description: str
    interview_type: str
    company_nature: str
    questions_generated: int
    created_at: datetime

class FavoriteItem(BaseModel):
    """Favorite questions schema"""
    id: int
    topic: str
    job_description: str
    questions: List[Question]
    created_at: datetime

class HistoryResponse(BaseModel):
    """Response containing user history"""
    items: List[HistoryItem]
    total_count: int

class FavoritesResponse(BaseModel):
    """Response containing user favorites"""
    items: List[FavoriteItem]
    total_count: int

# --- Token Schemas (for Login) ---

class Token(BaseModel):
    """The access token you send after a successful login."""
    access_token: str
    token_type: str
    user: Optional[UserPublic] = None  # Include user data for immediate frontend use

class TokenData(BaseModel):
    """The data you'll store inside the JWT token."""
    email: Optional[str] = None
    user_id: Optional[int] = None

# --- Chat Schemas ---

class ChatRequest(BaseModel):
    """Request for chat with AI"""
    topic: str = Field(..., min_length=2, max_length=100, description="Topic for discussion")
    message: str = Field(..., min_length=1, max_length=1000, description="Message to send to AI")

class ChatResponse(BaseModel):
    """Response from chat with AI"""
    response: str

# --- Export Schemas ---

class ExportRequest(BaseModel):
    """Request for exporting data"""
    format: str = Field("pdf", description="Export format: pdf, json, csv")
    content_type: str = Field("questions", description="Content type: questions, quiz, analytics")
    data_ids: Optional[List[int]] = None

class ExportResponse(BaseModel):
    """Export response with download URL"""
    download_url: str
    file_name: str
    file_size: Optional[int] = None

# --- Error Schema ---

class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str
    details: Optional[str] = None
    code: Optional[str] = None  # Error code for frontend handling

# --- Health Check Schema ---

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: Optional[str] = None
    timestamp: Optional[str] = None
    features: Optional[Dict[str, bool]] = None
    apis_configured: Optional[Dict[str, bool]] = None

# --- Generic Response Schemas ---

class SuccessResponse(BaseModel):
    """Generic success response"""
    message: str
    data: Optional[Dict[str, Any]] = None

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int

# --- API Info Schema ---

class APIInfoResponse(BaseModel):
    """API information response"""
    name: str
    version: str
    description: str
    features: List[str]
    documentation: str