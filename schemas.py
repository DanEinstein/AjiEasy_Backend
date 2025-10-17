# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Union, Optional
# (Cleaned up duplicate import)

# --- User Schemas ---

class UserCreate(BaseModel):
    """Data required to create a new user."""
    name: str
    email: EmailStr  # Pydantic validates this is a real email format
    password: str

class UserPublic(BaseModel):
    """Data you send back to the client (NEVER send the password)."""
    id: int
    name: str
    email: EmailStr
    is_active: bool

    class Config:
        # This tells Pydantic to read data from SQLAlchemy models
        from_attributes = True

# --- AI Service Schemas ---

class AiServiceRequest(BaseModel):
    """Data the client sends to request questions."""
    service_name: str

class AiServiceResponse(BaseModel):
    """Data you send back with the AI's answer."""
    questions: Union[str, dict]  # Can be the string of questions or a dict error

# <-- ADD THIS SECTION -->
class AiServiceCreate(BaseModel):
    """Data required to manually create a new service."""
    name: str
    description: str

class AiServicePublic(BaseModel):
    """Data you send back when a service is created or retrieved."""
    id: int
    name: str
    description: str
    is_active: bool

    class Config:
        from_attributes = True
# <-- END OF ADDED SECTION -->


# --- Token Schemas (for Login) ---

class Token(BaseModel):
    """The access token you send after a successful login."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """The data you'll store inside the JWT token."""
    email: Optional[str] = None