# main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uvicorn

# Import all your logic and models
import auth
import ai_service
import schemas
from database import Database

# Create all database tables (on startup)
Database.create_tables()

# Initialize the FastAPI app
app = FastAPI(title="AjiEasy API")

# --- Database Dependency ---

def get_db():
    """
    FastAPI dependency to get a database session.
    Ensures the session is always closed after the request.
    """
    db = Database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the AjiEasy API"}


# --- Authentication Endpoints ---

@app.post("/register/", response_model=schemas.UserPublic)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    """
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
    """
    Login endpoint. Takes username (which is email) and password.
    Returns a JWT access token.
    """
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(
        data={"sub": user.email} # "sub" is the standard name for the token subject
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- Core Feature Endpoints (Protected) ---

# <-- THIS IS THE NEW ENDPOINT -->
@app.post("/services/", response_model=schemas.AiServicePublic)
def add_new_service(
    service: schemas.AiServiceCreate,
    db: Session = Depends(get_db),
    current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    A protected endpoint to manually add new AI services to the database.
    """
    db_service = auth.get_ai_service_by_name(db, name=service.name)
    if db_service:
        raise HTTPException(status_code=400, detail="Service with this name already exists")
    
    return auth.create_ai_service(
        db_session=db, 
        name=service.name, 
        description=service.description
    )
# <-- END OF NEW ENDPOINT -->

@app.post("/generate-questions/", response_model=schemas.AiServiceResponse)
def generate_questions(
    request: schemas.AiServiceRequest, 
    db: Session = Depends(get_db),
    current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    """
    Generate interview questions based on a service name.
    If the service doesn't exist, it will be created by the AI.
    Requires a valid login token.
    """
    print(f"User {current_user.email} is requesting questions for {request.service_name}")
    
    response = ai_service.generate_questions_for_service(
        db, 
        service_name=request.service_name
    )
    
    if isinstance(response, dict) and "error" in response:
        # Use a 500 error code if the AI itself fails
        if "generate" in response["error"]:
             raise HTTPException(status_code=500, detail=response["error"])
        # Use a 404 if the service can't be found/created
        raise HTTPException(status_code=404, detail=response["error"])
        
    return schemas.AiServiceResponse(questions=response)

@app.get("/users/me", response_model=schemas.UserPublic)
def read_users_me(current_user: schemas.UserPublic = Depends(auth.get_current_user)):
    """
    A test endpoint to check if you are logged in.
    """
    return current_user

# --- Run the App ---

if __name__ == "__main__":
    # This line allows you to run the app with `python main.py`
    uvicorn.run(app, host="127.0.0.1", port=8000)