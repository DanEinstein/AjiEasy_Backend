from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uvicorn

import auth
import ai_service
import schemas
from database import Database

# Create database tables on startup
Database.create_tables()

app = FastAPI(title="AjiEasy API")

# Serve frontend as static files
app.mount("/frontend", StaticFiles(directory="../frontend", html=True), name="frontend")



# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URLs in production
    allow_credentials=True,
    allow_methods=["*"],
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

@app.post("/generate-questions/", response_model=schemas.AiServiceResponse)
async def generate_questions(
        request: schemas.AiServiceRequest,
        db: Session = Depends(get_db),
        current_user: schemas.UserPublic = Depends(auth.get_current_user)
):
    print(f"User {current_user.email} is requesting questions for {request.service_name}")
    response = await ai_service.generate_questions_for_service_async(db, service_name=request.service_name)
    if isinstance(response, dict) and "error" in response:
        if "generate" in response["error"]:
            raise HTTPException(status_code=500, detail=response["error"])
        raise HTTPException(status_code=404, detail=response["error"])
    return schemas.AiServiceResponse(questions=response)

@app.get("/users/me", response_model=schemas.UserPublic)
def read_users_me(current_user: schemas.UserPublic = Depends(auth.get_current_user)):
    return current_user

# Optional root endpoint (overridden by static files mount)
@app.get("/")
def read_root():
    return {"message": "Welcome to the AjiEasy API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
