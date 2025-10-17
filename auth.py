# auth.py

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# Import your config, database, and schemas
from config import settings
from database import Database
import schemas

# --- Configuration ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Tells FastAPI where to get the token

DBUser = Database.User
DBServices = Database.AiService

# --- New JWT Functions ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=30) # Default: 30 mins
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(Database.get_db)):
    """
    Dependency to get the current user. This function protects your endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


# --- Your Existing Functions (Unchanged) ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db_session: Session, email: str, password: str):
    user = db_session.query(DBUser).filter(DBUser.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def is_active_user(user: DBUser):
    return user.is_active

def get_user_by_email(db_session: Session, email: str):
    return db_session.query(DBUser).filter(DBUser.email == email).first()

def get_ai_service_by_name(db_session: Session, name: str):
    return db_session.query(DBServices).filter(DBServices.name == name).first()

def create_user(db_session: Session, name: str, email: str, password: str):
    hashed_password = get_password_hash(password)
    db_user = DBUser(name=name, email=email, hashed_password=hashed_password)
    db_session.add(db_user)
    db_session.commit()
    db_session.refresh(db_user)
    return db_user

def create_ai_service(db_session: Session, name: str, description: str):
    db_service = DBServices(name=name, description=description)
    db_session.add(db_service)
    db_session.commit()
    db_session.refresh(db_service)
    return db_service

def deactivate_user(db_session: Session, user: DBUser):
    user.is_active = False
    db_session.commit()
    db_session.refresh(user)
    return user

def deactivate_ai_service(db_session: Session, service: DBServices):
    service.is_active = False
    db_session.commit()
    db_session.refresh(service)
    return service

def activate_user(db_session: Session, user: DBUser):
    user.is_active = True
    db_session.commit()
    db_session.refresh(user)
    return user

def activate_ai_service(db_session: Session, service: DBServices):
    service.is_active = True
    db_session.commit()
    db_session.refresh(service)
    return service