from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging

# Import your config, database, and schemas
from config import settings
from database import User, AiService, get_db
import schemas

# Setup logging
logger = logging.getLogger(__name__)

# --- Configuration ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Tells FastAPI where to get the token

DBUser = User
DBServices = AiService

# --- JWT Functions ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token with configurable expiration
    """
    to_encode = data.copy()

    # Use settings for token expiration
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),  # Issued at timestamp
        "type": "access_token"
    })

    try:
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Dependency to get the current user. This function protects your endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")

        if email is None:
            logger.warning("Token missing email subject")
            raise credentials_exception

        # Validate token type
        if payload.get("type") != "access_token":
            logger.warning("Invalid token type")
            raise credentials_exception

        token_data = schemas.TokenData(email=email)

    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected token error: {e}")
        raise credentials_exception

    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        logger.warning(f"User not found for email: {token_data.email}")
        raise credentials_exception

    if not user.is_active:
        logger.warning(f"Inactive user attempt: {token_data.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    logger.info(f"User authenticated: {user.email}")
    return user

# --- Password and User Management Functions ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password"""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process password"
        )

def authenticate_user(db_session: Session, email: str, password: str):
    """Authenticate user with email and password"""
    try:
        user = db_session.query(DBUser).filter(DBUser.email == email).first()
        if not user:
            logger.warning(f"Authentication failed: user not found - {email}")
            return False

        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: invalid password - {email}")
            return False

        if not user.is_active:
            logger.warning(f"Authentication failed: inactive user - {email}")
            return False

        logger.info(f"User authenticated successfully: {email}")
        return user

    except Exception as e:
        logger.error(f"Authentication error for {email}: {e}")
        return False

def is_active_user(user: DBUser) -> bool:
    """Check if user is active"""
    return user.is_active

def get_user_by_email(db_session: Session, email: str):
    """Get user by email"""
    try:
        return db_session.query(DBUser).filter(DBUser.email == email).first()
    except Exception as e:
        logger.error(f"Error fetching user by email {email}: {e}")
        return None

def get_ai_service_by_name(db_session: Session, name: str):
    """Get AI service by name"""
    try:
        return db_session.query(DBServices).filter(DBServices.name == name).first()
    except Exception as e:
        logger.error(f"Error fetching AI service by name {name}: {e}")
        return None

def create_user(db_session: Session, name: str, email: str, password: str):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = get_user_by_email(db_session, email)
        if existing_user:
            logger.warning(f"User creation failed: email already exists - {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        hashed_password = get_password_hash(password)
        db_user = DBUser(
            name=name,
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(db_user)
        db_session.commit()
        db_session.refresh(db_user)

        logger.info(f"User created successfully: {email}")
        return db_user

    except HTTPException:
        raise
    except Exception as e:
        db_session.rollback()
        logger.error(f"User creation error for {email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user"
        )

def create_ai_service(db_session: Session, name: str, description: str):
    """Create a new AI service"""
    try:
        db_service = DBServices(
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc)
        )
        db_session.add(db_service)
        db_session.commit()
        db_session.refresh(db_service)

        logger.info(f"AI service created: {name}")
        return db_service

    except Exception as e:
        db_session.rollback()
        logger.error(f"AI service creation error for {name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create AI service"
        )

def deactivate_user(db_session: Session, user: DBUser):
    """Deactivate a user"""
    try:
        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        db_session.commit()
        db_session.refresh(user)

        logger.info(f"User deactivated: {user.email}")
        return user

    except Exception as e:
        db_session.rollback()
        logger.error(f"User deactivation error for {user.email}: {e}")
        raise

def deactivate_ai_service(db_session: Session, service: DBServices):
    """Deactivate an AI service"""
    try:
        service.is_active = False
        service.updated_at = datetime.now(timezone.utc)
        db_session.commit()
        db_session.refresh(service)

        logger.info(f"AI service deactivated: {service.name}")
        return service

    except Exception as e:
        db_session.rollback()
        logger.error(f"AI service deactivation error for {service.name}: {e}")
        raise

def activate_user(db_session: Session, user: DBUser):
    """Activate a user"""
    try:
        user.is_active = True
        user.updated_at = datetime.now(timezone.utc)
        db_session.commit()
        db_session.refresh(user)

        logger.info(f"User activated: {user.email}")
        return user

    except Exception as e:
        db_session.rollback()
        logger.error(f"User activation error for {user.email}: {e}")
        raise

def activate_ai_service(db_session: Session, service: DBServices):
    """Activate an AI service"""
    try:
        service.is_active = True
        service.updated_at = datetime.now(timezone.utc)
        db_session.commit()
        db_session.refresh(service)

        logger.info(f"AI service activated: {service.name}")
        return service

    except Exception as e:
        db_session.rollback()
        logger.error(f"AI service activation error for {service.name}: {e}")
        raise