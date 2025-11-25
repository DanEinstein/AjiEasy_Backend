from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

from config import settings

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "sslmode": "require",
    }
)
logger.info("Configured for PostgreSQL database with psycopg2")

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                      onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"User(id={self.id}, email={self.email}, is_active={self.is_active})"

# AiService model
class AiService(Base):
    __tablename__ = "ai_services"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                      onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"AiService(id={self.id}, name={self.name}, is_active={self.is_active})"

# Database dependency function
def get_db():
    db = SessionLocal()
    try:
        yield db
        logger.debug("Database session created successfully")
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed")

def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        tables = Base.metadata.tables.keys()
        logger.info(f"Available tables: {list(tables)}")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def get_database_info():
    try:
        with engine.connect() as conn:
            if "postgresql" in settings.DATABASE_URL.lower():
                result = conn.execute("SELECT version();")
                db_version = result.scalar()
                db_type = "PostgreSQL"
            else:
                db_version = "SQLite"
                db_type = "SQLite"

            return {
                "type": db_type,
                "version": db_version,
                "url": settings.DATABASE_URL.split('@')[0] + '@[hidden]' if '@' in settings.DATABASE_URL else settings.DATABASE_URL,
                "tables": list(Base.metadata.tables.keys())
            }
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}

logger.info("Database module initialized")