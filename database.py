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

class Database:
    """
    Database class encapsulates SQLAlchemy engine, session, and model definitions.
    """
    
    # Use regular PostgreSQL driver (not asyncpg)
    engine = create_engine(
        settings.DATABASE_URL,  # Keep as postgresql:// not postgresql+asyncpg://
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "sslmode": "require",  # Changed from "ssl" to "sslmode"
        }
    )
    logger.info("Configured for PostgreSQL database with psycopg2")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    # Your existing User and AiService classes remain the same
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

    # Your existing get_db, create_tables, test_connection, get_database_info methods remain the same
    @staticmethod
    def get_db():
        db = Database.SessionLocal()
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

    @staticmethod
    def create_tables():
        try:
            Database.Base.metadata.create_all(bind=Database.engine)
            logger.info("Database tables created successfully")
            tables = Database.Base.metadata.tables.keys()
            logger.info(f"Available tables: {list(tables)}")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    @staticmethod
    def test_connection():
        try:
            with Database.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    @staticmethod
    def get_database_info():
        try:
            with Database.engine.connect() as conn:
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
                    "tables": list(Database.Base.metadata.tables.keys())
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}

logger.info("Database module initialized")