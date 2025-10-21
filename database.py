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

# Import the settings from your new config.py file
from config import settings

# Setup logging
logger = logging.getLogger(__name__)

class Database:
    """
    Database class encapsulates SQLAlchemy engine, session, and model definitions.
    """
    
    # Determine database type and configure engine accordingly
    if "postgresql" in settings.DATABASE_URL.lower():
        # PostgreSQL configuration for Render
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,  # Verify connection before using
            pool_recycle=300,    # Recycle connections after 5 minutes
            connect_args={
                "sslmode": "require",  # Require SSL for security
                "connect_timeout": 10  # Connection timeout
            }
        )
        logger.info("Configured for PostgreSQL database")
    else:
        # SQLite configuration for local development
        engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False}
        )
        logger.info("Configured for SQLite database")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), index=True, nullable=False)  # Added length limit for PostgreSQL
        email = Column(String(255), unique=True, index=True, nullable=False)  # Added length limit
        hashed_password = Column(String(255), nullable=False)  # Added length limit
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Fixed timezone
        updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                          onupdate=lambda: datetime.now(timezone.utc))  # Fixed timezone

        def __repr__(self):
            return f"User(id={self.id}, email={self.email}, is_active={self.is_active})"

    class AiService(Base):
        __tablename__ = "ai_services"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String(100), index=True, nullable=False)  # Added length limit
        description = Column(Text)  # Changed to Text for longer descriptions
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Fixed timezone
        updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                          onupdate=lambda: datetime.now(timezone.utc))  # Fixed timezone

        def __repr__(self):
            return f"AiService(id={self.id}, name={self.name}, is_active={self.is_active})"

    @staticmethod
    def get_db():
        """
        Generator that yields a new session and ensures it gets closed.
        Used as a FastAPI dependency.
        """
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
        """
        Creates all tables in the database.
        Handles both development and production environments.
        """
        try:
            Database.Base.metadata.create_all(bind=Database.engine)
            logger.info("Database tables created successfully")
            
            # Log table information
            tables = Database.Base.metadata.tables.keys()
            logger.info(f"Available tables: {list(tables)}")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    @staticmethod
    def test_connection():
        """
        Test database connection - useful for health checks
        """
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
        """
        Get database information for debugging and monitoring
        """
        try:
            with Database.engine.connect() as conn:
                # Try to get database version
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

# Initialize database on import
logger.info("Database module initialized")