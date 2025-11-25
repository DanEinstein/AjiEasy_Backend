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


def _build_engine():
    """
    Create a SQLAlchemy engine with environment-aware connection arguments.
    Ensures local SQLite runs without SSL while production Postgres enforces it.
    """
    database_url = settings.DATABASE_URL

    connect_args = {}
    engine_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    if database_url.startswith("sqlite"):
        # SQLite cannot accept sslmode; also allow multi-thread access for FastAPI
        connect_args["check_same_thread"] = False
        engine_kwargs.pop("pool_recycle")  # Not applicable to SQLite
    elif database_url.startswith("postgresql"):
        connect_args["sslmode"] = "require"

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    engine = create_engine(database_url, **engine_kwargs)
    logger.info("Configured SQLAlchemy engine for %s", "SQLite" if database_url.startswith("sqlite") else "PostgreSQL")
    return engine


class Database:
    """
    Database class encapsulates SQLAlchemy engine, session, and model definitions.
    """

    engine = _build_engine()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

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
                conn.execute("SELECT 1")
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


# Backwards-compatible aliases for modules that import the previous API
engine = Database.engine
SessionLocal = Database.SessionLocal
Base = Database.Base
User = Database.User
AiService = Database.AiService


def get_db():
    """Compatibility wrapper around Database.get_db."""
    yield from Database.get_db()


def create_tables():
    Database.create_tables()


def test_connection():
    return Database.test_connection()


def get_database_info():
    return Database.get_database_info()

logger.info("Database module initialized")