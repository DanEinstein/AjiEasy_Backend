from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import the settings from your new config.py file
from config import settings

# Use the DATABASE_URL from your .env file via the settings object
DATABASE_URL = settings.DATABASE_URL


class Database:
    """
    Database class encapsulates SQLAlchemy engine, session, and model definitions.
    """
    engine = create_engine(
        DATABASE_URL,
        # This connect_args is necessary for SQLite
        connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class User(Base):
        __tablename__ = "users"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String, index=True)
        email = Column(String, unique=True, index=True)
        hashed_password = Column(String)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        def __repr__(self):
            return f"User(email={self.email}, is_active={self.is_active})"

    class AiService(Base):
        __tablename__ = "ai_services"

        id = Column(Integer, primary_key=True, index=True)
        name = Column(String, index=True)
        description = Column(String)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        def __repr__(self):
            return f"AiService(name={self.name}, is_active={self.is_active})"

    @staticmethod
    def get_db():
        """
        Generator that yields a new session and ensures it gets closed.
        """
        db = Database.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def create_tables():
        """
        Creates all tables in the database.
        """
        Database.Base.metadata.create_all(bind=Database.engine)