from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from .models import Base

# Default to a local postgres instance or allow override via env var
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5433/compliance_db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Scoped session for thread safety if needed
db_session = scoped_session(SessionLocal)

def init_db():
    """Creates the database tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

