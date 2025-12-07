from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import os
import yaml
from pathlib import Path
from .models import Base

# Load database configuration from config file
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

def get_database_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            return config.get("database", {})
    return {}

# Load config
db_config = get_database_config()

# Construct URL from components
db_type = db_config.get("database_type", "postgresql")
db_user = db_config.get("db_user", "user")
db_password = db_config.get("db_password", "password")
db_host = db_config.get("db_host", "localhost")
db_port = db_config.get("db_port", 5433)
db_name = db_config.get("db_name", "compliance_db")

# Handle password (if empty, might need special handling depending on driver, but usually empty string is fine)
# Format: dialect+driver://username:password@host:port/database
if db_password:
    DATABASE_URL = f"{db_type}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
else:
    DATABASE_URL = f"{db_type}://{db_user}@{db_host}:{db_port}/{db_name}"

# Fallback to env var if needed (though constructing from defaults above handles most cases)
if not db_config and os.getenv("DATABASE_URL"):
    DATABASE_URL = os.getenv("DATABASE_URL")

#

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
