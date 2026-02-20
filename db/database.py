from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ----- PostgreSQL connection -----
DATABASE_URL = "postgresql://postgres:pngalele@localhost:5432/trafficdb"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
