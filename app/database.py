"""
Database models and access layer.

Using SQLite initially via SQLAlchemy.

STEP 5:
- Define submissions table
- Provide helpers to store / load vectors (JSON)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence
import json

import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'plagiarism.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String, index=True)  # "baseline" | "student"
    student_id = Column(String, nullable=True, index=True)
    language = Column(String, index=True)
    # Raw source code (optional but recommended for baselines so models can be rebuilt).
    code = Column(Text, nullable=True)
    # JSON-encoded vectors (list[float]); we keep them as TEXT in SQLite.
    token_vector = Column(Text, nullable=True)
    ast_vector = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


# -----------------------------
# Session helper
# -----------------------------

def get_db() -> Session:
    """
    Get a new SQLAlchemy Session.

    In FastAPI, you'll normally use this as a dependency with `yield`, but this
    simple helper is enough for scripts and services.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Vector serialization helpers
# -----------------------------

def vector_to_json(vec: Optional[np.ndarray]) -> Optional[str]:
    """
    Serialize a 1D numpy vector to JSON string.
    """
    if vec is None:
        return None
    # Ensure 1D array
    arr = np.asarray(vec).ravel().tolist()
    return json.dumps(arr)


def json_to_vector(data: Optional[str]) -> Optional[np.ndarray]:
    """
    Deserialize a JSON string back into a numpy vector.
    """
    if data is None:
        return None
    try:
        arr = json.loads(data)
    except json.JSONDecodeError:
        return None
    return np.asarray(arr, dtype=float)


# -----------------------------
# CRUD helpers
# -----------------------------

def create_submission(
    db: Session,
    *,
    source_type: str,
    language: str,
    token_vector: Optional[np.ndarray],
    ast_vector: Optional[np.ndarray],
    student_id: Optional[str] = None,
    code: Optional[str] = None,
) -> Submission:
    """
    Create and persist a Submission row.
    """
    db_obj = Submission(
        source_type=source_type,
        student_id=student_id,
        language=language,
        code=code,
        token_vector=vector_to_json(token_vector),
        ast_vector=vector_to_json(ast_vector),
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_all_submissions(db: Session) -> Sequence[Submission]:
    """
    Return all submissions (baseline + student).
    """
    return db.query(Submission).order_by(Submission.id).all()


def get_submissions_by_type(db: Session, source_type: str) -> Sequence[Submission]:
    """
    Return submissions filtered by source_type.
    """
    return (
        db.query(Submission)
        .filter(Submission.source_type == source_type)
        .order_by(Submission.id)
        .all()
    )

