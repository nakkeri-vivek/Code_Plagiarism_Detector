"""
FastAPI routes for the plagiarism detection service.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import Submission, create_submission, get_db, get_submissions_by_type
from app.model_state import (
    add_baseline_example,
    compute_similarity_against_baselines,
    has_baselines,
    reset_baselines,
    get_baseline_submission_ids,
    total_baseline_count,
)
from app.similarity import top_k_indices

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


# -----------------------------
# Pydantic schemas
# -----------------------------


class BaselineSubmissionRequest(BaseModel):
    code: str = Field(..., description="Reference or teacher solution source code")
    language: str = Field(
        "python", description="Programming language of the code (currently Python-only)"
    )
    label: Optional[str] = Field(
        None,
        description="Optional label, e.g., assignment or solution identifier. Stored as student_id for baselines.",
    )


class BaselineSubmissionResponse(BaseModel):
    id: int
    language: str
    label: Optional[str]


class StudentSubmissionRequest(BaseModel):
    code: str = Field(..., description="Student submission source code")
    language: str = Field(
        "python", description="Programming language of the code (currently Python-only)"
    )
    student_id: Optional[str] = Field(
        None,
        description="Identifier for the student or submission (stored on the record)",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of most similar baselines to return",
    )


class SimilarityMatch(BaseModel):
    submission_id: int
    source_type: str
    label: Optional[str]
    language: str
    similarity: float


class PlagiarismResult(BaseModel):
    query_submission_id: int
    matches: List[SimilarityMatch]


class StatusResponse(BaseModel):
    baseline_count: int


# -----------------------------
# Routes
# -----------------------------


@router.get("/status", response_model=StatusResponse)
def status_endpoint(db: Session = Depends(get_db)) -> StatusResponse:
    """
    Lightweight status endpoint to see how many baselines are loaded.
    """
    baseline_count = total_baseline_count()
    return StatusResponse(baseline_count=baseline_count)


@router.post("/baseline/reset", response_model=StatusResponse)
def reset_baselines_endpoint(db: Session = Depends(get_db)) -> StatusResponse:
    """
    Rebuild the in-memory baseline corpus and TF-IDF models from the database.

    This does not delete baseline rows from the database; it:
    - Clears in-memory models
    - Reloads all baseline submissions (all languages) and re-trains models
    """
    reset_baselines()
    baselines = get_submissions_by_type(db, source_type="baseline")
    for submission in baselines:
        if not submission.code:
            continue
        add_baseline_example(
            submission_id=submission.id,
            code=submission.code,
            language=submission.language,
        )
    return StatusResponse(baseline_count=len(baselines))


@router.post("/baseline", response_model=BaselineSubmissionResponse)
def add_baseline_route(
    payload: BaselineSubmissionRequest, db: Session = Depends(get_db)
) -> BaselineSubmissionResponse:
    """
    Register a baseline (reference) solution.

    This:
    - Stores a baseline Submission in the database (including raw code)
    - Adds the code to the in-memory TF-IDF models for later similarity checks
    """
    submission = create_submission(
        db,
        source_type="baseline",
        language=payload.language,
        token_vector=None,
        ast_vector=None,
        student_id=payload.label,
        code=payload.code,
    )

    # Update in-memory model state
    add_baseline_example(
        submission_id=submission.id, code=payload.code, language=payload.language
    )

    return BaselineSubmissionResponse(
        id=submission.id,
        language=submission.language,
        label=submission.student_id,
    )


@router.post("/submit-code", response_model=PlagiarismResult)
def submit_code_route(
    payload: StudentSubmissionRequest, db: Session = Depends(get_db)
) -> PlagiarismResult:
    """
    Submit a student's code and compute similarity against all registered baselines.

    Returns the top-k most similar baseline submissions with cosine similarity scores.
    """
    if not has_baselines():
        raise HTTPException(
            status_code=400,
            detail="No baselines registered yet. Add at least one baseline via POST /baseline (any language).",
        )

    # Record this submission in the database (vectors left None for now).
    submission = create_submission(
        db,
        source_type="student",
        language=payload.language,
        token_vector=None,
        ast_vector=None,
        student_id=payload.student_id,
        code=payload.code,
    )

    scores = compute_similarity_against_baselines(
        code=payload.code, language=payload.language
    )
    if scores is None:
        # This should not happen if has_baselines() is true,
        # but we guard for safety.
        raise HTTPException(
            status_code=500,
            detail="Similarity model is not ready. Please add baselines again.",
        )

    baseline_ids = list(get_baseline_submission_ids())
    indices = top_k_indices(scores, k=payload.top_k)

    if not indices:
        return PlagiarismResult(query_submission_id=submission.id, matches=[])

    # Fetch baseline metadata from the database in a single query.
    selected_ids = [baseline_ids[i] for i in indices]
    baselines: List[Submission] = (
        db.query(Submission).filter(Submission.id.in_(selected_ids)).all()
    )
    baseline_by_id = {b.id: b for b in baselines}

    matches: List[SimilarityMatch] = []
    for idx in indices:
        baseline_id = baseline_ids[idx]
        baseline = baseline_by_id.get(baseline_id)
        if baseline is None:
            continue
        matches.append(
            SimilarityMatch(
                submission_id=baseline.id,
                source_type=baseline.source_type,
                label=baseline.student_id,
                language=baseline.language,
                similarity=float(scores[idx]),
            )
        )

    return PlagiarismResult(query_submission_id=submission.id, matches=matches)

