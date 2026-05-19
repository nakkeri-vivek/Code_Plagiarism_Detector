"""
In-memory model state for plagiarism detection.

Uses a single global corpus: all baselines (any language) are in one TF-IDF model.
Similarity is structure-based: any submitted code is compared against all baselines
by token/structure patterns, so e.g. Java can match against Python or C++ baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from app.feature_extraction import (
    AstTfidfVectorizer,
    TokenTfidfVectorizer,
    extract_features,
)
from app.similarity import compute_cosine_similarity


@dataclass
class PlagiarismModelState:
    token_vectorizer: Optional[TokenTfidfVectorizer] = None
    ast_vectorizer: Optional[AstTfidfVectorizer] = None

    baseline_submission_ids: List[int] = field(default_factory=list)
    baseline_token_docs: List[str] = field(default_factory=list)
    baseline_ast_docs: List[str] = field(default_factory=list)

    baseline_token_matrix: Optional[np.ndarray] = None
    baseline_ast_matrix: Optional[np.ndarray] = None


# Single global state: one corpus for all languages (structure-based similarity).
state = PlagiarismModelState()


def _rebuild_token_model() -> None:
    """Fit/re-fit the token TF-IDF model on all baseline token documents."""
    if not state.baseline_token_docs:
        state.token_vectorizer = None
        state.baseline_token_matrix = None
        return

    vectorizer = TokenTfidfVectorizer()
    matrix = vectorizer.fit_transform(state.baseline_token_docs)
    state.token_vectorizer = vectorizer
    state.baseline_token_matrix = matrix.toarray().astype(np.float32)


def _rebuild_ast_model() -> None:
    """Fit/re-fit the AST TF-IDF model on all non-empty baseline AST documents."""
    if not any(doc.strip() for doc in state.baseline_ast_docs):
        state.ast_vectorizer = None
        state.baseline_ast_matrix = None
        return

    vectorizer = AstTfidfVectorizer()
    matrix = vectorizer.fit_transform(state.baseline_ast_docs)
    state.ast_vectorizer = vectorizer
    state.baseline_ast_matrix = matrix.toarray().astype(np.float32)


def add_baseline_example(submission_id: int, code: str, language: str = "python") -> None:
    """
    Add a baseline to the global corpus. Language is used only for preprocessing
    (how to tokenize this code); all baselines are merged into one model.
    """
    features = extract_features(code, language=language)
    token_doc = features["token_document"]
    ast_doc = features["ast_document"] or ""

    state.baseline_submission_ids.append(submission_id)
    state.baseline_token_docs.append(token_doc)
    state.baseline_ast_docs.append(ast_doc)

    _rebuild_token_model()
    _rebuild_ast_model()


def has_baselines() -> bool:
    """True if at least one baseline exists (any language)."""
    return bool(state.baseline_submission_ids)


def reset_baselines() -> None:
    """Clear in-memory corpus and models. Does not delete database rows."""
    state.token_vectorizer = None
    state.ast_vectorizer = None
    state.baseline_submission_ids.clear()
    state.baseline_token_docs.clear()
    state.baseline_ast_docs.clear()
    state.baseline_token_matrix = None
    state.baseline_ast_matrix = None


def get_baseline_submission_ids() -> Sequence[int]:
    """Ordered list of baseline IDs (same order as rows in the similarity matrix)."""
    return list(state.baseline_submission_ids)


def total_baseline_count() -> int:
    return len(state.baseline_submission_ids)


def compute_similarity_against_baselines(
    code: str, language: str = "python"
) -> Optional[np.ndarray]:
    """
    Compare the given code against all baselines (all languages) using
    structure/token similarity. language is used only to preprocess the query code.
    """
    if not state.baseline_submission_ids:
        return None

    features = extract_features(code, language=language)
    token_doc = features["token_document"]
    ast_doc = features["ast_document"] or ""

    token_scores: Optional[np.ndarray] = None
    ast_scores: Optional[np.ndarray] = None

    if state.token_vectorizer is not None and state.baseline_token_matrix is not None:
        token_vec = state.token_vectorizer.transform(token_doc).toarray().astype(
            np.float32
        )[0]
        token_scores = compute_cosine_similarity(token_vec, state.baseline_token_matrix)

    if state.ast_vectorizer is not None and state.baseline_ast_matrix is not None:
        ast_vec = state.ast_vectorizer.transform(ast_doc).toarray().astype(
            np.float32
        )[0]
        ast_scores = compute_cosine_similarity(ast_vec, state.baseline_ast_matrix)

    if token_scores is None and ast_scores is None:
        return None
    if token_scores is None:
        return ast_scores
    if ast_scores is None:
        return token_scores

    return (token_scores + ast_scores) / 2.0


