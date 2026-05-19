"""
Similarity computation utilities.

Responsibilities (STEP 7):
- One-to-many cosine similarity between a query vector and stored vectors.
"""

from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(
    query_vector: np.ndarray, corpus_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and many corpus vectors.
    """
    # sklearn expects 2D arrays
    return cosine_similarity(query_vector.reshape(1, -1), corpus_vectors).flatten()


def top_k_indices(scores: np.ndarray, k: int = 5) -> List[int]:
    """
    Return indices of top-k scores in descending order.
    """
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    return np.argsort(scores)[-k:][::-1].tolist()


