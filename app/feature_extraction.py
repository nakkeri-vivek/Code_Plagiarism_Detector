"""
Feature extraction utilities.

STEP 3:
- Token-based features using Pygments + TF-IDF.

STEP 4 (later):
- AST-based features using Python ast + TF-IDF.
"""

from __future__ import annotations

from typing import Any, Dict, List

import ast
import numpy as np
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from pygments.token import Comment, Text, Token
from sklearn.feature_extraction.text import TfidfVectorizer

from app.preprocessing import preprocess_code


# -----------------------------
# Token sequence extraction
# -----------------------------

IGNORED_TOKEN_TYPES = {
    Comment,          # comments
    Text,             # whitespace / newlines
    Token.Literal.String.Doc,  # docstrings (already removed by preprocessing, but safe)
}


def extract_token_sequence(code: str, language: str = "python") -> List[str]:
    """
    Extract a sequence of tokens from source code using Pygments.

    - Uses the preprocessing step first (removes comments/docstrings, normalizes code, renames vars)
    - Ignores comments and whitespace-like tokens
    - Returns a list of token strings (for TF-IDF)
    """
    language = language.lower()
    preprocessed = preprocess_code(code, language=language)

    # For non-Python languages, preprocess_generic_code already returns a
    # whitespace-separated token stream. Splitting is fast and works well.
    if language != "python":
        return preprocessed.split()

    # Python path: use the Python lexer so we preserve punctuation and operators.
    try:
        lexer = get_lexer_by_name("python")
    except ClassNotFound:
        return preprocessed.split()
    tokens: List[str] = []

    for tok_type, tok_value in lex(preprocessed, lexer):
        # Skip comments, whitespace, and docstring-literal tokens
        if tok_type in IGNORED_TOKEN_TYPES:
            continue
        # Many token types are subtypes; skip if base category is Comment or Text.
        if tok_type in Comment or tok_type in Text:
            continue
        value = tok_value.strip()
        if not value:
            continue
        tokens.append(value)

    return tokens


# -----------------------------
# TF-IDF token vectorizer
# -----------------------------

class TokenTfidfVectorizer:
    """
    Wrapper around scikit-learn's TfidfVectorizer for code tokens.

    Usage:
        extractor = TokenTfidfVectorizer()
        docs = [" ".join(extract_token_sequence(code1)), " ".join(extract_token_sequence(code2))]
        extractor.fit(docs)
        vec = extractor.transform(" ".join(extract_token_sequence(new_code)))
    """

    def __init__(self) -> None:
        # We supply our own "tokenizer" by pre-splitting on spaces;
        # lowercasing is disabled because code is case-sensitive.
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda s: s.split(),
            preprocessor=None,
            lowercase=False,
        )

    def fit(self, documents: List[str]) -> None:
        self.vectorizer.fit(documents)

    def fit_transform(self, documents: List[str]):
        return self.vectorizer.fit_transform(documents)

    def transform(self, document: str):
        return self.vectorizer.transform([document])

    def to_dict(self) -> Dict[str, Any]:
        """
        Export vectorizer configuration and vocabulary.
        This can be serialized (e.g., via JSON or pickle) if desired.
        """
        return {
            "vocabulary_": self.vectorizer.vocabulary_,
            "idf_": getattr(self.vectorizer, "idf_", None).tolist()
            if getattr(self.vectorizer, "idf_", None) is not None
            else None,
        }


# -----------------------------
# AST-based features (STEP 4)
# -----------------------------


def ast_to_normalized_sequence(code: str, language: str = "python") -> str:
    """
    Convert code into a normalized AST representation string.

    For Python:
    - Uses the preprocessed code (variables already normalized)
    - Parses to AST and walks nodes
    - Produces a simple space-separated sequence of node types

    If parsing fails, returns an empty string.
    """
    language = language.lower()
    if language != "python":
        return ""

    preprocessed = preprocess_code(code, language=language)
    if not preprocessed:
        return ""

    try:
        tree = ast.parse(preprocessed)
    except SyntaxError:
        return ""

    node_types: List[str] = []

    for node in ast.walk(tree):
        node_types.append(type(node).__name__)

    return " ".join(node_types)


class AstTfidfVectorizer:
    """
    TF-IDF vectorizer specialized for AST node-type sequences.

    Works similarly to TokenTfidfVectorizer but on node-type strings.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda s: s.split(),
            preprocessor=None,
            lowercase=False,
        )

    def fit(self, documents: List[str]) -> None:
        self.vectorizer.fit(documents)

    def fit_transform(self, documents: List[str]):
        return self.vectorizer.fit_transform(documents)

    def transform(self, document: str):
        return self.vectorizer.transform([document])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocabulary_": self.vectorizer.vocabulary_,
            "idf_": getattr(self.vectorizer, "idf_", None).tolist()
            if getattr(self.vectorizer, "idf_", None) is not None
            else None,
        }


def extract_features(code: str, language: str = "python") -> Dict[str, Any]:
    """
    High-level feature extraction entrypoint.

    Returns:
        - token_document: string of tokens (for TokenTfidfVectorizer)
        - ast_document: string of AST node type names (for AstTfidfVectorizer)
        - token_vector / ast_vector: placeholders, to be filled by callers
    """
    tokens = extract_token_sequence(code, language=language)
    token_document = " ".join(tokens)

    ast_document = ast_to_normalized_sequence(code, language=language)

    return {
        "token_document": token_document,
        "ast_document": ast_document,
        "token_vector": None,
        "ast_vector": None,
    }




