"""
Code preprocessing utilities.

STEP 2 responsibilities:
- Remove comments / docstrings
- Normalize whitespace
- Rename variables consistently
Supports Python best; provides a generic fallback for other languages.
"""

from __future__ import annotations

import ast
import keyword
import re
from textwrap import dedent
from typing import Dict, Optional

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from pygments.token import Comment, Name, String, Text, Whitespace, Keyword


def preprocess_code(code: str, language: str = "python") -> str:
    """
    Preprocess raw source code, dispatching by language.
    """
    language = language.lower()
    if language == "python":
        return preprocess_python_code(code)
    return preprocess_generic_code(code, language=language)


def preprocess_python_code(code: str) -> str:
    """
    Preprocess Python code:
    - Dedent and strip leading/trailing whitespace
    - Remove docstrings (module, class, function)
    - Rename local variables and arguments to canonical names
    - Re-unparse the AST, which also normalizes whitespace and removes comments
    """
    # Normalize basic indentation and leading/trailing whitespace first.
    normalized = dedent(code).strip()
    if not normalized:
        return ""

    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        # If parsing fails, return a stripped version so we don't break on bad code.
        return normalized

    # First remove docstrings, then rename variables.
    tree = DocstringRemover().visit(tree)
    tree = VariableRenamer().visit(tree)
    ast.fix_missing_locations(tree)

    try:
        # ast.unparse is available in Python 3.9+
        formatted = ast.unparse(tree)
    except Exception:
        # Fallback: return normalized original if unparsing fails for some reason.
        return normalized

    # Final whitespace normalization: strip leading/trailing space.
    return formatted.strip()


def _safe_get_lexer(language: str):
    """
    Return a Pygments lexer for a language name, or None if unavailable.
    """
    try:
        return get_lexer_by_name(language)
    except ClassNotFound:
        return None


def preprocess_generic_code(code: str, language: str) -> str:
    """
    Generic preprocessing for non-Python languages using Pygments.

    Goals:
    - Remove comments (//, /* */ etc) and docstring-like tokens when available
    - Normalize identifiers to reduce variable-renaming noise
    - Normalize whitespace

    This is intentionally heuristic to work across many languages.
    """
    normalized = dedent(code).strip()
    if not normalized:
        return ""

    lexer = _safe_get_lexer(language)
    if lexer is None:
        # If we can't tokenize, at least normalize whitespace.
        return re.sub(r"\s+", " ", normalized).strip()

    name_map: Dict[str, str] = {}
    counter = 0

    def get_name(original: str) -> str:
        nonlocal counter
        if original in name_map:
            return name_map[original]
        new_name = f"var_{counter}"
        counter += 1
        name_map[original] = new_name
        return new_name

    out_parts = []

    for tok_type, tok_value in lex(normalized, lexer):
        # Drop comments entirely.
        if tok_type in Comment or tok_type is Comment:
            continue

        # Drop docstring-like strings if lexer marks them.
        if tok_type in String.Doc or tok_type is String.Doc:
            continue

        # Ignore pure whitespace tokens; we reinsert spaces later.
        if tok_type in (Text, Whitespace) or tok_type in Text or tok_type in Whitespace:
            continue

        value = tok_value.strip()
        if not value:
            continue

        # Keep keywords as-is.
        if tok_type in Keyword or tok_type is Keyword:
            out_parts.append(value)
            continue

        # Normalize identifiers (variable/class/function names) across languages.
        if tok_type in Name or tok_type is Name:
            out_parts.append(get_name(value))
            continue

        out_parts.append(value)

    return " ".join(out_parts).strip()


class DocstringRemover(ast.NodeTransformer):
    """
    Remove docstrings from modules, classes, and functions.
    """

    def _strip_docstring(self, body):
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node


class VariableRenamer(ast.NodeTransformer):
    """
    Rename variables and function arguments to canonical names.

    The goal is to reduce the effect of different naming choices on similarity:
    - Local variables: var_0, var_1, ...
    - Arguments: arg_0, arg_1, ...
    Special names like 'self' and 'cls', keywords, and builtins are left as-is.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name_map: Dict[str, str] = {}
        self.counter = 0
        self.arg_counter = 0
        self._builtins = set(dir(__builtins__))  # type: ignore[arg-type]

    def _is_special(self, name: str) -> bool:
        return name in {"self", "cls"}

    def _should_rename(self, name: str) -> bool:
        if self._is_special(name):
            return False
        if keyword.iskeyword(name):
            return False
        if name in self._builtins:
            return False
        return True

    def _get_or_create_name(self, original: str) -> str:
        if original in self.name_map:
            return self.name_map[original]
        new_name = f"var_{self.counter}"
        self.counter += 1
        self.name_map[original] = new_name
        return new_name

    def _get_or_create_arg_name(self, original: str) -> str:
        if original in self.name_map:
            return self.name_map[original]
        new_name = f"arg_{self.arg_counter}"
        self.arg_counter += 1
        self.name_map[original] = new_name
        return new_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Only rename identifiers that correspond to variables, not attributes, etc.
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            if self._should_rename(node.id):
                node.id = self._get_or_create_name(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if self._should_rename(node.arg):
            node.arg = self._get_or_create_arg_name(node.arg)
        return self.generic_visit(node)



