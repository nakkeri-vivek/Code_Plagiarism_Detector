## CodeShield - AI-powered Code Plagiarism Detection API

**CodeShield** is a FastAPI-based backend for detecting plagiarism in source code.
It uses code preprocessing, token (all languages) + AST (Python-only) TF-IDF
features, and cosine similarity to compare student submissions against known
baseline solutions.

### 1. Setup

- **Python**: Use Python 3.9+.
- Install dependencies (ideally in a virtualenv):

```bash
pip install -r requirements.txt
```

### 2. Run the API

From the project root:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

You can explore it via the automatic docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 3. Main endpoints

- **Health check**

  - `GET /health`
  - Simple liveness check: returns `{"status": "ok"}`.

- **Status**

  - `GET /status`
  - Returns how many baseline submissions are currently loaded in memory:
    - `{"baseline_count": <int>}`

- **Add baseline solution**

  - `POST /baseline`
  - Body:

```json
{
  "code": "def add(a, b): return a + b",
  "language": "python",
  "label": "assignment1_solution"
}
```

  - Response:

```json
{
  "id": 1,
  "language": "python",
  "label": "assignment1_solution"
}
```

  - This stores a baseline submission in the database and updates the
    in-memory TF-IDF models so it can be used for similarity checks.

- **Submit student code for plagiarism check**

  - `POST /submit-code`
  - Body:

```json
{
  "code": "def add(x, y): return x + y",
  "language": "python",
  "student_id": "student123",
  "top_k": 5
}
```

  - Response example:

```json
{
  "query_submission_id": 2,
  "matches": [
    {
      "submission_id": 1,
      "source_type": "baseline",
      "label": "assignment1_solution",
      "language": "python",
      "similarity": 0.93
    }
  ]
}
```

### 4. How it works (high level)

- **Preprocessing** (`app/preprocessing.py`)
  - **All languages (generic)**: removes comments (via Pygments tokenization), normalizes identifiers, and normalizes whitespace.
  - **Python (enhanced)**: removes docstrings, normalizes identifiers, and normalizes code via Python AST unparse.

- **Feature extraction** (`app/feature_extraction.py`)
  - **All languages**: token-based features + TF-IDF.
  - **Python only**: AST node-type sequence + TF-IDF (extra signal).

- **Similarity** (`app/similarity.py`)
  - Cosine similarity between vectors and top-k retrieval.

- **Model state** (`app/model_state.py`)
  - **Single global corpus**: all baselines (any language) live in one TF-IDF model.
  - Similarity is **structure-based**: any submitted code (Java, C++, Python, etc.) is
    compared against all baselines. You do not need baselines in the same language.
  - After a process restart, use "Rebuild baselines" (or `POST /baseline/reset`) to
    reload from the database.

### 5. Notes and extensions

- **Language-agnostic similarity**:
  - Baselines can be in any mix of languages (e.g. only Python). Submissions in
    Java or C++ are still compared against those baselines using structural token
    patterns (keywords, operators, normalized identifiers).
  - Python gets an extra AST signal when baselines include Python; other languages
    use token similarity only.
- You can add additional endpoints for:
  - Listing stored submissions.
  - Returning raw baseline code (if you choose to store it).
  - Exporting/importing model state.

