"""
Entry point for the FastAPI application.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import init_db
from app.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Code Plagiarism Detection API")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Allow all origins for simplicity in a project setting; in production, restrict this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    # Initialize database tables
    init_db()


app.include_router(api_router)


