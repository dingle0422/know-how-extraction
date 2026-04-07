"""
FastAPI application entry point.
"""

import os
import sys

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_ROOT = os.path.dirname(_SERVER_DIR)
for _p in (
    _SKILL_ROOT,
    os.path.join(_SKILL_ROOT, "extraction"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_2"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_1"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes import sessions, batch_test, versions

app = FastAPI(title="Know-How Incremental Training", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api")
app.include_router(batch_test.router, prefix="/api")
app.include_router(versions.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}
