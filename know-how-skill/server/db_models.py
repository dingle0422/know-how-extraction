"""
SQLAlchemy ORM models.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, Integer, JSON
from .database import Base


def _utcnow():
    return datetime.now(timezone.utc)


def _uuid():
    return uuid.uuid4().hex


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True, default=_uuid)
    state = Column(String(40), nullable=False, default="idle")
    question = Column(Text, default="")
    knowledge_dirs = Column(JSON, default=list)
    inference_result = Column(JSON, default=dict)
    expert_evaluation = Column(JSON, default=dict)
    expert_correction = Column(JSON, default=dict)
    activated_knowhow = Column(JSON, default=list)
    selected_entry_keys = Column(JSON, default=list)
    patches = Column(JSON, default=dict)
    test_result = Column(JSON, default=dict)
    history = Column(JSON, default=list)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class KnowledgeVersion(Base):
    __tablename__ = "knowledge_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_dir = Column(Text, nullable=False)
    snapshot = Column(JSON, nullable=False)
    description = Column(Text, default="")
    session_id = Column(String(32), default="")
    created_at = Column(DateTime, default=_utcnow)


class BatchTestRecord(Base):
    __tablename__ = "batch_tests"

    id = Column(String(32), primary_key=True, default=_uuid)
    knowledge_dirs = Column(JSON, default=list)
    status = Column(String(20), default="pending")
    total = Column(Integer, default=0)
    completed = Column(Integer, default=0)
    results = Column(JSON, default=list)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
