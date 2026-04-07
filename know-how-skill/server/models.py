"""
Pydantic request / response models.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


# ── Session ──────────────────────────────────────────────────────────────────

class CreateSessionInput(BaseModel):
    knowledge_dirs: list[str] = []


class InferInput(BaseModel):
    question: str


class EvaluationInput(BaseModel):
    error_type: Literal[
        "correct",
        "conclusion_error",
        "logic_error",
        "detail_error",
        "other",
    ]
    notes: str = ""


class CorrectionInput(BaseModel):
    corrected_answer: str
    corrected_reasoning: str


class KnowHowSelectionInput(BaseModel):
    entry_keys: list[str]


class AiUpdateInput(BaseModel):
    entry_key: str


class ManualUpdateInput(BaseModel):
    entry_key: str
    patched_json: dict[str, Any]


class GenerateNewKHResponse(BaseModel):
    knowhow_json: dict[str, Any]
    knowhow_text: str


class AddNewKHInput(BaseModel):
    knowhow_json: dict[str, Any]
    knowledge_dir: str = ""


class SinglePatchResponse(BaseModel):
    entry_key: str
    title: str
    original_text: str
    patched_text: str
    original_json: dict[str, Any]
    patched_json: dict[str, Any]
    operations: list[dict[str, Any]]
    diff_description: str


class RollbackInput(BaseModel):
    target_step: Literal[
        "awaiting_eval",
        "awaiting_correction",
        "awaiting_kh_selection",
    ]


# ── Responses ────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    id: str
    state: str
    question: str
    knowledge_dirs: list[str]
    inference_result: dict[str, Any]
    expert_evaluation: dict[str, Any]
    expert_correction: dict[str, Any]
    activated_knowhow: list[dict[str, Any]]
    selected_entry_keys: list[str]
    patches: dict[str, Any]
    test_result: dict[str, Any]

    class Config:
        from_attributes = True


class DiffItem(BaseModel):
    entry_key: str
    title: str
    original_text: str
    patched_text: str
    original_json: dict[str, Any]
    patched_json: dict[str, Any]
    operations: list[dict[str, Any]]
    diff_description: str


class DiffResponse(BaseModel):
    items: list[DiffItem]


class ActivatedKnowHowItem(BaseModel):
    entry_key: str
    source_dir: str
    knowledge_dir: str
    title: str
    scope: str
    kh_text: str
    reasoning_chain: str
    derived_answer: str


class ActivatedKnowHowResponse(BaseModel):
    items: list[ActivatedKnowHowItem]


# ── Batch test ───────────────────────────────────────────────────────────────

class BatchTestResponse(BaseModel):
    id: str
    status: str
    total: int
    completed: int
    results: list[dict[str, Any]]


# ── Versions ─────────────────────────────────────────────────────────────────

class VersionResponse(BaseModel):
    id: int
    knowledge_dir: str
    description: str
    session_id: str
    created_at: str
