"""
Workflow state machine for incremental training sessions.
"""

from __future__ import annotations

import copy
from typing import Any

from sqlalchemy.orm import Session as DBSession

from ..db_models import SessionRecord

VALID_STATES = {
    "idle",
    "inferring",
    "awaiting_eval",
    "awaiting_correction",
    "showing_activated_kh",
    "awaiting_kh_selection",
    "showing_diff",
    "testing",
    "awaiting_test_eval",
    "saved",
}

TRANSITIONS: dict[str, dict[str, str]] = {
    "idle":                  {"infer": "inferring"},
    "inferring":             {"inference_done": "awaiting_eval"},
    "awaiting_eval":         {"correct": "idle", "incorrect": "awaiting_correction"},
    "awaiting_correction":   {"submit_correction": "showing_activated_kh"},
    "showing_activated_kh":  {"kh_loaded": "awaiting_kh_selection"},
    "awaiting_kh_selection": {"select_kh": "showing_diff"},
    "showing_diff":          {"confirm_patch": "testing"},
    "testing":               {"test_done": "awaiting_test_eval"},
    "awaiting_test_eval":    {
        "satisfied": "saved",
        "rollback_eval": "awaiting_eval",
        "rollback_correction": "awaiting_correction",
        "rollback_kh_selection": "awaiting_kh_selection",
    },
    "saved":                 {"infer": "inferring"},
}

_SNAPSHOT_FIELDS = (
    "state", "question", "inference_result", "expert_evaluation",
    "expert_correction", "activated_knowhow", "selected_entry_keys",
    "patches", "test_result",
)


def _snapshot(rec: SessionRecord) -> dict[str, Any]:
    return {f: copy.deepcopy(getattr(rec, f)) for f in _SNAPSHOT_FIELDS}


def _restore(rec: SessionRecord, snap: dict[str, Any]):
    for f in _SNAPSHOT_FIELDS:
        if f in snap:
            setattr(rec, f, copy.deepcopy(snap[f]))


def transition(db: DBSession, rec: SessionRecord, action: str, **extra) -> SessionRecord:
    current = rec.state or "idle"
    allowed = TRANSITIONS.get(current, {})
    if action not in allowed:
        raise ValueError(
            f"Invalid transition: state={current!r}, action={action!r}. "
            f"Allowed actions: {list(allowed.keys())}"
        )

    history: list = list(rec.history or [])
    history.append(_snapshot(rec))
    rec.history = history

    new_state = allowed[action]
    rec.state = new_state

    for k, v in extra.items():
        if hasattr(rec, k):
            setattr(rec, k, v)

    db.commit()
    db.refresh(rec)
    return rec


def rollback_to(db: DBSession, rec: SessionRecord, target_state: str) -> SessionRecord:
    history: list = list(rec.history or [])
    if not history:
        raise ValueError("No history to rollback to")

    for i in range(len(history) - 1, -1, -1):
        if history[i].get("state") == target_state:
            snap = history[i]
            rec.history = history[:i]
            _restore(rec, snap)
            db.commit()
            db.refresh(rec)
            return rec

    raise ValueError(f"State {target_state!r} not found in history")
