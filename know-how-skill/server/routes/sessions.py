"""
Session workflow API routes.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..db_models import SessionRecord
from ..models import (
    CreateSessionInput, InferInput, EvaluationInput, CorrectionInput,
    KnowHowSelectionInput, RollbackInput,
    AiUpdateInput, ManualUpdateInput, AddNewKHInput,
    SessionResponse, DiffItem, DiffResponse,
    ActivatedKnowHowItem, ActivatedKnowHowResponse,
    SinglePatchResponse, GenerateNewKHResponse,
)
from ..services import workflow_service as wf
from ..services import inference_service as infer_svc
from ..services import knowhow_service as kh_svc
from ..services import version_service as ver_svc
from ..config import KNOWLEDGE_DIRS

router = APIRouter(tags=["sessions"])

_executor = ThreadPoolExecutor(max_workers=2)


def _get_rec(session_id: str, db: DBSession) -> SessionRecord:
    rec = db.query(SessionRecord).get(session_id)
    if rec is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return rec


def _to_response(rec: SessionRecord) -> SessionResponse:
    return SessionResponse(
        id=rec.id,
        state=rec.state or "idle",
        question=rec.question or "",
        knowledge_dirs=rec.knowledge_dirs or [],
        inference_result=rec.inference_result or {},
        expert_evaluation=rec.expert_evaluation or {},
        expert_correction=rec.expert_correction or {},
        activated_knowhow=rec.activated_knowhow or [],
        selected_entry_keys=rec.selected_entry_keys or [],
        patches=rec.patches or {},
        test_result=rec.test_result or {},
    )


# ── CRUD ─────────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse)
def create_session(body: CreateSessionInput, db: DBSession = Depends(get_db)):
    kd = body.knowledge_dirs if body.knowledge_dirs else list(KNOWLEDGE_DIRS)
    rec = SessionRecord(knowledge_dirs=kd)
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return _to_response(rec)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    return _to_response(_get_rec(session_id, db))


# ── Step 1: Infer ────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/infer", response_model=SessionResponse)
def submit_inference(session_id: str, body: InferInput, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    wf.transition(db, rec, "infer", question=body.question)

    try:
        result = infer_svc.run_single_inference(
            question=body.question,
            knowledge_dirs=rec.knowledge_dirs or [],
        )
        activated = infer_svc.extract_activated_knowhow(result)
        wf.transition(
            db, rec, "inference_done",
            inference_result=result,
            activated_knowhow=activated,
        )
    except Exception as e:
        rec.state = "idle"
        db.commit()
        raise HTTPException(500, f"Inference failed: {e}")

    return _to_response(rec)


# ── Step 2: Evaluate ─────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/evaluate", response_model=SessionResponse)
def submit_evaluation(session_id: str, body: EvaluationInput, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    eval_data = {"error_type": body.error_type, "notes": body.notes}

    if body.error_type == "correct":
        wf.transition(db, rec, "correct", expert_evaluation=eval_data)
    else:
        wf.transition(db, rec, "incorrect", expert_evaluation=eval_data)

    return _to_response(rec)


# ── Step 3: Submit correction ────────────────────────────────────────────────

@router.post("/sessions/{session_id}/correct", response_model=SessionResponse)
def submit_correction(session_id: str, body: CorrectionInput, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    correction = {
        "corrected_answer": body.corrected_answer,
        "corrected_reasoning": body.corrected_reasoning,
        **(rec.expert_evaluation or {}),
    }
    wf.transition(db, rec, "submit_correction", expert_correction=correction)

    activated = rec.activated_knowhow or []
    if activated:
        wf.transition(db, rec, "kh_loaded")

    return _to_response(rec)


# ── Step 4: Get activated know-how ───────────────────────────────────────────

@router.get("/sessions/{session_id}/activated-knowhow", response_model=ActivatedKnowHowResponse)
def get_activated_knowhow(session_id: str, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    items = []
    for kh in (rec.activated_knowhow or []):
        items.append(ActivatedKnowHowItem(
            entry_key=kh.get("entry_key", ""),
            source_dir=kh.get("source_dir", ""),
            knowledge_dir=kh.get("knowledge_dir", ""),
            title=kh.get("title", ""),
            scope=kh.get("scope", ""),
            kh_text=kh.get("kh_text", ""),
            reasoning_chain=kh.get("reasoning_chain", ""),
            derived_answer=kh.get("derived_answer", ""),
        ))
    return ActivatedKnowHowResponse(items=items)


# ── Step 5: Select know-how blocks (legacy batch) ───────────────────────────

@router.post("/sessions/{session_id}/select-knowhow", response_model=SessionResponse)
def select_knowhow(session_id: str, body: KnowHowSelectionInput, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)

    patches = kh_svc.apply_corrections_batch(
        activated_knowhow=rec.activated_knowhow or [],
        selected_keys=body.entry_keys,
        question=rec.question or "",
        expert_correction=rec.expert_correction or {},
    )

    wf.transition(
        db, rec, "select_kh",
        selected_entry_keys=body.entry_keys,
        patches=patches,
    )
    return _to_response(rec)


# ── Step 5b: Per-block operations (AI update / manual update / new KH) ──────

def _find_kd_for_key(rec: SessionRecord, entry_key: str) -> str:
    for kh in (rec.activated_knowhow or []):
        if kh.get("entry_key") == entry_key:
            return kh.get("knowledge_dir", "")
    return ""


@router.post("/sessions/{session_id}/ai-update-knowhow", response_model=SinglePatchResponse)
def ai_update_knowhow(session_id: str, body: AiUpdateInput, db: DBSession = Depends(get_db)):
    """AI-generate a patch for a single know-how block."""
    rec = _get_rec(session_id, db)
    if rec.state not in ("awaiting_kh_selection",):
        raise HTTPException(400, f"Cannot AI-update in state {rec.state}")

    knowledge_dir = _find_kd_for_key(rec, body.entry_key)
    if not knowledge_dir:
        raise HTTPException(404, f"Knowledge dir not found for entry_key={body.entry_key}")

    result = kh_svc.ai_update_single(
        knowledge_dir=knowledge_dir,
        entry_key=body.entry_key,
        question=rec.question or "",
        expert_correction=rec.expert_correction or {},
    )

    patches = dict(rec.patches or {})
    patches[body.entry_key] = result
    rec.patches = patches
    selected = list(rec.selected_entry_keys or [])
    if body.entry_key not in selected:
        selected.append(body.entry_key)
    rec.selected_entry_keys = selected
    db.commit()
    db.refresh(rec)

    return SinglePatchResponse(
        entry_key=body.entry_key,
        title=result.get("title", ""),
        original_text=result.get("original_text", ""),
        patched_text=result.get("patched_text", ""),
        original_json=result.get("original", {}),
        patched_json=result.get("patched", {}),
        operations=result.get("operations", []),
        diff_description=result.get("diff_description", ""),
    )


@router.post("/sessions/{session_id}/manual-update-knowhow", response_model=SinglePatchResponse)
def manual_update_knowhow(session_id: str, body: ManualUpdateInput, db: DBSession = Depends(get_db)):
    """Save a manually-edited know-how block."""
    rec = _get_rec(session_id, db)
    if rec.state not in ("awaiting_kh_selection",):
        raise HTTPException(400, f"Cannot manual-update in state {rec.state}")

    knowledge_dir = _find_kd_for_key(rec, body.entry_key)
    if not knowledge_dir:
        raise HTTPException(404, f"Knowledge dir not found for entry_key={body.entry_key}")

    result = kh_svc.manual_update_single(
        knowledge_dir=knowledge_dir,
        entry_key=body.entry_key,
        patched_json=body.patched_json,
    )

    patches = dict(rec.patches or {})
    patches[body.entry_key] = result
    rec.patches = patches
    selected = list(rec.selected_entry_keys or [])
    if body.entry_key not in selected:
        selected.append(body.entry_key)
    rec.selected_entry_keys = selected
    db.commit()
    db.refresh(rec)

    return SinglePatchResponse(
        entry_key=body.entry_key,
        title=result.get("title", ""),
        original_text=result.get("original_text", ""),
        patched_text=result.get("patched_text", ""),
        original_json=result.get("original", {}),
        patched_json=result.get("patched", {}),
        operations=result.get("operations", []),
        diff_description=result.get("diff_description", ""),
    )


@router.post("/sessions/{session_id}/generate-new-knowhow", response_model=GenerateNewKHResponse)
def generate_new_knowhow(session_id: str, db: DBSession = Depends(get_db)):
    """AI-generate a brand-new know-how block based on the question + correction."""
    rec = _get_rec(session_id, db)
    if rec.state not in ("awaiting_kh_selection",):
        raise HTTPException(400, f"Cannot generate new KH in state {rec.state}")

    result = kh_svc.generate_new_knowhow(
        question=rec.question or "",
        expert_correction=rec.expert_correction or {},
    )

    return GenerateNewKHResponse(
        knowhow_json=result.get("knowhow_json", {}),
        knowhow_text=result.get("knowhow_text", ""),
    )


@router.post("/sessions/{session_id}/add-new-knowhow", response_model=SinglePatchResponse)
def add_new_knowhow(session_id: str, body: AddNewKHInput, db: DBSession = Depends(get_db)):
    """Confirm and add a new know-how block (AI-generated or manually created)."""
    rec = _get_rec(session_id, db)
    if rec.state not in ("awaiting_kh_selection",):
        raise HTTPException(400, f"Cannot add new KH in state {rec.state}")

    knowledge_dir = body.knowledge_dir
    if not knowledge_dir:
        kd_list = rec.knowledge_dirs or []
        if kd_list:
            knowledge_dir = kd_list[0]
        else:
            raise HTTPException(400, "No knowledge_dir available")

    result = kh_svc.add_new_knowhow_entry(knowledge_dir, body.knowhow_json)
    new_key = result["entry_key"]

    patches = dict(rec.patches or {})
    patches[new_key] = result
    rec.patches = patches
    selected = list(rec.selected_entry_keys or [])
    if new_key not in selected:
        selected.append(new_key)
    rec.selected_entry_keys = selected

    activated = list(rec.activated_knowhow or [])
    activated.append({
        "entry_key": new_key,
        "source_dir": "",
        "knowledge_dir": knowledge_dir,
        "title": body.knowhow_json.get("title", ""),
        "scope": body.knowhow_json.get("scope", ""),
        "kh_text": kh_svc.render_knowhow_text(body.knowhow_json),
        "reasoning_chain": "",
        "derived_answer": "",
        "is_new": True,
    })
    rec.activated_knowhow = activated
    db.commit()
    db.refresh(rec)

    return SinglePatchResponse(
        entry_key=new_key,
        title=result.get("title", ""),
        original_text=result.get("original_text", ""),
        patched_text=result.get("patched_text", ""),
        original_json=result.get("original", {}),
        patched_json=result.get("patched", {}),
        operations=result.get("operations", []),
        diff_description=result.get("diff_description", ""),
    )


@router.post("/sessions/{session_id}/confirm-patches", response_model=SessionResponse)
def confirm_patches(session_id: str, db: DBSession = Depends(get_db)):
    """Transition from awaiting_kh_selection → showing_diff once per-block edits are done."""
    rec = _get_rec(session_id, db)
    if rec.state != "awaiting_kh_selection":
        raise HTTPException(400, f"Cannot confirm patches in state {rec.state}")

    patches = rec.patches or {}
    if not patches:
        raise HTTPException(400, "No patches to confirm. Please AI-update or manually edit at least one block.")

    wf.transition(db, rec, "select_kh")
    return _to_response(rec)


# ── Step 6: Get diff ─────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/diff", response_model=DiffResponse)
def get_diff(session_id: str, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    patches = rec.patches or {}
    items = []
    for key, p in patches.items():
        items.append(DiffItem(
            entry_key=key,
            title=p.get("title", ""),
            original_text=p.get("original_text", ""),
            patched_text=p.get("patched_text", ""),
            original_json=p.get("original", {}),
            patched_json=p.get("patched", {}),
            operations=p.get("operations", []),
            diff_description=p.get("diff_description", ""),
        ))
    return DiffResponse(items=items)


# ── Step 7: Confirm and test ─────────────────────────────────────────────────

@router.post("/sessions/{session_id}/test", response_model=SessionResponse)
def run_test(session_id: str, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    wf.transition(db, rec, "confirm_patch")

    patches = rec.patches or {}
    knowledge_dirs = rec.knowledge_dirs or []

    kd_patches: dict[str, dict] = {}
    for kh in (rec.activated_knowhow or []):
        ek = kh.get("entry_key", "")
        kd = kh.get("knowledge_dir", "")
        if ek in patches and kd:
            kd_patches.setdefault(kd, {})[ek] = patches[ek]

    temp_dirs = []
    test_knowledge_dirs = list(knowledge_dirs)
    try:
        for kd, kd_p in kd_patches.items():
            temp_kd = ver_svc.create_temp_knowledge(kd, kd_p)
            idx = None
            for i, d in enumerate(test_knowledge_dirs):
                if d == kd:
                    idx = i
                    break
            if idx is not None:
                test_knowledge_dirs[idx] = temp_kd
            else:
                test_knowledge_dirs.append(temp_kd)
            temp_dirs.append(temp_kd)

        test_result = infer_svc.run_single_inference(
            question=rec.question or "",
            knowledge_dirs=test_knowledge_dirs,
        )
        wf.transition(db, rec, "test_done", test_result=test_result)
    except Exception as e:
        rec.state = "showing_diff"
        db.commit()
        raise HTTPException(500, f"Test inference failed: {e}")
    finally:
        for td in temp_dirs:
            parent = td
            if not td.endswith("_knowledge"):
                parent = td
            try:
                shutil.rmtree(parent, ignore_errors=True)
            except Exception:
                pass

    return _to_response(rec)


# ── Step 8: Save version ─────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/save", response_model=SessionResponse)
def save_version(session_id: str, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    wf.transition(db, rec, "satisfied")

    patches = rec.patches or {}
    kd_patches: dict[str, dict] = {}
    for kh in (rec.activated_knowhow or []):
        ek = kh.get("entry_key", "")
        kd = kh.get("knowledge_dir", "")
        if ek in patches and kd:
            kd_patches.setdefault(kd, {})[ek] = patches[ek]

    for kd, kd_p in kd_patches.items():
        ver_svc.save_version(db, kd, description=f"Session {session_id}", session_id=session_id)
        ver_svc.persist_patches(kd, kd_p)

    return _to_response(rec)


# ── Rollback ─────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/rollback", response_model=SessionResponse)
def rollback(session_id: str, body: RollbackInput, db: DBSession = Depends(get_db)):
    rec = _get_rec(session_id, db)
    wf.rollback_to(db, rec, body.target_step)
    return _to_response(rec)
