"""
Batch testing API routes.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session as DBSession

from ..database import get_db, SessionLocal
from ..db_models import BatchTestRecord
from ..models import BatchTestResponse
from ..services import inference_service as infer_svc
from ..config import KNOWLEDGE_DIRS

router = APIRouter(tags=["batch-test"])


def _run_batch(batch_id: str, file_path: str, knowledge_dirs: list[str]):
    """Background task: run inference for each question and compare with answer."""
    import pandas as pd

    db = SessionLocal()
    try:
        rec = db.query(BatchTestRecord).get(batch_id)
        if rec is None:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file_path, sheet_name=0)

        if "question" not in df.columns:
            rec.status = "failed"
            rec.results = [{"error": "Missing 'question' column"}]
            db.commit()
            return

        has_answer = "answer" in df.columns
        rec.total = len(df)
        rec.status = "running"
        db.commit()

        results = []
        for idx, row in df.iterrows():
            q = str(row["question"]).strip()
            expected = str(row["answer"]).strip() if has_answer else ""

            try:
                infer_result = infer_svc.run_single_inference(q, knowledge_dirs)
                model_answer = infer_result.get("final_answer", "")
            except Exception as e:
                model_answer = f"Error: {e}"

            results.append({
                "index": int(idx),
                "question": q,
                "expected_answer": expected,
                "model_answer": model_answer,
                "match": "",
            })

            rec.completed = len(results)
            rec.results = results
            db.commit()

        rec.status = "completed"
        db.commit()
    except Exception as e:
        try:
            rec = db.query(BatchTestRecord).get(batch_id)
            if rec:
                rec.status = "failed"
                rec.results = [{"error": str(e)}]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
        try:
            os.unlink(file_path)
        except Exception:
            pass


@router.post("/batch-test", response_model=BatchTestResponse)
def create_batch_test(
    file: UploadFile = File(...),
    knowledge_dirs: str = Form(""),
    db: DBSession = Depends(get_db),
):
    kd = json.loads(knowledge_dirs) if knowledge_dirs else list(KNOWLEDGE_DIRS)

    suffix = os.path.splitext(file.filename or "test.csv")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()

    rec = BatchTestRecord(knowledge_dirs=kd, status="pending")
    db.add(rec)
    db.commit()
    db.refresh(rec)

    t = threading.Thread(target=_run_batch, args=(rec.id, tmp.name, kd), daemon=True)
    t.start()

    return BatchTestResponse(
        id=rec.id, status=rec.status, total=0, completed=0, results=[],
    )


@router.get("/batch-test/{batch_id}", response_model=BatchTestResponse)
def get_batch_test(batch_id: str, db: DBSession = Depends(get_db)):
    rec = db.query(BatchTestRecord).get(batch_id)
    if rec is None:
        raise HTTPException(404, "Batch test not found")
    return BatchTestResponse(
        id=rec.id,
        status=rec.status or "pending",
        total=rec.total or 0,
        completed=rec.completed or 0,
        results=rec.results or [],
    )
