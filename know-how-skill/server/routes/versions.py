"""
Knowledge version management API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import VersionResponse
from ..services import version_service as ver_svc

router = APIRouter(tags=["versions"])


@router.get("/versions", response_model=list[VersionResponse])
def list_versions(
    knowledge_dir: str = Query(default=None),
    db: DBSession = Depends(get_db),
):
    versions = ver_svc.list_versions(db, knowledge_dir)
    return [
        VersionResponse(
            id=v.id,
            knowledge_dir=v.knowledge_dir,
            description=v.description or "",
            session_id=v.session_id or "",
            created_at=v.created_at.isoformat() if v.created_at else "",
        )
        for v in versions
    ]


@router.post("/versions/{version_id}/restore")
def restore_version(version_id: int, db: DBSession = Depends(get_db)):
    try:
        kd = ver_svc.restore_version(db, version_id)
        return {"status": "ok", "knowledge_dir": kd}
    except ValueError as e:
        raise HTTPException(404, str(e))
