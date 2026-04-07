"""
Version management: snapshot knowledge.json, restore, rebuild index.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any

from sqlalchemy.orm import Session as DBSession

from ..db_models import KnowledgeVersion

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (
    _SKILL_ROOT,
    os.path.join(_SKILL_ROOT, "extraction"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_2"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def save_version(
    db: DBSession,
    knowledge_dir: str,
    description: str = "",
    session_id: str = "",
) -> KnowledgeVersion:
    """Snapshot current knowledge.json into the database."""
    kj = os.path.join(knowledge_dir, "knowledge.json")
    with open(kj, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    ver = KnowledgeVersion(
        knowledge_dir=knowledge_dir,
        snapshot=snapshot,
        description=description,
        session_id=session_id,
    )
    db.add(ver)
    db.commit()
    db.refresh(ver)
    return ver


def restore_version(db: DBSession, version_id: int) -> str:
    """Restore knowledge.json from a snapshot and rebuild index."""
    ver = db.query(KnowledgeVersion).get(version_id)
    if ver is None:
        raise ValueError(f"Version {version_id} not found")

    kj = os.path.join(ver.knowledge_dir, "knowledge.json")
    with open(kj, "w", encoding="utf-8") as f:
        json.dump(ver.snapshot, f, ensure_ascii=False, indent=2)

    _rebuild_index(ver.knowledge_dir)
    return ver.knowledge_dir


def list_versions(db: DBSession, knowledge_dir: str | None = None) -> list[KnowledgeVersion]:
    q = db.query(KnowledgeVersion)
    if knowledge_dir:
        q = q.filter(KnowledgeVersion.knowledge_dir == knowledge_dir)
    return q.order_by(KnowledgeVersion.created_at.desc()).all()


def persist_patches(
    knowledge_dir: str,
    patches: dict[str, dict],
) -> None:
    """Write patched know-how back to knowledge.json (updates + new entries)."""
    kj = os.path.join(knowledge_dir, "knowledge.json")
    with open(kj, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry_key, patch_info in patches.items():
        patched = patch_info.get("patched")
        if not patched:
            continue

        if patch_info.get("is_new"):
            data[entry_key] = {
                "know_how": patched,
                "source_qa_ids": [],
                "edge_qa_ids": [],
            }
        elif entry_key in data:
            data[entry_key]["know_how"] = patched
            if "retrieval_keywords" in data[entry_key]:
                del data[entry_key]["retrieval_keywords"]

    with open(kj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    _rebuild_index(knowledge_dir)


def create_temp_knowledge(knowledge_dir: str, patches: dict[str, dict]) -> str:
    """Create a temporary copy of knowledge dir with patches applied.

    Returns the temp directory path. Caller should clean up when done.
    """
    temp_dir = tempfile.mkdtemp(prefix="kh_test_")
    temp_kd = os.path.join(temp_dir, os.path.basename(knowledge_dir))
    shutil.copytree(knowledge_dir, temp_kd)

    kj = os.path.join(temp_kd, "knowledge.json")
    with open(kj, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry_key, patch_info in patches.items():
        patched = patch_info.get("patched")
        if not patched:
            continue

        if patch_info.get("is_new"):
            data[entry_key] = {
                "know_how": patched,
                "source_qa_ids": [],
                "edge_qa_ids": [],
            }
        elif entry_key in data:
            data[entry_key]["know_how"] = patched
            if "retrieval_keywords" in data[entry_key]:
                del data[entry_key]["retrieval_keywords"]

    with open(kj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    _rebuild_index(temp_kd)
    return temp_kd


def _rebuild_index(knowledge_dir: str):
    """Rebuild retrieval_index.json for the given knowledge directory."""
    try:
        from utils import build_retrieval_index as _build
        from llm_client import chat

        kj = os.path.join(knowledge_dir, "knowledge.json")
        if os.path.exists(kj):
            emb_func = None
            try:
                _utils_path = os.path.join(_SKILL_ROOT, "utils.py")
                if os.path.exists(_utils_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("_skill_utils", _utils_path)
                    m = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(m)
                    emb_func = m.get_embeddings
            except Exception:
                pass

            _build(
                knowledge_json_path=kj,
                knowledge_dir=knowledge_dir,
                embedding_func=emb_func,
                llm_func=chat,
            )
    except Exception as e:
        print(f"[VersionService] Index rebuild failed: {e}")
