"""
Server configuration.
"""

import os

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_DIRS: list[str] = []

_default_kd = os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "knowledge")
if os.path.isdir(_default_kd):
    for name in os.listdir(_default_kd):
        full = os.path.join(_default_kd, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "knowledge.json")):
            KNOWLEDGE_DIRS.append(full)

_doc_kd = os.path.join(_SKILL_ROOT, "extraction", "doc_know_how_build", "knowledge")
if os.path.isdir(_doc_kd):
    for name in os.listdir(_doc_kd):
        full = os.path.join(_doc_kd, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "knowledge.json")):
            KNOWLEDGE_DIRS.append(full)

DATABASE_URL = f"sqlite:///{os.path.join(_SKILL_ROOT, 'server', 'incremental.db')}"

SKILL_ROOT = _SKILL_ROOT
