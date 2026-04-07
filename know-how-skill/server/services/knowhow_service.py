"""
Know-How service: load, patch, diff structured know-how blocks.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any, Callable

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (
    _SKILL_ROOT,
    os.path.join(_SKILL_ROOT, "extraction"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_2"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _get_llm():
    from llm_client import chat
    return chat


def load_knowhow_entry(knowledge_dir: str, entry_key: str) -> dict[str, Any]:
    """Load a structured know-how dict from knowledge.json."""
    kj = os.path.join(knowledge_dir, "knowledge.json")
    with open(kj, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(entry_key, {})
    return entry.get("know_how", entry.get("Final_Know_How", {}))


def render_knowhow_text(kh: dict) -> str:
    """Render structured know-how to readable text (for diff display)."""
    if not isinstance(kh, dict):
        return str(kh)

    lines: list[str] = []
    if kh.get("title"):
        lines.append(f"【{kh['title']}】")
    if kh.get("scope"):
        lines.append(f"适用场景: {kh['scope']}")
    lines.append("")

    for s in kh.get("steps", []):
        step_id = s.get("step", "?")
        depth = step_id.count(".") if isinstance(step_id, str) else 0
        indent = "  " * depth
        line = f"{indent}{step_id}. {s.get('action', '')}"
        if s.get("condition"):
            line += f"  [条件: {s['condition']}]"
        if s.get("constraint"):
            line += f"  【约束: {s['constraint']}】"
        if s.get("policy_basis"):
            line += f"  【依据: {s['policy_basis']}】"
        if s.get("outcome"):
            line += f"  → {s['outcome']}"
        lines.append(line)

    if kh.get("exceptions"):
        lines.append("")
        lines.append("例外情况:")
        for ex in kh["exceptions"]:
            lines.append(f"  - 当 {ex.get('when', '?')} → {ex.get('then', '?')}")

    return "\n".join(lines)


def generate_patch(
    kh_original: dict,
    question: str,
    expert_correction: dict[str, str],
    llm_func: Callable | None = None,
) -> dict[str, Any]:
    """Generate patch operations using the existing kh_minimal_update prompt."""
    from prompts_v2 import kh_minimal_update
    from prompts import safe_parse_json_with_llm_repair

    llm = llm_func or _get_llm()

    error_type = expert_correction.get("error_type", "other")
    corrected_answer = expert_correction.get("corrected_answer", "")
    corrected_reasoning = expert_correction.get("corrected_reasoning", "")

    mismatch = (
        f"错误类型: {error_type}\n"
        f"专家修正的正确答案: {corrected_answer}\n"
        f"专家修正的正确思维链: {corrected_reasoning}"
    )

    prompt = kh_minimal_update(
        know_how_json=json.dumps(kh_original, ensure_ascii=False, indent=2),
        question=question,
        answer=corrected_answer,
        mismatch_analysis=mismatch,
    )

    for attempt in range(3):
        try:
            response = llm(prompt)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            result = safe_parse_json_with_llm_repair(content, llm_func=llm)
            return result
        except Exception:
            if attempt == 2:
                return {"operations": [], "diff_description": "Patch generation failed"}
            time.sleep(2)

    return {"operations": [], "diff_description": ""}


def apply_correction(
    knowledge_dir: str,
    entry_key: str,
    question: str,
    expert_correction: dict[str, str],
) -> dict[str, Any]:
    """Full flow: load original → generate patch → apply patch → return diff."""
    from patch_engine import apply_patch

    kh_original = load_knowhow_entry(knowledge_dir, entry_key)
    if not kh_original or not isinstance(kh_original, dict):
        return {
            "entry_key": entry_key,
            "title": "",
            "original": kh_original,
            "patched": kh_original,
            "original_text": str(kh_original),
            "patched_text": str(kh_original),
            "operations": [],
            "patch_log": [],
            "diff_description": "Cannot patch non-structured know-how",
        }

    patch_result = generate_patch(kh_original, question, expert_correction)
    operations = patch_result.get("operations", [])

    if operations:
        patched_kh, patch_log = apply_patch(kh_original, operations)
    else:
        patched_kh = copy.deepcopy(kh_original)
        patch_log = []

    return {
        "entry_key": entry_key,
        "title": kh_original.get("title", ""),
        "original": kh_original,
        "patched": patched_kh,
        "original_text": render_knowhow_text(kh_original),
        "patched_text": render_knowhow_text(patched_kh),
        "operations": operations,
        "patch_log": patch_log,
        "diff_description": patch_result.get("diff_description", ""),
    }


def apply_corrections_batch(
    activated_knowhow: list[dict],
    selected_keys: list[str],
    question: str,
    expert_correction: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Apply corrections to multiple selected know-how blocks."""
    results: dict[str, dict[str, Any]] = {}

    kd_map: dict[str, str] = {}
    for item in activated_knowhow:
        kd_map[item["entry_key"]] = item.get("knowledge_dir", "")

    for key in selected_keys:
        knowledge_dir = kd_map.get(key, "")
        if not knowledge_dir:
            continue
        results[key] = apply_correction(knowledge_dir, key, question, expert_correction)

    return results


# ── Per-block operations ─────────────────────────────────────────────────────

def ai_update_single(
    knowledge_dir: str,
    entry_key: str,
    question: str,
    expert_correction: dict[str, str],
) -> dict[str, Any]:
    """AI-generate patch for a single know-how block. Returns the patch result
    so the user can review/edit before confirming."""
    return apply_correction(knowledge_dir, entry_key, question, expert_correction)


def manual_update_single(
    knowledge_dir: str,
    entry_key: str,
    patched_json: dict,
) -> dict[str, Any]:
    """Create a patch result from user-provided edited know-how."""
    kh_original = load_knowhow_entry(knowledge_dir, entry_key)
    return {
        "entry_key": entry_key,
        "title": kh_original.get("title", "") if isinstance(kh_original, dict) else "",
        "original": kh_original,
        "patched": patched_json,
        "original_text": render_knowhow_text(kh_original),
        "patched_text": render_knowhow_text(patched_json),
        "operations": [{"op": "manual_edit", "description": "专家手动编辑"}],
        "patch_log": [],
        "diff_description": "专家手动修改",
    }


def generate_new_knowhow(
    question: str,
    expert_correction: dict[str, str],
    llm_func: Callable | None = None,
) -> dict[str, Any]:
    """AI-generate a brand-new know-how block from the question + correction."""
    from prompts_v2 import structured_kh_generate
    from prompts import safe_parse_json_with_llm_repair

    llm = llm_func or _get_llm()

    corrected_answer = expert_correction.get("corrected_answer", "")
    corrected_reasoning = expert_correction.get("corrected_reasoning", "")

    free_text = (
        f"根据专家修正生成新知识块。\n"
        f"问题: {question}\n"
        f"修正答案: {corrected_answer}\n"
        f"修正思维链: {corrected_reasoning}"
    )

    prompt = structured_kh_generate(
        know_how_text=free_text,
        question=question,
        answer=corrected_answer,
        extra_info="",
        reasoning=corrected_reasoning,
    )

    for attempt in range(3):
        try:
            response = llm(prompt)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            kh_json = safe_parse_json_with_llm_repair(content, llm_func=llm)
            return {
                "knowhow_json": kh_json,
                "knowhow_text": render_knowhow_text(kh_json),
            }
        except Exception:
            if attempt == 2:
                return {
                    "knowhow_json": {
                        "title": "",
                        "scope": "",
                        "steps": [],
                        "exceptions": [],
                    },
                    "knowhow_text": "",
                }
            time.sleep(2)

    return {"knowhow_json": {}, "knowhow_text": ""}


def add_new_knowhow_entry(
    knowledge_dir: str,
    knowhow_json: dict,
) -> dict[str, Any]:
    """Prepare a patch result for a new know-how block (not yet persisted).

    Returns a dict keyed by the new entry_key that can be merged into session.patches.
    """
    kj = os.path.join(knowledge_dir, "knowledge.json")
    existing_keys: list[str] = []
    if os.path.exists(kj):
        with open(kj, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing_keys = list(data.keys())

    max_key = 0
    for k in existing_keys:
        try:
            max_key = max(max_key, int(k))
        except ValueError:
            pass
    new_key = str(max_key + 1)

    empty_original: dict = {}
    return {
        "entry_key": new_key,
        "knowledge_dir": knowledge_dir,
        "title": knowhow_json.get("title", ""),
        "original": empty_original,
        "patched": knowhow_json,
        "original_text": "",
        "patched_text": render_knowhow_text(knowhow_json),
        "operations": [{"op": "add_new_entry", "description": "新增知识块"}],
        "patch_log": [],
        "diff_description": "新增知识块",
        "is_new": True,
    }
