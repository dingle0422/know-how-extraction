"""
Inference service: wraps existing mapreduce_infer pipeline for single-question use.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (
    _SKILL_ROOT,
    os.path.join(_SKILL_ROOT, "extraction"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_2"),
    os.path.join(_SKILL_ROOT, "extraction", "qa_know_how_build", "v_1"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _get_llm_func() -> Callable:
    from llm_client import chat
    return chat


def _get_embedding_func() -> Callable | None:
    try:
        from utils import get_embeddings
        return get_embeddings
    except Exception:
        return None


def run_single_inference(
    question: str,
    knowledge_dirs: list[str],
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
) -> dict[str, Any]:
    """Run the full MapReduce inference pipeline for a single question.

    Returns the raw result dict from the pipeline, which includes
    valid_results, final_answer, map_results, etc.
    """
    if not knowledge_dirs:
        print("[Infer] 警告: knowledge_dirs 为空，跳过推理流程")
        return {
            "q_idx": 0,
            "question": question,
            "final_answer": "",
            "valid_results": [],
            "map_results": [],
            "synthesis_analysis": "",
            "no_knowledge_dirs": True,
        }

    from inference.mapreduce_infer import run_mapreduce_inference
    from inference.retrieval import build_retrievers
    from inference.prompts_infer import infer_v1, summary_v0

    llm = _get_llm_func()
    emb_func = _get_embedding_func()

    retrievers = build_retrievers(knowledge_dirs)

    results = run_mapreduce_inference(
        knowledge_dirs=knowledge_dirs,
        questions=[{"q_idx": 0, "question": question}],
        map_llm_func=llm,
        reduce_llm_func=llm,
        infer_prompt_func=infer_v1,
        summary_prompt_func=summary_v0,
        embedding_func=emb_func,
        tfidf_top_n=tfidf_top_n,
        embedding_top_n=embedding_top_n,
        map_max_workers=4,
        enable_edge_cases=True,
        enable_qa_direct=True,
        pre_built_retrievers=retrievers,
        reduce_batch_size=3,
    )

    if results:
        return results[0]
    return {
        "q_idx": 0,
        "question": question,
        "final_answer": "",
        "valid_results": [],
        "map_results": [],
        "synthesis_analysis": "",
    }


def extract_activated_knowhow(inference_result: dict) -> list[dict[str, Any]]:
    """Extract the list of activated (valid) know-how blocks from inference result."""
    items = []
    seen_keys = set()

    for r in inference_result.get("valid_results", []):
        if r.get("is_llm_bare"):
            continue
        entry_key = r.get("entry_key", "")
        source_dir = r.get("source_dir", "")
        uid = f"{source_dir}:{entry_key}"
        if uid in seen_keys:
            continue
        seen_keys.add(uid)

        kh_text = r.get("kh_text", "")
        knowledge_dir = r.get("knowledge_dir", "")

        title, scope = _parse_title_scope(kh_text, knowledge_dir, entry_key)

        items.append({
            "entry_key": entry_key,
            "source_dir": source_dir,
            "knowledge_dir": knowledge_dir,
            "title": title,
            "scope": scope,
            "kh_text": kh_text,
            "reasoning_chain": r.get("Reasoning_Chain", ""),
            "derived_answer": r.get("Derived_Answer", ""),
        })

    return items


def _parse_title_scope(kh_text: str, knowledge_dir: str, entry_key: str) -> tuple[str, str]:
    """Try to parse title/scope from knowledge.json entry."""
    if not knowledge_dir:
        first_line = kh_text.split("\n", 1)[0] if kh_text else ""
        return first_line[:60], ""

    kj_path = os.path.join(knowledge_dir, "knowledge.json")
    if not os.path.exists(kj_path):
        return kh_text.split("\n", 1)[0][:60] if kh_text else "", ""

    try:
        with open(kj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(entry_key, {})
        kh = entry.get("know_how", {})
        if isinstance(kh, dict):
            return kh.get("title", ""), kh.get("scope", "")
    except Exception:
        pass
    return kh_text.split("\n", 1)[0][:60] if kh_text else "", ""
