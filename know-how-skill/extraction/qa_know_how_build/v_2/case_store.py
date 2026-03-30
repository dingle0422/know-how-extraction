"""
案例库管理：通用案例库（Level 1 空 Know-How）+ 边缘案例库（Level 2 不匹配样本）。
支持增量写入和线程安全。
"""

import json
import os
from threading import Lock

_store_lock = Lock()


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── 通用案例库 ──────────────────────────────────────────────────────────────

def save_general_cases(cases: list[dict], output_path: str, source_file: str = ""):
    """
    将 Level 1 提炼为空的样本写入通用案例库。

    Parameters
    ----------
    cases : 列表，每项含 index, question, answer 等原始字段。
    output_path : 输出 JSON 路径。
    source_file : 源数据文件名（用于标注来源）。
    """
    data = {
        "source_file": source_file,
        "total_cases": len(cases),
        "cases": cases,
    }
    with _store_lock:
        _save_json(output_path, data)
    print(f"[CaseStore] 通用案例库已写入 {len(cases)} 条: {output_path}")


# ─── 边缘案例库 ──────────────────────────────────────────────────────────────

def append_edge_cases(cluster_key: str, edge_cases: list[dict],
                      output_path: str):
    """
    向边缘案例库追加某个簇的不匹配样本。

    通过 cluster_key 与 level2_refinement.json 中的同名 key 建立索引关联。

    Parameters
    ----------
    cluster_key : 簇标识（如 "cluster_0"），对应 level2 输出中的 key。
    edge_cases : 边缘案例列表，每项含 index, input, inference_result, mismatch_reason。
    output_path : 输出 JSON 路径。
    """
    with _store_lock:
        data = _load_json(output_path)
        data[cluster_key] = {
            "edge_cases": edge_cases,
        }
        _save_json(output_path, data)
    if edge_cases:
        print(f"[CaseStore] 边缘案例库 {cluster_key}: 新增 {len(edge_cases)} 条")


def load_edge_cases(input_path: str) -> dict:
    """加载边缘案例库。"""
    return _load_json(input_path)


def get_edge_case_summary(input_path: str) -> list[dict]:
    """获取边缘案例库的摘要统计。"""
    data = _load_json(input_path)
    summary = []
    for cluster_key, cluster_data in data.items():
        cases = cluster_data.get("edge_cases", [])
        summary.append({
            "cluster": cluster_key,
            "edge_case_count": len(cases),
        })
    return summary
