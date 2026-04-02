"""
V2 二级提炼：质心驱动的增量精炼。
对每个聚类簇：质心样本生成结构化 Know-How → 逐样本推理验证 → 最小改动更新 / 边缘案例归档。
支持多线程并发（簇间并行）+ 断点续传。
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

for _p in (_V_DIR, _PACKAGE_DIR, _SKILL_ROOT, _EXTRACTION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompts import safe_parse_json_with_llm_repair
from prompts_v2 import structured_kh_generate, kh_inference_validate, kh_minimal_update, kh_normalize_steps
from patch_engine import apply_patch, append_qa_footnote
from case_store import append_edge_cases

_file_lock = Lock()


def _update_json_file(file_path: str, key: str, value: dict):
    data_dict = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
        except (json.JSONDecodeError, IOError):
            data_dict = {}
    data_dict[key] = value
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)


# ─── Level 1 结果加载（V2 版：保留完整 input 数据）──────────────────────────

def load_level1_results_full(level1_file: str) -> tuple[list[dict], list[dict]]:
    """
    加载一级提炼结果，分流为有效条目和空条目。

    Returns
    -------
    valid_items : Know_How 非空的成功条目（含 index, Know_How, input）
    empty_items : Know_How 为空或状态非 success 的条目（用于通用案例库）
    """
    with open(level1_file, "r", encoding="utf-8") as f:
        kh_data = json.load(f)

    valid_items = []
    empty_items = []

    for v in kh_data.values():
        item = {
            "index": v.get("index", -1),
            "Know_How": v.get("Know_How", ""),
            "input": v.get("input", {}),
        }
        if v.get("status") == "success" and v.get("Know_How", "").strip():
            valid_items.append(item)
        else:
            empty_items.append({
                "index": item["index"],
                "question": item["input"].get("question", ""),
                "answer": item["input"].get("answer", ""),
                "extra_info": item["input"].get("Extra_Information", ""),
                "reasoning": item["input"].get("reasoning", ""),
                "reason": "Level 1 未提炼出可泛化的 Know-How",
            })

    valid_items.sort(key=lambda x: x["index"])
    print(f"[Level-2] 有效 Know_How: {len(valid_items)}, 空/失败: {len(empty_items)}")
    return valid_items, empty_items


# ─── 单步 LLM 调用（带重试）────────────────────────────────────────────────

def _llm_call_with_retry(llm_func, prompt: str, parse_json: bool = True,
                         max_retries: int = 5) -> dict:
    """带重试的 LLM 调用。parse_json=True 时自动解析 JSON 响应。"""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm_func(prompt)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            if not parse_json:
                return {"content": content}
            return safe_parse_json_with_llm_repair(content, llm_func=llm_func)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2)
            if attempt % 3 == 0:
                print(f"  [LLM] 第 {attempt}/{max_retries} 次失败: {str(e)[:120]}")
    raise Exception(f"LLM 调用 {max_retries} 次均失败。最后错误: {last_err}")


# ─── 单簇增量精炼 ──────────────────────────────────────────────────────────

def _refine_single_cluster(
    cluster_idx: int,
    cluster: dict,
    total_clusters: int,
    llm_func,
    output_file: str,
    edge_cases_file: str,
    max_retries_per_step: int = 5,
):
    """
    对单个簇执行增量精炼：
    1. 质心样本 → 生成结构化 Know-How
    2. 其余样本按 cosine 降序逐个验证
    3. full → 跳过, partial → 最小更新, none → 边缘案例
    """
    centroid = cluster["centroid_item"]
    sorted_others = cluster["sorted_others"]
    cluster_key = f"cluster_{cluster_idx}"

    inp = centroid.get("input", {})
    centroid_q = inp.get("question", "")
    centroid_a = inp.get("answer", "")
    centroid_ei = inp.get("Extra_Information", "")
    centroid_r = inp.get("reasoning", "")

    # ── Step 1: 质心样本 → 结构化 Know-How ────────────────────────────────
    print(f"  [{cluster_key}] 质心 index={centroid['index']}，开始结构化生成...")
    prompt = structured_kh_generate(
        know_how_text=centroid["Know_How"],
        question=centroid_q,
        answer=centroid_a,
        extra_info=centroid_ei,
        reasoning=centroid_r,
    )
    know_how = _llm_call_with_retry(llm_func, prompt, parse_json=True,
                                     max_retries=max_retries_per_step)

    _ci = centroid["index"]
    for _step in know_how.get("steps", []):
        if "action" in _step:
            _step["action"] = append_qa_footnote(_step["action"], _ci)
    for _exc in know_how.get("exceptions", []):
        if "then" in _exc:
            _exc["then"] = append_qa_footnote(_exc["then"], _ci)
    for _j, _con in enumerate(know_how.get("constraints", [])):
        know_how["constraints"][_j] = append_qa_footnote(_con, _ci)

    refinement_log = [{
        "index": centroid["index"],
        "role": "centroid",
        "match_level": "centroid",
        "action": "generated structured know-how",
    }]

    # ── Step 2: 逐样本验证 ────────────────────────────────────────────────
    edge_cases = []

    for seq, sample in enumerate(sorted_others, 1):
        s_inp = sample.get("input", {})
        s_q = s_inp.get("question", "")
        s_a = s_inp.get("answer", "")
        s_ei = s_inp.get("Extra_Information", "")
        s_idx = sample["index"]
        s_kh_text = sample.get("Know_How", "")

        # 推理验证
        validate_prompt = kh_inference_validate(
            know_how_json=json.dumps(know_how, ensure_ascii=False, indent=2),
            question=s_q,
            answer=s_a,
            extra_info=s_ei,
        )
        validation = _llm_call_with_retry(llm_func, validate_prompt, parse_json=True,
                                           max_retries=max_retries_per_step)

        match_level = validation.get("match_level", "none").strip().lower()
        derived_answer = validation.get("derived_answer", "")
        mismatch_analysis = validation.get("mismatch_analysis", "")

        if match_level == "full":
            refinement_log.append({
                "index": s_idx,
                "role": "validated",
                "match_level": "full",
                "action": "skip",
            })
            print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: full match ✓")

        elif match_level == "partial":
            update_prompt = kh_minimal_update(
                know_how_json=json.dumps(know_how, ensure_ascii=False, indent=2),
                question=s_q,
                answer=s_a,
                mismatch_analysis=mismatch_analysis,
                extra_info=s_ei,
            )
            update_result = _llm_call_with_retry(llm_func, update_prompt,
                                                  parse_json=True,
                                                  max_retries=max_retries_per_step)

            ops = update_result.get("operations", [])
            diff_desc = update_result.get("diff_description", "")

            if ops:
                know_how, patch_log = apply_patch(know_how, ops, qa_index=s_idx)
                applied = [p for p in patch_log if p["status"] == "applied"]
                skipped = [p for p in patch_log if p["status"] == "skipped"]
                if skipped:
                    print(f"    [{cluster_key}]   ⚠ {len(skipped)} 个操作被跳过: "
                          f"{[s['detail'] for s in skipped]}")
            else:
                patch_log = []
                applied = []

            if not applied:
                edge_cases.append({
                    "index": s_idx,
                    "input": {
                        "question": s_q,
                        "answer": s_a,
                        "extra_info": s_ei,
                        "know_how": s_kh_text,
                    },
                    "inference_result": derived_answer,
                    "mismatch_reason": mismatch_analysis,
                    "patch_failure": True,
                    "patch_log": patch_log,
                })
                refinement_log.append({
                    "index": s_idx,
                    "role": "validated",
                    "match_level": "partial",
                    "action": f"patch_failed (0/{len(ops)} ops applied) → edge_case",
                    "patch_log": patch_log,
                })
                print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: partial → patch 全部失败 → 转入边缘案例")
            else:
                refinement_log.append({
                    "index": s_idx,
                    "role": "validated",
                    "match_level": "partial",
                    "action": f"patched ({len(applied)} ops): {diff_desc}",
                    "patch_log": patch_log,
                })
                print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: partial → patched {len(applied)} ops "
                      f"({diff_desc[:60]})")

        else:
            edge_cases.append({
                "index": s_idx,
                "input": {
                    "question": s_q,
                    "answer": s_a,
                    "extra_info": s_ei,
                    "know_how": s_kh_text,
                },
                "inference_result": derived_answer,
                "mismatch_reason": mismatch_analysis,
            })
            refinement_log.append({
                "index": s_idx,
                "role": "validated",
                "match_level": "none",
                "action": "edge_case",
            })
            print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: none → edge case")

    # ── Step 3: LLM 步骤编号归一化 ──────────────────────────────────────────
    if know_how.get("steps"):
        try:
            norm_prompt = kh_normalize_steps(
                json.dumps(know_how, ensure_ascii=False, indent=2)
            )
            norm_result = _llm_call_with_retry(
                llm_func, norm_prompt, parse_json=True,
                max_retries=max_retries_per_step,
            )
            new_steps = norm_result.get("steps") if isinstance(norm_result, dict) else None
            if new_steps and isinstance(new_steps, list) and len(new_steps) == len(know_how["steps"]):
                know_how["steps"] = new_steps
                print(f"  [{cluster_key}] 步骤编号归一化完成 ({len(new_steps)} steps)")
            elif new_steps and isinstance(new_steps, list):
                know_how["steps"] = new_steps
                print(f"  [{cluster_key}] 步骤编号归一化完成 "
                      f"(steps 数量变化: {len(know_how.get('steps', []))} → {len(new_steps)}，"
                      f"可能发生了合并)")
            else:
                print(f"  [{cluster_key}] 步骤归一化返回格式异常，保留原始编号")
        except Exception as e:
            print(f"  [{cluster_key}] 步骤编号归一化失败，保留原始编号: {str(e)[:120]}")

    # ── Step 4: 写入结果 ──────────────────────────────────────────────────
    kh_title = know_how.get("title", cluster_key)

    result = {
        "cluster_index": cluster_idx,
        "know_how": know_how,
        "centroid_index": centroid["index"],
        "absorbed_indices": [centroid["index"]] + [
            log["index"] for log in refinement_log
            if log.get("match_level") in ("full", "partial")
            and log.get("role") != "centroid"
        ],
        "edge_case_indices": [ec["index"] for ec in edge_cases],
        "cluster_keywords": cluster.get("keywords", []),
        "cluster_cohesion": cluster.get("cohesion"),
        "group_label": cluster.get("group_label", ""),
        "refinement_log": refinement_log,
        "status": "success",
    }

    with _file_lock:
        _update_json_file(output_file, str(cluster_idx), result)

    if edge_cases:
        append_edge_cases(cluster_key, edge_cases, edge_cases_file)

    total_in_cluster = 1 + len(sorted_others)
    absorbed = len(result["absorbed_indices"])
    print(f"  [{cluster_key}] 完成: {absorbed}/{total_in_cluster} 样本被吸收, "
          f"{len(edge_cases)} 边缘案例, title=\"{kh_title}\"")

    return cluster_idx, "success", result


def _refine_single_cluster_safe(cluster_idx, cluster, total_clusters,
                                 llm_func, output_file, edge_cases_file,
                                 max_retries_per_step):
    """带顶层异常捕获的簇精炼，确保不会崩溃整个线程池。"""
    try:
        return _refine_single_cluster(
            cluster_idx, cluster, total_clusters,
            llm_func, output_file, edge_cases_file, max_retries_per_step,
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"  [cluster_{cluster_idx}] 致命错误: {error_msg[:300]}")
        error_result = {
            "cluster_index": cluster_idx,
            "status": "failed",
            "error": str(e),
            "traceback": error_msg[:1000],
        }
        with _file_lock:
            _update_json_file(output_file, str(cluster_idx), error_result)
        return cluster_idx, "failed", None


# ─── 多线程入口 ──────────────────────────────────────────────────────────────

def _group_items_by_extra_info(items: list[dict]) -> dict[str, list[dict]]:
    """按 Extra_Information 标签组合将样本分组。返回 {标签: 样本列表} 有序字典。"""
    groups: dict[str, list[dict]] = {}
    for item in items:
        label = item.get("input", {}).get("Extra_Information", "").strip()
        if not label:
            label = "__未分类__"
        groups.setdefault(label, []).append(item)
    return groups


def run_level2_refinement(
    level1_file: str,
    llm_func,
    output_file: str = "./output/kh_level2_refinement.json",
    edge_cases_file: str = "./output/edge_cases.json",
    general_cases_file: str = "./output/general_cases.json",
    cosine_threshold: float = 0.75,
    max_workers: int = 4,
    max_retries_per_step: int = 5,
    source_file: str = "",
    embedding_func=None,
    tfidf_weight: float = 1.0,
    embedding_weight: float = 0.0,
    max_cluster_samples: int = 0,
    group_by_extra: bool = True,
):
    """
    V2 二级知识精炼入口。

    Parameters
    ----------
    level1_file : 一级提炼结果 JSON 路径
    llm_func : LLM 调用函数
    output_file : 二级精炼结果 JSON 输出路径
    edge_cases_file : 边缘案例库 JSON 输出路径
    general_cases_file : 通用案例库 JSON 输出路径
    cosine_threshold : 聚类 cosine 相似度阈值
    max_workers : 簇间并行线程数
    max_retries_per_step : 每个 LLM 调用步骤的最大重试次数
    source_file : 源数据文件名（用于案例库标注）
    embedding_func : Dense embedding 函数，为 None 时回退纯 TF-IDF
    tfidf_weight : TF-IDF 相似度权重，设为 0 跳过 TF-IDF
    embedding_weight : Dense Embedding 相似度权重，设为 0 跳过 Embedding
    max_cluster_samples : 每个簇的最大样本数，超出部分拆分为新簇。0 表示不限制。
    group_by_extra : 是否按 Extra_Information 标签组合预分组后再聚类（默认开启）。
                     为 True 时，先按 Extra_Information 值将样本分组，
                     然后在每个分组内独立执行聚类，最终合并所有簇。
                     若所有样本 Extra_Information 为空，自动退化为单组。
    """
    from clustering import make_clusters
    from case_store import save_general_cases

    # ── 加载与分流 ────────────────────────────────────────────────────────
    valid_items, empty_items = load_level1_results_full(level1_file)

    if empty_items:
        os.makedirs(os.path.dirname(general_cases_file) or ".", exist_ok=True)
        save_general_cases(empty_items, general_cases_file, source_file=source_file)

    if not valid_items:
        print("[Level-2] 无有效 Know_How 可供精炼，流程结束")
        return output_file

    # ── 聚类（可选：先按 Extra_Information 预分组）─────────────────────────
    if group_by_extra:
        groups = _group_items_by_extra_info(valid_items)
        print(f"[Level-2] 按 Extra_Information 预分组: "
              f"{len(groups)} 个标签组合，共 {len(valid_items)} 条样本")
        for label, group_items in groups.items():
            print(f"  ├── [{label}]: {len(group_items)} 条")

        clusters = []
        for label, group_items in groups.items():
            if not group_items:
                continue
            group_clusters = make_clusters(
                group_items,
                cosine_threshold=cosine_threshold,
                embedding_func=embedding_func,
                tfidf_weight=tfidf_weight,
                embedding_weight=embedding_weight,
                max_cluster_samples=max_cluster_samples,
            )
            for c in group_clusters:
                c["group_label"] = label
            clusters.extend(group_clusters)
        print(f"[Level-2] 预分组聚类完成: {len(groups)} 个分组 → {len(clusters)} 个簇")
    else:
        clusters = make_clusters(
            valid_items,
            cosine_threshold=cosine_threshold,
            embedding_func=embedding_func,
            tfidf_weight=tfidf_weight,
            embedding_weight=embedding_weight,
            max_cluster_samples=max_cluster_samples,
        )

    # ── 断点续传检查 ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 个簇记录，自动续传")
        except Exception:
            pass

    pending = [
        (i, c)
        for i, c in enumerate(clusters)
        if str(i) not in existing_data
        or existing_data.get(str(i), {}).get("status") != "success"
    ]
    completed = len(clusters) - len(pending)
    print(f"  总簇数: {len(clusters)}，已完成: {completed}，"
          f"待处理: {len(pending)}，并发数: {max_workers}")

    # ── 多线程精炼 ────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _refine_single_cluster_safe,
                i, c, len(clusters), llm_func,
                output_file, edge_cases_file, max_retries_per_step,
            ): i
            for i, c in pending
        }
        for future in as_completed(future_to_idx):
            cidx = future_to_idx[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    pct = completed / len(clusters) * 100
                    print(f"  进度: {completed}/{len(clusters)} ({pct:.1f}%)")
            except Exception as e:
                print(f"  簇 {cidx} 处理异常: {e}")

    print(f"[Level-2] 全部完成！结果保存于: {output_file}")
    return output_file
