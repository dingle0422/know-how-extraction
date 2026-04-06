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
from utils import sanitize_for_json

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
        json.dump(sanitize_for_json(data_dict), f, ensure_ascii=False, indent=2)


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

_MAX_EDGE_RECURSE_DEPTH = 5


def _generate_single_edge_kh(sample: dict, llm_func, max_retries: int) -> dict:
    """为单个 irrelevant 样本独立生成一个新的结构化 Know-How（仅当子簇仅 1 个样本时使用）。"""
    s_inp = sample.get("input", {})
    prompt = structured_kh_generate(
        know_how_text=sample.get("Know_How", ""),
        question=s_inp.get("question", ""),
        answer=s_inp.get("answer", ""),
        extra_info=s_inp.get("Extra_Information", ""),
        reasoning=s_inp.get("reasoning", ""),
    )
    edge_kh = _llm_call_with_retry(llm_func, prompt, parse_json=True,
                                    max_retries=max_retries)
    ci = sample["index"]
    for _step in edge_kh.get("steps", []):
        _oc = _step.get("outcome")
        _step["outcome"] = append_qa_footnote(
            "" if _oc is None else str(_oc), ci
        )
    for _exc in edge_kh.get("exceptions", []):
        if "then" in _exc:
            _exc["then"] = append_qa_footnote(_exc["then"], ci)
    edge_kh["source_qa_ids"] = [ci]
    edge_kh["edge_qa_ids"] = []
    return edge_kh


def _refine_edge_samples(
    edge_samples: list[dict],
    llm_func,
    max_retries_per_step: int,
    cluster_key: str,
    depth: int = 1,
    embedding_func=None,
    cosine_threshold: float = 0.75,
    tfidf_weight: float = 1.0,
    embedding_weight: float = 0.0,
) -> list[dict]:
    """
    对边缘样本进行递归聚类 → 质心生成 → 验证/patch，返回所有平级的 know-how 列表。

    与主流程共用同一套逻辑：
    1. edge_samples 聚类（复用 make_clusters）
    2. 每个子簇走 _refine_sub_cluster 逻辑（质心 → 验证 → patch）
    3. 子簇内新产生的 irrelevant 样本递归处理（depth+1）
    4. 递归深度达到 _MAX_EDGE_RECURSE_DEPTH 时，剩余 edge 各自独立生成 KH
    """
    from clustering import make_clusters

    tag = f"{cluster_key}/edge-L{depth}"
    if len(edge_samples) == 0:
        return []

    if len(edge_samples) == 1:
        print(f"    [{tag}] 仅 1 个边缘样本，直接生成独立 KH")
        return [_generate_single_edge_kh(edge_samples[0], llm_func, max_retries_per_step)]

    if depth > _MAX_EDGE_RECURSE_DEPTH:
        print(f"    [{tag}] 达到最大递归深度 {_MAX_EDGE_RECURSE_DEPTH}，"
              f"{len(edge_samples)} 个边缘样本各自独立生成 KH")
        return [_generate_single_edge_kh(s, llm_func, max_retries_per_step)
                for s in edge_samples]

    print(f"    [{tag}] {len(edge_samples)} 个边缘样本，开始二次聚类...")
    sub_clusters = make_clusters(
        edge_samples,
        cosine_threshold=cosine_threshold,
        embedding_func=embedding_func,
        tfidf_weight=tfidf_weight,
        embedding_weight=embedding_weight,
    )
    print(f"    [{tag}] 二次聚类完成: {len(sub_clusters)} 个子簇")

    all_khs: list[dict] = []

    for sc_idx, sc in enumerate(sub_clusters):
        sc_tag = f"{tag}/sub{sc_idx}"
        sc_centroid = sc["centroid_item"]
        sc_others = sc["sorted_others"]

        kh, sub_edge_samples, sub_log = _refine_sub_cluster(
            sc_centroid, sc_others, llm_func, max_retries_per_step, sc_tag,
        )

        if kh.get("steps"):
            try:
                norm_prompt = kh_normalize_steps(
                    json.dumps(kh, ensure_ascii=False, indent=2)
                )
                norm_result = _llm_call_with_retry(
                    llm_func, norm_prompt, parse_json=True,
                    max_retries=max_retries_per_step,
                )
                new_steps = norm_result.get("steps") if isinstance(norm_result, dict) else None
                if new_steps and isinstance(new_steps, list):
                    kh["steps"] = new_steps
            except Exception:
                pass

        all_khs.append(kh)

        if sub_edge_samples:
            child_khs = _refine_edge_samples(
                sub_edge_samples, llm_func, max_retries_per_step,
                cluster_key, depth=depth + 1,
                embedding_func=embedding_func,
                cosine_threshold=cosine_threshold,
                tfidf_weight=tfidf_weight,
                embedding_weight=embedding_weight,
            )
            all_khs.extend(child_khs)

    return all_khs


def _refine_sub_cluster(
    centroid: dict,
    sorted_others: list[dict],
    llm_func,
    max_retries: int,
    tag: str,
) -> tuple[dict, list[dict], list[dict]]:
    """
    对单个子簇执行质心生成 → 验证 → patch 流程（与主流程相同逻辑，但不写文件）。

    Returns
    -------
    (know_how, edge_samples, refinement_log)
    - know_how: 该子簇生成的结构化 KH
    - edge_samples: 无法被吸收的边缘样本列表（用于递归）
    - refinement_log: 精炼日志
    """
    inp = centroid.get("input", {})

    print(f"      [{tag}] 质心 index={centroid['index']}，开始结构化生成...")
    prompt = structured_kh_generate(
        know_how_text=centroid["Know_How"],
        question=inp.get("question", ""),
        answer=inp.get("answer", ""),
        extra_info=inp.get("Extra_Information", ""),
        reasoning=inp.get("reasoning", ""),
    )
    know_how = _llm_call_with_retry(llm_func, prompt, parse_json=True,
                                     max_retries=max_retries)

    _ci = centroid["index"]
    for _step in know_how.get("steps", []):
        _oc = _step.get("outcome")
        _step["outcome"] = append_qa_footnote(
            "" if _oc is None else str(_oc), _ci
        )
    for _exc in know_how.get("exceptions", []):
        if "then" in _exc:
            _exc["then"] = append_qa_footnote(_exc["then"], _ci)

    know_how["source_qa_ids"] = [_ci]
    know_how["edge_qa_ids"] = []

    refinement_log = [{
        "index": _ci, "role": "centroid",
        "match_level": "centroid",
        "action": "generated structured know-how",
    }]

    edge_samples: list[dict] = []

    for seq, sample in enumerate(sorted_others, 1):
        s_inp = sample.get("input", {})
        s_q = s_inp.get("question", "")
        s_a = s_inp.get("answer", "")
        s_ei = s_inp.get("Extra_Information", "")
        s_r = s_inp.get("reasoning", "")
        s_idx = sample["index"]

        validate_prompt = kh_inference_validate(
            know_how_json=json.dumps(know_how, ensure_ascii=False, indent=2),
            question=s_q, answer=s_a, extra_info=s_ei, reasoning=s_r,
        )
        validation = _llm_call_with_retry(llm_func, validate_prompt,
                                           parse_json=True, max_retries=max_retries)

        match_level = validation.get("match_level", "irrelevant").strip().lower()
        mismatch_analysis = validation.get("mismatch_analysis", "")

        if match_level == "answerable":
            know_how["source_qa_ids"].append(s_idx)
            refinement_log.append({
                "index": s_idx, "role": "validated",
                "match_level": "answerable", "action": "skip",
            })
            print(f"      [{tag}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: answerable ✓")

        elif match_level == "augmentable":
            update_prompt = kh_minimal_update(
                know_how_json=json.dumps(know_how, ensure_ascii=False, indent=2),
                question=s_q, answer=s_a,
                mismatch_analysis=mismatch_analysis, extra_info=s_ei,
            )
            update_result = _llm_call_with_retry(llm_func, update_prompt,
                                                  parse_json=True, max_retries=max_retries)

            ops = update_result.get("operations", [])
            diff_desc = update_result.get("diff_description", "")

            if ops:
                know_how, patch_log = apply_patch(know_how, ops, qa_index=s_idx)
                applied = [p for p in patch_log if p["status"] == "applied"]
            else:
                applied = []

            if applied:
                know_how["source_qa_ids"].append(s_idx)
                refinement_log.append({
                    "index": s_idx, "role": "validated",
                    "match_level": "augmentable",
                    "action": f"patched ({len(applied)} ops): {diff_desc}",
                })
                print(f"      [{tag}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: augmentable → patched {len(applied)} ops")
            else:
                know_how["edge_qa_ids"].append(s_idx)
                edge_samples.append(sample)
                refinement_log.append({
                    "index": s_idx, "role": "validated",
                    "match_level": "augmentable",
                    "action": "patch_failed → edge",
                })
                print(f"      [{tag}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: augmentable → patch 全失败 → edge")

        else:
            know_how["edge_qa_ids"].append(s_idx)
            edge_samples.append(sample)
            refinement_log.append({
                "index": s_idx, "role": "validated",
                "match_level": "irrelevant", "action": "edge",
            })
            print(f"      [{tag}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: irrelevant → edge")

    return know_how, edge_samples, refinement_log


def _refine_single_cluster(
    cluster_idx: int,
    cluster: dict,
    total_clusters: int,
    llm_func,
    output_file: str,
    edge_cases_file: str,
    max_retries_per_step: int = 5,
    embedding_func=None,
    cosine_threshold: float = 0.75,
    tfidf_weight: float = 1.0,
    embedding_weight: float = 0.0,
):
    """
    对单个簇执行增量精炼：
    1. 质心样本 → 生成结构化 Know-How
    2. 其余样本按 cosine 降序逐个验证（三档：answerable / augmentable / irrelevant）
    3. answerable → 挂钩 source_qa_ids
       augmentable → patch 补充后挂钩 source_qa_ids（patch 全失败则降级 irrelevant）
       irrelevant → 挂钩 edge_qa_ids
    4. 边缘样本递归聚类 → 质心 → 验证/patch，生成与主 KH 平级的 know-how
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
        _oc = _step.get("outcome")
        _step["outcome"] = append_qa_footnote(
            "" if _oc is None else str(_oc), _ci
        )
    for _exc in know_how.get("exceptions", []):
        if "then" in _exc:
            _exc["then"] = append_qa_footnote(_exc["then"], _ci)

    know_how["source_qa_ids"] = [centroid["index"]]
    know_how["edge_qa_ids"] = []

    refinement_log = [{
        "index": centroid["index"],
        "role": "centroid",
        "match_level": "centroid",
        "action": "generated structured know-how",
    }]

    # ── Step 2: 逐样本验证 ────────────────────────────────────────────────
    edge_cases = []
    edge_samples_for_recurse: list[dict] = []

    for seq, sample in enumerate(sorted_others, 1):
        s_inp = sample.get("input", {})
        s_q = s_inp.get("question", "")
        s_a = s_inp.get("answer", "")
        s_ei = s_inp.get("Extra_Information", "")
        s_r = s_inp.get("reasoning", "")
        s_idx = sample["index"]
        s_kh_text = sample.get("Know_How", "")

        validate_prompt = kh_inference_validate(
            know_how_json=json.dumps(know_how, ensure_ascii=False, indent=2),
            question=s_q,
            answer=s_a,
            extra_info=s_ei,
            reasoning=s_r,
        )
        validation = _llm_call_with_retry(llm_func, validate_prompt, parse_json=True,
                                           max_retries=max_retries_per_step)

        match_level = validation.get("match_level", "irrelevant").strip().lower()
        derived_answer = validation.get("derived_answer", "")
        mismatch_analysis = validation.get("mismatch_analysis", "")

        if match_level == "answerable":
            know_how["source_qa_ids"].append(s_idx)
            refinement_log.append({
                "index": s_idx,
                "role": "validated",
                "match_level": "answerable",
                "action": "skip",
            })
            print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: answerable ✓")

        elif match_level == "augmentable":
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
                          f"{[sk['detail'] for sk in skipped]}")
            else:
                patch_log = []
                applied = []

            if applied:
                know_how["source_qa_ids"].append(s_idx)
                refinement_log.append({
                    "index": s_idx,
                    "role": "validated",
                    "match_level": "augmentable",
                    "action": f"patched ({len(applied)} ops): {diff_desc}",
                    "patch_log": patch_log,
                })
                print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: augmentable → patched {len(applied)} ops "
                      f"({diff_desc[:60]})")
            else:
                know_how["edge_qa_ids"].append(s_idx)
                edge_samples_for_recurse.append(sample)
                edge_cases.append({
                    "index": s_idx,
                    "input": {
                        "question": s_q, "answer": s_a,
                        "extra_info": s_ei, "know_how": s_kh_text,
                    },
                    "inference_result": derived_answer,
                    "mismatch_reason": mismatch_analysis,
                    "patch_failure": True,
                    "patch_log": patch_log,
                })
                refinement_log.append({
                    "index": s_idx,
                    "role": "validated",
                    "match_level": "augmentable",
                    "action": f"patch_failed (0/{len(ops)} ops) → edge",
                    "patch_log": patch_log,
                })
                print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                      f"index={s_idx}: augmentable → patch 全失败 → 降级 edge")

        else:  # irrelevant
            know_how["edge_qa_ids"].append(s_idx)
            edge_samples_for_recurse.append(sample)
            edge_cases.append({
                "index": s_idx,
                "input": {
                    "question": s_q, "answer": s_a,
                    "extra_info": s_ei, "know_how": s_kh_text,
                },
                "inference_result": derived_answer,
                "mismatch_reason": mismatch_analysis,
            })
            refinement_log.append({
                "index": s_idx,
                "role": "validated",
                "match_level": "irrelevant",
                "action": "edge_case",
            })
            print(f"    [{cluster_key}] sample {seq}/{len(sorted_others)} "
                  f"index={s_idx}: irrelevant → edge")

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

    # ── Step 4: 边缘样本递归聚类 → 生成平级 KH ─────────────────────────────
    edge_know_hows: list[dict] = []
    if edge_samples_for_recurse:
        print(f"  [{cluster_key}] {len(edge_samples_for_recurse)} 个边缘样本，"
              f"进入递归聚类...")
        edge_know_hows = _refine_edge_samples(
            edge_samples_for_recurse, llm_func, max_retries_per_step,
            cluster_key, depth=1,
            embedding_func=embedding_func,
            cosine_threshold=cosine_threshold,
            tfidf_weight=tfidf_weight,
            embedding_weight=embedding_weight,
        )
        print(f"  [{cluster_key}] 边缘递归完成: 生成 {len(edge_know_hows)} 个平级 KH")

    # ── Step 5: 写入结果 ──────────────────────────────────────────────────
    kh_title = know_how.get("title", cluster_key)

    result = {
        "cluster_index": cluster_idx,
        "know_how": know_how,
        "centroid_index": centroid["index"],
        "absorbed_indices": list(know_how["source_qa_ids"]),
        "edge_case_indices": list(know_how["edge_qa_ids"]),
        "edge_know_hows": edge_know_hows,
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
    absorbed = len(know_how["source_qa_ids"])
    print(f"  [{cluster_key}] 完成: {absorbed}/{total_in_cluster} 样本被吸收, "
          f"{len(edge_cases)} 边缘案例 → 递归生成 {len(edge_know_hows)} 个平级KH, "
          f"title=\"{kh_title}\"")

    return cluster_idx, "success", result


def _refine_single_cluster_safe(cluster_idx, cluster, total_clusters,
                                 llm_func, output_file, edge_cases_file,
                                 max_retries_per_step,
                                 embedding_func=None,
                                 cosine_threshold=0.75,
                                 tfidf_weight=1.0,
                                 embedding_weight=0.0):
    """带顶层异常捕获的簇精炼，确保不会崩溃整个线程池。"""
    try:
        return _refine_single_cluster(
            cluster_idx, cluster, total_clusters,
            llm_func, output_file, edge_cases_file, max_retries_per_step,
            embedding_func=embedding_func,
            cosine_threshold=cosine_threshold,
            tfidf_weight=tfidf_weight,
            embedding_weight=embedding_weight,
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
                embedding_func=embedding_func,
                cosine_threshold=cosine_threshold,
                tfidf_weight=tfidf_weight,
                embedding_weight=embedding_weight,
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
