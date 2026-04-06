"""
MapReduce 推理模块（v2）
======================
基于检索增强推理架构，完整实现四阶段流水线:
  Phase 1: 双路并行检索（TF-IDF + Dense Embedding）
  Phase 2: 并行 Map 推理（含有效性校验）
  Phase 3: 边缘案例兜底（仅 QA Know-How）
  Phase 4: 分层 Reduce 融合推理（流式批次 + 递归归并）
"""

import json
import os
import concurrent.futures
import threading
from typing import Callable
from .prompts_infer import reduce_batch_v0, reduce_final_v0
from tqdm import tqdm

import time

from .retrieval import (
    build_retrievers,
    retrieve_candidates,
    load_knowledge_content,
    load_edge_cases,
    load_edge_know_hows,
    render_edge_know_how,
    load_level1_knowhow_map,
    build_qa_to_cluster_map,
    retrieve_edge_cases,
    format_edge_cases_text,
    QADirectRetriever,
    retrieve_qa_direct_candidates,
)


def _clean_json_string(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ─── Phase 2: Map 推理 ───────────────────────────────────────────────────────

def _run_map_task(
    task_input: dict,
    llm_func: Callable,
    infer_prompt_func: Callable,
    pitfalls_func: Callable = None,
) -> dict:
    """对单个候选知识块执行 Map 推理（有效性校验 + 候选答案生成）。"""
    question = task_input["question"]
    kh_text = task_input["kh_text"]

    try:
        if pitfalls_func is not None:
            pp = pitfalls_func()
            prompt = infer_prompt_func(question, kh_text, pp)
        else:
            prompt = infer_prompt_func(question, kh_text)

        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "Match_Status": "NO",
            "Rejection_Reason": f"模型调用或JSON解析失败: {e}",
            "Reasoning_Chain": "",
            "Derived_Answer": "",
        }

    result["q_idx"] = task_input["q_idx"]
    result["source_dir"] = task_input["source_dir"]
    result["entry_key"] = task_input["entry_key"]
    result["knowledge_type"] = task_input["knowledge_type"]
    result["kh_source_id"] = f"{task_input['source_dir']}:{task_input['entry_key']}"
    result["kh_text"] = task_input["kh_text"]
    return result


# ─── Phase 2b: QA 直检并行 Map 推理 ──────────────────────────────────────────

def _run_qa_direct_map_task(
    task_input: dict,
    llm_func: Callable,
    qa_direct_prompt_func: Callable,
    pitfalls_func: Callable = None,
) -> dict:
    """对单条 QA 直检命中的原始 QA 对执行 Map 推理。"""
    question = task_input["question"]
    qa_text = task_input["qa_text"]

    try:
        pp = pitfalls_func() if pitfalls_func is not None else ""
        prompt = qa_direct_prompt_func(question, qa_text, pp)
        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "Match_Status": "NO",
            "Rejection_Reason": f"QA 直检推理失败: {e}",
            "Reasoning_Chain": "",
            "Derived_Answer": "",
        }

    result["q_idx"] = task_input["q_idx"]
    result["source_dir"] = task_input["source_dir"]
    result["qa_index"] = task_input["qa_index"]
    result["knowledge_type"] = "qa_direct"
    result["kh_source_id"] = f"{task_input['source_dir']}:qa_{task_input['qa_index']}"
    result["is_qa_direct"] = True
    result["qa_text"] = task_input["qa_text"]
    result["qa_question"] = task_input.get("qa_question", "")
    result["qa_answer"] = task_input.get("qa_answer", "")
    result["qa_know_how"] = task_input.get("qa_know_how", "")
    return result


# ─── Phase 2c: LLM 裸考并行 Map 推理 ──────────────────────────────────────────

def _run_llm_bare_map_task(
    task_input: dict,
    llm_func: Callable,
    vendor: str = "volc",
    model: str = "deepseek-v3.2",
) -> dict:
    """直接将原问题交给 LLM，用模型自身知识生成候选答案（与知识块 Map 并行）。"""
    question = task_input["question"]
    prompt = (
        f"作为一个资深行业顾问，请结合你的专业知识，"
        f"直接且详尽地解答以下问题：\n【用户问题】：{question}"
    )
    try:
        raw_answer = llm_func(prompt, vendor=vendor, model=model)["content"]
        result = {
            "Match_Status": "YES",
            "Rejection_Reason": "",
            "Reasoning_Chain": "基于大模型自身专业知识的直接推理。",
            "Derived_Answer": raw_answer,
        }
    except Exception as e:
        result = {
            "Match_Status": "NO",
            "Rejection_Reason": f"LLM裸考推理失败: {e}",
            "Reasoning_Chain": "",
            "Derived_Answer": "",
        }

    result["q_idx"] = task_input["q_idx"]
    result["source_dir"] = "llm_internal"
    result["entry_key"] = "bare_llm"
    result["knowledge_type"] = "llm_bare"
    result["kh_source_id"] = "llm_bare_reasoning"
    result["is_llm_bare"] = True
    return result


# ─── Phase 3: 边缘案例兜底（基于 edge_know_hows 结构化 KH 推理）────────────

def _run_edge_kh_infer(
    task_input: dict,
    llm_func: Callable,
    infer_prompt_func: Callable,
) -> dict:
    """对单个 edge KH 执行推理，流程与 Phase 2 的 Map 推理完全一致。"""
    question = task_input["question"]
    kh_text = task_input["kh_text"]
    pitfalls = task_input.get("pitfalls", "")

    _base = {
        "q_idx": task_input["q_idx"],
        "source_dir": task_input["source_dir"],
        "entry_key": task_input["entry_key"],
        "knowledge_type": "qa_v2",
        "kh_source_id": task_input["kh_source_id"],
        "is_edge_case_fallback": True,
        "edge_kh_index": task_input.get("edge_kh_index"),
        "edge_kh_title": task_input.get("edge_kh_title", ""),
        "edge_kh_source_qa_ids": task_input.get("edge_kh_source_qa_ids", []),
    }

    try:
        prompt = infer_prompt_func(question, kh_text, pitfalls)
        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "Match_Status": "NO",
            "Rejection_Reason": f"Edge KH 推理失败: {e}",
            "Reasoning_Chain": "",
            "Derived_Answer": "",
        }

    result.update(_base)
    return result


def _run_edge_case_fallback_batch(
    task_input: dict,
    llm_func: Callable,
    infer_prompt_func: Callable,
    pitfalls_text: str = "",
    max_workers: int = 4,
) -> list[dict]:
    """对指定 cluster 的所有 edge_know_hows 并行执行推理。

    edge_know_hows 是扁平列表，包含所有递归层级产出的 KH。
    每个 edge KH 之间无依赖，全部并行推理。
    """
    knowledge_dir = task_input["knowledge_dir"]
    entry_key = task_input["entry_key"]

    edge_khs = load_edge_know_hows(knowledge_dir, entry_key)
    if not edge_khs:
        return []

    sub_tasks = []
    for i, ekh in enumerate(edge_khs):
        kh_text = render_edge_know_how(ekh)
        if not kh_text.strip():
            continue
        sub_tasks.append({
            "q_idx": task_input["q_idx"],
            "question": task_input["question"],
            "source_dir": task_input["source_dir"],
            "entry_key": entry_key,
            "kh_source_id": f"{task_input['source_dir']}:{entry_key}:edge_{i}",
            "kh_text": kh_text,
            "pitfalls": pitfalls_text,
            "edge_kh_index": i,
            "edge_kh_title": ekh.get("title", ""),
            "edge_kh_source_qa_ids": ekh.get("source_qa_ids", []),
        })

    if not sub_tasks:
        return []

    if len(sub_tasks) == 1:
        return [_run_edge_kh_infer(sub_tasks[0], llm_func, infer_prompt_func)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(sub_tasks))) as executor:
        futures = {
            executor.submit(_run_edge_kh_infer, st, llm_func, infer_prompt_func): st
            for st in sub_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                st = futures[future]
                results.append({
                    "q_idx": st["q_idx"],
                    "source_dir": st["source_dir"],
                    "entry_key": st["entry_key"],
                    "knowledge_type": "qa_v2",
                    "kh_source_id": st["kh_source_id"],
                    "is_edge_case_fallback": True,
                    "edge_kh_index": st.get("edge_kh_index"),
                    "Match_Status": "NO",
                    "Rejection_Reason": f"Edge KH 并行推理异常: {e}",
                    "Reasoning_Chain": "",
                    "Derived_Answer": "",
                })
    return results


# ─── 分层 Reduce 辅助函数 ──────────────────────────────────────────────────

def _format_reduce_item(item: dict) -> str:
    """将一个推理结果（原始 Map 结果或中间 Reduce 结果）格式化为文本片段。"""
    if "conclusion" in item:
        text = f"结论: {item['conclusion']}\n推理: {item['reasoning']}"
        if item.get("dissenting_view"):
            text += f"\n少数派观点: {item['dissenting_view']}"
        return text
    source_tag = "知识块"
    if item.get("is_llm_bare"):
        source_tag = "LLM裸考"
    elif item.get("is_qa_direct"):
        source_tag = "QA直检"
    elif item.get("is_edge_case_fallback"):
        source_tag = "边缘案例兜底"
    return (
        f"来源: {source_tag} [{item.get('kh_source_id', '?')}]\n"
        f"推导逻辑: {item.get('Reasoning_Chain', '')}\n"
        f"候选答案: {item.get('Derived_Answer', '')}"
    )


def _format_items_for_reduce(items: list) -> str:
    """将多个推理结果格式化为 Reduce prompt 输入文本。"""
    fragments = []
    for i, item in enumerate(items, 1):
        fragments.append(f"--- 候选 {i} ---\n{_format_reduce_item(item)}")
    return "\n\n".join(fragments)


def _call_batch_reduce(
    question: str,
    answers_text: str,
    llm_func: Callable,
) -> dict:
    """调用中间层批次 Reduce（reduce_batch_v0）。"""
    prompt = reduce_batch_v0(question, answers_text)
    try:
        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "conclusion": f"批次Reduce解析失败: {e}",
            "reasoning": "",
            "vote_distribution": "",
            "dissenting_view": "",
        }
    return result


def _call_final_reduce(
    question: str,
    answers_text: str,
    result_count: int,
    llm_func: Callable,
) -> dict:
    """调用最终层 Reduce（reduce_final_v0）。"""
    prompt = reduce_final_v0(question, answers_text, result_count)
    try:
        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "Synthesis_Analysis": f"最终Reduce解析失败: {e}",
            "Final_Answer": "",
        }
    return result


def _drain_pending_reduce(
    pending_reduce: list,
    reduce_batch_size: int,
    question: str,
    llm_func: Callable,
    reduce_executor,
    layer1_futures: list,
    _log: Callable,
):
    """将 pending_reduce 中达到水位线的部分提交为 Layer 1 Reduce 任务。"""
    while len(pending_reduce) >= reduce_batch_size:
        batch = pending_reduce[:reduce_batch_size]
        del pending_reduce[:reduce_batch_size]
        answers_text = _format_items_for_reduce(batch)
        fut = reduce_executor.submit(
            _call_batch_reduce, question, answers_text, llm_func,
        )
        layer1_futures.append(fut)
        _log("Reduce-L1",
             f"提交批次 ({reduce_batch_size} 条) -> Layer1 Reduce #{len(layer1_futures)}")


def _run_hierarchical_reduce(
    items: list,
    question: str,
    llm_func: Callable,
    batch_size: int = 3,
    _log: Callable = None,
) -> dict:
    """Layer 2+ 递归归并 + 最终 Reduce。

    接收 Layer 1 Reduce 中间结论 + 剩余未凑满批次的原始有效结果，
    递归分批 Reduce 直到结果数 <= batch_size，然后执行最终 Reduce。
    """
    if _log is None:
        _log = lambda phase, msg: None

    if not items:
        return {
            "Synthesis_Analysis": "无有效推理结果",
            "Final_Answer": "",
        }

    if len(items) == 1:
        item = items[0]
        if "conclusion" in item:
            return {
                "Synthesis_Analysis": f"仅一条候选结论（投票: {item.get('vote_distribution', '?')}）",
                "Final_Answer": item.get("conclusion", "") + "\n" + item.get("reasoning", ""),
            }
        return {
            "Synthesis_Analysis": "仅一条有效推理结果",
            "Final_Answer": item.get("Derived_Answer", ""),
        }

    if len(items) <= batch_size:
        answers_text = _format_items_for_reduce(items)
        _log("Reduce", f"最终层 Reduce: {len(items)} 条结果 -> Final Reduce")
        return _call_final_reduce(question, answers_text, len(items), llm_func)

    layer = 2
    while len(items) > batch_size:
        next_layer = []
        i = 0
        batch_count = 0
        reduce_futures = []

        workers = min(4, max(2, len(items) // batch_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            while i + batch_size <= len(items):
                batch = items[i:i + batch_size]
                answers_text = _format_items_for_reduce(batch)
                reduce_futures.append(
                    executor.submit(_call_batch_reduce, question, answers_text, llm_func)
                )
                batch_count += 1
                i += batch_size

            remainder = items[i:]

            for fut in concurrent.futures.as_completed(reduce_futures):
                try:
                    next_layer.append(fut.result())
                except Exception as e:
                    next_layer.append({
                        "conclusion": f"Layer {layer} Reduce 异常: {e}",
                        "reasoning": "",
                        "vote_distribution": "",
                        "dissenting_view": "",
                    })

        next_layer.extend(remainder)
        _log("Reduce", f"Layer {layer}: {len(items)} -> {len(next_layer)} 条（{batch_count} 个批次）")
        items = next_layer
        layer += 1

    answers_text = _format_items_for_reduce(items)
    _log("Reduce", f"最终层 Reduce: {len(items)} 条结果 -> Final Reduce")
    return _call_final_reduce(question, answers_text, len(items), llm_func)


# ─── Phase 4: Reduce 融合 ───────────────────────────────────────────────────

def _run_reduce_task(
    task_input: dict,
    llm_func: Callable,
    summary_prompt_func: Callable,
) -> dict:
    """[DEPRECATED] 旧版单层 Reduce，已被分层 Reduce 替代。保留仅为向后兼容。"""
    q_idx = task_input["q_idx"]
    question = task_input["question"]
    valid_results = task_input["valid_results"]

    if not valid_results:
        candidates_text = "【空】所有候选知识块均被判定为不相关，未匹配到有效 Know-How。"
    else:
        fragments = []
        for r in valid_results:
            if r.get("is_llm_bare"):
                source_tag = "LLM裸考"
            elif r.get("is_qa_direct"):
                source_tag = "QA直检"
            elif r.get("is_edge_case_fallback"):
                source_tag = "边缘案例兜底"
            else:
                source_tag = "知识块"
            fragments.append(
                f"--- 来自{source_tag} [{r.get('kh_source_id')}] 的推理 ---\n"
                f"推导逻辑: {r.get('Reasoning_Chain')}\n"
                f"候选答案: {r.get('Derived_Answer')}"
            )
        candidates_text = "\n\n".join(fragments)

    prompt = summary_prompt_func(question, "", candidates_text)

    try:
        raw = llm_func(prompt)["content"]
        reduce_result = json.loads(_clean_json_string(raw))
    except Exception as e:
        reduce_result = {
            "Synthesis_Analysis": f"解析失败: {e}",
            "Final_Answer": raw if "raw" in dir() else "处理失败",
        }

    reduce_result["q_idx"] = q_idx
    return reduce_result


# ─── 完整 4 阶段流水线入口 ──────────────────────────────────────────────────

def run_mapreduce_inference(
    knowledge_dirs: list[str],
    questions: list[dict],
    map_llm_func: Callable,
    reduce_llm_func: Callable,
    infer_prompt_func: Callable,
    summary_prompt_func: Callable,
    edge_case_prompt_func: Callable = None,
    qa_direct_prompt_func: Callable = None,
    pitfalls_func: Callable = None,
    extra_llm_func: Callable = None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
    force_extra_llm: bool = False,
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    question_max_workers: int = 1,
    enable_edge_cases: bool = True,
    enable_qa_direct: bool = True,
    pre_built_retrievers=None,
    on_question_done: Callable = None,
    reduce_batch_size: int = 3,
) -> list[dict]:
    """
    完整 MapReduce 推理流水线（含 QA 直检并行路径）。

    Parameters
    ----------
    knowledge_dirs           : 指定参与推理的 knowledge 目录列表
    questions                : 问题列表，每项为 {"q_idx": int, "question": str, ...}
    map_llm_func             : Map 阶段 LLM 函数
    reduce_llm_func          : Reduce 阶段 LLM 函数
    infer_prompt_func        : Map prompt 构造函数
    summary_prompt_func      : Reduce prompt 构造函数
    edge_case_prompt_func    : （已废弃）Phase 3 现在直接用 infer_prompt_func 对 edge_know_hows 推理，此参数保留兼容
    qa_direct_prompt_func    : QA 直检 Map prompt 构造函数
    pitfalls_func            : 陷阱提示函数，可选
    extra_llm_func           : 额外信息 LLM 函数（裸考），始终参与并行推理
    force_extra_llm          : 是否强制将 LLM 裸考结果纳入 Reduce 融合。
                               False（默认）= 仅在无其他有效匹配时才使用裸考结果；
                               True = 始终将裸考结果作为候选纳入 Reduce。
    embedding_func           : Dense embedding 函数，可选
    tfidf_top_n              : 所有双路检索共享的 TF-IDF Top-N
    embedding_top_n          : 所有双路检索共享的 Dense Embedding Top-N
    map_max_workers          : 单题内 Map/Phase3 并发线程数
    question_max_workers     : 问题级别并发数（默认 1 即串行；>1 时多题同时处理，
                               总并发 ≈ question_max_workers × map_max_workers）
    enable_edge_cases        : 是否启用 Phase 3 边缘案例兜底（默认开启）
    enable_qa_direct         : 是否启用 QA 直检并行路径（默认开启）
    pre_built_retrievers     : 预构建的 KnowledgeRetriever 列表，避免每题重复加载索引
    on_question_done         : 每道题完成后的回调 on_question_done(result)，用于增量保存
    reduce_batch_size        : 分层 Reduce 批次大小（必须为奇数，默认 3）。
                               Map 结果达到此水位线时立即提交 Layer 1 Reduce

    Returns
    -------
    推理结果列表，每项包含问题、Map 详情、Reduce 最终答案等
    """
    total_q = len(questions)
    print(f"[Infer] 收到 {total_q} 个问题, {len(knowledge_dirs)} 个知识目录, "
          f"问题级并发={question_max_workers}, Map并发={map_max_workers}")

    # ── 预构建检索器（避免每道题重复加载索引 JSON）──
    if pre_built_retrievers is not None:
        retrievers = pre_built_retrievers
        print(f"[Infer] 使用预构建检索器: {len(retrievers)} 个目录")
    else:
        print("[Infer] 首次构建检索器（建议外部预构建后传入以提速）...")
        retrievers = build_retrievers(knowledge_dirs)
        print(f"[Infer] 检索器构建完成: {len(retrievers)} 个目录")

    # ── 预加载 Level-1 Know-How 映射（Phase 3 已改用 edge_know_hows，此处保留供 QA 直检等使用）──
    level1_maps: dict[str, dict[int, str]] = {}

    # ── 预加载 QA 直检 Retriever（仅 QA 类目录）──
    qa_direct_retrievers: list[QADirectRetriever] = []
    if enable_qa_direct and qa_direct_prompt_func is not None:
        for d in knowledge_dirs:
            traceback_path = os.path.join(d, "knowledge_traceback.json")
            ktype_path = os.path.join(d, "retrieval_index.json")
            if not os.path.exists(traceback_path):
                continue
            is_qa = True
            if os.path.exists(ktype_path):
                try:
                    import json as _j
                    with open(ktype_path, "r", encoding="utf-8") as _f:
                        kt = _j.load(_f).get("knowledge_type", "")
                    is_qa = kt.startswith("qa")
                except Exception:
                    pass
            if not is_qa:
                continue
            ret = QADirectRetriever(d, embedding_func=embedding_func)
            if ret.qa_pairs:
                qa_direct_retrievers.append(ret)
        if qa_direct_retrievers:
            total_qa = sum(len(r.qa_pairs) for r in qa_direct_retrievers)
            print(f"[Infer] 已加载 QA 直检索引: {total_qa} 条原始 QA "
                  f"(来自 {len(qa_direct_retrievers)} 个目录)")

    # ── 预加载 QA index → Level-2 entry_key 反向映射（用于 QA 直检锚点关联）──
    qa_to_cluster_maps: dict[str, dict[int, str]] = {}
    if qa_direct_retrievers:
        for d in knowledge_dirs:
            m = build_qa_to_cluster_map(d)
            if m:
                qa_to_cluster_maps[d] = m
        if qa_to_cluster_maps:
            total_mapped = sum(len(v) for v in qa_to_cluster_maps.values())
            print(f"[Infer] 已加载 QA→Cluster 反向映射: {total_mapped} 条")

    print("")

    # ── 每道题的完整 4 阶段处理（封装成函数供并发调用）──
    _q_counter_lock = threading.Lock()
    _q_counter = [0]
    _callback_lock = threading.Lock()
    pbar = tqdm(total=total_q, desc="Questions")

    def _process_one_question(q_item: dict) -> dict:
        with _q_counter_lock:
            _q_counter[0] += 1
            q_num = _q_counter[0]

        q_idx = q_item["q_idx"]
        question = q_item["question"]
        q_preview = question[:40].replace("\n", " ")

        def _log(phase: str, msg: str):
            print(f"  [Q{q_num}/{total_q}][{phase}] {msg}")

        t_start = time.time()
        print(f"\n{'─'*60}")
        print(f"[Q{q_num}/{total_q}] idx={q_idx} | {q_preview}...")

        # ── Phase 1: 双路并行检索 ──────────────────────────────────────
        candidates = retrieve_candidates(
            question=question,
            tfidf_top_n=tfidf_top_n,
            embedding_top_n=embedding_top_n,
            embedding_func=embedding_func,
            pre_built_retrievers=retrievers,
        )
        _log("Phase1", f"Level-2 候选知识块: {len(candidates)} 个")

        # ── Phase 1b: QA 直检（锚点模式：仅用于关联 Level-2 知识块）────
        qa_direct_hits = []
        qa_anchored_count = 0
        if qa_direct_retrievers:
            query_emb = None
            if embedding_func is not None:
                try:
                    query_emb = embedding_func([question])[0]
                except Exception:
                    pass
            qa_direct_hits = retrieve_qa_direct_candidates(
                question, qa_direct_retrievers,
                tfidf_top_n=tfidf_top_n,
                embedding_top_n=embedding_top_n,
                query_embedding=query_emb,
            )
            if qa_direct_hits and qa_to_cluster_maps:
                existing_keys = {(c["source_dir"], c["entry_key"]) for c in candidates}
                for hit in qa_direct_hits:
                    kd = hit["knowledge_dir"]
                    cluster_map = qa_to_cluster_maps.get(kd, {})
                    entry_key = cluster_map.get(hit["qa_index"])
                    if entry_key is not None and (hit["source_dir"], entry_key) not in existing_keys:
                        candidates.append({
                            "source_dir": hit["source_dir"],
                            "knowledge_dir": kd,
                            "entry_key": entry_key,
                            "knowledge_type": "qa_v2",
                            "score": hit.get("score", 0.0),
                            "retrieval_method": "qa_anchor",
                        })
                        existing_keys.add((hit["source_dir"], entry_key))
                        qa_anchored_count += 1
        _log("Phase1", f"QA 直检命中: {len(qa_direct_hits)} 条 → 关联 Level-2 新增: {qa_anchored_count} 个")
        _log("Phase1", f"合并后候选知识块总数: {len(candidates)} 个")

        if not candidates:
            _log("Phase1", "无任何候选，跳过此题")
            pbar.update(1)
            return _build_empty_result(q_idx, question)

        # 加载知识块文本内容，构建 Map 任务
        map_tasks = []
        for cand in candidates:
            kh_text = load_knowledge_content(
                cand["knowledge_dir"], cand["entry_key"],
            )
            if not kh_text.strip():
                continue
            map_tasks.append({
                "q_idx": q_idx,
                "question": question,
                "source_dir": cand["source_dir"],
                "knowledge_dir": cand["knowledge_dir"],
                "entry_key": cand["entry_key"],
                "knowledge_type": cand["knowledge_type"],
                "kh_text": kh_text,
                "retrieval_score": cand["score"],
            })

        if not map_tasks:
            _log("Phase1", "知识块内容为空，跳过此题")
            pbar.update(1)
            return _build_empty_result(q_idx, question)

        # ── 分层 Reduce 基础设施 ──
        pending_reduce = []
        layer1_reduce_futures = []
        reduce_exec = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        all_valid = []

        # ── Phase 2: 并行 Map 推理（Level-2 知识块 + LLM 裸考 同时进行）──
        bare_enabled = extra_llm_func is not None
        total_map_tasks = len(map_tasks) + (1 if bare_enabled else 0)
        _log("Phase2", f"启动并行 Map: L2={len(map_tasks)}, "
             f"裸考={1 if bare_enabled else 0}，共 {total_map_tasks} 个任务，并发={map_max_workers}")

        map_results = []
        llm_bare_result = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=map_max_workers) as executor:
            l2_futures = {
                executor.submit(
                    _run_map_task, task, map_llm_func,
                    infer_prompt_func, pitfalls_func,
                ): ("l2", task)
                for task in map_tasks
            }
            bare_futures = {}
            if extra_llm_func is not None:
                bare_task = {"q_idx": q_idx, "question": question}
                bare_futures = {
                    executor.submit(
                        _run_llm_bare_map_task, bare_task,
                        extra_llm_func, extra_vendor, extra_model,
                    ): ("bare", bare_task)
                }
            all_futures = {**l2_futures, **bare_futures}
            done_count = 0
            for future in concurrent.futures.as_completed(all_futures):
                tag, _task = all_futures[future]
                try:
                    result = future.result()
                    status = result.get("Match_Status", "?")
                    done_count += 1
                    if tag == "l2":
                        map_results.append(result)
                        if status == "YES":
                            all_valid.append(result)
                            pending_reduce.append(result)
                            _drain_pending_reduce(
                                pending_reduce, reduce_batch_size, question,
                                reduce_llm_func, reduce_exec, layer1_reduce_futures, _log,
                            )
                        _log("Phase2", f"  L2 [{done_count}/{total_map_tasks}] key={result.get('entry_key')} → {status}")
                    else:
                        llm_bare_result = result
                        _log("Phase2", f"  裸考 [{done_count}/{total_map_tasks}] → {status}")
                except Exception as exc:
                    done_count += 1
                    print(f"  [-] Map 异常 (q_idx={q_idx}, type={tag}): {exc}")

        l2_valid = [r for r in map_results if r.get("Match_Status") == "YES"]
        rejected_results = [r for r in map_results if r.get("Match_Status") != "YES"]
        bare_valid = llm_bare_result is not None and llm_bare_result.get("Match_Status") == "YES"

        _log("Phase2", f"完成: L2 {len(l2_valid)}/{len(map_results)} 有效, "
             f"裸考 {'YES' if bare_valid else ('NO' if llm_bare_result else '未启用')}")

        # ── Phase 3: 边缘案例兜底（基于 edge_know_hows 结构化 KH）──
        edge_fallback_results = []
        if enable_edge_cases:
            qa_rejected = [
                r for r in rejected_results
                if r.get("knowledge_type") == "qa_v2"
            ]
            if qa_rejected:
                _log("Phase3", f"触发边缘案例兜底: {len(qa_rejected)} 个被拒绝的 QA 知识块")
                pp_text = pitfalls_func() if pitfalls_func is not None else ""
                edge_tasks = []
                for r in qa_rejected:
                    kd = _find_knowledge_dir(knowledge_dirs, r["source_dir"])
                    if kd is None:
                        continue
                    edge_tasks.append({
                        "q_idx": q_idx,
                        "question": question,
                        "source_dir": r["source_dir"],
                        "knowledge_dir": kd,
                        "entry_key": r["entry_key"],
                    })
                with concurrent.futures.ThreadPoolExecutor(max_workers=map_max_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_edge_case_fallback_batch, task,
                            map_llm_func, infer_prompt_func,
                            pp_text, map_max_workers,
                        ): task
                        for task in edge_tasks
                    }
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_results = future.result()
                            for ec_result in batch_results:
                                edge_fallback_results.append(ec_result)
                                if ec_result.get("Match_Status") == "YES":
                                    all_valid.append(ec_result)
                                    pending_reduce.append(ec_result)
                                    _drain_pending_reduce(
                                        pending_reduce, reduce_batch_size, question,
                                        reduce_llm_func, reduce_exec, layer1_reduce_futures, _log,
                                    )
                                _log("Phase3",
                                     f"  兜底 key={ec_result.get('entry_key')}:edge_{ec_result.get('edge_kh_index')} "
                                     f"({ec_result.get('edge_kh_title', '')[:30]}) → {ec_result.get('Match_Status')}")
                        except Exception as exc:
                            print(f"  [-] EdgeCase 异常 (q_idx={q_idx}): {exc}")

                edge_success = len([r for r in edge_fallback_results if r.get("Match_Status") == "YES"])
                _log("Phase3", f"完成: {edge_success}/{len(edge_fallback_results)} 个 edge KH 兜底成功")
            else:
                _log("Phase3", "无被拒绝的 QA 知识块，跳过兜底")
        elif not enable_edge_cases:
            _log("Phase3", "已禁用（--no-edge-cases）")

        # ── Phase 4: 分层 Reduce 融合 ─────────────────────────────────
        bare_included = False
        if bare_valid:
            if force_extra_llm:
                all_valid.append(llm_bare_result)
                pending_reduce.append(llm_bare_result)
                bare_included = True
                _log("Phase4", "LLM 裸考结果已强制纳入 Reduce（--force-extra-llm）")
            elif not all_valid:
                all_valid.append(llm_bare_result)
                pending_reduce.append(llm_bare_result)
                bare_included = True
                _log("Phase4", "无其他有效匹配，LLM 裸考结果作为兜底纳入 Reduce")
            else:
                _log("Phase4", f"已有 {len(all_valid)} 个有效匹配，LLM 裸考结果仅记录不参与 Reduce")

        # 等待 Layer 1 Reduce 完成
        layer1_results = []
        if layer1_reduce_futures:
            for fut in concurrent.futures.as_completed(layer1_reduce_futures):
                try:
                    layer1_results.append(fut.result())
                except Exception as e:
                    layer1_results.append({
                        "conclusion": f"Layer 1 Reduce 异常: {e}",
                        "reasoning": "",
                        "vote_distribution": "",
                        "dissenting_view": "",
                    })
        reduce_exec.shutdown(wait=True)

        reduce_items = layer1_results + pending_reduce
        _log("Phase4", f"分层 Reduce: L1中间结论={len(layer1_results)}, "
             f"剩余={len(pending_reduce)}, 总计={len(reduce_items)}")

        reduce_result = _run_hierarchical_reduce(
            reduce_items, question, reduce_llm_func, reduce_batch_size, _log,
        )

        llm_bare_answer = ""
        if bare_valid:
            llm_bare_answer = llm_bare_result.get("Derived_Answer", "")

        final_ans = reduce_result.get("Final_Answer", "")
        elapsed = time.time() - t_start
        _log("Phase4", f"完成 | 耗时 {elapsed:.1f}s | 最终答案: {str(final_ans)[:60]}...")
        pbar.update(1)

        return {
            "q_idx": q_idx,
            "question": question,
            "retrieval_candidates_count": len(candidates),
            "map_total_evaluated": len(map_results),
            "map_match_count": len(l2_valid),
            "qa_direct_count": len(qa_direct_hits),
            "qa_direct_match_count": qa_anchored_count,
            "edge_fallback_count": len(edge_fallback_results),
            "edge_fallback_match_count": len([
                r for r in edge_fallback_results if r.get("Match_Status") == "YES"
            ]),
            "total_valid_count": len(all_valid),
            "map_results": map_results,
            "qa_direct_results": [],
            "edge_fallback_results": edge_fallback_results,
            "valid_results": all_valid,
            "rejected_reasons": [
                f"KH[{r.get('kh_source_id')}]: {r.get('Rejection_Reason')}"
                for r in map_results if r.get("Match_Status") != "YES"
            ],
            "extra_information": llm_bare_answer,
            "synthesis_analysis": reduce_result.get("Synthesis_Analysis", ""),
            "final_answer": final_ans,
            "valid_kh_source_qa": _format_valid_kh_sources(l2_valid),
            "valid_edge_source_qa": _format_valid_edge_sources(edge_fallback_results),
            "valid_qa_direct_source_qa": "",
        }

    # ── 串行 or 问题级并发 ──────────────────────────────────────────────────
    all_output = []

    if question_max_workers <= 1:
        for q_item in questions:
            result = _process_one_question(q_item)
            all_output.append(result)
            with _callback_lock:
                if on_question_done:
                    on_question_done(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=question_max_workers) as executor:
            futures = {
                executor.submit(_process_one_question, q_item): q_item
                for q_item in questions
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_output.append(result)
                    with _callback_lock:
                        if on_question_done:
                            on_question_done(result)
                except Exception as exc:
                    q_item = futures[future]
                    print(f"[-] 问题处理异常 (q_idx={q_item.get('q_idx')}): {exc}")

    pbar.close()
    all_output.sort(key=lambda x: x["q_idx"])

    print(f"\n{'='*60}")
    print(f"[Infer] 推理完成！共处理 {len(all_output)} 个问题")
    return all_output


def _format_valid_kh_sources(l2_valid_results: list[dict]) -> str:
    """将有效 L2 知识块的源 QA + Know-How 格式化为可读文本。"""
    if not l2_valid_results:
        return ""
    parts = []
    for i, r in enumerate(l2_valid_results, 1):
        src = r.get("kh_source_id", "?")
        kh = r.get("kh_text", "")
        parts.append(f"=== 有效知识块 {i} [{src}] ===\n{kh}")
    return "\n\n".join(parts)


def _format_valid_edge_sources(edge_results: list[dict]) -> str:
    """将有效边缘案例兜底的源 QA + Know-How 格式化为可读文本。"""
    valid = [r for r in edge_results if r.get("Match_Status") == "YES"]
    if not valid:
        return ""
    parts = []
    for i, r in enumerate(valid, 1):
        src = r.get("kh_source_id", "?")
        kh_key = r.get("ec_knowhow_entry_key", "?")
        qa_indices = r.get("ec_matched_qa_indices", [])
        ec = r.get("ec_text", "")
        header = f"=== 有效边缘案例 {i} [{src}] | Know-How序号={kh_key} | QA序号={qa_indices} ==="
        parts.append(f"{header}\n{ec}")
    return "\n\n".join(parts)


def _format_valid_qa_direct_sources(qa_direct_valid_results: list[dict]) -> str:
    """将 QA 直检有效命中的源 QA + Know-How 格式化为可读文本。"""
    if not qa_direct_valid_results:
        return ""
    parts = []
    for i, r in enumerate(qa_direct_valid_results, 1):
        src = r.get("kh_source_id", "?")
        q = r.get("qa_question", "")
        a = r.get("qa_answer", "")
        kh = r.get("qa_know_how", "")
        lines = [f"=== QA直检 {i} [{src}] ==="]
        if q:
            lines.append(f"原始问题: {q}")
        if a:
            lines.append(f"原始答案: {a}")
        if kh:
            lines.append(f"关联知识 (Know-How): {kh}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _build_empty_result(q_idx: int, question: str) -> dict:
    """构建未检索到候选知识块时的空结果。"""
    return {
        "q_idx": q_idx,
        "question": question,
        "retrieval_candidates_count": 0,
        "map_total_evaluated": 0,
        "map_match_count": 0,
        "qa_direct_count": 0,
        "qa_direct_match_count": 0,
        "edge_fallback_count": 0,
        "edge_fallback_match_count": 0,
        "total_valid_count": 0,
        "map_results": [],
        "qa_direct_results": [],
        "edge_fallback_results": [],
        "valid_results": [],
        "rejected_reasons": [],
        "extra_information": "",
        "synthesis_analysis": "未检索到相关候选知识块",
        "final_answer": "",
        "valid_kh_source_qa": "",
        "valid_edge_source_qa": "",
        "valid_qa_direct_source_qa": "",
    }


def _find_knowledge_dir(knowledge_dirs: list[str], source_dir: str) -> str | None:
    """根据 source_dir 名称在 knowledge_dirs 列表中找到对应的完整路径。"""
    import os
    for d in knowledge_dirs:
        if os.path.basename(d) == source_dir:
            return d
    return None


# ─── 文件 I/O 便捷入口（支持 CSV + Excel）────────────────────────────────────

def _read_input_file(filepath: str):
    """读取输入文件（自动识别 CSV / Excel），返回 DataFrame。"""
    import pandas as pd
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"不支持的文件格式: {ext}（仅支持 .csv / .xlsx / .xls）")


def _write_output_file(df, filepath: str):
    """将 DataFrame 写入输出文件（根据扩展名自动选择 CSV / Excel）。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls"):
        df.to_excel(filepath, index=False, engine="openpyxl")
    else:
        raise ValueError(f"不支持的输出格式: {ext}")


def _append_results_to_df(df, results: list[dict]):
    """将推理结果追加到 DataFrame 的新列中。"""
    results_map = {r["q_idx"]: r for r in results}

    df["Retrieval_Candidates"] = 0
    df["Map_Total_Evaluated"] = 0
    df["Map_Match_Count"] = 0
    df["QA_Direct_Count"] = 0
    df["QA_Direct_Match"] = 0
    df["Edge_Fallback_Count"] = 0
    df["Edge_Fallback_Match"] = 0
    df["Total_Valid_Count"] = 0
    df["Map_Valid_Details"] = ""
    df["Map_Rejected_Reasons"] = ""
    df["Valid_KH_Source_QA"] = ""
    df["Valid_Edge_Source_QA"] = ""
    df["Valid_QADirect_Source_QA"] = ""
    df["Extra_Information"] = ""
    df["Reduce_Analysis"] = ""
    df["Final_Inference_Answer"] = ""

    for index in df.index:
        r = results_map.get(index)
        if r is None:
            continue
        df.at[index, "Retrieval_Candidates"] = r["retrieval_candidates_count"]
        df.at[index, "Map_Total_Evaluated"] = r["map_total_evaluated"]
        df.at[index, "Map_Match_Count"] = r["map_match_count"]
        df.at[index, "QA_Direct_Count"] = r["qa_direct_count"]
        df.at[index, "QA_Direct_Match"] = r["qa_direct_match_count"]
        df.at[index, "Edge_Fallback_Count"] = r["edge_fallback_count"]
        df.at[index, "Edge_Fallback_Match"] = r["edge_fallback_match_count"]
        df.at[index, "Total_Valid_Count"] = r["total_valid_count"]
        df.at[index, "Map_Valid_Details"] = (
            json.dumps(r["valid_results"], ensure_ascii=False)
            if r["valid_results"] else ""
        )
        df.at[index, "Map_Rejected_Reasons"] = "\n".join(r["rejected_reasons"])
        df.at[index, "Valid_KH_Source_QA"] = r.get("valid_kh_source_qa", "")
        df.at[index, "Valid_Edge_Source_QA"] = r.get("valid_edge_source_qa", "")
        df.at[index, "Valid_QADirect_Source_QA"] = r.get("valid_qa_direct_source_qa", "")
        df.at[index, "Extra_Information"] = r["extra_information"]
        df.at[index, "Reduce_Analysis"] = r["synthesis_analysis"]
        df.at[index, "Final_Inference_Answer"] = r["final_answer"]

    return df


def run_mapreduce_inference_file(
    knowledge_dirs: list[str],
    input_path: str,
    output_path: str,
    map_llm_func: Callable,
    reduce_llm_func: Callable,
    infer_prompt_func: Callable,
    summary_prompt_func: Callable,
    edge_case_prompt_func: Callable = None,
    qa_direct_prompt_func: Callable = None,
    pitfalls_func: Callable = None,
    extra_llm_func: Callable = None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
    force_extra_llm: bool = False,
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    question_max_workers: int = 1,
    enable_edge_cases: bool = True,
    enable_qa_direct: bool = True,
    question_column: str = "question",
    reduce_batch_size: int = 3,
) -> str:
    """
    文件便捷入口：读取测试集（CSV/Excel），执行推理，输出结果文件。

    在输入数据的基础上追加以下列：
      - Retrieval_Candidates  : Phase 1 检索到的候选知识块数
      - Map_Total_Evaluated   : Phase 2 实际评估的知识块数（Level-2 知识块）
      - Map_Match_Count       : Phase 2 判定有效的知识块数
      - QA_Direct_Count       : QA 直检评估数
      - QA_Direct_Match       : QA 直检命中数
      - Edge_Fallback_Count   : Phase 3 边缘案例兜底尝试数
      - Edge_Fallback_Match   : Phase 3 兜底成功数
      - Total_Valid_Count     : 最终有效推理结果总数
      - Map_Valid_Details     : 所有有效 Map 推理中间结果（JSON）
      - Map_Rejected_Reasons  : 所有被拒绝的原因
      - Extra_Information     : 额外参考信息
      - Reduce_Analysis       : Reduce 融合分析
      - Final_Inference_Answer: 最终推理答案

    Parameters
    ----------
    input_path         : 输入文件路径（.csv / .xlsx）
    output_path        : 输出文件路径（.csv / .xlsx，格式以此为准）
    question_column    : 问题列名，默认 "question"
    tfidf_top_n        : 所有双路检索共享的 TF-IDF Top-N
    embedding_top_n    : 所有双路检索共享的 Dense Embedding Top-N
    enable_edge_cases  : 是否启用 Phase 3 边缘案例兜底
    enable_qa_direct   : 是否启用 QA 直检并行路径
    其余参数与 run_mapreduce_inference 一致

    Returns
    -------
    输出文件路径
    """
    import pandas as pd

    df = _read_input_file(input_path)
    print(f"[Infer] 加载 {len(df)} 条数据 ({os.path.basename(input_path)})")

    if question_column not in df.columns:
        raise ValueError(
            f"输入文件中未找到问题列 '{question_column}'，"
            f"可用列: {list(df.columns)}"
        )

    questions = []
    for index, row in df.iterrows():
        q = str(row[question_column]).strip()
        if q:
            questions.append({"q_idx": index, "question": q})

    print(f"[Infer] 有效问题: {len(questions)} 条")

    # ── 预构建检索器（所有问题共享，避免每题重复加载索引 JSON）──
    print("[Infer] 预构建检索器...")
    retrievers = build_retrievers(knowledge_dirs)
    print(f"[Infer] 检索器就绪: {len(retrievers)} 个目录\n")

    # ── 增量保存：每道题完成后立即写入检查点文件（线程安全）──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    checkpoint_path = output_path + ".ckpt.csv"
    collected_results: list[dict] = []
    saved_count = 0
    _ckpt_lock = threading.Lock()

    def _on_question_done(result: dict):
        nonlocal saved_count
        with _ckpt_lock:
            collected_results.append(result)
            try:
                partial_df = _append_results_to_df(df.copy(), collected_results)
                partial_df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
                saved_count += 1
                print(f"  [Checkpoint] 已保存 {saved_count}/{len(questions)} 题 → {os.path.basename(checkpoint_path)}")
            except Exception as e:
                print(f"  [Checkpoint] 保存失败: {e}")

    results = run_mapreduce_inference(
        knowledge_dirs=knowledge_dirs,
        questions=questions,
        map_llm_func=map_llm_func,
        reduce_llm_func=reduce_llm_func,
        infer_prompt_func=infer_prompt_func,
        summary_prompt_func=summary_prompt_func,
        edge_case_prompt_func=edge_case_prompt_func,
        qa_direct_prompt_func=qa_direct_prompt_func,
        pitfalls_func=pitfalls_func,
        extra_llm_func=extra_llm_func,
        extra_vendor=extra_vendor,
        extra_model=extra_model,
        force_extra_llm=force_extra_llm,
        embedding_func=embedding_func,
        tfidf_top_n=tfidf_top_n,
        embedding_top_n=embedding_top_n,
        map_max_workers=map_max_workers,
        question_max_workers=question_max_workers,
        enable_edge_cases=enable_edge_cases,
        enable_qa_direct=enable_qa_direct,
        pre_built_retrievers=retrievers,
        on_question_done=_on_question_done,
        reduce_batch_size=reduce_batch_size,
    )

    df = _append_results_to_df(df, results)
    _write_output_file(df, output_path)
    print(f"\n[Infer] 最终结果已保存至: {output_path}")

    # 清理检查点文件
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"[Infer] 检查点文件已清理: {os.path.basename(checkpoint_path)}")
        except Exception:
            pass

    return output_path


# ─── 独立测试 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 60)
    print("[mapreduce_infer] 开始独立测试（v2 四阶段流水线）")
    print("=" * 60)

    def _mock_llm(prompt: str) -> dict:
        if "边缘案例" in prompt or "Edge_Case" in prompt:
            return {
                "content": json.dumps({
                    "Match_Status": "YES",
                    "Rejection_Reason": "",
                    "Reasoning_Chain": "在参考案例中发现与问题相关的信息。",
                    "Derived_Answer": "根据参考案例，答案是...",
                }, ensure_ascii=False)
            }
        return {
            "content": json.dumps({
                "Match_Status": "YES",
                "Rejection_Reason": "",
                "Reasoning_Chain": "问题与知识块高度相关，可直接推导答案。",
                "Derived_Answer": "需及时取得合法凭证，关注合同条款的税务合规性。",
            }, ensure_ascii=False)
        }

    def _mock_reduce_llm(prompt: str) -> dict:
        return {
            "content": json.dumps({
                "Synthesis_Analysis": "综合多个知识块分析：建议在合同签订阶段即明确税务条款。",
                "Final_Answer": "签合同时应明确税务条款，及时索取合法凭证，关注发票认证期限。",
            }, ensure_ascii=False)
        }

    def _mock_infer_prompt(question: str, kh_text: str) -> str:
        return f"问题：{question}\n知识：{kh_text}"

    def _mock_summary_prompt(
        question: str, extra_info: str, candidates_text: str,
    ) -> str:
        return f"汇总问题：{question}\n候选答案：{candidates_text[:200]}"

    questions = [
        {"q_idx": 0, "question": "签合同时需要注意哪些税务风险?"},
        {"q_idx": 1, "question": "进项发票过了认证期限怎么处理?"},
    ]

    results = run_mapreduce_inference(
        knowledge_dirs=[],
        questions=questions,
        map_llm_func=_mock_llm,
        reduce_llm_func=_mock_reduce_llm,
        infer_prompt_func=_mock_infer_prompt,
        summary_prompt_func=_mock_summary_prompt,
        map_max_workers=2,
    )

    print(f"\n测试结果（共 {len(results)} 条）:")
    for r in results:
        print(f"  Q: {r['question']}")
        print(f"  检索候选: {r['retrieval_candidates_count']}")
        print(f"  Map命中: {r['map_match_count']}/{r['map_total_evaluated']}")
        print(f"  A: {str(r.get('final_answer', ''))[:80]}")
        print()

    print("[mapreduce_infer] 测试完成！")
