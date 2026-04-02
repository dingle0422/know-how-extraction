"""
MapReduce 推理模块（v2）
======================
基于检索增强推理架构，完整实现四阶段流水线:
  Phase 1: 双路并行检索（TF-IDF + Dense Embedding）
  Phase 2: 并行 Map 推理（含有效性校验）
  Phase 3: 边缘案例兜底（仅 QA Know-How）
  Phase 4: Reduce 融合推理
"""

import json
import os
import concurrent.futures
import threading
from typing import Callable
from tqdm import tqdm

import time

from .retrieval import (
    build_retrievers,
    retrieve_candidates,
    load_knowledge_content,
    load_edge_cases,
    load_level1_knowhow_map,
    retrieve_edge_cases,
    format_edge_cases_text,
    QADirectRetriever,
    retrieve_qa_direct_candidates,
    format_qa_direct_text,
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


# ─── Phase 3: 边缘案例兜底 ──────────────────────────────────────────────────

def _run_edge_case_fallback(
    task_input: dict,
    llm_func: Callable,
    edge_case_prompt_func: Callable,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    embedding_func: Callable = None,
) -> dict:
    """对 QA Know-How 中 Map 判定无效的条目，用边缘案例做兜底推理。

    通过双路独立检索（TF-IDF Top-N + Dense Top-N → 并集去重）筛选最相关的边缘案例，
    并附带 Level-1 Know-How 作为辅助推理上下文。
    """
    question = task_input["question"]
    knowledge_dir = task_input["knowledge_dir"]
    entry_key = task_input["entry_key"]
    level1_map = task_input.get("level1_map", {})

    _empty = {
        "q_idx": task_input["q_idx"],
        "source_dir": task_input["source_dir"],
        "entry_key": entry_key,
        "knowledge_type": "qa_v2",
        "kh_source_id": f"{task_input['source_dir']}:{entry_key}:edge",
        "Match_Status": "NO",
        "Rejection_Reason": "无可用边缘案例",
        "Reasoning_Chain": "",
        "Derived_Answer": "",
        "is_edge_case_fallback": True,
    }

    all_edge_cases = load_edge_cases(knowledge_dir, entry_key)
    if not all_edge_cases:
        return _empty

    edge_cases = retrieve_edge_cases(
        question, all_edge_cases,
        tfidf_top_n=tfidf_top_n,
        embedding_top_n=embedding_top_n,
        embedding_func=embedding_func,
        level1_map=level1_map,
    )
    if not edge_cases:
        return _empty

    ec_text = format_edge_cases_text(edge_cases, level1_map=level1_map)

    try:
        prompt = edge_case_prompt_func(question, ec_text)
        raw = llm_func(prompt)["content"]
        result = json.loads(_clean_json_string(raw))
    except Exception as e:
        result = {
            "Match_Status": "NO",
            "Rejection_Reason": f"边缘案例推理失败: {e}",
            "Reasoning_Chain": "",
            "Derived_Answer": "",
        }

    result["q_idx"] = task_input["q_idx"]
    result["source_dir"] = task_input["source_dir"]
    result["entry_key"] = entry_key
    result["knowledge_type"] = "qa_v2"
    result["kh_source_id"] = f"{task_input['source_dir']}:{entry_key}:edge"
    result["is_edge_case_fallback"] = True
    return result


# ─── Phase 4: Reduce 融合 ───────────────────────────────────────────────────

def _run_reduce_task(
    task_input: dict,
    llm_func: Callable,
    summary_prompt_func: Callable,
) -> dict:
    """汇总所有有效 Map 推理结果（含 LLM 裸考），融合生成最终答案。"""
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
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    question_max_workers: int = 1,
    enable_edge_cases: bool = True,
    enable_qa_direct: bool = True,
    pre_built_retrievers=None,
    on_question_done: Callable = None,
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
    edge_case_prompt_func    : Phase 3 边缘案例兜底 prompt 构造函数
    qa_direct_prompt_func    : QA 直检 Map prompt 构造函数
    pitfalls_func            : 陷阱提示函数，可选
    extra_llm_func           : 额外信息 LLM 函数（裸考），可选
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

    # ── 预加载 Level-1 Know-How 映射（用于 Phase 3 边缘案例兜底）──
    level1_maps: dict[str, dict[int, str]] = {}
    if enable_edge_cases and edge_case_prompt_func is not None:
        for d in knowledge_dirs:
            l1 = load_level1_knowhow_map(d)
            if l1:
                level1_maps[d] = l1
        if level1_maps:
            total_l1 = sum(len(v) for v in level1_maps.values())
            print(f"[Infer] 已加载 Level-1 Know-How 映射: {total_l1} 条")

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

        # ── Phase 1b: QA 直检并行检索 ─────────────────────────────────
        qa_direct_hits = []
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
        _log("Phase1", f"QA 直检命中: {len(qa_direct_hits)} 条")

        if not candidates and not qa_direct_hits:
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

        # 构建 QA 直检 Map 任务
        qa_direct_tasks = []
        for hit in qa_direct_hits:
            qa_direct_tasks.append({
                "q_idx": q_idx,
                "question": question,
                "source_dir": hit["source_dir"],
                "qa_index": hit["qa_index"],
                "qa_text": format_qa_direct_text(hit),
            })

        if not map_tasks and not qa_direct_tasks:
            _log("Phase1", "知识块内容为空，跳过此题")
            pbar.update(1)
            return _build_empty_result(q_idx, question)

        # ── Phase 2: 并行 Map 推理（Level-2 知识块 + QA 直检 + LLM 裸考 同时进行）──
        bare_enabled = extra_llm_func is not None
        total_map_tasks = len(map_tasks) + len(qa_direct_tasks) + (1 if bare_enabled else 0)
        _log("Phase2", f"启动并行 Map: L2={len(map_tasks)}, QA直检={len(qa_direct_tasks)}, "
             f"裸考={1 if bare_enabled else 0}，共 {total_map_tasks} 个任务，并发={map_max_workers}")

        map_results = []
        qa_direct_results = []
        llm_bare_result = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=map_max_workers) as executor:
            l2_futures = {
                executor.submit(
                    _run_map_task, task, map_llm_func,
                    infer_prompt_func, pitfalls_func,
                ): ("l2", task)
                for task in map_tasks
            }
            qd_futures = {
                executor.submit(
                    _run_qa_direct_map_task, task, map_llm_func,
                    qa_direct_prompt_func, pitfalls_func,
                ): ("qd", task)
                for task in qa_direct_tasks
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
            all_futures = {**l2_futures, **qd_futures, **bare_futures}
            done_count = 0
            for future in concurrent.futures.as_completed(all_futures):
                tag, _task = all_futures[future]
                try:
                    result = future.result()
                    status = result.get("Match_Status", "?")
                    done_count += 1
                    if tag == "l2":
                        map_results.append(result)
                        _log("Phase2", f"  L2 [{done_count}/{total_map_tasks}] key={result.get('entry_key')} → {status}")
                    elif tag == "qd":
                        qa_direct_results.append(result)
                        _log("Phase2", f"  QA直检 [{done_count}/{total_map_tasks}] qa_idx={result.get('qa_index')} → {status}")
                    else:
                        llm_bare_result = result
                        _log("Phase2", f"  裸考 [{done_count}/{total_map_tasks}] → {status}")
                except Exception as exc:
                    done_count += 1
                    print(f"  [-] Map 异常 (q_idx={q_idx}, type={tag}): {exc}")

        l2_valid = [r for r in map_results if r.get("Match_Status") == "YES"]
        rejected_results = [r for r in map_results if r.get("Match_Status") != "YES"]
        qa_direct_valid = [r for r in qa_direct_results if r.get("Match_Status") == "YES"]
        bare_valid = llm_bare_result is not None and llm_bare_result.get("Match_Status") == "YES"

        valid_results = list(l2_valid)
        valid_results.extend(qa_direct_valid)
        if bare_valid:
            valid_results.append(llm_bare_result)

        _log("Phase2", f"完成: L2 {len(l2_valid)}/{len(map_results)} 有效, "
             f"QA直检 {len(qa_direct_valid)}/{len(qa_direct_results)} 有效, "
             f"裸考 {'YES' if bare_valid else ('NO' if llm_bare_result else '未启用')}")

        # ── Phase 3: 边缘案例兜底（仅 QA Know-How）──────────────────
        edge_fallback_results = []
        if enable_edge_cases and edge_case_prompt_func is not None:
            qa_rejected = [
                r for r in rejected_results
                if r.get("knowledge_type") == "qa_v2"
            ]
            if qa_rejected:
                _log("Phase3", f"触发边缘案例兜底: {len(qa_rejected)} 个被拒绝的 QA 知识块")
                edge_tasks = []
                for r in qa_rejected:
                    kd = _find_knowledge_dir(knowledge_dirs, r["source_dir"])
                    edge_tasks.append({
                        "q_idx": q_idx,
                        "question": question,
                        "source_dir": r["source_dir"],
                        "knowledge_dir": kd,
                        "entry_key": r["entry_key"],
                        "level1_map": level1_maps.get(kd, {}),
                    })
                with concurrent.futures.ThreadPoolExecutor(max_workers=map_max_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_edge_case_fallback, task,
                            map_llm_func, edge_case_prompt_func,
                            tfidf_top_n, embedding_top_n, embedding_func,
                        ): task
                        for task in edge_tasks
                        if task["knowledge_dir"] is not None
                    }
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            ec_result = future.result()
                            edge_fallback_results.append(ec_result)
                            if ec_result.get("Match_Status") == "YES":
                                valid_results.append(ec_result)
                            _log("Phase3", f"  兜底 key={ec_result.get('entry_key')} → {ec_result.get('Match_Status')}")
                        except Exception as exc:
                            print(f"  [-] EdgeCase 异常 (q_idx={q_idx}): {exc}")

                edge_success = len([r for r in edge_fallback_results if r.get("Match_Status") == "YES"])
                _log("Phase3", f"完成: {edge_success}/{len(edge_fallback_results)} 兜底成功")
            else:
                _log("Phase3", "无被拒绝的 QA 知识块，跳过兜底")
        elif not enable_edge_cases:
            _log("Phase3", "已禁用（--no-edge-cases）")

        # ── Phase 4: Reduce 融合 ───────────────────────────────────────
        _log("Phase4", f"Reduce 融合: 汇总 {len(valid_results)} 个有效推理结果...")
        reduce_result = _run_reduce_task(
            {"q_idx": q_idx, "question": question, "valid_results": valid_results},
            reduce_llm_func, summary_prompt_func,
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
            "qa_direct_count": len(qa_direct_results),
            "qa_direct_match_count": len(qa_direct_valid),
            "edge_fallback_count": len(edge_fallback_results),
            "edge_fallback_match_count": len([
                r for r in edge_fallback_results if r.get("Match_Status") == "YES"
            ]),
            "total_valid_count": len(valid_results),
            "map_results": map_results,
            "qa_direct_results": qa_direct_results,
            "edge_fallback_results": edge_fallback_results,
            "valid_results": valid_results,
            "rejected_reasons": [
                f"KH[{r.get('kh_source_id')}]: {r.get('Rejection_Reason')}"
                for r in map_results if r.get("Match_Status") != "YES"
            ],
            "extra_information": llm_bare_answer,
            "synthesis_analysis": reduce_result.get("Synthesis_Analysis", ""),
            "final_answer": final_ans,
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
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    question_max_workers: int = 1,
    enable_edge_cases: bool = True,
    enable_qa_direct: bool = True,
    question_column: str = "question",
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
        embedding_func=embedding_func,
        tfidf_top_n=tfidf_top_n,
        embedding_top_n=embedding_top_n,
        map_max_workers=map_max_workers,
        question_max_workers=question_max_workers,
        enable_edge_cases=enable_edge_cases,
        enable_qa_direct=enable_qa_direct,
        pre_built_retrievers=retrievers,
        on_question_done=_on_question_done,
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
