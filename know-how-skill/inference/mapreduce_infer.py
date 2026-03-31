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
from typing import Callable
from tqdm import tqdm

from .retrieval import (
    retrieve_candidates,
    load_knowledge_content,
    load_edge_cases,
    format_edge_cases_text,
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


# ─── Phase 3: 边缘案例兜底 ──────────────────────────────────────────────────

def _run_edge_case_fallback(
    task_input: dict,
    llm_func: Callable,
    edge_case_prompt_func: Callable,
) -> dict:
    """对 QA Know-How 中 Map 判定无效的条目，用边缘案例做兜底推理。"""
    question = task_input["question"]
    knowledge_dir = task_input["knowledge_dir"]
    entry_key = task_input["entry_key"]

    edge_cases = load_edge_cases(knowledge_dir, entry_key)
    if not edge_cases:
        return {
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

    ec_text = format_edge_cases_text(edge_cases)

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
    extra_llm_func: Callable = None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
) -> dict:
    """汇总所有有效 Map 推理结果，融合生成最终答案。"""
    q_idx = task_input["q_idx"]
    question = task_input["question"]
    valid_results = task_input["valid_results"]

    extra_info = ""
    if extra_llm_func is not None:
        try:
            extra_prompt = (
                f"作为一个资深行业顾问，请结合你的专业知识，"
                f"直接且详尽地解答以下问题：\n【用户问题】：{question}"
            )
            extra_info = extra_llm_func(
                extra_prompt, vendor=extra_vendor, model=extra_model,
            )["content"]
        except Exception as e:
            extra_info = f"获取额外信息失败: {e}"

    if not valid_results:
        candidates_text = "【空】所有候选知识块均被判定为不相关，未匹配到有效 Know-How。"
    else:
        fragments = []
        for r in valid_results:
            source_tag = "边缘案例兜底" if r.get("is_edge_case_fallback") else "知识块"
            fragments.append(
                f"--- 来自{source_tag} [{r.get('kh_source_id')}] 的推理 ---\n"
                f"推导逻辑: {r.get('Reasoning_Chain')}\n"
                f"候选答案: {r.get('Derived_Answer')}"
            )
        candidates_text = "\n\n".join(fragments)

    prompt = summary_prompt_func(question, extra_info, candidates_text)

    try:
        raw = llm_func(prompt)["content"]
        reduce_result = json.loads(_clean_json_string(raw))
    except Exception as e:
        reduce_result = {
            "Synthesis_Analysis": f"解析失败: {e}",
            "Final_Answer": raw if "raw" in dir() else "处理失败",
        }

    reduce_result["q_idx"] = q_idx
    reduce_result["Extra_Information"] = extra_info
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
    pitfalls_func: Callable = None,
    extra_llm_func: Callable = None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    enable_edge_case_fallback: bool = True,
) -> list[dict]:
    """
    完整 4 阶段 MapReduce 推理流水线。

    Parameters
    ----------
    knowledge_dirs           : 指定参与推理的 knowledge 目录列表
    questions                : 问题列表，每项为 {"q_idx": int, "question": str, ...}
    map_llm_func             : Map 阶段 LLM 函数
    reduce_llm_func          : Reduce 阶段 LLM 函数
    infer_prompt_func        : Map prompt 构造函数
    summary_prompt_func      : Reduce prompt 构造函数
    edge_case_prompt_func    : Phase 3 边缘案例兜底 prompt 构造函数
    pitfalls_func            : 陷阱提示函数，可选
    extra_llm_func           : 额外信息 LLM 函数（裸考），可选
    embedding_func           : Dense embedding 函数，可选
    tfidf_top_n              : TF-IDF 检索 Top-N
    embedding_top_n          : Dense Embedding 检索 Top-N
    map_max_workers          : Map 阶段并发线程数
    enable_edge_case_fallback: 是否启用边缘案例兜底

    Returns
    -------
    推理结果列表，每项包含问题、Map 详情、Reduce 最终答案等
    """
    print(f"[Infer] 收到 {len(questions)} 个问题, "
          f"{len(knowledge_dirs)} 个知识目录")

    all_output = []

    for q_item in tqdm(questions, desc="Questions"):
        q_idx = q_item["q_idx"]
        question = q_item["question"]

        # ── Phase 1: 双路并行检索 ──────────────────────────────────────
        candidates = retrieve_candidates(
            question=question,
            knowledge_dirs=knowledge_dirs,
            tfidf_top_n=tfidf_top_n,
            embedding_top_n=embedding_top_n,
            embedding_func=embedding_func,
        )

        if not candidates:
            all_output.append(_build_empty_result(q_idx, question))
            continue

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
            all_output.append(_build_empty_result(q_idx, question))
            continue

        # ── Phase 2: 并行 Map 推理 ─────────────────────────────────────
        map_results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=map_max_workers,
        ) as executor:
            futures = {
                executor.submit(
                    _run_map_task, task, map_llm_func,
                    infer_prompt_func, pitfalls_func,
                ): task
                for task in map_tasks
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    map_results.append(future.result())
                except Exception as exc:
                    print(f"[-] Map 异常 (q_idx={q_idx}): {exc}")

        valid_results = [
            r for r in map_results if r.get("Match_Status") == "YES"
        ]
        rejected_results = [
            r for r in map_results if r.get("Match_Status") != "YES"
        ]

        # ── Phase 3: 边缘案例兜底（仅 QA Know-How）──────────────────
        edge_fallback_results = []
        if enable_edge_case_fallback and edge_case_prompt_func is not None:
            qa_rejected = [
                r for r in rejected_results
                if r.get("knowledge_type") == "qa_v2"
            ]
            if qa_rejected:
                edge_tasks = []
                for r in qa_rejected:
                    edge_tasks.append({
                        "q_idx": q_idx,
                        "question": question,
                        "source_dir": r["source_dir"],
                        "knowledge_dir": _find_knowledge_dir(
                            knowledge_dirs, r["source_dir"],
                        ),
                        "entry_key": r["entry_key"],
                    })

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=map_max_workers,
                ) as executor:
                    futures = {
                        executor.submit(
                            _run_edge_case_fallback, task,
                            map_llm_func, edge_case_prompt_func,
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
                        except Exception as exc:
                            print(f"[-] EdgeCase 异常 (q_idx={q_idx}): {exc}")

        # ── Phase 4: Reduce 融合 ───────────────────────────────────────
        reduce_input = {
            "q_idx": q_idx,
            "question": question,
            "valid_results": valid_results,
        }
        reduce_result = _run_reduce_task(
            reduce_input, reduce_llm_func, summary_prompt_func,
            extra_llm_func, extra_vendor, extra_model,
        )

        # ── 组装输出 ───────────────────────────────────────────────────
        all_output.append({
            "q_idx": q_idx,
            "question": question,
            "retrieval_candidates_count": len(candidates),
            "map_total_evaluated": len(map_results),
            "map_match_count": len([
                r for r in map_results if r.get("Match_Status") == "YES"
            ]),
            "edge_fallback_count": len(edge_fallback_results),
            "edge_fallback_match_count": len([
                r for r in edge_fallback_results
                if r.get("Match_Status") == "YES"
            ]),
            "total_valid_count": len(valid_results),
            "map_results": map_results,
            "edge_fallback_results": edge_fallback_results,
            "valid_results": valid_results,
            "rejected_reasons": [
                f"KH[{r.get('kh_source_id')}]: {r.get('Rejection_Reason')}"
                for r in map_results
                if r.get("Match_Status") != "YES"
            ],
            "extra_information": reduce_result.get("Extra_Information", ""),
            "synthesis_analysis": reduce_result.get("Synthesis_Analysis", ""),
            "final_answer": reduce_result.get("Final_Answer", ""),
        })

    print(f"\n[Infer] 推理完成！共处理 {len(all_output)} 个问题")
    return all_output


def _build_empty_result(q_idx: int, question: str) -> dict:
    """构建未检索到候选知识块时的空结果。"""
    return {
        "q_idx": q_idx,
        "question": question,
        "retrieval_candidates_count": 0,
        "map_total_evaluated": 0,
        "map_match_count": 0,
        "edge_fallback_count": 0,
        "edge_fallback_match_count": 0,
        "total_valid_count": 0,
        "map_results": [],
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
    pitfalls_func: Callable = None,
    extra_llm_func: Callable = None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
    embedding_func: Callable = None,
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    map_max_workers: int = 4,
    enable_edge_case_fallback: bool = True,
    question_column: str = "question",
) -> str:
    """
    文件便捷入口：读取测试集（CSV/Excel），执行推理，输出结果文件。

    在输入数据的基础上追加以下列：
      - Retrieval_Candidates  : Phase 1 检索到的候选知识块数
      - Map_Total_Evaluated   : Phase 2 实际评估的知识块数
      - Map_Match_Count       : Phase 2 判定有效的知识块数
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
    input_path       : 输入文件路径（.csv / .xlsx）
    output_path      : 输出文件路径（.csv / .xlsx，格式以此为准）
    question_column  : 问题列名，默认 "question"
    其余参数与 run_mapreduce_inference 一致

    Returns
    -------
    输出文件路径
    """
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

    results = run_mapreduce_inference(
        knowledge_dirs=knowledge_dirs,
        questions=questions,
        map_llm_func=map_llm_func,
        reduce_llm_func=reduce_llm_func,
        infer_prompt_func=infer_prompt_func,
        summary_prompt_func=summary_prompt_func,
        edge_case_prompt_func=edge_case_prompt_func,
        pitfalls_func=pitfalls_func,
        extra_llm_func=extra_llm_func,
        extra_vendor=extra_vendor,
        extra_model=extra_model,
        embedding_func=embedding_func,
        tfidf_top_n=tfidf_top_n,
        embedding_top_n=embedding_top_n,
        map_max_workers=map_max_workers,
        enable_edge_case_fallback=enable_edge_case_fallback,
    )

    df = _append_results_to_df(df, results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _write_output_file(df, output_path)
    print(f"\n[Infer] 结果已保存至: {output_path}")
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
