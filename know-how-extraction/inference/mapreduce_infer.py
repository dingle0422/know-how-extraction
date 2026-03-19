"""
MapReduce 推理模块
Map 阶段：将用户问题与每个 Know-How 块并发匹配推理。
Reduce 阶段：融合有效候选 + 额外信息，生成最终答案。
"""

import json
import concurrent.futures
from tqdm import tqdm
import pandas as pd


def _clean_json_string(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ─── Map 阶段 ───────────────────────────────────────────────────────────────

def _run_map_task(
    task_input: dict,
    llm_func,
    infer_prompt_func,
    pitfalls_func=None,
) -> dict:
    q_idx = task_input["q_idx"]
    question = task_input["question"]
    kh_id = task_input["kh_id"]
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

    result["q_idx"] = q_idx
    result["kh_source_id"] = kh_id
    return result


# ─── Reduce 阶段 ────────────────────────────────────────────────────────────

def _run_reduce_task(
    task_input: dict,
    llm_func,
    summary_prompt_func,
    extra_llm_func=None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
) -> dict:
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
                extra_prompt, vendor=extra_vendor, model=extra_model
            )["content"]
        except Exception as e:
            extra_info = f"获取额外信息失败: {e}"

    if not valid_results:
        candidates_text = "【空】所有并发线程均拒答，未匹配到有效KNOW-HOW。"
    else:
        candidates_text = "\n\n".join(
            [
                f"--- 来自知识块 [{r.get('kh_source_id')}] 的推理 ---\n"
                f"推导逻辑: {r.get('Reasoning_Chain')}\n"
                f"候选答案: {r.get('Derived_Answer')}"
                for r in valid_results
            ]
        )

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


# ─── 完整 MapReduce 流程入口 ─────────────────────────────────────────────────

def run_mapreduce_inference(
    kh_json_path: str,
    test_csv_path: str,
    output_csv_path: str,
    map_llm_func,
    reduce_llm_func,
    infer_prompt_func,
    summary_prompt_func,
    pitfalls_func=None,
    extra_llm_func=None,
    extra_vendor: str = "volc",
    extra_model: str = "deepseek-v3.2",
    max_workers: int = 16,
):
    """
    MapReduce 推理全流程。

    Parameters
    ----------
    kh_json_path : 二级压缩结果 JSON（Know-How 知识块）
    test_csv_path : 测试集 CSV
    output_csv_path : 推理结果输出 CSV
    map_llm_func : Map 阶段 LLM 函数
    reduce_llm_func : Reduce 阶段 LLM 函数
    infer_prompt_func : Map prompt（如 infer_v1）
    summary_prompt_func : Reduce prompt（如 summary_v0）
    pitfalls_func : 陷阱提示函数（如 potential_pitfalls），可选
    extra_llm_func : 额外信息 LLM 函数（裸考），可选
    """
    with open(kh_json_path, "r", encoding="utf-8") as f:
        kh_data = json.load(f)
    print(f"[Infer] 加载 {len(kh_data)} 条 KNOW-HOW 知识块")

    df = pd.read_csv(test_csv_path)
    print(f"[Infer] 加载 {len(df)} 条测试问题")

    # ── Map 任务池 ──
    map_task_inputs = []
    for index, row in df.iterrows():
        question = str(row.get("question", ""))
        if not question.strip():
            continue
        for kh_id, kh_info in kh_data.items():
            kh_text = kh_info.get("Final_Know_How", "")
            map_task_inputs.append({
                "q_idx": index,
                "question": question,
                "kh_id": kh_id,
                "kh_text": kh_text,
            })

    print(f"\n[阶段一] Map 并发推理... (共 {len(map_task_inputs)} 个子任务)")
    map_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_map_task, task, map_llm_func, infer_prompt_func, pitfalls_func,
            ): task
            for task in map_task_inputs
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Map Progress",
        ):
            try:
                map_results.append(future.result())
            except Exception as exc:
                print(f"[-] Map 异常: {exc}")

    # ── 按问题聚合 ──
    grouped = {i: [] for i in df.index}
    for res in map_results:
        q_idx = res.get("q_idx")
        if q_idx is not None:
            grouped[q_idx].append(res)

    reduce_task_inputs = []
    for q_idx, results in grouped.items():
        question = str(df.at[q_idx, "question"])
        valid_results = [r for r in results if r.get("Match_Status") == "YES"]
        reduce_task_inputs.append({
            "q_idx": q_idx,
            "question": question,
            "valid_results": valid_results,
            "all_map_results": results,
        })

    # ── Reduce ──
    print(f"\n[阶段二] Reduce 融合推理... (共 {len(reduce_task_inputs)} 个子任务)")
    reduce_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_reduce_task,
                task,
                reduce_llm_func,
                summary_prompt_func,
                extra_llm_func,
                extra_vendor,
                extra_model,
            ): task
            for task in reduce_task_inputs
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reduce Progress",
        ):
            try:
                res = future.result()
                reduce_results[res["q_idx"]] = res
            except Exception as exc:
                print(f"[-] Reduce 异常: {exc}")

    # ── 回写 DataFrame ──
    df["Extra_Information"] = ""
    df["Map_Total_Evaluated"] = 0
    df["Map_Match_Count"] = 0
    df["Map_Valid_Details"] = ""
    df["Map_Rejected_Reasons"] = ""
    df["Reduce_Analysis"] = ""
    df["Final_Inference_Answer"] = ""

    for task_info in reduce_task_inputs:
        q_idx = task_info["q_idx"]
        all_maps = task_info["all_map_results"]
        valids = task_info["valid_results"]

        rejected = [
            f"KH[{r.get('kh_source_id')}]: {r.get('Rejection_Reason')}"
            for r in all_maps
            if r.get("Match_Status") == "NO"
        ]
        r_res = reduce_results.get(q_idx, {})

        df.at[q_idx, "Extra_Information"] = r_res.get("Extra_Information", "")
        df.at[q_idx, "Map_Total_Evaluated"] = len(all_maps)
        df.at[q_idx, "Map_Match_Count"] = len(valids)
        df.at[q_idx, "Map_Valid_Details"] = (
            json.dumps(valids, ensure_ascii=False) if valids else "None"
        )
        df.at[q_idx, "Map_Rejected_Reasons"] = "\n".join(rejected)
        df.at[q_idx, "Reduce_Analysis"] = r_res.get("Synthesis_Analysis", "")
        df.at[q_idx, "Final_Inference_Answer"] = r_res.get("Final_Answer", "")

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Infer] 推理完成！结果保存至: {output_csv_path}")
    return output_csv_path
