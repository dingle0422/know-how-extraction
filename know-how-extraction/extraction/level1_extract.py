"""
一级提炼：对每条问答样本独立调用 LLM，抽取泛化 Know-How 片段。
支持多线程并发 + 断点续传 + 指数退避重试。
"""

import os
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import json5
import pandas as pd

file_lock = Lock()


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


def _process_single_item(
    c: int,
    data_train: pd.DataFrame,
    eb: str,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 100,
):
    q = data_train["question"].iloc[c]
    r = data_train["reasoning"].iloc[c]
    a = data_train["answer"].iloc[c]

    retry_count = 0
    last_error_msg = ""

    while True:
        if max_retries is not None and retry_count >= max_retries:
            error_info = {
                "index": c,
                "status": "failed",
                "error": "达到最大重试次数",
                "retry_count": retry_count,
                "last_error": last_error_msg,
            }
            with file_lock:
                _update_json_file(output_file, str(c), error_info)
            print(f"[Failed] 第 {c} 项在重试 {max_retries} 次后放弃")
            return c, "failed", None

        try:
            response = llm_func(prompt_func(eb, q, r, a))
            try:
                content = json5.loads(response["content"])
            except Exception as json_err:
                raise Exception(
                    f"JSON解析失败: {json_err} | 原始内容: "
                    f"{str(response.get('content', 'N/A'))[:100]}"
                )

            if "Know_How" not in content:
                raise Exception("响应缺少必需字段 'Know_How'")

            result = {
                "index": c,
                "Know_How": content["Know_How"],
                "Logic_Diagnosis": content.get("Logic_Diagnosis", ""),
                "status": "success",
                "retry_count": retry_count,
            }
            with file_lock:
                _update_json_file(output_file, str(c), result)

            msg = f"第 {c}/{len(data_train)} 项完成"
            if retry_count > 0:
                msg += f"（历经 {retry_count} 次重试）"
            print(f"[Success] {msg}")
            return c, "success", content["Know_How"]

        except Exception:
            retry_count += 1
            last_error_msg = traceback.format_exc()
            if retry_count % 5 == 1:
                print(f"[Error] 第 {c} 项第 {retry_count} 次失败: {last_error_msg[:150]}...")
            time.sleep(3)


def run_level1_extraction(
    data_train: pd.DataFrame,
    eb: str,
    llm_func,
    prompt_func,
    output_file: str = "./kh_results_level1.json",
    max_workers: int = 16,
    max_retries: int = 100,
):
    """
    多线程一级知识提炼入口。

    Parameters
    ----------
    data_train : 训练数据（需含 question, reasoning, answer 列）
    eb : 业务领域标签
    llm_func : LLM 调用函数（如 chat 或 qwen）
    prompt_func : 一级提炼 prompt 构造函数（如 single_v1）
    output_file : JSON 输出路径（支持断点续传）
    max_workers : 并发线程数
    max_retries : 每项最大重试次数
    """
    total = len(data_train)
    print(f"[Level-1] 开始多线程提炼，总数据量: {total}, 并发数: {max_workers}")

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 条记录，自动续传")
        except Exception:
            existing_data = {}

    pending_indices = [
        c for c in range(total)
        if str(c) not in existing_data
        or existing_data.get(str(c), {}).get("status") != "success"
    ]
    completed = total - len(pending_indices)
    print(f"  已完成: {completed}, 待处理: {len(pending_indices)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_single_item,
                c, data_train, eb, llm_func, prompt_func, output_file, max_retries,
            ): c
            for c in pending_indices
        }
        for future in as_completed(future_to_idx):
            c = future_to_idx[future]
            try:
                idx, status, _ = future.result()
                if status == "success":
                    completed += 1
                    print(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")
            except Exception as e:
                print(f"  第 {c} 项处理异常: {e}")

    print(f"[Level-1] 全部完成！结果保存于: {output_file}")
    return output_file


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("[level1_extract] 开始独立测试")
    print("=" * 60)

    def _mock_llm(prompt: str) -> dict:
        return {
            "content": (
                '{"Know_How": "合同签订时需双方签字盖章，条款需合法合规，'
                '涉及税务事项应及时取得合法凭证。", "Logic_Diagnosis": "逻辑正常"}'
            )
        }

    def _mock_prompt(eb, q, r, a):
        return f"领域:{eb}\n问题:{q}\n答案:{a}"

    test_df = pd.DataFrame(
        {
            "question": [
                "合同签订需要注意什么?",
                "增值税发票如何认证?",
                "企业所得税汇算清缴截止日期是?",
            ],
            "reasoning": ["...", "...", "..."],
            "answer": [
                "需双方签字盖章，条款合法合规。",
                "登录增值税发票综合服务平台进行勾选认证。",
                "次年5月31日前完成汇算清缴。",
            ],
        }
    )

    tmp_dir = tempfile.mkdtemp()
    output_file = os.path.join(tmp_dir, "kh_level1_test.json")

    result = run_level1_extraction(
        data_train=test_df,
        eb="财税",
        llm_func=_mock_llm,
        prompt_func=_mock_prompt,
        output_file=output_file,
        max_workers=2,
        max_retries=3,
    )

    with open(result, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n测试结果预览（共 {len(data)} 条）:")
    for k, v in sorted(data.items(), key=lambda x: int(x[0])):
        status = v.get("status")
        kh_preview = str(v.get("Know_How", ""))[:60]
        print(f"  [{k}] status={status} | Know_How: {kh_preview}...")

    print("\n[level1_extract] 测试完成！")
