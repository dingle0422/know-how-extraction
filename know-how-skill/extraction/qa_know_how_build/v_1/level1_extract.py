"""
一级提炼：对每条问答样本独立调用 LLM，抽取泛化 Know-How 片段。
支持多线程并发 + 断点续传 + 指数退避重试。
"""

import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

try:
    from prompts import safe_parse_json_with_llm_repair
except ImportError:
    sys.path.insert(0, _SKILL_ROOT)
    from prompts import safe_parse_json_with_llm_repair

sys.path.insert(0, _EXTRACTION_DIR)
from utils import sanitize_for_json

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
        json.dump(sanitize_for_json(data_dict), f, ensure_ascii=False, indent=2)


def _process_single_item(
    c: int,
    data_train: pd.DataFrame,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 100,
):
    q = data_train["question"].iloc[c]
    r = data_train["reasoning"].iloc[c]
    a = data_train["answer"].iloc[c]
    ei = data_train["Extra_Information"].iloc[c]

    retry_count = 0
    last_error_msg = ""

    while True:
        if max_retries is not None and retry_count >= max_retries:
            error_info = {
                "index": c,
                "input": {
                    "question": q,
                    "reasoning": r,
                    "answer": a,
                    "Extra_Information": ei,
                },
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
            response = llm_func(prompt_func(ei, q, r, a))
            try:
                content = safe_parse_json_with_llm_repair(
                    response["content"], llm_func=llm_func
                )
            except Exception as json_err:
                raise Exception(
                    f"JSON解析失败（含LLM修复）: {json_err} | 原始内容: "
                    f"{str(response.get('content', 'N/A'))[:100]}"
                )

            if "Know_How" not in content:
                raise Exception("响应缺少必需字段 'Know_How'")

            result = {
                "index": c,
                "input": {
                    "question": q,
                    "reasoning": r,
                    "answer": a,
                    "Extra_Information": ei,
                },
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
    llm_func,
    prompt_func,
    output_file: str = "./output/kh_results_level1.json",
    max_workers: int = os.cpu_count() or 4,
    max_retries: int = 100,
):
    """
    多线程一级知识提炼入口。

    Parameters
    ----------
    data_train : 训练数据（需含 question, reasoning, answer 列）
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
                c, data_train, llm_func, prompt_func, output_file, max_retries,
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


_SUPPORTED_QA_EXTS = {".csv", ".xlsx", ".xls"}


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, _SKILL_ROOT)
    sys.path.insert(0, _EXTRACTION_DIR)
    sys.path.insert(0, _PACKAGE_DIR)
    from utils import get_source_stem
    from llm_client import chat
    from prompts_v1 import single_v1

    parser = argparse.ArgumentParser(
        description="QA 一级知识提炼（Level 1）"
    )
    parser.add_argument(
        "--files", "-f", nargs="+", default=None,
        help="指定要处理的 QA 源数据文件路径（支持多个）；不指定则处理 input 目录下所有文件",
    )
    args = parser.parse_args()

    input_dir = os.path.join(_PACKAGE_DIR, "input")
    output_dir = os.path.join(_PACKAGE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.files:
        source_files = []
        for fp in args.files:
            fp = os.path.abspath(fp)
            if not os.path.isfile(fp):
                print(f"[警告] 文件不存在，已跳过: {fp}")
                continue
            if os.path.splitext(fp)[1].lower() not in _SUPPORTED_QA_EXTS:
                print(f"[警告] 不支持的文件类型，已跳过: {fp}（支持: {_SUPPORTED_QA_EXTS}）")
                continue
            source_files.append(fp)
        if not source_files:
            raise FileNotFoundError("指定的文件中没有可处理的有效文件")
        mode_desc = "指定文件模式"
    else:
        source_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_QA_EXTS
        ])
        if not source_files:
            raise FileNotFoundError(
                f"input 目录中未找到支持的数据文件（{_SUPPORTED_QA_EXTS}）：{input_dir}"
            )
        mode_desc = "全量扫描模式"

    print("=" * 60)
    print(f"[level1_extract] {mode_desc}，共 {len(source_files)} 个源数据文件（仅一级提炼）")
    print("=" * 60)
    for i, fp in enumerate(source_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    for idx, source_file in enumerate(source_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{len(source_files)}] 处理: {os.path.basename(source_file)}")
        print(f"{'═' * 60}")

        source_stem = get_source_stem(source_file)
        ext = os.path.splitext(source_file)[1].lower()
        if ext == ".csv":
            data_train = pd.read_csv(source_file, encoding="utf-8-sig")
        else:
            data_train = pd.read_excel(source_file, sheet_name=0)

        required_cols = {"question", "answer"}
        missing = required_cols - set(data_train.columns)
        if missing:
            print(f"  [跳过] 缺少必需列: {missing}")
            continue

        if "reasoning" not in data_train.columns:
            data_train["reasoning"] = ""
            print(f"  提示: 源文件中不含 reasoning 列，已自动创建并置空")

        core_cols = {"question", "reasoning", "answer"}
        if "Extra_Information" not in data_train.columns:
            extra_cols = [c for c in data_train.columns if c not in core_cols]
            if extra_cols:
                data_train["Extra_Information"] = data_train[extra_cols].apply(
                    lambda row: "; ".join(f"{k}={v}" for k, v in row.items() if pd.notna(v)),
                    axis=1,
                )
            else:
                data_train["Extra_Information"] = ""
        print(f"  数据加载完成: {len(data_train)} 条记录")

        output_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
        run_level1_extraction(
            data_train=data_train,
            llm_func=chat,
            prompt_func=single_v1,
            output_file=output_file,
            max_workers=os.cpu_count() or 4,
            max_retries=100,
        )

        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        md_file = output_file.replace(".json", ".md")
        with open(md_file, "w", encoding="utf-8") as f:
            for k, v in sorted(data.items(), key=lambda x: int(x[0])):
                kh = v.get("Know_How", "").strip()
                if kh:
                    f.write(kh)
                    f.write("\n\n---\n\n")
        print(f"  Markdown 预览文件已导出: {md_file}")

    print(f"\n{'═' * 60}")
    print(f"[level1_extract] 全部 {len(source_files)} 个数据文件处理完成！")
    print(f"{'═' * 60}")
