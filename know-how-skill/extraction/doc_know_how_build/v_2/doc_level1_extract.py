"""
文档型一级提炼 v2：按片段独立抽取（类 QA 模式）
================================================
对文档切分后的每个片段独立调用 LLM 进行 Know-How 抽取，
输出格式与 QA Level 1 完全一致，可直接接入 Level 2 压缩流程。

流程：
  1. 解析文档 → 段落切分 → 按字数窗口合并
  2. 抽取目录结构作为上下文（可选）
  3. 对每个片段独立调用 LLM 提炼 Know-How
  4. 输出与 QA Level 1 一致的 JSON 结构
"""

import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

_V1_DIR = os.path.join(_PACKAGE_DIR, "v_1")
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)
if _SKILL_ROOT not in sys.path:
    sys.path.insert(0, _SKILL_ROOT)

from doc_structure_parse import (
    parse_document,
    merge_segments_by_length,
    extract_toc,
    extract_toc_keywords,
    build_paragraphs,
)
from prompts import safe_parse_json_with_llm_repair

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


# ─── 文档解析 → 片段任务构建 ──────────────────────────────────────────────


def build_segment_tasks(
    doc_path: str,
    llm_func=None,
    min_seg_chars: int = 500,
    max_seg_chars: int = 2000,
    force_llm_toc: bool = True,
    llm_toc_workers: int = 8,
) -> tuple[list[dict], str, dict]:
    """
    解析文档 → 合并段落 → 抽取目录 → 构建片段任务列表。

    Returns
    -------
    tasks            : [{index, segment_idx, segment_text, toc_section, toc_level, keywords}]
    file_type        : 文档类型
    segment_content  : 合并后的 {序号: 文本}
    """
    print(f"  [解析] 解析文档原始内容...")
    full_text, segment_content, file_type = parse_document(doc_path)
    raw_count = len(segment_content)

    if min_seg_chars > 0:
        print(f"  [合并] 按字数窗口合并段落（min={min_seg_chars}, max={max_seg_chars}）...")
        segment_content, full_text = merge_segments_by_length(
            segment_content, min_chars=min_seg_chars, max_chars=max_seg_chars,
        )
    merged_count = len(segment_content)
    print(f"  文档类型: {file_type}, 原始段落: {raw_count}, 合并后: {merged_count}")

    print(f"  [目录] 抽取目录结构（用作片段上下文）...")
    toc_items = extract_toc(
        doc_path, full_text, segment_content, file_type,
        llm_func, llm_toc_workers=llm_toc_workers, force_llm_toc=force_llm_toc,
    )
    toc_items = extract_toc_keywords(toc_items, segment_content, llm_func)

    paragraphs = build_paragraphs(segment_content, toc_items)

    toc_kw_map = {item["title"]: item.get("keywords", []) for item in toc_items}

    tasks = []
    for i, para in enumerate(paragraphs):
        tasks.append({
            "index": i,
            "segment_idx": para["idx"],
            "segment_text": para["text"],
            "toc_section": para["toc_section"],
            "toc_level": para["toc_level"],
            "keywords": toc_kw_map.get(para["toc_section"], []),
        })

    return tasks, file_type, segment_content


# ─── 单片段处理（含重试） ─────────────────────────────────────────────────


def _process_single_segment(
    task: dict,
    llm_func,
    prompt_func,
    output_file: str,
    source_file_name: str,
    max_retries: int = 100,
):
    """
    处理单个片段：构造输入 → 调用 LLM → 解析 JSON → 写入输出文件。
    输出字段与 QA Level 1 完全一致（index / input / Know_How / Logic_Diagnosis / status / retry_count）。
    """
    idx = task["index"]

    whole_text = (
        "## 所属章节:\n\n" + task["toc_section"]
        + "\n\n## 关键词:\n\n" + ", ".join(task.get("keywords", []))
        + "\n\n## 具体内容:\n\n" + task["segment_text"]
    )

    input_snapshot = {
        "segment_text": task["segment_text"],
        "toc_section": task["toc_section"],
        "toc_level": task["toc_level"],
        "keywords": task.get("keywords", []),
        "source_file": source_file_name,
    }

    retry_count = 0
    last_error_msg = ""

    while True:
        if retry_count >= max_retries:
            error_info = {
                "index": idx,
                "input": input_snapshot,
                "status": "failed",
                "error": "达到最大重试次数",
                "retry_count": retry_count,
                "last_error": last_error_msg,
            }
            with file_lock:
                _update_json_file(output_file, str(idx), error_info)
            print(f"[Failed] 片段 {idx} 在重试 {max_retries} 次后放弃")
            return idx, "failed", None

        try:
            response = llm_func(prompt_func(whole_text))
            try:
                content = safe_parse_json_with_llm_repair(
                    response["content"], llm_func=llm_func
                )
            except Exception as json_err:
                raise Exception(
                    f"JSON解析失败（含LLM修复）: {json_err} | 原始内容: "
                    f"{str(response.get('content', 'N/A'))[:100]}"
                )

            result = {
                "index": idx,
                "input": input_snapshot,
                "Know_How": content.get("Know_How", ""),
                "Logic_Diagnosis": content.get("Logic_Diagnosis", ""),
                "status": "success",
                "retry_count": retry_count,
            }
            with file_lock:
                _update_json_file(output_file, str(idx), result)

            msg = f"片段 {idx} 完成"
            if retry_count > 0:
                msg += f"（历经 {retry_count} 次重试）"
            print(f"[Success] {msg}")
            return idx, "success", content.get("Know_How", "")

        except Exception:
            retry_count += 1
            last_error_msg = traceback.format_exc()
            if retry_count % 5 == 1:
                print(
                    f"[Error] 片段 {idx} 第 {retry_count} 次失败: "
                    f"{last_error_msg[:150]}..."
                )
            wait = min(2 ** (retry_count - 1), 60)
            time.sleep(wait)


# ─── 主入口 ───────────────────────────────────────────────────────────────


def run_doc_level1_extraction(
    doc_path: str,
    llm_func,
    prompt_func,
    output_file: str = "./output/doc_kh_level1.json",
    max_workers: int = None,
    max_retries: int = 100,
    min_seg_chars: int = 500,
    max_seg_chars: int = 2000,
    force_llm_toc: bool = True,
    llm_toc_workers: int = 8,
):
    """
    文档型 Level 1 入口（v2）：对文档切分后的每个片段独立进行 Know-How 抽取。

    输出格式与 QA Level 1 完全一致：
      {str(index): {index, input, Know_How, Logic_Diagnosis, status, retry_count}}

    Parameters
    ----------
    doc_path          : 文档文件路径（PDF/DOCX/TXT/PPTX）
    llm_func          : LLM 调用函数
    prompt_func       : 提炼 prompt 构造函数（如 doc_extract_v1）
    output_file       : JSON 输出路径（支持断点续传）
    max_workers       : 并发线程数（默认 CPU 核心数）
    max_retries       : 每片段最大重试次数
    min_seg_chars     : 段落合并下限
    max_seg_chars     : 段落合并上限
    force_llm_toc     : 强制使用 LLM 生成目录（跳过规则方式）
    llm_toc_workers   : LLM 目录摘要并发数
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"[Doc-Level-1-v2] 开始处理: {os.path.basename(doc_path)}")

    tasks, file_type, segment_content = build_segment_tasks(
        doc_path, llm_func, min_seg_chars, max_seg_chars,
        force_llm_toc, llm_toc_workers,
    )

    total = len(tasks)
    print(f"[Doc-Level-1-v2] 共 {total} 个片段任务，并发数: {max_workers}")

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 条记录，自动续传")
        except Exception:
            existing_data = {}

    pending = [
        task for task in tasks
        if str(task["index"]) not in existing_data
        or existing_data.get(str(task["index"]), {}).get("status") != "success"
    ]
    completed = total - len(pending)
    print(f"  已完成: {completed}, 待处理: {len(pending)}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    source_file_name = os.path.basename(doc_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_single_segment,
                task, llm_func, prompt_func, output_file,
                source_file_name, max_retries,
            ): task["index"]
            for task in pending
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    print(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")
            except Exception as e:
                print(f"  片段 {idx} 处理异常: {e}")

    print(f"[Doc-Level-1-v2] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 支持的文档扩展名 ─────────────────────────────────────────────────────
_SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".txt", ".pptx"}


# ─── 独立运行入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from llm_client import chat
    from prompts import doc_extract_v1

    sys.path.insert(0, _EXTRACTION_DIR)
    from utils import get_source_stem

    parser = argparse.ArgumentParser(
        description="文档知识抽取 v2 — Level 1（按片段独立抽取，仅一级提炼）"
    )
    parser.add_argument(
        "--files", "-f", nargs="+", default=None,
        help="指定要处理的文档文件路径（支持多个）；不指定则处理 input 目录下所有文件",
    )
    args = parser.parse_args()

    input_dir = os.path.join(_PACKAGE_DIR, "input")
    output_dir = os.path.join(_PACKAGE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.files:
        doc_files = []
        for fp in args.files:
            fp = os.path.abspath(fp)
            if not os.path.isfile(fp):
                print(f"[警告] 文件不存在，已跳过: {fp}")
                continue
            if os.path.splitext(fp)[1].lower() not in _SUPPORTED_DOC_EXTS:
                print(f"[警告] 不支持的文件类型，已跳过: {fp}（支持: {_SUPPORTED_DOC_EXTS}）")
                continue
            doc_files.append(fp)
        if not doc_files:
            raise FileNotFoundError("指定的文件中没有可处理的有效文件")
        mode_desc = "指定文件模式"
    else:
        doc_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_DOC_EXTS
        ])
        if not doc_files:
            raise FileNotFoundError(
                f"input 目录中未找到支持的文档文件（{_SUPPORTED_DOC_EXTS}）：{input_dir}"
            )
        mode_desc = "全量扫描模式"

    print("=" * 60)
    print(f"[doc_level1_extract v2] {mode_desc}，共 {len(doc_files)} 个源文档（仅一级提炼）")
    print("=" * 60)
    for i, fp in enumerate(doc_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    for idx, doc_path in enumerate(doc_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{len(doc_files)}] 处理: {os.path.basename(doc_path)}")
        print(f"{'═' * 60}")

        source_stem = get_source_stem(doc_path)
        output_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")

        run_doc_level1_extraction(
            doc_path=doc_path,
            llm_func=chat,
            prompt_func=doc_extract_v1,
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
    print(f"[doc_level1_extract v2] 全部 {len(doc_files)} 个文档处理完成！")
    print(f"{'═' * 60}")
