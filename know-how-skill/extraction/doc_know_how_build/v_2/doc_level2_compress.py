"""
文档型完整流水线 v2（升级版）：Level 1（按片段）→ 废料分流 → 新聚类 → Level 2（压缩）→ Knowledge
======================================================================================
相比原版 v2 的改动：
  1. 聚类算法从 KMeansConstrained 替换为 AgglomerativeClustering（cosine 阈值）
  2. Level 1 结果中 Know_How 为空的片段按长度分流：
     - 原文 < min_case_chars 字 → 丢弃（标题/页码等结构性废料）
     - 原文 >= min_case_chars 字 → 写入废料备份库
  3. compression 逻辑保持不变（compression_v2 多主题合并）

独立运行：
  python doc_level2_compress.py              # 处理 input/ 下所有文档
  python doc_level2_compress.py -f a.pdf     # 处理指定文件
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

_V1_DIR = os.path.join(_PACKAGE_DIR, "v_1")
_QA_V2_DIR = os.path.join(_EXTRACTION_DIR, "qa_know_how_build", "v_2")

for _p in (_V_DIR, _V1_DIR, _QA_V2_DIR, _EXTRACTION_DIR, _SKILL_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# v2 目录强制置顶，确保优先于 v1 的同名模块
if _V_DIR in sys.path:
    sys.path.remove(_V_DIR)
sys.path.insert(0, _V_DIR)

from doc_level1_extract import run_doc_level1_extraction, _SUPPORTED_DOC_EXTS
from doc_structure_parse import parse_document
from prompts import safe_parse_json_with_llm_repair
from utils import get_source_stem, publish_to_knowledge
from clustering import make_clusters

_compress_lock = Lock()


# ─── Level 1 结果加载 + 废料分流 ─────────────────────────────────────────────

def _load_and_triage_level1(level1_file: str,
                            min_case_chars: int = 50) -> tuple[list[dict], list[dict]]:
    """
    加载 Level 1 结果，分流为：
      - valid_items : Know_How 非空的成功条目
      - waste_items : Know_How 为空但原文 >= min_case_chars 的片段（废料备份库）
    原文 < min_case_chars 的片段直接丢弃（标题/页码等结构性废料）。
    """
    with open(level1_file, "r", encoding="utf-8") as f:
        kh_data = json.load(f)

    valid_items = []
    waste_items = []
    discarded = 0

    for v in kh_data.values():
        index = v.get("index", -1)
        know_how = v.get("Know_How", "").strip()
        inp = v.get("input", {})
        segment_text = inp.get("segment", inp.get("question", ""))

        if v.get("status") == "success" and know_how:
            valid_items.append({
                "index": index,
                "Know_How": know_how,
                "input": inp,
            })
        elif len(segment_text) >= min_case_chars:
            waste_items.append({
                "index": index,
                "segment": segment_text[:2000],
                "toc_title": inp.get("toc_title", ""),
                "keywords": inp.get("keywords", ""),
                "reason": "Level 1 未提炼出可泛化的 Know-How",
            })
        else:
            discarded += 1

    valid_items.sort(key=lambda x: x["index"])
    print(f"[Level-2] 有效 Know_How: {len(valid_items)}, "
          f"废料备份: {len(waste_items)}, 丢弃(过短): {discarded}")
    return valid_items, waste_items


def _save_waste_backup(waste_items: list[dict], output_path: str,
                       source_file: str = ""):
    """将废料片段写入备份库。"""
    data = {
        "source_file": source_file,
        "description": "Level 1 未提炼出 Know-How 的文档片段备份（原文 >= 阈值字数）",
        "total_items": len(waste_items),
        "items": waste_items,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[废料备份库] 已写入 {len(waste_items)} 条: {output_path}")


# ─── Compression 处理（复用 v1 逻辑）──────────────────────────────────────────

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


def _format_batch_snippets(batch: list[dict]) -> str:
    parts = []
    for i, item in enumerate(batch, 1):
        parts.append(f"【片段 {i}】（原始 index: {item['index']}）\n{item['Know_How']}")
    return "\n\n---\n\n".join(parts)


def _process_compression_batch(
    batch_idx: int,
    batch: list[dict],
    total_batches: int,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 999,
    batch_keywords: list[str] | None = None,
    batch_cohesion: dict | None = None,
):
    retry_count = 0
    last_error_msg = ""

    while True:
        if retry_count > max_retries:
            error_result = {
                "batch_index": batch_idx,
                "source_indices": [item["index"] for item in batch],
                "item_count": len(batch),
                "batch_keywords": batch_keywords or [],
                "batch_cohesion": batch_cohesion,
                "status": "failed",
                "error": last_error_msg,
                "retry_count": retry_count,
            }
            with _compress_lock:
                _update_json_file(output_file, str(batch_idx), error_result)
            print(f"[Failed] 批次 {batch_idx} 超过最大重试次数 ({max_retries})")
            return batch_idx, "failed", None

        try:
            snippets_text = _format_batch_snippets(batch)
            response = llm_func(prompt_func(f=snippets_text))

            try:
                content = safe_parse_json_with_llm_repair(
                    response["content"], llm_func=llm_func
                )
            except Exception as json_err:
                raise Exception(f"JSON 解析失败（含LLM修复）: {json_err}")

            if "Final_Know_How" not in content:
                raise Exception("响应缺少必需字段 'Final_Know_How'")

            final_kh = content["Final_Know_How"]
            if isinstance(final_kh, str):
                final_kh = [final_kh] if final_kh.strip() else []
            if not isinstance(final_kh, list):
                raise Exception(
                    f"Final_Know_How 应为 list 类型，实际为 {type(final_kh).__name__}"
                )

            result = {
                "batch_index": batch_idx,
                "source_indices": [item["index"] for item in batch],
                "item_count": len(batch),
                "batch_keywords": batch_keywords or [],
                "batch_cohesion": batch_cohesion,
                "Synthesis_Summary": content.get("Synthesis_Summary", ""),
                "Final_Know_How": final_kh,
                "status": "success",
                "retry_count": retry_count,
            }
            with _compress_lock:
                _update_json_file(output_file, str(batch_idx), result)

            suffix = f"（历经 {retry_count} 次重试）" if retry_count > 0 else ""
            print(f"[Success] 批次 {batch_idx + 1}/{total_batches} 完成"
                  f"（{len(batch)} 条知识）{suffix}")
            return batch_idx, "success", result

        except Exception as e:
            retry_count += 1
            last_error_msg = str(e)
            print(f"[Error] 批次 {batch_idx} 第 {retry_count} 次失败: "
                  f"{last_error_msg[:150]}")
            time.sleep(3)


# ─── Level 2 入口（新聚类 + compression）──────────────────────────────────────

def run_level2_compression_v2(
    level1_file: str,
    llm_func,
    prompt_func,
    output_file: str,
    waste_backup_file: str = "",
    cosine_threshold: float = 0.75,
    min_case_chars: int = 50,
    max_workers: int = 16,
    source_file: str = "",
):
    """
    V2 二级知识压缩入口（AgglomerativeClustering + 废料分流）。

    Parameters
    ----------
    level1_file : 一级提炼结果 JSON 路径
    llm_func : LLM 调用函数
    prompt_func : 二级压缩 prompt 构造函数（如 compression_v2）
    output_file : 输出 JSON 路径
    waste_backup_file : 废料备份库 JSON 路径
    cosine_threshold : 聚类 cosine 相似度阈值
    min_case_chars : 废料备份最小字数阈值（低于此值直接丢弃）
    max_workers : 并发线程数
    source_file : 源文件名（用于备份库标注）
    """
    # ── 加载 + 分流 ──
    valid_items, waste_items = _load_and_triage_level1(
        level1_file, min_case_chars=min_case_chars
    )

    if waste_items and waste_backup_file:
        _save_waste_backup(waste_items, waste_backup_file, source_file=source_file)

    if not valid_items:
        print("[Level-2] 无有效 Know_How 可供压缩，流程结束")
        return output_file

    # ── 新聚类 ──
    clusters = make_clusters(valid_items, cosine_threshold=cosine_threshold)

    # ── 断点续传检查 ──
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 个批次记录，自动续传")
        except Exception:
            pass

    pending = [
        (i, c)
        for i, c in enumerate(clusters)
        if str(i) not in existing_data
        or existing_data.get(str(i), {}).get("status") != "success"
    ]
    completed = len(clusters) - len(pending)
    print(f"  总批次: {len(clusters)}，已完成: {completed}，"
          f"待处理: {len(pending)}，并发数: {max_workers}")

    # ── 多线程压缩 ──
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_compression_batch,
                i, c["items"], len(clusters), llm_func, prompt_func, output_file,
                batch_keywords=c.get("keywords"),
                batch_cohesion=c.get("cohesion"),
            ): i
            for i, c in pending
        }
        for future in as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    pct = completed / len(clusters) * 100
                    print(f"  进度: {completed}/{len(clusters)} ({pct:.1f}%)")
            except Exception as e:
                print(f"  批次 {batch_idx} 处理异常: {e}")

    print(f"[Level-2] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 单文件完整流水线 ─────────────────────────────────────────────────────────

def run_full_pipeline_for_doc(
    doc_path: str,
    llm_func,
    level1_prompt_func,
    level2_prompt_func,
    output_dir: str,
    knowledge_dir: str,
    cosine_threshold: float = 0.75,
    min_case_chars: int = 50,
    level1_max_workers: int = 4,
    level2_max_workers: int = 16,
    max_retries: int = 100,
    min_seg_chars: int = 500,
    max_seg_chars: int = 2000,
    force_llm_toc: bool = True,
    llm_toc_workers: int = 8,
):
    """
    对单个源文档执行完整的 Level 1 → 废料分流 → 新聚类 → Level 2 → Knowledge 流水线。
    """
    source_stem = get_source_stem(doc_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Level 1: 按片段独立提炼 ──
    level1_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
    structure_file = os.path.join(output_dir, f"{source_stem}_structure.json")
    print(f"\n  [Level 1] 按片段独立提炼: {os.path.basename(doc_path)}")
    run_doc_level1_extraction(
        doc_path=doc_path,
        llm_func=llm_func,
        prompt_func=level1_prompt_func,
        output_file=level1_file,
        structure_file=structure_file,
        max_workers=level1_max_workers,
        max_retries=max_retries,
        min_seg_chars=min_seg_chars,
        max_seg_chars=max_seg_chars,
        force_llm_toc=force_llm_toc,
        llm_toc_workers=llm_toc_workers,
    )

    # ── Level 1 Markdown 预览 ──
    with open(level1_file, encoding="utf-8") as f:
        l1_data = json.load(f)
    md_file = level1_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for k, v in sorted(l1_data.items(), key=lambda x: int(x[0])):
            kh = v.get("Know_How", "").strip()
            if kh:
                f.write(kh)
                f.write("\n\n---\n\n")

    # ── Level 2: 废料分流 + 新聚类 + 压缩 ──
    level2_file = os.path.join(output_dir, f"{source_stem}_level2_compression.json")
    waste_file = os.path.join(output_dir, f"{source_stem}_waste_backup.json")
    print(f"\n  [Level 2] 废料分流 + 新聚类压缩: {os.path.basename(doc_path)}")
    result = run_level2_compression_v2(
        level1_file=level1_file,
        llm_func=llm_func,
        prompt_func=level2_prompt_func,
        output_file=level2_file,
        waste_backup_file=waste_file,
        cosine_threshold=cosine_threshold,
        min_case_chars=min_case_chars,
        max_workers=level2_max_workers,
        source_file=os.path.basename(doc_path),
    )

    # ── Level 2 Markdown 预览 ──
    with open(result, encoding="utf-8") as f:
        l2_data = json.load(f)
    md_file = level2_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for k, v in sorted(l2_data.items(), key=lambda x: int(x[0])):
            kh = v.get("Final_Know_How", [])
            if isinstance(kh, str):
                kh = [kh]
            for topic in kh:
                topic = topic.strip()
                if topic:
                    f.write(topic)
                    f.write("\n\n---\n\n")
    print(f"  Markdown 预览文件已导出: {md_file}")

    # ── 发布到 knowledge 目录 ──
    knowledge_sub = os.path.join(knowledge_dir, f"{source_stem}_knowledge")
    knowledge_json = os.path.join(knowledge_sub, "knowledge.json")
    knowledge_md = os.path.join(knowledge_sub, "knowledge.md")
    if os.path.exists(knowledge_json) and os.path.exists(knowledge_md):
        print(f"  [跳过] Knowledge 目录已存在: {knowledge_sub}")
    else:
        print(f"\n  [Knowledge] 发布到 knowledge 目录...")
        full_text, _, _ = parse_document(doc_path)
        source_text_head = full_text[:20000]
        publish_to_knowledge(
            source_stem=source_stem,
            final_json_path=level2_file,
            knowledge_base_dir=knowledge_dir,
            llm_func=llm_func,
            source_text_head=source_text_head,
            level1_json_path=level1_file,
        )

    # 复制废料备份到 knowledge 目录
    if os.path.exists(waste_file):
        import shutil
        dst = os.path.join(knowledge_sub, "waste_backup.json")
        os.makedirs(knowledge_sub, exist_ok=True)
        shutil.copy2(waste_file, dst)
        print(f"  [Knowledge] 废料备份库已复制到: {dst}")

    return level2_file


# ─── 独立运行入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from llm_client import chat
    from prompts import doc_extract_v1, compression_v2

    parser = argparse.ArgumentParser(
        description="文档知识抽取 v2（升级版）— Level 1 → 废料分流 → 新聚类 → Level 2 → Knowledge"
    )
    parser.add_argument(
        "--files", "-f", nargs="+", default=None,
        help="指定要处理的文档文件路径（支持多个）；不指定则处理 input 目录下所有文件",
    )
    parser.add_argument(
        "--cosine-threshold", "-t", type=float, default=0.60,
        help="聚类 cosine 相似度阈值 (默认 0.60)",
    )
    parser.add_argument(
        "--min-case-chars", type=int, default=50,
        help="废料备份最小字数阈值，低于此值直接丢弃 (默认 50)",
    )
    args = parser.parse_args()

    input_dir = os.path.join(_PACKAGE_DIR, "input")
    output_dir = os.path.join(_PACKAGE_DIR, "output")
    knowledge_dir = os.path.join(_PACKAGE_DIR, "knowledge")

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
    print(f"[doc v2 升级版] {mode_desc}，共 {len(doc_files)} 个源文档")
    print(f"  cosine_threshold={args.cosine_threshold}, "
          f"min_case_chars={args.min_case_chars}")
    print("=" * 60)
    for i, fp in enumerate(doc_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    for idx, doc_path in enumerate(doc_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{len(doc_files)}] 处理: {os.path.basename(doc_path)}")
        print(f"{'═' * 60}")

        run_full_pipeline_for_doc(
            doc_path=doc_path,
            llm_func=chat,
            level1_prompt_func=doc_extract_v1,
            level2_prompt_func=compression_v2,
            output_dir=output_dir,
            knowledge_dir=knowledge_dir,
            cosine_threshold=args.cosine_threshold,
            min_case_chars=args.min_case_chars,
            level1_max_workers=os.cpu_count() or 4,
            level2_max_workers=os.cpu_count() or 4,
            max_retries=100,
        )

    print(f"\n{'═' * 60}")
    print(f"[doc v2 升级版] 全部 {len(doc_files)} 个文档处理完成！")
    print(f"{'═' * 60}")
