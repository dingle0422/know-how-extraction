"""
文档型完整流水线 v2：Level 1（按片段）→ Level 2（压缩）→ Knowledge
================================================================
复用 QA 的 Level 2 压缩逻辑（TF-IDF 语义聚类 + 批次压缩），
确保最终输出结构与 QA 完全一致。

独立运行：
  python doc_level2_compress.py              # 处理 input/ 下所有文档
  python doc_level2_compress.py -f a.pdf     # 处理指定文件
"""

import os
import sys
import json

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

if _V_DIR not in sys.path:
    sys.path.insert(0, _V_DIR)
if _SKILL_ROOT not in sys.path:
    sys.path.insert(0, _SKILL_ROOT)

_QA_V1_DIR = os.path.join(_EXTRACTION_DIR, "qa_know_how_build", "v_1")
if _QA_V1_DIR not in sys.path:
    sys.path.insert(0, _QA_V1_DIR)

_V1_DIR = os.path.join(_PACKAGE_DIR, "v_1")
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from doc_level1_extract import run_doc_level1_extraction, _SUPPORTED_DOC_EXTS
from level2_compress import run_level2_compression
from doc_structure_parse import parse_document

sys.path.insert(0, _EXTRACTION_DIR)
from utils import get_source_stem, publish_to_knowledge


# ─── 单文件完整流水线 ─────────────────────────────────────────────────────


def run_full_pipeline_for_doc(
    doc_path: str,
    llm_func,
    level1_prompt_func,
    level2_prompt_func,
    output_dir: str,
    knowledge_dir: str,
    level1_max_workers: int = 4,
    level2_max_workers: int = 16,
    level2_batch_size: int = 5,
    max_retries: int = 100,
    min_seg_chars: int = 500,
    max_seg_chars: int = 2000,
    force_llm_toc: bool = True,
    llm_toc_workers: int = 8,
):
    """
    对单个源文档执行完整的 Level 1 → Level 2 → Knowledge 发布流水线。

    输出格式与 QA 完全一致：
      - Level 1: {str(index): {index, input, Know_How, Logic_Diagnosis, status, ...}}
      - Level 2: {str(batch): {batch_index, source_indices, Final_Know_How, ...}}
    """
    source_stem = get_source_stem(doc_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Level 1: 按片段独立提炼 ──
    level1_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
    print(f"\n  [Level 1] 按片段独立提炼: {os.path.basename(doc_path)}")
    run_doc_level1_extraction(
        doc_path=doc_path,
        llm_func=llm_func,
        prompt_func=level1_prompt_func,
        output_file=level1_file,
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

    # ── Level 2: 二级压缩（复用 QA 的 TF-IDF 聚类 + 批次压缩） ──
    level2_file = os.path.join(output_dir, f"{source_stem}_level2_compression.json")
    print(f"\n  [Level 2] 二级知识压缩: {os.path.basename(doc_path)}")
    result = run_level2_compression(
        level1_file=level1_file,
        llm_func=llm_func,
        prompt_func=level2_prompt_func,
        output_file=level2_file,
        batch_size=level2_batch_size,
        max_workers=level2_max_workers,
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

    return level2_file


# ─── 独立运行入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from llm_client import chat
    from prompts import doc_extract_v1, compression_v2

    parser = argparse.ArgumentParser(
        description="文档知识抽取 v2 — 完整流水线（Level 1 → Level 2 → Knowledge）"
    )
    parser.add_argument(
        "--files", "-f", nargs="+", default=None,
        help="指定要处理的文档文件路径（支持多个）；不指定则处理 input 目录下所有文件",
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
    print(f"[doc_level2_compress v2] {mode_desc}，共 {len(doc_files)} 个源文档")
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
            level1_max_workers=os.cpu_count() or 4,
            level2_max_workers=os.cpu_count() or 4,
            level2_batch_size=5,
            max_retries=100,
        )

    print(f"\n{'═' * 60}")
    print(f"[doc_level2_compress v2] 全部 {len(doc_files)} 个文档处理完成！")
    print(f"{'═' * 60}")
