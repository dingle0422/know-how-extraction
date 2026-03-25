"""
文档型一级提炼（Layer 1）
=========================
基于 Layer 0（doc_structure_parse）生成的 DocStructure，按目录-内容映射关系
并发调用 LLM，抽取泛化 Know-How 片段。

支持两种输入模式：
  A. 接收 DocStructure dict / JSON 文件（推荐，走完整 2 层流水线）
  B. 直接传入 PDF + title_page（向后兼容旧接口）

支持多线程并发 + 断点续传 + 指数退避重试。
"""

import os
import re
import sys
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from prompts import safe_parse_json_with_llm_repair
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from prompts import safe_parse_json_with_llm_repair

file_lock = Lock()


# ─── 文件辅助 ─────────────────────────────────────────────────────────────────

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


# ─── 从 DocStructure 构建章节任务 ────────────────────────────────────────────

def _build_tasks_from_doc_structure(doc_structure: dict) -> list[dict]:
    """
    从 Layer 0 的 DocStructure 构建 Layer 1 的章节任务列表。

    按 toc_section 聚合 paragraphs，形成"目录节→内容"映射，
    每个目录节的内容 = 该节下所有段落文本拼接。

    Returns
    -------
    [{title, content, toc_level, keywords, seg_indices}]
    """
    toc_items = doc_structure.get("toc", [])
    paragraphs = doc_structure.get("paragraphs", [])

    toc_kw_map = {item["title"]: item.get("keywords", []) for item in toc_items}
    toc_level_map = {item["title"]: item.get("level", 1) for item in toc_items}

    section_paras: dict[str, list[dict]] = defaultdict(list)
    for p in paragraphs:
        section_paras[p["toc_section"]].append(p)

    tasks = []
    for toc_item in toc_items:
        title = toc_item["title"]
        paras = section_paras.get(title, [])
        if not paras:
            continue

        content = "\n\n".join(p["text"] for p in paras)
        if not content.strip():
            continue

        tasks.append({
            "title": title,
            "content": content,
            "toc_level": toc_level_map.get(title, 1),
            "keywords": toc_kw_map.get(title, []),
            "seg_indices": [p["idx"] for p in paras],
        })

    return tasks


# ─── 旧版 PDF 解析（向后兼容）────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> tuple:
    """解析 PDF，返回 (全文拼接文本, {页码: 页面文本})。向后兼容接口。"""
    if pdfplumber is None:
        raise ImportError("pdfplumber 未安装，请运行: pip install pdfplumber")

    text = ""
    page_content = defaultdict(str)
    with pdfplumber.open(pdf_path) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text += f"page:{p}\n\n{page_text}\n" + "—" * 50 + "\n"
            page_content[p] = page_text
    return text, dict(page_content)


def parse_toc(text: str, page_content: dict, toc_marker: str = "...........") -> dict:
    """从全文文本中识别目录行，构建 {章节标题: (起始页, 结束页)}。向后兼容接口。"""
    menu = [line for line in text.split("\n") if toc_marker in line]
    if not menu:
        return {}

    total_pages = max(page_content.keys()) if page_content else 1
    title_page = {}

    for i in range(len(menu) - 1):
        title = menu[i].split("..")[0]
        try:
            start = int(re.sub(r"\D", "", menu[i][-10:]))
            end = int(re.sub(r"\D", "", menu[i + 1][-10:]))
            title_page[title] = (start, end)
        except (ValueError, IndexError):
            continue

    last_title = menu[-1].split("..")[0]
    try:
        last_start = int(re.sub(r"\D", "", menu[-1][-10:]))
        title_page[last_title] = (last_start, total_pages)
    except ValueError:
        pass

    return title_page


# ─── 单任务处理（含重试）─────────────────────────────────────────────────────

def _process_single_task(
    task: dict,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 100,
):
    """
    处理单个章节任务：构造输入 → 调用 LLM → 解析 JSON → 写入输出文件。
    task 结构：{title, content, toc_level, keywords, group_ids} 或旧格式。
    """
    title = task["title"]

    if "content" in task:
        whole_text = (
            "## 内容标题:\n\n" + title
            + "\n\n## 关键词:\n\n" + ", ".join(task.get("keywords", []))
            + "\n\n## 具体内容:\n\n" + task["content"]
        )
        extra_meta = {
            "toc_level": task.get("toc_level", 1),
            "keywords": task.get("keywords", []),
            "seg_indices": task.get("seg_indices", []),
        }
    else:
        page_range = task["page_range"]
        page_content = task["page_content"]
        p_list = list(range(page_range[0], page_range[1] + 1))
        core_content = [page_content.get(p, "") for p in p_list]
        whole_text = (
            "## 内容标题:\n\n" + title
            + "\n\n## 具体内容:\n\n" + "\n\n".join(core_content)
        )
        extra_meta = {"page_range": list(page_range)}

    retry_count = 0
    last_error_msg = ""

    while True:
        if retry_count >= max_retries:
            error_info = {
                "title": title,
                **extra_meta,
                "status": "failed",
                "error": "达到最大重试次数",
                "retry_count": retry_count,
                "last_error": last_error_msg,
            }
            with file_lock:
                _update_json_file(output_file, title, error_info)
            print(f"[Failed] 章节 '{title}' 在重试 {max_retries} 次后放弃")
            return title, "failed", None

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
                "title": title,
                **extra_meta,
                "Logic_Diagnosis": content.get("Logic_Diagnosis", ""),
                "Know_How": content.get("Know_How", ""),
                "status": "success",
                "retry_count": retry_count,
            }
            with file_lock:
                _update_json_file(output_file, title, result)

            msg = f"章节 '{title}' 完成"
            if retry_count > 0:
                msg += f"（历经 {retry_count} 次重试）"
            print(f"[Success] {msg}")
            return title, "success", content.get("Know_How", "")

        except Exception:
            retry_count += 1
            last_error_msg = traceback.format_exc()
            if retry_count % 5 == 1:
                print(
                    f"[Error] 章节 '{title}' 第 {retry_count} 次失败: "
                    f"{last_error_msg[:150]}..."
                )
            wait = min(2 ** (retry_count - 1), 60)
            time.sleep(wait)


# ─── 主入口（新：基于 DocStructure）──────────────────────────────────────────

def run_doc_level1_extraction(
    llm_func,
    prompt_func,
    doc_structure: dict = None,
    doc_structure_file: str = None,
    output_file: str = "./output/doc_kh_level1.json",
    max_workers: int = None,
    max_retries: int = 100,
):
    """
    Layer 1 主入口（推荐）：基于 DocStructure 进行知识提炼。

    Parameters
    ----------
    llm_func           : LLM 调用函数
    prompt_func        : 文档提炼 prompt 构造函数（如 doc_extract_v1）
    doc_structure      : Layer 0 输出的 DocStructure dict
    doc_structure_file : 或者传入 DocStructure JSON 文件路径
    output_file        : JSON 输出路径（支持断点续传）
    max_workers        : 并发线程数（默认 CPU 核心数）
    max_retries        : 每章最大重试次数
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    if doc_structure is None and doc_structure_file:
        with open(doc_structure_file, "r", encoding="utf-8") as f:
            doc_structure = json.load(f)

    if doc_structure is None:
        raise ValueError("必须提供 doc_structure 或 doc_structure_file")

    tasks = _build_tasks_from_doc_structure(doc_structure)
    total = len(tasks)
    print(f"[Layer-1] 基于 DocStructure 构建 {total} 个章节任务，并发数: {max_workers}")

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
        if task["title"] not in existing_data
        or existing_data.get(task["title"], {}).get("status") != "success"
    ]
    completed = total - len(pending)
    print(f"  已完成: {completed}, 待处理: {len(pending)}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {
            executor.submit(
                _process_single_task,
                task, llm_func, prompt_func, output_file, max_retries,
            ): task["title"]
            for task in pending
        }
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    print(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")
            except Exception as e:
                print(f"  章节 '{title}' 处理异常: {e}")

    print(f"[Layer-1] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 向后兼容入口（旧：直接传 PDF）─────────────────────────────────────────

def run_doc_level1_extraction_legacy(
    pdf_path: str,
    llm_func,
    prompt_func,
    output_file: str = "./output/doc_kh_level1.json",
    toc_marker: str = "...........",
    max_workers: int = None,
    max_retries: int = 100,
    title_page: dict = None,
):
    """
    向后兼容入口：直接传入 PDF 路径，内部解析 TOC 并执行提炼。
    新项目建议使用 run_doc_level1_extraction + DocStructure。
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"[Doc-Level-1-Legacy] 解析 PDF: {pdf_path}")
    text, page_content = parse_pdf(pdf_path)

    if title_page is None:
        title_page = parse_toc(text, page_content, toc_marker)

    total = len(title_page)
    print(f"[Doc-Level-1-Legacy] 共 {total} 个章节，并发数: {max_workers}")

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 条记录，自动续传")
        except Exception:
            existing_data = {}

    pending = [
        {"title": title, "page_range": page_range, "page_content": page_content}
        for title, page_range in title_page.items()
        if title not in existing_data
        or existing_data.get(title, {}).get("status") != "success"
    ]
    completed = total - len(pending)
    print(f"  已完成: {completed}, 待处理: {len(pending)}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {
            executor.submit(
                _process_single_task,
                task, llm_func, prompt_func, output_file, max_retries,
            ): task["title"]
            for task in pending
        }
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    print(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")
            except Exception as e:
                print(f"  章节 '{title}' 处理异常: {e}")

    print(f"[Doc-Level-1-Legacy] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 支持的文档扩展名 ─────────────────────────────────────────────────────
_SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".txt", ".pptx"}


# ─── 单文件完整流水线 ─────────────────────────────────────────────────────
def run_full_pipeline_for_doc(
    doc_path: str,
    llm_func,
    prompt_func,
    output_dir: str,
    knowledge_dir: str,
    force_llm_toc: bool = True,
    max_workers: int = 2,
    max_retries: int = 100,
):
    """
    对单个源文档执行完整的 Layer 0 → Layer 1 → Knowledge 发布流水线。

    若中间产物已存在，自动跳过对应阶段（断点续传由各子函数内部处理）。
    """
    from doc_structure_parse import run_doc_structure_parse, parse_document
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_source_stem, publish_to_knowledge

    source_stem = get_source_stem(doc_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Layer 0: 文档结构化解析 ──
    structure_file = os.path.join(output_dir, f"{source_stem}_structure.json")
    if os.path.exists(structure_file):
        print(f"  [跳过] Layer 0 结构化文件已存在: {structure_file}")
        with open(structure_file, "r", encoding="utf-8") as f:
            doc_structure = json.load(f)
    else:
        print(f"\n  [Layer 0] 文档结构化解析: {os.path.basename(doc_path)}")
        doc_structure = run_doc_structure_parse(
            doc_path=doc_path,
            llm_func=llm_func,
            output_file=structure_file,
            force_llm_toc=force_llm_toc,
        )

    # ── Layer 1: 知识提炼（断点续传由 run_doc_level1_extraction 内部处理）──
    output_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
    print(f"\n  [Layer 1] Know-How 提炼: {os.path.basename(doc_path)}")
    result = run_doc_level1_extraction(
        llm_func=llm_func,
        prompt_func=prompt_func,
        doc_structure=doc_structure,
        output_file=output_file,
        max_workers=max_workers,
        max_retries=max_retries,
    )

    # ── 导出 Markdown 预览 ──
    with open(result, encoding="utf-8") as f:
        data = json.load(f)

    md_file = output_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for title, v in data.items():
            kh = v.get("Know_How", "").strip()
            if kh:
                f.write(f"<!-- {title} -->\n\n")
                f.write(kh)
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
            final_json_path=output_file,
            knowledge_base_dir=knowledge_dir,
            llm_func=llm_func,
            source_text_head=source_text_head,
            level1_json_path=output_file,
        )

    return output_file


# ─── 独立运行入口：扫描 input 文件夹全部源文件 ──────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from llm_client import chat
    from prompts import doc_extract_v1

    input_dir = os.path.join(os.path.dirname(__file__), "input")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")

    doc_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _SUPPORTED_DOC_EXTS
    ])

    if not doc_files:
        raise FileNotFoundError(
            f"input 目录中未找到支持的文档文件（{_SUPPORTED_DOC_EXTS}）：{input_dir}"
        )

    print("=" * 60)
    print(f"[doc_level1_extract] 扫描到 {len(doc_files)} 个源文档，开始批量流水线")
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
            prompt_func=doc_extract_v1,
            output_dir=output_dir,
            knowledge_dir=knowledge_dir,
            force_llm_toc=True,
            max_workers=2,
            max_retries=100,
        )

    print(f"\n{'═' * 60}")
    print(f"[doc_level1_extract] 全部 {len(doc_files)} 个文档处理完成！")
    print(f"{'═' * 60}")
