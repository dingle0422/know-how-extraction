"""
文档型一级提炼：解析 PDF 书籍目录，按章节并发调用 LLM，抽取泛化 Know-How 片段。
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


# ─── PDF 解析 ─────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> tuple:
    """
    解析 PDF，返回 (全文拼接文本, {页码: 页面文本} 字典)。
    页码从 1 开始计数。

    Parameters
    ----------
    pdf_path : PDF 文件路径

    Returns
    -------
    text         : 含页码标记的全文字符串，用于目录行识别
    page_content : {int: str}，每页的原始文本
    """
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
    """
    从全文文本中识别目录行，构建 {章节标题: (起始页, 结束页)} 字典。

    目录行通过 toc_marker（如连续省略号）定位。起止页范围与 notebook 原版保持一致：
    第 i 章的结束页 = 第 i+1 章的起始页（页面略有重叠，符合原始提取逻辑）。
    最后一章的结束页为 PDF 总页数。

    Parameters
    ----------
    text       : parse_pdf 返回的全文字符串
    page_content : parse_pdf 返回的页码字典（用于获取总页数）
    toc_marker : 识别目录行的标志字符串，默认为 "..........."

    Returns
    -------
    {str: (int, int)}  标题 -> (起始页, 结束页)
    """
    menu = [line for line in text.split("\n") if toc_marker in line]
    if not menu:
        return {}

    total_pages = max(page_content.keys()) if page_content else 1
    title_page = {}

    # 计算每个menu项的内容覆盖范围页
    for i in range(len(menu) - 1):
        title = menu[i].split("..")[0]
        try:
            start = int(re.sub(r"\D", "", menu[i][-10:]))
            end = int(re.sub(r"\D", "", menu[i + 1][-10:]))
            title_page[title] = (start, end)
        except (ValueError, IndexError):
            continue

    # 最后一个目录条目，结束页设为 PDF 末页
    last_title = menu[-1].split("..")[0]
    try:
        last_start = int(re.sub(r"\D", "", menu[-1][-10:]))
        title_page[last_title] = (last_start, total_pages)
    except ValueError:
        pass

    return title_page


# ─── 单章节处理（含重试）─────────────────────────────────────────────────────

def _process_single_chapter(
    title: str,
    page_range: tuple,
    page_content: dict,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 100,
):
    """
    处理单个章节：拼装文本 -> 调用 LLM -> 解析 JSON -> 写入输出文件。
    支持无限重试（直到 max_retries），每次失败后指数退避等待。
    """
    p_list = list(range(page_range[0], page_range[1] + 1))
    core_content = [page_content.get(p, "") for p in p_list]
    whole_text = (
        "## 内容标题:\n\n"
        + title
        + "\n\n## 具体内容:\n\n"
        + "\n\n".join(core_content)
    )

    retry_count = 0
    last_error_msg = ""

    while True:
        if retry_count >= max_retries:
            error_info = {
                "title": title,
                "page_range": list(page_range),
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
                "page_range": list(page_range),
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


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def run_doc_level1_extraction(
    pdf_path: str,
    llm_func,
    prompt_func,
    output_file: str = "./output/doc_kh_level1.json",
    toc_marker: str = "...........",
    max_workers: int = os.cpu_count() or 4,
    max_retries: int = 100,
    title_page: dict = None,
):
    """
    文档型多线程一级知识提炼入口。

    Parameters
    ----------
    pdf_path    : PDF 文件路径
    llm_func    : LLM 调用函数（如 chat 或 qwen），签名：str -> {"content": str, ...}
    prompt_func : 文档提炼 prompt 构造函数（如 doc_extract_v1），签名：str -> str
    output_file : JSON 输出路径（支持断点续传）
    toc_marker  : 目录行识别标志，默认 "..........."
    max_workers : 并发线程数
    max_retries : 每章最大重试次数
    title_page  : 可选，外部传入 {标题: (起始页, 结束页)} 以跳过 TOC 自动解析
    """
    print(f"[Doc-Level-1] 解析 PDF: {pdf_path}")
    text, page_content = parse_pdf(pdf_path)

    if title_page is None:
        title_page = parse_toc(text, page_content, toc_marker)

    total = len(title_page)
    print(f"[Doc-Level-1] 共 {total} 个章节，并发数: {max_workers}")

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 条记录，自动续传")
        except Exception:
            existing_data = {}

    pending = [
        (title, page_range)
        for title, page_range in title_page.items()
        if title not in existing_data
        or existing_data.get(title, {}).get("status") != "success"
    ]
    completed = total - len(pending)
    print(f"  已完成: {completed}, 待处理: {len(pending)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {
            executor.submit(
                _process_single_chapter,
                title, page_range, page_content, llm_func, prompt_func,
                output_file, max_retries,
            ): title
            for title, page_range in pending
        }
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    print(
                        f"  进度: {completed}/{total} "
                        f"({completed / total * 100:.1f}%)"
                    )
            except Exception as e:
                print(f"  章节 '{title}' 处理异常: {e}")

    print(f"[Doc-Level-1] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from llm_client import chat
    from prompts import doc_extract_v1

    print("=" * 60)
    print("[doc_level1_extract] 开始独立测试（真实 LLM 调用）")
    print("=" * 60)

    pdf_path = os.path.join(os.path.dirname(__file__), "input", "电商行业财税合规与实务指南.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"未找到测试 PDF：{pdf_path}\n"
            "请将 PDF 文件放入 input/ 目录后重新运行。"
        )

    # 解析 PDF 与目录
    print("\n[1/3] 解析 PDF 文档...")
    text, page_content = parse_pdf(pdf_path)
    title_page_full = parse_toc(text, page_content)
    print(f"  共解析到 {len(title_page_full)} 个章节")

    # 仅取前 3 个章节进行测试，避免全量运行耗时过长
    TEST_CHAPTERS = 10
    title_page_test = dict(list(title_page_full.items())[:TEST_CHAPTERS])
    print(f"\n[2/3] 本次仅测试前 {TEST_CHAPTERS} 个章节：")
    for t, pr in title_page_test.items():
        print(f"  - {t}  (页 {pr[0]}~{pr[1]})")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "doc_kh_level1_test.json")
    if os.path.exists(output_file):
        os.remove(output_file)

    print("\n[3/3] 开始提炼...")
    result = run_doc_level1_extraction(
        pdf_path=pdf_path,
        llm_func=chat,
        prompt_func=doc_extract_v1,
        output_file=output_file,
        max_workers=2,
        max_retries=3,
        title_page=title_page_test,
    )

    with open(result, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n测试结果预览（共 {len(data)} 个章节）:")
    for title, v in data.items():
        status = v.get("status")
        kh_preview = str(v.get("Know_How", ""))[:80]
        print(f"  [{status}] {title}\n    Know_How: {kh_preview}...")

    # 导出 Markdown 预览
    md_file = output_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for title, v in data.items():
            kh = v.get("Know_How", "").strip()
            if kh:
                f.write(f"<!-- {title} -->\n\n")
                f.write(kh)
                f.write("\n\n---\n\n")
    print(f"\nMarkdown 预览文件已导出：{md_file}")
    print("\n[doc_level1_extract] 测试完成！")
