"""
Extraction 公共工具
==================
提供源文件名解析、knowledge 目录发布等跨模块复用功能。
"""

import json
import os
import shutil


def get_source_stem(filepath: str) -> str:
    """从文件路径提取不含扩展名的文件名（源文件名前缀）。"""
    return os.path.splitext(os.path.basename(filepath))[0]


def publish_to_knowledge(
    source_stem: str,
    final_json_path: str,
    knowledge_base_dir: str,
    llm_func,
    source_text_head: str,
    level1_json_path: str = None,
    knowledge_desc_prompt_func=None,
):
    """
    将最终抽取结果发布到 knowledge 子文件夹。

    目录结构:
        {knowledge_base_dir}/{source_stem}_knowledge/
            ├── knowledge.json       (最终抽取结果副本)
            ├── knowledge.md         (LLM 基于源文档头部生成的知识概述)
            └── knowledge_traceback.json     (一级抽取结果，含源数据回溯信息，可选)

    Parameters
    ----------
    source_stem            : 源文件名（不含扩展名）
    final_json_path        : 最终抽取结果 JSON 路径
    knowledge_base_dir     : knowledge 根目录（如 .../doc_know_how_build/knowledge）
    llm_func               : LLM 调用函数
    source_text_head       : 源文档头部文本（建议取前 2 万字）
    level1_json_path       : 一级抽取结果 JSON 路径（含源数据回溯），可选
    knowledge_desc_prompt_func : 生成 knowledge.md 的 prompt 函数
    """
    from prompts import knowledge_description_prompt as _default_prompt

    if knowledge_desc_prompt_func is None:
        knowledge_desc_prompt_func = _default_prompt

    sub_dir = os.path.join(knowledge_base_dir, f"{source_stem}_knowledge")
    os.makedirs(sub_dir, exist_ok=True)

    dst_json = os.path.join(sub_dir, "knowledge.json")
    shutil.copy2(final_json_path, dst_json)
    print(f"[Knowledge] 最终结果已复制到: {dst_json}")

    if level1_json_path and os.path.exists(level1_json_path):
        dst_traceback = os.path.join(sub_dir, "knowledge_traceback.json")
        shutil.copy2(level1_json_path, dst_traceback)
        print(f"[Knowledge] 一级回溯文件已复制到: {dst_traceback}")

    print("[Knowledge] 正在生成 knowledge.md（基于源文档头部摘要）...")
    prompt = knowledge_desc_prompt_func(source_text_head)
    try:
        response = llm_func(prompt)
        description = response.get("content", "") if isinstance(response, dict) else str(response)
    except Exception as e:
        description = f"（knowledge.md 自动生成失败: {e}）"
        print(f"[Knowledge] 警告: knowledge.md 生成失败: {e}")

    md_path = os.path.join(sub_dir, "knowledge.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(description)
    print(f"[Knowledge] 知识描述文件已生成: {md_path}")

    append_knowhow_content_to_md(dst_json, md_path)

    return sub_dir


# ─── Know-How 内容追加到 knowledge.md ─────────────────────────────────────────

def _render_structured_kh(kh: dict) -> str:
    """将 QA v2 的结构化 JSON know-how 渲染为可读文本。"""
    lines = []
    lines.append(f"### {kh.get('title', '未命名')}")
    if kh.get("scope"):
        lines.append(f"**适用场景**: {kh['scope']}")
    lines.append("")

    steps = kh.get("steps", [])
    if steps:
        lines.append("**操作步骤**:")
        for s in steps:
            line = f"  {s.get('step', '?')}. {s.get('action', '')}"
            if s.get("condition"):
                line += f" （条件: {s['condition']}）"
            if s.get("outcome"):
                line += f" → {s['outcome']}"
            lines.append(line)
        lines.append("")

    exceptions = kh.get("exceptions", [])
    if exceptions:
        lines.append("**例外情况**:")
        for ex in exceptions:
            lines.append(f"  - 当 {ex.get('when', '?')} → {ex.get('then', '?')}")
        lines.append("")

    constraints = kh.get("constraints", [])
    if constraints:
        lines.append("**约束/依据**:")
        for c in constraints:
            lines.append(f"  - {c}")
        lines.append("")

    return "\n".join(lines)


def append_knowhow_content_to_md(knowledge_json_path: str, knowledge_md_path: str):
    """
    读取 knowledge.json，将所有 know-how 内容转为文本追加到 knowledge.md 末尾，
    并在最后添加总字数统计。

    自动检测两种格式：
      - Doc compression 格式: Final_Know_How (list[str])
      - QA v2 结构化格式: know_how (dict with title/steps/...)
    """
    try:
        with open(knowledge_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    kh_texts = []

    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = data[key]
        if entry.get("status") != "success":
            continue

        # QA v2 结构化格式
        if "know_how" in entry and isinstance(entry["know_how"], dict):
            rendered = _render_structured_kh(entry["know_how"])
            if rendered.strip():
                kh_texts.append(rendered)

        # Doc compression 格式
        elif "Final_Know_How" in entry:
            fkh = entry["Final_Know_How"]
            if isinstance(fkh, str):
                fkh = [fkh]
            if isinstance(fkh, list):
                for topic in fkh:
                    topic = topic.strip()
                    if topic:
                        kh_texts.append(topic)

    if not kh_texts:
        return

    total_content = "\n\n---\n\n".join(kh_texts)
    total_chars = len(total_content)

    with open(knowledge_md_path, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write("# Know-How 全文内容\n\n")
        f.write(total_content)
        f.write("\n\n---\n\n")
        f.write(f"> **Know-How 总字数: {total_chars:,} 字 | 共 {len(kh_texts)} 条知识节点**\n")

    print(f"[Knowledge] Know-How 全文已追加到 knowledge.md "
          f"({len(kh_texts)} 条, {total_chars:,} 字)")
