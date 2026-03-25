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
            └── *_traceback.json     (一级抽取结果，含源数据回溯信息，可选)

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
        dst_traceback = os.path.join(sub_dir, f"{source_stem}_traceback.json")
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

    return sub_dir
