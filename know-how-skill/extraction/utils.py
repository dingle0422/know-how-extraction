"""
Extraction 公共工具
==================
提供源文件名解析、knowledge 目录发布、检索索引构建等跨模块复用功能。
"""

import json
import os
import shutil
from datetime import datetime


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
            condition_prefix = ""
            if s.get("condition"):
                condition_prefix = f"【触发条件：{s['condition']}】 → "
            line = f"  {s.get('step', '?')}. {condition_prefix}{s.get('action', '')}"
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


# ─── 检索索引构建 ────────────────────────────────────────────────────────────

def _extract_retrieval_text(entry: dict) -> str:
    """从 knowledge.json 的单条记录中提取用于检索向量化的文本。

    自动适配两种知识格式：
      - QA v2 结构化: know_how dict → 拼接 title/scope/steps/exceptions/constraints
      - Doc compression: Final_Know_How list[str] → 拼接所有主题文本
    """
    # QA v2 结构化格式
    if "know_how" in entry and isinstance(entry["know_how"], dict):
        kh = entry["know_how"]
        parts = []
        if kh.get("title"):
            parts.append(kh["title"])
        if kh.get("scope"):
            parts.append(kh["scope"])
        for s in kh.get("steps", []):
            if s.get("action"):
                parts.append(s["action"])
            if s.get("condition"):
                parts.append(s["condition"])
            if s.get("outcome"):
                parts.append(s["outcome"])
        for ex in kh.get("exceptions", []):
            if ex.get("when"):
                parts.append(ex["when"])
            if ex.get("then"):
                parts.append(ex["then"])
        for c in kh.get("constraints", []):
            if isinstance(c, str):
                parts.append(c)
        return " ".join(parts)

    # Doc compression 格式
    if "Final_Know_How" in entry:
        fkh = entry["Final_Know_How"]
        if isinstance(fkh, str):
            fkh = [fkh]
        if isinstance(fkh, list):
            return " ".join(t.strip() for t in fkh if t.strip())

    return ""


def _detect_knowledge_type(data: dict) -> str:
    """检测 knowledge.json 的知识类型。"""
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("status") == "success":
            if "know_how" in entry and isinstance(entry["know_how"], dict):
                return "qa_v2"
            if "Final_Know_How" in entry:
                return "doc_v2"
    return "unknown"


def build_retrieval_index(
    knowledge_json_path: str,
    knowledge_dir: str,
    embedding_func=None,
) -> str | None:
    """
    为 knowledge 目录构建检索索引文件 retrieval_index.json。

    对 knowledge.json 中每个成功的知识条目：
      1. 提取检索用文本
      2. 构建 TF-IDF 向量（保存 vocabulary + IDF 权重 + 条目向量）
      3. 调用 embedding 服务构建 Dense 向量（可选，服务不可用时跳过）

    Parameters
    ----------
    knowledge_json_path : knowledge.json 文件路径
    knowledge_dir       : knowledge 子目录路径（索引文件保存位置）
    embedding_func      : Dense embedding 函数，签名 (texts: list[str]) -> list[list[float]]；
                          为 None 或调用失败时仅保存 TF-IDF 索引

    Returns
    -------
    retrieval_index.json 的路径，构建失败返回 None
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        with open(knowledge_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[RetrievalIndex] 无法加载 knowledge.json: {e}")
        return None

    knowledge_type = _detect_knowledge_type(data)
    if knowledge_type == "unknown":
        print("[RetrievalIndex] 无法识别 knowledge.json 格式，跳过索引构建")
        return None

    # ── 提取检索文本 ──────────────────────────────────────────────────────
    entry_keys = []
    retrieval_texts = []
    entry_meta = {}

    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = data[key]
        if not isinstance(entry, dict) or entry.get("status") != "success":
            continue
        text = _extract_retrieval_text(entry)
        if not text.strip():
            continue

        entry_keys.append(key)
        retrieval_texts.append(text)

        meta = {"retrieval_text": text}
        if knowledge_type == "qa_v2":
            kh = entry.get("know_how", {})
            meta["title"] = kh.get("title", "")
            meta["scope"] = kh.get("scope", "")
        meta["keywords"] = entry.get("cluster_keywords", entry.get("batch_keywords", []))
        entry_meta[key] = meta

    if not retrieval_texts:
        print("[RetrievalIndex] 无有效知识条目，跳过索引构建")
        return None

    print(f"[RetrievalIndex] 检测到 {knowledge_type} 格式，"
          f"共 {len(retrieval_texts)} 条有效知识条目")

    # ── TF-IDF 向量化 ────────────────────────────────────────────────────
    try:
        import jieba
        jieba.setLogLevel(20)

        _stopwords = {
            "的", "了", "在", "是", "有", "和", "就", "不", "都", "一", "也", "很",
            "到", "说", "要", "会", "着", "看", "好", "上", "去", "来", "过", "把",
            "与", "及", "并", "或", "等", "中", "其", "该", "此", "以", "为", "从",
            "由", "被", "让", "使", "于", "对", "将", "已", "可", "能", "时", "后",
            "前", "这", "那", "个", "这个", "那个", "这些", "那些", "什么", "怎么",
            "因为", "所以", "但是", "如果", "可以", "应该", "需要", "已经", "通过",
            "进行", "相关", "包括", "属于", "具有", "情况", "方面", "问题", "方式",
            "他们", "我们", "我", "你", "他", "她", "它", "您", "自己",
        }

        def _tokenizer(text: str) -> list[str]:
            return [
                tok.strip() for tok in jieba.cut(text)
                if len(tok.strip()) >= 2 and tok.strip() not in _stopwords
            ]

        vectorizer = TfidfVectorizer(
            tokenizer=_tokenizer, max_features=512, token_pattern=None,
        )
        tfidf_tokenizer_type = "jieba"
    except ImportError:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=512,
        )
        tfidf_tokenizer_type = "char_wb"

    X = vectorizer.fit_transform(retrieval_texts).toarray()
    vocabulary = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
    idf = vectorizer.idf_.tolist()

    tfidf_vectors = {}
    for i, key in enumerate(entry_keys):
        vec = X[i]
        nonzero_mask = vec != 0
        if nonzero_mask.any():
            indices = np.where(nonzero_mask)[0].tolist()
            values = vec[nonzero_mask].tolist()
            tfidf_vectors[key] = {"indices": indices, "values": values}
        else:
            tfidf_vectors[key] = {"indices": [], "values": []}

    tfidf_section = {
        "tokenizer": tfidf_tokenizer_type,
        "max_features": 512,
        "vocab_size": len(vocabulary),
        "vocabulary": vocabulary,
        "idf": idf,
        "vectors": tfidf_vectors,
    }
    if tfidf_tokenizer_type == "char_wb":
        tfidf_section["ngram_range"] = [2, 4]

    print(f"[RetrievalIndex] TF-IDF 构建完成: vocab_size={len(vocabulary)}, "
          f"tokenizer={tfidf_tokenizer_type}")

    # ── Dense Embedding 向量化（可选）────────────────────────────────────
    dense_section = None
    if embedding_func is not None:
        try:
            print(f"[RetrievalIndex] 正在计算 Dense Embedding "
                  f"({len(retrieval_texts)} 条)...")
            embeddings = embedding_func(retrieval_texts)
            dim = len(embeddings[0]) if embeddings else 0
            dense_vectors = {}
            for i, key in enumerate(entry_keys):
                dense_vectors[key] = [round(v, 6) for v in embeddings[i]]
            dense_section = {
                "model": "bge-m3",
                "dimension": dim,
                "vectors": dense_vectors,
            }
            print(f"[RetrievalIndex] Dense Embedding 构建完成: "
                  f"dim={dim}, entries={len(dense_vectors)}")
        except Exception as e:
            print(f"[RetrievalIndex] Dense Embedding 构建失败（已跳过）: {e}")

    # ── 组装并写入索引 ────────────────────────────────────────────────────
    index = {
        "version": "1.0",
        "knowledge_type": knowledge_type,
        "created_at": datetime.now().isoformat(),
        "entry_count": len(entry_keys),
        "entry_keys": entry_keys,
        "tfidf": tfidf_section,
        "entries": entry_meta,
    }
    if dense_section is not None:
        index["dense"] = dense_section

    index_path = os.path.join(knowledge_dir, "retrieval_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(index_path) / 1024
    print(f"[RetrievalIndex] 索引已保存: {index_path} ({size_kb:.1f} KB)")
    return index_path
