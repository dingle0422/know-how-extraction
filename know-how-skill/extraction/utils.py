"""
Extraction 公共工具
==================
提供源文件名解析、knowledge 目录发布、检索索引构建等跨模块复用功能。
"""

import json
import math
import os
import shutil
from datetime import datetime


def sanitize_for_json(obj):
    """递归清理数据结构中的 None / NaN / Infinity，统一替换为空字符串。

    确保 json.dump 输出的 JSON 不含 null、NaN、Infinity 等
    可能导致某些解析器（如 Excel）无法打开的值。
    """
    if obj is None:
        return ""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return ""
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def get_source_stem(filepath: str) -> str:
    """从文件路径提取不含扩展名的文件名（源文件名前缀）。"""
    return os.path.splitext(os.path.basename(filepath))[0]


def publish_to_knowledge(
    source_stem: str,
    final_json_path: str,
    knowledge_base_dir: str,
    llm_func=None,
    source_text_head: str = "",
    level1_json_path: str = None,
    knowledge_desc_prompt_func=None,
):
    """
    将最终抽取结果发布到 knowledge 子文件夹。

    目录结构:
        {knowledge_base_dir}/{source_stem}_knowledge/
            ├── knowledge.json       (最终抽取结果副本)
            ├── knowledge.md         (Know-How 目录索引 + 全文内容)
            └── knowledge_traceback.json     (一级抽取结果，含源数据回溯信息，可选)

    Parameters
    ----------
    source_stem            : 源文件名（不含扩展名）
    final_json_path        : 最终抽取结果 JSON 路径
    knowledge_base_dir     : knowledge 根目录（如 .../doc_know_how_build/knowledge）
    llm_func               : （已废弃，保留兼容）
    source_text_head       : （已废弃，保留兼容）
    level1_json_path       : 一级抽取结果 JSON 路径（含源数据回溯），可选
    knowledge_desc_prompt_func : （已废弃，保留兼容）
    """
    sub_dir = os.path.join(knowledge_base_dir, f"{source_stem}_knowledge")
    os.makedirs(sub_dir, exist_ok=True)

    dst_json = os.path.join(sub_dir, "knowledge.json")
    shutil.copy2(final_json_path, dst_json)
    print(f"[Knowledge] 最终结果已复制到: {dst_json}")

    if level1_json_path and os.path.exists(level1_json_path):
        dst_traceback = os.path.join(sub_dir, "knowledge_traceback.json")
        shutil.copy2(level1_json_path, dst_traceback)
        print(f"[Knowledge] 一级回溯文件已复制到: {dst_traceback}")

    md_path = os.path.join(sub_dir, "knowledge.md")
    print("[Knowledge] 正在生成 knowledge.md（目录索引 + 全文内容）...")
    write_knowhow_md_with_toc(dst_json, md_path)

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
            step_id = s.get("step", "?")
            depth = step_id.count(".") if isinstance(step_id, str) else 0
            indent = "  " * (depth + 1)
            condition_prefix = ""
            if s.get("condition"):
                condition_prefix = f"【触发条件：{s['condition']}】 → "
            line = f"{indent}{step_id}. {condition_prefix}{s.get('action', '')}"
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


def _extract_title_from_text(text: str) -> str:
    """从渲染后的文本块中提取标题（第一个 Markdown heading 或首行非空文本）。"""
    import re as _re
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return _re.sub(r'^#+\s*', '', line).strip()
        if line:
            return line[:80]
    return "未命名"


def write_knowhow_md_with_toc(knowledge_json_path: str, knowledge_md_path: str):
    """
    读取 knowledge.json，生成带目录索引的 knowledge.md。

    文件结构：
      1. Know-How 目录（标题 + 对应行号）
      2. Know-How 全文内容
      3. 统计信息

    自动检测两种格式：
      - QA v2 结构化格式: know_how (dict with title/steps/...)
      - Doc compression 格式: Final_Know_How (list[str])
    """
    try:
        with open(knowledge_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    kh_blocks = []

    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = data[key]
        if not isinstance(entry, dict) or entry.get("status") != "success":
            continue

        if "know_how" in entry and isinstance(entry["know_how"], dict):
            rendered = _render_structured_kh(entry["know_how"])
            title = entry["know_how"].get("title", "未命名")
            if rendered.strip():
                kh_blocks.append((title, rendered))

        elif "Final_Know_How" in entry:
            fkh = entry["Final_Know_How"]
            if isinstance(fkh, str):
                fkh = [fkh]
            if isinstance(fkh, list):
                for topic in fkh:
                    topic = topic.strip()
                    if topic:
                        title = _extract_title_from_text(topic)
                        kh_blocks.append((title, topic))

    if not kh_blocks:
        with open(knowledge_md_path, "w", encoding="utf-8") as f:
            f.write("# Know-How 知识目录\n\n> 暂无知识条目\n")
        return

    # ── 构建全文内容区，并记录每个标题在内容区中的行偏移 ──────────────
    content_lines = ["# Know-How 全文内容", ""]
    title_offsets = []

    for i, (title, text) in enumerate(kh_blocks):
        if i > 0:
            content_lines.append("---")
            content_lines.append("")
        title_offsets.append((title, len(content_lines)))
        content_lines.extend(text.split("\n"))
        content_lines.append("")

    total_chars = sum(len(text) for _, text in kh_blocks)
    content_lines.append("---")
    content_lines.append("")
    content_lines.append(
        f"> **Know-How 总字数: {total_chars:,} 字 | 共 {len(kh_blocks)} 条知识节点**"
    )

    # ── 构建目录区 ────────────────────────────────────────────────────────
    toc_header = ["# Know-How 知识目录", ""]
    toc_footer = ["", "---", ""]
    toc_line_count = len(toc_header) + len(kh_blocks) + len(toc_footer)

    toc_entries = []
    for i, (title, offset_in_content) in enumerate(title_offsets):
        actual_line = toc_line_count + offset_in_content + 1
        toc_entries.append(f"{i + 1}. {title}  (行 {actual_line})")

    # ── 组装并写入文件 ────────────────────────────────────────────────────
    all_lines = toc_header + toc_entries + toc_footer + content_lines

    with open(knowledge_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
        f.write("\n")

    print(
        f"[Knowledge] knowledge.md 已生成: {knowledge_md_path} "
        f"({len(kh_blocks)} 条, {total_chars:,} 字)"
    )


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
        json.dump(sanitize_for_json(index), f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(index_path) / 1024
    print(f"[RetrievalIndex] 索引已保存: {index_path} ({size_kb:.1f} KB)")
    return index_path
