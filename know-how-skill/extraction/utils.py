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
    write_knowhow_md_with_toc(dst_json, md_path, llm_func=llm_func)

    return sub_dir


# ─── Know-How 内容追加到 knowledge.md ─────────────────────────────────────────

def _render_structured_kh(kh: dict, cluster_index=None) -> str:
    """将 QA v2 的结构化 JSON know-how 渲染为可读文本。"""
    lines = []
    title = kh.get('title', '未命名')
    if cluster_index is not None:
        title = f"[{cluster_index}] {title}"
    lines.append(f"### {title}")
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
            if s.get("constraint"):
                line += f" 【约束: {s['constraint']}】"
            if s.get("policy_basis"):
                line += f" 【依据: {s['policy_basis']}】"
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


def write_knowhow_md_with_toc(
    knowledge_json_path: str,
    knowledge_md_path: str,
    llm_func=None,
):
    """
    读取 knowledge.json，生成带目录索引的 knowledge.md。

    文件结构：
      1. Know-How 目录（标题 + 对应行号）
      2. Know-How 全文内容
      3. 统计信息

    当提供 llm_func 且知识条目数超过阈值时，使用 LLM 生成多级分类目录；
    否则回退到扁平编号目录。

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
            cluster_idx = entry.get("cluster_index")
            rendered = _render_structured_kh(entry["know_how"], cluster_index=cluster_idx)
            title = entry["know_how"].get("title", "未命名")
            if cluster_idx is not None:
                title = f"[{cluster_idx}] {title}"
            if rendered.strip():
                kh_blocks.append((title, rendered))
            for ekh in entry.get("edge_know_hows", []):
                if isinstance(ekh, dict):
                    e_rendered = _render_structured_kh(ekh, cluster_index=cluster_idx)
                    e_title = ekh.get("title", "未命名")
                    if cluster_idx is not None:
                        e_title = f"[{cluster_idx}] {e_title}"
                    if e_rendered.strip():
                        kh_blocks.append((e_title, e_rendered))

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
    titles = [t for t, _ in kh_blocks]
    use_hierarchical = (
        llm_func is not None
        and len(kh_blocks) >= _HIERARCHICAL_TOC_THRESHOLD
    )

    if use_hierarchical:
        print(f"[Knowledge] 知识条目 {len(kh_blocks)} 条，启用 LLM 多级目录分类...")
        categories = _generate_hierarchical_toc(titles, llm_func)
    else:
        categories = None

    if categories is not None:
        toc_header = ["# Know-How 知识目录（多级分类）", ""]
        toc_footer = ["", "---", ""]

        # 先预估目录区行数以计算正确的行号引用
        # 两遍渲染: 第一遍用占位行号确定行数，第二遍用实际行号
        dummy_line_map = {i: 0 for i in range(len(titles))}
        dummy_toc = _render_hierarchical_toc(categories, titles, dummy_line_map)
        toc_line_count = len(toc_header) + len(dummy_toc) + len(toc_footer)

        title_line_map = {}
        for i, (_title, offset_in_content) in enumerate(title_offsets):
            title_line_map[i] = toc_line_count + offset_in_content + 1

        toc_entries = _render_hierarchical_toc(categories, titles, title_line_map)
        print(f"[Knowledge] 多级目录已生成: "
              f"{len(categories)} 个一级分类")
    else:
        if use_hierarchical:
            print("[Knowledge] LLM 多级目录生成失败，回退扁平目录")
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

    toc_type = "多级分类" if categories is not None else "扁平"
    print(
        f"[Knowledge] knowledge.md 已生成: {knowledge_md_path} "
        f"({len(kh_blocks)} 条, {total_chars:,} 字, {toc_type}目录)"
    )


# ─── 多级目录 LLM 分类 ────────────────────────────────────────────────────

_HIERARCHICAL_TOC_PROMPT = """\
你是一个知识体系架构师。下面有 {count} 条知识块标题，请将它们组织为**两级分类目录**，方便用户层层检索。

# 输入标题列表
{titles_text}

# 分类要求
1. **一级分类**：按业务领域/主题大类划分（通常 5~15 个大类），名称简洁有概括性（如"增值税免税政策""进项税额抵扣""会计处理"等）。
2. **二级分类**：在每个一级分类下，按更细粒度的子主题/场景进一步分组（每个一级下通常 2~8 个二级），如果某一级分类下条目数 ≤ 5，可以不设二级，直接放条目。
3. 每条知识块只能归入**一个**二级分类（不可重复归类）。
4. 所有输入标题都必须被归类，不得遗漏。
5. 分类名称应尽量短小精炼（≤15字），体现该类的核心主题。

# 输出格式
严格输出合法 JSON（不要用 ```json 包裹），结构如下：
{{
  "categories": [
    {{
      "name": "一级分类名称",
      "subcategories": [
        {{
          "name": "二级分类名称",
          "items": [0, 3, 5]
        }}
      ]
    }},
    {{
      "name": "另一个一级分类（条目少时可无二级子分类）",
      "subcategories": [
        {{
          "name": "",
          "items": [10, 11]
        }}
      ]
    }}
  ]
}}

说明：
- items 数组中是标题前面的序号（0-based）。
- 当某一级分类不需要二级子分类时，subcategories 中放一个 name 为空字符串的项，items 包含该类全部条目序号。
- categories 数组按逻辑顺序排列（如先税种政策、再操作流程、再会计处理等）。"""


_HIERARCHICAL_TOC_THRESHOLD = 20


def _generate_hierarchical_toc(
    titles: list[str],
    llm_func,
    max_retries: int = 3,
) -> list[dict] | None:
    """调用 LLM 将扁平标题列表组织为多级分类结构。

    返回格式: [{"name": "...", "subcategories": [{"name": "...", "items": [idx, ...]}]}]
    失败时返回 None，调用方回退到扁平目录。
    """
    titles_text = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    prompt = _HIERARCHICAL_TOC_PROMPT.format(count=len(titles), titles_text=titles_text)

    for attempt in range(max_retries):
        try:
            raw = llm_func(prompt)["content"]
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            result = json.loads(raw)
            categories = result.get("categories", [])
            if not categories:
                continue

            seen = set()
            for cat in categories:
                for sub in cat.get("subcategories", []):
                    for idx in sub.get("items", []):
                        seen.add(idx)

            if len(seen) < len(titles) * 0.9:
                print(f"    [TOC] 第 {attempt+1} 次尝试覆盖率不足 "
                      f"({len(seen)}/{len(titles)})，重试...")
                continue

            return categories
        except Exception as e:
            print(f"    [TOC] 第 {attempt+1} 次尝试失败: {e}")

    return None


def _render_hierarchical_toc(
    categories: list[dict],
    titles: list[str],
    title_line_map: dict[int, int],
) -> list[str]:
    """将 LLM 返回的分类结构渲染为多级 Markdown 目录行。

    Parameters
    ----------
    categories     : LLM 返回的分类树
    titles         : 知识块标题列表
    title_line_map : {知识块序号: 实际行号} 映射
    """
    lines = []
    for ci, cat in enumerate(categories, 1):
        lines.append(f"## {ci}. {cat['name']}")
        lines.append("")
        subcats = cat.get("subcategories", [])
        for si, sub in enumerate(subcats, 1):
            sub_name = sub.get("name", "")
            items = sub.get("items", [])
            if sub_name:
                lines.append(f"### {ci}.{si} {sub_name}")
                lines.append("")
            for idx in items:
                if 0 <= idx < len(titles):
                    line_no = title_line_map.get(idx, "?")
                    lines.append(f"- {titles[idx]}  (行 {line_no})")
            lines.append("")
    return lines


# ─── 检索关键词生成 ────────────────────────────────────────────────────────

_RETRIEVAL_KEYWORDS_PROMPT = """\
你是一个信息检索专家。下面是一条知识块的内容，请提取能帮助用户通过搜索找到它的关键词列表。

# 知识块内容
{content}

# 提取要求
提取 10-20 个关键词/短语，重点覆盖：
1. **领域实体**：涉及的专业术语、政策名称、业务对象（如"进项发票""增值税""认证期限"）
2. **场景动作**：描述该场景的动词或短语（如"逾期处理""补开发票""税前扣除"）
3. **用户表达**：用户提问时可能使用的口语化说法和同义词（如"过期了怎么办""没有发票""扣税"）

关键词应尽量短（2-6 字），避免与知识块标题和适用场景高度重复。

# 输出格式
直接输出 JSON 数组（不要用 ```json 包裹）：
["关键词1", "关键词2", ...]"""


def _render_entry_for_keywords(entry: dict) -> str:
    """将知识条目渲染为供关键词提取 LLM 阅读的可读文本。"""
    if "know_how" in entry and isinstance(entry["know_how"], dict):
        kh = entry["know_how"]
        parts = []
        if kh.get("title"):
            parts.append(f"标题: {kh['title']}")
        if kh.get("scope"):
            parts.append(f"适用场景: {kh['scope']}")
        for s in kh.get("steps", []):
            line = f"步骤: {s.get('action', '')}"
            if s.get("condition"):
                line = f"[当 {s['condition']}] {line}"
            if s.get("constraint"):
                line += f" [约束: {s['constraint']}]"
            if s.get("policy_basis"):
                line += f" [依据: {s['policy_basis']}]"
            if s.get("outcome"):
                line += f" → {s['outcome']}"
            parts.append(line)
        for ex in kh.get("exceptions", []):
            parts.append(f"例外: 当 {ex.get('when', '?')} → {ex.get('then', '?')}")
        return "\n".join(parts)

    if "Final_Know_How" in entry:
        fkh = entry["Final_Know_How"]
        if isinstance(fkh, str):
            return fkh[:3000]
        if isinstance(fkh, list):
            text = "\n".join(t.strip() for t in fkh if t.strip())
            return text[:3000]

    return ""


def _generate_retrieval_keywords(
    entry: dict,
    llm_func,
    existing_keywords: list[str] | None = None,
) -> list[str]:
    """为单条知识条目生成面向检索的关键词列表。

    使用 LLM 从知识内容中提取领域实体、场景动作和用户常用表达，
    失败时退回 existing_keywords（聚类阶段的 TF-IDF 关键词）。
    """
    content = _render_entry_for_keywords(entry)
    if not content.strip():
        return existing_keywords or []

    try:
        prompt = _RETRIEVAL_KEYWORDS_PROMPT.format(content=content)
        raw = llm_func(prompt)["content"]
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        keywords = json.loads(raw)
        if isinstance(keywords, list):
            cleaned = [str(kw).strip() for kw in keywords if str(kw).strip()]
            if cleaned:
                return cleaned
    except Exception as e:
        print(f"    [Keywords] LLM 提取失败，使用已有关键词: {e}")

    return existing_keywords or []


def _generate_all_retrieval_keywords(
    data: dict,
    entry_keys: list[str],
    llm_func,
    max_workers: int = 4,
) -> dict[str, list[str]]:
    """批量为多条知识块并行生成检索关键词。

    已有 retrieval_keywords 的条目会跳过生成，直接复用。
    返回 {entry_key: [keyword, ...]} 映射。
    """
    import concurrent.futures

    to_generate = []
    results: dict[str, list[str]] = {}
    for key in entry_keys:
        entry = data[key]
        if entry.get("retrieval_keywords"):
            results[key] = entry["retrieval_keywords"]
        else:
            to_generate.append(key)

    if not to_generate:
        print(f"[RetrievalIndex] 所有 {len(entry_keys)} 条知识块已有检索关键词，跳过生成")
        return results

    reused = len(entry_keys) - len(to_generate)
    if reused:
        print(f"[RetrievalIndex] 复用已有关键词 {reused} 条，"
              f"新生成 {len(to_generate)} 条（并发={max_workers}）...")
    else:
        print(f"[RetrievalIndex] 正在为 {len(to_generate)} 条知识块"
              f"生成检索关键词（并发={max_workers}）...")

    counter = [0]
    total = len(to_generate)

    def _process_one(key: str) -> tuple[str, list[str]]:
        entry = data[key]
        existing = entry.get("cluster_keywords", entry.get("batch_keywords", []))
        return key, _generate_retrieval_keywords(entry, llm_func, existing)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, key): key
            for key in to_generate
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                k, kws = future.result()
                results[k] = kws
                counter[0] += 1
                if counter[0] % 5 == 0 or counter[0] == total:
                    print(f"    [{counter[0]}/{total}] 关键词已生成")
            except Exception as e:
                fallback = data[key].get(
                    "cluster_keywords", data[key].get("batch_keywords", []),
                )
                results[key] = fallback
                counter[0] += 1
                print(f"    [{counter[0]}/{total}] key={key} 失败: {e}")

    print(f"[RetrievalIndex] 检索关键词生成完成")
    return results


# ─── 检索索引构建 ────────────────────────────────────────────────────────────

def _extract_retrieval_text(
    entry: dict,
    retrieval_keywords: list[str] | None = None,
) -> str:
    """从 knowledge.json 的单条记录中提取用于检索向量化的文本。

    检索文本策略:
      - 有 retrieval_keywords: title + scope + 关键词（语义集中，检索效果好）
      - 无 retrieval_keywords: 全字段拼接（向后兼容）

    自动适配两种知识格式:
      - QA v2 结构化: know_how dict
      - Doc compression: Final_Know_How list[str]
    """
    # QA v2 结构化格式
    if "know_how" in entry and isinstance(entry["know_how"], dict):
        kh = entry["know_how"]

        if retrieval_keywords:
            parts = []
            if kh.get("title"):
                parts.append(kh["title"])
            if kh.get("scope"):
                parts.append(kh["scope"])
            parts.append(" ".join(retrieval_keywords))
            return " ".join(parts)

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
            if s.get("constraint"):
                parts.append(s["constraint"])
            if s.get("policy_basis"):
                parts.append(s["policy_basis"])
            if s.get("outcome"):
                parts.append(s["outcome"])
        for ex in kh.get("exceptions", []):
            if ex.get("when"):
                parts.append(ex["when"])
            if ex.get("then"):
                parts.append(ex["then"])
        return " ".join(parts)

    # Doc compression 格式
    if "Final_Know_How" in entry:
        if retrieval_keywords:
            summary = entry.get("Synthesis_Summary", "")
            if not summary:
                fkh = entry["Final_Know_How"]
                if isinstance(fkh, str):
                    summary = fkh.strip().split("\n")[0][:200]
                elif isinstance(fkh, list) and fkh:
                    summary = fkh[0].strip().split("\n")[0][:200]
            parts = []
            if summary:
                parts.append(summary)
            parts.append(" ".join(retrieval_keywords))
            return " ".join(parts)

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
    llm_func=None,
) -> str | None:
    """
    为 knowledge 目录构建检索索引文件 retrieval_index.json。

    对 knowledge.json 中每个成功的知识条目：
      1. 生成检索关键词（可选，需要 llm_func）
      2. 提取检索用文本（有关键词时使用 title+scope+keywords 策略）
      3. 构建 TF-IDF 向量（保存 vocabulary + IDF 权重 + 条目向量）
      4. 调用 embedding 服务构建 Dense 向量（可选，服务不可用时跳过）

    Parameters
    ----------
    knowledge_json_path : knowledge.json 文件路径
    knowledge_dir       : knowledge 子目录路径（索引文件保存位置）
    embedding_func      : Dense embedding 函数，签名 (texts: list[str]) -> list[list[float]]；
                          为 None 或调用失败时仅保存 TF-IDF 索引
    llm_func            : LLM 调用函数，签名 (prompt: str) -> {"content": str}；
                          用于生成检索关键词。为 None 时复用已有关键词或退回全字段拼接

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

    # ── 第 1 步：收集有效条目 ─────────────────────────────────────────────
    valid_keys = []
    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = data[key]
        if isinstance(entry, dict) and entry.get("status") == "success":
            valid_keys.append(key)

    if not valid_keys:
        print("[RetrievalIndex] 无有效知识条目，跳过索引构建")
        return None

    print(f"[RetrievalIndex] 检测到 {knowledge_type} 格式，"
          f"共 {len(valid_keys)} 条有效知识条目")

    # ── 第 2 步：生成检索关键词 ───────────────────────────────────────────
    entry_keywords: dict[str, list[str]] = {}
    if llm_func is not None:
        entry_keywords = _generate_all_retrieval_keywords(
            data, valid_keys, llm_func,
        )
        keywords_updated = False
        for key, kws in entry_keywords.items():
            if kws and data[key].get("retrieval_keywords") != kws:
                data[key]["retrieval_keywords"] = kws
                keywords_updated = True
        if keywords_updated:
            with open(knowledge_json_path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(data), f, ensure_ascii=False, indent=2)
            print(f"[RetrievalIndex] 检索关键词已写回 knowledge.json")
    else:
        for key in valid_keys:
            saved_kw = data[key].get("retrieval_keywords")
            if saved_kw:
                entry_keywords[key] = saved_kw
        if entry_keywords:
            print(f"[RetrievalIndex] 复用已有检索关键词: {len(entry_keywords)} 条")

    # ── 第 3 步：提取检索文本 ─────────────────────────────────────────────
    entry_keys = []
    retrieval_texts = []
    entry_meta = {}

    for key in valid_keys:
        entry = data[key]
        kw = entry_keywords.get(key)
        text = _extract_retrieval_text(entry, retrieval_keywords=kw)
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
        meta["retrieval_keywords"] = kw or []
        entry_meta[key] = meta

    if not retrieval_texts:
        print("[RetrievalIndex] 无有效知识条目，跳过索引构建")
        return None

    kw_count = sum(1 for k in entry_keys if entry_keywords.get(k))
    strategy = (f"title+scope+keywords ({kw_count}/{len(entry_keys)} 条)"
                if kw_count else "全字段拼接（兼容模式）")
    print(f"[RetrievalIndex] 检索文本策略: {strategy}")

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
        "version": "1.1",
        "knowledge_type": knowledge_type,
        "retrieval_strategy": "title+scope+keywords" if kw_count else "full_fields",
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
