"""
二级提炼：将一级提炼的碎片 Know-How 按批次压缩整合。
支持多线程并发 + 断点续传。
"""

import math
import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from prompts import safe_parse_json_with_llm_repair
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from prompts import safe_parse_json_with_llm_repair

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    from k_means_constrained import KMeansConstrained
    _CLUSTERING_AVAILABLE = True
except ImportError:
    _CLUSTERING_AVAILABLE = False

import re

# 中文常用停用词（虚词、助词、连词等对主题无区分度的词）
_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "有", "和", "就", "不", "都", "一", "也", "很",
    "到", "说", "要", "会", "着", "看", "好", "上", "去", "来", "过", "把",
    "与", "及", "并", "或", "等", "中", "其", "该", "此", "以", "为", "从",
    "由", "被", "让", "使", "于", "对", "将", "已", "可", "能", "时", "后",
    "前", "这", "那", "个", "这个", "那个", "这些", "那些", "什么", "怎么",
    "因为", "所以", "但是", "如果", "可以", "应该", "需要", "已经", "通过",
    "进行", "相关", "包括", "属于", "具有", "情况", "方面", "问题", "方式",
    "他们", "我们", "我", "你", "他", "她", "它", "您", "自己",
}

# 匹配「纯符号 / 纯数字 / markdown 标记」的 token，直接丢弃
_NOISE_RE = re.compile(
    r'^['
    r'\s\d'                      # 空白、数字
    r'#\*\-\_\~\`\|\>\.\,\!\?'  # markdown & 英文标点
    r'\uff0c\u3002\uff1a\uff1b'  # 中文，。：；
    r'\u300a\u300b\u3010\u3011'  # 《》【】
    r'\uff08\uff09\u2014\u2026'  # （）—…
    r']+$'
)

try:
    import jieba
    jieba.setLogLevel(20)  # 关闭 jieba 的 DEBUG 日志

    def _jieba_tokenizer(text: str) -> list[str]:
        tokens = []
        for tok in jieba.cut(text):
            tok = tok.strip()
            if len(tok) < 2:           # 过滤单字
                continue
            if _NOISE_RE.match(tok):   # 过滤纯符号 / 数字 / markdown
                continue
            if tok in _STOPWORDS:      # 过滤停用词
                continue
            tokens.append(tok)
        return tokens

    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

compress_lock = Lock()


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


def load_level1_results(level1_file: str) -> list[dict]:
    """加载一级提炼结果，只保留成功且非空的条目，按 index 排序。"""
    with open(level1_file, "r", encoding="utf-8") as f:
        kh_data = json.load(f)
    valid = sorted(
        [
            {"index": v["index"], "Know_How": v["Know_How"]}
            for v in kh_data.values()
            if v.get("status") == "success" and v.get("Know_How", "").strip()
        ],
        key=lambda x: x["index"],
    )
    print(f"[Level-2] 有效 Know_How 总数: {len(valid)}")
    return valid


def _batch_metadata(idx_list: list[int], X, feature_names) -> dict:
    """计算单个批次的 TF-IDF top-5 关键词与 K-Means 内聚度指标。"""
    sub = X[idx_list]
    centroid = np.asarray(sub.mean(axis=0)).flatten()

    # top-5 关键词：取批次 TF-IDF 均值向量中权重最高的词
    top_idx = centroid.argsort()[::-1][:5]
    keywords = [feature_names[i] for i in top_idx if centroid[i] > 0]

    # 内聚度：各条目与批次质心的平均余弦相似度（越接近 1.0 越紧密）
    sims = _cosine_sim(sub, centroid.reshape(1, -1)).flatten()
    avg_cosine = round(float(sims.mean()), 4) if len(sims) > 0 else 0.0

    # 惯性贡献：各条目到质心的欧氏距离平方和（越小越紧密，与 K-Means 内部指标一致）
    dense = np.asarray(sub.todense()) if hasattr(sub, "todense") else np.asarray(sub)
    inertia = round(float(np.sum((dense - centroid) ** 2)), 4)

    return {
        "keywords": keywords,
        "cohesion": {
            "avg_cosine_to_centroid": avg_cosine,
            "inertia": inertia,
            "item_count": len(idx_list),
        },
    }


def _make_vectorizer() -> "TfidfVectorizer":
    """
    构造 TF-IDF 向量化器。
    优先使用 jieba 分词（正确切分中文词语）；
    若 jieba 不可用则回退到字符级 2-4 gram（避免整句被当成单 token）。
    """
    if _JIEBA_AVAILABLE:
        return TfidfVectorizer(
            tokenizer=_jieba_tokenizer,
            max_features=512,
            token_pattern=None,
        )
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=512,
    )


def make_batches(items: list[dict], batch_size: int = 10) -> list[dict]:
    """
    将 items 按语义相似度分批：使用 TF-IDF 向量化 + 均衡 K-Means 聚类，
    保证每批条数 <= batch_size。
    若依赖库未安装，自动回退到顺序分块（仍会计算关键词和内聚度）。

    返回 list[dict]，每个元素包含：
      - items    : 本批次的 know-how 条目列表
      - keywords : TF-IDF top-5 关键词
      - cohesion : {"avg_cosine_to_centroid": float, "inertia": float, "item_count": int}
    """
    n = len(items)
    texts = [item["Know_How"] for item in items]

    def _fallback_sequential(X, feature_names) -> list[dict]:
        result = []
        for start in range(0, n, batch_size):
            idx_list = list(range(start, min(start + batch_size, n)))
            meta = _batch_metadata(idx_list, X, feature_names)
            result.append({"items": [items[i] for i in idx_list], **meta})
        return result

    if n <= batch_size:
        print(f"[Level-2] 共 1 个批次，每批最多 {batch_size} 条")
        if _CLUSTERING_AVAILABLE:
            vectorizer = _make_vectorizer()
            X = vectorizer.fit_transform(texts)
            meta = _batch_metadata(list(range(n)), X, vectorizer.get_feature_names_out())
            return [{"items": items, **meta}]
        return [{"items": items, "keywords": [], "cohesion": None}]

    if not _CLUSTERING_AVAILABLE:
        print(
            "[Level-2] 警告：未找到 k-means-constrained，回退到顺序分块。"
            "可运行 `pip install scikit-learn k-means-constrained` 启用语义聚类。"
        )
        vectorizer = _make_vectorizer()
        X = vectorizer.fit_transform(texts)
        result = _fallback_sequential(X, vectorizer.get_feature_names_out())
        print(f"[Level-2] 共 {len(result)} 个批次（顺序分块），每批最多 {batch_size} 条")
        return result

    vectorizer = _make_vectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    feature_names = vectorizer.get_feature_names_out()

    k = math.ceil(n / batch_size)
    clf = KMeansConstrained(
        n_clusters=k,
        size_max=batch_size,
        random_state=42,
        n_init=10,
    )
    labels = clf.fit_predict(X)

    cluster_indices: dict[int, list[int]] = {}
    for pos, label in enumerate(labels):
        cluster_indices.setdefault(int(label), []).append(pos)

    result = []
    for label in sorted(cluster_indices):
        idx_list = cluster_indices[label]
        meta = _batch_metadata(idx_list, X, feature_names)
        result.append({"items": [items[i] for i in idx_list], **meta})

    sizes = [len(b["items"]) for b in result]
    print(
        f"[Level-2] 共 {len(result)} 个批次（TF-IDF 语义聚类），"
        f"每批最多 {batch_size} 条，实际批次大小：min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
    )
    return result


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
            with compress_lock:
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

            result = {
                "batch_index": batch_idx,
                "source_indices": [item["index"] for item in batch],
                "item_count": len(batch),
                "batch_keywords": batch_keywords or [],
                "batch_cohesion": batch_cohesion,
                "Synthesis_Summary": content.get("Synthesis_Summary", ""),
                "Final_Know_How": content["Final_Know_How"],
                "status": "success",
                "retry_count": retry_count,
            }
            with compress_lock:
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


def run_level2_compression(
    level1_file: str,
    llm_func,
    prompt_func,
    output_file: str = "./kh_compression_level2.json",
    batch_size: int = 10,
    max_workers: int = 16,
):
    """
    多线程二级知识压缩入口。

    Parameters
    ----------
    level1_file : 一级提炼结果 JSON 路径
    llm_func : LLM 调用函数
    prompt_func : 二级压缩 prompt 构造函数（如 compression_v2）
    output_file : 输出 JSON 路径
    batch_size : 每批条数
    max_workers : 并发线程数
    """
    valid_items = load_level1_results(level1_file)
    batches = make_batches(valid_items, batch_size)

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 个批次记录，自动续传")
        except Exception:
            pass

    pending = [
        (i, bi)
        for i, bi in enumerate(batches)
        if str(i) not in existing_data
        or existing_data.get(str(i), {}).get("status") != "success"
    ]
    completed = len(batches) - len(pending)
    print(f"  总批次: {len(batches)}，已完成: {completed}，"
          f"待处理: {len(pending)}，并发数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_compression_batch,
                i, bi["items"], len(batches), llm_func, prompt_func, output_file,
                batch_keywords=bi.get("keywords"),
                batch_cohesion=bi.get("cohesion"),
            ): i
            for i, bi in pending
        }
        for future in as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    pct = completed / len(batches) * 100
                    print(f"  进度: {completed}/{len(batches)} ({pct:.1f}%)")
            except Exception as e:
                print(f"  批次 {batch_idx} 处理异常: {e}")

    print(f"[Level-2] 全部完成！结果保存于: {output_file}")
    return output_file


# ─── 支持的 QA 源数据扩展名 ───────────────────────────────────────────────
_SUPPORTED_QA_EXTS = {".csv", ".xlsx", ".xls"}


# ─── 单文件完整流水线（Level 1 → Level 2 → Knowledge）─────────────────────
def run_full_pipeline_for_qa(
    source_file: str,
    llm_func,
    level1_prompt_func,
    level2_prompt_func,
    output_dir: str,
    knowledge_dir: str,
    level1_max_workers: int = 4,
    level2_max_workers: int = 16,
    level2_batch_size: int = 10,
    max_retries: int = 100,
):
    """
    对单个 QA 源文件执行完整的 Level 1 → Level 2 → Knowledge 发布流水线。

    若中间产物已存在且全部成功，自动跳过对应阶段。
    """
    import pandas as pd
    from level1_extract import run_level1_extraction
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_source_stem, publish_to_knowledge

    source_stem = get_source_stem(source_file)
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(source_file)[1].lower()
    if ext == ".csv":
        data_train = pd.read_csv(source_file, encoding="utf-8-sig")
    else:
        data_train = pd.read_excel(source_file, sheet_name=0)

    required_cols = {"question", "answer"}
    missing = required_cols - set(data_train.columns)
    if missing:
        raise ValueError(
            f"源文件 {os.path.basename(source_file)} 缺少必需列: {missing}\n"
            f"要求: question, answer 为必需列；reasoning 可选（缺失时自动置空）"
        )

    if "reasoning" not in data_train.columns:
        data_train["reasoning"] = ""
        print(f"  提示: 源文件中不含 reasoning 列，已自动创建并置空")

    core_cols = {"question", "reasoning", "answer"}
    if "Extra_Information" not in data_train.columns:
        extra_cols = [c for c in data_train.columns if c not in core_cols]
        if extra_cols:
            data_train["Extra_Information"] = data_train[extra_cols].apply(
                lambda row: "; ".join(f"{k}={v}" for k, v in row.items() if pd.notna(v)),
                axis=1,
            )
        else:
            data_train["Extra_Information"] = ""
    print(f"  数据加载完成: {len(data_train)} 条记录")

    # ── Level 1: 一级提炼（断点续传由内部处理）──
    level1_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
    print(f"\n  [Level 1] 一级知识提炼: {os.path.basename(source_file)}")
    run_level1_extraction(
        data_train=data_train,
        llm_func=llm_func,
        prompt_func=level1_prompt_func,
        output_file=level1_file,
        max_workers=level1_max_workers,
        max_retries=max_retries,
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

    # ── Level 2: 二级压缩（断点续传由内部处理）──
    level2_file = os.path.join(output_dir, f"{source_stem}_level2_compression.json")
    print(f"\n  [Level 2] 二级知识压缩: {os.path.basename(source_file)}")
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
            kh = v.get("Final_Know_How", "").strip()
            if kh:
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
        _ext = os.path.splitext(source_file)[1].lower()
        if _ext == ".csv":
            with open(source_file, "r", encoding="utf-8") as f:
                source_text_head = f.read(20000)
        else:
            source_text_head = data_train.head(200).to_string(index=False, max_colwidth=120)[:20000]
        publish_to_knowledge(
            source_stem=source_stem,
            final_json_path=level2_file,
            knowledge_base_dir=knowledge_dir,
            llm_func=llm_func,
            source_text_head=source_text_head,
            level1_json_path=level1_file,
        )

    return level2_file


# ─── 独立运行入口：扫描 input 文件夹全部源数据 ──────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from llm_client import chat
    from prompts import single_v1, compression_v2

    input_dir = os.path.join(os.path.dirname(__file__), "input")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")

    source_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _SUPPORTED_QA_EXTS
    ])

    if not source_files:
        raise FileNotFoundError(
            f"input 目录中未找到支持的数据文件（{_SUPPORTED_QA_EXTS}）：{input_dir}"
        )

    print("=" * 60)
    print(f"[level2_compress] 扫描到 {len(source_files)} 个源数据文件，开始批量流水线")
    print("=" * 60)
    for i, fp in enumerate(source_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    for idx, source_file in enumerate(source_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{len(source_files)}] 处理: {os.path.basename(source_file)}")
        print(f"{'═' * 60}")

        run_full_pipeline_for_qa(
            source_file=source_file,
            llm_func=chat,
            level1_prompt_func=single_v1,
            level2_prompt_func=compression_v2,
            output_dir=output_dir,
            knowledge_dir=knowledge_dir,
            level1_max_workers=os.cpu_count() or 4,
            level2_max_workers=os.cpu_count() or 4,
            level2_batch_size=5,
            max_retries=100,
        )

    print(f"\n{'═' * 60}")
    print(f"[level2_compress] 全部 {len(source_files)} 个数据文件处理完成！")
    print(f"{'═' * 60}")
