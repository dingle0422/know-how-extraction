"""
V2 聚类模块：jieba + TF-IDF + AgglomerativeClustering (cosine 阈值)。
用 cosine 相似度阈值自适应控制簇数量和边界，替代 V1 的固定簇大小约束。
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# ─── 中文分词 & 停用词（与 v1 一致）─────────────────────────────────────────

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

_NOISE_RE = re.compile(
    r'^['
    r'\s\d'
    r'#\*\-\_\~\`\|\>\.\,\!\?'
    r'\uff0c\u3002\uff1a\uff1b'
    r'\u300a\u300b\u3010\u3011'
    r'\uff08\uff09\u2014\u2026'
    r']+$'
)

try:
    import jieba
    jieba.setLogLevel(20)

    def _jieba_tokenizer(text: str) -> list[str]:
        tokens = []
        for tok in jieba.cut(text):
            tok = tok.strip()
            if len(tok) < 2:
                continue
            if _NOISE_RE.match(tok):
                continue
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
        return tokens

    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False


def _make_vectorizer() -> TfidfVectorizer:
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


# ─── 聚类核心逻辑 ──────────────────────────────────────────────────────────

def _cluster_metadata(idx_list: list[int], X: np.ndarray,
                      feature_names: np.ndarray) -> dict:
    """计算单个簇的 TF-IDF top-5 关键词与内聚度指标。"""
    sub = X[idx_list]
    centroid = sub.mean(axis=0)

    top_idx = centroid.argsort()[::-1][:5]
    keywords = [feature_names[i] for i in top_idx if centroid[i] > 0]

    sims = cosine_similarity(sub, centroid.reshape(1, -1)).flatten()
    avg_cosine = round(float(sims.mean()), 4) if len(sims) > 0 else 0.0

    nearest_to_centroid = int(idx_list[int(sims.argmax())])

    return {
        "keywords": keywords,
        "cohesion": {
            "avg_cosine_to_centroid": avg_cosine,
            "item_count": len(idx_list),
        },
        "centroid_item_idx": nearest_to_centroid,
        "cosine_to_centroid": {idx_list[i]: float(sims[i]) for i in range(len(idx_list))},
    }


def make_clusters(items: list[dict], cosine_threshold: float = 0.75) -> list[dict]:
    """
    将 items 按语义相似度聚类：TF-IDF 向量化 + AgglomerativeClustering。
    以 cosine 相似度阈值（而非固定簇大小）控制聚类边界。

    Parameters
    ----------
    items : Level 1 有效条目列表，每项需含 'Know_How' 字段。
    cosine_threshold : 簇内最低 cosine 相似度，默认 0.75。

    Returns
    -------
    list[dict]，每个元素包含：
      - items           : 本簇的条目列表
      - centroid_item   : 距簇质心最近的条目（质心样本）
      - keywords        : TF-IDF top-5 关键词
      - cohesion        : 内聚度指标
      - sorted_others   : 非质心样本按 cosine 降序排列
    """
    n = len(items)
    texts = [item["Know_How"] for item in items]

    vectorizer = _make_vectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    feature_names = vectorizer.get_feature_names_out()

    if n == 1:
        meta = _cluster_metadata([0], X, feature_names)
        return [{
            "items": items,
            "centroid_item": items[0],
            "keywords": meta["keywords"],
            "cohesion": meta["cohesion"],
            "sorted_others": [],
            "tfidf_vectors": X,
            "global_indices": [0],
        }]

    distance_threshold = 1.0 - cosine_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    cluster_indices: dict[int, list[int]] = {}
    for pos, label in enumerate(labels):
        cluster_indices.setdefault(int(label), []).append(pos)

    result = []
    for label in sorted(cluster_indices):
        idx_list = cluster_indices[label]
        meta = _cluster_metadata(idx_list, X, feature_names)
        centroid_global = meta["centroid_item_idx"]

        sim_map = meta["cosine_to_centroid"]
        others_sorted = sorted(
            [i for i in idx_list if i != centroid_global],
            key=lambda i: sim_map.get(i, 0.0),
            reverse=True,
        )

        result.append({
            "items": [items[i] for i in idx_list],
            "centroid_item": items[centroid_global],
            "keywords": meta["keywords"],
            "cohesion": meta["cohesion"],
            "sorted_others": [items[i] for i in others_sorted],
            "tfidf_vectors": X[idx_list],
            "global_indices": idx_list,
        })

    sizes = [len(c["items"]) for c in result]
    print(
        f"[Level-2] 共 {len(result)} 个簇（cosine 阈值={cosine_threshold}），"
        f"簇大小：min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
    )
    return result
