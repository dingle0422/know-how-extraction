"""
V2 聚类模块：TF-IDF + Dense Embedding 混合相似度 + AgglomerativeClustering。
支持通过权重参数灵活切换：纯 TF-IDF / 纯 Dense / 混合聚类。
"""

import re
from typing import Callable

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


def _build_hybrid_similarity(
    X_tfidf: np.ndarray,
    texts: list[str],
    embedding_func: Callable | None,
    tfidf_weight: float,
    embedding_weight: float,
) -> np.ndarray:
    """根据权重构建融合相似度矩阵。权重为 0 的分支跳过计算。"""
    S = np.zeros((len(texts), len(texts)))

    if tfidf_weight > 0:
        S_tfidf = cosine_similarity(X_tfidf)
        S += tfidf_weight * S_tfidf
        print(f"[Clustering] TF-IDF 相似度已计算 (weight={tfidf_weight})")

    if embedding_weight > 0 and embedding_func is not None:
        try:
            print(f"[Clustering] 正在计算 Dense Embedding ({len(texts)} 条)...")
            embeddings = embedding_func(texts)
            E = np.array(embeddings)
            S_emb = cosine_similarity(E)
            S += embedding_weight * S_emb
            print(f"[Clustering] Dense Embedding 相似度已计算 "
                  f"(weight={embedding_weight}, dim={E.shape[1]})")
        except Exception as e:
            print(f"[Clustering] Dense Embedding 计算失败，回退纯 TF-IDF: {e}")
            if tfidf_weight == 0:
                S_tfidf = cosine_similarity(X_tfidf)
                S += S_tfidf
    elif embedding_weight > 0 and embedding_func is None:
        print(f"[Clustering] embedding_weight={embedding_weight} 但未提供 "
              f"embedding_func，回退纯 TF-IDF")
        if tfidf_weight == 0:
            S_tfidf = cosine_similarity(X_tfidf)
            S += S_tfidf

    total_weight = tfidf_weight + embedding_weight
    if total_weight > 0 and total_weight != 1.0:
        S /= total_weight

    return S


def _split_oversized_clusters(
    raw_clusters: list[dict],
    items: list[dict],
    X: np.ndarray,
    feature_names: np.ndarray,
    max_cluster_samples: int,
) -> list[dict]:
    """将超过 max_cluster_samples 的簇按质心相似度倒排拆分为多个子簇。"""
    final: list[dict] = []

    for cluster in raw_clusters:
        idx_list: list[int] = cluster["global_indices"]
        if len(idx_list) <= max_cluster_samples:
            final.append(cluster)
            continue

        remaining_idx = list(idx_list)
        while len(remaining_idx) > max_cluster_samples:
            meta = _cluster_metadata(remaining_idx, X, feature_names)
            centroid_global = meta["centroid_item_idx"]
            sim_map = meta["cosine_to_centroid"]

            sorted_all = sorted(
                remaining_idx,
                key=lambda i: (0 if i == centroid_global else 1,
                               -sim_map.get(i, 0.0)),
            )

            keep_idx = sorted_all[:max_cluster_samples]
            overflow_idx = sorted_all[max_cluster_samples:]

            keep_centroid = centroid_global
            keep_others_sorted = [i for i in keep_idx if i != keep_centroid]
            keep_others_sorted.sort(
                key=lambda i: sim_map.get(i, 0.0), reverse=True
            )

            keep_meta = _cluster_metadata(keep_idx, X, feature_names)
            final.append({
                "items": [items[i] for i in keep_idx],
                "centroid_item": items[keep_centroid],
                "keywords": keep_meta["keywords"],
                "cohesion": keep_meta["cohesion"],
                "sorted_others": [items[i] for i in keep_others_sorted],
                "tfidf_vectors": X[keep_idx],
                "global_indices": keep_idx,
            })

            remaining_idx = overflow_idx

        if remaining_idx:
            sub_meta = _cluster_metadata(remaining_idx, X, feature_names)
            sub_centroid = sub_meta["centroid_item_idx"]
            sub_sim = sub_meta["cosine_to_centroid"]
            sub_others = sorted(
                [i for i in remaining_idx if i != sub_centroid],
                key=lambda i: sub_sim.get(i, 0.0),
                reverse=True,
            )
            final.append({
                "items": [items[i] for i in remaining_idx],
                "centroid_item": items[sub_centroid],
                "keywords": sub_meta["keywords"],
                "cohesion": sub_meta["cohesion"],
                "sorted_others": [items[i] for i in sub_others],
                "tfidf_vectors": X[remaining_idx],
                "global_indices": remaining_idx,
            })

    return final


def make_clusters(
    items: list[dict],
    cosine_threshold: float = 0.75,
    embedding_func: Callable | None = None,
    tfidf_weight: float = 1.0,
    embedding_weight: float = 0.0,
    max_cluster_samples: int = 0,
) -> list[dict]:
    """
    将 items 按相似度聚类，支持 TF-IDF / Dense Embedding / 混合三种模式。

    通过权重控制聚类模式：
      - tfidf_weight=1, embedding_weight=0 → 纯 TF-IDF（默认，向后兼容）
      - tfidf_weight=0, embedding_weight=1 → 纯 Dense Embedding
      - tfidf_weight=0.5, embedding_weight=0.5 → 混合
    权重为 0 的分支跳过计算，避免不必要的开销。

    Parameters
    ----------
    items : Level 1 有效条目列表，每项需含 'Know_How' 字段。
    cosine_threshold : 融合相似度空间中的最低阈值，默认 0.75。
    embedding_func : Dense embedding 函数，签名 (texts: list[str]) -> list[list[float]]；
                     为 None 时自动回退纯 TF-IDF。
    tfidf_weight : TF-IDF 相似度权重，默认 1.0。设为 0 跳过 TF-IDF 相似度计算。
    embedding_weight : Dense Embedding 相似度权重，默认 0.0。设为 0 跳过 Embedding 计算。
    max_cluster_samples : 每个簇的最大样本数，超出部分按相似度倒排拆分为新簇。
                          设为 0 或负数时不限制（向后兼容）。

    Returns
    -------
    list[dict]，每个元素包含：
      - items           : 本簇的条目列表
      - centroid_item   : 距簇质心最近的条目（质心样本）
      - keywords        : TF-IDF top-5 关键词
      - cohesion        : 内聚度指标（基于 TF-IDF）
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

    use_hybrid = (embedding_weight > 0 and embedding_func is not None)
    use_pure_embedding = (tfidf_weight == 0 and embedding_weight > 0)

    if use_hybrid or use_pure_embedding:
        S = _build_hybrid_similarity(
            X, texts, embedding_func, tfidf_weight, embedding_weight,
        )
        D = np.clip(1.0 - S, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)

        distance_threshold = 1.0 - cosine_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(D)
    else:
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

    # ── 超限簇拆分 ──────────────────────────────────────────────────────
    if max_cluster_samples > 0:
        oversized = [c for c in result if len(c["items"]) > max_cluster_samples]
        if oversized:
            pre_split = len(result)
            result = _split_oversized_clusters(
                result, items, X, feature_names, max_cluster_samples,
            )
            print(
                f"[Level-2] 簇大小上限={max_cluster_samples}，"
                f"{len(oversized)} 个超限簇被拆分 → 簇数 {pre_split} → {len(result)}"
            )

    sizes = [len(c["items"]) for c in result]
    mode_desc = "纯TF-IDF"
    if use_pure_embedding:
        mode_desc = "纯Embedding"
    elif use_hybrid:
        mode_desc = f"混合(tfidf={tfidf_weight}, emb={embedding_weight})"
    print(
        f"[Level-2] 共 {len(result)} 个簇（{mode_desc}，"
        f"cosine 阈值={cosine_threshold}），"
        f"簇大小：min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
    )
    return result
