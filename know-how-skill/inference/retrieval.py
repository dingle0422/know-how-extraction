"""
Phase 1: 双路并行检索模块
========================
加载 extraction 阶段预构建的 retrieval_index.json，
基于 TF-IDF 和 Dense Embedding 两路并行检索，返回候选知识块集合。
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np


# ─── 内部工具 ──────────────────────────────────────────────────────────────────

def _build_jieba_tokenizer():
    """构建与 extraction 阶段一致的 jieba 分词器（含停用词过滤）。"""
    import jieba
    jieba.setLogLevel(20)

    stopwords = {
        "的", "了", "在", "是", "有", "和", "就", "不", "都", "一", "也", "很",
        "到", "说", "要", "会", "着", "看", "好", "上", "去", "来", "过", "把",
        "与", "及", "并", "或", "等", "中", "其", "该", "此", "以", "为", "从",
        "由", "被", "让", "使", "于", "对", "将", "已", "可", "能", "时", "后",
        "前", "这", "那", "个", "这个", "那个", "这些", "那些", "什么", "怎么",
        "因为", "所以", "但是", "如果", "可以", "应该", "需要", "已经", "通过",
        "进行", "相关", "包括", "属于", "具有", "情况", "方面", "问题", "方式",
        "他们", "我们", "我", "你", "他", "她", "它", "您", "自己",
    }

    def tokenize(text: str) -> list[str]:
        return [
            tok.strip() for tok in jieba.cut(text)
            if len(tok.strip()) >= 2 and tok.strip() not in stopwords
        ]

    return tokenize


def _build_charwb_tokenizer(ngram_range=(2, 4)):
    """退化分词器：字符 n-gram，与 extraction 阶段 fallback 保持一致。"""
    lo, hi = ngram_range

    def tokenize(text: str) -> list[str]:
        text = f" {text.strip()} "
        tokens = []
        for n in range(lo, hi + 1):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                if gram.strip():
                    tokens.append(gram)
        return tokens

    return tokenize


def _cosine_sparse(q_indices, q_values, d_indices, d_values) -> float:
    """计算两个稀疏向量的余弦相似度。"""
    q_map = dict(zip(q_indices, q_values))
    dot = sum(q_map.get(idx, 0.0) * v for idx, v in zip(d_indices, d_values))
    norm_q = math.sqrt(sum(v * v for v in q_values)) if q_values else 0.0
    norm_d = math.sqrt(sum(v * v for v in d_values)) if d_values else 0.0
    if norm_q == 0.0 or norm_d == 0.0:
        return 0.0
    return dot / (norm_q * norm_d)


def _cosine_dense(vec_a, vec_b) -> float:
    """计算两个稠密向量的余弦相似度。"""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


# ─── 单目录检索索引加载 & 查询 ───────────────────────────────────────────────

class KnowledgeRetriever:
    """对单个 knowledge 目录的检索索引进行加载与查询。"""

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.dir_name = os.path.basename(knowledge_dir)

        index_path = os.path.join(knowledge_dir, "retrieval_index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self._index = json.load(f)

        self.knowledge_type = self._index.get("knowledge_type", "unknown")
        self.entry_keys = self._index.get("entry_keys", [])
        self._tfidf = self._index.get("tfidf", {})
        self._dense = self._index.get("dense")
        self._entries = self._index.get("entries", {})

        tokenizer_type = self._tfidf.get("tokenizer", "jieba")
        try:
            if tokenizer_type == "jieba":
                self._tokenizer = _build_jieba_tokenizer()
            else:
                ngram = self._tfidf.get("ngram_range", [2, 4])
                self._tokenizer = _build_charwb_tokenizer(tuple(ngram))
        except ImportError:
            ngram = self._tfidf.get("ngram_range", [2, 4])
            self._tokenizer = _build_charwb_tokenizer(tuple(ngram))

        self._vocabulary = self._tfidf.get("vocabulary", {})
        self._idf = self._tfidf.get("idf", [])
        self._tfidf_vectors = self._tfidf.get("vectors", {})

    def _query_tfidf_vector(self, query: str) -> tuple[list[int], list[float]]:
        """将查询文本转为 TF-IDF 稀疏向量（与索引构建时保持一致）。"""
        tokens = self._tokenizer(query)
        if not tokens:
            return [], []

        tf_counts: dict[int, int] = {}
        for tok in tokens:
            idx = self._vocabulary.get(tok)
            if idx is not None:
                tf_counts[idx] = tf_counts.get(idx, 0) + 1

        if not tf_counts:
            return [], []

        indices = sorted(tf_counts.keys())
        raw_values = []
        for idx in indices:
            tf = tf_counts[idx]
            idf = self._idf[idx] if idx < len(self._idf) else 1.0
            raw_values.append(tf * idf)

        norm = math.sqrt(sum(v * v for v in raw_values))
        values = [v / norm for v in raw_values] if norm > 0 else raw_values
        return indices, values

    def search_tfidf(self, query: str, top_n: int = 5) -> list[dict]:
        """TF-IDF 路线检索：返回 top_n 个最相似的知识块。"""
        q_indices, q_values = self._query_tfidf_vector(query)
        if not q_indices:
            return []

        scores = []
        for key in self.entry_keys:
            vec = self._tfidf_vectors.get(key, {})
            d_indices = vec.get("indices", [])
            d_values = vec.get("values", [])
            if not d_indices:
                continue
            sim = _cosine_sparse(q_indices, q_values, d_indices, d_values)
            if sim > 0:
                scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for key, sim in scores[:top_n]:
            results.append({
                "source_dir": self.dir_name,
                "knowledge_dir": self.knowledge_dir,
                "entry_key": key,
                "knowledge_type": self.knowledge_type,
                "score": round(sim, 6),
                "retrieval_method": "tfidf",
                "meta": self._entries.get(key, {}),
            })
        return results

    def search_dense(
        self, query: str, top_n: int = 5, embedding_func: Callable = None,
    ) -> list[dict]:
        """Dense Embedding 路线检索：返回 top_n 个最相似的知识块。"""
        if self._dense is None or embedding_func is None:
            return []
        try:
            q_embedding = embedding_func([query])[0]
        except Exception as e:
            print(f"[Retrieval] Dense embedding 计算失败 ({self.dir_name}): {e}")
            return []
        return self.search_dense_with_embedding(q_embedding, top_n=top_n)

    def search_dense_with_embedding(
        self, query_embedding: list[float], top_n: int = 5,
    ) -> list[dict]:
        """Dense 检索（使用预计算的查询向量，避免重复调用 embedding 服务）。"""
        if self._dense is None:
            return []

        dense_vectors = self._dense.get("vectors", {})
        if not dense_vectors:
            return []

        scores = []
        for key in self.entry_keys:
            d_vec = dense_vectors.get(key)
            if d_vec is None:
                continue
            sim = _cosine_dense(query_embedding, d_vec)
            if sim > 0:
                scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for key, sim in scores[:top_n]:
            results.append({
                "source_dir": self.dir_name,
                "knowledge_dir": self.knowledge_dir,
                "entry_key": key,
                "knowledge_type": self.knowledge_type,
                "score": round(sim, 6),
                "retrieval_method": "dense",
                "meta": self._entries.get(key, {}),
            })
        return results


# ─── Phase 1 入口：多目录双路并行检索 ─────────────────────────────────────────

def retrieve_candidates(
    question: str,
    knowledge_dirs: list[str],
    tfidf_top_n: int = 5,
    embedding_top_n: int = 5,
    embedding_func: Callable = None,
) -> list[dict]:
    """
    Phase 1: 对指定的 knowledge 目录列表执行双路并行检索。

    优化策略：
      1. Query embedding 只调用一次，所有目录共享
      2. 多个 knowledge 目录之间并行检索
      3. 每个目录内 TF-IDF 和 Dense 双路同步执行

    Parameters
    ----------
    question         : 用户问题
    knowledge_dirs   : knowledge 子目录路径列表
    tfidf_top_n      : TF-IDF 路线每目录返回数量
    embedding_top_n  : Dense Embedding 路线每目录返回数量
    embedding_func   : Dense embedding 函数，签名 (texts: list[str]) -> list[list[float]]

    Returns
    -------
    去重后的候选知识块列表，每项包含 source_dir, entry_key, knowledge_type, score 等信息
    """
    retrievers = []
    for d in knowledge_dirs:
        index_path = os.path.join(d, "retrieval_index.json")
        if not os.path.exists(index_path):
            print(f"[Retrieval] 跳过目录（无检索索引）: {d}")
            continue
        try:
            retrievers.append(KnowledgeRetriever(d))
        except Exception as e:
            print(f"[Retrieval] 加载索引失败 ({d}): {e}")

    if not retrievers:
        print("[Retrieval] 无可用的检索索引")
        return []

    # ── Query embedding 只计算一次，所有目录共享 ──
    query_embedding = None
    if embedding_func is not None:
        has_dense = any(ret._dense is not None for ret in retrievers)
        if has_dense:
            try:
                query_embedding = embedding_func([question])[0]
            except Exception as e:
                print(f"[Retrieval] Query embedding 计算失败（退化为纯 TF-IDF）: {e}")

    # ── 单个 retriever 的双路检索（TF-IDF + Dense）──
    def _search_one_dir(ret: KnowledgeRetriever) -> list[dict]:
        hits = ret.search_tfidf(question, top_n=tfidf_top_n)
        if query_embedding is not None:
            hits += ret.search_dense_with_embedding(
                query_embedding, top_n=embedding_top_n,
            )
        return hits

    # ── 多目录并行检索 ──
    all_results: list[dict] = []
    if len(retrievers) == 1:
        all_results = _search_one_dir(retrievers[0])
    else:
        with ThreadPoolExecutor(max_workers=len(retrievers)) as executor:
            futures = {
                executor.submit(_search_one_dir, ret): ret
                for ret in retrievers
            }
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    ret = futures[future]
                    print(f"[Retrieval] 目录检索异常 ({ret.dir_name}): {e}")

    # ── 去重：按 (source_dir, entry_key) 保留最高分 ──
    seen: dict[tuple, dict] = {}
    for r in all_results:
        uid = (r["source_dir"], r["entry_key"])
        if uid not in seen or r["score"] > seen[uid]["score"]:
            seen[uid] = r

    deduplicated = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return deduplicated


# ─── 知识内容加载工具 ──────────────────────────────────────────────────────────

def load_knowledge_content(knowledge_dir: str, entry_key: str) -> str:
    """从 knowledge.json 加载指定条目的知识块文本内容。"""
    kj_path = os.path.join(knowledge_dir, "knowledge.json")
    with open(kj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entry = data.get(entry_key)
    if entry is None:
        return ""

    if "know_how" in entry and isinstance(entry["know_how"], dict):
        return _render_qa_knowhow(entry["know_how"])

    if "Final_Know_How" in entry:
        fkh = entry["Final_Know_How"]
        if isinstance(fkh, str):
            return fkh
        if isinstance(fkh, list):
            return "\n".join(t.strip() for t in fkh if t.strip())

    return ""


def _render_qa_knowhow(kh: dict) -> str:
    """将 QA v2 结构化 Know-How 渲染为可读文本。"""
    parts = []
    if kh.get("title"):
        parts.append(f"【{kh['title']}】")
    if kh.get("scope"):
        parts.append(f"适用场景: {kh['scope']}")

    for s in kh.get("steps", []):
        line = f"步骤{s.get('step', '?')}: {s.get('action', '')}"
        if s.get("condition"):
            line = f"[当 {s['condition']}] " + line
        if s.get("outcome"):
            line += f" → {s['outcome']}"
        parts.append(line)

    for ex in kh.get("exceptions", []):
        parts.append(f"例外: 当 {ex.get('when', '?')} → {ex.get('then', '?')}")

    for c in kh.get("constraints", []):
        if isinstance(c, str):
            parts.append(f"约束: {c}")

    return "\n".join(parts)


def load_edge_cases(knowledge_dir: str, entry_key: str) -> list[dict]:
    """从 edge_cases.json 中加载指定 cluster 的边缘案例。

    entry_key (如 "3") 映射到 edge_cases.json 中的 "cluster_3"。
    """
    ec_path = os.path.join(knowledge_dir, "edge_cases.json")
    if not os.path.exists(ec_path):
        return []

    with open(ec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cluster_key = f"cluster_{entry_key}"
    cluster_data = data.get(cluster_key, {})
    return cluster_data.get("edge_cases", [])


def load_level1_knowhow_map(knowledge_dir: str) -> dict[int, str]:
    """从 knowledge_traceback.json 加载 index → Level-1 Know_How 文本的映射。

    用于在 Phase 3 边缘案例兜底时，为每个边缘案例补充其一级提炼知识。
    """
    traceback_path = os.path.join(knowledge_dir, "knowledge_traceback.json")
    if not os.path.exists(traceback_path):
        return {}
    try:
        with open(traceback_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    mapping: dict[int, str] = {}
    for _key, entry in data.items():
        idx = entry.get("index", -1)
        kh = entry.get("Know_How", "")
        if kh and idx >= 0:
            mapping[idx] = kh
    return mapping


def retrieve_edge_cases(
    question: str,
    edge_cases: list[dict],
    top_n: int = 3,
    embedding_func: Callable = None,
) -> list[dict]:
    """对单个 cluster 的边缘案例执行混合检索，返回 top-N 最相关的案例。

    检索策略:
      1. TF-IDF: jieba 分词后计算 token overlap cosine（轻量级，无需预构建索引）
      2. Dense Embedding: 若提供 embedding_func，计算语义相似度
      3. 两路分数相加，取 top-N
    """
    if not edge_cases or top_n <= 0:
        return []
    if len(edge_cases) <= top_n:
        return edge_cases

    tokenizer = _build_jieba_tokenizer()
    q_tokens = tokenizer(question)
    q_token_set = set(q_tokens)

    tfidf_scores: list[float] = []
    ec_question_texts: list[str] = []
    for ec in edge_cases:
        inp = ec.get("input", {})
        ec_q = inp.get("question", ec.get("question", ""))
        ec_a = inp.get("answer", ec.get("answer", ""))
        ec_question_texts.append(ec_q)
        ec_text = f"{ec_q} {ec_a}"
        ec_token_set = set(tokenizer(ec_text))
        if not ec_token_set or not q_token_set:
            tfidf_scores.append(0.0)
        else:
            overlap = q_token_set & ec_token_set
            tfidf_scores.append(
                len(overlap) / math.sqrt(len(q_token_set) * len(ec_token_set))
            )

    dense_scores = [0.0] * len(edge_cases)
    if embedding_func is not None:
        try:
            all_texts = [question] + ec_question_texts
            embeddings = embedding_func(all_texts)
            q_emb = embeddings[0]
            for i in range(len(edge_cases)):
                dense_scores[i] = _cosine_dense(q_emb, embeddings[i + 1])
        except Exception as e:
            print(f"[Retrieval] Edge case dense embedding 失败: {e}")

    combined = [
        (i, tfidf_scores[i] + dense_scores[i])
        for i in range(len(edge_cases))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return [edge_cases[i] for i, _ in combined[:top_n]]


def format_edge_cases_text(
    edge_cases: list[dict],
    level1_map: dict[int, str] | None = None,
) -> str:
    """将边缘案例列表格式化为 LLM 可读的参考文本（含 Level-1 Know-How）。"""
    if not edge_cases:
        return ""

    parts = []
    for i, ec in enumerate(edge_cases, 1):
        inp = ec.get("input", {})
        q = inp.get("question", ec.get("question", ""))
        a = inp.get("answer", ec.get("answer", ""))
        lines = [f"--- 参考案例 {i} ---"]
        if q:
            lines.append(f"问题: {q}")
        if a:
            lines.append(f"答案: {a}")
        if level1_map:
            ec_idx = ec.get("index")
            if ec_idx is not None and ec_idx in level1_map:
                lines.append(f"关联知识 (Level-1 Know-How): {level1_map[ec_idx]}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ─── QA 直检：原始 QA 对 + Level-1 Know-How 并行检索 ──────────────────────────

class QADirectRetriever:
    """对单个 QA knowledge 目录的原始 QA 对进行加载与检索。

    从 knowledge_traceback.json 加载 Level-1 成功提炼的 QA 对，
    支持 TF-IDF + Dense 混合检索，返回与用户问题最相关的原始 QA。
    """

    def __init__(
        self,
        knowledge_dir: str,
        embedding_func: Callable = None,
    ):
        self.knowledge_dir = knowledge_dir
        self.dir_name = os.path.basename(knowledge_dir)
        self.qa_pairs: list[dict] = []

        traceback_path = os.path.join(knowledge_dir, "knowledge_traceback.json")
        if not os.path.exists(traceback_path):
            return
        try:
            with open(traceback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        for _key, entry in data.items():
            if entry.get("status") != "success":
                continue
            kh = entry.get("Know_How", "")
            if not kh or not kh.strip():
                continue
            inp = entry.get("input", {})
            q = inp.get("question", "")
            a = inp.get("answer", "")
            if not q:
                continue
            self.qa_pairs.append({
                "index": entry.get("index", -1),
                "question": q,
                "answer": a,
                "know_how": kh,
                "extra_info": inp.get("Extra_Information", ""),
            })

        if not self.qa_pairs:
            return

        try:
            self._tokenizer = _build_jieba_tokenizer()
        except Exception:
            self._tokenizer = _build_charwb_tokenizer()

        self._qa_token_sets: list[set[str]] = []
        for p in self.qa_pairs:
            tokens = self._tokenizer(f"{p['question']} {p['answer']}")
            self._qa_token_sets.append(set(tokens))

        self._qa_embeddings: list[list[float]] | None = None
        if embedding_func is not None and self.qa_pairs:
            try:
                texts = [f"{p['question']} {p['answer']}" for p in self.qa_pairs]
                self._qa_embeddings = embedding_func(texts)
            except Exception as e:
                print(f"[QADirect] Dense embedding 预计算失败 ({self.dir_name}): {e}")

    def search(
        self,
        question: str,
        top_n: int = 3,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        """混合检索 top-N 最相关的原始 QA 对。"""
        if not self.qa_pairs or top_n <= 0:
            return []

        q_tokens = set(self._tokenizer(question))

        tfidf_scores = []
        for ts in self._qa_token_sets:
            if not ts or not q_tokens:
                tfidf_scores.append(0.0)
            else:
                overlap = q_tokens & ts
                tfidf_scores.append(
                    len(overlap) / math.sqrt(len(q_tokens) * len(ts))
                )

        dense_scores = [0.0] * len(self.qa_pairs)
        if query_embedding is not None and self._qa_embeddings is not None:
            for i, emb in enumerate(self._qa_embeddings):
                dense_scores[i] = _cosine_dense(query_embedding, emb)

        combined = [
            (i, tfidf_scores[i] + dense_scores[i])
            for i in range(len(self.qa_pairs))
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in combined[:top_n]:
            p = self.qa_pairs[idx]
            results.append({
                "source_dir": self.dir_name,
                "knowledge_dir": self.knowledge_dir,
                "qa_index": p["index"],
                "question": p["question"],
                "answer": p["answer"],
                "know_how": p["know_how"],
                "extra_info": p["extra_info"],
                "score": round(score, 6),
            })
        return results


def retrieve_qa_direct_candidates(
    question: str,
    qa_retrievers: list[QADirectRetriever],
    top_n: int = 3,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """对多个 QA 目录执行 QA 直检，汇总去重后返回全局 top-N。"""
    all_hits: list[dict] = []
    for ret in qa_retrievers:
        all_hits.extend(ret.search(question, top_n=top_n, query_embedding=query_embedding))

    all_hits.sort(key=lambda x: x["score"], reverse=True)

    seen: set[tuple] = set()
    deduplicated = []
    for h in all_hits:
        uid = (h["source_dir"], h["qa_index"])
        if uid in seen:
            continue
        seen.add(uid)
        deduplicated.append(h)

    return deduplicated[:top_n]


def format_qa_direct_text(qa_hit: dict) -> str:
    """将单条 QA 直检结果格式化为 LLM 可读的推理素材。"""
    lines = []
    lines.append(f"原始问题: {qa_hit['question']}")
    lines.append(f"原始答案: {qa_hit['answer']}")
    if qa_hit.get("extra_info"):
        lines.append(f"补充信息: {qa_hit['extra_info']}")
    lines.append(f"关联知识 (Level-1 Know-How): {qa_hit['know_how']}")
    return "\n".join(lines)
