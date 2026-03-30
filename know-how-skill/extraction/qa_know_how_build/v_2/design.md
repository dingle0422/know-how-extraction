# QA Know-How 提取 V2 — 设计文档

## 1. 版本概述

V2 在 V1 的基础上进行了三项核心升级：

1. **Know-How 存在性过滤 + 案例库沉淀**：Level 1 提炼结果为空的样本不再丢弃，而是写入通用案例库供后续检索。
2. **基于 Cosine 相似度阈值的自适应聚类**：用 `AgglomerativeClustering` 替代 `KMeansConstrained`，以 cosine 相似度阈值（而非固定簇大小）控制聚类边界。
3. **质心驱动的增量精炼**：用结构化 JSON Schema 表示 Know-How，通过「质心生成 → 逐样本推理验证 → 最小改动更新 / 边缘案例归档」的闭环迭代取代一次性 compression。

## 2. 整体流程

```
输入: QA 源文件 (.csv / .xlsx)
         │
  ┌──────▼──────────────────────────────────────┐
  │ Level 1: 逐样本提炼 (复用 v1 level1_extract) │
  └──────┬──────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │ 加载 Level 1 结果并分流                      │
  │  ├─ KH 非空 → valid_items                   │
  │  └─ KH 为空 → general_cases (写入案例库)     │
  └──────┬──────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │ 聚类: jieba+TF-IDF + AgglomerativeClustering │
  │  • metric='cosine'                           │
  │  • distance_threshold = 1 - cosine_threshold │
  │  • linkage='average'                         │
  └──────┬──────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────────────┐
  │ Level 2: 多线程 per-cluster 增量精炼                 │
  │  1. 找质心样本 → 生成结构化 Know-How (JSON Schema)    │
  │  2. 其余样本按 cosine 降序逐个验证:                   │
  │     ├─ full    → 跳过                               │
  │     ├─ partial → 最小改动更新 Know-How                │
  │     └─ none    → 写入边缘案例库                       │
  └──────┬──────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │ 输出                                         │
  │  ├─ know_how_level2.json    (结构化 KH 集)   │
  │  ├─ general_cases.json      (通用案例库)      │
  │  └─ edge_cases.json         (边缘案例库)      │
  └─────────────────────────────────────────────┘
```

## 3. Know-How 结构化 Schema

```json
{
  "title": "简洁的方法论标题",
  "scope": "适用场景的一句话描述",
  "steps": [
    {
      "step": "1",
      "action": "具体操作描述（尽量一句话）",
      "condition": null,
      "outcome": null
    },
    {
      "step": "2a",
      "action": "分支A的操作",
      "condition": "当满足条件A时",
      "outcome": null
    },
    {
      "step": "2b",
      "action": "分支B的操作",
      "condition": "当满足条件B时",
      "outcome": null
    }
  ],
  "exceptions": [
    {
      "when": "异常/特殊条件",
      "then": "对应处理方式"
    }
  ],
  "constraints": ["关键约束或注意事项"]
}
```

**设计原则**：
- **增量友好**：partial match 时，LLM 只需在 steps/exceptions/constraints 中做原子级插入或修改。
- **分叉友好**：`step` 为 string 类型，支持子标签分叉（"2a", "2b"），无需嵌套结构即可表达多路径决策树。
- **篇幅可控**：每个字段都是精炼短句/列表项，不会产生长段落。
- **推理友好**：steps 是有序 SOP（含分叉），exceptions 是罕见异常，天然适合闭卷推理。

## 4. 聚类策略

| 维度 | V1 | V2 |
|------|-----|-----|
| 算法 | `KMeansConstrained` | `AgglomerativeClustering` |
| 控制参数 | `batch_size`（固定簇大小上限） | `cosine_threshold`（簇内相似度下限） |
| 簇数 | `ceil(n / batch_size)` | 由算法自动确定 |
| 簇大小 | ≤ batch_size | 不固定，由数据分布决定 |
| 向量化 | jieba + TF-IDF (512 features) | 同 V1 |

核心参数：
- `cosine_threshold`: 默认 0.75。对应 `distance_threshold = 1 - 0.75 = 0.25`。
- `linkage`: 默认 `'average'`。若簇太松散可切换为 `'complete'`。

## 5. 增量精炼算法

```
function refine_cluster(cluster):
    centroid_sample = find_nearest_to_centroid(cluster)
    know_how = generate_structured_kh(centroid_sample)
    edge_cases = []

    others = cluster.remove(centroid_sample)
    others.sort_by(cosine_sim_to_centroid, descending)

    for sample in others:
        result = validate_with_kh(know_how, sample)
        switch result.match_level:
            case "full":
                skip
            case "partial":
                know_how = minimal_update(know_how, sample, result)
            case "none":
                edge_cases.append({
                    sample, result.derived_answer, result.mismatch_reason
                })

    return know_how, edge_cases
```

## 6. 案例库设计

### 6.1 通用案例库 (general_cases.json)

存储 Level 1 提炼后 Know_How 为空的样本（无法升维提炼的原始数据）。

```json
{
  "source_file": "xxx.csv",
  "cases": [
    {
      "index": 5,
      "question": "...",
      "answer": "...",
      "extra_info": "...",
      "reason": "Level 1 未提炼出可泛化的 Know-How"
    }
  ]
}
```

### 6.2 边缘案例库 (edge_cases.json)

存储增量精炼时完全不匹配的样本，按 Know-How 节点挂钩。

```json
{
  "cluster_0": {
    "edge_cases": [
      {
        "index": 42,
        "input": { "question": "...", "answer": "...", "extra_info": "..." },
        "inference_result": "用 know-how 推理得到的结果",
        "mismatch_reason": "LLM 给出的不匹配原因简述"
      }
    ]
  }
}
```

**挂钩机制**：`cluster_0` 与 `level2_refinement.json` 中的 key `"0"` 一一对应（同次 pipeline 产出），通过索引即可回溯到完整的 Know-How 对象、质心样本、被吸收样本等全部信息。

当某个 Know-How 节点的边缘案例达到阈值后，人工审核决定是否启动新一轮提炼。

## 7. 模块结构

```
qa_know_how_build/v_2/
├── design.md           # 本文档
├── __init__.py
├── prompts_v2.py       # 结构化 KH 生成 / 推理验证 / 最小更新 prompt
├── clustering.py       # TF-IDF + AgglomerativeClustering
├── case_store.py       # 通用案例库 + 边缘案例库管理
├── level2_refine.py    # 核心增量精炼逻辑
└── pipeline.py         # 完整流水线入口 + CLI
```

## 8. LLM 调用矩阵

| 环节 | Prompt 函数 | 调用时机 | 输入 | 输出 |
|------|------------|---------|------|------|
| Level 1 | `single_v1` (复用 v1) | 每个样本 1 次 | QA 四元组 | 自由文本 Know_How |
| 结构化生成 | `structured_kh_generate` | 每个簇的质心样本 1 次 | Level 1 Know_How + QA 上下文 | JSON Schema Know-How |
| 推理验证 | `kh_inference_validate` | 簇内每个非质心样本 1 次 | 结构化 KH + QA | match_level + derived_answer |
| 最小更新 | `kh_minimal_update` | partial match 时 | 当前 KH + 样本 + 分析 | 更新后的 KH |

## 9. 依赖

```
scikit-learn >= 1.0    # AgglomerativeClustering, TfidfVectorizer, cosine_similarity
numpy
jieba                  # 中文分词（可选，缺失时回退 char_wb）
pandas                 # 数据加载
```

相比 V1 移除了 `k-means-constrained` 依赖。
