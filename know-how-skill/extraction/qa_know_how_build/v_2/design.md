# QA Know-How 提取 V2 — 设计文档

## 1. 版本概述

V2 在 V1 的基础上进行了三项核心升级：

1. **Know-How 存在性过滤 + 案例库沉淀**：Level 1 提炼结果为空的样本不再丢弃，而是写入通用案例库供后续检索。
2. **基于 Cosine 相似度阈值的自适应聚类**：用 `AgglomerativeClustering` 替代 `KMeansConstrained`，以 cosine 相似度阈值（而非固定簇大小）控制聚类边界。
3. **质心驱动的增量精炼**：用结构化 JSON Schema 表示 Know-How，通过「质心生成 → 逐样本三档验证 → 增量补充 / 边缘案例递归聚类建块」的闭环迭代。

### V2.1 变更（知识块 Patch 逻辑重构）

- **三档语义重定义**：full/partial/none → answerable/augmentable/irrelevant
  - **answerable**：KH 完全可以推导出正确答案（含 reasoning 逻辑路径一致）→ 挂钩 `source_qa_ids`
  - **augmentable**：KH 方向正确但缺信息 → patch 补充后挂钩 `source_qa_ids`（patch 全失败则降级 irrelevant）
  - **irrelevant**：完全无关 / 核心结论矛盾 / 需大幅重构 KH 才能覆盖 reasoning → 挂钩 `edge_qa_ids`
- **双重溯源机制**：
  - KH 级别的 `source_qa_ids` / `edge_qa_ids` 数组（追踪整体覆盖）
  - 元素级 inline footnote `[1,2,3]`（step.outcome / exception.then 末尾，追踪每个细节源头）
- **Schema 变更**：移除顶层 `constraints`，合并到 step 的 `constraint` / `policy_basis` 字段
- **Exceptions 严格纪律**：仅当 QA 原文/reasoning 中明确提及时才抽取，LLM 严禁脑补
- **Reasoning 感知验证**：当训练数据提供 reasoning 时，answerable 要求推理路径也与 KH steps 一致
- **边缘案例递归聚类**：irrelevant 样本不再各自独立生成 KH，而是递归聚类 → 质心 → 验证/patch，生成与主 KH 平级的 know-how

### V2.2 变更（推理阶段分层 Reduce）

- **分层流式 Reduce**：Phase 4 从单层 Reduce 重构为多层递归归并架构
  - **Layer 1**（流式）：Map/Phase3 结果实时积累，达到 `--parallel-reduce-batch`（奇数，默认 3）时立即提交批次 Reduce
  - **Layer 2+**（递归归并）：Layer 1 中间结论 + 剩余结果递归分批 Reduce，直到 <= batch_size
  - **Final Reduce**：最终裁决，奇数投票制，偶数且对立时进行条件假设性分析
- **精准谨慎原则**：只采纳绝对必要的信息，法规时效性提醒不可省略
- **新增 Prompt**：`reduce_batch_v0`（中间层投票 Reduce）、`reduce_final_v0`（最终层含偶数对立处理）
- **新增 CLI 参数**：`--parallel-reduce-batch`（分层 Reduce 批次大小，必须为奇数）

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
  └──────┬──────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────────────┐
  │ Level 2: 多线程 per-cluster 增量精炼                 │
  │  1. 找质心样本 → 生成结构化 Know-How (JSON Schema)    │
  │     设置 source_qa_ids = [centroid_index]            │
  │     追加 inline footnote 到 outcome/then             │
  │  2. 其余样本按 cosine 降序逐个验证:                   │
  │     ├─ answerable   → source_qa_ids.append           │
  │     ├─ augmentable  → patch 补充 → source_qa_ids     │
  │     │   (patch 全失败 → 降级 irrelevant)              │
  │     └─ irrelevant   → edge_qa_ids.append             │
  │  3. 步骤编号归一化                                    │
  │  4. 边缘样本递归处理:                                  │
  │     ├─ ≥2 个样本 → 二次聚类 → 子簇走同样流程          │
  │     ├─ 1 个样本 → 独立生成 KH                         │
  │     └─ 递归深度上限: 3 层                              │
  │     产出的所有 KH 与主 KH **平级**存储                  │
  └──────┬──────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │ 输出                                         │
  │  ├─ know_how_level2.json    (结构化 KH 集)   │
  │  │   每个簇含主 KH + edge_know_hows 列表     │
  │  │   (edge_know_hows 中的 KH 与主 KH 平级)  │
  │  ├─ general_cases.json      (通用案例库)      │
  │  └─ edge_cases.json         (边缘案例库)      │
  └─────────────────────────────────────────────┘
```

## 3. Know-How 结构化 Schema

```json
{
  "title": "简洁的方法论标题",
  "scope": "适用场景的一句话描述",
  "source_qa_ids": [0, 3, 5],
  "edge_qa_ids": [7, 12],
  "steps": [
    {
      "step": "1",
      "action": "具体操作描述",
      "condition": "触发/前置条件（可选，null）",
      "constraint": "约束条件（可选，null）",
      "policy_basis": "政策依据-文件名+文号（可选，null）",
      "outcome": "预期结果[0]（末尾为溯源角标，标注来源 QA 序号）"
    },
    {
      "step": "2.1",
      "action": "分支A的操作",
      "condition": "当满足条件A时",
      "constraint": null,
      "policy_basis": null,
      "outcome": null
    }
  ],
  "exceptions": [
    {
      "when": "异常/特殊条件",
      "then": "对应处理方式[0,3]"
    }
  ]
}
```

**设计原则**：
- **增量友好**：augmentable 时，LLM 只需在 steps/exceptions 中做原子级插入或修改。
- **分叉友好**：`step` 为 string 类型，支持数字+点号分级（如 "2.1", "2.2"），无需嵌套结构即可表达多路径决策树。
- **双重溯源**：
  - `source_qa_ids` / `edge_qa_ids`：KH 级别，追踪哪些 QA 被覆盖/未覆盖
  - `outcome` / `then` 末尾的 `[1,2,3]`：元素级别，追踪每个细节的具体 QA 来源
- **严格抽取**：`exceptions` 仅从 QA 原文提取，`constraint` / `policy_basis` 仅从原文提取，LLM 严禁脑补。

## 4. 三档验证语义

| 级别 | 含义 | 后续动作 |
|------|------|---------|
| **answerable** | KH 能严格推导出与标准答案一致的结论；当提供 reasoning 时，KH 的 steps 也能自然支撑相同推理链 | 挂钩 `source_qa_ids`，不修改 KH |
| **augmentable** | KH 方向正确，但缺少该样本提到的步骤/条件/例外；或结论正确但 steps 不足以覆盖 reasoning 推理链（小幅补充即可） | patch 补充后挂钩 `source_qa_ids`；patch 全失败则降级 irrelevant |
| **irrelevant** | KH 与该问题无关或核心结论矛盾；或需要大幅重构 KH 核心逻辑才能覆盖 reasoning 推理链 | 挂钩 `edge_qa_ids`，进入边缘递归流程 |

## 5. 边缘案例递归聚类

当簇内验证完成后，所有 irrelevant 样本（含 augmentable 降级的）进入递归处理：

```
edge_samples (≥2)
     │
  ┌──▼───────────────────────────────┐
  │ make_clusters() 二次聚类          │
  │ (复用同一套 TF-IDF / Embedding)  │
  └──┬───────────────────────────────┘
     │
  ┌──▼───────────────────────────────┐
  │ 每个子簇:                         │
  │  质心 → 结构化 KH                 │
  │  逐样本验证 → patch               │
  │  子簇内新 edge → 递归 (depth+1)   │
  └──┬───────────────────────────────┘
     │
  结果: 所有递归层级产出的 KH 扁平收集
        与主 KH 同级存储在 edge_know_hows[]
```

**递归终止条件**：
- 边缘样本 = 0：不处理
- 边缘样本 = 1：直接独立生成 KH
- 递归深度 > `_MAX_EDGE_RECURSE_DEPTH`（默认 3）：剩余样本各自独立生成 KH

## 6. 聚类策略

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

## 7. 案例库设计

### 7.1 通用案例库 (general_cases.json)

存储 Level 1 提炼后 Know_How 为空的样本。

### 7.2 边缘案例库 (edge_cases.json)

存储 irrelevant 样本的验证元数据（用于调试/审计）。

```json
{
  "cluster_0": {
    "edge_cases": [
      {
        "index": 42,
        "input": { "question": "...", "answer": "...", "extra_info": "..." },
        "inference_result": "...",
        "mismatch_reason": "..."
      }
    ]
  }
}
```

## 8. 模块结构

```
qa_know_how_build/v_2/
├── design.md           # 本文档
├── __init__.py
├── prompts_v2.py       # 结构化 KH 生成 / 推理验证 / 最小更新 prompt
├── clustering.py       # TF-IDF + AgglomerativeClustering
├── patch_engine.py     # 结构化补丁执行引擎 + inline footnote 溯源
├── case_store.py       # 通用案例库 + 边缘案例库管理
├── level2_refine.py    # 核心增量精炼逻辑 + 边缘案例递归聚类
└── pipeline.py         # 完整流水线入口 + CLI
```

## 9. LLM 调用矩阵

| 环节 | Prompt 函数 | 调用时机 | 输入 | 输出 |
|------|------------|---------|------|------|
| Level 1 | `single_v1` (复用 v1) | 每个样本 1 次 | QA 四元组 | 自由文本 Know_How |
| 结构化生成 | `structured_kh_generate` | 每个簇的质心 + 递归子簇质心 + 单独 edge 样本 | Level 1 Know_How + QA 上下文 | JSON Schema Know-How |
| 推理验证 | `kh_inference_validate` | 簇/子簇内每个非质心样本 1 次 | 结构化 KH + QA + reasoning(可选) | match_level + reasoning_alignment |
| 最小更新 | `kh_minimal_update` | augmentable 时 | 当前 KH + 样本 + 分析 | patch operations |
| 步骤归一化 | `kh_normalize_steps` | 每个簇/子簇精炼完成后 | 完整 KH JSON | 重新编号的 steps |
| 批次 Reduce | `reduce_batch_v0` | 推理阶段 Map 结果达到水位线时 | 用户问题 + 候选答案批次 | 投票结论 + 精炼推理 |
| 最终 Reduce | `reduce_final_v0` | 推理阶段递归归并完成后 | 用户问题 + 候选结论 | Synthesis_Analysis + Final_Answer |

## 10. 依赖

```
scikit-learn >= 1.0    # AgglomerativeClustering, TfidfVectorizer, cosine_similarity
numpy
jieba                  # 中文分词（可选，缺失时回退 char_wb）
pandas                 # 数据加载
```
