# 推理阶段（Inference）方案设计

## 概述

推理阶段基于 **检索增强推理（Retrieval-Augmented Reasoning）** 架构，将用户问题与抽取阶段产出的 Know-How 知识库进行匹配、校验和推理，最终生成高质量答案。

核心流程：**知识指定 → 双路检索 → 并行 Map 推理（含有效性校验）→ 边缘案例兜底 → Reduce 融合**

---

## 整体架构

```
用户输入
  │
  ├── 用户问题（question）
  └── 指定的 knowledge 目录列表（QA / Doc 可混合）
         │
         ▼
┌─────────────────────────────────────────────────┐
│           Phase 1: 双路并行检索                    │
│                                                   │
│  对每个指定的 knowledge 目录:                       │
│    ├─ 路线 A: TF-IDF cosine → Top-N_a            │
│    └─ 路线 B: Dense Embedding cosine → Top-N_b   │
│                                                   │
│  所有 knowledge 目录的检索结果取并集（去重）          │
│  → 候选知识块集合                                   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Phase 2: 并行 Map 推理                   │
│                                                   │
│  对每个候选知识块【独立并行】:                        │
│    1. LLM 推理验证（知识有效性校验）                  │
│    2. 有效 → 产出推理结果 + 候选答案                 │
│    3. 无效 → 进入 Phase 3 边缘案例兜底              │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│     Phase 3: 边缘案例兜底（仅 QA Know-How）         │
│                                                   │
│  当某条 QA Know-How 被判定为对当前问题无效时:         │
│    1. 取该 cluster 的边缘案例（原始 QA 样本）         │
│    2. 边缘案例作为参考素材交给 LLM 再做推理验证        │
│       （避免模型强行使用不相关知识）                    │
│    3. 有效 → 产出推理结果                            │
│    4. 仍然无效 → 标记为不相关，不参与 Reduce          │
│                                                   │
│  注: Doc Know-How 不走此兜底，Map 判定无效即终止       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Phase 4: Reduce 融合推理                  │
│                                                   │
│  汇总所有有效 Map 推理结果:                          │
│    1. 综合各候选答案的推理链                          │
│    2. 融合分析，消解矛盾                             │
│    3. 输出最终答案                                   │
└─────────────────────────────────────────────────┘
```

---

## 各阶段详细说明

### Phase 1: 双路并行检索

用户指定 extraction 目录下的 knowledge 文件夹（QA 和 Doc 均可），程序加载每个 knowledge 目录中预构建的 `retrieval_index.json`，并行执行两路检索：

| 路线 | 方法 | 擅长场景 | 输出 |
|------|------|---------|------|
| **A** | jieba 分词 → TF-IDF 向量 → cosine 相似度 | 精确关键词匹配（专有名词、术语） | Top-N_a 个知识块 |
| **B** | BGE-M3 Dense Embedding → cosine 相似度 | 语义匹配（同义表达、近义词） | Top-N_b 个知识块 |

- 两路 **独立并行** 执行，各自返回指定数量的 Top-N 结果
- 两路结果取 **并集**，按知识块唯一标识（knowledge 来源 + entry key）**去重**，避免同一知识块被重复送入 Map 推理
- N_a 和 N_b 可独立配置，默认值待定

### Phase 2: 并行 Map 推理

对候选集中的每个知识块，**独立并行** 执行以下逻辑：

1. **构造推理 prompt**：将用户问题 + 知识块内容组合
2. **LLM 推理验证**：模型同时完成两件事：
   - 判断该知识块是否与用户问题相关（有效性校验）
   - 如果相关，基于知识块推导候选答案
3. **输出标记**：
   - `有效`：返回推理链（Reasoning Chain）+ 候选答案（Derived Answer）
   - `无效`：返回拒绝原因（Rejection Reason），进入 Phase 3

### Phase 3: 边缘案例兜底（仅 QA Know-How）

**触发条件**：某条 QA Know-How 在 Phase 2 被判定为无效。

**设计动机**：抽取阶段的 Level 2 聚类精炼会将同簇样本压缩为结构化 Know-How，这个泛化过程可能丢失某些具体场景的细节。而被归为边缘案例（edge_cases）的原始 QA 样本恰好保留了这些未被抽象的具体知识——它们可能包含与用户问题直接相关的信息。

**执行逻辑**：
1. 通过 `cluster_key` 从 `edge_cases.json` 中取出该 cluster 的边缘案例（原始 QA 样本）
2. 将边缘案例作为 **参考素材** 连同用户问题一起交给 LLM，再做一次推理验证
   - 边缘案例仅供 LLM 参考判断，避免模型在缺乏上下文时强行使用不相关知识
   - LLM 仍需独立判定是否与用户问题相关
3. 有效 → 产出推理结果，参与 Reduce
4. 仍然无效 → 最终标记为不相关，不参与 Reduce

**适用范围**：
- **仅 QA Know-How** 走此兜底路径。QA 抽取阶段的 Level 2 聚类精炼会产生 edge_cases.json，其中保留了未被结构化吸收的具体 QA 样本。
- **Doc Know-How 不走边缘案例兜底**。Doc 抽取阶段产生的 waste_backup.json 不用于推理阶段，Map 判定无效即为最终结论。

### Phase 4: Reduce 融合推理

汇总 Phase 2 和 Phase 3 中所有标记为「有效」的推理结果：

1. 整合来自不同知识源（不同 QA 文件、不同文档）的候选答案
2. 分析各推理链的一致性与互补性
3. 消解可能的矛盾结论
4. 融合生成最终答案

---

## 数据依赖

推理阶段依赖抽取阶段在每个 knowledge 目录下产出的以下文件：

```
{source_stem}_knowledge/
├── knowledge.json           # 最终知识库（知识块内容）
├── retrieval_index.json     # 检索索引（TF-IDF + Dense Embedding 预计算向量）
├── edge_cases.json          # 边缘案例库（仅 QA Know-How，Phase 3 兜底用）
├── general_cases.json       # 通用案例库（仅 QA Know-How，备选参考）
├── knowledge.md             # 知识概述（人工审核用，推理不直接使用）
└── knowledge_traceback.json # 一级回溯（调试/审计用，推理不直接使用）
```

---

## 配置参数（预期）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `knowledge_dirs` | 指定参与推理的 knowledge 目录列表 | 必填 |
| `tfidf_top_n` | 路线 A (TF-IDF) 检索返回数量 | 待定 |
| `embedding_top_n` | 路线 B (Dense Embedding) 检索返回数量 | 待定 |
| `map_max_workers` | Map 阶段并发线程数 | 4 |
| `reduce_llm_func` | Reduce 阶段使用的 LLM | 同 Map |
| `enable_edge_case_fallback` | 是否启用边缘案例兜底 | True |



# 输出新增列
| 新增列                   | 阶段      | 说明                                  |
|------------------------|----------|-------------------------------------|
| Retrieval_Candidates   | Phase 1  | 检索到的候选知识块数量                  |
| Map_Total_Evaluated    | Phase 2  | 实际评估的知识块数量                    |
| Map_Match_Count        | Phase 2  | 判定有效的知识块数量                    |
| Edge_Fallback_Count    | Phase 3  | 边缘案例兜底尝试数                      |
| Edge_Fallback_Match    | Phase 3  | 兜底成功数                              |
| Total_Valid_Count      | 汇总      | 最终有效推理结果总数                     |
| Map_Valid_Details      | Phase 2+3| 所有有效推理中间结果（JSON）             |
| Map_Rejected_Reasons   | Phase 2  | 被拒绝的原因列表                        |
| Extra_Information      | Phase 4  | 额外参考信息                            |
| Reduce_Analysis        | Phase 4  | Reduce 融合分析                         |
| Final_Inference_Answer | Phase 4  | 最终推理答案                            |