# 推理阶段（Inference）

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
│  Query Embedding 只计算一次，所有目录共享            │
│  对每个指定的 knowledge 目录（多目录并行）:           │
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
│    2. 有效 → 产出推理链 + 候选答案                   │
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

## 快速上手

### 运行方式

从 `know-how-skill/` 目录执行：

```bash
python inference/run_infer.py \
    --input 借款业务评测集_324.csv \
    --knowledge-dirs \
        extraction/qa_know_how_build/knowledge/aaa_knowledge \
        extraction/doc_know_how_build/knowledge/bbb_knowledge
```

### 输入/输出约定

- **输入**：放在 `inference/input/` 目录下，支持 `.csv` 和 `.xlsx`
- **输出**：自动生成到 `inference/output/`，格式与输入一致，文件名带时间戳
- `--knowledge-dirs` 支持多个目录，QA 和 Doc 类型可自由混合

### 完整参数

```bash
python inference/run_infer.py \
    --input test.xlsx \                    # 输入文件（相对于 input/ 或绝对路径）
    --knowledge-dirs kd1 kd2 \             # knowledge 目录列表（必填，支持多个）
    --output result.xlsx \                 # 输出路径（可选，默认自动生成）
    --output-format xlsx \                 # 输出格式（可选，默认与输入一致）
    --question-column question \           # 问题列名（默认: question）
    --tfidf-top-n 5 \                      # TF-IDF 检索 Top-N（默认: 5）
    --embedding-top-n 5 \                  # Dense Embedding 检索 Top-N（默认: 5）
    --max-workers 4 \                      # 并发线程数（默认: 4）
    --no-edge-fallback \                   # 禁用 Phase 3 边缘案例兜底
    --no-extra-llm                         # 禁用 Reduce 阶段额外 LLM 信息
```

---

## 代码结构

```
inference/
├── run_infer.py           # CLI 入口脚本
├── retrieval.py           # Phase 1: 双路并行检索模块
├── mapreduce_infer.py     # Phase 2/3/4: MapReduce 推理流水线
├── prompts_infer.py       # Prompt 模板（Map / Reduce / 边缘案例兜底）
├── __init__.py            # 模块导出
├── README.md              # 本文件
├── input/                 # 待推理数据存放目录
│   └── *.csv / *.xlsx
└── output/                # 推理结果输出目录
    └── *_result_MMDD_HHMM.csv / .xlsx
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `retrieval.py` | 加载 `retrieval_index.json`，执行 TF-IDF + Dense 双路检索，多目录并行，query embedding 单次计算 |
| `mapreduce_infer.py` | 4 阶段流水线编排；Map 推理、边缘案例兜底、Reduce 融合；文件 I/O（CSV/Excel 双格式） |
| `prompts_infer.py` | `infer_v0/v1`（Map 推理）、`edge_case_fallback_v0`（Phase 3 兜底）、`summary_v0`（Reduce 融合）、`potential_pitfalls`（陷阱提示） |
| `run_infer.py` | CLI 参数解析、路径解析、LLM/Embedding 服务加载、调用推理流水线 |

---

## 各阶段详细说明

### Phase 1: 双路并行检索

加载每个 knowledge 目录中预构建的 `retrieval_index.json`，执行两路检索：

| 路线 | 方法 | 擅长场景 | 输出 |
|------|------|---------|------|
| **A** | jieba 分词 → TF-IDF 向量 → cosine 相似度 | 精确关键词匹配（专有名词、术语） | Top-N_a 个知识块 |
| **B** | BGE-M3 Dense Embedding → cosine 相似度 | 语义匹配（同义表达、近义词） | Top-N_b 个知识块 |

**并行优化策略**：
- Query embedding **只调用一次** API，所有 knowledge 目录共享同一个查询向量
- 多个 knowledge 目录之间通过 `ThreadPoolExecutor` **并行检索**
- 两路结果取 **并集**，按 `(source_dir, entry_key)` **去重**，保留最高分

### Phase 2: 并行 Map 推理

对候选集中的每个知识块，通过 `ThreadPoolExecutor` **独立并行** 执行：

1. **构造推理 prompt**：将用户问题 + 知识块内容组合
2. **LLM 推理验证**：模型同时完成两件事：
   - 判断该知识块是否与用户问题相关（有效性校验）
   - 如果相关，基于知识块推导候选答案
3. **输出**：
   - `有效`（Match_Status=YES）：返回推理链（Reasoning_Chain）+ 候选答案（Derived_Answer）→ 直接参与 Reduce
   - `无效`（Match_Status=NO）：返回拒绝原因（Rejection_Reason）→ QA 类型进入 Phase 3

### Phase 3: 边缘案例兜底（仅 QA Know-How）

**触发条件**：某条 QA Know-How 在 Phase 2 被判定为无效。

**设计动机**：抽取阶段的 Level 2 聚类精炼会将同簇样本压缩为结构化 Know-How，这个泛化过程可能丢失某些具体场景的细节。而被归为边缘案例（edge_cases）的原始 QA 样本恰好保留了这些未被抽象的具体知识——它们可能包含与用户问题直接相关的信息。

**执行逻辑**（通过 `ThreadPoolExecutor` 并行）：
1. 通过 `cluster_key` 从 `edge_cases.json` 中取出该 cluster 的边缘案例
2. 将边缘案例作为 **参考素材** 连同用户问题一起交给 LLM（`edge_case_fallback_v0` prompt），再做一次推理验证
3. 有效 → 产出推理结果，参与 Reduce
4. 仍然无效 → 最终标记为不相关，不参与 Reduce

**适用范围**：
- **仅 QA Know-How** 走此兜底路径（有 `edge_cases.json`）
- **Doc Know-How 不走边缘案例兜底**，Map 判定无效即为最终结论
- 可通过 `--no-edge-fallback` 禁用

### Phase 4: Reduce 融合推理

汇总 Phase 2 和 Phase 3 中所有标记为「有效」的推理结果：

1. 整合来自不同知识源（不同 QA 文件、不同文档）的候选答案及推理链
2. 分析各推理链的一致性与互补性
3. 消解可能的矛盾结论
4. 融合生成最终答案

---

## 数据依赖

推理阶段依赖抽取阶段在每个 knowledge 目录下产出的以下文件：

```
{source_stem}_knowledge/
├── knowledge.json           # 最终知识库（知识块内容）          ← Phase 2 加载
├── retrieval_index.json     # 检索索引（TF-IDF + Dense 预计算向量）← Phase 1 加载
├── edge_cases.json          # 边缘案例库（仅 QA Know-How）     ← Phase 3 加载
├── general_cases.json       # 通用案例库（仅 QA Know-How，备选参考）
├── knowledge.md             # 知识概述（人工审核用，推理不直接使用）
└── knowledge_traceback.json # 一级回溯（调试/审计用，推理不直接使用）
```

---

## 配置参数

| 参数 | CLI 参数 | 说明 | 默认值 |
|------|---------|------|--------|
| `knowledge_dirs` | `--knowledge-dirs` | 指定参与推理的 knowledge 目录列表 | 必填 |
| `tfidf_top_n` | `--tfidf-top-n` | 路线 A (TF-IDF) 检索返回数量 | 5 |
| `embedding_top_n` | `--embedding-top-n` | 路线 B (Dense Embedding) 检索返回数量 | 5 |
| `map_max_workers` | `--max-workers` | Map / Phase 3 并发线程数 | 4 |
| `enable_edge_case_fallback` | `--no-edge-fallback` | 是否启用边缘案例兜底 | True |
| `question_column` | `--question-column` | 输入文件中问题列名 | question |

---

## 输出列说明

在输入数据的原始列之后，追加以下结果列：

| 新增列 | 阶段 | 说明 |
|--------|------|------|
| `Retrieval_Candidates` | Phase 1 | 检索到的候选知识块数量 |
| `Map_Total_Evaluated` | Phase 2 | 实际评估的知识块数量 |
| `Map_Match_Count` | Phase 2 | 判定有效的知识块数量 |
| `Edge_Fallback_Count` | Phase 3 | 边缘案例兜底尝试数 |
| `Edge_Fallback_Match` | Phase 3 | 兜底成功数 |
| `Total_Valid_Count` | 汇总 | 最终有效推理结果总数 |
| `Map_Valid_Details` | Phase 2+3 | 所有有效推理中间结果（JSON，含每条的推理链和候选答案） |
| `Map_Rejected_Reasons` | Phase 2 | 被拒绝的原因列表 |
| `Extra_Information` | Phase 4 | 额外参考信息（裸考 LLM） |
| `Reduce_Analysis` | Phase 4 | Reduce 融合分析过程 |
| `Final_Inference_Answer` | Phase 4 | **最终推理答案** |
