# 推理阶段（Inference）

## 概述

推理阶段基于 **检索增强推理（Retrieval-Augmented Reasoning）** 架构，将用户问题与抽取阶段产出的 Know-How 知识库进行匹配、校验和推理，最终生成高质量答案。

核心流程：**知识指定 → 双路检索（含 QA 直检并行）→ 并行 Map 推理（含有效性校验 + LLM 裸考）→ 边缘案例混合检索兜底 → Reduce 融合**

---

## 整体架构

```
用户输入
  │
  ├── 用户问题（question）
  └── 指定的 knowledge 目录列表（QA / Doc 可混合）
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: 双路并行检索 + QA 直检                   │
│                                                               │
│  Query Embedding 只计算一次，所有目录共享                        │
│                                                               │
│  ┌── 路径 A: Level-2 知识块检索 ──────────────────────┐        │
│  │  对每个 knowledge 目录（多目录并行）:                 │        │
│  │    ├─ TF-IDF cosine → Top-N_a                      │        │
│  │    └─ Dense Embedding cosine → Top-N_b             │        │
│  │  → 候选知识块集合                                    │        │
│  └──────────────────────────────────────────────────┘        │
│                                                               │
│  ┌── 路径 B: QA 直检（仅 QA 目录）────────────────────┐        │
│  │  从 knowledge_traceback.json 加载原始 QA 对          │        │
│  │  文档侧检索文本: Q + A + Level-1 Know-How           │        │
│  │    ├─ jieba token overlap cosine                   │        │
│  │    └─ Dense Embedding cosine                       │        │
│  │  → Top-N 原始 QA 对（附带 Level-1 Know-How）        │        │
│  └──────────────────────────────────────────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: 并行 Map 推理                            │
│                                                               │
│  Level-2 知识块、QA 直检、LLM 裸考 同时并行进入 Map:              │
│                                                               │
│  ┌── Level-2 Map ───────────────────────────┐                │
│  │  对每个候选知识块【独立并行】:               │                │
│  │    1. LLM 推理验证（有效性校验）             │                │
│  │    2. 有效 → 推理链 + 候选答案              │                │
│  │    3. 无效 → QA 类型进入 Phase 3 兜底       │                │
│  └────────────────────────────────────────┘                │
│                                                               │
│  ┌── QA 直检 Map ──────────────────────────┐                │
│  │  对每条 QA 直检命中【独立并行】:             │                │
│  │    1. LLM 基于原始 QA + Level-1 推理验证    │                │
│  │    2. 有效 → 推理链 + 候选答案              │                │
│  │    3. 无效 → 标记为不相关                   │                │
│  └────────────────────────────────────────┘                │
│                                                               │
│  ┌── LLM 裸考 Map（默认开启）────────────────┐                │
│  │  直接将原问题交给 LLM（无知识上下文）:        │                │
│  │    → 基于模型自身知识生成候选答案             │                │
│  │    → 产出推理链 + 候选答案，参与 Reduce       │                │
│  └────────────────────────────────────────┘                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      Phase 3: 边缘案例混合检索兜底（仅 QA Know-How）             │
│                                                               │
│  当某条 QA Know-How 被判定为对当前问题无效时:                      │
│    1. 加载该 cluster 的全部边缘案例                               │
│    2. 加载关联的 Level-1 Know-How                                │
│    3. 混合检索（token overlap + Dense，文本: Q+A+Know-How）→ Top-N│
│    4. 边缘案例 + Level-1 Know-How 交给 LLM 推理                  │
│    5. 有效 → 产出推理结果                                        │
│    6. 仍然无效 → 标记为不相关，不参与 Reduce                       │
│                                                               │
│  注: Doc Know-How 不走此兜底，Map 判定无效即终止                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 4: Reduce 融合推理                          │
│                                                               │
│  汇总所有有效推理结果:                                           │
│  （Level-2 Map + QA 直检 + LLM 裸考 + 边缘案例兜底）             │
│    1. 综合各来源的候选答案推理链（标注来源类型）                    │
│    2. 融合分析，消解矛盾                                        │
│    3. 输出最终答案                                              │
└─────────────────────────────────────────────────────────────┘
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
    --tfidf-top-n 5 \                      # 所有双路检索共享的 TF-IDF Top-N（默认: 5）
    --embedding-top-n 5 \                  # 所有双路检索共享的 Dense Embedding Top-N（默认: 5）
    --max-workers 4 \                      # 并发线程数（默认: 4）
    --no-edge-cases \                      # 禁用 Phase 3 边缘案例兜底（默认开启）
    --no-qa-direct \                       # 禁用 QA 直检并行路径（默认开启）
    --no-extra-llm                         # 禁用 Reduce 阶段额外 LLM 裸考推理（默认开启）
```

---

## 代码结构

```
inference/
├── run_infer.py           # CLI 入口脚本
├── retrieval.py           # Phase 1: 双路并行检索模块 + Phase 3: 边缘案例混合检索
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
| `retrieval.py` | 加载 `retrieval_index.json`，执行 TF-IDF + Dense 双路检索，多目录并行，query embedding 单次计算；Phase 3 边缘案例混合检索（`retrieve_edge_cases`）；Level-1 Know-How 加载（`load_level1_knowhow_map`）；QA 直检（`QADirectRetriever`） |
| `mapreduce_infer.py` | 流水线编排；Level-2 Map 推理、QA 直检 Map 推理、边缘案例混合检索兜底（含 Level-1 Know-How）、Reduce 融合；文件 I/O（CSV/Excel 双格式） |
| `prompts_infer.py` | `infer_v0/v1`（Map 推理）、`qa_direct_infer_v0`（QA 直检 Map 推理）、`edge_case_fallback_v0`（Phase 3 兜底）、`summary_v0`（Reduce 融合）、`potential_pitfalls`（陷阱提示） |
| `run_infer.py` | CLI 参数解析、路径解析、LLM/Embedding 服务加载、调用推理流水线 |

---

## 各阶段详细说明

### Phase 1: 双路并行检索 + QA 直检

#### 路径 A: Level-2 知识块检索

加载每个 knowledge 目录中预构建的 `retrieval_index.json`，执行两路检索：

| 路线 | 方法 | 擅长场景 | 输出 |
|------|------|---------|------|
| **A** | jieba 分词 → TF-IDF 向量 → cosine 相似度 | 精确关键词匹配（专有名词、术语） | Top-N_a 个知识块 |
| **B** | BGE-M3 Dense Embedding → cosine 相似度 | 语义匹配（同义表达、近义词） | Top-N_b 个知识块 |

**并行优化策略**：
- Query embedding **只调用一次** API，所有 knowledge 目录共享同一个查询向量
- 多个 knowledge 目录之间通过 `ThreadPoolExecutor` **并行检索**
- 两路结果取 **并集**，按 `(source_dir, entry_key)` **去重**，保留最高分

#### 路径 B: QA 直检（仅 QA 目录）

**设计动机**：Level-2 知识块是对原始 QA 的高度聚类精炼，这个抽象过程可能丢失某些具体场景的细节。QA 直检绕过 Level-2 抽象层，直接在原始 QA 对中检索与用户问题最相关的案例，为 Reduce 阶段提供一条独立的推理依据来源。

**执行逻辑**：
1. 从 `knowledge_traceback.json` 加载所有 Level-1 提炼成功的原始 QA 对（`QADirectRetriever`）
2. 通过 **混合检索**（jieba token overlap cosine + Dense Embedding cosine）筛选 Top-N 最相关的 QA 对；两路检索统一使用 **Q + A + Level-1 Know-How** 作为文档侧检索文本，Level-1 Know-How 的泛化表述有助于弥合用户问法与原始 QA 之间的词汇鸿沟
3. 每条 QA 对自带其 **Level-1 Know-How**（一级提炼的泛化知识），作为辅助推理上下文
4. 多目录结果汇总去重，按综合得分取全局 Top-N

**与 Level-2 检索的关系**：两条路径 **完全并行** 执行，互不依赖。QA 直检的结果与 Level-2 候选知识块一同进入 Phase 2 的 Map 推理。

**适用范围**：
- **仅 QA 目录** 参与 QA 直检（需存在 `knowledge_traceback.json`）
- **Doc 目录不参与**（无原始 QA 对）
- 可通过 `--qa-direct-top-n 0` 禁用

### Phase 2: 并行 Map 推理

Level-2 知识块、QA 直检结果和 LLM 裸考**在同一个线程池中同时并行**执行 Map 推理：

#### Level-2 Map（与原有逻辑一致）

对候选集中的每个知识块，通过 `ThreadPoolExecutor` **独立并行** 执行：

1. **构造推理 prompt**：将用户问题 + 知识块内容组合
2. **LLM 推理验证**：模型同时完成两件事：
   - 判断该知识块是否与用户问题相关（有效性校验）
   - 如果相关，基于知识块推导候选答案
3. **输出**：
   - `有效`（Match_Status=YES）：返回推理链 + 候选答案 → 直接参与 Reduce
   - `无效`（Match_Status=NO）：返回拒绝原因 → QA 类型进入 Phase 3

#### QA 直检 Map

对每条 QA 直检命中，**独立并行** 执行：

1. **构造推理 prompt**：将用户问题 + 原始 QA 对 + Level-1 Know-How 组合（`qa_direct_infer_v0`）
2. **LLM 推理验证**：判断该原始 QA 及其泛化知识是否能回答用户问题
3. **输出**：
   - `有效`（Match_Status=YES）：返回推理链 + 候选答案 → 直接参与 Reduce（标记来源为"QA直检"）
   - `无效`（Match_Status=NO）：标记为不相关，不参与 Reduce（QA 直检无二次兜底）

#### LLM 裸考 Map（默认开启）

**设计动机**：知识库检索依赖于抽取阶段的覆盖范围，可能存在知识盲区。直接利用大模型自身的预训练知识作为一条独立的并行推理渠道，能为 Reduce 提供额外的参考视角，尤其在知识库覆盖不足时起到兜底作用。

**执行逻辑**：
1. 直接将原始用户问题交给 LLM（不提供任何知识上下文），请求其以"资深行业顾问"角色进行全面解答
2. 结果作为一条独立的候选推理（标记来源为"LLM裸考"），与其他 Map 结果一同参与 Reduce 融合

**与知识块 Map 的关系**：LLM 裸考与 Level-2 Map、QA 直检 Map **完全并行** 执行，互不依赖。在 Reduce 阶段，知识库来源的推理结果优先级高于裸考结果（Reduce prompt 中知识库来源标注为"知识块"/"QA直检"/"边缘案例兜底"，裸考标注为"LLM裸考"）。

**适用范围**：可通过 `--no-extra-llm` 禁用

### Phase 3: 边缘案例混合检索兜底（仅 QA Know-How）

**触发条件**：某条 QA Know-How 在 Phase 2 被判定为无效。

**设计动机**：抽取阶段的 Level 2 聚类精炼会将同簇样本压缩为结构化 Know-How，这个泛化过程可能丢失某些具体场景的细节。而被归为边缘案例（edge_cases）的原始 QA 样本恰好保留了这些未被抽象的具体知识——它们可能包含与用户问题直接相关的信息。

**执行逻辑**（通过 `ThreadPoolExecutor` 并行）：
1. 通过 `cluster_key` 从 `edge_cases.json` 中加载该 cluster 的全部边缘案例
2. 从 `knowledge_traceback.json` 加载 Level-1 Know-How 映射（`load_level1_knowhow_map`）
3. **混合检索**（jieba token overlap cosine + Dense Embedding cosine），筛选出与用户问题最相关的 **Top-N** 条边缘案例；两路检索统一使用 **Q + A + Level-1 Know-How** 作为文档侧检索文本，与 QA 直检保持一致
4. 将 Top-N 边缘案例（含 Level-1 Know-How）作为参考素材，连同用户问题一起交给 LLM（`edge_case_fallback_v0` prompt），再做一次推理验证
5. 有效 → 产出推理结果，参与 Reduce
6. 仍然无效 → 最终标记为不相关，不参与 Reduce

**与旧版的区别**：
- **旧版**：全量加载 cluster 下所有边缘案例，直接发给 LLM，无检索筛选，不使用 Level-1 Know-How
- **新版**：通过混合检索精选 Top-N 最相关的案例，减少噪声；检索和展示阶段均融入 Level-1 Know-How——检索时作为文档侧补充语义（Q + A + Know-How），推理时作为辅助上下文，提升召回率和推理准确性

**适用范围**：
- **仅 QA Know-How** 走此兜底路径（有 `edge_cases.json`）
- **Doc Know-How 不走边缘案例兜底**，Map 判定无效即为最终结论
- 可通过 `--edge-cases-top-n 0` 禁用

### Phase 4: Reduce 融合推理

汇总 Phase 2（Level-2 Map + QA 直检 Map + LLM 裸考 Map）和 Phase 3 中所有标记为「有效」的推理结果：

1. 整合来自不同知识源（Level-2 知识块、QA 直检、LLM 裸考、边缘案例兜底）的候选答案及推理链（每条标注来源类型）
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
├── knowledge_traceback.json # 一级回溯（Level-1 Know-How）     ← Phase 1 QA 直检 + Phase 3 边缘案例兜底（检索与推理均使用）
├── general_cases.json       # 通用案例库（仅 QA Know-How，备选参考）
└── knowledge.md             # 知识概述（人工审核用，推理不直接使用）
```

---

## 配置参数

| 参数 | CLI 参数 | 说明 | 默认值 |
|------|---------|------|--------|
| `knowledge_dirs` | `--knowledge-dirs` | 指定参与推理的 knowledge 目录列表 | 必填 |
| `tfidf_top_n` | `--tfidf-top-n` | 所有双路检索共享的 TF-IDF Top-N | 5 |
| `embedding_top_n` | `--embedding-top-n` | 所有双路检索共享的 Dense Embedding Top-N | 5 |
| `map_max_workers` | `--max-workers` | Map / Phase 3 并发线程数 | 4 |
| `question_column` | `--question-column` | 输入文件中问题列名 | question |
| `no_edge_cases` | `--no-edge-cases` | 禁用 Phase 3 边缘案例兜底 | False（默认开启） |
| `no_qa_direct` | `--no-qa-direct` | 禁用 QA 直检并行路径 | False（默认开启） |
| `no_extra_llm` | `--no-extra-llm` | 禁用 Reduce 阶段额外 LLM 裸考推理 | False（默认开启） |

---

## 输出列说明

在输入数据的原始列之后，追加以下结果列：

| 新增列 | 阶段 | 说明 |
|--------|------|------|
| `Retrieval_Candidates` | Phase 1 | 检索到的候选知识块数量 |
| `Map_Total_Evaluated` | Phase 2 | 实际评估的知识块数量 |
| `Map_Match_Count` | Phase 2 | 判定有效的知识块数量 |
| `QA_Direct_Count` | Phase 2 | QA 直检评估的原始 QA 对数量 |
| `QA_Direct_Match` | Phase 2 | QA 直检判定有效的数量 |
| `Edge_Fallback_Count` | Phase 3 | 边缘案例兜底尝试数 |
| `Edge_Fallback_Match` | Phase 3 | 兜底成功数 |
| `Total_Valid_Count` | 汇总 | 最终有效推理结果总数 |
| `Map_Valid_Details` | Phase 2+3 | 所有有效推理中间结果（JSON，含每条的推理链和候选答案） |
| `Map_Rejected_Reasons` | Phase 2 | 被拒绝的原因列表 |
| `Extra_Information` | Phase 4 | 额外参考信息（LLM 裸考推理结果） |
| `Reduce_Analysis` | Phase 4 | Reduce 融合分析过程 |
| `Final_Inference_Answer` | Phase 4 | **最终推理答案** |
