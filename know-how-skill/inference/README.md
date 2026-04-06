# 推理阶段（Inference）

## 概述

推理阶段基于 **检索增强推理（Retrieval-Augmented Reasoning）** 架构，将用户问题与抽取阶段产出的 Know-How 知识库进行匹配、校验和推理，最终生成高质量答案。

核心流程：**知识指定 → 双路检索（含 QA 直检锚点关联）→ 并行 Map 推理（含有效性校验 + LLM 裸考）→ 边缘 KH 结构化推理兜底 → 分层 Reduce 融合（流式批次 + 递归归并 + 投票制裁决）**

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
│         Phase 1: 双路并行检索 + QA 直检锚点关联                 │
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
│  ┌── 路径 B: QA 直检锚点（仅 QA 目录）────────────────┐        │
│  │  从 knowledge_traceback.json 加载原始 QA 对          │        │
│  │  文档侧检索文本: Q + A + Level-1 Know-How           │        │
│  │    ├─ jieba token overlap cosine                   │        │
│  │    └─ Dense Embedding cosine                       │        │
│  │  → Top-N 原始 QA 对                                │        │
│  │    ↓                                               │        │
│  │  通过 knowledge.json 反向映射回关联的 Level-2 知识块  │        │
│  │    ↓                                               │        │
│  │  融入路径 A 候选集（按 source_dir+entry_key 去重）   │        │
│  └──────────────────────────────────────────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: 并行 Map 推理                            │
│                                                               │
│  Level-2 知识块（含 QA 锚点关联的）、LLM 裸考 同时并行进入 Map:   │
│                                                               │
│  ┌── Level-2 Map ───────────────────────────┐                │
│  │  对每个候选知识块【独立并行】:               │                │
│  │    1. LLM 推理验证（有效性校验）             │                │
│  │    2. 有效 → 推理链 + 候选答案              │                │
│  │    3. 无效 → QA 类型进入 Phase 3 兜底       │                │
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
│      Phase 3: 边缘 KH 结构化推理兜底（仅 QA Know-How）              │
│                                                               │
│  当某条 QA Know-How 被判定为对当前问题无效时:                      │
│    1. 从 knowledge.json 加载该 cluster 的 edge_know_hows         │
│       （由提取阶段边缘案例递归聚类生成的结构化 KH）                  │
│    2. 每个 edge KH 走与 Phase 2 完全相同的 Map 推理流程            │
│       （使用 infer_prompt_func，非旧版 edge_case_fallback）       │
│    3. 有效 → 产出推理结果                                        │
│    4. 仍然无效 → 标记为不相关，不参与 Reduce                       │
│                                                               │
│  注: Doc Know-How 不走此兜底，Map 判定无效即终止                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      Phase 4: 分层 Reduce 融合推理（流式批次 + 递归归并）          │
│                                                               │
│  Layer 1（流式）: Map/Phase3 结果实时积累，达到水位线时立即        │
│    提交批次 Reduce（投票制裁决 + 推理精炼）                       │
│                                                               │
│  Layer 2+（递归归并）: Layer 1 中间结论 + 剩余结果递归分批         │
│    Reduce，直到结果数 <= batch_size                              │
│                                                               │
│  Final Reduce: 最终裁决，输出面向用户的答案                       │
│    奇数→投票制; 偶数且对立→条件假设性分析                          │
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
    --parallel-reduce-batch 3 \            # 分层 Reduce 批次大小（必须为奇数，默认: 3）
    --no-extra-llm                         # 禁用 Reduce 阶段额外 LLM 裸考推理（默认开启）
```

---

## 代码结构

```
inference/
├── run_infer.py           # CLI 入口脚本
├── retrieval.py           # Phase 1: 双路并行检索模块 + Phase 3: edge_know_hows 加载
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
| `retrieval.py` | 加载 `retrieval_index.json`，执行 TF-IDF + Dense 双路检索，多目录并行，query embedding 单次计算；Phase 3 边缘案例混合检索（`retrieve_edge_cases`）；Level-1 Know-How 加载（`load_level1_knowhow_map`）；QA→Cluster 反向映射（`build_qa_to_cluster_map`）；QA 直检（`QADirectRetriever`，作为锚点关联 Level-2 知识块） |
| `mapreduce_infer.py` | 流水线编排；Level-2 Map 推理（含 QA 锚点关联的知识块）、边缘 KH 结构化推理兜底、分层 Reduce 融合（流式 L1 + 递归归并 L2+ + 最终裁决）；文件 I/O（CSV/Excel 双格式） |
| `prompts_infer.py` | `infer_v0/v1`（Map 推理）、`edge_case_fallback_v0`（Phase 3 兜底，已废弃）、`summary_v0`（旧版 Reduce 融合）、`reduce_batch_v0`（中间层批次 Reduce）、`reduce_final_v0`（最终层 Reduce）、`potential_pitfalls`（陷阱提示） |
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

**检索文本策略（v1.1）**：

索引构建时，每条知识块的检索文本（用于 TF-IDF 和 Dense Embedding 向量化）采用 **title + scope + retrieval_keywords** 策略：

- **title + scope**：描述知识块"是关于什么的"，语义集中，避免长文本稀释
- **retrieval_keywords**：LLM 从知识内容中提取的 10-20 个面向检索的关键词，覆盖三类：
  1. **领域实体**：专业术语、政策名称、业务对象
  2. **场景动作**：动词/短语，描述用户可能遇到的操作场景
  3. **用户表达**：口语化说法和同义词，弥补知识块表述与用户提问之间的词汇鸿沟

关键词在索引构建时自动生成（需传入 `llm_func`），并持久化到 `knowledge.json` 的 `retrieval_keywords` 字段。
后续重建索引时自动复用已有关键词，无需重复调用 LLM。若知识块内容更新，删除对应条目的 `retrieval_keywords` 字段后重建索引即可触发重新生成。

若未提供 `llm_func` 且无已有关键词，则退回全字段拼接模式（向后兼容）。

**并行优化策略**：
- Query embedding **只调用一次** API，所有 knowledge 目录共享同一个查询向量
- 多个 knowledge 目录之间通过 `ThreadPoolExecutor` **并行检索**
- 两路结果取 **并集**，按 `(source_dir, entry_key)` **去重**，保留最高分

#### 路径 B: QA 直检锚点关联（仅 QA 目录）

**设计动机**：Level-2 知识块检索依赖于知识块文本与用户问题的直接匹配，但部分关键知识块的抽象表述可能与用户问法存在词汇鸿沟。QA 直检通过在原始 QA 对中检索，可以发现那些虽然 Level-2 文本未被直接命中、但原始 QA 场景与用户问题高度相关的知识块。QA 直检命中仅作为**锚点**，反向关联到其所属的 Level-2 知识块，扩充路径 A 的候选集。

**执行逻辑**：
1. 从 `knowledge_traceback.json` 加载所有 Level-1 提炼成功的原始 QA 对（`QADirectRetriever`）
2. 通过 **混合检索**（jieba token overlap cosine + Dense Embedding cosine）筛选 Top-N 最相关的 QA 对；两路检索统一使用 **Q + A + Level-1 Know-How** 作为文档侧检索文本
3. 通过 `knowledge.json` 中的 `absorbed_indices` / `edge_case_indices` **反向映射**，找到每条 QA 命中所属的 Level-2 知识块（entry_key）
4. 将这些关联的 Level-2 知识块**融入路径 A 候选集**，按 `(source_dir, entry_key)` **去重**（已存在的不重复添加）

**与 Level-2 检索的关系**：两条路径 **完全并行** 执行检索，但 QA 直检的最终产出是 Level-2 知识块（而非原始 QA 对）。所有候选知识块统一进入 Phase 2 的 Level-2 Map 推理，不存在独立的 QA 直检 Map 路径。

**适用范围**：
- **仅 QA 目录** 参与 QA 直检（需存在 `knowledge_traceback.json`）
- **Doc 目录不参与**（无原始 QA 对）
- 可通过 `--no-qa-direct` 禁用

### Phase 2: 并行 Map 推理

Level-2 知识块（含 QA 锚点关联的）和 LLM 裸考**在同一个线程池中同时并行**执行 Map 推理：

#### Level-2 Map

对候选集中的每个知识块（包括路径 A 直接检索和路径 B QA 锚点关联的），通过 `ThreadPoolExecutor` **独立并行** 执行：

1. **构造推理 prompt**：将用户问题 + 知识块内容组合
2. **LLM 推理验证**：模型同时完成两件事：
   - 判断该知识块是否与用户问题相关（有效性校验）
   - 如果相关，基于知识块推导候选答案
3. **输出**：
   - `有效`（Match_Status=YES）：返回推理链 + 候选答案 → 直接参与 Reduce
   - `无效`（Match_Status=NO）：返回拒绝原因 → QA 类型进入 Phase 3

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

### Phase 4: 分层 Reduce 融合推理

采用**分层流式 Reduce** 架构，解决大量 Map 结果一次性汇总导致的上下文过长、幻觉和信息丢失问题。

**Layer 1（流式批次 Reduce）**：
- 嵌入 Phase 2/3 的 `as_completed` 循环，实时监控有效推理结果的积累
- 当有效结果数量达到 `--parallel-reduce-batch`（默认 3，必须为奇数）时，立即提交一条独立 Reduce 线程
- 使用 `reduce_batch_v0` prompt 进行投票制裁决：多数派结论即为批次结论，仅保留与结论直接相关的推理细节
- 多个 Layer 1 Reduce 线程可并行执行

**Layer 2+（递归归并）**：
- 所有 Map/Phase3 结束后，收集 Layer 1 中间结论 + 剩余未凑满批次的原始结果
- 递归分批调用 `reduce_batch_v0` 进行归并，直到结果数 <= batch_size

**Final Reduce（最终裁决）**：
- 使用 `reduce_final_v0` prompt 输出面向用户的最终答案
- 奇数结论：多数派投票确定最终结论
- 偶数且对立：以谦逊审慎语气进行条件假设性分析
- **核心原则**：只采纳绝对必要的信息，精准谨慎，避免额外非必要信息引入逻辑漏洞；法规时效性提醒不可省略

---

## 数据依赖

推理阶段依赖抽取阶段在每个 knowledge 目录下产出的以下文件：

```
{source_stem}_knowledge/
├── knowledge.json           # 最终知识库（知识块内容 + retrieval_keywords）← Phase 1 QA 锚点反向映射 + Phase 2 加载
├── retrieval_index.json     # 检索索引（v1.1: TF-IDF + Dense + 检索策略标记）← Phase 1 加载
├── edge_cases.json          # 边缘案例库（仅 QA Know-How）     ← Phase 3 加载
├── knowledge_traceback.json # 一级回溯（Level-1 Know-How）     ← Phase 1 QA 直检锚点 + Phase 3 边缘案例兜底（检索与推理均使用）
├── general_cases.json       # 通用案例库（仅 QA Know-How，备选参考）
└── knowledge.md             # 知识概述（人工审核用，推理不直接使用）
```

> **v1.1 新增**：`knowledge.json` 中每条成功的知识条目新增 `retrieval_keywords` 字段（list[str]），
> 由索引构建阶段 LLM 自动生成，持久化供后续复用。`retrieval_index.json` 新增 `retrieval_strategy`
> 和 `version: "1.1"` 标记，`entries` 中每条记录新增 `retrieval_keywords` 以便调试追踪。

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
| `reduce_batch_size` | `--parallel-reduce-batch` | 分层 Reduce 批次大小（必须为奇数） | 3 |

---

## 输出列说明

在输入数据的原始列之后，追加以下结果列：

| 新增列 | 阶段 | 说明 |
|--------|------|------|
| `Retrieval_Candidates` | Phase 1 | 检索到的候选知识块数量 |
| `Map_Total_Evaluated` | Phase 2 | 实际评估的知识块数量 |
| `Map_Match_Count` | Phase 2 | 判定有效的知识块数量 |
| `QA_Direct_Count` | Phase 1 | QA 直检命中的原始 QA 锚点数量 |
| `QA_Direct_Match` | Phase 1 | 经 QA 锚点关联新增的 Level-2 知识块数量 |
| `Edge_Fallback_Count` | Phase 3 | 边缘案例兜底尝试数 |
| `Edge_Fallback_Match` | Phase 3 | 兜底成功数 |
| `Total_Valid_Count` | 汇总 | 最终有效推理结果总数 |
| `Map_Valid_Details` | Phase 2+3 | 所有有效推理中间结果（JSON，含每条的推理链和候选答案） |
| `Map_Rejected_Reasons` | Phase 2 | 被拒绝的原因列表 |
| `Extra_Information` | Phase 4 | 额外参考信息（LLM 裸考推理结果） |
| `Reduce_Analysis` | Phase 4 | Reduce 融合分析过程 |
| `Final_Inference_Answer` | Phase 4 | **最终推理答案** |
