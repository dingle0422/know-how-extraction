"""
V2 Prompts：结构化 Know-How 生成 / 推理验证 / 结构化补丁更新。
"""

import json as _json

# ═══════════════════════════════════════════════════════════════════════════════
#  Know-How JSON Schema（供 prompt 内联引用）
# ═══════════════════════════════════════════════════════════════════════════════

KH_JSON_SCHEMA = """{
  "title": "简洁的方法论标题",
  "scope": "适用场景的一句话描述",
  "steps": [
    {
      "step": "1",
      "action": "具体操作描述（尽量一句话）",
      "condition": "触发/前置条件（可选，无条件时为 null）",
      "outcome": "预期结果（可选，无则为 null）"
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
}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 1: 结构化 Know-How 生成（质心样本 → JSON Schema）
# ═══════════════════════════════════════════════════════════════════════════════

def structured_kh_generate(know_how_text: str, question: str, answer: str,
                           extra_info: str = "", reasoning: str = "") -> str:
    return f"""# Role
你是一台「知识结构化引擎」，负责将一段自由文本格式的 Know-How 片段转换为严格的 JSON 结构化表示。

# Objective
阅读输入的自由文本 Know-How（来自一级提炼），结合原始问答上下文，将其**精准重构**为下述 JSON Schema 格式。
- 不要脑补原文未提及的内容。
- 不要丢失原文中的任何关键规则、条件、约束或政策依据。
- 如果原文中某些字段无对应内容（如无例外情况），对应数组留空即可。

# Target JSON Schema
{KH_JSON_SCHEMA}

# 字段填写指引
- **title**: 用一句话概括该 Know-How 的核心方法论主题（10~30 字）。
- **scope**: 描述该 Know-How 适用于什么场景/业务类型。
- **steps**: 将核心方法论拆解为有序的操作步骤。
  - `step` 为字符串编号。**无分叉时**用 "1", "2", "3" 线性递增。
  - **有分叉时**，用子标签表示同层分支：如 "2a", "2b" 表示在第 2 步有两条路径，分别用 `condition` 标注走哪条路。后续步骤若只属于某条分支，也沿用同一字母后缀（如 "3a" 承接 "2a"）。
  - 每步的 `action` 精炼为一句话；`condition` 仅在该步骤有特定触发条件或属于某条分支时填写，否则为 null；`outcome` 为预期结果，无则为 null。
- **exceptions**: 原文中提到的例外/特殊场景处理方式（与分叉不同：分叉是正常的并行路径，exception 是罕见/异常情况）。
- **constraints**: 关键约束、红线、注意事项。包括但不限于政策依据（文件名+文号）。

# Input Data
<Free_Text_Know_How>
{know_how_text}
</Free_Text_Know_How>

<Original_Question>
{question}
</Original_Question>

<Original_Answer>
{answer}
</Original_Answer>

<Extra_Information>
{extra_info}
</Extra_Information>

<Reasoning_Chain>
{reasoning}
</Reasoning_Chain>

# Output Format
严格输出合法 JSON，不要包含 Markdown 围栏或任何额外文字。JSON 中的字符串值如有换行请用 \\n 替代。
输出必须完全符合上述 Target JSON Schema 的结构。"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 2: 推理验证（用结构化 Know-How 推理，对比标准答案）
# ═══════════════════════════════════════════════════════════════════════════════

def _render_know_how_readable(kh_json: str) -> str:
    """将 JSON know-how 渲染为人类可读的自然语言摘要，辅助 LLM 理解逻辑流。"""
    import json as _json
    try:
        kh = _json.loads(kh_json) if isinstance(kh_json, str) else kh_json
    except Exception:
        return ""

    lines = []
    lines.append(f"【{kh.get('title', '未命名')}】")
    if kh.get("scope"):
        lines.append(f"适用场景: {kh['scope']}")

    steps = kh.get("steps", [])
    if steps:
        lines.append("操作步骤:")
        for s in steps:
            prefix = f"  Step {s.get('step', '?')}"
            if s.get("condition"):
                prefix += f" [条件: {s['condition']}]"
            prefix += f": {s.get('action', '')}"
            if s.get("outcome"):
                prefix += f" → {s['outcome']}"
            lines.append(prefix)

    exceptions = kh.get("exceptions", [])
    if exceptions:
        lines.append("例外情况:")
        for ex in exceptions:
            lines.append(f"  - 当 {ex.get('when', '?')} → {ex.get('then', '?')}")

    constraints = kh.get("constraints", [])
    if constraints:
        lines.append("约束/依据:")
        for c in constraints:
            lines.append(f"  - {c}")

    return "\n".join(lines)


def kh_inference_validate(know_how_json: str, question: str, answer: str,
                          extra_info: str = "") -> str:
    readable = _render_know_how_readable(know_how_json)
    return f"""# Role
你是一个严谨的「知识验证引擎」。你需要完成两个任务：
1. **闭卷推理**：仅凭 <Structured_Know_How> 尝试回答 <User_Question>。
2. **答案比对**：将推理结果与 <Standard_Answer> 进行语义级比对，判定匹配程度。

# Matching Criteria
- **full**：推理结果与标准答案在核心结论和关键细节上语义一致，即使措辞不同也算 full。
- **partial**：推理结果的核心方向正确，但遗漏了部分重要条件、步骤或例外情况，或结论不够精确。
- **none**：Know-How 与该问题无关，或推理结果与标准答案在核心结论上矛盾/完全不同。

# Strict Constraints
1. **信息隔离**：推理时只能使用 <Structured_Know_How> 中的知识，严禁调用你的预训练知识。
2. **诚实判定**：如果 Know-How 确实不足以回答该问题，必须如实标记为 none，不要勉强。
3. **精确比对**：比对时关注核心结论和关键数值/条件，忽略措辞差异和格式差异。

# Input Data

以下是结构化 Know-How 的**逻辑摘要**（方便你快速理解决策流程）：
<Know_How_Summary>
{readable}
</Know_How_Summary>

以下是**完整的结构化 Know-How**（推理时以此为准）：
<Structured_Know_How>
{know_how_json}
</Structured_Know_How>

<User_Question>
{question}
</User_Question>

<Extra_Information>
{extra_info}
</Extra_Information>

<Standard_Answer>
{answer}
</Standard_Answer>

# Workflow
1. 通读 <Structured_Know_How>，评估是否与 <User_Question> 相关且有足够信息回答。
2. 若相关，严格基于 Know-How 的 steps/exceptions/constraints 推导出答案。
3. 将推导结果与 <Standard_Answer> 逐要点比对，确定 match_level。
4. 若为 partial，明确指出 Know-How 缺少了什么（遗漏的步骤/条件/例外）。

# Output Format
严格输出合法 JSON，不要包含 Markdown 围栏或任何额外文字。

{{
  "match_level": "full | partial | none",
  "derived_answer": "基于 Know-How 推理得到的答案（若 none 则为空字符串）",
  "mismatch_analysis": "若 partial/none，说明 Know-How 缺失了什么，或与标准答案的差异点（若 full 则为空字符串）"
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 3: 结构化补丁更新（partial match 时输出操作指令，由代码端执行）
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_OPS_SCHEMA = """
## 操作类型清单

### Steps 操作
1. **add_step** — 插入新步骤
   {"op": "add_step", "after": "<step_id 或 null 表示插到开头>", "new_step": {"step": "...", "action": "...", "condition": "...|null", "outcome": "...|null"}}

2. **modify_step** — 修改已有步骤的部分字段（只传需要改的字段；若需重命名步骤编号，在 updates 中包含 "step" 字段）
   {"op": "modify_step", "target": "<step_id>", "updates": {"action?": "...", "condition?": "...", "outcome?": "...", "step?": "<new_label>"}}

3. **remove_step** — 删除步骤（仅当该步骤被证明错误/多余时使用，极少见）
   {"op": "remove_step", "target": "<step_id>"}

### Exceptions 操作
4. **add_exception** — 追加新的例外情况
   {"op": "add_exception", "exception": {"when": "...", "then": "..."}}

5. **modify_exception** — 修改已有例外（通过 0-based 索引定位）
   {"op": "modify_exception", "index": 0, "updates": {"when?": "...", "then?": "..."}}

6. **remove_exception** — 删除例外（仅当该例外被证明不成立时使用）
   {"op": "remove_exception", "index": 0}

### Constraints 操作
7. **add_constraint** — 追加新约束/依据
   {"op": "add_constraint", "constraint": "..."}

8. **modify_constraint** — 修改已有约束（通过 0-based 索引定位）
   {"op": "modify_constraint", "index": 0, "new_value": "..."}

9. **remove_constraint** — 删除约束
   {"op": "remove_constraint", "index": 0}

### 顶层字段操作
10. **update_scope** — 修改适用范围
    {"op": "update_scope", "new_scope": "..."}

11. **update_title** — 修改标题（最低优先级，仅在其他操作都无法覆盖时使用）
    {"op": "update_title", "new_title": "..."}
"""


def kh_minimal_update(know_how_json: str, question: str, answer: str,
                      mismatch_analysis: str, extra_info: str = "") -> str:
    readable = _render_know_how_readable(know_how_json)
    return f"""# Role
你是一个精密的「知识增量补丁引擎」。你的任务是针对一份已有的结构化 Know-How，输出一组**结构化操作指令**（patch），由程序端精确执行。

# 核心原则
- **最小改动**：只输出必要的操作，不动已经正确的部分。你不需要输出完整的 Know-How，只需要输出改动操作。
- **泛化抽象**：新增内容应向上抽象为通用规则，不要就事论事地复述样本细节。
- **操作按顺序执行**：operations 数组中的操作将按顺序依次执行。如果前一个操作重命名了某个 step（如 "2" → "2a"），后续操作应引用新名称 "2a"。
- **分叉组合**：若需从已有步骤分叉，请组合使用 modify_step（重命名原步骤并标注条件）+ add_step（插入新分支步骤）。

# 操作指令规范
{PATCH_OPS_SCHEMA}

# step 编号格式约束（严格遵守）
- step 编号**只允许**两种格式：**纯数字**（如 "1", "2", "3"）或 **数字+单个小写字母**（如 "2a", "2b", "3a"）。
- **禁止**使用：step "0"、中文后缀（如 "2a-预缴"）、短横线（如 "3a-2"）、三级嵌套（如 "3a1"）。
- 新增步骤的编号必须与插入位置的上下文保持逻辑连续。

# 操作优先级（优先使用排在前面的操作）
1. add_step / modify_step（补充步骤或完善已有步骤的条件/结果）
2. add_exception（追加例外）
3. add_constraint（追加约束/依据）
4. update_scope（调整适用范围）
5. update_title（调整标题，仅在极少数情况下使用）
6. remove 类操作（极少使用，仅当有明确证据表明现有内容错误时）

# 分叉示例
若需将 step "2" 拆分为两条路径：
```json
[
  {{"op": "modify_step", "target": "2", "updates": {{"step": "2a", "condition": "当满足条件A时"}}}},
  {{"op": "add_step", "after": "2a", "new_step": {{"step": "2b", "action": "分支B的操作", "condition": "当满足条件B时", "outcome": null}}}}
]
```

# Input Data

以下是当前 Know-How 的**逻辑摘要**（方便你快速理解现有决策流程）：
<Know_How_Summary>
{readable}
</Know_How_Summary>

以下是**完整的结构化 Know-How**（操作指令以此为基准）：
<Current_Know_How>
{know_how_json}
</Current_Know_How>

<Mismatch_Analysis>
{mismatch_analysis}
</Mismatch_Analysis>

<New_Sample_Question>
{question}
</New_Sample_Question>

<New_Sample_Answer>
{answer}
</New_Sample_Answer>

<Extra_Information>
{extra_info}
</Extra_Information>

# Output Format
严格输出合法 JSON，不要包含 Markdown 围栏或任何额外文字。

{{
  "operations": [
    // 操作指令数组，按执行顺序排列
  ],
  "diff_description": "简述本次修改内容（如：将 step 2 分叉为 2a/2b 处理不同纳税人类型；在 constraints 中追加了 XX 政策依据）"
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 4: 步骤编号归一化（cluster 精炼完成后，LLM 重整 steps 编号与顺序）
# ═══════════════════════════════════════════════════════════════════════════════

def kh_normalize_steps(know_how_json: str) -> str:
    return f"""# Role
你是一个结构化知识整理专家。你的任务是对一份 Know-How 的操作步骤进行**编号归一化和排序整理**。

# 核心原则
- **只调整步骤的 `step` 编号值和数组中的排列顺序**。
- **不修改任何步骤的 action、condition、outcome 文本内容**。
- **不增删步骤**，只做重新编号和排序。

# 编号规范（严格遵守）
1. 无分叉的线性步骤使用纯数字递增：`"1"`, `"2"`, `"3"` ...
2. 同一决策点的并行分支使用"数字+单个小写字母"：`"2a"`, `"2b"`, `"2c"` ...
3. 分支后若回归主线，后续步骤恢复纯数字递增（取分支所在数字的下一个）：如 `"2a"`, `"2b"` 之后是 `"3"`
4. **禁止使用以下格式**：
   - step `"0"`（步骤必须从 `"1"` 开始）
   - 中文后缀（如 `"2a-预缴"`）
   - 短横线连接（如 `"3a-2"`）
   - 三级及以上嵌套（如 `"3a1"`, `"3a2"`, `"3a2-1"`）
5. 如果现有步骤存在三级嵌套（如 `"3a1"`, `"3a2"`），请将它们**扁平化**：提升为独立的同级分支（如 `"4a"`, `"4b"`），或在语义合理时合并到父步骤中。

# 排序规则
步骤应按决策流程的逻辑顺序排列：
1. 主数字升序
2. 同主数字下，无字母后缀排在字母后缀前（`"2"` 排在 `"2a"` 前）
3. 字母后缀按字母表顺序（`"2a"` 排在 `"2b"` 前）

# 如何判断"分叉" vs "线性"
- 如果两个步骤的 condition 描述了同一决策点的不同条件路径（互斥或并列），它们是**分支**，应使用同一数字+不同字母（如 `"2a"`, `"2b"`）。
- 如果步骤之间有先后依赖关系（先做 A 才能做 B），它们是**线性**的，应使用递增数字（如 `"2"`, `"3"`）。

# Input
<Current_Know_How>
{know_how_json}
</Current_Know_How>

# Output Format
严格输出合法 JSON 对象（不要包含 Markdown 围栏或任何额外文字），格式如下：

{{
  "steps": [
    {{"step": "1", "action": "...", "condition": "...|null", "outcome": "...|null"}},
    {{"step": "2a", "action": "...", "condition": "当...", "outcome": "..."}},
    {{"step": "2b", "action": "...", "condition": "当...", "outcome": "..."}},
    {{"step": "3", "action": "...", "condition": null, "outcome": "..."}}
  ]
}}

注意：只输出 steps 数组包裹在 JSON 对象中，不要输出 title、scope、exceptions、constraints 等其他字段。"""
