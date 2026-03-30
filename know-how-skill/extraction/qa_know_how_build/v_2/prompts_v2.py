"""
V2 Prompts：结构化 Know-How 生成 / 推理验证 / 最小改动更新。
"""

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
#  Prompt 3: 最小改动更新（partial match 时增量修补 Know-How）
# ═══════════════════════════════════════════════════════════════════════════════

def kh_minimal_update(know_how_json: str, question: str, answer: str,
                      mismatch_analysis: str, extra_info: str = "") -> str:
    readable = _render_know_how_readable(know_how_json)
    return f"""# Role
你是一个精密的「知识增量更新引擎」。你的任务是对一份已有的结构化 Know-How 进行**最小改动**，使其能够覆盖一个新的业务样本。

# Objective
根据 <Mismatch_Analysis> 指出的缺失点，对 <Current_Know_How> 进行精准补充。
- **最小改动原则**：只修改/新增必要的内容，不动已经正确的部分。
- **结构保持**：输出必须严格遵循原有的 JSON Schema 结构。
- **泛化原则**：新增内容仍应向上抽象为通用规则，不要就事论事地复述样本细节。

# Allowed Operations (按优先级)
1. 在 steps 中插入新步骤，或在现有步骤中补充 condition/outcome
2. **从已有步骤分叉**：若新样本揭示了一条不同的处理路径，在分叉点用子标签新建分支（如在 step "2" 处拆分为 "2a" 原路径 + "2b" 新路径），并为后续步骤延续分支标签
3. 在 exceptions 中追加新的例外情况
4. 在 constraints 中追加新的约束/依据
5. 微调 scope 以扩大/缩小适用范围
6. 仅在上述操作都无法覆盖时，才考虑修改 title

# Target JSON Schema
{KH_JSON_SCHEMA}

# Input Data

以下是当前 Know-How 的**逻辑摘要**（方便你快速理解现有决策流程）：
<Know_How_Summary>
{readable}
</Know_How_Summary>

以下是**完整的结构化 Know-How**（修改时以此为准）：
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
  "updated_know_how": {{ ... }},
  "diff_description": "简述本次修改内容（如：在 steps 第2步后插入了新步骤处理 XX 情况；在 exceptions 中追加了 YY 例外）"
}}

其中 updated_know_how 必须是完整的、符合 Target JSON Schema 的 JSON 对象（不是增量 diff，而是修改后的完整版本）。"""
