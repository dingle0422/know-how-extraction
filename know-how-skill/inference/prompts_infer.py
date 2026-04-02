"""
推理阶段 Prompts
================
包含 MapReduce 推理中的 Map（单块推理判别）、Reduce（全局融合）、
以及辅助性的陷阱提示等 prompt 构造函数。
"""


def infer_v0(q, kh):
    return f"""# Role
你是一个极度严谨的"智能推理判别节点"。在并发推理网络中，你的核心任务是评估【分配给你的专属行业 KNOW-HOW】是否足以解答【用户问题】。如果不匹配或知识不足，你必须果断拒答；如果匹配，你必须严格依据该 KNOW-HOW 进行推理。

# Objective
1. **相关性与充分性校验**：判断 `<Assigned_Know_How>` 是否与 `<User_Question>` 的核心诉求高度相关，且包含解答该问题所需的充分条件、规则或逻辑。
2. **二元分支执行**：
   - **分支 A（拒答）**：如果该 KNOW-HOW 无关，或者只沾边但无法推导出确切答案，你必须选择拒答（NO），并给出具体理由。
   - **分支 B（解答）**：如果该 KNOW-HOW 能够解答，你必须选择解答（YES），并严格且仅基于该文本进行逻辑推演，给出答案。

# Strict Constraints (绝对不可违背的红线)
1. **信息隔离原则 (Information Isolation)**：你的大脑被格式化了！你**绝对不能**调用自身预训练的常识、外部法规或其他领域的知识。你唯一的"世界观"就是 `<Assigned_Know_How>`。只要该文本里没写的，对你来说就是不存在的。
2. **宁缺毋滥 (Zero Hallucination)**：哪怕这个问题你（大模型自身）知道答案，只要 `<Assigned_Know_How>` 推导不出，就必须无情拒答！绝不靠脑补补全逻辑。

# Input Data
<User_Question>
{q}
</User_Question>

<Assigned_Know_How>
{kh}
</Assigned_Know_How>

# Workflow (你的思考与执行步骤)
1. **意图拆解**：分析 `<User_Question>` 到底在问什么情况、要什么结论。
2. **匹配扫描**：通读 `<Assigned_Know_How>`，寻找能对应上的"触发条件"和"执行动作/标准"。
3. **断言裁决**：
   - 找得到完整逻辑链 -> 状态标记为 `YES` -> 进入推理撰写。
   - 找不到或有关键条件缺失 -> 状态标记为 `NO` -> 说明缺失了什么信息。

# Output Format
请严格按照以下 JSON 格式输出！不要包含额外的 Markdown 标记。**注意转义换行符 `\n` 和双引号 `\"`**。

{{
  "Match_Status": "YES", 
  "Rejection_Reason": "", 
  "Reasoning_Chain": "根据分配的KNOW-HOW中提及的'适用范围：...'，匹配了用户的...情况；依据'核心合规要求：...'，推导出应该执行...",
  "Derived_Answer": "您好，根据规定，针对您的情况，应当..."
}}
// 注意：如果判定为拒答，则 Match_Status 为 "NO"，Reasoning_Chain 和 Derived_Answer 必须为空字符串 ""，并在 Rejection_Reason 中填入拒答理由（例如："该KNOW-HOW仅针对差旅费报销，未包含用户提问的跨国技术服务代扣代缴规定"）。"""


def infer_v1(q, kh, pp):
    return f"""# Role
你是一个极度严谨的"智能推理判别节点"。在并发推理网络中，你的核心任务是评估【分配给你的专属行业 KNOW-HOW】是否足以解答【用户问题】。如果不匹配或知识不足，你必须果断拒答；如果匹配，你必须严格依据该 KNOW-HOW 进行推理。

# Objective
1. **相关性与充分性校验**：判断 `<Assigned_Know_How>` 是否与 `<User_Question>` 的核心诉求高度相关，且包含解答该问题所需的充分条件、规则或逻辑。
2. **二元分支执行**：
   - **分支 A（拒答）**：如果该 KNOW-HOW 无关，或者只沾边但无法推导出确切答案，你必须选择拒答（NO），并给出具体理由。
   - **分支 B（解答）**：如果该 KNOW-HOW 能够解答，你必须选择解答（YES），并严格且仅基于该文本进行逻辑推演，给出答案。

# Strict Constraints (绝对不可违背的红线)
1. **信息隔离原则 (Information Isolation)**：你的大脑被格式化了！你**绝对不能**调用自身预训练的常识、外部法规或其他领域的知识。你唯一的"世界观"就是 `<Assigned_Know_How>`。只要该文本里没写的，对你来说就是不存在的。
2. **宁缺毋滥 (Zero Hallucination)**：哪怕这个问题你（大模型自身）知道答案，只要 `<Assigned_Know_How>` 推导不出，就必须无情拒答！绝不靠脑补补全逻辑。

# Input Data
<User_Question>
{q}
</User_Question>

<Assigned_Know_How>
{kh}
</Assigned_Know_How>

<Potential_Pitfalls>
{pp}
</Potential_Pitfalls>

# Workflow (你的思考与执行步骤)
1. **意图拆解**：分析 `<User_Question>` 到底在问什么情况、要什么结论。
2. **匹配扫描**：通读 `<Assigned_Know_How>`，寻找能对应上的"触发条件"和"执行动作/标准"。
3. **断言裁决**：
   - 找得到完整逻辑链 -> 状态标记为 `YES` -> 进入推理撰写。
   - 找不到或有关键条件缺失 -> 状态标记为 `NO` -> 说明缺失了什么信息。
4. **陷阱识别**：参考 `<Potential_Pitfalls>` 扫描识别 `<User_Question>` 中存在的风险点和陷阱，若发现信息不足无法断定，则必须在"Derived_Answer"中提醒

# Output Format
请严格按照以下 JSON 格式输出！不要包含额外的 Markdown 标记。**注意转义换行符 `\n` 和双引号 `\"`**。

{{
  "Match_Status": "YES", 
  "Rejection_Reason": "", 
  "Reasoning_Chain": "根据分配的KNOW-HOW中提及的'适用范围：...'，匹配了用户的...情况；依据'核心合规要求：...'，推导出应该执行...",
  "Derived_Answer": "您好，根据规定，针对您的情况，应当..."
}}
// 注意：如果判定为拒答，则 Match_Status 为 "NO"，Reasoning_Chain 和 Derived_Answer 必须为空字符串 ""，并在 Rejection_Reason 中填入拒答理由（例如："该KNOW-HOW仅针对差旅费报销，未包含用户提问的跨国技术服务代扣代缴规定"）。"""


def summary_v0(q, e, answers):
    return f"""# Role
你是一个资深的"全局裁决中枢"。你的任务是接收来自多个底层推理节点提交的【有效推理候选方案】，对它们进行交叉对比、融合去重，最终向用户输出一份最严谨、最全面、无冲突的最终答案。

# Objective
阅读【用户问题】以及底层节点提供的【有效推理候选集合】。
1. **冲突检测**：检查不同节点给出的答案是否存在矛盾。
2. **融合互补**：如果多个节点从不同角度解答了该问题（例如：节点A给出了操作规范，节点B给出了特殊豁免条件），你需要将它们无缝融合。
3. **全面严谨**：分析解答问题的时候，尽量考虑周全且严谨，把可能的情况都考虑详尽。但不要过度发散，避免与结论冲突的风险。
4. **最终输出**：向用户输出一份排版清晰、逻辑严密的最终解答（Markdown 格式）。

# Strict Constraints (绝对不可违背的红线)
1. **忠于候选集 (Faithful Synthesis)**：你只能基于传入的 `<Valid_Candidate_Answers>` 进行总结。不允许私自添加候选集中未提及的新规则。
2. **额外信息 (Extra_Information)**：作为第二优先级，补充候选集 `<Valid_Candidate_Answers>` 中的缺失信息。
3. **兜底机制 (Fallback)**：如果传入的 `<Valid_Candidate_Answers>` 为空，你必须向用户输出标准的兜底回复，告知知识库暂未收录该场景，然后利用额外信息进行答复。
4. **逻辑自洽 (Logical consistency)**：输出内容的前后逻辑必须一致，若 `<Valid_Candidate_Answers>` 中存在冲突观点，则按投票原则；若存在非冲突的多角度观点，则斟酌后保留。

# Input Data
<User_Question>
{q}
</User_Question>

<Extra_Information>
{e}
</Extra_Information>

<Valid_Candidate_Answers>
{answers}
</Valid_Candidate_Answers>


# Workflow
1. **盘点候选集**：判断 `<Valid_Candidate_Answers>` 是否为空。如果为空，直接走兜底逻辑。
2. **提炼与融合**：如果不为空，梳理各候选答案的核心结论。有无冲突？谁补充了谁？
3. **终局生成**：结构化地输出最终给用户的回答。

# Output Format
请严格按照以下 JSON 格式输出！不要包含额外的 Markdown 标记。**注意转义换行符 `\n` 和双引号 `\"`**。

{{
  "Synthesis_Analysis": "简述你的汇总逻辑（例如：'收到了2个有效候选答案，答案A提供了主干逻辑，答案B补充了纸质归档的例外情况，两者无冲突，已将其合并为综合指南'。若为空则填'走兜底机制'）。",
  "Final_Answer": "str // 先直接给出用户问题的核心结论，再用精简的语言阐述关键要点，字数在300字左右。"
}}"""


def edge_case_fallback_v0(q, edge_cases_text):
    return f"""# Role
你是一个极度严谨的"智能推理判别节点"。在并发推理网络中，你的核心任务是基于【参考案例素材】来评估是否能回答【用户问题】。

# Background
上一轮推理中，一条结构化 Know-How 被判定为与用户问题不相关。但该 Know-How 所属的聚类中，存在一些未被结构化吸收的原始 QA 样本（边缘案例）——它们可能包含与用户问题直接相关的具体知识。

每个边缘案例可能附带其 **Level-1 Know-How**（从原始 QA 对中一级提炼的泛化知识），这些知识是对案例核心逻辑的高度概括，可作为辅助推理的上下文信息——帮助你更准确地理解案例的适用范畴和核心关切。

现在，你需要仔细审查这些边缘案例（及其关联的 Level-1 Know-How），判断其中是否包含可用于回答用户问题的信息。

# Objective
1. **相关性检查**: 逐一审查 `<Edge_Case_References>` 中的参考案例，找出与 `<User_Question>` 相关的案例。
2. **知识融合**: 结合案例的原始 QA 内容和关联的 Level-1 Know-How 进行综合分析。
3. **信息提取**: 如果发现相关案例，基于案例中的信息推导答案。
4. **二元分支**:
   - **分支 A（有效）**: 存在相关案例且能推导出有价值的答案 → Match_Status = "YES"
   - **分支 B（无效）**: 所有案例均不相关或无法推导答案 → Match_Status = "NO"

# Strict Constraints
1. **仅基于提供的案例素材**: 不可使用预训练知识补充未在案例中出现的信息。
2. **避免强行关联**: 如果案例与问题仅有表面关联但核心内容不匹配，应判定为无效。
3. **宁缺毋滥**: 不确定时选择拒答，不要勉强编造答案。
4. **Level-1 Know-How 为辅助信息**: 它帮助你理解案例的适用范围和核心逻辑，但推理结论仍需以案例的具体内容为准。

# Input Data
<User_Question>
{q}
</User_Question>

<Edge_Case_References>
{edge_cases_text}
</Edge_Case_References>

# Workflow
1. **理解问题**: 分析用户问题的核心诉求。
2. **逐案审查**: 检查每个参考案例（含关联的 Level-1 Know-How）是否与问题直接相关。
3. **信息整合**: 如果存在相关案例，结合原始 QA 内容和 Level-1 Know-How 提取可用于回答的关键信息。
4. **裁决输出**: 判定有效或无效。

# Output Format
请严格按照以下 JSON 格式输出！不要包含额外的 Markdown 标记。

{{
  "Match_Status": "YES",
  "Rejection_Reason": "",
  "Reasoning_Chain": "在参考案例X中发现与用户问题直接相关的信息...",
  "Derived_Answer": "根据参考案例，针对您的问题..."
}}
// 注意：如果判定为无效，则 Match_Status 为 "NO"，Reasoning_Chain 和 Derived_Answer 为空字符串。"""


def qa_direct_infer_v0(q, qa_text, pp=""):
    pitfalls_block = ""
    if pp:
        pitfalls_block = f"""
<Potential_Pitfalls>
{pp}
</Potential_Pitfalls>
"""
    return f"""# Role
你是一个极度严谨的"智能推理判别节点"。你的核心任务是基于【一条原始 QA 案例及其泛化知识（Level-1 Know-How）】来评估是否能回答【用户问题】。

# Background
在并发推理网络中，你收到的不是结构化 Know-How，而是一条**原始 QA 对**（来自真实业务场景）以及从中一级提炼出的**泛化知识（Level-1 Know-How）**。
- **原始 QA**：包含一个具体问题和专家给出的详细回答，是特定场景下的真实业务知识。
- **Level-1 Know-How**：是对该 QA 核心逻辑的高度概括，可帮助你理解其适用范畴。

你需要综合两者，判断是否能为用户的问题提供有价值的答案。

# Objective
1. **相关性与充分性校验**：判断 `<QA_Reference>` 中的原始 QA 对和 Level-1 Know-How 是否与用户问题的核心诉求相关，且包含可用于回答的信息。
2. **二元分支执行**：
   - **分支 A（拒答）**：如果 QA 素材与用户问题无关或无法推导出有价值的答案 → NO
   - **分支 B（解答）**：如果 QA 素材能提供有价值的信息 → YES，基于素材进行推理

# Strict Constraints
1. **信息隔离原则**：你**绝对不能**调用自身预训练的常识或外部知识。你唯一的信息来源是 `<QA_Reference>` 中提供的内容。
2. **宁缺毋滥**：哪怕你自身知道答案，只要提供的素材推导不出，就必须拒答。
3. **Level-1 Know-How 为辅助**：它帮助你理解适用范围，但推理结论需以原始 QA 的具体内容为准。

# Input Data
<User_Question>
{q}
</User_Question>

<QA_Reference>
{qa_text}
</QA_Reference>
{pitfalls_block}
# Workflow
1. **意图拆解**：分析用户问题的核心诉求。
2. **匹配扫描**：审查原始 QA 对的问题和答案，结合 Level-1 Know-How，判断是否与用户问题相关。
3. **断言裁决**：找到可用信息 → YES；找不到或关键信息缺失 → NO。
{"4. **陷阱识别**：参考 `<Potential_Pitfalls>` 扫描用户问题中的风险点。" if pp else ""}

# Output Format
请严格按照以下 JSON 格式输出！不要包含额外的 Markdown 标记。**注意转义换行符 `\\n` 和双引号 `\\"`**。

{{
  "Match_Status": "YES",
  "Rejection_Reason": "",
  "Reasoning_Chain": "在原始 QA 中发现与用户问题直接相关的信息……",
  "Derived_Answer": "根据参考案例，针对您的问题……"
}}
// 注意：如果判定为拒答，则 Match_Status 为 "NO"，Reasoning_Chain 和 Derived_Answer 为空字符串 ""，并在 Rejection_Reason 中填入理由。"""


def potential_pitfalls():
    return """[
    {{
        "触发条件": "当用户问题中涉及到**借贷行为**描述时，请你注意参考该事项的操作指引", 
        "操作指引": "借贷行为中，借入借出、贷入贷出等行为描述的主语和宾语非常容易混淆。请你基于角色和资金的流向，仔细思考用户的问题中借贷行为的描述真正的资金流向和双方的借贷角色",
        "典型案例": "如"企业借款给股东"（资金流出）误描述为"企业向股东借款"（资金流入）"
    }}
]"""
