"""
通用 Prompt 工具
================
提供 JSON 解析修复、知识概述生成等跨模块复用的基础 prompt 工具函数。

领域专用 prompts 已按模块拆分：
  - QA 提取: extraction/qa_know_how_build/prompts_v1.py
  - QA v2 提取: extraction/qa_know_how_build/prompts_v2.py
  - 文档提取: extraction/doc_know_how_build/prompts_doc.py
  - 推理: inference/prompts_infer.py
"""

import re
import json
import json5 as _json5


def json_repair_prompt(raw_output: str) -> str:
    """构造 JSON 修复 prompt，让大模型将无法解析的输出修正为合法 JSON。"""
    return f"""# Task
你是一个 JSON 格式修复引擎。你收到了一段**原本应当是合法 JSON 但实际无法被解析**的文本。
请你将它修复为**严格合法的 JSON**，然后**只输出修复后的 JSON，不要输出任何其他内容**。

# 常见错误类型（你需要修复的）
1. **Markdown 围栏残留**：输出被 ```json ... ``` 包裹，需要剥离
2. **未转义的特殊字符**：JSON 字符串值内部出现了裸换行符、制表符、未转义的双引号或反斜杠
3. **尾部逗号**：对象或数组的最后一个元素后面多了逗号（如 {{"a": 1,}}）
4. **单引号代替双引号**：键或值使用了单引号而非双引号
5. **注释残留**：JSON 中混入了 // 或 /* */ 注释
6. **键未加引号**：对象的键没有用双引号包裹（如 {{key: "value"}}）
7. **截断或多余内容**：JSON 前后有多余的解释性文字，或 JSON 在末尾被截断（缺少闭合括号）
8. **控制字符**：字符串内包含 U+0000–U+001F 范围内的控制字符未被转义

# 修复规则
- **保持语义不变**：只修复格式问题，绝不修改、删除或新增任何业务数据内容
- **Markdown 文本中的换行**：如果某个字段的值是 Markdown 格式文本，确保其中的换行符被替换为 `\\n`，双引号被替换为 `\\"`
- **截断修复**：如果 JSON 明显在末尾被截断（缺少闭合的 `}}` 或 `]`），请补全闭合符号
- **纯净输出**：你的回复中只能包含修复后的 JSON 本身，不要包含任何解释、前言或 Markdown 标记

# 待修复文本
<Raw_Output>
{raw_output}
</Raw_Output>

# Output
请直接输出修复后的合法 JSON（不要用 ```json 包裹）："""


def safe_parse_json_with_llm_repair(
    text: str,
    llm_func=None,
    max_repair_attempts: int = 5,
) -> dict:
    """
    带 LLM 兜底的 JSON 解析。

    先尝试 safe_parse_json（纯规则修复），失败后调用 llm_func + json_repair_prompt
    让大模型修正格式，再重新解析。可用于所有要求大模型输出 JSON 的场景。

    Parameters
    ----------
    text : 大模型原始输出文本
    llm_func : LLM 调用函数（签名：str -> dict，返回值含 "content" 键）。
               为 None 时退化为 safe_parse_json。
    max_repair_attempts : LLM 修复最大尝试次数
    """
    try:
        return safe_parse_json(text)
    except Exception as first_err:
        if llm_func is None:
            raise first_err

    last_err = first_err
    raw = text
    for attempt in range(1, max_repair_attempts + 1):
        repaired_text = None
        try:
            repair_response = llm_func(json_repair_prompt(raw))
            repaired_text = repair_response.get("content", "")
            return safe_parse_json(repaired_text)
        except Exception as e:
            last_err = e
            if repaired_text is not None:
                raw = repaired_text
            print(
                f"[JSON Repair] 第 {attempt}/{max_repair_attempts} 次 LLM 修复仍失败: "
                f"{str(e)[:120]}"
            )

    raise Exception(
        f"safe_parse_json_with_llm_repair：规则修复与 {max_repair_attempts} 次 LLM 修复均失败。"
        f"最后错误: {last_err}"
    )


def safe_parse_json(text: str) -> dict:
    """
    多策略 JSON 解析，专为大模型输出设计。

    策略顺序：
    1. json5 直接解析
    2. 剥离 Markdown 代码围栏（```json … ```）后解析
    3. 用状态机修复字符串内未转义的换行/制表符后解析
    4. 正则提取最外层 {...} 块，再执行策略 3
    """

    def _fix_unescaped_in_strings(s: str) -> str:
        """状态机遍历，将 JSON 字符串值内的裸换行、制表符替换为合法转义序列。"""
        result = []
        in_string = False
        escape_next = False
        for ch in s:
            if escape_next:
                result.append(ch)
                escape_next = False
            elif ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
            elif in_string and ch == '\n':
                result.append('\\n')
            elif in_string and ch == '\r':
                result.append('\\r')
            elif in_string and ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        return ''.join(result)

    # 策略 1：直接解析
    try:
        return _json5.loads(text)
    except Exception:
        pass

    # 策略 2：剥离 Markdown 围栏
    stripped = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r'\s*```\s*$', '', stripped.strip(), flags=re.MULTILINE)
    stripped = stripped.strip()
    try:
        return _json5.loads(stripped)
    except Exception:
        pass

    # 策略 3：修复裸换行后解析
    try:
        return _json5.loads(_fix_unescaped_in_strings(stripped))
    except Exception:
        pass

    # 策略 4：正则提取 {...} 最外层块，再修复裸换行
    m = re.search(r'\{[\s\S]*\}', stripped)
    if m:
        try:
            return _json5.loads(_fix_unescaped_in_strings(m.group(0)))
        except Exception:
            pass

    raise (
        f"safe_parse_json：所有解析策略均失败。原始内容前 300 字符：\n{text[:300]}"
    )


def knowledge_description_prompt(source_text_head: str) -> str:
    """
    生成 knowledge.md 的 prompt：基于源文档头部内容，生成知识概述。
    输出供智能体判断该知识库是否适用于当前问题。
    """
    return f"""# Task
你是一个专业的知识库索引引擎。请阅读下面的文档内容（截取自源文档头部），为该知识库生成一份**简洁、精准的知识概述**。

# 目标
生成的概述将作为该知识库的"目录卡片"，帮助智能体快速判断：**当收到一个用户问题时，该知识库是否包含可用于回答该问题的相关知识。**

# 输出要求
1. **核心主题**（1-2 句话）：该知识库主要覆盖什么领域/行业/业务场景？
2. **关键知识点**（3-10 个要点）：列出该知识库覆盖的主要知识模块、核心概念或业务主题，每个要点一句话。
3. **适用问题类型**（2-5 条）：列出该知识库适合回答哪些类型的问题。
4. **不适用场景**（1-3 条）：列出该知识库明确不涉及的领域或问题类型。

# 格式要求
直接输出 Markdown 格式，不要用 JSON 包裹，不要包含任何代码围栏标记。

# 源文档内容（头部节选）
<Source_Document_Head>
{source_text_head[:20000]}
</Source_Document_Head>
"""
