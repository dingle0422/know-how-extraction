"""
Know-How 结构化补丁引擎
======================
接收 LLM 输出的 operations 数组，按顺序对 know-how JSON 执行原子操作。
无效操作不中断流程，跳过并记录警告日志。
"""

import copy
import re

_STEP_ID_RE = re.compile(r'^(\d+)([a-zA-Z]?)')


def apply_patch(know_how: dict, operations: list[dict]) -> tuple[dict, list[dict]]:
    """
    对 know-how 执行一组结构化补丁操作。

    Parameters
    ----------
    know_how : 当前的结构化 know-how dict（会被深拷贝，不修改原对象）
    operations : LLM 输出的操作指令列表

    Returns
    -------
    (updated_know_how, patch_log)
    patch_log 中每条记录: {"op": ..., "status": "applied"|"skipped", "detail": ...}
    """
    kh = copy.deepcopy(know_how)
    patch_log: list[dict] = []

    for i, op_raw in enumerate(operations):
        op_type = op_raw.get("op", "")
        handler = _OP_DISPATCH.get(op_type)
        if handler is None:
            patch_log.append({
                "seq": i, "op": op_type,
                "status": "skipped", "detail": f"未知操作类型: {op_type}",
            })
            continue
        try:
            detail = handler(kh, op_raw)
            patch_log.append({
                "seq": i, "op": op_type, "status": "applied", "detail": detail,
            })
        except _PatchSkip as e:
            patch_log.append({
                "seq": i, "op": op_type, "status": "skipped", "detail": str(e),
            })

    if "steps" in kh and kh["steps"]:
        _sort_steps(kh["steps"])

    return kh, patch_log


class _PatchSkip(Exception):
    """操作校验不通过时抛出，用于跳过该操作。"""


# ─── Steps 排序 ───────────────────────────────────────────────────────────────

def _parse_step_sort_key(step_id: str) -> tuple:
    """将 step 编号解析为可排序的 tuple，如 "2a" → (2, 'a', 0)。"""
    s = str(step_id).strip()
    m = _STEP_ID_RE.match(s)
    if not m:
        return (9999, '', 0)
    major = int(m.group(1))
    branch = m.group(2).lower() if m.group(2) else ''
    rest = s[m.end():]
    sub_m = re.match(r'[-]?(\d+)', rest)
    sub = int(sub_m.group(1)) if sub_m else 0
    return (major, branch, sub)


def _sort_steps(steps: list[dict]) -> None:
    """按 step 编号的逻辑顺序原地排序。"""
    steps.sort(key=lambda s: _parse_step_sort_key(s.get("step", "")))


# ─── Steps 操作 ───────────────────────────────────────────────────────────────

def _find_step_idx(steps: list[dict], step_id: str) -> int:
    for idx, s in enumerate(steps):
        if str(s.get("step", "")) == step_id:
            return idx
    return -1


def _op_add_step(kh: dict, op: dict) -> str:
    new_step = op.get("new_step")
    if not new_step or not isinstance(new_step, dict):
        raise _PatchSkip("new_step 缺失或格式错误")

    steps = kh.setdefault("steps", [])
    after = op.get("after")

    if after is None or after == "":
        steps.insert(0, new_step)
        return f"在开头插入 step {new_step.get('step', '?')}"

    anchor_idx = _find_step_idx(steps, str(after))
    if anchor_idx < 0:
        raise _PatchSkip(f"after 引用的 step '{after}' 不存在")

    steps.insert(anchor_idx + 1, new_step)
    return f"在 step {after} 之后插入 step {new_step.get('step', '?')}"


def _op_modify_step(kh: dict, op: dict) -> str:
    target = op.get("target")
    updates = op.get("updates")
    if not target or not updates or not isinstance(updates, dict):
        raise _PatchSkip("target 或 updates 缺失/格式错误")

    steps = kh.get("steps", [])
    idx = _find_step_idx(steps, str(target))
    if idx < 0:
        raise _PatchSkip(f"target step '{target}' 不存在")

    changed_fields = []
    for field, value in updates.items():
        if field in ("step", "action", "condition", "outcome"):
            steps[idx][field] = value
            changed_fields.append(field)

    if not changed_fields:
        raise _PatchSkip("updates 中无合法字段")
    return f"修改 step {target}: {', '.join(changed_fields)}"


def _op_remove_step(kh: dict, op: dict) -> str:
    target = op.get("target")
    if not target:
        raise _PatchSkip("target 缺失")

    steps = kh.get("steps", [])
    idx = _find_step_idx(steps, str(target))
    if idx < 0:
        raise _PatchSkip(f"target step '{target}' 不存在")

    steps.pop(idx)
    return f"删除 step {target}"


# ─── Exceptions 操作 ──────────────────────────────────────────────────────────

def _check_list_index(lst: list, index, name: str):
    if not isinstance(index, int):
        raise _PatchSkip(f"{name} index 非整数: {index}")
    if index < 0 or index >= len(lst):
        raise _PatchSkip(f"{name} index {index} 越界 (当前长度 {len(lst)})")


def _op_add_exception(kh: dict, op: dict) -> str:
    exc = op.get("exception")
    if not exc or not isinstance(exc, dict):
        raise _PatchSkip("exception 缺失或格式错误")
    kh.setdefault("exceptions", []).append(exc)
    return f"追加 exception: when={exc.get('when', '?')}"


def _op_modify_exception(kh: dict, op: dict) -> str:
    exceptions = kh.get("exceptions", [])
    index = op.get("index")
    updates = op.get("updates")
    _check_list_index(exceptions, index, "exceptions")
    if not updates or not isinstance(updates, dict):
        raise _PatchSkip("updates 缺失或格式错误")

    changed = []
    for field in ("when", "then"):
        if field in updates:
            exceptions[index][field] = updates[field]
            changed.append(field)
    if not changed:
        raise _PatchSkip("updates 中无合法字段")
    return f"修改 exception[{index}]: {', '.join(changed)}"


def _op_remove_exception(kh: dict, op: dict) -> str:
    exceptions = kh.get("exceptions", [])
    index = op.get("index")
    _check_list_index(exceptions, index, "exceptions")
    removed = exceptions.pop(index)
    return f"删除 exception[{index}]: when={removed.get('when', '?')}"


# ─── Constraints 操作 ─────────────────────────────────────────────────────────

def _op_add_constraint(kh: dict, op: dict) -> str:
    constraint = op.get("constraint")
    if not constraint or not isinstance(constraint, str):
        raise _PatchSkip("constraint 缺失或非字符串")
    kh.setdefault("constraints", []).append(constraint)
    return f"追加 constraint: {constraint[:60]}"


def _op_modify_constraint(kh: dict, op: dict) -> str:
    constraints = kh.get("constraints", [])
    index = op.get("index")
    new_value = op.get("new_value")
    _check_list_index(constraints, index, "constraints")
    if not new_value or not isinstance(new_value, str):
        raise _PatchSkip("new_value 缺失或非字符串")
    constraints[index] = new_value
    return f"修改 constraint[{index}]"


def _op_remove_constraint(kh: dict, op: dict) -> str:
    constraints = kh.get("constraints", [])
    index = op.get("index")
    _check_list_index(constraints, index, "constraints")
    removed = constraints.pop(index)
    return f"删除 constraint[{index}]: {removed[:60]}"


# ─── 顶层字段操作 ─────────────────────────────────────────────────────────────

def _op_update_scope(kh: dict, op: dict) -> str:
    new_scope = op.get("new_scope")
    if not new_scope or not isinstance(new_scope, str):
        raise _PatchSkip("new_scope 缺失或非字符串")
    old = kh.get("scope", "")
    kh["scope"] = new_scope
    return f"scope 已更新 (旧: {old[:40]}…)"


def _op_update_title(kh: dict, op: dict) -> str:
    new_title = op.get("new_title")
    if not new_title or not isinstance(new_title, str):
        raise _PatchSkip("new_title 缺失或非字符串")
    old = kh.get("title", "")
    kh["title"] = new_title
    return f"title 已更新: {old} → {new_title}"


# ─── 操作分发表 ───────────────────────────────────────────────────────────────

_OP_DISPATCH = {
    "add_step":           _op_add_step,
    "modify_step":        _op_modify_step,
    "remove_step":        _op_remove_step,
    "add_exception":      _op_add_exception,
    "modify_exception":   _op_modify_exception,
    "remove_exception":   _op_remove_exception,
    "add_constraint":     _op_add_constraint,
    "modify_constraint":  _op_modify_constraint,
    "remove_constraint":  _op_remove_constraint,
    "update_scope":       _op_update_scope,
    "update_title":       _op_update_title,
}
