"""
三级提炼：对二级压缩结果做累增式合并 + 自动收缩，产出最终全局知识库。
串行执行（有状态依赖），支持断点续传。
"""

import os
import sys
import json
import time

try:
    from prompts import safe_parse_json
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prompts import safe_parse_json


def _save_progress(progress_file: str, batch_idx: int, kh_text: str):
    snapshot = {"last_merged_batch": batch_idx, "current_kh": kh_text}
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def _shrink_kh(
    kh_text: str,
    target_chars: int,
    llm_func,
    shrink_prompt_func,
    label: str = "",
) -> str:
    """调用 shrink_v0 将 kh_text 压缩到 target_chars 字符以内。"""
    for attempt in range(1, 4):
        try:
            resp = llm_func(shrink_prompt_func(kh_text, target_chars=target_chars))
            parsed = safe_parse_json(resp["content"])
            shrunk = parsed.get("Shrunken_Know_How", "")
            if not shrunk.strip():
                raise ValueError("Shrunken_Know_How 为空")
            log = parsed.get("Shrink_Log", "")
            print(f"  [Shrink] {label}压缩: {len(kh_text)}->{len(shrunk)} 字符 | {log[:80]}")
            return shrunk
        except Exception as e:
            wait = 3 * (2 ** (attempt - 1))
            print(f"  [Shrink] {label}第 {attempt} 次失败: {str(e)[:100]}，{wait}s 后重试...")
            time.sleep(wait)
    print(f"  [Shrink] {label}全部失败，保留原文（长度 {len(kh_text)}）")
    return kh_text


def run_level3_merge(
    level2_file: str,
    llm_func,
    merge_prompt_func,
    shrink_prompt_func,
    progress_file: str = "./kh_merge_progress.json",
    final_output_file: str = "./kh_final.json",
    max_kh_len: int = 7000,
    target_kh_len: int = 5000,
    final_max_len: int = 10000,
    max_retries_per_batch: int = 5,
):
    """
    累增式三级合并入口。

    Parameters
    ----------
    level2_file : 二级压缩结果 JSON 路径
    llm_func : LLM 调用函数（如 qwen）
    merge_prompt_func : 合并 prompt（如 merge_v0）
    shrink_prompt_func : 收缩 prompt（如 shrink_v0）
    progress_file : 断点续传进度文件
    final_output_file : 最终输出 JSON
    max_kh_len : 超过此字符数则合并前先收缩
    target_kh_len : 收缩目标字符数
    final_max_len : 最终输出强制不超过此字符数
    """
    with open(level2_file, "r", encoding="utf-8") as f:
        compression_data = json.load(f)

    batches_l2 = sorted(
        [
            v for v in compression_data.values()
            if v.get("status") == "success" and v.get("Final_Know_How", "").strip()
        ],
        key=lambda x: x["batch_index"],
    )
    total = len(batches_l2)
    print(f"[Level-3] 二级有效批次: {total} 个")

    current_kh = ""
    start_from = 0

    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
            last_done = progress.get("last_merged_batch", -1)
            current_kh = progress.get("current_kh", "")
            start_from = last_done + 1
            print(f"  发现进度存档，已完成第 0~{last_done} 批，从第 {start_from} 批继续")
        except Exception:
            print("  进度文件损坏，从头开始")

    print(f"  长度策略：>{max_kh_len} 先收缩至 {target_kh_len}，最终 <={final_max_len}\n")

    for i in range(start_from, total):
        batch = batches_l2[i]
        increment = batch["Final_Know_How"]
        src = batch.get("source_indices", [])
        print(f"--- 第 {i + 1}/{total} 批（src: {src}，当前 KH: {len(current_kh)} 字符）---")

        if len(current_kh) > max_kh_len:
            print(f"  [!] KH 超过 {max_kh_len} 字符，触发收缩...")
            current_kh = _shrink_kh(
                current_kh, target_kh_len, llm_func, shrink_prompt_func,
                label=f"第{i + 1}批前置-",
            )

        retry_count = 0
        while True:
            if retry_count > max_retries_per_batch:
                print(f"  [Failed] 第 {i + 1} 批超过最大重试次数，跳过")
                break
            try:
                response = llm_func(merge_prompt_func(existing=current_kh, increment=increment))
                content = safe_parse_json(response["content"])

                if "Merged_Know_How" not in content:
                    raise Exception("响应缺少字段 Merged_Know_How")

                current_kh = content["Merged_Know_How"]
                merge_log = content.get("Merge_Log", "")
                _save_progress(progress_file, i, current_kh)

                suffix = f"（历经 {retry_count} 次重试）" if retry_count > 0 else ""
                print(f"  [OK] 合并成功{suffix}，KH 长度: {len(current_kh)} 字符")
                log_preview = merge_log[:120] + "..." if len(merge_log) > 120 else merge_log
                print(f"  Merge_Log: {log_preview}")
                break
            except Exception as e:
                retry_count += 1
                wait_time = min(3 * (2 ** (retry_count - 1)), 60)
                print(f"  [Error] 第 {i + 1} 批第 {retry_count} 次失败: {str(e)[:150]}")
                time.sleep(wait_time)

    print(f"\n合并完成，最终 KH 长度: {len(current_kh)} 字符")
    if len(current_kh) > final_max_len:
        print(f"[!] 超过最终上限 {final_max_len} 字符，执行最终压缩...")
        current_kh = _shrink_kh(
            current_kh, final_max_len - 500, llm_func, shrink_prompt_func, label="最终-",
        )

    final_result = {
        "status": "success",
        "total_l2_batches_merged": total,
        "final_kh_length": len(current_kh),
        "Final_Know_How": current_kh,
    }
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"[Level-3] 三级提炼完成！保存至: {final_output_file}")
    print(f"  最终 KH 字符数: {len(current_kh)}")
    print(f"\n--- 最终 Know-How 预览（前 500 字符）---")
    print(current_kh[:500], "..." if len(current_kh) > 500 else "")
    return final_output_file


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("[level3_merge] 开始独立测试")
    print("=" * 60)

    mock_level2 = {
        "0": {
            "batch_index": 0,
            "source_indices": [0, 1],
            "item_count": 2,
            "Final_Know_How": "合同签订及增值税发票认证的综合管理要点：合同须双方签字盖章，发票认证需在360天内完成。",
            "status": "success",
        },
        "1": {
            "batch_index": 1,
            "source_indices": [2, 3],
            "item_count": 2,
            "Final_Know_How": "企业所得税汇算清缴（次年5月31日前）及个税汇算清缴（次年3-6月）的期限与扣除要点。",
            "status": "success",
        },
    }

    call_count = {"n": 0}

    def _mock_llm(prompt: str) -> dict:
        call_count["n"] += 1
        return {
            "content": (
                f'{{"Merged_Know_How": "合同、发票、企业所得税及个税汇算清缴综合财税知识库（第{call_count["n"]}次合并）。", '
                f'"Merge_Log": "正常合并，新增知识已整合"}}'
            )
        }

    def _mock_merge_prompt(existing: str, increment: str) -> str:
        return f"请合并以下知识：\n已有：{existing[:50]}\n新增：{increment[:50]}"

    def _mock_shrink_prompt(kh_text: str, target_chars: int = 5000) -> str:
        return f"请将以下知识压缩至{target_chars}字以内：\n{kh_text[:50]}"

    tmp_dir = tempfile.mkdtemp()
    level2_file = os.path.join(tmp_dir, "kh_level2_test.json")
    progress_file = os.path.join(tmp_dir, "kh_merge_progress_test.json")
    final_output = os.path.join(tmp_dir, "kh_final_test.json")

    with open(level2_file, "w", encoding="utf-8") as f:
        json.dump(mock_level2, f, ensure_ascii=False, indent=2)

    result = run_level3_merge(
        level2_file=level2_file,
        llm_func=_mock_llm,
        merge_prompt_func=_mock_merge_prompt,
        shrink_prompt_func=_mock_shrink_prompt,
        progress_file=progress_file,
        final_output_file=final_output,
    )

    with open(result, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n测试结果预览:")
    print(f"  status: {data.get('status')}")
    print(f"  合并批次数: {data.get('total_l2_batches_merged')}")
    print(f"  最终 KH 字符数: {data.get('final_kh_length')}")
    print(f"  内容前100字: {str(data.get('Final_Know_How', ''))[:100]}")

    print("\n[level3_merge] 测试完成！")
