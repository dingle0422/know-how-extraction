"""
二级提炼：将一级提炼的碎片 Know-How 按批次压缩整合。
支持多线程并发 + 断点续传。
"""

import math
import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from prompts import safe_parse_json_with_llm_repair
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from prompts import safe_parse_json_with_llm_repair

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from k_means_constrained import KMeansConstrained
    _CLUSTERING_AVAILABLE = True
except ImportError:
    _CLUSTERING_AVAILABLE = False

compress_lock = Lock()


def _update_json_file(file_path: str, key: str, value: dict):
    data_dict = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
        except (json.JSONDecodeError, IOError):
            data_dict = {}
    data_dict[key] = value
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)


def load_level1_results(level1_file: str) -> list[dict]:
    """加载一级提炼结果，只保留成功且非空的条目，按 index 排序。"""
    with open(level1_file, "r", encoding="utf-8") as f:
        kh_data = json.load(f)
    valid = sorted(
        [
            {"index": v["index"], "Know_How": v["Know_How"]}
            for v in kh_data.values()
            if v.get("status") == "success" and v.get("Know_How", "").strip()
        ],
        key=lambda x: x["index"],
    )
    print(f"[Level-2] 有效 Know_How 总数: {len(valid)}")
    return valid


def make_batches(items: list[dict], batch_size: int = 10) -> list[list[dict]]:
    """
    将 items 按语义相似度分批：使用 TF-IDF 向量化 + 均衡 K-Means 聚类，
    保证每批条数 <= batch_size。
    若依赖库未安装，自动回退到顺序分块。
    """
    n = len(items)
    k = math.ceil(n / batch_size)

    if n <= batch_size:
        print(f"[Level-2] 共 1 个批次，每批最多 {batch_size} 条")
        return [items]

    if not _CLUSTERING_AVAILABLE:
        print(
            "[Level-2] 警告：未找到 sklearn 或 k-means-constrained，"
            "回退到顺序分块。可运行 `pip install scikit-learn k-means-constrained` 启用语义聚类。"
        )
        batches = [items[i: i + batch_size] for i in range(0, n, batch_size)]
        print(f"[Level-2] 共 {len(batches)} 个批次（顺序分块），每批最多 {batch_size} 条")
        return batches

    texts = [item["Know_How"] for item in items]
    vectorizer = TfidfVectorizer(max_features=512)
    X = vectorizer.fit_transform(texts)

    clf = KMeansConstrained(
        n_clusters=k,
        size_max=batch_size,
        random_state=42,
        n_init=10,
    )
    labels = clf.fit_predict(X)

    cluster_map: dict[int, list[dict]] = {}
    for item, label in zip(items, labels):
        cluster_map.setdefault(int(label), []).append(item)

    batches = list(cluster_map.values())
    sizes = [len(b) for b in batches]
    print(
        f"[Level-2] 共 {len(batches)} 个批次（TF-IDF 语义聚类），"
        f"每批最多 {batch_size} 条，实际批次大小：min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}"
    )
    return batches


def _format_batch_snippets(batch: list[dict]) -> str:
    parts = []
    for i, item in enumerate(batch, 1):
        parts.append(f"【片段 {i}】（原始 index: {item['index']}）\n{item['Know_How']}")
    return "\n\n---\n\n".join(parts)


def _process_compression_batch(
    batch_idx: int,
    batch: list[dict],
    total_batches: int,
    llm_func,
    prompt_func,
    output_file: str,
    max_retries: int = 999,
):
    retry_count = 0
    last_error_msg = ""

    while True:
        if retry_count > max_retries:
            error_result = {
                "batch_index": batch_idx,
                "source_indices": [item["index"] for item in batch],
                "item_count": len(batch),
                "status": "failed",
                "error": last_error_msg,
                "retry_count": retry_count,
            }
            with compress_lock:
                _update_json_file(output_file, str(batch_idx), error_result)
            print(f"[Failed] 批次 {batch_idx} 超过最大重试次数 ({max_retries})")
            return batch_idx, "failed", None

        try:
            snippets_text = _format_batch_snippets(batch)
            response = llm_func(prompt_func(f=snippets_text))

            try:
                content = safe_parse_json_with_llm_repair(
                    response["content"], llm_func=llm_func
                )
            except Exception as json_err:
                raise Exception(f"JSON 解析失败（含LLM修复）: {json_err}")

            if "Final_Know_How" not in content:
                raise Exception("响应缺少必需字段 'Final_Know_How'")

            result = {
                "batch_index": batch_idx,
                "source_indices": [item["index"] for item in batch],
                "item_count": len(batch),
                "Synthesis_Summary": content.get("Synthesis_Summary", ""),
                "Final_Know_How": content["Final_Know_How"],
                "status": "success",
                "retry_count": retry_count,
            }
            with compress_lock:
                _update_json_file(output_file, str(batch_idx), result)

            suffix = f"（历经 {retry_count} 次重试）" if retry_count > 0 else ""
            print(f"[Success] 批次 {batch_idx + 1}/{total_batches} 完成"
                  f"（{len(batch)} 条知识）{suffix}")
            return batch_idx, "success", result

        except Exception as e:
            retry_count += 1
            last_error_msg = str(e)
            print(f"[Error] 批次 {batch_idx} 第 {retry_count} 次失败: "
                  f"{last_error_msg[:150]}")
            time.sleep(3)


def run_level2_compression(
    level1_file: str,
    llm_func,
    prompt_func,
    output_file: str = "./kh_compression_level2.json",
    batch_size: int = 10,
    max_workers: int = 16,
):
    """
    多线程二级知识压缩入口。

    Parameters
    ----------
    level1_file : 一级提炼结果 JSON 路径
    llm_func : LLM 调用函数
    prompt_func : 二级压缩 prompt 构造函数（如 compression_v2）
    output_file : 输出 JSON 路径
    batch_size : 每批条数
    max_workers : 并发线程数
    """
    valid_items = load_level1_results(level1_file)
    batches = make_batches(valid_items, batch_size)

    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"  发现已有进度文件，包含 {len(existing_data)} 个批次记录，自动续传")
        except Exception:
            pass

    pending = [
        (i, b)
        for i, b in enumerate(batches)
        if str(i) not in existing_data
        or existing_data.get(str(i), {}).get("status") != "success"
    ]
    completed = len(batches) - len(pending)
    print(f"  总批次: {len(batches)}，已完成: {completed}，"
          f"待处理: {len(pending)}，并发数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_compression_batch,
                i, b, len(batches), llm_func, prompt_func, output_file,
            ): i
            for i, b in pending
        }
        for future in as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            try:
                _, status, _ = future.result()
                if status == "success":
                    completed += 1
                    pct = completed / len(batches) * 100
                    print(f"  进度: {completed}/{len(batches)} ({pct:.1f}%)")
            except Exception as e:
                print(f"  批次 {batch_idx} 处理异常: {e}")

    print(f"[Level-2] 全部完成！结果保存于: {output_file}")
    return output_file


if __name__ == "__main__":
    from llm_client import chat
    from prompts import compression_v2

    print("=" * 60)
    print("[level2_compress] 开始独立测试（真实 LLM 调用）")
    print("=" * 60)

    # 直接读取 level1 的真实输出文件（json.load 会自动将 \n 还原为真正换行符）
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    level1_file = os.path.join(output_dir, "kh_level1_test.json")
    if not os.path.exists(level1_file):
        raise FileNotFoundError(
            f"未找到 level1 输出文件：{level1_file}\n"
            "请先运行 level1_extract.py 生成该文件。"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "kh_level2_test.json")
    if os.path.exists(output_file):
        os.remove(output_file)

    result = run_level2_compression(
        level1_file=level1_file,
        llm_func=chat,
        prompt_func=compression_v2,
        output_file=output_file,
        batch_size=10,
        max_workers=2,
    )

    with open(result, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n测试结果预览（共 {len(data)} 个批次）:")
    for k, v in sorted(data.items(), key=lambda x: int(x[0])):
        status = v.get("status")
        kh_preview = str(v.get("Final_Know_How", ""))[:60]
        print(f"  [批次 {k}] status={status} | Final_Know_How: {kh_preview}...")

    # 导出 Markdown 预览文件
    md_file = output_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for k, v in sorted(data.items(), key=lambda x: int(x[0])):
            kh = v.get("Final_Know_How", "").strip()
            if kh:
                f.write(kh)
                f.write("\n\n---\n\n")
    print(f"Markdown 预览文件已导出：{md_file}")

    print("\n[level2_compress] 测试完成！")
