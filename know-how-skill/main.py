"""
Know-How 提炼 & 推理 全流程入口
=================================

两大阶段：
  1. 提炼阶段（Extraction）：一级 → 二级 → 三级，从 QA 样本中提炼全局知识库
  2. 推理阶段（Inference）：基于知识库对测试问题做 MapReduce 推理

用法:
  python main.py                    # 运行全流程
  python main.py --stage extract    # 仅运行提炼阶段
  python main.py --stage infer      # 仅运行推理阶段
"""

import argparse
import os
import sys

# ─── 路径与配置 ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "extraction", "qa_know_how_build"))
sys.path.insert(0, os.path.join(BASE_DIR, "inference"))

# 输入文件（按需修改）
ANSWER_FILE = os.path.join(BASE_DIR, "output_kh_zqrz317_experts.xlsx")
QUESTION_FILE = os.path.join(BASE_DIR, "zqrz_fjrjg_data.xlsx")
TEST_CSV = os.path.join(BASE_DIR, "fjrjg_test_data.csv")

# 中间产物
LEVEL1_OUTPUT = os.path.join(BASE_DIR, "kh_results_level1.json")
LEVEL2_OUTPUT = os.path.join(BASE_DIR, "kh_compression_level2.json")
LEVEL3_PROGRESS = os.path.join(BASE_DIR, "kh_merge_progress_level3.json")
LEVEL3_OUTPUT = os.path.join(BASE_DIR, "kh_final_level3.json")

# 最终产物
EXCEL_OUTPUT = os.path.join(BASE_DIR, "output_kh_final.xlsx")
INFER_OUTPUT = os.path.join(BASE_DIR, "infer_mapreduce_result.csv")

# 业务领域标签
DOMAIN_TAG = "债权融资"

# 并发数
MAX_WORKERS = 16

# ─── 推理阶段配置 ────────────────────────────────────────────────────────────

# 指定参与推理的 knowledge 目录列表（QA / Doc 可混合）
KNOWLEDGE_DIRS: list[str] = [
    # 示例:
    # os.path.join(BASE_DIR, "extraction", "qa_know_how_build", "knowledge", "xxx_knowledge"),
    # os.path.join(BASE_DIR, "extraction", "doc_know_how_build", "knowledge", "yyy_knowledge"),
]

# 检索 Top-N 参数
TFIDF_TOP_N = 5
EMBEDDING_TOP_N = 5


def run_extraction():
    """提炼阶段：一级 → 二级 → 三级"""
    from llm_client import chat, qwen
    from prompts_v1 import single_v1, compression_v2, merge_v0, shrink_v0
    from data_loader import load_and_prepare
    from extraction.level1_extract import run_level1_extraction
    from extraction.level2_compress import run_level2_compression
    from extraction.level3_merge import run_level3_merge

    print("=" * 60)
    print("  阶段一：数据加载")
    print("=" * 60)
    data, data_train, data_test = load_and_prepare(
        answer_file=ANSWER_FILE,
        question_file=QUESTION_FILE,
    )
    print(f"  全量: {len(data)} 条, 训练: {len(data_train)} 条, 测试: {len(data_test)} 条\n")

    # 保存测试集供推理阶段使用
    from data_loader import save_test_data
    save_test_data(data_test, TEST_CSV)
    print(f"  测试集已保存至: {TEST_CSV}\n")

    print("=" * 60)
    print("  阶段二：一级提炼（单样本 Know-How 抽取）")
    print("=" * 60)
    run_level1_extraction(
        data_train=data_train,
        eb=DOMAIN_TAG,
        llm_func=chat,
        prompt_func=single_v1,
        output_file=LEVEL1_OUTPUT,
        max_workers=MAX_WORKERS,
    )

    print("\n" + "=" * 60)
    print("  阶段三：二级提炼（批次压缩整合）")
    print("=" * 60)
    run_level2_compression(
        level1_file=LEVEL1_OUTPUT,
        llm_func=chat,
        prompt_func=compression_v2,
        output_file=LEVEL2_OUTPUT,
        batch_size=10,
        max_workers=MAX_WORKERS,
    )

    print("\n" + "=" * 60)
    print("  阶段四：三级提炼（累增式合并）")
    print("=" * 60)
    run_level3_merge(
        level2_file=LEVEL2_OUTPUT,
        llm_func=qwen,
        merge_prompt_func=merge_v0,
        shrink_prompt_func=shrink_v0,
        progress_file=LEVEL3_PROGRESS,
        final_output_file=LEVEL3_OUTPUT,
    )

    print("\n" + "=" * 60)
    print("  阶段五：导出 Excel")
    print("=" * 60)
    from export import export_to_excel
    export_to_excel(
        data=data,
        level1_file=LEVEL1_OUTPUT,
        level2_file=LEVEL2_OUTPUT,
        output_path=EXCEL_OUTPUT,
    )

    print("\n提炼阶段全部完成！")


def run_inference():
    """推理阶段：4 阶段 MapReduce 知识推理"""
    from llm_client import chat, qwen
    from prompts_infer import infer_v1, summary_v0, potential_pitfalls, edge_case_fallback_v0
    from inference.mapreduce_infer import run_mapreduce_inference_file

    print("=" * 60)
    print("  推理阶段：MapReduce 知识推理（v2 四阶段）")
    print("=" * 60)

    if not os.path.exists(TEST_CSV):
        print(f"[!] 未找到测试集文件 {TEST_CSV}，请先运行提炼阶段或手动准备。")
        return

    # 指定参与推理的 knowledge 目录列表（按需修改）
    knowledge_dirs = KNOWLEDGE_DIRS
    if not knowledge_dirs:
        print("[!] 未配置 knowledge 目录列表，请在 KNOWLEDGE_DIRS 中指定。")
        return

    missing = [d for d in knowledge_dirs if not os.path.isdir(d)]
    if missing:
        print(f"[!] 以下 knowledge 目录不存在: {missing}")
        return

    embedding_func = None
    try:
        from utils import get_embeddings
        embedding_func = get_embeddings
        print("[Infer] Dense Embedding 服务已加载")
    except Exception as e:
        print(f"[Infer] Dense Embedding 不可用（仅使用 TF-IDF）: {e}")

    run_mapreduce_inference_file(
        knowledge_dirs=knowledge_dirs,
        input_path=TEST_CSV,
        output_path=INFER_OUTPUT,
        map_llm_func=chat,
        reduce_llm_func=chat,
        infer_prompt_func=infer_v1,
        summary_prompt_func=summary_v0,
        edge_case_prompt_func=edge_case_fallback_v0,
        pitfalls_func=potential_pitfalls,
        extra_llm_func=chat,
        extra_vendor="volc",
        extra_model="deepseek-v3.2",
        embedding_func=embedding_func,
        tfidf_top_n=TFIDF_TOP_N,
        embedding_top_n=EMBEDDING_TOP_N,
        map_max_workers=MAX_WORKERS,
        enable_edge_case_fallback=True,
    )

    print("\n推理阶段全部完成！")


def main():
    parser = argparse.ArgumentParser(description="Know-How 提炼与推理全流程")
    parser.add_argument(
        "--stage",
        choices=["extract", "infer", "all"],
        default="all",
        help="运行阶段: extract=仅提炼, infer=仅推理, all=全流程（默认）",
    )
    args = parser.parse_args()

    print("\n" + "+" * 60)
    print("  Know-How 提炼 & 推理 Pipeline")
    print("+" * 60 + "\n")

    if args.stage in ("extract", "all"):
        run_extraction()

    if args.stage in ("infer", "all"):
        if args.stage == "all":
            print("\n\n" + "#" * 60)
            print("  提炼阶段结束，进入推理阶段...")
            print("#" * 60 + "\n")
        run_inference()

    print("\n" + "+" * 60)
    print("  全部流程结束")
    print("+" * 60)


if __name__ == "__main__":
    main()
