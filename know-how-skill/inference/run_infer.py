"""
推理阶段 CLI 入口
=================
从 inference/input/ 读取待推理数据，执行 4 阶段 MapReduce 推理，
结果输出到 inference/output/，在原始数据后追加中间结果和最终答案列。

用法:
  # 从 know-how-skill/ 目录运行
  python inference/run_infer.py --input 借款业务评测集_324.csv --knowledge-dirs path/to/kd1 path/to/kd2

  # 指定输出格式（默认与输入一致）
  python inference/run_infer.py --input test.xlsx --knowledge-dirs ./extraction/qa_know_how_build/knowledge/xxx_knowledge --output-format xlsx

  # 完整参数示例
  python inference/run_infer.py \\
      --input test.csv \\
      --knowledge-dirs kd1 kd2 \\
      --tfidf-top-n 5 \\
      --embedding-top-n 5 \\
      --edge-cases-top-n 3 \\
      --max-workers 8 \\
      --question-column question
"""

import argparse
import os
import sys
from datetime import datetime

INFER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(INFER_DIR)
INPUT_DIR = os.path.join(INFER_DIR, "input")
OUTPUT_DIR = os.path.join(INFER_DIR, "output")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, INFER_DIR)


def _resolve_input_path(input_arg: str) -> str:
    """解析输入文件路径：支持绝对路径和相对于 input/ 目录的路径。"""
    if os.path.isabs(input_arg):
        return input_arg
    candidate = os.path.join(INPUT_DIR, input_arg)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(input_arg):
        return os.path.abspath(input_arg)
    raise FileNotFoundError(
        f"找不到输入文件: {input_arg}\n"
        f"  已搜索: {candidate}, {os.path.abspath(input_arg)}"
    )


def _resolve_output_path(input_path: str, output_format: str = None) -> str:
    """根据输入文件名生成输出文件路径，放在 output/ 目录下。"""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if output_format is None:
        ext = os.path.splitext(input_path)[1]
    else:
        ext = f".{output_format.lstrip('.')}"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_name = f"{stem}_result_{timestamp}{ext}"
    return os.path.join(OUTPUT_DIR, output_name)


def main():
    parser = argparse.ArgumentParser(
        description="Know-How 推理（4 阶段 MapReduce）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python inference/run_infer.py \\\n"
            "      --input 借款业务评测集_324.csv \\\n"
            "      --knowledge-dirs extraction/qa_know_how_build/knowledge/xxx_knowledge"
        ),
    )
    parser.add_argument(
        "--input", required=True,
        help="输入文件（.csv/.xlsx），可为绝对路径或相对于 inference/input/ 的文件名",
    )
    parser.add_argument(
        "--knowledge-dirs", nargs="+", required=True,
        help="参与推理的 knowledge 目录列表（绝对路径或相对于项目根目录）",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出文件路径（默认自动生成到 inference/output/）",
    )
    parser.add_argument(
        "--output-format", choices=["csv", "xlsx"], default=None,
        help="输出格式（默认与输入一致）",
    )
    parser.add_argument(
        "--question-column", default="question",
        help="输入文件中问题所在的列名（默认: question）",
    )
    parser.add_argument(
        "--tfidf-top-n", type=int, default=5,
        help="TF-IDF 检索 Top-N（默认: 5）",
    )
    parser.add_argument(
        "--embedding-top-n", type=int, default=5,
        help="Dense Embedding 检索 Top-N（默认: 5）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="单题内 Map/Phase3 并发线程数（默认: 4）",
    )
    parser.add_argument(
        "--question-workers", type=int, default=1,
        help="问题级别并发数（默认: 1 即串行；设为 2~4 可同时处理多道题，"
             "总并发 ≈ question-workers × max-workers，请根据 API 并发上限设置）",
    )
    parser.add_argument(
        "--no-edge-cases", action="store_true",
        help="禁用 Phase 3 边缘案例兜底（默认开启）",
    )
    parser.add_argument(
        "--no-qa-direct", action="store_true",
        help="禁用 QA 直检并行路径（默认开启）",
    )
    parser.add_argument(
        "--no-extra-llm", action="store_true",
        help="禁用 Reduce 阶段额外 LLM 裸考推理（默认开启）",
    )
    args = parser.parse_args()

    # ── 路径解析 ──
    input_path = _resolve_input_path(args.input)
    print(f"[Input]  {input_path}")

    knowledge_dirs = []
    for d in args.knowledge_dirs:
        if os.path.isabs(d):
            resolved = d
        else:
            resolved = os.path.join(PROJECT_DIR, d)
        if not os.path.isdir(resolved):
            print(f"[!] Knowledge 目录不存在: {resolved}")
            sys.exit(1)
        knowledge_dirs.append(resolved)
    print(f"[Knowledge] {len(knowledge_dirs)} 个目录:")
    for d in knowledge_dirs:
        print(f"  - {d}")

    if args.output:
        output_path = args.output
    else:
        output_path = _resolve_output_path(input_path, args.output_format)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[Output] {output_path}")

    # ── 加载 LLM 和 Prompt ──
    print("\n[Init] 加载 LLM 和 Prompt 模块...")
    try:
        from llm_client import chat
    except ImportError as e:
        print(f"[!] 无法加载 llm_client: {e}")
        sys.exit(1)

    from prompts_infer import infer_v1, summary_v0, potential_pitfalls, edge_case_fallback_v0, qa_direct_infer_v0

    embedding_func = None
    try:
        from utils import get_embeddings
        embedding_func = get_embeddings
        print("[Init] Dense Embedding 服务已加载")
    except Exception as e:
        print(f"[Init] Dense Embedding 不可用（仅使用 TF-IDF）: {e}")

    extra_llm_func = None if args.no_extra_llm else chat

    # ── 执行推理 ──
    print("\n" + "=" * 60)
    print("  开始推理（4 阶段 MapReduce）")
    print("=" * 60 + "\n")

    from inference.mapreduce_infer import run_mapreduce_inference_file

    enable_edge = not args.no_edge_cases
    enable_qa_direct = not args.no_qa_direct
    qa_direct_prompt = qa_direct_infer_v0 if enable_qa_direct else None

    run_mapreduce_inference_file(
        knowledge_dirs=knowledge_dirs,
        input_path=input_path,
        output_path=output_path,
        map_llm_func=chat,
        reduce_llm_func=chat,
        infer_prompt_func=infer_v1,
        summary_prompt_func=summary_v0,
        edge_case_prompt_func=edge_case_fallback_v0,
        qa_direct_prompt_func=qa_direct_prompt,
        pitfalls_func=potential_pitfalls,
        extra_llm_func=extra_llm_func,
        embedding_func=embedding_func,
        tfidf_top_n=args.tfidf_top_n,
        embedding_top_n=args.embedding_top_n,
        map_max_workers=args.max_workers,
        question_max_workers=args.question_workers,
        enable_edge_cases=enable_edge,
        enable_qa_direct=enable_qa_direct,
        question_column=args.question_column,
    )

    print("\n" + "=" * 60)
    print("  推理完成！")
    print(f"  结果文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
