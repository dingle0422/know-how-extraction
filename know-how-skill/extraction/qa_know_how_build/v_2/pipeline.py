"""
V2 QA 知识提取完整流水线：Level 1 (复用 v1) → Level 2 增量精炼 → Knowledge 发布。
支持命令行独立运行和函数调用。
"""

import json
import os
import sys

_V_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_V_DIR)
_EXTRACTION_DIR = os.path.dirname(_PACKAGE_DIR)
_SKILL_ROOT = os.path.dirname(_EXTRACTION_DIR)

sys.path.insert(0, _SKILL_ROOT)
sys.path.insert(0, _EXTRACTION_DIR)
sys.path.insert(0, _V_DIR)
sys.path.insert(0, _PACKAGE_DIR)
sys.path.insert(0, os.path.join(_PACKAGE_DIR, "v_1"))

_SUPPORTED_QA_EXTS = {".csv", ".xlsx", ".xls"}


def run_full_pipeline_for_qa_v2(
    source_file: str,
    llm_func,
    level1_prompt_func,
    output_dir: str,
    knowledge_dir: str,
    cosine_threshold: float = 0.75,
    level1_max_workers: int = 4,
    level2_max_workers: int = 4,
    max_retries: int = 100,
    max_retries_per_step: int = 5,
    column_map: dict[str, str] | None = None,
    extra_columns: list[str] | None = None,
    embedding_func=None,
    tfidf_weight: float = 1.0,
    embedding_weight: float = 0.0,
):
    """
    V2 完整流水线：对单个 QA 源文件执行 Level 1 → Level 2 增量精炼 → Knowledge 发布。

    Parameters
    ----------
    source_file : QA 源数据文件路径 (.csv / .xlsx / .xls)
    llm_func : LLM 调用函数
    level1_prompt_func : Level 1 一级提炼 prompt 函数 (如 single_v1)
    output_dir : 中间产物输出目录
    knowledge_dir : 最终 knowledge 发布目录
    cosine_threshold : 聚类 cosine 相似度阈值
    level1_max_workers : Level 1 并发线程数
    level2_max_workers : Level 2 簇间并发线程数
    max_retries : Level 1 每项最大重试次数
    max_retries_per_step : Level 2 每步 LLM 调用最大重试次数
    column_map : 列名映射 {标准名: 实际列名}，如 {"question": "问题", "answer": "回答"}
    extra_columns : 指定哪些非核心列纳入 Extra_Information，为 None 或空时不生成额外信息
    embedding_func : Dense embedding 函数，为 None 时回退纯 TF-IDF
    tfidf_weight : TF-IDF 相似度权重，设为 0 跳过 TF-IDF
    embedding_weight : Dense Embedding 相似度权重，设为 0 跳过 Embedding
    """
    import pandas as pd
    from level1_extract import run_level1_extraction
    from level2_refine import run_level2_refinement
    from utils import get_source_stem, publish_to_knowledge, build_retrieval_index

    source_stem = get_source_stem(source_file)
    os.makedirs(output_dir, exist_ok=True)

    # ── 数据加载 ──────────────────────────────────────────────────────────
    ext = os.path.splitext(source_file)[1].lower()
    if ext == ".csv":
        data_train = pd.read_csv(source_file, encoding="utf-8-sig")
    else:
        data_train = pd.read_excel(source_file, sheet_name=0)

    if column_map:
        rename = {v: k for k, v in column_map.items() if v in data_train.columns}
        missing_src = [
            f"{k} -> '{v}'"
            for k, v in column_map.items()
            if v not in data_train.columns
        ]
        if missing_src:
            print(f"  [警告] column-map 中以下源列在文件中不存在，已忽略: {missing_src}")
        if rename:
            data_train = data_train.rename(columns=rename)
            print(f"  列名映射: {rename}")

    required_cols = {"question", "answer"}
    missing = required_cols - set(data_train.columns)
    if missing:
        raise ValueError(
            f"源文件 {os.path.basename(source_file)} 缺少必需列: {missing}\n"
            f"要求: question, answer 为必需列；reasoning 可选"
        )

    if "reasoning" not in data_train.columns:
        data_train["reasoning"] = ""
        print(f"  提示: 源文件中不含 reasoning 列，已自动创建并置空")

    if "Extra_Information" not in data_train.columns:
        if extra_columns:
            valid_extra = [c for c in extra_columns if c in data_train.columns]
            invalid_extra = [c for c in extra_columns if c not in data_train.columns]
            if invalid_extra:
                print(f"  [警告] --extra-columns 中以下列在文件中不存在，已忽略: {invalid_extra}")
            if valid_extra:
                data_train["Extra_Information"] = data_train[valid_extra].apply(
                    lambda row: "; ".join(
                        f"{k}={v}" for k, v in row.items() if pd.notna(v)
                    ),
                    axis=1,
                )
                print(f"  Extra_Information 已从以下列生成: {valid_extra}")
            else:
                data_train["Extra_Information"] = ""
        else:
            data_train["Extra_Information"] = ""
    print(f"  数据加载完成: {len(data_train)} 条记录")

    # ── Level 1: 一级提炼（复用 v1）──────────────────────────────────────
    level1_file = os.path.join(output_dir, f"{source_stem}_level1_extraction.json")
    print(f"\n  [Level 1] 一级知识提炼: {os.path.basename(source_file)}")
    run_level1_extraction(
        data_train=data_train,
        llm_func=llm_func,
        prompt_func=level1_prompt_func,
        output_file=level1_file,
        max_workers=level1_max_workers,
        max_retries=max_retries,
    )

    # Level 1 Markdown 预览
    with open(level1_file, encoding="utf-8") as f:
        l1_data = json.load(f)
    md_file = level1_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for k, v in sorted(l1_data.items(), key=lambda x: int(x[0])):
            kh = v.get("Know_How", "").strip()
            if kh:
                f.write(kh)
                f.write("\n\n---\n\n")

    # ── Level 2: V2 增量精炼 ─────────────────────────────────────────────
    level2_file = os.path.join(output_dir, f"{source_stem}_level2_refinement.json")
    edge_cases_file = os.path.join(output_dir, f"{source_stem}_edge_cases.json")
    general_cases_file = os.path.join(output_dir, f"{source_stem}_general_cases.json")

    print(f"\n  [Level 2] V2 增量精炼: {os.path.basename(source_file)}")
    run_level2_refinement(
        level1_file=level1_file,
        llm_func=llm_func,
        output_file=level2_file,
        edge_cases_file=edge_cases_file,
        general_cases_file=general_cases_file,
        cosine_threshold=cosine_threshold,
        max_workers=level2_max_workers,
        max_retries_per_step=max_retries_per_step,
        source_file=os.path.basename(source_file),
        embedding_func=embedding_func,
        tfidf_weight=tfidf_weight,
        embedding_weight=embedding_weight,
    )

    # Level 2 Markdown 预览
    with open(level2_file, encoding="utf-8") as f:
        l2_data = json.load(f)
    md_file = level2_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        for k, v in sorted(l2_data.items(), key=lambda x: int(x[0])):
            kh = v.get("know_how", {})
            if not kh or v.get("status") != "success":
                continue
            title = kh.get("title", "Untitled")
            scope = kh.get("scope", "")
            f.write(f"## {title}\n\n")
            if scope:
                f.write(f"**适用场景**: {scope}\n\n")
            steps = kh.get("steps", [])
            if steps:
                f.write("**操作步骤**:\n")
                for s in steps:
                    line = f"{s.get('step', '?')}. {s.get('action', '')}"
                    if s.get("condition"):
                        line += f" （条件: {s['condition']}）"
                    if s.get("outcome"):
                        line += f" → {s['outcome']}"
                    f.write(f"  {line}\n")
                f.write("\n")
            exceptions = kh.get("exceptions", [])
            if exceptions:
                f.write("**例外情况**:\n")
                for ex in exceptions:
                    f.write(f"  - 当 {ex.get('when', '?')} → {ex.get('then', '?')}\n")
                f.write("\n")
            constraints = kh.get("constraints", [])
            if constraints:
                f.write("**约束/依据**:\n")
                for c in constraints:
                    f.write(f"  - {c}\n")
                f.write("\n")
            f.write("---\n\n")
    print(f"  Markdown 预览已导出: {md_file}")

    # ── 发布到 knowledge 目录 ─────────────────────────────────────────────
    knowledge_sub = os.path.join(knowledge_dir, f"{source_stem}_knowledge")
    knowledge_json = os.path.join(knowledge_sub, "knowledge.json")
    knowledge_md = os.path.join(knowledge_sub, "knowledge.md")
    if os.path.exists(knowledge_json) and os.path.exists(knowledge_md):
        print(f"  [跳过] Knowledge 目录已存在: {knowledge_sub}")
    else:
        print(f"\n  [Knowledge] 发布到 knowledge 目录...")
        _ext = os.path.splitext(source_file)[1].lower()
        if _ext == ".csv":
            with open(source_file, "r", encoding="utf-8") as f:
                source_text_head = f.read(20000)
        else:
            source_text_head = data_train.head(200).to_string(
                index=False, max_colwidth=120
            )[:20000]
        publish_to_knowledge(
            source_stem=source_stem,
            final_json_path=level2_file,
            knowledge_base_dir=knowledge_dir,
            llm_func=llm_func,
            source_text_head=source_text_head,
            level1_json_path=level1_file,
        )

    # ── 构建检索索引 ─────────────────────────────────────────────────
    _knowledge_json = os.path.join(knowledge_sub, "knowledge.json")
    if os.path.exists(_knowledge_json):
        _emb_func = None
        try:
            import importlib.util as _ilu
            _skill_utils_path = os.path.join(_SKILL_ROOT, "utils.py")
            if os.path.exists(_skill_utils_path):
                _spec = _ilu.spec_from_file_location("_skill_utils", _skill_utils_path)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                _emb_func = _m.get_embeddings
        except Exception:
            pass
        build_retrieval_index(
            knowledge_json_path=_knowledge_json,
            knowledge_dir=knowledge_sub,
            embedding_func=_emb_func,
        )

    # 复制案例库到 knowledge 目录
    import shutil
    if os.path.exists(general_cases_file):
        dst = os.path.join(knowledge_sub, "general_cases.json")
        shutil.copy2(general_cases_file, dst)
        print(f"  [Knowledge] 通用案例库已复制到: {dst}")
    if os.path.exists(edge_cases_file):
        dst = os.path.join(knowledge_sub, "edge_cases.json")
        shutil.copy2(edge_cases_file, dst)
        print(f"  [Knowledge] 边缘案例库已复制到: {dst}")

    return level2_file


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from llm_client import chat
    from prompts_v1 import single_v1

    parser = argparse.ArgumentParser(
        description="QA 知识抽取 V2 流水线（Level 1 → Level 2 增量精炼 → Knowledge）"
    )
    parser.add_argument(
        "--files", "-f", nargs="+", default=None,
        help="指定要处理的源数据文件路径（支持多个）；不指定则处理 input 目录下所有文件",
    )
    parser.add_argument(
        "--cosine-threshold", "-t", type=float, default=0.60,
        help="聚类 cosine 相似度阈值 (默认 0.60)",
    )
    parser.add_argument(
        "--level1-workers", type=int, default=os.cpu_count() or 4,
        help="Level 1 并发线程数",
    )
    parser.add_argument(
        "--level2-workers", type=int, default=4,
        help="Level 2 簇间并发线程数",
    )
    parser.add_argument(
        "--column-map", "-m", nargs="*", metavar="FIELD=COL_NAME",
        help=(
            "列名映射，格式: 标准字段名=文件实际列名。"
            "标准字段: question, answer, reasoning。"
            "示例: --column-map question=问题 answer=回答 reasoning=推理过程"
        ),
    )
    parser.add_argument(
        "--extra-columns", "-e", nargs="*", metavar="COL_NAME",
        help=(
            "指定哪些非核心列纳入 Extra_Information。"
            "不指定则不生成 Extra_Information。"
            "示例: --extra-columns 产品类型 客户等级 渠道"
        ),
    )
    parser.add_argument(
        "--tfidf-weight", type=float, default=1.0,
        help="聚类时 TF-IDF 相似度权重 (默认 1.0，设为 0 跳过 TF-IDF)",
    )
    parser.add_argument(
        "--embedding-weight", type=float, default=0.0,
        help="聚类时 Dense Embedding 相似度权重 (默认 0.0，设为 0 跳过 Embedding)",
    )
    args = parser.parse_args()

    column_map: dict[str, str] | None = None
    if args.column_map:
        _VALID_FIELDS = {"question", "answer", "reasoning"}
        column_map = {}
        for item in args.column_map:
            if "=" not in item:
                parser.error(f"column-map 格式错误 '{item}'，应为 FIELD=COL_NAME")
            field, col = item.split("=", 1)
            if field not in _VALID_FIELDS:
                parser.error(
                    f"未知标准字段 '{field}'，可选: {', '.join(sorted(_VALID_FIELDS))}"
                )
            column_map[field] = col

    input_dir = os.path.join(_PACKAGE_DIR, "input")
    output_dir = os.path.join(_PACKAGE_DIR, "output")
    knowledge_dir = os.path.join(_PACKAGE_DIR, "knowledge")

    if args.files:
        source_files = []
        for fp in args.files:
            fp = os.path.abspath(fp)
            if not os.path.isfile(fp):
                print(f"[警告] 文件不存在，已跳过: {fp}")
                continue
            if os.path.splitext(fp)[1].lower() not in _SUPPORTED_QA_EXTS:
                print(f"[警告] 不支持的文件类型，已跳过: {fp}")
                continue
            source_files.append(fp)
        if not source_files:
            raise FileNotFoundError("指定的文件中没有可处理的有效文件")
        mode_desc = "指定文件模式"
    else:
        source_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_QA_EXTS
        ])
        if not source_files:
            raise FileNotFoundError(
                f"input 目录中未找到支持的数据文件（{_SUPPORTED_QA_EXTS}）：{input_dir}"
            )
        mode_desc = "全量扫描模式"

    _emb_func = None
    if args.embedding_weight > 0:
        try:
            import importlib.util as _ilu
            _skill_utils_path = os.path.join(_SKILL_ROOT, "utils.py")
            if os.path.exists(_skill_utils_path):
                _spec = _ilu.spec_from_file_location("_skill_utils", _skill_utils_path)
                _m = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_m)
                _emb_func = _m.get_embeddings
        except Exception as _e:
            print(f"[警告] 无法加载 embedding 服务: {_e}，聚类将回退纯 TF-IDF")

    print("=" * 60)
    print(f"[V2 Pipeline] {mode_desc}，共 {len(source_files)} 个源数据文件")
    print(f"  cosine_threshold={args.cosine_threshold}, "
          f"tfidf_weight={args.tfidf_weight}, "
          f"embedding_weight={args.embedding_weight}")
    print("=" * 60)
    for i, fp in enumerate(source_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    for idx, source_file in enumerate(source_files, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{idx}/{len(source_files)}] 处理: {os.path.basename(source_file)}")
        print(f"{'═' * 60}")

        run_full_pipeline_for_qa_v2(
            source_file=source_file,
            llm_func=chat,
            level1_prompt_func=single_v1,
            output_dir=output_dir,
            knowledge_dir=knowledge_dir,
            cosine_threshold=args.cosine_threshold,
            level1_max_workers=args.level1_workers,
            level2_max_workers=args.level2_workers,
            max_retries=100,
            max_retries_per_step=5,
            column_map=column_map,
            extra_columns=args.extra_columns or None,
            embedding_func=_emb_func,
            tfidf_weight=args.tfidf_weight,
            embedding_weight=args.embedding_weight,
        )

    print(f"\n{'═' * 60}")
    print(f"[V2 Pipeline] 全部 {len(source_files)} 个数据文件处理完成！")
    print(f"{'═' * 60}")
