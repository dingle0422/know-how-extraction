import json
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INFER_RESULT_PATH  = os.path.join(BASE_DIR, "data", "infer_result.json")
ANCESTRY_MAP_PATH  = os.path.join(BASE_DIR, "data", "ancestry_map.json")
DATA_CSV_PATH      = os.path.join(BASE_DIR, "data", "data_0317_zqrz.csv")
OUTPUT_XLSX_PATH   = os.path.join(BASE_DIR, "data", "infer_result.xlsx")


def assemble():
    with open(INFER_RESULT_PATH, "r", encoding="utf-8") as f:
        infer_result = json.load(f)

    with open(ANCESTRY_MAP_PATH, "r", encoding="utf-8") as f:
        ancestry_map = json.load(f)

    # 读取 data.csv，取 QID 列（唯一问题编号）和问题列
    df_data = pd.read_csv(DATA_CSV_PATH, encoding="utf-8-sig").reset_index()
    df_data = df_data.rename(columns={"index": "question_id"})

    # 找出 QID 列（含"唯一"的列名）
    qid_col = next((c for c in df_data.columns if "唯一" in c), df_data.columns[1])
    question_col = "问题"

    df_qid = df_data[["question_id", qid_col, question_col, "经济活动"]].rename(
        columns={qid_col: "QID", question_col: "question"}
    ).copy()

    def get_level_label(fine_label: str, level_key: str) -> str:
        """
        将最细粒度的经济活动标签反推到指定层级的事项名称。
        若该层级不存在（如 fine_label 本身比 level_key 更浅），
        则取 ancestry_map 中最深的可用层级。
        """
        chain = ancestry_map.get(fine_label, {})
        if level_key in chain:
            return chain[level_key]
        # 取链中最深层作为兜底
        if chain:
            deepest = max(chain.keys(), key=lambda k: int(k[5:]))
            return chain[deepest]
        return fine_label

    levels = sorted(
        [k for k in infer_result if k.startswith("level")],
        key=lambda k: int(k[5:])
    )

    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        for level_key in levels:
            records = [
                item for item in infer_result[level_key]
                if item.get("judgement") is True
            ]
            if not records:
                continue

            # label 始终由 data.csv 的经济活动反推到本层级，与是否命中无关
            df_qid[f"label_{level_key}"] = df_qid["经济活动"].apply(
                lambda fine: get_level_label(str(fine), level_key)
                if pd.notna(fine) else ""
            )

            # confirmed：judgement=true 的记录（问题已被确认属于某 target）
            df_confirmed = pd.DataFrame(records)[["question_id", "target", "reasoning"]] \
                if records else pd.DataFrame(columns=["question_id", "target", "reasoning"])
            df_confirmed = df_confirmed.rename(columns={"target": "confirmed", "reasoning": "reasoning"})

            # fallback_reasoning：judgement!=true 但 target==label 的记录（解释为何正确答案未被选中）
            all_records = infer_result[level_key]
            df_qid_labels = df_qid[["question_id", f"label_{level_key}"]].rename(
                columns={f"label_{level_key}": "label"}
            )
            fallback_rows = []
            for item in all_records:
                if item.get("judgement") is True:
                    continue
                qid = item["question_id"]
                item_label_row = df_qid_labels[df_qid_labels["question_id"] == qid]
                if item_label_row.empty:
                    continue
                row_label = item_label_row.iloc[0]["label"]
                if item.get("target") == row_label:
                    fallback_rows.append({
                        "question_id": qid,
                        "fallback_reasoning": item["reasoning"]
                    })
            df_fallback = pd.DataFrame(fallback_rows) \
                if fallback_rows else pd.DataFrame(columns=["question_id", "fallback_reasoning"])

            # outer join：以 data.csv 为基准，依次合并 confirmed 和 fallback_reasoning
            df = df_qid[["question_id", "QID", "question", f"label_{level_key}"]].merge(
                df_confirmed, on="question_id", how="left"
            ).merge(
                df_fallback, on="question_id", how="left"
            )

            # confirmed 为空时，用 fallback_reasoning 填充 reasoning 列
            df["reasoning"] = df["reasoning"].where(df["confirmed"].notna(), df["fallback_reasoning"])

            df = df.rename(columns={f"label_{level_key}": "label"})
            df = df[["QID", "question", "label", "confirmed", "reasoning"]]

            df.to_excel(writer, sheet_name=level_key, index=False)
            print(f"{level_key}: {len(df)} 行")

    print(f"\n已保存至 {OUTPUT_XLSX_PATH}")


def single_question_columns_assemble():
    with open(INFER_RESULT_PATH, "r", encoding="utf-8") as f:
        infer_result = json.load(f)

    # 读取CSV（只有一列问题数据，无QID、无经济活动列）
    df_data = pd.read_csv(DATA_CSV_PATH, encoding="utf-8-sig")
    
    # 自动识别问题列：优先"问题"/"question"列，否则使用第一列
    question_candidates = [c for c in df_data.columns 
                          if "问题" in c or "question" in str(c).lower()]
    question_col = question_candidates[0] if question_candidates else df_data.columns[0]
    
    # 构建基础数据表：自动生成question_id和QID
    df_base = pd.DataFrame({
        "question_id": range(len(df_data)),
        "QID": [f"Q{i+1}" for i in range(len(df_data))],  # 生成Q1, Q2...
        "question": df_data[question_col].astype(str)
    })

    # 获取所有level并排序（level1, level2...）
    levels = sorted(
        [k for k in infer_result if k.startswith("level")],
        key=lambda k: int(k[5:])
    )

    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        for level_key in levels:
            records = infer_result[level_key]
            if not records:
                continue

            # 将当前level的推理结果转为DataFrame
            df_infer = pd.DataFrame(records)
            
            # 防御性处理：确保有question_id，如果没有则假设顺序对应
            if "question_id" not in df_infer.columns:
                df_infer.insert(0, "question_id", range(len(df_infer)))
            
            # 只保留需要的列，并将target重命名为confirmed
            keep_cols = ["question_id", "target", "reasoning"]
            if "judgement" in df_infer.columns:
                keep_cols.append("judgement")
                
            # 确保列存在（防止字段缺失报错）
            for col in keep_cols:
                if col not in df_infer.columns:
                    df_infer[col] = None
                    
            df_infer = df_infer[keep_cols].rename(columns={"target": "confirmed"})
            
            # 以df_base为基准（保证所有问题都出现），合并推理结果
            df_merged = df_base.merge(df_infer, on="question_id", how="left")
            
            # 整理输出列顺序：QID, 问题, 判定结果, 推理过程, (是否正确)
            output_cols = ["QID", "question", "confirmed", "reasoning"]
            if "judgement" in df_infer.columns:
                output_cols.append("judgement")
                
            df_output = df_merged[[c for c in output_cols if c in df_merged.columns]]
            
            # 写入Excel，每个level一个sheet
            df_output.to_excel(writer, sheet_name=level_key, index=False)
            print(f"{level_key}: {len(df_output)} 行")

    print(f"\n已保存至 {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    single_question_columns_assemble()
