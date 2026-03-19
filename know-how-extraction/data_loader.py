"""
数据加载与预处理模块
负责读取原始 Excel 数据、合并专家答案、切分训练/测试集。
"""

import pandas as pd


def load_and_prepare(
    answer_file: str,
    question_file: str,
    train_ratio: float = 1.0,
    test_start_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载数据并完成预处理，返回 (全量 data, 训练集, 测试集)。

    Parameters
    ----------
    answer_file : 专家答案 Excel 路径（含 question, answer 列）
    question_file : 原始问题 Excel 路径（含 序号, 问题, 答案 列）
    train_ratio : 取全量数据的前 N% 作为训练集基数
    test_start_ratio : 从训练集基数的 N% 位置开始划分测试集
    """
    ans_data = pd.read_excel(answer_file, engine="openpyxl")[["question", "answer"]]

    data = pd.read_excel(question_file, engine="openpyxl")
    data.rename(columns={"序号": "id", "问题": "question", "答案": "answer"}, inplace=True)
    data["policy"] = None
    data["reasoning"] = None

    data = ans_data.merge(data, how="inner", on="question")

    data["policy"] = data["policy"].fillna("暂无")
    data["reasoning"] = data["reasoning"].fillna("暂无")
    data["answer"] = data["answer"] + "\n\n- 政策依据\n" + data["policy"]

    cut = int(len(data) * train_ratio)
    data_train = data.loc[:cut, :].copy()
    test_cut = int(cut * test_start_ratio)
    data_test = data.loc[test_cut:, :].copy().reset_index(drop=True)

    return data, data_train, data_test


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """直接从已有 CSV 加载测试数据。"""
    return pd.read_csv(csv_path)


def save_test_data(data_test: pd.DataFrame, output_path: str):
    """将测试集保存为 CSV。"""
    data_test[["question", "answer", "confirmed"]].to_csv(output_path, index=False)
