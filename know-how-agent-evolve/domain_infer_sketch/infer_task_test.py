import pandas as pd
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# 将项目根目录添加到sys.path，确保项目内部模块可被正确导入
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from llm.llm_engine import qwen, chat
from domain_infer_sketch.prompt.domain_infer import domain_infer_v0, domain_infer_v1
import json5
from tqdm import tqdm

def parse_name_level_map(file_path: str) -> dict:
    """
    解析 levels.csv，以层级为外层 key，返回该层级下每个名称与描述的映射。

    Returns:
        {
            "level1": [{"name": "生命周期", "desc": "..."}, ...],
            "level2": [{"name": "设立",     "desc": "..."}, ...],
            ...
        }
        同一层级内名称不重复，以首次出现的 desc 为准。
    """
    df = pd.read_csv(file_path, header=0, encoding='gbk')
    df = df.dropna(axis=1, how='all')
    num_levels = len(df.columns) // 2

    result = {f'level{i + 1}': [] for i in range(num_levels)}
    seen = {f'level{i + 1}': set() for i in range(num_levels)}

    for _, row in df.iterrows():
        for level_idx in range(num_levels):
            col_i = level_idx * 2
            data_value = row.iloc[col_i] if col_i < len(row) else None
            desc_value = row.iloc[col_i + 1] if col_i + 1 < len(row) else None

            if pd.isna(data_value) or not str(data_value).strip():
                continue

            name = str(data_value).strip()
            desc = str(desc_value).strip() if pd.notna(desc_value) else ''

            level_key = f'level{level_idx + 1}'
            if name not in seen[level_key]:
                seen[level_key].add(name)
                result[level_key].append({'name': name, 'desc': desc})

    return result


def parse_ancestry_map(file_path: str) -> dict:
    """
    解析 levels.csv，返回每个名称的完整层级链（祖先 + 自身）。

    通过逐行维护「当前路径」来推断每个名称的完整层级归属。

    Returns:
        {
            "生命周期": {"level1": "生命周期"},
            "设立":     {"level1": "生命周期", "level2": "设立", "level3": "设立"},
            "采购货物": {"level1": "生命周期", "level2": "采购", "level3": "采购货物"},
            ...
        }
        所有层级的节点均包含在内，链中包含节点自身。
        同名节点跨层级出现时，保留最深（最完整）的链。
    """
    df = pd.read_csv(file_path, header=0, encoding='gbk')
    df = df.dropna(axis=1, how='all')
    num_levels = len(df.columns) // 2

    result = {}
    current_path = [None] * num_levels

    for _, row in df.iterrows():
        for level_idx in range(num_levels):
            col_i = level_idx * 2
            data_value = row.iloc[col_i] if col_i < len(row) else None

            if pd.isna(data_value) or not str(data_value).strip():
                continue

            name = str(data_value).strip()

            current_path[level_idx] = name
            for j in range(level_idx + 1, num_levels):
                current_path[j] = None

            chain = {}
            for j in range(level_idx + 1):
                if current_path[j] is not None:
                    chain[f'level{j + 1}'] = current_path[j]

            if name not in result or len(chain) > len(result[name]):
                result[name] = chain

    return result


def parse_descendant_map(file_path: str) -> dict:
    """
    解析 levels.csv，返回按层级分组、每个节点附带完整子树的结构。

    与 parse_ancestry_map（子 → 祖先，自底向上）互补，
    本函数构建自顶向下的多层递进关系，并以 levelN 为外层 key 分组。

    Returns:
        {
            "level1": {
                "生命周期": {
                    "设立": {"设立": {}},
                    "采购": {"采购货物": {}, "采购服务": {}, ...},
                    ...
                },
                "特殊事项": {...}
            },
            "level2": {
                "设立": {"设立": {}},
                "采购": {"采购货物": {}, "采购服务": {}, ...},
                "企业重组": {...},
                ...
            },
            "level3": {
                "设立": {},
                "采购货物": {},
                "采购服务": {},
                ...
            }
        }
        每个 levelN 下列出该层级的所有节点，值为其完整子树。
        同名节点在不同层级各自独立，子树不会因同名而被合并为空。
    """
    df = pd.read_csv(file_path, header=0, encoding='gbk')
    df = df.dropna(axis=1, how='all')
    num_levels = len(df.columns) // 2

    # 用 (level_idx, name) 作为唯一节点标识，避免不同层级同名节点合并
    children_of: dict[tuple, list] = {}
    nodes_at_level: dict[int, list[tuple]] = {i: [] for i in range(num_levels)}
    seen_at_level: dict[int, set[tuple]] = {i: set() for i in range(num_levels)}
    current_path: list[tuple | None] = [None] * num_levels

    for _, row in df.iterrows():
        for level_idx in range(num_levels):
            col_i = level_idx * 2
            data_value = row.iloc[col_i] if col_i < len(row) else None

            if pd.isna(data_value) or not str(data_value).strip():
                continue

            name = str(data_value).strip()
            node_key = (level_idx, name)

            current_path[level_idx] = node_key
            for j in range(level_idx + 1, num_levels):
                current_path[j] = None

            if node_key not in children_of:
                children_of[node_key] = []

            if node_key not in seen_at_level[level_idx]:
                seen_at_level[level_idx].add(node_key)
                nodes_at_level[level_idx].append(node_key)

            if level_idx > 0:
                parent_key = current_path[level_idx - 1]
                if parent_key is not None:
                    if parent_key not in children_of:
                        children_of[parent_key] = []
                    if node_key not in children_of[parent_key]:
                        children_of[parent_key].append(node_key)

    def build_tree(node_key: tuple) -> dict:
        return {child[1]: build_tree(child) for child in children_of.get(node_key, [])}

    return {
        f'level{level_idx + 1}': {
            node[1]: build_tree(node) for node in nodes_at_level[level_idx]
        }
        for level_idx in range(num_levels)
    }


def level_traceback(data: pd.DataFrame, ancestry: dict, level: str,
                    name_level: dict = None) -> pd.DataFrame:
    """
    基于层级回溯表，将 DataFrame 中指定列的每个标签展开为完整的层级链，
    并将各层级的标签（及描述）作为新列追加到 DataFrame 中返回。

    Args:
        data:       原始 DataFrame
        ancestry:   parse_ancestry_map 返回的祖先映射
                    {"采购货物": {"level1": "生命周期", "level2": "采购"}, ...}
        level:      DataFrame 中作为查找键的列名，如 "经济活动"；为空（None/空串）或列不存在时直接返回副本
        name_level: parse_name_level_map 返回的层级映射（可选），
                    提供时为每个层级额外追加 levelN_desc 列

    Returns:
        新增了 levelN_label（及 levelN_desc）列的 DataFrame 副本。
        例如 level="经济活动"，某行值为"采购货物"，则新增：
            level1_label="生命周期", level2_label="采购", level3_label="采购货物"
    """
    df = data.copy()
    # 兼容 level 为空或列不存在：直接返回副本，不追加层级列
    if level is None or (isinstance(level, str) and not level.strip()):
        return df
    if level not in df.columns:
        return df

    # 构建 name -> desc 的扁平查找表（仅在 name_level 不为空时使用）
    desc_lookup = {}
    if name_level:
        for items in name_level.values():
            for item in items:
                desc_lookup[item['name']] = item['desc']

    def build_chain(name) -> dict:
        """给定一个名称，返回其完整层级链 {level1: ..., level2: ..., levelN: name}"""
        if not isinstance(name, str) or not name.strip():
            return {}
        name = name.strip()
        ancestors = ancestry.get(name, {})
        # 自身层级号 = 最深祖先层级号 + 1；若无祖先则为 level1
        if ancestors:
            self_level = max(int(k[5:]) for k in ancestors) + 1
        else:
            self_level = 1
        return {**ancestors, f'level{self_level}': name}

    chains = df[level].map(build_chain)

    # 收集全部出现的层级 key，按层级号排序
    all_level_keys = sorted(
        {k for chain in chains for k in chain},
        key=lambda x: int(x[5:])
    )

    for lk in all_level_keys:
        df[f'{lk}_label'] = chains.map(lambda c, k=lk: c.get(k, ''))
        if name_level:
            df[f'{lk}_desc'] = df[f'{lk}_label'].map(lambda n: desc_lookup.get(n, ''))

    return df


def domain_infer(input_, domain_name, domain_level, domain_desc):
    """
    调用大模型，判断用户问所属业务领域

    params:
        input_: str | 用户问
        domain_name: str | 业务事项名称
        domain_level: str | 业务事项层级
        domain_desc: str | 业务事项描述
    """
    prompt = domain_infer_v1(input_=input_, domain_name=domain_name, domain_level=domain_level, domain_desc=domain_desc)
    res = chat(prompt)
    return res

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # 解析level数据
    FILE = "/root/know-how/know-how-extraction/know-how-agent-evolve/domain_infer_sketch/data/levels_0318_jdyw.csv" # os.path.join(DATA_DIR, "levels_0318_jdyw.csv")
    name_level = parse_name_level_map(FILE)
    ancestry = parse_ancestry_map(FILE)
    descendant = parse_descendant_map(FILE)

    with open(os.path.join(DATA_DIR, "name_level_map.json"), "w", encoding="utf-8") as f:
        json.dump(name_level, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "ancestry_map.json"), "w", encoding="utf-8") as f:
        json.dump(ancestry, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "descendant_map.json"), "w", encoding="utf-8") as f:
        json.dump(descendant, f, ensure_ascii=False, indent=2)


    # 读取待推理数据data.csv，其中‘问题’列为input，‘经济活动’列为label
    # df = pd.read_csv(os.path.join(DATA_DIR, "data_0317_zqrz.csv"), encoding="utf-8").dropna(how="all")
    df = pd.read_excel("/root/know-how/know-how-extraction/know-how-agent-evolve/domain_infer_sketch/data/keywords_jdyw_data.xlsx",engine='openpyxl').dropna(how="all")
    df.rename(columns={"原问题":"问题"}, inplace = True)
    print("数据总量:", df.shape)

    result = level_traceback(df, ancestry, level="", name_level=name_level)
    label_cols = [c for c in result.columns if c.endswith("_label") or c.endswith("_desc")]
    preview = result[["问题"] + label_cols].head(3).to_dict(orient="records")

    # 创建结果表，记录每个问题的逐层推理结果
    df_copy = df[["问题"]].copy(deep=True)
    # infer_result = {
    #     "level1":[
    #         {"question": "问题", "label": "经济活动标签", "target": "业务事项名称", "judgements": "业务事项名称判断结果", "reasoning": "业务事项名称判断思考"}
    #         ...
            # ]
    #     ...
    # }
    infer_result = {
        "level1":[],
        # "level2":[],
        # "level3":[],
    }

    MAX_RETRIES = 999
    RETRY_DELAY = 5  # 秒
    MAX_WORKERS = 1  # 每层级并发推理的线程数，可按需调整
    json_write_lock = threading.Lock()

    def _infer_one(question_text, task_tuple):
        """单条推理：用于线程池。返回 (n, domain_name, judgement, reasoning)。"""
        n, domain_name, domain_desc, domain_descendant = task_tuple
        judgement_res = domain_infer(question_text, domain_name, domain_descendant, domain_desc)
        res_content = json5.loads(judgement_res["content"])
        return (n, domain_name, res_content["judgement"], res_content["reasoning"])

#     # 循环对每个问题进行逐层推理，并判断是否属于制定的业务事项
#     for index, row in tqdm(df_copy.iterrows()):  # 遍历每个问题
#         input_ = row["问题"]
#         fine_level_label = ""#row["事项"]  # 最细粒度层级的业务事项标签

#         for attempt in range(1, MAX_RETRIES + 1):
#             try:
#                 upsteam_set = set()
#                 question_results = {level: [] for level in infer_result}  # 本题临时结果，失败时不污染 infer_result

#                 for l in range(len(name_level)):  # 遍历每个层级
#                     level_number = l + 1
#                     domain_level = f"level{level_number}"
#                     # 收集本层级需推理的 (n, domain_name, domain_desc, domain_descendant)
#                     tasks = []
#                     for n in range(len(name_level[domain_level])):
#                         domain_name = name_level[domain_level][n]["name"]
#                         domain_desc = name_level[domain_level][n]["desc"]
#                         domain_descendant = descendant[domain_level][domain_name]
#                         if domain_level != "level1":
#                             upstream_domain = ancestry[domain_name][f"level{level_number-1}"]
#                             if upstream_domain not in upsteam_set:
#                                 continue
#                         tasks.append((n, domain_name, domain_desc, domain_descendant))

#                     # 本层级多线程并发推理
#                     level_results = []
#                     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                         futures = [executor.submit(_infer_one, input_, t) for t in tasks]
#                         for future in as_completed(futures):
#                             level_results.append(future.result())
#                     level_results.sort(key=lambda x: x[0])  # 按 n 排序，保证顺序一致

#                     for _, domain_name, judgement, reasoning in level_results:
#                         question_results[domain_level].append({
#                             "question_id": index,
#                             "question": input_,
#                             "label": fine_level_label,
#                             "target": domain_name,
#                             "judgement": judgement,
#                             "reasoning": reasoning,
#                         })
#                         if judgement:
#                             upsteam_set.add(domain_name)

#                 # 本题全部层级推理成功，加锁合并到总结果并持久化
#                 with json_write_lock:
#                     for level in infer_result:
#                         infer_result[level].extend(question_results[level])
#                     with open(os.path.join(DATA_DIR, "domain_infer_result_jdyw.json"), "w", encoding="utf-8") as f:
#                         json.dump(infer_result, f, ensure_ascii=False, indent=2)
#                 break  # 成功，退出重试循环

#             except Exception as e:
#                 print(f"\n[question_id={index}] 第 {attempt}/{MAX_RETRIES} 次尝试失败：{e}")
#                 if attempt < MAX_RETRIES:
#                     time.sleep(RETRY_DELAY)
#                 else:
#                     print(f"[question_id={index}] 已达最大重试次数，跳过该问题。")


    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # 全局锁和结果容器
    json_write_lock = threading.Lock()
    infer_result = {f"level{i+1}": [] for i in range(len(name_level))}

    def process_single_question(question_data):
        """处理单个问题的完整逻辑（包含内部层级循环）"""
        index, row = question_data
        input_ = row["问题"]
        fine_level_label = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                upsteam_set = set()
                question_results = {level: [] for level in infer_result}

                # 内部层级循环（保持串行，每层内部并发）
                for l in range(len(name_level)):
                    level_number = l + 1
                    domain_level = f"level{level_number}"

                    # 构建当前层级任务
                    tasks = []
                    for n in range(len(name_level[domain_level])):
                        domain_name = name_level[domain_level][n]["name"]
                        domain_desc = name_level[domain_level][n]["desc"]
                        domain_descendant = descendant[domain_level][domain_name]

                        if domain_level != "level1":
                            upstream_domain = ancestry[domain_name][f"level{level_number-1}"]
                            if upstream_domain not in upsteam_set:
                                continue
                        tasks.append((n, domain_name, domain_desc, domain_descendant))

                    # 单层并发推理
                    level_results = []
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(_infer_one, input_, t) for t in tasks]
                        for future in as_completed(futures):
                            level_results.append(future.result())

                    level_results.sort(key=lambda x: x[0])

                    # 处理结果
                    for _, domain_name, judgement, reasoning in level_results:
                        question_results[domain_level].append({
                            "question_id": index,
                            "question": input_,
                            "label": fine_level_label,
                            "target": domain_name,
                            "judgement": judgement,
                            "reasoning": reasoning,
                        })
                        if judgement:
                            upsteam_set.add(domain_name)

                # 成功后写入（加锁保护）
                with json_write_lock:
                    for level in infer_result:
                        infer_result[level].extend(question_results[level])
                    with open(os.path.join(DATA_DIR, "domain_infer_result_jdyw.json"), "w", encoding="utf-8") as f:
                        json.dump(infer_result, f, ensure_ascii=False, indent=2)
                return index, "success", None

            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    return index, "failed", str(e)

    # 外层并发执行
    question_items = list(df_copy.iterrows())
    with ThreadPoolExecutor(max_workers=16) as executor:  # 控制并发问题数
        futures = {executor.submit(process_single_question, item): item[0] for item in question_items}

        # 进度追踪
        with tqdm(total=len(question_items)) as pbar:
            for future in as_completed(futures):
                idx, status, error = future.result()
                if status == "failed":
                    print(f"[question_id={idx}] 处理失败：{error}")
                pbar.update(1)