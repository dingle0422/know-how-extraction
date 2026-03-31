import logging
from typing import Union

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

EMBEDDING_URL = "http://mlp.paas.dc.servyou-it.com/hw-embedding-bge/v1/embeddings"
EMBEDDING_MODEL = "bge-m3"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_embeddings(
    texts: Union[str, list[str]],
    model: str = EMBEDDING_MODEL,
    url: str = EMBEDDING_URL,
) -> list[list[float]]:
    """调用远程 embedding 服务，将文本转换为向量。

    Args:
        texts: 单条文本或文本列表。
        model: 向量化模型名称。
        url: embedding 服务地址。

    Returns:
        与输入顺序一致的向量列表，每个向量为 float 列表。
    """
    if isinstance(texts, str):
        texts = [texts]

    headers = {"Content-Type": "application/json; charset=UTF-8"}
    data = {"input": texts, "model": model}

    response = requests.post(url=url, json=data, headers=headers, timeout=60)
    response.raise_for_status()

    results = response.json()["data"]
    embeddings = [item["embedding"] for item in results]
    return embeddings


if __name__ == "__main__":
    vecs = get_embeddings(["你好", "世界", "测试"])
    for i, v in enumerate(vecs):
        print(f"[{i}] dim={len(v)}, first 5: {v[:5]}")
