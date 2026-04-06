"""
LLM API 客户端模块
提供 qwen（阿里云 DashScope）和 chat（内部网关）两个调用接口。
"""

import os
import time
import json
import functools
import traceback
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─── 重试装饰器 ─────────────────────────────────────────────────────────────

def retry(max_retries: int = 5, sleep_seconds: float = 15.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 2):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt <= max_retries:
                        print(f"[retry] {func.__name__} 第{attempt}次失败: "
                              f"{traceback.format_exc()}，{sleep_seconds}秒后重试...")
                        time.sleep(sleep_seconds)
                    else:
                        print(f"[retry] {func.__name__} 已达最大重试次数({max_retries})，放弃。")
            raise last_exc
        return wrapper
    return decorator


# ─── qwen：阿里云 DashScope 兼容接口 ────────────────────────────────────────

@retry(max_retries=3, sleep_seconds=5.0)
def qwen(prompt: str, web_search: bool = False, enable_thinking: bool = False) -> dict:
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY", "sk-fb07e345e2b04562ad5acf2d4bfee8fa"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=os.getenv("INFER_LLM_NAME", "qwen3-max"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        extra_body={
            "enable_search": web_search,
            "enable_thinking": enable_thinking,
        },
    )
    msg = completion.choices[0].message
    return {
        "content": msg.content,
        "reasoning_content": getattr(msg, "reasoning_content", "") or "",
    }


# ─── chat：内部 MLP 网关接口 ────────────────────────────────────────────────

#"sk-5a5c8fea7cc14d779a201d8ab0be8f91"

@retry(max_retries=3, sleep_seconds=5.0)
def chat(messages: str, vendor: str = "aliyun", model: str = "deepseek-v3.2") -> str:
    if vendor == "servyou":
        URL = f"http://10.199.0.7:5000/api/llm/{vendor}/v1/chat/completions"
        app_id = "sk-d75b519b704d4d348245efe435f08ff3"
    else:
        URL = f"http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/{vendor}/v1/chat/completions"
        app_id = "sk-0609aa6d08de4413a72e14b3fb8fbab1"
        
    HEADERS = {"Content-Type": "application/json", "Authorization": app_id}
    messages = [{"role": "user", "content": messages}]
    PAYLOAD = {
        "appId": app_id,
        "model": model,
        "messages": messages,
        "stream": False,
        "top_p": 0.7,
        "temperature": 0.5,
        "enable_thinking": True,
        # "enable_search":True
    }
    response = requests.post(URL, data=json.dumps(PAYLOAD), headers=HEADERS).json()
    if "success" in response:
        raise Exception(response["errorContext"])
    return response["choices"][0]["message"]#["content"]


if __name__ == "__main__":
    print(chat("草木灰属于初级农产品嘛？", vendor = "aliyun"))