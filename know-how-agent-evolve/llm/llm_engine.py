import os
import time
import functools
from openai import OpenAI
from dotenv import load_dotenv
import traceback

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.ENV'))


def retry(max_retries: int = 5, sleep_seconds: float = 15.0):
    """
    重试装饰器：函数报错时等待 sleep_seconds 秒后重试，最多重试 max_retries 次。
    超过重试上限后，抛出最后一次的异常。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 2):  # 1次正常 + max_retries次重试
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt <= max_retries:
                        print(f"[retry] {func.__name__} 第{attempt}次失败: {traceback.format_exc()}，{sleep_seconds}秒后重试...")
                        time.sleep(sleep_seconds)
                    else:
                        print(f"[retry] {func.__name__} 已达最大重试次数({max_retries})，放弃。")
            raise last_exc
        return wrapper
    return decorator


@retry(max_retries=3, sleep_seconds=5.0)
def qwen(prompt, web_search: bool = False, enable_thinking: bool = False):
    client = OpenAI(
        # 从环境变量读取，或直接填写 api_key="sk-xxx"
        api_key=os.getenv("QWEN_API_KEY"),
        # 北京地域（默认）
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

    )

    completion = client.chat.completions.create(
        model=os.getenv("INFER_LLM_NAME", "qwen3-max"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        # stream=True,  # 如需流式输出，取消注释
        extra_body={
            "enable_search": web_search,  # 开启联网搜索
            "enable_thinking": enable_thinking,  # 开关思考功能
            # "thinking_budget": 81920   # 可选：限制思考过程最大Token数（默认32768）
            # "search_options": {
            #     "forced_search": True,        # 强制开启搜索（默认false，由模型判断是否使用）
            #     "search_strategy": "turbo",   # 搜索策略：turbo(默认)/max/agent/agent_max
            #     "enable_source": True,        # 返回搜索结果来源（仅部分模型支持）
            #     "enable_citation": True,      # 开启角标标注[1]样式（需enable_source为true）
            #     "citation_format": "[ref_<number>]"  # 角标样式：[<number>] 或 [ref_<number>]
            # }
        }

    )
    if enable_thinking:
        res ={
            "content": completion.choices[0].message.content,
            "reasoning_content":completion.choices[0].message.reasoning_content
        }
    else:
        res ={
            "content": completion.choices[0].message.content,
            "reasoning_content":""
        }

    return res

if __name__ == "__main__":
    print(qwen("我应该走路去离我只有50米的洗车店洗车嘛？"))