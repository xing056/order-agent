"""厂商能力探测：选型前先问清楚"这家到底支不支持我要的东西"。

做这个项目的第一步不是写代码，是确认技术路线走不走得通：
对话模型要**支持 Function Calling**，RAG 要**有 embedding 接口**——
缺哪一样这条路线就得换。下面四个探测就是当时干的活，留作复用。

用法（Key 从环境变量 / .env 读，别写死在代码里）：
    python probe_provider.py

    # 或在 .env 里加临时的一组：
    # PROBE_CHAT_KEY=...   PROBE_CHAT_URL=https://api.deepseek.com/v1   PROBE_CHAT_MODEL=deepseek-chat
    # PROBE_EMBED_KEY=...  PROBE_EMBED_URL=https://open.bigmodel.cn/api/paas/v4  PROBE_EMBED_MODEL=embedding-3
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _client(key: str, url: str):
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=url)


def probe_chat(base_url: str, key: str, model: str) -> None:
    """1. 对话模型能不能用（顺带验证网络与 Key 是否有效）。"""
    print(f"=== 对话模型 {model} ===")
    if not key:
        print("  跳过：没配 Key")
        return
    try:
        c = _client(key, base_url)
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：可用"}],
            max_tokens=10,
        )
        print("  可用，返回:", r.choices[0].message.content)
    except Exception as e:
        print("  不可用:", type(e).__name__, str(e)[:200])


def probe_tool_calling(base_url: str, key: str, model: str) -> None:
    """2. 是否支持 Function Calling —— **这是智能体路线的生死线**，不支持就直接换厂商。"""
    print(f"=== Function Calling（{model}）===")
    if not key:
        print("  跳过：没配 Key")
        return
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "查询订单状态",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    try:
        c = _client(key, base_url)
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "帮我查一下订单 o1002 的状态"}],
            tools=tools,
        )
        tc = r.choices[0].message.tool_calls
        if tc:
            print("  支持！返回 tool_calls:", tc[0].function.name, tc[0].function.arguments)
        else:
            print("  没报错但没返回 tool_calls，需人工确认")
    except Exception as e:
        print("  不支持:", type(e).__name__, str(e)[:200])


def probe_embedding(base_url: str, key: str, model: str) -> None:
    """3. 向量接口是否存在（对话厂商多数没有，所以 RAG 常常要另配一家）。"""
    print(f"=== 向量模型 {model} ===")
    if not key:
        print("  跳过：没配 Key")
        return
    try:
        c = _client(key, base_url)
        r = c.embeddings.create(model=model, input=["测试文本"])
        print(f"  可用，维度 {len(r.data[0].embedding)}")
    except Exception as e:
        print("  不可用:", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    chat_url = os.getenv("PROBE_CHAT_URL", "")
    chat_key = os.getenv("PROBE_CHAT_KEY", "")
    chat_model = os.getenv("PROBE_CHAT_MODEL", "")

    embed_url = os.getenv("PROBE_EMBED_URL", os.getenv("LLM_EMBED_BASE_URL", ""))
    embed_key = os.getenv("PROBE_EMBED_KEY", os.getenv("LLM_EMBED_API_KEY", ""))
    embed_model = os.getenv("PROBE_EMBED_MODEL", os.getenv("LLM_EMBED_MODEL", "embedding-3"))

    probe_chat(chat_url, chat_key, chat_model)
    print()
    probe_tool_calling(chat_url, chat_key, chat_model)
    print()
    probe_embedding(embed_url, embed_key, embed_model)
