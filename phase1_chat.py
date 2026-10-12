"""Phase 1: 命令行聊天 — 打通 DeepSeek API。

目的：走通一次真实的 LLM API 调用，看清三个核心概念：
1. system / user / assistant 三种消息角色
2. messages 列表是"对话记忆"，多轮对话 = 不断往列表里追加
3. 每次调用都消耗 token（成本），打印出来便于建立成本感

运行：python phase1_chat.py
退出：输入 exit
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY or API_KEY == "sk-your-key-here":
    raise SystemExit("请先在 .env 里填上你的 DeepSeek API Key（复制 .env.example 并编辑）")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# system 角色：定义模型扮演什么角色、按什么规则行事
SYSTEM_PROMPT = (
    "你是一个乐于助人的智能助手。回答要简洁、准确、友好。"
    "当你不确定时，直接说不知道，不要编造。"
)

# messages 是整个对话的"记忆"，每个角色一条记录
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("智能体已就绪（输入 exit 退出）")
print("=" * 40)

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("再见！")
        break
    if not user_input:
        continue

    # 1. 把用户的话追加进对话记忆
    messages.append({"role": "user", "content": user_input})

    # 2. 请求模型生成回复（一次 API 调用）
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
    )

    # 3. 取出回复，并追加进对话记忆（这样模型能记住上下文）
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}")
    # 打印 token 消耗，建立成本概念（deepseek-chat 约 ¥2/百万输入 token，约 ¥8/百万输出 token）
    usage = resp.usage
    print(f"[本次消耗: 输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens} token]")
