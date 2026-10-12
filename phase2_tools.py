"""Phase 2: 最小智能体 — 工具调用（Function Calling）+ 行动循环（Agent Loop）。

这是"调模型"和"智能体"的分水岭。本文件只做一件新事：

    让模型"决定"要调用哪个函数，代码执行后把真实结果回灌，再由模型组织回答。

流程（就是智能体的全部秘密）：
    用户提问
      -> 把问题连同工具清单发给模型
      -> 模型说："我要调用 get_user_orders(uid='u001')"
      -> 代码真正执行这个函数（这里用模拟数据，之后换成真实查询）
      -> 把执行结果回灌给模型
      -> 模型拿着结果组织成人话回复
      -> 如果模型还要用别的工具，循环继续；不再调用工具，任务结束

运行：python phase2_tools.py
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY or API_KEY == "sk-your-key-here":
    raise SystemExit("请先在 .env 里填上你的 DeepSeek API Key")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ---------------------------------------------------------------
# 第 1 步：写工具（就是你的"手"）
# 这里用内存里的假数据模拟，真实项目里把函数体换成你的接口/数据库查询即可
# ---------------------------------------------------------------

ORDERS = {
    "u001": [
        {"order_id": "O20260815001", "item": "华为 Mate70 Pro", "amount": 6999, "status": "已发货"},
        {"order_id": "O20260820002", "item": "蓝牙耳机 FreeBuds 6", "amount": 899, "status": "待发货"},
    ],
    "u002": [
        {"order_id": "O20260825001", "item": "智能手表 Watch GT5", "amount": 1488, "status": "已完成"},
    ],
}


def get_user_orders(uid: str) -> str:
    """查询某个用户的所有订单。"""
    orders = ORDERS.get(uid, [])
    return json.dumps(orders, ensure_ascii=False)


def get_order_status(order_id: str) -> str:
    """按订单号查询订单状态。"""
    for orders in ORDERS.values():
        for o in orders:
            if o["order_id"] == order_id:
                return json.dumps(o, ensure_ascii=False)
    return json.dumps({"error": f"未找到订单 {order_id}"}, ensure_ascii=False)


# ---------------------------------------------------------------
# 第 2 步：把工具"介绍"给模型（JSON Schema 格式，模型据此决定调谁、传什么参数）
# ---------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "查询指定用户 ID 的全部订单列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "用户 ID，例如 u001"}
                },
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "按订单号查询单个订单的状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，例如 O20260815001"}
                },
                "required": ["order_id"],
            },
        },
    },
]

# 工具名 -> 实际函数 的映射表（模型只给名字和参数，执行权始终在你的代码手里）
TOOL_DISPATCH = {
    "get_user_orders": get_user_orders,
    "get_order_status": get_order_status,
}

SYSTEM_PROMPT = (
    "你是订单查询助手。用户询问订单信息时，必须调用工具获取真实数据，"
    "然后基于工具返回的数据组织回答。工具查不到就如实告知，不要编造。"
)


# ---------------------------------------------------------------
# 第 3 步：智能体循环（核心！约 20 行）
# ---------------------------------------------------------------

def run_agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for turn in range(10):  # 保险丝：最多 10 轮，防止死循环烧钱
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 情况 A：模型要求调用工具
        if msg.tool_calls:
            # ① assistant 这条消息必须原样回传（它带着模型的决定）
            messages.append(msg)
            # ② 逐个执行工具，把结果作为 role="tool" 的消息回灌
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"  [智能体] 调用工具 {fn_name}({args})")
                result = TOOL_DISPATCH[fn_name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,  # 必须带上，模型靠它对应上调用
                    "content": result,
                })
            continue  # 回到循环顶部，让模型基于工具结果继续

        # 情况 B：模型认为任务完成，直接输出最终回答
        return msg.content

    return "（达到最大轮数仍未完成，已安全终止）"


# ---------------------------------------------------------------
# 入口：命令行试几个问题
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("订单查询智能体已就绪。试试问：")
    print("  1. 帮我查一下 u001 的订单")
    print("  2. u002 买了什么？")
    print("  3. O20260815001 这个订单发货了吗？")
    print("  4. 我总共消费了多少钱？（这需要模型自己决定查两次再算）")
    print("=" * 50)
    while True:
        q = input("\n你: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer = run_agent(q)
        print(f"AI: {answer}")
