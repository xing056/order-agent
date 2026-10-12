"""工具层：模型能"看到"什么工具、真实执行如何路由，都在这一个文件。

【这个文件是理解智能体的钥匙 —— 三个部件各司其职】

  TOOLS（JSON Schema 列表）
      → 给模型看的"工具说明书"。模型只认识这种格式：
        名字、用途描述、参数类型。描述写得好不好直接影响模型
        会不会调、调得对不对 —— 这就是"提示词工程"的工程部分。

  CONFIRM_REQUIRED（危险操作名单）
      → 白名单的反面：写操作（取消订单、改地址）不直接执行，
        而是挂起循环等用户点确认。宁可多一次确认，
        不可错改一次数据。

  TOOL_DISPATCH（名字 → 真实函数的映射表）
      → 执行路由。模型只能"提议"调用某个名字，真正执行的是
        这张表指向的 Python 函数 —— 执行权永远在代码手里，
        模型从头到尾碰不到函数本体。这就是工具调用的安全边界。

【加一个新工具的三步操作】
  1. db.py / kb.py 里写真实执行函数（干活的）
  2. 在 TOOLS 里加一份 JSON Schema（让模型知道它）
  3. 在 TOOL_DISPATCH 里注册（让代码找得到它）
  写操作的话再进 CONFIRM_REQUIRED。全程不改 core.py 的循环 ——
  这就是"工具可插拔"的含义。
"""

import json

from agent import db, kb

# ---------- 1. 工具定义（模型看到的"说明书"） ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "查询某个用户的所有订单，返回订单号、商品、金额、状态、收货地址、下单时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "用户ID，例如 u001"}
                },
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "查询单个订单的实时状态（含下单人、商品、金额、当前状态、收货地址）",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，例如 o1001"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "取消一个订单（危险操作，系统会先让用户确认）。只有「待发货」状态的订单才能取消",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要取消的订单号，例如 o1003"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_address",
            "description": "修改订单的收货地址（危险操作，系统会先让用户确认）。只有「待发货」状态的订单才能修改地址",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要修改的订单号，例如 o1003"},
                    "new_address": {"type": "string", "description": "新的完整收货地址，例如：北京市朝阳区望京SOHO T1座"},
                },
                "required": ["order_id", "new_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索公司售后政策知识库（RAG）。用户问退货/运费/发票/保修/配送时效/积分/会员"
                "等政策类问题时必须调用它，依据检索到的原文回答，不要凭空编造政策"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题，例如：退货政策是什么"},
                },
                "required": ["query"],
            },
        },
    },
]

# ---------- 2. 危险操作名单（调用前必须经过用户确认） ----------
# 判断标准很简单：只读（查询、检索）直接执行；写库（改状态、改数据）必须确认。
# 延伸：生产环境这里还会叠加"权限校验"——校验发起人账号有没有权限
# 让智能体做这件事（校验的是人，不是智能体）。

CONFIRM_REQUIRED = {"cancel_order", "update_address"}

# ---------- 3. 分发表（名字 -> 真实函数） ----------
# 模型返回的只是字符串名字（如 "cancel_order"），由这张表翻译成
# 真正的 Python 函数调用。表里没有的名字一律拒绝 —— 这同时
# 就是"工具白名单"：模型想调用一个不存在的工具？没门。

TOOL_DISPATCH = {
    "get_user_orders": db.get_user_orders,
    "get_order_status": db.get_order_status,
    "cancel_order": db.cancel_order,
    "update_address": db.update_address,
    "search_knowledge_base": kb.search_knowledge_base,
}


def execute_tool(name: str, args: dict) -> dict:
    """执行工具并返回结果（统一入口，方便加日志/审计/耗时统计）。

    生产环境这个函数会变成：参数校验 → 权限检查 → 执行 →
    结果脱敏 → 写审计日志 —— 所有横切关注点都挂在这个收口上。
    """
    if name not in TOOL_DISPATCH:
        return {"error": f"未知工具: {name}"}
    return TOOL_DISPATCH[name](**args)
