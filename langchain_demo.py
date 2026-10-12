"""LangChain 版订单智能助手 —— 与手写版（agent/core.py + app.py）功能完全相同的第二种实现。

对照阅读指南：
  手写版 agent/core.py 里的 150 行 while 循环，这里被 LangChain 的
  create_agent 一行调用吃掉了 —— 这就是"框架 = 循环的封装"最直观的证据。

  | 手写版（agent/core.py）      | LangChain 1.x 版（本文件）               |
  |------------------------------|------------------------------------------|
  | TOOLS（JSON Schema 字典）    | @tool 装饰器自动生成 Schema              |
  | while True 循环              | create_agent 内部封装（LangGraph 驱动）  |
  | role="tool" 结果回灌         | 框架自动完成                             |
  | 10 轮保险丝                  | recursion_limit（运行时配置）            |
  | 确认机制（挂起/恢复）         | 确认做在工具函数内部（见下方说明）        |

版本说明（LangChain 的版本变迁）：
  LangChain 1.x（2025 起）淘汰了老教程里的 AgentExecutor，统一到 create_agent，
  底层由 LangGraph 状态机驱动。网上大量"LangChain 教程"教的还是 0.x 旧 API ——
  这也是很容易踩的坑：背的 API 和 pip 装到的版本对不上。

关于确认机制的差异：
  手写版能"挂起循环等用户点按钮"，因为循环在你手里，想停就停。
  LangChain 1.x 提供了 interrupt_before=["tools"] 参数可以中断在工具执行前，
  但恢复流程较绕；命令行 demo 里把确认做进工具函数内部（执行前 input 确认），
  效果等价。—— 这就是"框架给你便利，也拿走一部分控制权"的亲身体会。

运行：
    cd agent-lab
    python langchain_demo.py
    （先激活 .venv 再运行；Windows 下的完整环境准备可参考 run.bat）
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agent import db, kb

load_dotenv()

# ============================================================
# 1. 工具定义：@tool 装饰器
#    手写版要自己写 JSON Schema 字典；这里框架根据函数签名+docstring
#    自动生成 —— 但"说明书"的本质没变：模型看的还是 name/description/参数。
# ============================================================


@tool
def get_user_orders(uid: str) -> dict:
    """查询某个用户的所有订单，返回订单号、商品、金额、状态、下单时间。

    Args:
        uid: 用户ID，例如 u001
    """
    return db.get_user_orders(uid)


@tool
def get_order_status(order_id: str) -> dict:
    """查询单个订单的实时状态（含下单人、商品、金额、当前状态）。

    Args:
        order_id: 订单号，例如 o1001
    """
    return db.get_order_status(order_id)


@tool
def cancel_order(order_id: str) -> dict:
    """取消一个订单（危险操作，执行前会要求用户确认）。
    只有「待发货」状态的订单才能取消。

    Args:
        order_id: 要取消的订单号，例如 o1003
    """
    # 框架版确认机制：做在工具函数内部（手写版是挂起整个循环）
    ans = input(f"\n⚠️  模型提议取消订单 {order_id}，确认执行吗？(y/n): ").strip().lower()
    if ans != "y":
        return {"ok": False, "reason": "用户拒绝了本次操作"}
    return db.cancel_order(order_id)


@tool
def update_address(order_id: str, new_address: str) -> dict:
    """修改订单的收货地址（危险操作，执行前会要求用户确认）。
    只有「待发货」状态的订单才能修改地址。

    Args:
        order_id: 要修改的订单号，例如 o1003
        new_address: 新的完整收货地址
    """
    ans = input(f"\n⚠️  模型提议修改订单 {order_id} 的收货地址为「{new_address}」，确认执行吗？(y/n): ").strip().lower()
    if ans != "y":
        return {"ok": False, "reason": "用户拒绝了本次操作"}
    return db.update_address(order_id, new_address)


@tool
def search_knowledge_base(query: str) -> dict:
    """检索公司售后政策知识库（RAG）。退货/运费/发票/保修/配送/积分等
    政策类问题用这个工具，依据检索到的原文回答。

    Args:
        query: 要检索的问题，例如：退货政策是什么
    """
    # 注意：RAG 检索是只读操作，不需要确认；且检索逻辑与手写版
    # 完全复用同一个 kb.search_knowledge_base —— 双实现共用数据层与
    # 知识库层的又一证据，变的只有编排层。
    return kb.search_knowledge_base(query)


TOOLS_LC = [get_user_orders, get_order_status, cancel_order, update_address, search_knowledge_base]

# ============================================================
# 2. 大模型接入：ChatOpenAI 指向阿里百炼（兼容模式）
#    手写版你直接用 OpenAI SDK；这里多一层 LangChain 的模型抽象，
#    好处是以后换模型（Anthropic/本地模型）只改这一行。
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen-plus"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
)

# ============================================================
# 3. 组装智能体：create_agent
#    ★ 这一行就是手写版 150 行循环的全部替代品 ★
#    system_prompt 参数 = 手写版的 SYSTEM_PROMPT；
#    底层是一个 LangGraph 状态机：模型节点 ↔ 工具节点来回跑，直到不再调工具。
# ============================================================

SYSTEM_PROMPT = (
    "你是\"订单智能助手\"，一个能查数据、能办事、懂售后政策的客服助手。\n"
    "行为准则：\n"
    "1. 用户问订单相关的问题时，必须调用工具获取真实数据，绝不编造。\n"
    "2. 查询订单先用 get_user_orders 拿到订单号，再按需用 get_order_status 看详情。\n"
    "3. 用户要求\"取消订单\"时，调用 cancel_order 工具。\n"
    "4. 用户要求\"修改收货地址\"时，调用 update_address 工具。\n"
    "5. 用户问退货、运费、发票、保修等售后政策时，必须先调用 search_knowledge_base，"
    "依据检索原文回答，检索不到就说不知道。\n"
    "6. 用简洁友好的中文回答，金额带货币符号 ¥。\n"
    "7. 工具返回空结果时，如实说\"没有查到\"，不要猜测。"
)

agent = create_agent(
    model=llm,
    tools=TOOLS_LC,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 4. 可观测性：CallbackHandler —— 对应手写版的事件流
#    手写版你在循环里手动 append 事件；框架版用回调钩子"旁听"执行过程。
#    生产环境接 Langfuse/LangSmith 做追踪，用的就是同一套回调机制。
# ============================================================


class ToolTraceHandler(BaseCallbackHandler):
    """把每次工具调用打印出来，看清框架循环的每一步。"""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?") if serialized else "?"
        print(f"  🔧 [框架循环] 模型提议调用工具: {name}({input_str})")

    def on_tool_end(self, output, **kwargs):
        text = str(output).replace("\n", " ")
        print(f"  ✅ [框架循环] 工具返回: {text[:120]}{'...' if len(text) > 120 else ''}")


# ============================================================
# 5. 命令行交互（与手写版 demo 同样的体验）
# ============================================================


def main() -> None:
    print("=" * 60)
    print("LangChain 版订单智能助手（与手写版功能相同，循环层换成了框架）")
    print("=" * 60)
    print("试试：查询 u001 的订单 / u001 总共消费多少钱 / 退货政策是什么 / 帮我取消订单 o1003")
    print("输入 q 退出\n")

    history = []  # 多轮记忆由我们自己维护（框架不自动记）
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in {"q", "quit", "exit", "退出"}:
            break
        if not user_input:
            continue
        try:
            result = agent.invoke(
                {"messages": history + [("user", user_input)]},
                config={
                    "callbacks": [ToolTraceHandler()],
                    "recursion_limit": 20,  # 手写版"10 轮保险丝"的框架等价物
                },
            )
            # 结果是一个 messages 列表，最后一条就是最终回答
            last = result["messages"][-1]
            answer = getattr(last, "content", str(last))
            print(f"\nAI: {answer}\n")
            history.append(("user", user_input))
            history.append(("assistant", answer))
        except Exception as e:  # noqa: BLE001
            print(f"\nAI: 调用出错：{e}\n请检查 .env 的 Key / Base URL / 模型名。\n")


if __name__ == "__main__":
    main()
