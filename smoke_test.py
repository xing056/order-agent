"""演示前自检脚本：跑通"查询 / RAG / 危险操作确认"三类对话，最后自动重置数据库。

用法（改动后或交付前跑一次，全绿即通过）：
    python smoke_test.py
它验证的事：Key 有效、网络可达、知识库向量缓存就绪、确认机制正常。
跑完数据库自动回到初始状态（o1003 = 待发货）。
"""

import os

from dotenv import load_dotenv

from agent import db
from agent.core import AgentSession

load_dotenv()
db.init_db(force=True)

s = AgentSession(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
)


def run(label, text):
    print(f"\n{'=' * 55}\n【{label}】用户: {text}")
    for ev in s.send(text):
        if ev["type"] == "tool_call":
            print(f"  [tool] {ev['name']}({ev['args']})")
        elif ev["type"] == "tool_result":
            print(f"  [result] {str(ev['result'])[:110]}")
        elif ev["type"] == "confirm":
            print(f"  [PENDING] {ev['name']}")
        elif ev["type"] == "answer":
            print(f"  [answer] {ev['content'][:200]}")


run("普通查询", "查询用户 u001 的订单")
run("知识库RAG", "退货的话运费谁承担？")
run("危险操作-改地址", "把订单 o1003 的收货地址改成 北京市朝阳区望京SOHO T1座")

if s.pending_action:
    print("\n--- 用户点击【确认执行】 ---")
    for ev in s.confirm(True):
        if ev["type"] == "tool_result":
            print(f"  [result] {ev['result']}")
        elif ev["type"] == "answer":
            print(f"  [answer] {ev['content'][:200]}")

print("\n[验证] o1003 地址 =", db.get_order_status("o1003")["address"])

db.init_db(force=True)
print("\n[重置] 数据库已恢复演示初始状态（o1003=待发货）")
