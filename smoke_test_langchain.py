"""LangChain 版自检脚本 —— 与 smoke_test.py 对称，验证框架版在页面里能跑通。

用法（演示前跑一次）：
    python smoke_test_langchain.py

它验证的事：Key 有效、create_agent 循环能跑、RAG 复用同一套知识库、
**危险操作的「预演 → 确认 → 重放」链路正常**，跑完自动重置数据库。
"""

import os

from dotenv import load_dotenv

from agent import db
from agent.langchain_session import LangChainSession

load_dotenv()
db.init_db(force=True)

s = LangChainSession(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
)


def run(label, text):
    print(f"\n{'=' * 55}\n【{label}】用户: {text}")
    out = s.send(text)
    for ev in out:
        if ev["type"] == "tool_call":
            print(f"  [tool] {ev['name']}({ev['args']})")
        elif ev["type"] == "tool_result":
            print(f"  [result] {str(ev['result'])[:110]}")
        elif ev["type"] == "confirm":
            print(f"  [PENDING] 待确认 -> {ev['name']}({ev['args']})")
        elif ev["type"] == "answer":
            print(f"  [answer] {ev['content'][:200]}")
    return out


def show(events):
    for ev in events:
        if ev["type"] == "tool_result":
            print(f"  [result] {ev['result']}")
        elif ev["type"] == "answer":
            print(f"  [answer] {ev['content'][:200]}")


print("=" * 60)
print("LangChain 版自检（create_agent + 预演/确认/重放）")
print("=" * 60)

run("普通查询", "查询用户 u001 的订单")
run("知识库RAG", "退货的话运费谁承担？")

print("\n--- 危险操作：先拒绝（数据库不应变化）---")
run("危险操作-改地址", "把订单 o1003 的收货地址改成 北京市朝阳区望京SOHO T1座")
before = db.get_order_status("o1003")["address"]
print(f"  拒绝前地址 = {before}")
if s.pending_action:
    print("--- 用户点击【拒绝】---")
    show(s.confirm(False))
after_reject = db.get_order_status("o1003")["address"]
print(f"  [验证] 拒绝后地址 = {after_reject}  ->  {'OK 未变化' if after_reject == before else '!! 被改了'}")

print("\n--- 危险操作：再确认（数据库应变化）---")
run("危险操作-改地址", "把订单 o1003 的收货地址改成 北京市朝阳区望京SOHO T1座")
if s.pending_action:
    print("--- 用户点击【确认执行】---")
    show(s.confirm(True))
after_ok = db.get_order_status("o1003")["address"]
print(f"  [验证] 确认后地址 = {after_ok}  ->  {'OK 已修改' if after_ok != before else '!! 没生效'}")

db.init_db(force=True)
print("\n[重置] 数据库已恢复演示初始状态（o1003=待发货）")
