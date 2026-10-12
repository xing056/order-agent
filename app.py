"""订单智能助手 —— Streamlit 界面。

**两个实现共用同一套界面**（左侧边栏随时切换）：
    手写版       agent/core.py               —— while 循环 + 挂起确认，零框架
    LangChain 版 agent/langchain_session.py  —— create_agent 一行替代循环

两个会话类对外接口完全一致（send / confirm / pending_action），所以下面的
渲染代码不需要知道当前是哪一版在跑。**变的只有编排层** —— 这本身就是
"框架 = 循环的封装"最直观的证据。

运行方式：
    pip install -r requirements.txt
    cp .env.example .env      # 填入你的 API Key
    streamlit run app.py

    演示前跑 python smoke_test.py 一键回初始状态。
    框架版自检：python smoke_test_langchain.py
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from agent import db
from agent.core import AgentSession

# ---------- 可选依赖：框架版需要 langchain ----------
# 设计原则：可选依赖缺失要**优雅降级**，不能因为一个对照版本把主功能拖死。
# 没装 langchain 时页面照常打开、照常跑手写版，只在侧边栏提示怎么启用。
try:
    from agent.langchain_session import LangChainSession

    LANGCHAIN_READY = True
    LANGCHAIN_ERR = ""
except Exception as _e:  # noqa: BLE001 —— 缺依赖/版本不对都只降级，不崩页面
    LangChainSession = None
    LANGCHAIN_READY = False
    LANGCHAIN_ERR = f"{type(_e).__name__}: {_e}"

load_dotenv()

st.set_page_config(page_title="订单智能助手", page_icon=":material/support_agent:", layout="wide")


# ---------- 配置读取 ----------
def _secret(name: str, default: str = "") -> str:
    """取配置：先看环境变量 / .env，再回落到 Streamlit 平台的 secrets。

    为什么要两层：本地跑读 .env 最顺手；一旦部署到 Streamlit Community Cloud
    这类平台，Key 是填在平台的 Secrets 面板里的，本地 .env 根本不会跟着走。
    代码两处都认，同一份源码本地和线上都能跑，不用改。
    """
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


# ---------- 可选：公开演示的"密码门" ----------
# 不设 DEMO_PASSWORD 时（本地演示的默认情况）这整段直接跳过，行为跟以前一模一样。
# 只有在把应用挂到**公开网址**上时才设它 —— 原因见下方注释。
#
# 为什么必须有这道门：应用一旦公开，任何拿到链接的人（包括扫描全网的小爬虫）
# 都能直接跟你的智能体对话，而**每一次对话烧的都是你自己的模型额度**。
# Streamlit Community Cloud 的免费公开应用没有内置鉴权，这道门是自己加的锁。
DEMO_PASSWORD = _secret("DEMO_PASSWORD")
if DEMO_PASSWORD and st.session_state.get("_authed") is not True:
    st.title("订单智能助手 · 受限演示")
    st.caption(
        "这是一个受保护的演示环境（每次对话都会消耗真实的模型额度），"
        "所以只对受邀者开放。请输入访问口令。"
    )
    _pw = st.text_input("访问口令", type="password")
    if st.button("进入演示", type="primary"):
        if _pw == DEMO_PASSWORD:
            st.session_state._authed = True
            st.rerun()
        else:
            st.error("口令不正确。")
    st.stop()

# ---------- 配置 ----------
API_KEY = _secret("LLM_API_KEY")
BASE_URL = _secret("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = _secret("LLM_MODEL", "qwen-plus")

IMPL_HAND = "手写版（主推荐）"
IMPL_LANGCHAIN = "LangChain 版"

if not API_KEY or API_KEY.startswith("sk-在这里填"):
    st.error("未检测到 API Key：请把 agent-lab/.env.example 复制为 .env，填入你的 Key 后重启应用。")
    st.stop()

db.init_db()  # 首次运行自动建表 + 灌入演示数据


# ---------- 会话工厂 ----------
def make_session(impl: str):
    """按所选实现创建会话。两个类接口一致，UI 无差别对待。"""
    if impl == IMPL_LANGCHAIN and LANGCHAIN_READY:
        return LangChainSession(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    return AgentSession(api_key=API_KEY, base_url=BASE_URL, model=MODEL)


# ---------- 事件渲染（把智能体的"思考过程"展示出来） ----------
def render_event(ev: dict) -> None:
    t = ev["type"]
    if t == "user":
        with st.chat_message("user"):
            st.write(ev["content"])
    elif t == "tool_call":
        with st.chat_message("assistant"):
            st.markdown("**调用工具**")
            st.code(
                json.dumps({"工具": ev["name"], "参数": ev["args"]}, ensure_ascii=False, indent=2),
                language="json",
            )
    elif t == "tool_result":
        with st.chat_message("assistant"):
            st.markdown("**工具返回**")
            st.code(
                json.dumps(ev["result"], ensure_ascii=False, indent=2, default=str)[:1500],
                language="json",
            )
    elif t == "confirm":
        with st.chat_message("assistant"):
            st.markdown(f"**需要你确认**：{ev['name']}({json.dumps(ev['args'], ensure_ascii=False)})")
    elif t == "answer":
        with st.chat_message("assistant"):
            st.write(ev["content"])


def run_agent(fn) -> list:
    """执行智能体调用的统一入口。

    包一层 spinner 的原因：模型调用是**同步阻塞**的，一次完整流程要等
    模型思考 + 工具执行 + 再思考（几秒到十几秒）。这期间页面毫无变化，
    用户会以为程序卡死了 —— 转圈提示至少说明"它在干活"。

    顺带统一兜底异常：Key 错了、网络断了都给一句人话提示，而不是白屏。
    """
    try:
        with st.spinner("智能体正在思考并调用工具…"):
            return fn()
    except Exception as e:
        return [
            {
                "type": "answer",
                "content": f"调用模型出错：{e}\n\n请检查 .env 里的 API Key / Base URL 是否正确，以及网络是否可达。",
            }
        ]


def handle_prompt(prompt: str) -> None:
    """先把用户消息入列并立刻重跑一次，让刚发出去的文字马上显示出来。

    关键点：真正跑模型放在下一帧（见下方 pending_prompt 分支）。
    如果这里直接调 agent.send()，用户会先盯着空白页等十几秒，
    连自己刚打的字都看不到 —— 演示时观感极差。
    """
    st.session_state.events.append({"type": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.rerun()


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("订单智能助手")
    st.caption(f"模型：{MODEL}")
    st.caption(f"服务：{BASE_URL.split('//')[1].split('/')[0]}")
    st.divider()

    st.markdown("### 实现方式（演示对照）")
    _options = [IMPL_HAND] + ([IMPL_LANGCHAIN] if LANGCHAIN_READY else [])
    _captions = ["agent/core.py：while 循环，危险操作真正挂起等确认"] + (
        ["create_agent：一行吃掉循环，确认改用「预演→确认→重放」"] if LANGCHAIN_READY else []
    )
    impl = st.radio("实现方式", _options, captions=_captions, label_visibility="collapsed")
    if not LANGCHAIN_READY:
        st.caption(
            "⚠️ 框架版未启用：当前 Python 环境里没有 langchain（或其他可选依赖缺失）。"
            "装依赖后刷新即可：`pip install -r requirements.txt`。手写版不受影响。"
        )

    # —— 会话与对话记录按实现方式分开存：来回切换时，两版各自的上下文都不会串 ——
    if st.session_state.get("impl") != impl:
        st.session_state.impl = impl
        st.session_state.pop("pending_prompt", None)  # 丢掉上一版没跑完的提问

    session_key = f"session_{impl}"
    if session_key not in st.session_state:
        st.session_state[session_key] = make_session(impl)
    agent = st.session_state[session_key]

    # 让下面既有的渲染代码继续用 st.session_state.events（引用同一份列表，
    # 原地 append/+= 都会同步写回 events_<实现> 这个键）
    events_key = f"events_{impl}"
    if events_key not in st.session_state:
        st.session_state[events_key] = []
    st.session_state.events = st.session_state[events_key]

    st.divider()
    st.markdown("### 演示问题（点击直接发送）")
    demo_q = st.selectbox(
        "选择一个演示问题",
        [
            "查询用户 u001 的订单",
            "订单 o1002 现在什么状态",
            "帮我算一下 u001 总共消费了多少钱",
            "退货政策是什么？运费谁出？",
            "帮我取消订单 o1003",
            "把订单 o1003 的收货地址改成 北京市朝阳区望京SOHO T1座",
        ],
    )
    if st.button("发送该问题", width="stretch"):
        handle_prompt(demo_q)
    st.divider()
    st.markdown("### 数据库（演示用）")
    table = st.selectbox("查看表", ["orders", "users"])
    conn = db.get_conn()
    try:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    finally:
        conn.close()
    st.dataframe(rows, width="stretch", hide_index=True)

# ---------- 主区 ----------
st.title("订单智能助手")
st.caption(
    "智能体演示：模型提议调用工具 → 你的代码执行 → 结果回灌 → 直到任务完成。"
    "试试让它「查订单 / 算总消费 / 问售后政策 / 取消订单 / 改收货地址」，左侧边栏可以看到它真的改了数据库。"
)

if impl == IMPL_LANGCHAIN:
    st.info(
        "当前实现：**LangChain 1.x `create_agent`**（循环交给框架）。对照点两条 ——\n\n"
        "1. 手写版 `agent/core.py` 里那个 `while` 循环，这里被 `create_agent` 一行替代（底层是 LangGraph 状态机）；\n"
        "2. 确认机制不同：**手写版能真正挂起循环**，框架版 UI 里没法阻塞，改用「**预演 → 你点确认 → 重放**」。"
    )
else:
    st.success(
        "当前实现：**手写版**（`agent/core.py` 的 while 循环 + 危险操作挂起确认），不依赖任何 Agent 框架。"
    )

for ev in st.session_state.events:
    render_event(ev)

# ---------- 待处理的用户提问 ----------
# 放在历史渲染**之后**执行：这样用户刚发出的那句话已经显示在页面上了，
# 再开始等模型。否则会先空白十几秒，连自己打的字都看不到。
if st.session_state.get("pending_prompt"):
    _p = st.session_state.pop("pending_prompt")
    st.session_state.events += run_agent(lambda: agent.send(_p))
    st.rerun()

# ---------- 危险操作确认区 ----------
# pending_action 里可能是 cancel_order（取消）或 update_address（改地址），
# 按工具类型生成对应的确认文案 —— 但确认流程本身是同一套。
# 两个实现类的 pending_action / confirm() 接口一致，所以这里不用分支。
pa = agent.pending_action
if pa:
    args = pa["args"]
    if pa["name"] == "update_address":
        desc = f"把订单 {args.get('order_id')} 的收货地址改为「{args.get('new_address')}」"
    else:
        desc = f"取消订单 {args.get('order_id')}"
    st.warning(f"需要你确认：{desc}。此操作会真实修改数据库，是否执行？")
    c1, c2 = st.columns(2)
    if c1.button("确认执行", type="primary", width="stretch"):
        st.session_state.events += run_agent(lambda: agent.confirm(True))
        st.rerun()
    if c2.button("拒绝", width="stretch"):
        st.session_state.events += run_agent(lambda: agent.confirm(False))
        st.rerun()

prompt = st.chat_input("例如：查询 u001 的订单 / 退货政策是什么 / 帮我取消订单 o1003", disabled=pa is not None)
if prompt:
    handle_prompt(prompt)
