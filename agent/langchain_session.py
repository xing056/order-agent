"""LangChain 版会话封装 —— 让「框架版」也能跑在 Streamlit 页面上。

【为什么需要这个文件】
  langchain_demo.py 是命令行版：确认机制用 input() 就地阻塞。
  但 Streamlit 里没有 stdin，input() 会把页面卡死。
  所以这里把确认换成「预演 → 用户点确认 → 重放」，并对外暴露与
  手写版 AgentSession **完全相同**的接口：

      .send(text) -> list[event]      .confirm(yes) -> list[event]      .pending_action

  接口一致的意义：app.py 的渲染代码一行都不用改，两版共用同一套 UI，
  变的只有编排层 —— 这本身就是「框架 = 循环的封装」最直观的证据。

【事件流（与 agent/core.py 逐字一致）】
  {"type": "tool_call",   "name":…, "args":…}
  {"type": "tool_result", "name":…, "result":…}
  {"type": "confirm",     "name":…, "args":…}     危险操作，等待前端确认
  {"type": "answer",      "content":…}

【两个踩过的坑（框架版和手写版的关键差异）】

  坑 1｜确认机制：UI 里没法像 CLI 那样 input() 阻塞。
        手写版：循环在自己手里 → 遇到危险工具直接 return，**真正挂起**。
        框架版-命令行：确认做在工具函数内部（input 阻塞）。
        框架版-页面：工具先**只登记不执行**，用户点确认后由会话层执行、
        再把结果作为新消息回灌让模型收尾。
        想要"真正的挂起/恢复"，框架侧要用
        create_agent(interrupt_before=["tools"]) + checkpointer + thread_id 的 resume 流程。
        —— 一句话：**框架给你便利，也拿走一部分控制权。**

  坑 2｜事件从哪来：一开始用 CallbackHandler（on_tool_start/on_tool_end）旁听，
        结果 on_tool_start 给的 input_str 是 **Python repr 字符串**（单引号，
        还偶发流式分片导致的重复调用 / 脏 key），拿它渲染"工具参数"得先做字符串解析，
        既不干净也不稳。改成直接读 **messages** —— framework 返回的
        AIMessage.tool_calls 本来就是结构化 dict，ToolMessage 带 name + content。
        —— 结论：框架已经把结构给你了，就别再去解析字符串。
        （回调机制没白学：生产环境接 Langfuse / LangSmith 做追踪用的是同一套钩子。）
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agent import db, kb

load_dotenv()

# 与手写版 agent/tools.py 的 CONFIRM_REQUIRED 对齐
CONFIRM_REQUIRED = {"cancel_order", "update_address"}

# 递归上限：手写版"10 轮保险丝"的框架等价物（命令行版用的是 20，这里保持一致）
MAX_RECURSION = 20

SYSTEM_PROMPT = (
    '你是"订单智能助手"，一个能查数据、能办事、懂售后政策的客服助手。\n'
    "行为准则：\n"
    "1. 用户问订单相关的问题时，必须调用工具获取真实数据，绝不编造。\n"
    "2. 查询订单先用 get_user_orders 拿到订单号，再按需用 get_order_status 看详情。\n"
    '3. 用户要求"取消订单"时，调用 cancel_order 工具。\n'
    '4. 用户要求"修改收货地址"时，调用 update_address 工具。\n'
    "5. 用户问退货、运费、发票、保修等售后政策时，必须先调用 search_knowledge_base，"
    "依据检索原文回答，检索不到就说不知道。\n"
    "6. 用简洁友好的中文回答，金额带货币符号 ¥。\n"
    '7. 工具返回空结果时，如实说"没有查到"，不要猜测。\n'
    "8. 如果工具返回 pending_confirm（待用户确认），用一句话请用户确认即可，不要重复调用该工具。\n"
    "9. 如果系统消息告知某操作已处理完毕，直接答复用户结果，不要再调用工具。"
)


def _text(content) -> str:
    """把模型返回的 content 拍平成字符串（兼容 str / 分段列表两种形态）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content or "")


def _parse_result(output):
    """把 ToolMessage 还原成可读结果（dict / str）。

    on_tool_end 和 messages 里的 tool 消息，真正的返回值都在 .content 里
    （字符串形式的 JSON），所以这里先取 content 再尝试解析。
    """
    if isinstance(output, (dict, list)):
        return output
    content = getattr(output, "content", None)
    if content is not None and content is not output:
        return _parse_result(content)
    s = str(output).strip()
    if s.startswith(("{", "[")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _events_from_messages(messages) -> list[dict]:
    """把框架返回的 messages 流翻译成与手写版一致的事件流。

    手写版是在 while 循环里手动 append 事件；框架版没有那个循环，
    就在每轮 invoke 之后把**本轮新增的** messages 翻译一遍。
    """
    events: list[dict] = []
    for m in messages:
        t = getattr(m, "type", "")
        if t == "ai":
            # tool_calls 已经是结构化 dict：{"name":…, "args":{…}, "id":…}
            for tc in getattr(m, "tool_calls", None) or []:
                events.append(
                    {
                        "type": "tool_call",
                        "name": tc.get("name", "?"),
                        "args": tc.get("args") or {},
                    }
                )
        elif t == "tool":
            events.append(
                {
                    "type": "tool_result",
                    "name": getattr(m, "name", None) or "?",
                    "result": _parse_result(m),
                }
            )
    return events


@dataclass
class LangChainSession:
    """框架版会话。对外接口与 AgentSession 完全一致，UI 可以无差别对待。"""

    api_key: str
    base_url: str
    model: str
    # 框架版不自动记历史（命令行版也是自己维护），这里存 LangChain 的 message 列表
    history: list = field(default_factory=list)
    pending_action: dict | None = None

    def __post_init__(self):
        # 登记待确认操作 / 以及"确认后勿重复挂起"的开关
        self._holder: dict = {}
        # 确认后真正执行工具的落地函数（与手写版共用 db，只是执行时机不同）
        self._dispatch = {
            "cancel_order": lambda **kw: db.cancel_order(kw["order_id"]),
            "update_address": lambda **kw: db.update_address(kw["order_id"], kw["new_address"]),
        }
        llm_kwargs = dict(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
        )
        extra = self._extra_body()
        if extra:  # 只在需要时传，避免非智谱端点收到未知字段
            llm_kwargs["extra_body"] = extra
        llm = ChatOpenAI(**llm_kwargs)

        # ★ 这一行 = 手写版 agent/core.py 里那个 while 循环的全部替代品 ★
        self.graph = create_agent(
            model=llm,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def _extra_body(self) -> dict | None:
        """与手写版同款优化：只对智谱端点关掉深度思考（省 token、提速约一倍）。

        手写版在 core.py._model_kwargs() 里做同一件事 —— 这是"优化手段可以在
        两版之间平移"的又一个证据。
        """
        if "bigmodel.cn" not in (self.base_url or ""):
            return None
        if os.getenv("LLM_DISABLE_THINKING", "true").lower() == "false":
            return None
        return {"thinking": {"type": "disabled"}}

    # ---------- 工具定义：@tool 装饰器自动生成 JSON Schema ----------

    def _build_tools(self):
        holder = self._holder

        @tool
        def get_user_orders(uid: str) -> list:
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
            """取消一个订单（危险操作，需用户确认后才真正执行）。
            只有「待发货」状态的订单才能取消。

            Args:
                order_id: 要取消的订单号，例如 o1003
            """
            if holder.get("suppress"):
                return {"ok": False, "reason": "该操作已处理完毕，请勿重复调用，直接答复用户结果。"}
            holder["action"] = {"name": "cancel_order", "args": {"order_id": order_id}}
            return {
                "ok": False,
                "pending_confirm": True,
                "reason": "已登记待用户确认；前端确认后会真正执行，请先请用户确认。",
            }

        @tool
        def update_address(order_id: str, new_address: str) -> dict:
            """修改订单的收货地址（危险操作，需用户确认后才真正执行）。
            只有「待发货」状态的订单才能修改地址。

            Args:
                order_id: 要修改的订单号，例如 o1003
                new_address: 新的完整收货地址
            """
            if holder.get("suppress"):
                return {"ok": False, "reason": "该操作已处理完毕，请勿重复调用，直接答复用户结果。"}
            holder["action"] = {
                "name": "update_address",
                "args": {"order_id": order_id, "new_address": new_address},
            }
            return {
                "ok": False,
                "pending_confirm": True,
                "reason": "已登记待用户确认；前端确认后会真正执行，请先请用户确认。",
            }

        @tool
        def search_knowledge_base(query: str) -> dict:
            """检索公司售后政策知识库（RAG）。退货/运费/发票/保修/配送/积分等
            政策类问题用这个工具，依据检索到的原文回答。

            Args:
                query: 要检索的问题，例如：退货政策是什么
            """
            # RAG 是只读操作，无需确认；且与手写版复用同一个 kb.search_knowledge_base
            return kb.search_knowledge_base(query)

        return [get_user_orders, get_order_status, cancel_order, update_address, search_knowledge_base]

    # ---------- 主入口（与手写版同名同签名） ----------

    def send(self, user_text: str | None = None) -> list[dict]:
        """处理一轮对话，返回事件列表。"""
        self._holder.clear()  # 新一轮：清掉上一轮登记，解开"勿重复挂起"开关
        if user_text:
            self.history.append(("user", user_text))
        return self._run()

    def confirm(self, yes: bool) -> list[dict]:
        """用户对挂起操作表态后，执行（或放弃）工具，再让模型收尾。"""
        action = self.pending_action
        if not action:
            return []
        self.pending_action = None
        name, args = action["name"], action["args"]

        if yes:
            try:
                result = self._dispatch[name](**args)
            except Exception as e:  # noqa: BLE001 —— 执行失败也要回灌给模型，别让页面白屏
                result = {"ok": False, "reason": f"执行失败：{e}"}
            note = (
                "【系统】用户已点击确认，"
                f"{name} 已按上面登记的参数真实执行，执行结果："
                f"{json.dumps(result, ensure_ascii=False, default=str)}。"
                "请用一句话告知用户结果，不要再调用工具。"
            )
        else:
            result = {"ok": False, "reason": "用户拒绝了本次操作"}
            note = (
                "【系统】用户点击了拒绝，该操作未执行。请不要再次调用该工具，"
                "直接告诉用户已取消即可。"
            )

        self.history.append(("user", note))
        # suppress：万一模型仍重复调用危险工具，也不要再弹一次确认框（防死循环）
        self._holder.clear()
        self._holder["suppress"] = True
        return [{"type": "tool_result", "name": name, "result": result}] + self._run()

    # ---------- 内部 ----------

    def _run(self) -> list[dict]:
        sent = list(self.history)  # 本轮发进去的历史（用来切出"本轮新增"的消息）
        try:
            result = self.graph.invoke(
                {"messages": sent},
                config={"recursion_limit": MAX_RECURSION},
            )
        except Exception as e:  # noqa: BLE001 —— Key 错/网络断都给一句人话
            return [
                {
                    "type": "answer",
                    "content": f"框架版调用出错：{e}\n\n请检查 .env 里的 API Key / Base URL / 模型名。",
                }
            ]

        # 框架返回完整消息流（含它自己插入的 system）；存回来时去掉 system，
        # 否则下次 invoke 时 create_agent 会再插一条，变成两条 system。
        full = [m for m in result["messages"] if getattr(m, "type", "") != "system"]
        new_msgs = full[len(sent):]  # 只翻译本轮新增，避免历史事件被重复渲染
        self.history = full

        events = _events_from_messages(new_msgs)

        action = self._holder.get("action")
        if action:
            # 与手写版行为一致：挂起点只产出确认事件，不产出最终回答
            self.pending_action = action
            events.append({"type": "confirm", "name": action["name"], "args": action["args"]})
            return events

        self.pending_action = None
        # 只在本轮新增消息里找最终回答：退回到 full 会把上一轮的旧回答又显示一遍
        answer = self._last_ai(new_msgs) or "（模型本轮没有返回文本，请再试一次）"
        events.append({"type": "answer", "content": answer})
        return events

    @staticmethod
    def _last_ai(messages) -> str:
        for m in reversed(list(messages)):
            if getattr(m, "type", "") == "ai":
                text = _text(getattr(m, "content", ""))
                if text.strip():
                    return text
        return ""
