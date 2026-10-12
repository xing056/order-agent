"""智能体核心：AgentSession —— 行动循环 + 危险操作确认 + 轮数保险丝。

【这 130 行就是"手写智能体"的全部】
  循环五步（对照着读 send 方法）：
    1. 把对话历史 + 工具说明书发给模型
    2. 模型要么直接回答（结束），要么提议调用工具（tool_calls）
    3. 普通工具：立即执行，结果以 role="tool" 回灌进对话历史
    4. 危险操作：挂起循环，把控制权交回前端等用户确认
    5. 回到第 1 步 —— 模型读到工具结果后决定下一步，直到收工

  LangChain 的 create_agent / LangGraph 的状态机，封装的就是这个循环。
  想知道框架替你做了什么，答案全在这 130 行里。

【事件模型】（前端靠它渲染"工具调用过程"）
    {"type": "tool_call",    "name": ..., "args": ...}            模型提议调用工具
    {"type": "tool_result",  "name": ..., "result": ...}          工具执行完成
    {"type": "confirm",      "name": ..., "args": ...}            危险操作，等待用户确认（循环暂停）
    {"type": "answer",       "content": ...}                      最终回复（循环结束）
"""

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

from agent.tools import TOOLS, TOOL_DISPATCH, CONFIRM_REQUIRED

# ---------- 系统提示词（提示词工程的主战场） ----------
# 模型行为好不好，一半取决于这段话。三处关键设计：
#   · "必须调工具、绝不编造" —— 堵住模型幻觉数据的老毛病
#   · 点名各工具的使用时机 —— 相当于给新员工的岗位说明书
#   · "政策类问题先查知识库" —— RAG 的触发规则写在提示词里

SYSTEM_PROMPT = """你是"订单智能助手"，一个能查数据、能办事、懂售后政策的客服助手。

行为准则：
1. 用户问订单相关的问题时，必须调用工具获取真实数据，绝不编造。
2. 查询订单要先用 get_user_orders 拿到订单号，再按需用 get_order_status 看详情。
3. 用户要求"取消订单"时，调用 cancel_order 工具（系统会自动让用户确认）。
4. 用户要求"修改收货地址"时，调用 update_address 工具（系统会自动让用户确认）。
5. 用户问退货、运费、发票、保修、配送、积分等售后政策时，必须先调用 search_knowledge_base 检索知识库，依据检索到的原文回答；检索不到就说不知道，绝不编造政策。
6. 用简洁友好的中文回答，金额要带货币符号 ¥。
7. 工具返回空结果时，如实告诉用户"没有查到"，不要猜测。
"""

# ---------- 轮数保险丝（防死循环 / 防 token 失控） ----------
# 模型理论上可能反复调工具不收工（尤其是提示词有歧义时）。
# 到达上限就强制收口：把"已达轮数上限"作为信息回灌，让模型
# 基于已有内容给出最终回答。这是智能体特有的"软故障"防线之一。
MAX_ROUNDS = 10


def _assistant_msg_dict(msg) -> dict:
    """把 SDK 返回的 message 对象转成可回填 messages 的字典。

    为什么需要它：OpenAI 协议规定，role="tool" 的消息前面必须
    有一条带 tool_calls 的 assistant 消息（提议和结果要配对），
    否则 API 直接报错。这是新手最常踩的协议坑。
    """
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in (msg.tool_calls or [])
    ]
    return {"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls}


@dataclass
class AgentSession:
    """一个用户的会话。生产环境"一个会话一个实例"，状态外置到 Redis 后可跨进程恢复。"""

    api_key: str
    base_url: str
    model: str
    # 对话历史 = 智能体的"短期记忆"。system 提示词常驻第一条。
    messages: list = field(default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}])
    pending_action: dict | None = None  # 挂起的危险操作 {call_id, name, args, assistant_msg}

    def __post_init__(self):
        # 用 OpenAI SDK 直连兼容模式端点（百炼/DeepSeek 都是这个协议）。
        # 没有任何框架 —— 这就是"手写"的含义。
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _model_kwargs(self) -> dict:
        """按厂商拼装额外的请求参数（目前只用来关掉智谱的深度思考）。

        背景：智谱（open.bigmodel.cn）的新模型默认开启深度思考——模型在输出
        工具调用之前会先 internally 推理一段。对我们这种"查数据 / 调工具"的
        标准化任务是纯浪费，实测同一个"查订单"请求：

            开思考：3.5s，思考 42 token + 输出 68 token
            关思考：1.8s，输出 24 token，工具调用结果完全一样

        思考 token 照样计费，所以关掉 = 提速一倍 + 省掉一大半输出成本。

        只对智谱端点传这个参数（换 DeepSeek / 百炼时自动不传，避免未知字段报错）。
        想保留思考过程：.env 里设 LLM_DISABLE_THINKING=false。
        """
        if "bigmodel.cn" not in (self.base_url or ""):
            return {}
        if os.getenv("LLM_DISABLE_THINKING", "true").lower() == "false":
            return {}
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    # ---------- 主入口 ----------

    def send(self, user_text: str | None = None) -> list[dict]:
        """处理一轮对话（可带用户输入），返回事件列表。

        user_text=None 的场景：confirm() 之后继续跑循环用 ——
        用户确认/拒绝本身不是新的用户消息，只是"补发"工具结果。
        """
        events: list[dict] = []
        if user_text:
            self.messages.append({"role": "user", "content": user_text})

        rounds = 0
        while True:
            rounds += 1
            if rounds > MAX_ROUNDS:  # 保险丝熔断：强制收口，不让循环无限转下去
                self.messages.append(
                    {"role": "user", "content": "（系统提示：已达到工具调用轮数上限，请基于已有信息直接回答用户）"}
                )

            # 第 1 步：带着全部历史 + 工具说明书请求模型
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,          # 工具说明书随请求带上，模型每次都"看得到"
                tool_choice="auto",   # 自动：模型自己决定调不调工具
                **self._model_kwargs(),  # 厂商特有参数（目前=关掉智谱深度思考）
            )
            msg = resp.choices[0].message

            # 第 2 步：没有工具调用 = 模型认为任务完成，循环退出
            if not msg.tool_calls:
                answer = msg.content or "（无回复）"
                events.append({"type": "answer", "content": answer})
                self.messages.append({"role": "assistant", "content": answer})
                return events

            # 第 3 步：逐个处理模型提议的工具调用
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    # 模型返回的参数是 JSON 字符串，需要解析
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}  # 参数解析失败也回灌错误，让模型自己纠错重试

                events.append({"type": "tool_call", "name": name, "args": args})

                # 第 4 步：危险操作 → 挂起，把控制权交回前端
                if name in CONFIRM_REQUIRED:
                    self.pending_action = {"call_id": call.id, "name": name, "args": args, "assistant_msg": msg}
                    events.append({"type": "confirm", "name": name, "args": args})
                    return events  # 循环在这里暂停 —— 这就是"人留在决策环里"

                # 第 5 步：普通工具 → 立即执行，结果回灌
                result = TOOL_DISPATCH[name](**args)
                events.append({"type": "tool_result", "name": name, "result": result})
                self.messages.append(_assistant_msg_dict(msg))  # 先补提议（协议配对要求）
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
                )

    # ---------- 危险操作确认 ----------

    def confirm(self, yes: bool) -> list[dict]:
        """用户对挂起操作表态后，把结果回灌给模型并继续循环。

        整个流程：send 挂起 → 前端弹按钮 → 用户点击 → confirm 补发
        tool 结果 → 继续 send 直到最终回答。这对应 LangGraph 的
        interrupt / human-in-the-loop 机制，只不过这里循环在我们
        手里，想停就停，不需要框架支持。
        """
        pa = self.pending_action
        if pa is None:
            return []
        self.pending_action = None

        # 先把 assistant 的 tool_calls 提议补进历史（OpenAI 要求配对）
        self.messages.append(_assistant_msg_dict(pa["assistant_msg"]))

        if yes:
            result = TOOL_DISPATCH[pa["name"]](**pa["args"])
            content = json.dumps(result, ensure_ascii=False)
            events = [{"type": "tool_result", "name": pa["name"], "result": result}]
        else:
            # 拒绝也是一种"工具结果"——模型读到后会改口告知用户，而不是卡死
            result = {"ok": False, "reason": "用户拒绝了本次操作"}
            content = json.dumps(result, ensure_ascii=False)
            events = [{"type": "tool_result", "name": pa["name"], "result": result}]

        self.messages.append({"role": "tool", "tool_call_id": pa["call_id"], "content": content})
        events.extend(self.send(None))  # 继续循环，直到最终回答
        return events
