"""知识库层：RAG 的最小可用实现（向量检索 + 关键词兜底）。

【一句话原理】
  RAG = 开卷考试：模型答题前，先由这里替它把"书"翻到最相关的几页，
  把原文拼进上下文，模型照着答 —— 解决模型不知道公司私有政策的问题。

【两条流水线】（对应 RAG 的经典划分）
  离线建库（做一次）：
      FAQ 文档切成条 → 调 embedding API 把每条算成向量 → 缓存到 JSON 文件
  在线检索（每次提问）：
      用户问题 → 算成向量 → 和库里每条算余弦相似度 → 取 top-k 原文返回

【为什么这样实现而不是上向量数据库】
  Chroma/Milvus 是生产规模的选择（百万级文档、持久化、索引）。
  demo 只有十几条 FAQ，用"JSON 文件 + 纯 Python 余弦"把原理摊开，
  每一行都看得懂 —— 也就是说，这里故意没用向量库：我想让检索这一步保持透明，
  换向量库只是把这个 for 循环换成一次查询。

【生产思维：外部依赖必须有降级路径】
  embedding API 挂了不等于知识库瘫痪 —— 自动降级为关键词匹配，
  精度下降但服务可用。这和"数据库挂了读缓存"是同一种思路。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "kb_vectors.json"

# embedding（向量化模型）允许单独配置厂商 —— 因为很多 chat 厂商不提供 embedding
# （DeepSeek 就是一个典型：对话很强，但没有向量接口）。
# 这样就能"DeepSeek 聊天 + 智谱做向量"混搭。留空则回退到通用配置。
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "text-embedding-v3")
EMBED_API_KEY = os.getenv("LLM_EMBED_API_KEY") or os.getenv("LLM_API_KEY")
EMBED_BASE_URL = os.getenv("LLM_EMBED_BASE_URL") or os.getenv("LLM_BASE_URL")
# 是否"向量走另一家"：单独配了 LLM_EMBED_BASE_URL 就说明是跨厂商混搭
_SEPARATE_EMBED_VENDOR = bool(os.getenv("LLM_EMBED_BASE_URL"))


def _embed_enabled() -> bool:
    """是否启用了向量检索。

    没配 embedding 模型（或 Key 缺失）时直接走关键词模式 —— 不浪费一次
    注定失败的 API 往返（演示现场每多等 1 秒都是风险）。

    跨厂商时必须配独立的 Key：否则会拿着对话厂商的 Key 去打向量厂商的接口，
    必然 401。与其白跑一趟，不如干净地降级到关键词模式。
    """
    if not EMBED_MODEL:
        return False
    if _SEPARATE_EMBED_VENDOR and not os.getenv("LLM_EMBED_API_KEY"):
        return False
    return bool(EMBED_API_KEY)

# ---------- 1. 知识源：公司售后政策 FAQ ----------
# 生产环境这些内容来自文档系统（Word/PDF/网页），需要额外的切块(chunking)步骤；
# demo 直接写成结构化的问答对，等价于"已经切好的块"。

FAQ_ENTRIES = [
    {"question": "退货政策是什么？", "answer": "签收后 7 天内可无理由退货，商品需保持不影响二次销售的状态；食品、定制品等特殊商品不支持无理由退货。"},
    {"question": "退货运费谁承担？", "answer": "质量问题退货，运费由商家全额承担；无理由退货，运费由买家承担，可使用赠送的运费券抵扣。"},
    {"question": "怎么开发票？", "answer": "订单完成后 90 天内可在「订单详情-开具发票」中申请电子普票或纸质专票，抬头信息可修改，1 个工作日内开出。"},
    {"question": "保修政策是什么？", "answer": "数码类商品自签收起 1 年保修，耗材类（鼠标垫、支架等）不在保修范围；保修期内非人为损坏免费维修或换新。"},
    {"question": "订单什么时候发货？", "answer": "工作日 16:00 前支付的订单当天发货，之后次日发货；大促期间可能延迟 1~2 天，可在订单页查看预计发货时间。"},
    {"question": "配送一般多久到？", "answer": "同城次日达，省内 1~2 天，跨省 2~4 天，偏远地区 5~7 天；发货后会推送含物流单号的通知。"},
    {"question": "什么情况下可以取消订单？", "answer": "只有「待发货」状态的订单可以直接取消，已发货订单需等签收后走退货流程；已取消订单的退款 3 个工作日内原路退回。"},
    {"question": "可以修改收货地址吗？", "answer": "「待发货」状态的订单可以修改收货地址，已发货后无法修改，可以联系快递员协商转寄。"},
    {"question": "会员积分规则是什么？", "answer": "每消费 1 元积 1 分，评价晒图额外送 50 分；100 积分抵 1 元，下单时可勾选使用，积分有效期 1 年。"},
    {"question": "售后怎么联系人工客服？", "answer": "App 内「我的-客服中心」可转人工，工作日 9:00-21:00 在线；紧急问题可拨打 400-800-1234。"},
]


# ---------- 2. 向量化：文本 → 一串数字（语义坐标） ----------
# embedding 模型把"意思相近的句子"映射到空间中相近的位置，
# 所以"退货要花钱吗"和"退货运费谁承担"的向量距离会很近 ——
# 这就是"语义检索"能听懂人话的原因（关键词匹配做不到）。

def _embed(texts: list[str]) -> list[list[float]]:
    """调 embedding API 把一批文本转成向量。"""
    client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：两串数字的"方向"有多一致（1=完全相同，0=毫无关系）。

    纯 Python 实现，就是教科书公式：点积 / (模 × 模)。
    向量库里这行是 C/Rust 写的索引，快几个数量级，但数学上就是这件事。
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------- 3. 离线建库（带缓存：只在知识变化或首次运行时调 API） ----------

def _build_index() -> list[dict]:
    """给每条 FAQ 算向量。结果缓存到 data/kb_vectors.json，避免每次启动都调 API。"""
    texts = [f"{e['question']}\n{e['answer']}" for e in FAQ_ENTRIES]
    vectors = _embed(texts)
    cache = {
        "entries": FAQ_ENTRIES,
        "vectors": vectors,
        # 记下"这份缓存是谁算的"：换厂商/换模型后向量空间完全不同，
        # 混用会导致余弦相似度算出一堆无意义的数 —— 而且不报错，静默出错。
        "meta": {"model": EMBED_MODEL, "base_url": EMBED_BASE_URL, "dim": len(vectors[0])},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def _load_index() -> dict:
    """加载知识库索引：有缓存且"对得上"就读缓存，否则重建。

    三重校验（任何一条不满足都重建）：
      1. FAQ 内容有没有改过
      2. embedding 模型是不是同一个
      3. 向量维度一不一致
    """
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        meta = cache.get("meta") or {}
        if (
            cache.get("entries") == FAQ_ENTRIES
            and cache.get("vectors")
            and meta.get("model") == EMBED_MODEL
            and meta.get("base_url") == EMBED_BASE_URL
            and meta.get("dim") == len(cache["vectors"][0])
        ):
            return cache
    return _build_index()


# ---------- 4. 在线检索：对外的工具函数 ----------

def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """知识库检索入口（智能体工具之一）。

    主路径：向量检索（问题向量化 → 和每条 FAQ 算余弦 → 取 top-k）；
    降级路径：没配 embedding、或 embedding API 调用失败时，自动切换关键词匹配。
    两种降级都会写清原因，让模型和调试者知道这次走的是哪条路。
    """
    # 没配 embedding 就别白跑一趟 API（演示现场每多等 1 秒都是风险）
    if not _embed_enabled():
        # 区分两种"没配"，让调试的人一眼知道该补哪个变量
        if _SEPARATE_EMBED_VENDOR and not os.getenv("LLM_EMBED_API_KEY"):
            reason = "未填 LLM_EMBED_API_KEY（向量厂商的独立 Key 缺失）"
        else:
            reason = "未配置 embedding（LLM_EMBED_MODEL 为空）"
        return _keyword_search(query, top_k, reason)
    try:
        index = _load_index()
        query_vec = _embed([query])[0]
        scored = [
            {**entry, "score": round(_cosine(query_vec, vec), 4)}
            for entry, vec in zip(index["entries"], index["vectors"])
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        hits = [
            {"question": h["question"], "answer": h["answer"], "score": h["score"]}
            for h in scored[:top_k]
            if h["score"] >= 0.3  # 相似度阈值：太低的块拼进 prompt 只是浪费 token
        ]
        return {"mode": "向量检索", "results": hits}
    except Exception as e:  # noqa: BLE001  # embedding 挂了 → 关键词兜底
        return _keyword_search(query, top_k, f"向量服务不可用：{type(e).__name__}")


def _keyword_search(query: str, top_k: int, reason: str) -> dict:
    """降级检索：2 字滑窗做关键词匹配（讲不清原理时就说"关键词兜底"）。"""
    keywords = [w for w in _tokenize(query)]
    scored = []
    for entry in FAQ_ENTRIES:
        text = entry["question"] + entry["answer"]
        score = sum(1 for w in keywords if w in text)
        if score > 0:
            scored.append({**entry, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "mode": f"关键词兜底（{reason}）",
        "results": [{"question": h["question"], "answer": h["answer"]} for h in scored[:top_k]],
    }


def _tokenize(text: str) -> list[str]:
    """中文简易分词：2 字滑窗。够兜底用，不讲效率。"""
    text = text.strip()
    return [text[i: i + 2] for i in range(len(text) - 1)] if len(text) > 1 else [text]


if __name__ == "__main__":
    # 单独运行本文件 = 手动触发"离线建库"，提前把向量缓存好
    idx = _build_index()
    print(f"知识库建库完成：{len(idx['entries'])} 条 FAQ，向量缓存于 {CACHE_PATH}")
