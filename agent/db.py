"""数据层：SQLite 订单库（用户表 + 订单表），首次运行自动建表造数。

【设计要点】
  这一层和任何传统后端数据层没有区别 —— 智能体的"手"最终
  摸到的还是你熟悉的 SQL、连接管理、业务校验、事务提交。
  所谓"工具"，本质就是这些普通函数：被 JSON Schema 描述之后
  暴露给模型而已。模型负责"决定调谁"，这里负责"真正干活"。

【订单状态约定（演示用中文，非技术人也能看懂）】
  待发货 → 已发货 → 已完成
     └──────→ 已取消（只有"待发货"允许取消 / 改地址）
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.db"


def get_conn() -> sqlite3.Connection:
    """获取数据库连接。

    row_factory=sqlite3.Row 让查询结果可以像字典一样按列名取值
    （row["order_id"]），比按下标取值可读得多。
    每次操作都"开连接→用完关"，demo 规模够用；
    生产环境会换连接池（每次建连的开销在高并发下不可接受）。
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    """建表；表为空或 force=True 时灌入演示数据。

    force=True 是重置数据库的开关 —— 需要干净的初始数据时跑一次，
    保证 o1003 回到"待发货"状态，取消/改地址的演示永远可复现：
        python -c "from agent import db; db.init_db(force=True)"
    """
    conn = get_conn()
    try:
        if force:
            # force=True = 彻底重置：先删旧表再建，保证结构（列）也是最新的。
            # 否则旧的 orders.db 里没有 address 列，CREATE TABLE IF NOT EXISTS
            # 不会补列，插入就会报错 —— 演示环境换电脑/改表结构后靠它兜底。
            conn.executescript("DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS users;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                uid   TEXT PRIMARY KEY,
                name  TEXT NOT NULL,
                phone TEXT
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id   TEXT PRIMARY KEY,
                uid        TEXT NOT NULL,
                item       TEXT NOT NULL,
                amount     REAL NOT NULL,
                status     TEXT NOT NULL DEFAULT '待发货',
                address    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (uid) REFERENCES users(uid)
            );
            """
        )
        count = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"]
        if count == 0 or force:
            seed(conn)
        conn.commit()
    finally:
        conn.close()


def seed(conn: sqlite3.Connection) -> None:
    """灌入演示数据：3 个用户、8 笔订单，状态覆盖 待发货/已发货/已完成。

    演示数据是"导演安排"过的：每个用户都有一笔"待发货"订单，
    保证取消 / 改地址的演示永远有素材。
    """
    conn.executemany(
        "INSERT OR REPLACE INTO users (uid, name, phone) VALUES (?, ?, ?)",
        [
            ("u001", "张伟", "13800000001"),
            ("u002", "李娜", "13800000002"),
            ("u003", "王强", "13800000003"),
        ],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO orders "
        "(order_id, uid, item, amount, status, address, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("o1001", "u001", "无线蓝牙耳机", 299.0, "已完成", "北京市海淀区中关村大街1号", "2026-08-01 10:00:00"),
            ("o1002", "u001", "机械键盘", 459.0, "已发货", "北京市海淀区中关村大街1号", "2026-08-15 14:30:00"),
            ("o1003", "u001", "显示器支架", 129.0, "待发货", "北京市海淀区中关村大街1号", "2026-08-28 09:20:00"),
            ("o2001", "u002", "保温杯", 89.0, "已完成", "上海市浦东新区张江路100号", "2026-08-05 16:00:00"),
            ("o2002", "u002", "便携充电宝", 159.0, "已发货", "上海市浦东新区张江路100号", "2026-08-20 11:45:00"),
            ("o2003", "u002", "鼠标垫", 39.9, "待发货", "上海市浦东新区张江路100号", "2026-08-29 20:10:00"),
            ("o3001", "u003", "USB-C 扩展坞", 329.0, "已完成", "广州市天河区体育西路50号", "2026-08-10 09:00:00"),
            ("o3002", "u003", "4K 显示器", 1899.0, "待发货", "广州市天河区体育西路50号", "2026-08-30 15:40:00"),
        ],
    )


# ---------- 以下为工具层调用的查询/操作函数 ----------
# 这些就是"智能体的手"：每个函数都会被 tools.py 里的 JSON Schema
# 描述一次，模型看到描述后就能"提议"调用，由 TOOL_DISPATCH 真正执行。


def get_user_orders(uid: str) -> list[dict]:
    """查询某用户全部订单（按下单时间倒序）。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT order_id, item, amount, status, address, created_at FROM orders WHERE uid = ? "
            "ORDER BY created_at DESC",
            (uid,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_order_status(order_id: str) -> dict | None:
    """查询单个订单详情（联表带出下单人姓名）。"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT o.order_id, u.name AS user_name, o.item, o.amount, o.status, o.address, o.created_at "
            "FROM orders o JOIN users u ON o.uid = u.uid WHERE o.order_id = ?",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cancel_order(order_id: str) -> dict:
    """取消订单（危险操作，调用前会被系统挂起等用户确认）。

    业务校验：只有"待发货"状态允许取消 —— 已发货/已完成的订单
    走退货流程而不是取消。校验放在数据层而不是靠提示词约束，
    因为"绝不相信模型输出的正确性"是安全底线。
    返回值统一是 dict（成功带 ok=True，失败带 ok=False + reason），
    这样回灌给模型的信息结构稳定，模型能读懂并如实转述给用户。
    """
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT order_id, item, status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return {"ok": False, "reason": "订单不存在"}
        if row["status"] != "待发货":
            return {"ok": False, "reason": f"订单当前状态为「{row['status']}」，只有「待发货」的订单可以取消"}
        conn.execute("UPDATE orders SET status = '已取消' WHERE order_id = ?", (order_id,))
        conn.commit()
        return {"ok": True, "order_id": order_id, "item": row["item"], "new_status": "已取消"}
    finally:
        conn.close()


def update_address(order_id: str, new_address: str) -> dict:
    """修改订单收货地址（危险操作，调用前会被系统挂起等用户确认）。

    业务校验：同样只有"待发货"状态允许改地址 —— 已发货的包裹
    追不回来，改了也没意义。这是从取消订单复制过来的同一套
    "状态机守卫"思路：写操作前先检查状态是否允许。
    """
    if not new_address or not new_address.strip():
        return {"ok": False, "reason": "新地址不能为空"}
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT order_id, item, status, address FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return {"ok": False, "reason": "订单不存在"}
        if row["status"] != "待发货":
            return {"ok": False, "reason": f"订单当前状态为「{row['status']}」，只有「待发货」的订单可以修改地址"}
        conn.execute(
            "UPDATE orders SET address = ? WHERE order_id = ?", (new_address.strip(), order_id)
        )
        conn.commit()
        return {
            "ok": True,
            "order_id": order_id,
            "item": row["item"],
            "old_address": row["address"],
            "new_address": new_address.strip(),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    init_db(force=True)
    print("数据库已初始化：", DB_PATH)
