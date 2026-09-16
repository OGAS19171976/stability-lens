"""stability_lens.datasets — 数据集加载（numpy + 标准库，不依赖 pandas）。

默认使用工作区里的真实宽表（数仓 demo 的订单表）做**回归**任务：
用下单时间、履约时长、状态、用户下单频次等特征预测 ``order_amount``。
这份数据是真实工作流的形状 —— 时间格式不统一、状态是枚举、存在异常时长 ——
正好用来演示"条件数很差时，η_max 是算出来的而不是试出来的"。

若文件不存在，退化为一个固定种子的合成回归集，保证实验在任何机器上可复现。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray

__all__ = ["Dataset", "standardize", "load_order_amount", "make_synthetic", "DEFAULT_CSV"]

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "dw-etl-demo" / "data" / "raw" / "order_info.csv"

# 真实数据里同时存在这两种格式，故意不统一
_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M")


@dataclass
class Dataset:
    name: str
    X: Array
    y: Array
    feature_names: list[str]
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


def parse_dt(text: str) -> datetime | None:
    text = (text or "").strip()
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def standardize(X: Array, y: Array) -> tuple[Array, Array, dict[str, Array]]:
    """列标准化 + 目标标准化。**这一步不是装饰**：它直接决定 Hessian 的条件数。"""
    xm = X.mean(axis=0)
    xs = X.std(axis=0)
    xs[xs == 0.0] = 1.0
    ym = float(y.mean())
    ys = float(y.std()) or 1.0
    return (X - xm) / xs, (y - ym) / ys, {"X_mean": xm, "X_std": xs, "y_mean": ym, "y_std": ys}


def load_order_amount(
    path: str | Path | None = None,
    max_rows: int = 1200,
    seed: int = 0,
) -> Dataset:
    """构造「预测订单金额」的回归集。

    特征（全部由原始字段现场算出来，含脏数据清洗）：

    ======================  ================================================
    ``hour``                下单小时（周期量的最粗粒度）
    ``weekday``             星期几
    ``lead_hours``          下单 → 操作 的时长（小时，截断到 [0, 72]）
    ``log_amount_prior``    该用户历史订单金额均值（频次+均值编码）
    ``user_order_count``    该用户订单数（log1p）
    ``status_1001`` …       ``order_status`` 的 one-hot（取最高频 4 个）
    ======================  ================================================

    目标：``log1p(order_amount)``。
    """
    path = Path(path) if path is not None else DEFAULT_CSV
    if not path.exists():
        ds = make_synthetic(seed=seed)
        ds.note = f"未找到 {path}，回退到合成数据；{ds.note}"
        return ds

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    # 第一遍：用户级统计（频次 / 金额均值），并挑出高频状态
    user_count: dict[str, int] = {}
    user_sum: dict[str, float] = {}
    status_count: dict[str, int] = {}
    for r in rows:
        uid = r.get("user_id", "")
        user_count[uid] = user_count.get(uid, 0) + 1
        try:
            amt = float(r.get("order_amount", "nan"))
        except ValueError:
            amt = float("nan")
        if math.isfinite(amt):
            user_sum[uid] = user_sum.get(uid, 0.0) + amt
        st = (r.get("order_status") or "").strip()
        status_count[st] = status_count.get(st, 0) + 1
    top_status = [s for s, _ in sorted(status_count.items(), key=lambda kv: -kv[1])[:4]]
    status_index = {s: i for i, s in enumerate(top_status)}

    feature_names = ["hour", "weekday", "lead_hours", "log_amount_prior",
                     "user_order_count"] + [f"status_{s}" for s in top_status]
    feats: list[list[float]] = []
    targets: list[float] = []
    skipped = 0
    for r in rows:
        t0 = parse_dt(r.get("create_time", ""))
        t1 = parse_dt(r.get("operate_time", ""))
        try:
            amt = float(r.get("order_amount", "nan"))
        except ValueError:
            amt = float("nan")
        if t0 is None or not math.isfinite(amt) or amt <= 0.0:
            skipped += 1
            continue
        lead = (t1 - t0).total_seconds() / 3600.0 if t1 is not None else 0.0
        lead = min(max(lead, 0.0), 72.0)
        uid = r.get("user_id", "")
        cnt = user_count.get(uid, 1)
        prior = user_sum.get(uid, amt) / max(cnt, 1)
        st = (r.get("order_status") or "").strip()
        status_oh = [0.0] * len(top_status)
        if st in status_index:
            status_oh[status_index[st]] = 1.0
        feats.append([float(t0.hour), float(t0.weekday()), lead,
                      math.log1p(max(prior, 0.0)), math.log1p(cnt)] + status_oh)
        targets.append(math.log1p(amt))
        if max_rows and len(targets) >= max_rows * 4:
            break  # 先多读一些，下面再抽样，避免只拿到时间上最早的一段

    X = np.asarray(feats, dtype=float)
    y = np.asarray(targets, dtype=float)
    if max_rows and X.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(X.shape[0], size=max_rows, replace=False))
        X, y = X[idx], y[idx]

    names = path.name
    note = (f"{names}: 读取 {len(rows)} 行，跳过 {skipped} 行脏数据，"
            f"抽样保留 {X.shape[0]} 行；状态取最高频 4 类 {top_status}")
    return Dataset(name=f"order_amount({names})", X=X, y=y,
                   feature_names=feature_names, note=note)


def make_synthetic(n: int = 800, d: int = 12, seed: int = 0,
                   condition: float = 30.0) -> Dataset:
    """固定种子的合成回归集：``y = f(x) + 噪声``，``f`` 带一次项与几个交互项。

    ``condition`` 控制设计矩阵的条件数（大 = 病态 = 刚性），用来演示
    "条件数决定需要的迭代数、λ_max 决定能用的最大步长"。
    """
    rng = np.random.default_rng(seed)
    # 让奇异值按 condition 递减，制造病态
    U, _ = np.linalg.qr(rng.normal(size=(n, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.geomspace(1.0, 1.0 / max(condition, 1.0), d)
    X = (U * s) @ V.T
    w = rng.normal(size=d)
    y = X @ w + 0.5 * (X[:, 0] * X[:, 1]) + 0.1 * rng.normal(size=n)
    return Dataset(name=f"synthetic(n={n}, d={d}, cond={condition:g})", X=X, y=y,
                   feature_names=[f"x{i}" for i in range(d)],
                   note="合成回归数据，用于无外部文件时的可复现回退")
