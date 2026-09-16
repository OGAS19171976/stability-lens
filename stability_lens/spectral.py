"""stability_lens.spectral — 对称算子的最大特征值估计。

微调场景里 Hessian 通常是 ``10^7 ~ 10^9`` 维、无法显式构造，
但 Hessian-向量积可以拿到。于是 ``λ_max`` 的标准做法是幂迭代。

本模块只依赖 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
MatVec = Callable[[Array], Array]

__all__ = ["PowerResult", "power_iteration", "spectral_summary"]


@dataclass
class PowerResult:
    """幂迭代结果。

    Attributes:
        lam: 最大特征值的估计（末次 Rayleigh 商）。
        iters: 实际迭代步数。
        residual: ``‖Hv − λv‖``（v 已归一化），衡量是否收敛到特征向量。
        trace: 每步的 ``λ`` 估计，画收敛曲线用。
        converged: 残差是否小于容差。
    """

    lam: float
    iters: int
    residual: float
    trace: list[float] = field(default_factory=list)
    converged: bool = False
    vec: Array | None = None


def power_iteration(
    matvec: MatVec,
    dim: int,
    iters: int = 300,
    tol: float = 1e-10,
    seed: int = 0,
    v0: Array | None = None,
    keep_trace: bool = True,
) -> PowerResult:
    """幂迭代估 ``λ_max``（要求算子对称）。

    收敛速率是 ``(λ2/λ1)^k`` —— Hessian 病态时很慢，所以默认给 300 步并把
    残差一起报出来，方便判断"这个 λ_max 到底可不可信"。

    Args:
        matvec: ``v -> H v``。
        dim: 参数维度。
        tol: 残差容差 ``‖Hv − λv‖ <= tol·max(1,|λ|)`` 时提前停。
        v0: 初始向量；默认用固定种子抽正态向量，保证可复现。
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(v0, dtype=float) if v0 is not None else rng.normal(size=dim)
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise ValueError("初始向量不能是零向量")
    v = v / nv
    trace: list[float] = []
    lam = 0.0
    residual = float("inf")
    k = 0
    for k in range(1, iters + 1):
        w = matvec(v)
        lam = float(v @ w)
        residual = float(np.linalg.norm(w - lam * v))
        nw = float(np.linalg.norm(w))
        if nw == 0.0:                     # H = 0
            trace.append(0.0)
            return PowerResult(lam=0.0, iters=k, residual=0.0, trace=trace,
                               converged=True, vec=v)
        v = w / nw
        if keep_trace:
            trace.append(lam)
        if residual <= tol * max(1.0, abs(lam)):
            break
    lam = float(v @ matvec(v))            # 用最终的 Rayleigh 商收尾
    return PowerResult(lam=lam, iters=k, residual=residual, trace=trace,
                       converged=residual <= tol * max(1.0, abs(lam)), vec=v)


def spectral_summary(H: Array) -> dict[str, float]:
    """显式矩阵的谱摘要（用于给幂迭代提供真值）。"""
    w = np.linalg.eigvalsh(0.5 * (H + H.T))
    return {
        "lam_min": float(w[0]),
        "lam_max": float(w[-1]),
        "condition": float(w[-1] / w[0]) if w[0] > 0 else float("inf"),
        "trace": float(w.sum()),
    }
