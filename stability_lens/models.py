"""stability_lens.models — 纯 numpy 的小模型，供「预测 η_max」实验使用。

只依赖 numpy（可选依赖，不在运行时强制要求：`core`/`diagnose` 仍然是零依赖的）。

提供两层 MLP 与线性模型，且每个模型都给三样东西：

1. **解析梯度** —— 与有限差分对拍（见 `numerical_gradient`）；
2. **Hessian-向量积** —— 用中心差分的解析梯度实现（``eps`` 可调），
   这样不必手推二阶反向传播；
3. **精确 Hessian 装配** —— 逐列差分，用来给幂迭代提供真值。

线性模型还提供**闭式 Hessian** ``XᵀX/n``（与 θ 无关），
它是不依赖任何差分的独立真值 —— 幂迭代 + HVP 的整条流水线都要先在这一关上通过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray

__all__ = [
    "init_params",
    "unpack",
    "pack",
    "param_count",
    "mlp_forward",
    "mlp_loss_grad",
    "mlp_hvp",
    "mlp_hessian",
    "linear_loss_grad",
    "linear_hessian",
    "numerical_gradient",
    "gradient_check",
]


# ===========================================================================
# 参数打包 / 解包
# ===========================================================================
def pack(W1: Array, b1: Array, W2: Array, b2: Array) -> Array:
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def unpack(theta: Array, d: int, h: int) -> tuple[Array, Array, Array, Array]:
    """θ 的内存布局：``[W1(h,d) | b1(h) | W2(1,h) | b2(1)]``。"""
    i = 0
    W1 = theta[i:i + h * d].reshape(h, d); i += h * d
    b1 = theta[i:i + h]; i += h
    W2 = theta[i:i + h].reshape(1, h); i += h
    b2 = theta[i:i + 1]
    return W1, b1, W2, b2


def param_count(d: int, h: int) -> int:
    return h * d + h + h + 1


def init_params(d: int, h: int, seed: int = 0, scale: float | None = None) -> Array:
    """Xavier 风格初始化；``b2`` 的初值取 ``y`` 均值可显著改善条件数（调用方决定）。"""
    rng = np.random.default_rng(seed)
    s = scale if scale is not None else 1.0 / np.sqrt(d)
    W1 = rng.normal(0.0, s, size=(h, d))
    b1 = np.zeros(h)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(h), size=(1, h))
    b2 = np.zeros(1)
    return pack(W1, b1, W2, b2)


# ===========================================================================
# 两层 MLP：f(x) = W2 · tanh(W1 x + b1) + b2，损失 = MSE
# ===========================================================================
def mlp_forward(theta: Array, X: Array, d: int, h: int) -> tuple[Array, Array]:
    W1, b1, W2, b2 = unpack(theta, d, h)
    a = np.tanh(X @ W1.T + b1)          # (n, h)
    f = (a @ W2.T + b2).ravel()          # (n,)
    return f, a


def mlp_loss_grad(theta: Array, X: Array, y: Array, d: int, h: int,
                  l2: float = 0.0) -> tuple[float, Array]:
    """``L = 1/(2n) Σ (f_i − y_i)² + l2/2·‖θ‖²`` 及其解析梯度。

    ``l2`` 默认 0；训练到极小后若 Hessian 仍有零/负特征值（不可辨识方向），
    加一个极小的 ``l2``（如 1e-6）可以让极小严格正定，便于讨论稳定域。
    """
    W1, b1, W2, b2 = unpack(theta, d, h)
    n = X.shape[0]
    a = np.tanh(X @ W1.T + b1)
    r = (a @ W2.T + b2).ravel() - y
    loss = float(r @ r / (2.0 * n) + 0.5 * l2 * float(theta @ theta))

    g = r / n                                    # (n,)
    gW2 = (g[:, None] * a).sum(axis=0, keepdims=True)   # (1,h)
    gb2 = np.array([g.sum()])
    dz = g[:, None] * W2.ravel()[None, :] * (1.0 - a ** 2)   # (n,h)
    gW1 = dz.T @ X                               # (h,d)
    gb1 = dz.sum(axis=0)
    return loss, pack(gW1, gb1, gW2, gb2) + l2 * theta


def mlp_loss(theta: Array, X: Array, y: Array, d: int, h: int, l2: float = 0.0) -> float:
    return mlp_loss_grad(theta, X, y, d, h, l2=l2)[0]


def mlp_hvp(theta: Array, X: Array, y: Array, d: int, h: int, v: Array,
            eps: float = 1e-5, l2: float = 0.0) -> Array:
    """Hessian-向量积，用中心差分的解析梯度实现：``(g(θ+εv) − g(θ−εv)) / 2ε``。

    截断误差 ``O(ε²·|∇³L|)``，``ε = 1e-5`` 时相对误差约 1e-9（tanh 光滑网络）。
    """
    _, gp = mlp_loss_grad(theta + eps * v, X, y, d, h, l2=l2)
    _, gm = mlp_loss_grad(theta - eps * v, X, y, d, h, l2=l2)
    return (gp - gm) / (2.0 * eps)


def mlp_hessian(theta: Array, X: Array, y: Array, d: int, h: int,
                eps: float = 1e-5, l2: float = 0.0) -> Array:
    """逐列差分装配精确 Hessian；返回对称化后的 ``(p, p)`` 矩阵。"""
    p = param_count(d, h)
    H = np.empty((p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = 1.0
        H[:, j] = mlp_hvp(theta, X, y, d, h, e, eps=eps, l2=l2)
    return 0.5 * (H + H.T)


# ===========================================================================
# 线性模型：闭式 Hessian 作独立真值
# ===========================================================================
def linear_loss_grad(theta: Array, X: Array, y: Array) -> tuple[float, Array]:
    n = X.shape[0]
    r = X @ theta - y
    return float(r @ r / (2.0 * n)), X.T @ r / n


def linear_hessian(X: Array) -> Array:
    """``∇²L = XᵀX/n`` —— 与 θ 无关的闭式真值，不含任何差分。"""
    return X.T @ X / X.shape[0]


# ===========================================================================
# 梯度自检
# ===========================================================================
def numerical_gradient(f: Callable[[Array], float], theta: Array, eps: float = 1e-6) -> Array:
    """中心差分梯度（只用来对拍，不参与生产路径）。"""
    g = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        g[j] = (f(theta + e) - f(theta - e)) / (2.0 * eps)
    return g


def gradient_check(f: Callable[[Array], float], analytic: Array, theta: Array,
                   eps: float = 1e-6) -> float:
    """返回解析梯度与差分梯度的最大相对误差（按整体范数归一）。"""
    num = numerical_gradient(f, theta, eps=eps)
    denom = max(float(np.linalg.norm(num)), float(np.linalg.norm(analytic)), 1e-300)
    return float(np.linalg.norm(num - analytic) / denom)
