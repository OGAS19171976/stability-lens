"""stability_lens.torch_backend — 可选 PyTorch 后端。

**torch 不是依赖**：本模块在没有 torch 的环境里也能 import，只有真正调用函数时才要求它。

接入方式刻意不假设模型结构 —— 只要你能给出"重新算一遍损失的闭包"和参数列表：

.. code-block:: python

    loss_fn = lambda: criterion(model(x), y)      # 标量、可微；每次调用重建计算图
    params = [p for p in model.parameters() if p.requires_grad]
    out = eta_max_torch(loss_fn, params, beta=0.0)
    print(out["lambda_max"], out["eta_max"])

Hessian-向量积用**双重反向传播**实现：

.. math:: g = \\nabla L(\\theta)\\ (\\text{create\\_graph}),\\qquad
          Hv = \\nabla_\\theta\\,(g \\cdot v)

拿到 ``λ_max`` 之后，下游全部复用前面的模块：``η_max = 2(1+β)/λ_max``、
非正规性的账（动量伴随矩阵的瞬态放大）等等，都不需要重写一遍。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "has_torch",
    "TorchPowerResult",
    "make_hvp",
    "lambda_max_torch",
    "eta_max_torch",
    "diagnose_torch",
]

INSTALL_HINT = (
    "这个功能需要 PyTorch（可选依赖）：\n"
    "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    "其余功能（check / sweep / selftest / predict / monitor / transient / adaptive）都不需要它。"
)


def has_torch() -> bool:
    """torch 是否可用（供测试与文档做 skip 判断）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 依赖环境
        raise ImportError(f"{INSTALL_HINT}\n原始错误：{exc}") from exc
    return torch


@dataclass
class TorchPowerResult:
    lam: float
    iters: int
    residual: float
    dim: int
    trace: list[float] = field(default_factory=list)
    converged: bool = False


def make_hvp(loss_fn: Callable[[], Any], params: Sequence[Any]) -> Callable[[Any], Any]:
    """返回 ``v -> H(θ)·v``（torch 张量进出，设备/精度跟随参数）。

    ``loss_fn`` 必须每次调用都重新前向一次（这样才有一张新图可求导），
    并且 ``params`` 必须是叶子张量。
    """
    torch = _require_torch()
    params = list(params)
    if not params:
        raise ValueError("params 不能为空")

    def hvp(v):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        if flat.numel() != v.numel():
            raise ValueError(f"v 的长度 {v.numel()} 与参数量 {flat.numel()} 不一致")
        dot = (flat * v).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv]).detach()

    return hvp


def lambda_max_torch(
    loss_fn: Callable[[], Any],
    params: Sequence[Any],
    iters: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
) -> TorchPowerResult:
    """在真实 PyTorch 模型上估 ``λ_max(H)``（幂迭代，只吃 HVP）。"""
    torch = _require_torch()
    params = list(params)
    if not params:
        raise ValueError("params 不能为空")
    dim = int(sum(p.numel() for p in params))
    dtype, device = params[0].dtype, params[0].device

    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, dtype=dtype, generator=gen).to(device)
    v = v / v.norm()

    hvp = make_hvp(loss_fn, params)
    trace: list[float] = []
    lam, residual, k = 0.0, float("inf"), 0
    for k in range(1, iters + 1):
        w = hvp(v)
        lam = float(v @ w)
        residual = float((w - lam * v).norm())
        trace.append(lam)
        nw = float(w.norm())
        if nw == 0.0:
            return TorchPowerResult(lam=0.0, iters=k, residual=0.0, dim=dim,
                                    trace=trace, converged=True)
        v = w / nw
        if residual <= tol * max(1.0, abs(lam)):
            break
    lam = float(v @ hvp(v))
    return TorchPowerResult(lam=lam, iters=k, residual=residual, dim=dim, trace=trace,
                            converged=residual <= tol * max(1.0, abs(lam)))


def eta_max_torch(
    loss_fn: Callable[[], Any],
    params: Sequence[Any],
    beta: float = 0.0,
    iters: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
) -> dict[str, float]:
    """``η_max = 2(1+β)/λ_max``（真实模型上）。"""
    res = lambda_max_torch(loss_fn, params, iters=iters, tol=tol, seed=seed)
    return {
        "lambda_max": res.lam,
        "eta_max": 2.0 * (1.0 + beta) / res.lam if res.lam > 0 else math.inf,
        "iters": res.iters,
        "residual": res.residual,
        "dim": float(res.dim),
    }


def diagnose_torch(
    loss_fn: Callable[[], Any],
    params: Sequence[Any],
    eta: float | None = None,
    beta: float = 0.0,
    iters: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
):
    """在真实模型上体检：估 ``λ_max`` → 复用 :func:`diagnose_heavy_ball` 出结论。

    返回的是 ``stability_lens.diagnose.Diagnosis``，
    因此 ``η_max``、判定、动量瞬态放大等下游全都自动就位。
    """
    from .diagnose import diagnose_heavy_ball

    out = eta_max_torch(loss_fn, params, beta=beta, iters=iters, tol=tol, seed=seed)
    lam = out["lambda_max"]
    if lam <= 0.0:
        raise ValueError(f"λ_max = {lam} ≤ 0：当前点不是正曲率区域，判据不适用")
    d = diagnose_heavy_ball(eta=eta, beta=beta, lambda_max=lam)
    d.title = f"PyTorch 模型（p={int(out['dim'])}，λ_max 由双重反向传播 + 幂迭代估出）"
    d.rows.insert(1, ("λ_max 估计", f"{lam:.8g}（{out['iters']} 步，残差 {out['residual']:.1e}）"))
    d.extra["torch"] = out
    return d
