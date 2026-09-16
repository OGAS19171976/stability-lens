"""stability_lens.adaptive — 自适应优化器的逐坐标步长界（Adam / RMSProp）。

自适应方法每步更新是

.. math:: \\theta \\leftarrow \\theta - \\eta\\, P\\, \\nabla L,
          \\qquad P = \\mathrm{diag}\\!\\left(\\frac{1}{\\sqrt{\\hat v_i} + \\epsilon}\\right)

其中 ``v̂`` 是梯度平方的滑动平均。线性化后迭代矩阵是 ``I − η P H``，于是

.. math:: \\eta_{\\max}^{\\text{Adam}} = \\frac{2}{\\lambda_{\\max}(P H)}

``P H`` 不对称，但和对称矩阵 ``P^{1/2} H P^{1/2}`` **相似**（``P`` 是对角正定），
所以只要在**预条件后的 Hessian-向量积**上跑幂迭代就够了：``w ↦ √p ⊙ (H (√p ⊙ w))``。

这就是"逐坐标"的含义：``v̂`` 小的坐标（少见特征、饱和单元）拿到大的 ``p_i``，
把有效曲率顶上去，全局 ``η`` 的上界随之被压紧 —— 这是 Adam 在稀疏特征上翻车的机制，
也是 ``ε`` 不能随便调小的原因（``p_i`` 上限是 ``1/ε``）。

**范围**：下面这套分析把更新当作 ``−η P g``，对应 ``β₁ = 0``（RMSProp）。
``β₁ > 0`` 时动量项还会引入非正规性与瞬态放大（见 ``stability_lens.transient``），
本模块不声称给出那一情形的紧界。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import datasets, models, spectral, training
from .datasets import Dataset
from .report import pad_left, pad_right

Array = np.ndarray

__all__ = [
    "AdaptiveStudy",
    "mlp_grad_sq_moments",
    "diagonal_preconditioner",
    "preconditioned_matvec",
    "preconditioned_lambda_max",
    "adam_eta_bound",
    "run_adaptive_study",
    "render_adaptive_study",
    "as_json",
]


# ===========================================================================
# v̂：梯度平方的二阶矩
# ===========================================================================
def mlp_grad_sq_moments(
    theta: Array, X: Array, y: Array, d: int, h: int, l2: float = 0.0
) -> Array:
    """逐参数的 ``v̂_i = E_x[(∂ℓ(x)/∂θ_i)²]``，``ℓ`` 是**单样本**损失（对角 Fisher）。

    约定必须说清楚，否则数字会差一个 ``√n``：全批梯度是
    ``∇L = (1/n)Σ_i ∂ℓ_i/∂θ``，而 Adam 的 ``v̂`` 建模成单样本损失梯度的二阶矩，
    所以这里**不**除以 ``n``。除以 ``n`` 会把预条件子整体放大 ``n`` 倍，
    进而把 ``η_max`` 压小 ``n`` 倍 —— 那是量纲错误，不是发现。

    对 MSE + 两层 tanh，单样本梯度是 ``(f_i − y_i)·∂f_i/∂θ``，平方后对样本取平均即可。
    """
    W1, b1, W2, b2 = models.unpack(theta, d, h)
    n = X.shape[0]
    a = np.tanh(X @ W1.T + b1)                       # (n,h)
    r = (a @ W2.T + b2).ravel() - y                  # (n,)  单样本残差
    s = (1.0 - a ** 2) * W2.ravel()[None, :]         # (n,h)  ∂f_i/∂z
    r2 = r ** 2

    g2_W1 = np.einsum("n,nh,nd->hd", r2, s ** 2, X ** 2) / n
    g2_b1 = (r2[:, None] * s ** 2).mean(axis=0)
    g2_W2 = (r2[:, None] * a ** 2).mean(axis=0).reshape(1, h)
    g2_b2 = np.array([float(r2.mean())])

    v = models.pack(g2_W1, g2_b1, g2_W2, g2_b2)
    if l2:
        # 正则项的梯度 l2·θ 与样本无关、且量级远小于数据项，这里只做一次粗加
        v = v + (l2 * theta) ** 2
    return v


def diagonal_preconditioner(v_hat: Array, eps: float = 1e-8) -> Array:
    """``p_i = 1 / (sqrt(v̂_i) + ε)``。"""
    if eps <= 0.0:
        raise ValueError("eps 必须为正（Adam 的分母保护项）")
    return 1.0 / (np.sqrt(np.maximum(v_hat, 0.0)) + eps)


# ===========================================================================
# 预条件后的谱
# ===========================================================================
def preconditioned_matvec(hvp: Callable[[Array], Array], sqrt_p: Array) -> Callable[[Array], Array]:
    """``w ↦ √p ⊙ (H (√p ⊙ w))``，即对称矩阵 ``P^{1/2} H P^{1/2}`` 的乘法。"""
    return lambda w: sqrt_p * hvp(sqrt_p * w)


def preconditioned_lambda_max(
    hvp: Callable[[Array], Array],
    p: Array,
    iters: int = 400,
    tol: float = 1e-12,
    seed: int = 0,
) -> spectral.PowerResult:
    """``λ_max(P^{1/2} H P^{1/2})``（= ``λ_max(P H)``）。"""
    sqrt_p = np.sqrt(p)
    return spectral.power_iteration(preconditioned_matvec(hvp, sqrt_p), p.size,
                                    iters=iters, tol=tol, seed=seed)


def adam_eta_bound(
    lam_max_precond: float,
    lam_max_plain: float | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """把两个 ``λ_max`` 换成步长上界与比值。"""
    out: dict[str, float] = {
        "eta_max_adam": 2.0 / lam_max_precond if lam_max_precond > 0 else math.inf,
    }
    if lam_max_plain is not None:
        out["eta_max_gd"] = 2.0 / lam_max_plain if lam_max_plain > 0 else math.inf
        out["ratio"] = (out["eta_max_adam"] / out["eta_max_gd"]
                        if math.isfinite(out["eta_max_gd"]) and out["eta_max_gd"] > 0 else math.nan)
    return out


# ===========================================================================
# 研究
# ===========================================================================
@dataclass
class AdaptiveStudy:
    dataset_name: str
    n: int
    d: int
    h: int
    p: int
    eps: float
    lam_max_plain: float
    lam_max_precond: float
    eta_max_gd: float
    eta_max_adam: float
    ratio: float
    vhat_min: float
    vhat_max: float
    p_max: float
    worst_index: int
    worst_effective_step: float
    measured: dict[str, Any] = field(default_factory=dict)
    linear_check: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def adam_is_tighter(self) -> bool:
        return self.ratio < 1.0


def run_adaptive_study(
    dataset: Dataset | None = None,
    h: int = 12,
    eps: float = 1e-8,
    l2: float = 1e-3,
    train_steps: int = 120,
    test_steps: int = 150,
    bisect_iters: int = 16,
    seed: int = 0,
) -> AdaptiveStudy:
    """在真实数据上给出 Adam 的逐坐标界，并用**实测边界**验证它。

    验证方式沿用前面的套路：把预条件后的梯度 ``P·∇L`` 当成新的"梯度"喂给固定步长迭代
    （这正是 RMSProp 的更新），于是理论上的 ``η_max = 2/λ_max(PH)`` 应当与实测边界吻合。

    另附一个**闭式线性对照**：线性模型上 ``H = XᵀX/n`` 与 ``v̂`` 都有闭式，
    所以预条件后的 ``λ_max`` 可以精确算出来，用来确认幂迭代那一步没算错。
    """
    ds = dataset if dataset is not None else datasets.load_order_amount()
    X, y, _ = datasets.standardize(ds.X, ds.y)
    n, d = X.shape

    theta0 = models.init_params(d, h, seed=seed)
    grad_fn = lambda th: models.mlp_loss_grad(th, X, y, d, h, l2=l2)     # noqa: E731
    hess_fn = lambda th: models.mlp_hessian(th, X, y, d, h, l2=l2)       # noqa: E731
    hvp = lambda th, v: models.mlp_hvp(th, X, y, d, h, v, l2=l2)         # noqa: E731

    theta_star = training.train_damped_newton(grad_fn, hess_fn, theta0,
                                             steps=train_steps, tol=1e-10).theta
    H = hess_fn(theta_star)
    lam_plain = spectral.spectral_summary(H)["lam_max"]

    v_hat = mlp_grad_sq_moments(theta_star, X, y, d, h, l2=l2)
    p = diagonal_preconditioner(v_hat, eps=eps)
    power = preconditioned_lambda_max(lambda v: hvp(theta_star, v), p,
                                      iters=400, tol=1e-12, seed=seed)
    lam_pre = power.lam

    bounds = adam_eta_bound(lam_pre, lam_plain, eps=eps)
    worst = int(np.argmax(p))

    # ---- 实测边界：把 P·∇L 当梯度（= RMSProp 的更新）----
    precond_grad = lambda th: (grad_fn(th)[0], p * grad_fn(th)[1])       # noqa: E731
    meas = training.boundary_from_prediction(
        precond_grad, theta_star, bounds["eta_max_adam"], beta=0.0,
        bisect_iters=bisect_iters, steps=test_steps, seed=seed, direction=power.vec)
    eta_meas = float(meas["eta_max"])

    # ---- 线性模型闭式对照 ----
    Hl = models.linear_hessian(X)
    th_l, *_ = np.linalg.lstsq(X, y, rcond=None)
    # 线性模型 f = Xθ，单样本梯度 = r_i x_i（同一个约定：不除以 n）
    r_l = X @ th_l - y
    v_l = ((r_l ** 2)[:, None] * (X ** 2)).mean(axis=0)
    p_l = diagonal_preconditioner(v_l, eps=eps)
    sqrt_p_l = np.sqrt(p_l)
    lam_pre_l = float(np.linalg.eigvalsh((sqrt_p_l[:, None] * Hl) * sqrt_p_l[None, :])[-1])
    power_l = preconditioned_lambda_max(
        lambda v: Hl @ v, p_l, iters=500, tol=1e-13, seed=seed)
    eta_pred_l = 2.0 / lam_pre_l
    meas_l = training.boundary_from_prediction(
        lambda th: (models.linear_loss_grad(th, X, y)[0],
                    p_l * models.linear_loss_grad(th, X, y)[1]),
        th_l, eta_pred_l, beta=0.0, bisect_iters=bisect_iters + 6,
        steps=test_steps, seed=seed, direction=power_l.vec)

    study = AdaptiveStudy(
        dataset_name=ds.name, n=n, d=d, h=h, p=int(theta_star.size), eps=eps,
        lam_max_plain=lam_plain, lam_max_precond=lam_pre,
        eta_max_gd=bounds["eta_max_gd"], eta_max_adam=bounds["eta_max_adam"],
        ratio=bounds["ratio"],
        vhat_min=float(v_hat.min()), vhat_max=float(v_hat.max()),
        p_max=float(p[worst]), worst_index=worst,
        worst_effective_step=float(bounds["eta_max_adam"] * p[worst]),
        measured={
            "eta_pred": bounds["eta_max_adam"],
            "eta_measured": eta_meas,
            "rel_err": (abs(eta_meas - bounds["eta_max_adam"]) / bounds["eta_max_adam"]
                        if math.isfinite(eta_meas) else float("nan")),
            "valid": bool(meas.get("valid")),
        },
        linear_check={
            "lam_max_precond_closed_form": lam_pre_l,
            "lam_max_precond_power": power_l.lam,
            "power_rel_err": abs(power_l.lam - lam_pre_l) / lam_pre_l,
            "eta_pred": eta_pred_l,
            "eta_measured": float(meas_l["eta_max"]),
            "rel_err": (abs(float(meas_l["eta_max"]) - eta_pred_l) / eta_pred_l
                        if math.isfinite(float(meas_l["eta_max"])) else float("nan")),
        },
    )

    study.notes.append(
        f"ε = {eps:g} 把预条件子上限钉在 1/ε = {1.0 / eps:.3g}；"
        f"本配置里 p_max = {study.p_max:.3g}（第 {worst} 号参数，v̂ = {v_hat[worst]:.3g}）。"
        "ε 调小会让这个上界变大，进而把 η_max 压得更紧、也更脆。")
    if study.adam_is_tighter:
        study.notes.append(
            f"Adam 的界比 SGD 紧 {1.0 / study.ratio:.3g} 倍："
            f"{study.eta_max_adam:.6g} vs {study.eta_max_gd:.6g}。"
            "逐坐标预条件把有效曲率顶上去了 —— 同一个 η 对 SGD 安全，对 Adam 未必。")
    else:
        study.notes.append(
            f"本配置下 Adam 的界反而比 SGD 松 {study.ratio:.3g} 倍"
            f"（{study.eta_max_adam:.6g} vs {study.eta_max_gd:.6g}）："
            "预条件恰好把最大的那几个曲率方向压下去了。")
    study.notes.append(
        "分析把更新当作 −ηPg，对应 β₁ = 0（RMSProp）。β₁ > 0 时动量项还会带来非正规性"
        "与瞬态放大（见 stability_lens.transient），那一情形下本模块不给紧界。")
    return study


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], left_cols: int = 1) -> str:
    cols = len(headers)
    w = [len(str(x)) for x in headers]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], len(str(r[i])))
    out = ["  ".join(pad_right(str(headers[i]), w[i]) for i in range(cols))]
    out.append("-" * (sum(w) + 2 * (cols - 1)))
    for r in rows:
        out.append("  ".join(pad_right(str(r[i]), w[i]) if i < left_cols
                             else pad_left(str(r[i]), w[i]) for i in range(cols)))
    return "\n".join(out)


def render_adaptive_study(s: AdaptiveStudy) -> str:
    L: list[str] = []
    L.append("-- Adam / RMSProp 的逐坐标步长界 " + "-" * 30)
    L.append(f"  数据集 {s.dataset_name}   n={s.n} d={s.d} h={s.h} p={s.p}   ε={s.eps:g}")
    L.append("")
    rows = [
        ["λ_max(H)（SGD 用）", f"{s.lam_max_plain:.8f}"],
        ["λ_max(P^{1/2} H P^{1/2})（Adam 用）", f"{s.lam_max_precond:.8f}"],
        ["η_max  SGD", f"{s.eta_max_gd:.6f}"],
        ["η_max  Adam/RMSProp", f"{s.eta_max_adam:.6f}"],
        ["比值 Adam/SGD", f"{s.ratio:.4f}" + ("  ← Adam 更紧" if s.adam_is_tighter else "  ← Adam 更松")],
        ["v̂ 范围", f"{s.vhat_min:.3g} … {s.vhat_max:.3g}"],
        ["p_max（第 %d 号参数）" % s.worst_index, f"{s.p_max:.3g}"],
        ["该坐标的有效步长 η·p", f"{s.worst_effective_step:.4g}"],
    ]
    L.append(_table(["量", "值"], rows))
    L.append("")
    L.append("  预测 vs 实测（把 P·∇L 当梯度迭代 = RMSProp）")
    m = s.measured
    L.append(_table(["量", "值"], [
        ["预测 η_max", f"{m['eta_pred']:.6f}"],
        ["实测边界", f"{m['eta_measured']:.6f}"],
        ["相对偏差", f"{m['rel_err'] * 100:.3f}%"],
    ]))
    lc = s.linear_check
    L.append("")
    L.append("  线性模型闭式对照（H 与 v̂ 都有闭式，不依赖差分）")
    L.append(_table(["量", "值"], [
        ["λ_max(P^{1/2}HP^{1/2}) 闭式", f"{lc['lam_max_precond_closed_form']:.8f}"],
        ["同上 幂迭代", f"{lc['lam_max_precond_power']:.8f}"],
        ["幂迭代相对误差", f"{lc['power_rel_err']:.2e}"],
        ["η_max 预测 / 实测", f"{lc['eta_pred']:.6f} / {lc['eta_measured']:.6f}"],
        ["相对偏差", f"{lc['rel_err'] * 100:.3f}%"],
    ]))
    L.append("")
    for note in s.notes:
        L.append(f"  · {note}")
    return "\n".join(L)


def as_json(s: AdaptiveStudy) -> dict[str, Any]:
    def conv(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v

    return {
        "dataset": s.dataset_name,
        "n": s.n, "d": s.d, "h": s.h, "params": s.p, "eps": s.eps,
        "lambda_max_plain": conv(s.lam_max_plain),
        "lambda_max_preconditioned": conv(s.lam_max_precond),
        "eta_max_gd": conv(s.eta_max_gd),
        "eta_max_adam": conv(s.eta_max_adam),
        "ratio_adam_over_gd": conv(s.ratio),
        "vhat_min": conv(s.vhat_min), "vhat_max": conv(s.vhat_max),
        "p_max": conv(s.p_max), "worst_index": s.worst_index,
        "measured": {k: conv(v) for k, v in s.measured.items()},
        "linear_check": {k: conv(v) for k, v in s.linear_check.items()},
        "notes": s.notes,
    }
