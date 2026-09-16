"""stability_lens.training — 训练与"实测稳定边界"。

实验协议（这是本模块存在的意义）：

1. 用很小的 ``η`` 把模型训到收敛，得到一个**局部极小** ``θ*``；
2. 在 ``θ*`` 附近加一个小扰动（``‖δ‖ ≈ 1e-3``），用它跑固定步长的迭代；
3. 观察 ``‖θ_k − θ*‖`` 是衰减还是增长 —— 这就是该 ``η`` 下的**实测**稳定性；
4. 对 ``η`` 做倍增定界 + 二分，得到实测稳定边界，与解析预测 ``2(1+β)/λ_max`` 对照。

在 ``θ*`` 附近加扰动而不是从随机初值出发，是因为线性化只在局部成立 ——
这样才能公平地检验 ``η_max = 2(1+β)/λ_max`` 这个**局部**判据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
GradFn = Callable[[Array], tuple[float, Array]]

__all__ = [
    "TrainResult",
    "train",
    "train_linesearch",
    "train_damped_newton",
    "decay_test",
    "find_boundary_empirical",
    "boundary_from_prediction",
]


@dataclass
class TrainResult:
    theta: Array
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    steps_run: int = 0
    diverged: bool = False

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def final_grad_norm(self) -> float:
        return self.grad_norms[-1] if self.grad_norms else float("nan")


def train(
    grad_fn: GradFn,
    theta0: Array,
    eta: float,
    steps: int,
    beta: float = 0.0,
    blowup: float = 1e12,
) -> TrainResult:
    """``θ_{k+1} = θ_k − η·∇L(θ_k) + β(θ_k − θ_{k-1})``。

    ``β = 0`` 是梯度下降；``β > 0`` 是 heavy-ball。
    """
    th = np.array(theta0, dtype=float)
    prev = th.copy()
    res = TrainResult(theta=th)
    for _ in range(steps):
        loss, g = grad_fn(th)
        res.losses.append(float(loss))
        res.grad_norms.append(float(np.linalg.norm(g)))
        nxt = th - eta * g + beta * (th - prev)
        prev, th = th, nxt
        res.steps_run += 1
        if (not np.all(np.isfinite(th))) or float(np.linalg.norm(th)) > blowup:
            res.diverged = True
            break
    res.theta = th
    return res


def train_linesearch(
    grad_fn: GradFn,
    theta0: Array,
    steps: int = 2000,
    init_eta: float = 0.1,
    tol: float = 1e-10,
    shrink: float = 0.5,
    grow: float = 1.5,
    c1: float = 1e-4,
    max_backtrack: int = 40,
) -> TrainResult:
    """回溯线搜索（Armijo）梯度下降，用来**找** θ*。

    这一步的优化器不是研究对象 —— 我们要的是"一个局部极小"，用什么方法到达都行。
    固定步长的稳定性分析在下一步才登场。没有用 scipy，因此包仍然只依赖 numpy。
    """
    th = np.array(theta0, dtype=float)
    loss, g = grad_fn(th)
    res = TrainResult(theta=th)
    res.losses.append(float(loss))
    res.grad_norms.append(float(np.linalg.norm(g)))
    eta = init_eta
    for _ in range(steps):
        gnorm2 = float(g @ g)
        if float(np.linalg.norm(g)) <= tol:
            break
        t = eta
        accepted = False
        for _ in range(max_backtrack):
            cand = th - t * g
            l_new, g_new = grad_fn(cand)
            if l_new <= loss - c1 * t * gnorm2:
                th, loss, g = cand, l_new, g_new
                eta = min(t * grow, 1e6)
                accepted = True
                break
            t *= shrink
        if not accepted:
            break                      # 步长已缩到机器精度，认为已收敛
        res.losses.append(float(loss))
        res.grad_norms.append(float(np.linalg.norm(g)))
        res.steps_run += 1
    res.theta = th
    return res


def _cholesky_solve(A: Array, b: Array) -> Array | None:
    """A 正定时解 ``A x = b``，否则返回 None（用来判定是否还需要加阻尼）。"""
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        return None


def train_damped_newton(
    grad_fn: GradFn,
    hess_fn: Callable[[Array], Array],
    theta0: Array,
    steps: int = 60,
    tol: float = 1e-10,
    mu0: float = 1e-6,
    c1: float = 1e-4,
    max_ls: int = 40,
) -> TrainResult:
    """阻尼牛顿法找 θ*：Cholesky 判定正定 + Armijo 线搜索。

    坑点（实测踩过）：只用"``l_new < loss`` 就接受、然后 μ×0.3"的朴素阻尼，
    一旦 μ 被推大就再也不会降下来 —— 每步退化成极小的梯度步，60 步也到不了极小。
    正确做法是：

    1. 用 **Cholesky 是否成功**判断 ``H + μI`` 是否正定，逐步加大 μ 直到正定；
    2. 在该方向上做 **Armijo 线搜索**，保证每步有足够下降（而不是"只要下降一点点"）；
    3. 接受之后才放松 μ。

    这一步的优化器不是研究对象，它只负责把模型送到一个局部极小。
    """
    th = np.array(theta0, dtype=float)
    p = th.size
    res = TrainResult(theta=th)
    loss, g = grad_fn(th)
    res.losses.append(float(loss))
    res.grad_norms.append(float(np.linalg.norm(g)))
    mu = mu0
    for _ in range(steps):
        if float(np.linalg.norm(g)) <= tol:
            break
        H = hess_fn(th)
        d = None
        for _ in range(60):
            d = _cholesky_solve(H + mu * np.eye(p), -g)
            if d is not None:
                break
            mu = max(2.0 * mu, 1e-10)
        gd = float(g @ d) if d is not None else 0.0
        if d is None or gd >= 0.0:            # 兜底：退回最速下降方向
            d = -g
            gd = -float(g @ g)
            mu = max(mu, 1e-4)
        t = 1.0
        accepted = False
        for _ in range(max_ls):
            cand = th + t * d
            l_new, g_new = grad_fn(cand)
            if np.isfinite(l_new) and l_new <= loss + c1 * t * gd:
                th, loss, g = cand, l_new, g_new
                mu = max(mu * 0.25, 1e-12)
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        res.losses.append(float(loss))
        res.grad_norms.append(float(np.linalg.norm(g)))
        res.steps_run += 1
    res.theta = th
    return res


def decay_test(
    grad_fn: GradFn,
    theta_star: Array,
    eta: float,
    beta: float = 0.0,
    steps: int = 150,
    seed: int = 0,
    growth_tol: float = 1.0,
    blowup: float = 1.0,
    direction: Array | None = None,
    symmetric: bool = True,
    delta_scale: float = 1e-6,
    noise_rel: float = 1e-13,
) -> dict[str, float | bool]:
    """在 ``θ*`` 附近加小扰动，测该 ``η`` 下扰动是衰减还是增长。

    三个必须同时做对的细节（都是实测踩出来的）：

    1. **对称扰动** ``θ* ± δ``，观测量取 ``‖θ⁺ − θ⁻‖/2``：θ* 处的残余梯度会形成
       常数漂移，扰动衰减到同一量级时它就主导读数（实测把增长率从 0.600 抬到 0.671）。
       对称差分把漂移与所有偶阶非线性项一起消掉。
    2. **噪声地板早停**：扰动衰减到 ``1e-13`` 量级后就只剩浮点噪声，读数变成常数、
       斜率归零 —— 若不早停，"明确稳定"会被判成"临界"。一旦触底就直接判稳定。
    3. **斜率而非端点比值**：动量法的距离是衰减正弦，端点比值会被相位污染（实测差 40%+），
       后半段 log 距离的最小二乘斜率能把振荡平均掉。

    Args:
        delta_scale: 扰动幅度（默认 ``1e-6``）。要小到非线性可忽略，又要大到有足够动态范围。
        direction: 扰动方向；传幂迭代给出的主特征向量可对准最危险的模态。
        blowup: 位移超过它就认为已经发散并提前结束。

    Returns:
        dict：``decayed``（是否稳定）、``rate``（每步几何倍率，``>1`` 发散）、
        ``rate_tail``（端点比值，仅作诊断）、``e_ratio``、``hit``
        （``"diverged"`` / ``"noise-floor"`` / ``"window"``）。
    """
    if direction is not None:
        d = np.asarray(direction, dtype=float)
        if d.shape != theta_star.shape:
            raise ValueError("direction 的形状必须与 theta_star 一致")
    else:
        d = np.random.default_rng(seed).normal(size=theta_star.size)
    nrm = float(np.linalg.norm(d))
    if nrm == 0.0:
        d = np.ones_like(theta_star)
        nrm = float(np.linalg.norm(d))
    step = delta_scale * (d / nrm)
    floor = noise_rel * max(1.0, float(np.linalg.norm(theta_star)))
    th_p = theta_star + step
    prev_p = th_p.copy()
    if symmetric:
        th_m = theta_star - step
        prev_m = th_m.copy()
    dists = [float(np.linalg.norm(th_p - th_m) / 2.0) if symmetric else float(delta_scale)]
    diverged = False
    hit_floor = False
    for _ in range(steps):
        _, g_p = grad_fn(th_p)
        nxt_p = th_p - eta * g_p + beta * (th_p - prev_p)
        prev_p, th_p = th_p, nxt_p
        if symmetric:
            _, g_m = grad_fn(th_m)
            nxt_m = th_m - eta * g_m + beta * (th_m - prev_m)
            prev_m, th_m = th_m, nxt_m
            e = float(np.linalg.norm(th_p - th_m) / 2.0)
            # 逃逸检查：只看两地之差会漏掉"两条轨迹一起跑远、差值反而饱和"的情形
            esc = max(float(np.linalg.norm(th_p - theta_star)),
                      float(np.linalg.norm(th_m - theta_star)))
        else:
            e = float(np.linalg.norm(th_p - theta_star))
            esc = e
        if (not np.isfinite(e)) or (not np.isfinite(esc)) or esc > blowup:
            dists.append(e if np.isfinite(e) else float("inf"))
            diverged = True
            break
        if e < floor and len(dists) >= 3:
            dists.append(e)
            hit_floor = True
            break
        dists.append(e)
    e0, eK = dists[0], dists[-1]
    k = len(dists) - 1
    arr = np.asarray(dists, dtype=float)
    ks = np.arange(arr.size, dtype=float)
    half = arr.size // 2
    # 斜率必须取**前半段**：越界之后轨迹会逃到一个更平坦的区域，差值随之饱和，
    # 后半段斜率回落到 ≈1（实测在 ρ_theory = 1.535 时给出 0.9975，把发散判成稳定）。
    # 前半段还在线性区里，反映的才是线性化判据真正预言的增长率。
    cut = max(3, half)
    seg_y = arr[:cut]
    seg_x = ks[:cut]
    mask = np.isfinite(seg_y) & (seg_y > 0.0)
    if mask.sum() >= 3:
        rate = float(np.exp(np.polyfit(seg_x[mask], np.log(seg_y[mask]), 1)[0]))
    elif k <= 0:
        rate = float("inf")
    elif e0 <= 0.0 or eK <= 0.0:
        rate = 0.0 if eK == 0.0 else float("inf")
    else:
        rate = float(np.exp((np.log(eK) - np.log(e0)) / k))
    # 端点比值（保留作诊断）：**不要**用它做判据，振荡与饱和都会让它失真
    if half > 0 and dists[half] > 0.0 and eK > 0.0:
        rate_tail = float(np.exp((np.log(eK) - np.log(dists[half])) / (k - half)))
    else:
        rate_tail = rate
    if diverged:
        decayed = False
        hit = "diverged"
    elif hit_floor:
        # 扰动衰减到浮点噪声之下 —— 这是"明确稳定"，不是"临界"
        decayed = True
        hit = "noise-floor"
    else:
        decayed = rate < growth_tol
        hit = "window"
    return {"decayed": decayed, "rate": rate, "rate_tail": rate_tail,
            "e_ratio": float(eK / e0) if e0 > 0 else float("inf"),
            "steps_run": k, "diverged": diverged, "hit": hit}


def find_boundary_empirical(
    grad_fn: GradFn,
    theta_star: Array,
    beta: float = 0.0,
    eta_start: float = 1e-6,
    max_doublings: int = 24,
    bisect_iters: int = 16,
    steps: int = 150,
    delta_scale: float = 1e-8,
    seed: int = 0,
    direction: Array | None = None,
) -> dict[str, float | bool]:
    """倍增定界 + 二分，实测该 ``β`` 下最大的稳定步长。

    **不假设**任何理论值：从很小的 ``η``（必然稳定）开始倍增，直到不稳定，
    再在区间内二分。返回 ``{eta_max, bracket_lo, bracket_hi, valid}``。
    """
    def stable(eta: float) -> bool:
        return bool(decay_test(grad_fn, theta_star, eta, beta=beta, steps=steps,
                               delta_scale=delta_scale, seed=seed,
                               direction=direction)["decayed"])

    lo = eta_start
    if not stable(lo):
        return {"eta_max": float("nan"), "bracket_lo": lo, "bracket_hi": lo, "valid": False}
    hi = lo
    ok = False
    for _ in range(max_doublings):
        hi = lo * 2.0
        if not stable(hi):
            ok = True
            break
        lo = hi
    if not ok:
        return {"eta_max": float("inf"), "bracket_lo": lo, "bracket_hi": hi, "valid": False}

    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return {"eta_max": 0.5 * (lo + hi), "bracket_lo": lo, "bracket_hi": hi, "valid": True}


def boundary_from_prediction(
    grad_fn: GradFn,
    theta_star: Array,
    eta_pred: float,
    beta: float = 0.0,
    span: float = 4.0,
    bisect_iters: int = 18,
    steps: int = 150,
    delta_scale: float = 1e-6,
    seed: int = 0,
    direction: Array | None = None,
) -> dict[str, float | bool]:
    """当已经知道预测值时，在 ``[η_pred/span, η_pred*span]`` 内二分求解。

    比 :func:`find_boundary_empirical` 省一大半前向评估；代价是**假设**了预测值
    落在这个区间里（返回值里的 ``bracket_ok`` 会说明这个假设是否成立）。
    """
    lo, hi = eta_pred / span, eta_pred * span

    def stable(eta: float) -> bool:
        return bool(decay_test(grad_fn, theta_star, eta, beta=beta, steps=steps,
                               delta_scale=delta_scale, seed=seed,
                               direction=direction)["decayed"])

    ok_lo, ok_hi = stable(lo), stable(hi)
    if not (ok_lo and not ok_hi):
        return {"eta_max": float("nan"), "bracket_lo": lo, "bracket_hi": hi,
                "valid": False, "bracket_ok": False}
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return {"eta_max": 0.5 * (lo + hi), "bracket_lo": lo, "bracket_hi": hi,
            "valid": True, "bracket_ok": True}
