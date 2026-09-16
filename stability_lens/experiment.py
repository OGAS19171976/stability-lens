"""stability_lens.experiment — 端到端研究：把 λ_max 变成 η_max，再和实测边界对拍。

流程：

1. 真实宽表 → 标准化 → 两层 MLP；
2. 小步长训到局部极小 θ*（并**验证** λ_min(H) > 0，否则线性化判据不适用）；
3. 幂迭代（只吃 HVP）估 λ_max，同时装配精确 Hessian 作真值；
4. 预测 η_max = 2(1+β)/λ_max；
5. 在 θ* 附近加小扰动，倍增定界 + 二分**实测**稳定边界；
6. 线性模型做一次闭式对照 —— 二次目标下实测边界应当**精确**等于 2/λ_max。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import datasets, models, monitor, spectral, training
from .core import heavy_ball_rho
from .datasets import Dataset
from .report import display_width, pad_left, pad_right

Array = np.ndarray

__all__ = ["BoundaryRow", "StudyResult", "run_study", "render_study", "as_json",
           "MonitorRow", "MonitorStudy", "run_monitor_study", "render_monitor_study",
           "monitor_as_json"]


@dataclass
class BoundaryRow:
    label: str
    beta: float
    eta_pred: float
    eta_measured: float
    rel_err: float
    valid: bool = True
    # 增长率对照：在 η_in 处的实测几何增长率 vs 理论 ρ
    eta_in: float = float("nan")
    rate_measured: float = float("nan")
    rate_theory: float = float("nan")
    rate_rel_err: float = float("nan")


@dataclass
class StudyResult:
    dataset_name: str
    dataset_note: str
    n: int
    d: int
    h: int
    p: int
    grad_check: float
    hessian_symmetry: float
    l2: float
    train_steps: int
    train_loss: float
    train_grad_norm: float
    lam_min: float
    lam_max_exact: float
    lam_max_power: float
    power_iters: int
    power_residual: float
    rows: list[BoundaryRow] = field(default_factory=list)
    linear: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_local_min(self) -> bool:
        return self.lam_min > 0.0


def run_study(
    dataset: Dataset | None = None,
    h: int = 12,
    betas: Sequence[float] = (0.0, 0.5, 0.9),
    seed: int = 0,
    l2: float = 1e-3,
    train_steps: int = 120,
    test_steps: int = 150,
    bisect_iters: int = 14,
    rate_frac: float = 0.8,
    rate_steps: int = 20,
    with_linear_reference: bool = True,
) -> StudyResult:
    """跑完整套研究。默认参数下耗时约数秒。

    Args:
        l2: 训练时加的 L2（默认 1e-3，即 weight decay）。它保证 θ* 是**严格**局部极小
            （λ_min(H) > 0）：实测下来 ``l2 = 1e-6`` 时噪声方向上的平坦谱会让梯度
            迟迟不收敛、权重继续膨胀，λ_max 一路从 3.7 涨到 22.8，线性化判据就失去意义了。
        train_steps: 阻尼牛顿的步数上限。这一步只为**找到** θ*，不是研究对象。
        rate_frac: 在 ``η_in = rate_frac · η_max`` 处额外做一次增长率对照。
    """
    ds = dataset if dataset is not None else datasets.load_order_amount()
    X, y, _ = datasets.standardize(ds.X, ds.y)
    n, d = X.shape

    theta0 = models.init_params(d, h, seed=seed)
    grad_fn = lambda th: models.mlp_loss_grad(th, X, y, d, h, l2=l2)  # noqa: E731

    # ---- 梯度自检（解析 vs 中心差分）----
    rng = np.random.default_rng(seed + 1)
    probe = theta0 + 0.1 * rng.normal(size=theta0.size)
    _, g_probe = grad_fn(probe)
    gcheck = models.gradient_check(lambda t: models.mlp_loss(t, X, y, d, h, l2=l2),
                                   g_probe, probe)

    # ---- 用阻尼牛顿法把模型送到局部极小 θ* ----
    hess_fn = lambda th: models.mlp_hessian(th, X, y, d, h, l2=l2)  # noqa: E731
    trained = training.train_damped_newton(grad_fn, hess_fn, theta0,
                                           steps=train_steps, tol=1e-10)
    theta_star = trained.theta

    # ---- 精确 Hessian（真值）与幂迭代（生产路径）----
    H = models.mlp_hessian(theta_star, X, y, d, h, l2=l2)
    sym = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
    info = spectral.spectral_summary(H)

    p = models.param_count(d, h)
    hvp = lambda v: models.mlp_hvp(theta_star, X, y, d, h, v, l2=l2)  # noqa: E731
    power = spectral.power_iteration(hvp, p, iters=400, tol=1e-12, seed=seed)

    # ---- 预测 vs 实测 ----
    # 扰动方向取幂迭代给出的主特征向量：对准最危险的模态，边界分辨率最高
    direction = power.vec
    rows: list[BoundaryRow] = []
    for beta in betas:
        eta_pred = 2.0 * (1.0 + beta) / info["lam_max"]
        meas = training.boundary_from_prediction(
            grad_fn, theta_star, eta_pred, beta=beta,
            bisect_iters=bisect_iters, steps=test_steps, seed=seed,
            direction=direction)
        eta_meas = float(meas["eta_max"])
        rel = abs(eta_meas - eta_pred) / eta_pred if math.isfinite(eta_meas) else float("nan")

        # 增长率对照：实测几何增长率 vs 理论 ρ
        # 注意用短窗口 + 较大的扰动（1e-6）：长窗口下稳定模态会下溢到 0，
        # 而扰动太大又会撞上 tanh 饱和 —— 两者都会让"增长率"失去意义。
        eta_in = rate_frac * eta_pred
        dt = training.decay_test(grad_fn, theta_star, eta_in, beta=beta,
                                 steps=rate_steps, delta_scale=1e-6,
                                 seed=seed, direction=direction)
        rate_meas = float(dt["rate"])
        if beta == 0.0:
            rate_theory = abs(1.0 - eta_in * info["lam_max"])
        else:
            rate_theory = heavy_ball_rho(eta_in, beta, info["lam_max"])
        rate_rel = (abs(rate_meas - rate_theory) / rate_theory
                    if rate_theory > 0 and math.isfinite(rate_meas) else float("nan"))

        rows.append(BoundaryRow(label=f"β={beta:g}", beta=beta, eta_pred=eta_pred,
                                eta_measured=eta_meas, rel_err=rel,
                                valid=bool(meas.get("valid")),
                                eta_in=eta_in, rate_measured=rate_meas,
                                rate_theory=rate_theory, rate_rel_err=rate_rel))

    # ---- 线性模型闭式对照 ----
    linear: dict[str, Any] = {}
    if with_linear_reference:
        Hl = models.linear_hessian(X)
        info_l = spectral.spectral_summary(Hl)
        lam_l = info_l["lam_max"]
        theta_l, *_ = np.linalg.lstsq(X, y, rcond=None)
        gfun_l = lambda th: models.linear_loss_grad(th, X, y)  # noqa: E731
        lam_pow_l = spectral.power_iteration(
            lambda v: (models.linear_loss_grad(theta_l + 1e-5 * v, X, y)[1]
                       - models.linear_loss_grad(theta_l - 1e-5 * v, X, y)[1]) / 2e-5,
            d, iters=400, tol=1e-13, seed=seed)
        eta_pred_l = 2.0 / lam_l
        # 同样沿主特征向量扰动：随机方向会混入次大模态，让边界略微偏移
        meas_l = training.boundary_from_prediction(
            gfun_l, theta_l, eta_pred_l, beta=0.0, bisect_iters=bisect_iters + 6,
            steps=test_steps, seed=seed, direction=lam_pow_l.vec)
        linear = {
            "lam_max_closed_form": lam_l,
            "lam_max_power": lam_pow_l.lam,
            "power_rel_err": abs(lam_pow_l.lam - lam_l) / lam_l,
            "lam_min": info_l["lam_min"],
            "eta_pred": eta_pred_l,
            "eta_measured": float(meas_l["eta_max"]),
            "rel_err": (abs(float(meas_l["eta_max"]) - eta_pred_l) / eta_pred_l
                        if math.isfinite(float(meas_l["eta_max"])) else float("nan")),
            "valid": bool(meas_l.get("valid")),
        }

    res = StudyResult(
        dataset_name=ds.name, dataset_note=ds.note, n=n, d=d, h=h, p=p,
        grad_check=gcheck, hessian_symmetry=sym,
        l2=l2, train_steps=trained.steps_run,
        train_loss=trained.final_loss, train_grad_norm=trained.final_grad_norm,
        lam_min=info["lam_min"], lam_max_exact=info["lam_max"], lam_max_power=power.lam,
        power_iters=power.iters, power_residual=power.residual,
        rows=rows, linear=linear,
    )

    if res.is_local_min:
        res.notes.append(
            f"λ_min(H) = {res.lam_min:.3e} > 0：θ* 是严格局部极小，"
            "线性化判据 η_max = 2(1+β)/λ_max 在此适用。")
    else:
        res.notes.append(
            f"λ_min(H) = {res.lam_min:.3e} ≤ 0：还停在鞍点/非凸方向，"
            "线性化判据不适用 —— 请先把它训到极小再分析。")
    if res.power_residual > 1e-6 * max(1.0, abs(res.lam_max_power)):
        res.notes.append(
            f"幂迭代残差 {res.power_residual:.2e} 偏大：λ₂/λ₁ 接近 1（病态），"
            "λ_max 的估计可能还不够收敛，可加大 iters。")
    res.notes.append(
        "实测边界：在 θ* 附近做**对称扰动** θ* ± δ（δ = 1e-6），观测量取 ‖θ⁺ − θ⁻‖/2。"
        "三个细节缺一不可 —— ①对称差分消掉残余梯度造成的常数漂移；"
        "②扰动触及 1e-13 噪声地板即判稳定（否则浮点噪声会把「明确稳定」读成「临界」）；"
        "③增长率取**前半段** log 距离的最小二乘斜率（越界后轨迹逃到平坦区、差值饱和，"
        "后半段斜率会回落，实测把 ρ=1.535 的发散读成 0.9975）。"
        "扰动若放到 1e-3，tanh 饱和会让长窗口判据把发散误判成稳定（实测偏差可达 145%）。")
    return res


# ===========================================================================
# 渲染
# ===========================================================================
# ===========================================================================
# 在线监控研究：告警步 vs 损失爆掉步
# ===========================================================================
@dataclass
class MonitorRow:
    model: str
    start: str
    frac: float
    eta: float
    eta_max_ref: float
    alarm_step: int | None
    blowup_step: int | None
    lead_time: int | None
    final_loss: float
    note: str = ""

    @property
    def alarmed(self) -> bool:
        return self.alarm_step is not None


@dataclass
class MonitorStudy:
    beta: float
    steps: int
    rows: list[MonitorRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def false_alarms(self) -> int:
        """稳定配置（frac < 1）里告警的次数 —— 应当是 0。"""
        return sum(1 for r in self.rows if r.frac < 1.0 and r.alarmed)

    @property
    def missed(self) -> int:
        """损失已经爆掉却没告警的次数 —— 应当是 0。"""
        return sum(1 for r in self.rows if r.blowup_step is not None and not r.alarmed)


def run_monitor_study(
    dataset: Dataset | None = None,
    h: int = 12,
    fracs: Sequence[float] = (0.5, 0.9, 1.01, 1.3),
    beta: float = 0.0,
    steps: int = 400,
    l2: float = 1e-3,
    check_every: int = 20,
    power_iters: int = 8,
    blowup_ratio: float = 4.0,
    seed: int = 0,
    with_mlp: bool = True,
) -> MonitorStudy:
    """在线监控的证据表：不同 η/η_max 下，告警发生在损失爆掉之前多少步。

    * **线性模型**：二次目标、``λ_max = XᵀX/n`` 恒定且不会饱和，发散没有歧义 ——
      监控器在第 0 步就报警，提前量就是真实超前量。
    * **两层 MLP**：``λ_max(θ_k)`` 随训练变化，监控器在线估它；非线性饱和会把损失
      增长截断，"损失爆掉"可能根本不来 —— 这本身是值得写进结论的现象。
    """
    ds = dataset if dataset is not None else datasets.load_order_amount()
    X, y, _ = datasets.standardize(ds.X, ds.y)
    n, d = X.shape
    study = MonitorStudy(beta=beta, steps=steps)

    # ---------------- 线性模型（基准） ----------------
    Hl = models.linear_hessian(X)
    lam_l = spectral.spectral_summary(Hl)["lam_max"]
    eta_max_l = 2.0 / lam_l
    gfun_l = lambda th: models.linear_loss_grad(th, X, y)      # noqa: E731
    hvp_l = lambda th, v: Hl @ v                               # noqa: E731
    th0_l = np.zeros(d)
    loss0_l = gfun_l(th0_l)[0]
    for frac in fracs:
        rep = monitor.run_with_monitor(
            gfun_l, hvp_l, th0_l, eta=frac * eta_max_l, steps=steps, beta=beta,
            blowup_loss=blowup_ratio * loss0_l, check_every=check_every,
            power_iters=power_iters, seed=seed)
        study.rows.append(MonitorRow(
            model="linear", start="θ=0", frac=frac, eta=frac * eta_max_l,
            eta_max_ref=eta_max_l, alarm_step=rep.alarm_step,
            blowup_step=rep.blowup_step, lead_time=rep.lead_time,
            final_loss=rep.final_loss))

    # ---------------- 两层 MLP ----------------
    if with_mlp:
        theta0 = models.init_params(d, h, seed=seed)
        gfun = lambda th: models.mlp_loss_grad(th, X, y, d, h, l2=l2)     # noqa: E731
        hvp = lambda th, v: models.mlp_hvp(th, X, y, d, h, v, l2=l2)      # noqa: E731
        hess_fn = lambda th: models.mlp_hessian(th, X, y, d, h, l2=l2)    # noqa: E731
        theta_star = training.train_damped_newton(gfun, hess_fn, theta0,
                                                  steps=120, tol=1e-10).theta
        lam_star = spectral.power_iteration(
            lambda v: hvp(theta_star, v), theta0.size, iters=400, tol=1e-12).lam
        eta_max_m = 2.0 * (1.0 + beta) / lam_star
        for start, th_init in (("随机初值", theta0), ("θ*", theta_star)):
            loss0_m = gfun(th_init)[0]
            for frac in fracs:
                rep = monitor.run_with_monitor(
                    gfun, hvp, th_init, eta=frac * eta_max_m, steps=steps, beta=beta,
                    blowup_loss=blowup_ratio * loss0_m, check_every=check_every,
                    power_iters=power_iters, seed=seed)
                note = "" if rep.blowup_step is not None else "饱和截断，损失未爆"
                study.rows.append(MonitorRow(
                    model="mlp", start=start, frac=frac, eta=frac * eta_max_m,
                    eta_max_ref=eta_max_m, alarm_step=rep.alarm_step,
                    blowup_step=rep.blowup_step, lead_time=rep.lead_time,
                    final_loss=rep.final_loss, note=note))

    study.notes.append(
        f"参考上界：linear 用闭式 2/λ_max = {eta_max_l:.6g}；"
        f"mlp 用 θ* 处的幂迭代值 2(1+β)/λ_max(θ*)（本配置 β={beta:g}）")
    study.notes.append(
        "监控器在每个检查点**在线**估 λ_max(θ_k)（热启动幂迭代），用的是当前点而非 θ* —— "
        "所以它既不需要事先知道最优解，也不需要等损失爆掉。")
    study.notes.append(
        "线性模型的 λ_max 与 θ 无关且二次目标永不饱和，告警出现在第 0 步："
        "提前量 = 损失爆掉步。MLP 上损失可能被 tanh 饱和截断而不爆，"
        "此时只有告警、没有「爆掉步」，不算漏报。")
    return study


def render_monitor_study(study: MonitorStudy) -> str:
    L: list[str] = []
    L.append("-- 在线稳定性监控 · 告警提前量 " + "-" * 33)
    L.append(f"  β = {study.beta:g}，最多 {study.steps} 步；"
             f"误报 {study.false_alarms} 次，漏报 {study.missed} 次")
    L.append("")
    rows = [[
        r.model, r.start, f"{r.frac:g}", f"{r.eta:.6g}",
        "—" if r.alarm_step is None else str(r.alarm_step),
        "未爆" if r.blowup_step is None else str(r.blowup_step),
        "—" if r.lead_time is None else str(r.lead_time),
        f"{r.final_loss:.3e}", r.note,
    ] for r in study.rows]
    L.append(_table(["模型", "起点", "η/η_max", "η", "告警步", "爆掉步", "提前量", "末损失", "备注"], rows))
    L.append("")
    for note in study.notes:
        L.append(f"  · {note}")
    return "\n".join(L)


def monitor_as_json(study: MonitorStudy) -> dict[str, Any]:
    return {
        "beta": study.beta,
        "steps": study.steps,
        "false_alarms": study.false_alarms,
        "missed": study.missed,
        "rows": [
            {"model": r.model, "start": r.start, "frac": r.frac, "eta": r.eta,
             "eta_max_ref": r.eta_max_ref,
             "alarm_step": r.alarm_step, "blowup_step": r.blowup_step,
             "lead_time": r.lead_time, "final_loss": r.final_loss, "note": r.note}
            for r in study.rows
        ],
        "notes": study.notes,
    }


def as_json(res: StudyResult) -> dict[str, Any]:
    """结构化输出。"""
    def num(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v

    return {
        "dataset": {"name": res.dataset_name, "note": res.dataset_note,
                    "n": res.n, "d": res.d, "hidden": res.h, "params": res.p},
        "checks": {
            "gradient_vs_finite_difference": res.grad_check,
            "hessian_symmetry": res.hessian_symmetry,
            "lambda_min": res.lam_min,
            "is_strict_local_min": res.is_local_min,
        },
        "training": {"l2": res.l2, "newton_steps": res.train_steps,
                     "loss": res.train_loss, "grad_norm": res.train_grad_norm},
        "lambda_max": {"assembled": res.lam_max_exact, "power_iteration": res.lam_max_power,
                       "power_iters": res.power_iters, "power_residual": res.power_residual,
                       "rel_err": abs(res.lam_max_power - res.lam_max_exact) / res.lam_max_exact},
        "boundaries": [
            {"beta": r.beta, "eta_pred": num(r.eta_pred), "eta_measured": num(r.eta_measured),
             "rel_err": num(r.rel_err), "valid": r.valid,
             "rate_eta_in": num(r.eta_in), "rate_theory": num(r.rate_theory),
             "rate_measured": num(r.rate_measured), "rate_rel_err": num(r.rate_rel_err)}
            for r in res.rows
        ],
        "linear_reference": {k: num(v) for k, v in res.linear.items()},
        "notes": res.notes,
    }


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           left_cols: int = 1) -> str:
    cols = len(headers)
    w = [display_width(str(x)) for x in headers]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], display_width(str(r[i])))
    out = ["  ".join(pad_right(str(headers[i]), w[i]) for i in range(cols))]
    out.append("-" * (sum(w) + 2 * (cols - 1)))
    for r in rows:
        cells = [pad_right(str(r[i]), w[i]) if i < left_cols else pad_left(str(r[i]), w[i])
                 for i in range(cols)]
        out.append("  ".join(cells))
    return "\n".join(out)


def render_study(res: StudyResult) -> str:
    L: list[str] = []
    L.append("-- 训练稳定性诊断 · 预测 η_max vs 实测边界 " + "-" * 28)
    L.append(f"  数据集  {res.dataset_name}")
    L.append(f"  说明    {res.dataset_note}")
    L.append(f"  规模    n={res.n}  d={res.d}  h={res.h}  参数量 p={res.p}")
    L.append("")
    L.append(f"  梯度对拍（解析 vs 中心差分）  {res.grad_check:.2e}")
    L.append(f"  Hessian 对称性 ‖H−Hᵀ‖/‖H‖   {res.hessian_symmetry:.2e}")
    L.append(f"  训练    阻尼牛顿 {res.train_steps} 步 → loss={res.train_loss:.3e}，"
             f"‖∇L‖={res.train_grad_norm:.3e}（L2 = {res.l2:g}）")
    L.append(f"  局部极小检查  λ_min(H) = {res.lam_min:.4e} → "
             f"{'严格局部极小 ✓' if res.is_local_min else '非极小 ✗（判据不适用）'}")
    L.append("")
    L.append(f"  λ_max：精确装配 {res.lam_max_exact:.8f}    "
             f"幂迭代 {res.lam_max_power:.8f}    "
             f"相对误差 {abs(res.lam_max_power - res.lam_max_exact) / res.lam_max_exact:.2e}"
             f"（{res.power_iters} 步，残差 {res.power_residual:.1e}）")
    L.append("")
    rows = [[r.label, f"{r.eta_pred:.6f}", f"{r.eta_measured:.6f}",
             f"{r.rel_err * 100:.2f}%", "✓" if r.valid else "区间失配"] for r in res.rows]
    L.append(_table(["β", "预测 η_max = 2(1+β)/λ_max", "实测边界", "相对偏差", "状态"], rows))
    # 增长率对照
    L.append("")
    L.append("  增长率对照（沿主特征向量扰动，在 η_in = 0.8·η_max 处测每步几何倍率）")
    L.append("  β>0 的有限窗口偏差来自动量振荡的相位；β=0 无振荡，故精确到 1e-5 量级。")
    rate_rows = [[r.label, f"{r.eta_in:.6f}", f"{r.rate_theory:.6f}", f"{r.rate_measured:.6f}",
                  f"{r.rate_rel_err * 100:.2f}%"] for r in res.rows]
    L.append(_table(["β", "η_in", "理论 ρ", "实测增长率", "相对偏差"], rate_rows))
    # (1+β) 缩放检验
    if len(res.rows) >= 2 and all(r.valid for r in res.rows):
        r0, rN = res.rows[0], res.rows[-1]
        pred_ratio = (1.0 + rN.beta) / (1.0 + r0.beta)
        meas_ratio = rN.eta_measured / r0.eta_measured
        L.append("")
        L.append(f"  (1+β) 缩放检验：预测 {pred_ratio:.4f} 倍 vs 实测 {meas_ratio:.4f} 倍"
                 f"（偏差 {abs(meas_ratio - pred_ratio) / pred_ratio * 100:.2f}%）")
    if res.linear:
        li = res.linear
        L.append("")
        L.append("  线性模型对照（闭式 Hessian = XᵀX/n，无任何差分）")
        L.append(f"    λ_max：闭式 {li['lam_max_closed_form']:.8f}  "
                 f"幂迭代 {li['lam_max_power']:.8f}  相对误差 {li['power_rel_err']:.2e}")
        L.append(f"    η_max：预测 {li['eta_pred']:.6f}  实测 {li['eta_measured']:.6f}  "
                 f"相对偏差 {li['rel_err'] * 100:.3f}%   "
                 f"（二次目标下两者应当一致到二分精度）")
    L.append("")
    for note in res.notes:
        L.append(f"  · {note}")
    return "\n".join(L)
