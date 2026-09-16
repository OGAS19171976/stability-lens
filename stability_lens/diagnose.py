"""stability_lens.diagnose — 把内核组装成一次「体检」。

对外的主入口是 :func:`diagnose`：给一个更新规则（或一串特征值）与一个步长，
返回结构化的 :class:`Diagnosis`，其中已经包含 η_max、谱半径、判定与设计建议。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import core

__all__ = [
    "Diagnosis",
    "parse_complex",
    "parse_spectrum",
    "diagnose_euler",
    "diagnose_gd",
    "diagnose_heavy_ball",
    "diagnose_lms",
    "diagnose",
    "sweep_euler",
    "sweep_heavy_ball",
]

# 退出码约定（可直接当作训练前的 gate）
EXIT_OK = 0        # 稳定 / 只是给出建议
EXIT_GAP = 2       # 连续稳定但离散不稳定（缝隙）
EXIT_STRUCTURAL = 3  # ODE 本身不稳定，调步长救不回来


@dataclass
class Diagnosis:
    """一次稳定性体检的完整结果。"""

    rule: str
    title: str
    eta: float | None
    eta_max: float
    rho: float | None
    stable: bool | None
    hurwitz: bool
    verdict: str
    level: str                 # "ok" | "gap" | "unstable" | "info"
    exit_code: int
    eigenvalues: list[complex] = field(default_factory=list)
    rows: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# 输入解析
# ===========================================================================
def parse_complex(text: str) -> complex:
    """解析 ``"1"``、``"-2.5"``、``"3i"``、``"-1+3i"``、``"0.5-2j"`` 这类写法。"""
    s = text.strip().replace(" ", "").replace("j", "i")
    if s in ("i", "+i"):
        return complex(0.0, 1.0)
    if s == "-i":
        return complex(0.0, -1.0)
    if s.endswith("i"):
        body = s[:-1]
        idx = max(body.rfind("+", 1), body.rfind("-", 1))
        if idx == -1:
            if body in ("", "+"):
                return complex(0.0, 1.0)
            if body == "-":
                return complex(0.0, -1.0)
            return complex(0.0, float(body))
        re_part, im_part = body[:idx], body[idx:]
        im = 1.0 if im_part == "+" else -1.0 if im_part == "-" else float(im_part)
        return complex(float(re_part), im)
    return complex(float(s), 0.0)


def parse_spectrum(text: Iterable[str] | str) -> list[complex]:
    """把 ``"1,4"`` 或 ``["-1+3i", "2"]`` 解析成特征值列表。"""
    parts: list[str] = []
    if isinstance(text, str):
        parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    else:
        for item in text:
            parts.extend(p for p in str(item).replace(";", ",").split(",") if p.strip())
    if not parts:
        raise ValueError("特征值列表为空")
    return [parse_complex(p) for p in parts]


def fmt_c(z: complex, digits: int = 4) -> str:
    """紧凑复数格式化：``-1.0000+3.0000i``。"""
    re, im = z.real, z.imag
    if im == 0.0:
        return f"{re:.{digits}f}"
    sign = "+" if im >= 0 else "-"
    return f"{re:.{digits}f}{sign}{abs(im):.{digits}f}i"


# ===========================================================================
# 各规则的体检
# ===========================================================================
def diagnose_euler(
    spectrum: Sequence[complex] | str,
    eta: float | None = None,
    title: str = "前向 Euler / 线性增量迭代",
    rule: str = "euler",
) -> Diagnosis:
    """谱直接给定（``lambda(J)``）时的体检。

    ``eta`` 给定则同时判定该步长下的离散稳定性；否则只报 η_max。
    """
    lams = parse_spectrum(spectrum) if isinstance(spectrum, str) else list(spectrum)
    em = core.euler_eta_max(lams)
    hurwitz = em.hurwitz

    rho = core.euler_rho(lams, eta) if eta is not None else None
    stable = (rho < 1.0) if rho is not None else None

    rows: list[tuple[str, str]] = [
        ("谱", "  ".join(fmt_c(z, 3) for z in lams)),
        ("Hurwitz（全部 Re λ < 0）", "是" if hurwitz else "否"),
        ("η_max = min 2(−Re λ)/|λ|²", f"{em.eta_max:.6g}" if hurwitz else "—"),
    ]
    if em.limiting is not None:
        rows.append(("决定 η_max 的模态", fmt_c(em.limiting, 4)))
    if eta is not None:
        rows.append(("η", f"{eta:.6g}"))
        rows.append(("η / η_max", f"{eta / em.eta_max:.4f}" if em.eta_max > 0 else "—"))
        rows.append(("ρ(I + ηJ)", f"{rho:.8g}"))
        rows.append(("离散性", verdict_text(hurwitz, stable)))

    notes: list[str] = []
    if not hurwitz:
        level, code, verdict = "unstable", EXIT_STRUCTURAL, "ODE 本身不稳定：调小 η 救不回来"
        notes.append(
            "存在 Re λ ≥ 0 的模态 —— 这是**结构性**不稳定。η 只改变时间尺度，不改变特征值符号。"
        )
        if all(z.imag == 0.0 and z.real > 0.0 for z in lams):
            notes.append(
                "提示：这些看起来像二次目标的**曲率 λ(A) > 0**。若确实如此，请改用 "
                "`--rule gd`（它内部取 J = −A）；`--rule euler` 的 --spectrum 期望的是 λ(J)。"
            )
    elif eta is None:
        level, code, verdict = "info", EXIT_OK, f"离散稳定的充要条件：0 < η < {em.eta_max:.6g}"
        notes.append("上界来自**积分格式的绝对稳定域**，不是来自 ODE：ODE 本身没有步长。")
    elif stable:
        level, code, verdict = "ok", EXIT_OK, "离散稳定"
        ratio = eta / em.eta_max
        if ratio > 0.8:
            notes.append(f"η 已用到 η_max 的 {ratio * 100:.0f}%，靠近边界；小扰动或非正规性可能让它翻车。")
    else:
        level, code = "gap", EXIT_GAP
        ratio = eta / em.eta_max
        verdict = f"连续稳定但离散不稳定（缝隙）：η 是上界的 {ratio:.3f} 倍"
        notes.append("连续时间的特征值全在左半平面，但 ηλ 已经跑出前向 Euler 的圆盘 |1+ηλ| < 1。")
        notes.append(f"把 η 降到 {em.eta_max:.6g} 以下即可恢复；或改用 A-稳定的隐式格式换取无条件稳定。")

    if eta is not None and hurwitz and eta > 0:
        slow = max(lams, key=lambda z: z.real)  # Re 最大 = 最慢模态
        rows.append(("最慢模态 −Re λ", f"{-slow.real:.6g}"))
        rows.append(("最慢模态的离散有效速率（<0 = 增长）", f"{core.euler_effective_rate(slow, eta):.6g}"))

    return Diagnosis(
        rule=rule, title=title, eta=eta, eta_max=em.eta_max, rho=rho, stable=stable,
        hurwitz=hurwitz, verdict=verdict, level=level, exit_code=code,
        eigenvalues=lams, rows=rows, notes=notes,
        extra={"limiting": em.limiting},
    )


def diagnose_gd(curvature: Sequence[float] | str, eta: float | None = None) -> Diagnosis:
    """二次目标 ``f = 1/2 x'Ax - b'x`` 上的梯度下降：``J = -A``。

    ``--spectrum`` 在这里解释为 **A 的曲率**（正的实数特征值）。
    """
    if isinstance(curvature, str):
        parsed = parse_spectrum(curvature)
    else:
        parsed = [complex(v, 0.0) for v in curvature]
    if any(z.imag != 0.0 for z in parsed):
        raise ValueError("梯度下降规则要求 A 的曲率为实数")
    cur = [z.real for z in parsed]
    d = diagnose_euler([complex(-c, 0.0) for c in cur], eta=eta,
                       title="梯度下降（二次目标，J = −A）", rule="gd")
    lam_max = max(cur)
    d.rows.insert(0, ("曲率 λ(A)", "  ".join(f"{c:.6g}" for c in cur)))
    d.rows.append(("λ_max / λ_min = κ", f"{lam_max / min(cur):.6g}"))
    d.notes.append(
        f"实特征值特例：η_max = 2/λ_max = {2.0 / lam_max:.6g}；"
        f"病态 κ = {lam_max / min(cur):.4g} ⟺ 刚性 ODE，步长被 λ_max 卡、迭代数被 κ 卡。"
    )
    return d


def diagnose_heavy_ball(
    eta: float | None = None, beta: float = 0.9, lambda_max: float = 1.0
) -> Diagnosis:
    """Heavy-ball / momentum 的体检（按最坏模态 ``λ_max`` 分析）。"""
    emax = core.heavy_ball_eta_max(beta, lambda_max)
    rho = core.heavy_ball_rho(eta, beta, lambda_max) if eta is not None else None
    stable = (rho < 1.0) if rho is not None else None
    disc = core.heavy_ball_disc(eta, beta, lambda_max) if eta is not None else None
    eigs = core.heavy_ball_eigs(eta, beta, lambda_max) if eta is not None else ()
    a = core.continuous_damping(eta, beta) if eta is not None else None

    rows: list[tuple[str, str]] = [
        ("β", f"{beta:.4g}"),
        ("λ_max", f"{lambda_max:.6g}"),
        ("η_max = 2(1+β)/λ_max", f"{emax:.6g}"),
    ]
    tr: dict[str, float] | None = None
    if eta is not None:
        rows += [
            ("η", f"{eta:.6g}"),
            ("特征值 μ₁,₂", f"{fmt_c(eigs[0], 4)} , {fmt_c(eigs[1], 4)}"),
            ("判别式 tr² − 4det", f"{disc:.6g}" + ("（复根，|μ| = √β）" if disc < 0 else "（实根）")),
            ("ρ = max|μ|", f"{rho:.8g}"),
            ("√β", f"{math.sqrt(beta):.8g}"),
            ("ODE 阻尼 a = (1−β)/√η", f"{a:.6g}" + ("  > 0 ⇒ ODE 无条件稳定" if a > 0 else "  ≤ 0 ⇒ ODE 本身不稳")),
            ("离散性", verdict_text(a > 0, stable)),
        ]
        # 非正规性的账：谱半径只管 k→∞，伴随矩阵非正规时会先放大
        tr = core.heavy_ball_transient(eta, beta, lambda_max)
        rows += [
            ("误差瞬态峰值 max_k‖e₁ᵀM^k‖", f"{tr['position_peak']:.6g}（第 {tr['position_peak_step']:.0f} 步）"),
            ("整状态峰值 max_k‖M^k‖", f"{tr['state_peak']:.6g}"),
            ("Kreiss 常数 K(M)", f"{tr['kreiss']:.6g}"),
        ]

    notes: list[str] = []
    if eta is None:
        level, code, verdict = "info", EXIT_OK, f"离散稳定的充要条件：0 < η < {emax:.6g}"
        notes.append(f"动量把步长上界线性放大了 (1+β) = {1 + beta:.4g} 倍，代价是进入欠阻尼振荡区。")
    elif a is not None and a <= 0:
        level, code, verdict = "unstable", EXIT_STRUCTURAL, "β ≥ 1：连 ODE 都不稳，调 η 无用"
        notes.append("连续时间阻尼 a ≤ 0，能量不再单调下降 —— 这是结构性不稳定。")
    elif stable:
        level, code, verdict = "ok", EXIT_OK, "离散稳定"
        if disc is not None and disc < 0:
            notes.append("复根区：|μ| = √β 与 η 无关，迭代以固定几何速率振荡收敛。")
        notes.append(f"η / η_max = {eta / emax:.4f}。")
    else:
        level, code = "gap", EXIT_GAP
        verdict = f"连续稳定但离散不稳定（缝隙）：η 是上界的 {eta / emax:.3f} 倍"
        notes.append(
            f"ODE 阻尼 a = {a:.6g} > 0，连续时间无可争议地稳定；"
            "但离散迭代已越过单位圆。这就是「连续稳定 ⇒ 离散稳定」的构造性反例。"
        )
        notes.append("注意：这**不是**因为 ODE 不稳定，而是因为前向 Euler 的绝对稳定域是有限的。")

    if tr is not None and tr["position_peak"] >= 1.5:
        notes.append(
            f"非正规性的账：ρ = {rho:.4f} < 1 只管 k→∞，而伴随矩阵非正规 —— "
            f"**误差本身**会先放大 {tr['position_peak']:.3g} 倍（第 {tr['position_peak_step']:.0f} 步）再衰减"
            f"（整状态范数峰值 {tr['state_peak']:.3g}，Kreiss 常数 {tr['kreiss']:.3g}）。"
            "「ρ < 1 所以误差单调下降」在这里是错的。"
        )
        notes.append(
            "GD 不会有这个问题（H 对称 ⟹ M = I−ηH 正规，位置放大恒为 1）；"
            "放大完全来自动量项。想压掉它可以减小 β 或 η。"
        )

    return Diagnosis(
        rule="heavy-ball", title=f"Heavy-ball / momentum（λ_max = {lambda_max:.6g}）",
        eta=eta, eta_max=emax, rho=rho, stable=stable, hurwitz=(beta < 1.0),
        verdict=verdict, level=level, exit_code=code, eigenvalues=list(eigs),
        rows=rows, notes=notes,
        extra={"beta": beta, "lambda_max": lambda_max, "disc": disc, "damping": a,
               "transient": tr},
    )


def diagnose_lms(eta: float, s2: float = 1.0, sigma2: float = 1.0) -> Diagnosis:
    """LMS / 随机逼近的噪声地板：常数步长的稳态方差 ``V = ησ²/(2 − ηs²)``。"""
    v = core.steady_variance(eta, s2, sigma2)
    rows = [
        ("η", f"{eta:.6g}"),
        ("回归子二阶矩 s²", f"{s2:.6g}"),
        ("噪声方差 σ²", f"{sigma2:.6g}"),
        ("稳定条件", f"η < 2/s² = {2.0 / s2:.6g}"),
        ("稳态方差 V", f"{v:.6g}" if math.isfinite(v) else "∞（已越界）"),
        ("稳态 RMS √V", f"{math.sqrt(v):.6g}" if math.isfinite(v) else "∞"),
        ("小 η 渐近 ησ²/2", f"{eta * sigma2 / 2.0:.6g}"),
        ("渐近相对误差", f"{abs(v - eta * sigma2 / 2.0) / v * 100:.3g}%"),
    ]
    notes = [
        "噪声不移动零点 h(x*) = 0，所以不改变 ODE 的平衡点；它决定误差**停在哪里**。",
        "V ∝ η ⇒ 常数步长的误差永久停在 O(√η) 的噪声地板上；要收敛到点必须用衰减步长 "
        "(Ση = ∞, Ση² < ∞)。",
        "想要置信区间或有限样本界，ODE 不够用，需要扩散逼近 / SDE。",
    ]
    if not math.isfinite(v):
        return Diagnosis(
            rule="lms", title="LMS / 随机逼近（噪声地板）", eta=eta, eta_max=math.inf,
            rho=None, stable=False, hurwitz=False,
            verdict=f"η ≥ 2/s² = {2.0 / s2:.6g}：连均值都不收敛", level="unstable",
            exit_code=EXIT_STRUCTURAL, rows=rows, notes=notes,
        )
    return Diagnosis(
        rule="lms", title="LMS / 随机逼近（噪声地板）", eta=eta, eta_max=2.0 / s2,
        rho=None, stable=True, hurwitz=True,
        verdict=f"收敛到 O(√η) 邻域：稳态 RMS = {math.sqrt(v):.6g}", level="info",
        exit_code=EXIT_OK, rows=rows, notes=notes,
    )


def verdict_text(continuous_stable: bool, discrete_stable: bool | None) -> str:
    if not continuous_stable:
        return "连续时间不稳定（结构性）"
    if discrete_stable is None:
        return "只报了上界"
    return "离散稳定" if discrete_stable else "连续稳定 / 离散不稳定（缝隙）"


def diagnose(rule: str, **kw: Any) -> Diagnosis:
    """按规则名分发。可用规则：``euler`` / ``gd`` / ``heavy-ball`` / ``lms``。"""
    table: dict[str, Callable[..., Diagnosis]] = {
        "euler": diagnose_euler,
        "gd": diagnose_gd,
        "heavy-ball": diagnose_heavy_ball,
        "momentum": diagnose_heavy_ball,
        "lms": diagnose_lms,
    }
    if rule not in table:
        raise ValueError(f"未知规则 {rule!r}；可用：{', '.join(sorted(table))}")
    return table[rule](**kw)


# ===========================================================================
# 扫描
# ===========================================================================
def sweep_euler(
    spectrum: Sequence[complex] | str,
    eta_lo: float,
    eta_hi: float,
    n: int = 25,
) -> dict[str, Any]:
    """扫 η，给出每个点的谱半径与数值临界点（与解析 η_max 对照）。"""
    lams = parse_spectrum(spectrum) if isinstance(spectrum, str) else list(spectrum)
    em = core.euler_eta_max(lams)
    pts = []
    for i in range(n):
        eta = eta_lo + (eta_hi - eta_lo) * i / (n - 1) if n > 1 else eta_lo
        pts.append((eta, core.euler_rho(lams, eta)))
    numeric = core.bisect_eta_max(lambda e: core.euler_rho(lams, e),
                                  max(eta_lo, 1e-12), max(eta_hi, 1e-9))
    return {"points": pts, "eta_max": em.eta_max, "eta_max_numeric": numeric, "spectrum": lams}


def sweep_heavy_ball(
    beta: float, lambda_max: float, eta_lo: float, eta_hi: float, n: int = 25
) -> dict[str, Any]:
    """扫 η，给出谱半径与解析 η_max = 2(1+β)/λ_max 的对照。"""
    pts = []
    for i in range(n):
        eta = eta_lo + (eta_hi - eta_lo) * i / (n - 1) if n > 1 else eta_lo
        pts.append((eta, core.heavy_ball_rho(eta, beta, lambda_max)))
    numeric = core.bisect_eta_max(lambda e: core.heavy_ball_rho(e, beta, lambda_max),
                                  max(eta_lo, 1e-12), max(eta_hi, 1e-9))
    return {
        "points": pts,
        "eta_max": core.heavy_ball_eta_max(beta, lambda_max),
        "eta_max_numeric": numeric,
    }
