"""stability_lens.transient — 非正规性的账：``ρ < 1`` 不保证误差单调下降。

``README`` 里一直挂着这条边界，但从来没被量化过。本模块把它变成可计算的量：

* **瞬态曲线** ``‖M^k‖₂``：谱半径只管 ``k → ∞``，而实际训练只跑有限步；
  非正规矩阵可以在 ``ρ(M) < 1`` 的前提下先放大若干数量级再衰减。
* **Kreiss 常数** ``K(M) = sup_{|z|>1} (|z| − 1)·‖(zI − M)⁻¹‖₂``：
  非正规性的严格度量。Kreiss 矩阵定理给出 ``sup_k‖M^k‖ ≤ e·n·K(M)``。
* **ε-伪谱** ``{z : σ_min(zI − M) < ε}``：特征值只告诉你谱在哪，
  伪谱告诉你"谱附近有多大一片区域实际上和特征值不可区分" —— 它才是瞬态增长的来源。

为什么和稳定性有关：迭代 ``x_{k+1} = M x_k`` 里 ``M = I − ηA``。
即使所有特征值都在单位圆内，``‖M^k‖`` 仍可能先冲到很大 ——
误差在若干步内被放大成 ``O(10³)``，这时候"ρ < 1 所以稳定"这个结论在实践中是没有意义的。

**范围说明**：本模块基于显式矩阵（精确 2-范数），适合 p 不大的场合，
例如 2×2 模态分析、或 ``θ*`` 处已经装配出来的局部 Hessian。
参数维度很大时应当改用随机探针估 ``‖M^k v‖``（那是下界）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

Array = np.ndarray

__all__ = [
    "TransientResult",
    "NonNormalDiagnosis",
    "parse_matrix",
    "iteration_matrix",
    "nonnormal_example",
    "transient_curve",
    "transient_growth",
    "position_amplification",
    "kreiss_constant",
    "pseudospectrum_sigma_min",
    "diagnose_nonnormal",
    "render",
    "as_json",
]


# ===========================================================================
# 输入
# ===========================================================================
def parse_matrix(text: str) -> Array:
    """解析 ``"0.9,10;0,0.9"`` 这样的方阵（行之间用 ``;``，元素用逗号/空格分隔）。"""
    rows = [r for r in str(text).replace("|", ";").split(";") if r.strip()]
    if not rows:
        raise ValueError("矩阵为空")
    data = [[float(x) for x in r.replace(",", " ").split()] for r in rows]
    M = np.asarray(data, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError('需要方阵，行之间用 ";" 分隔，例如 "0.9,10;0,0.9"')
    return M


def iteration_matrix(A: Array, eta: float) -> Array:
    """前向 Euler 的迭代矩阵 ``M = I − ηA``。"""
    A = np.asarray(A, dtype=float)
    return np.eye(A.shape[0]) - eta * A


def nonnormal_example(lam: float = 0.9, coupling: float = 10.0) -> Array:
    """经典最小非正规例子 ``M = [[λ, c], [0, λ]]``。

    ``ρ(M) = λ < 1``，但 ``M^k = [[λ^k, k·λ^{k-1}·c], [0, λ^k]]`` ——
    ``k`` 一大，非对角项先涨后落，``‖M^k‖`` 能冲到 ``λ`` 完全预测不到的量级。
    """
    return np.array([[lam, coupling], [0.0, lam]], dtype=float)


# ===========================================================================
# 瞬态增长
# ===========================================================================
@dataclass
class TransientResult:
    rho: float
    peak: float
    peak_step: int
    final: float
    steps: int
    curve: list[float] = field(default_factory=list)

    @property
    def asymptotic_decreasing(self) -> bool:
        return self.rho < 1.0

    @property
    def contradicts_spectral_picture(self) -> bool:
        """谱说会衰减，实际却放大了 —— 非正规性的直接证据。"""
        return self.rho < 1.0 and self.peak > 1.0


def transient_curve(M: Array, steps: int = 50) -> list[float]:
    """返回 ``[‖M⁰‖, ‖M¹‖, …, ‖M^steps‖]``（精确 2-范数）。"""
    A = np.asarray(M, dtype=float)
    n = A.shape[0]
    P = np.eye(n)
    out = [float(np.linalg.norm(P, 2))]
    for _ in range(steps):
        P = P @ A
        out.append(float(np.linalg.norm(P, 2)))
    return out


def transient_growth(M: Array, steps: int = 50) -> TransientResult:
    """瞬态曲线的摘要：峰值、峰值步、谱半径。"""
    curve = transient_curve(M, steps)
    peak = max(curve)
    rho = float(max(abs(np.linalg.eigvals(np.asarray(M, dtype=float)))))
    return TransientResult(rho=rho, peak=peak, peak_step=curve.index(peak),
                           final=curve[-1], steps=steps, curve=curve)


def kreiss_constant(
    M: Array,
    radii: Sequence[float] = (1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0,
                              100.0, 1000.0),
    n_angles: int = 96,
) -> float:
    """``K(M) = sup_{|z|>1} (|z| − 1)·‖(zI − M)⁻¹‖₂``（网格搜索）。

    正规矩阵的 ``K`` 趋近 1；非正规程度越高 ``K`` 越大。
    Kreiss 矩阵定理：``K(M) ≤ sup_k‖M^k‖ ≤ e·n·K(M)``。

    注意 ``sup`` 在 ``|z| → ∞`` 方向趋于 ``(|z|−1)/(|z|−ρ) → 1``，
    所以半径网格必须一直铺到很远（默认到 1000），否则正规矩阵会算出略小于 1 的值。
    """
    A = np.asarray(M, dtype=float)
    n = A.shape[0]
    I = np.eye(n)
    best = 0.0
    for r in radii:
        for t in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
            z = complex(r * math.cos(t), r * math.sin(t))
            try:
                inv = np.linalg.inv(z * I - A)
            except np.linalg.LinAlgError:
                continue
            best = max(best, (r - 1.0) * float(np.linalg.norm(inv, 2)))
    return best


def pseudospectrum_sigma_min(
    M: Array,
    re_range: tuple[float, float] = (-0.2, 1.4),
    im_range: tuple[float, float] = (-1.0, 1.0),
    n: int = 61,
) -> tuple[Array, Array, Array]:
    """在网格上算 ``σ_min(zI − M)``。

    返回 ``(re_axis, im_axis, grid)``，``grid[i, j] = σ_min(z_ij I − M)``。
    ε-伪谱就是 ``{σ_min < ε}``：正规矩阵的伪谱是特征值周围的圆盘，
    非正规矩阵会向外鼓出一大截 —— 那一截正是瞬态增长的来源。
    """
    A = np.asarray(M, dtype=float)
    n_dim = A.shape[0]
    I = np.eye(n_dim)
    re = np.linspace(*re_range, n)
    im = np.linspace(*im_range, n)
    grid = np.empty((n, n))
    for i, x in enumerate(re):
        for j, y in enumerate(im):
            grid[i, j] = np.linalg.svd((x + 1j * y) * I - A, compute_uv=False)[-1]
    return re, im, grid


# ===========================================================================
# 体检
# ===========================================================================
@dataclass
class NonNormalDiagnosis:
    title: str
    rho: float
    peak: float
    peak_step: int
    kreiss: float
    sigma_min_probe: float | None
    verdict: str
    level: str            # "ok" | "transient" | "unstable"
    exit_code: int
    rows: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_transient(self) -> bool:
        return self.level == "transient"


def position_amplification(M: Array, steps: int = 60) -> tuple[float, int]:
    """``max_k ‖e₁ᵀ M^k‖₂``：最坏初始条件下**第一个分量**（误差本身）的放大。

    与 :func:`transient_growth` 的区别很重要：后者量的是整状态范数 ``‖M^k‖``，
    在"把二阶递推写成一阶状态"时会把表示上的非正规性也算进去。
    例如 ``β = 0`` 的动量伴随矩阵 ``‖M^k‖`` 会报到 1.28，但误差分量根本不会放大。
    """
    A = np.asarray(M, dtype=float)
    P = np.eye(A.shape[0])
    best, best_step = float(np.linalg.norm(P[0, :], 2)), 0
    for k in range(1, steps + 1):
        P = P @ A
        val = float(np.linalg.norm(P[0, :], 2))
        if val > best:
            best, best_step = val, k
    return best, best_step


def diagnose_nonnormal(
    M: Array,
    threshold: float = 10.0,
    steps: int = 60,
    probe: complex | None = None,
    title: str = "非正规性 / 瞬态增长",
) -> NonNormalDiagnosis:
    """给出 ``ρ``、瞬态峰值、Kreiss 常数与判定。

    Args:
        threshold: 瞬态放大超过它才告警。默认 10 倍 —— 误差先被放大 10 倍，
            已经足够让"ρ < 1 所以稳定"这句话在实践中失去意义。
        probe: 想额外看伪谱的点（默认取距谱最远的那个特征值外的 1e-1 处）。
    """
    from .diagnose import EXIT_GAP, EXIT_OK, EXIT_STRUCTURAL

    A = np.asarray(M, dtype=float)
    tg = transient_growth(A, steps=steps)
    K = kreiss_constant(A)

    if probe is None:
        eigs = np.linalg.eigvals(A)
        k = int(np.argmax(np.abs(eigs)))
        probe = complex(eigs[k]) + 0.1
    sigma_min = float(np.linalg.svd(probe * np.eye(A.shape[0]) - A, compute_uv=False)[-1])

    rows: list[tuple[str, str]] = [
        ("维度", str(A.shape[0])),
        ("谱半径 ρ(M)", f"{tg.rho:.6g}"),
        ("瞬态峰值 max_k ‖M^k‖₂", f"{tg.peak:.6g}（第 {tg.peak_step} 步）"),
        ("Kreiss 常数 K(M)", f"{K:.6g}"),
        (f"σ_min(({probe.real:.3g}{probe.imag:+.3g}i)I − M)", f"{sigma_min:.3g}"),
        ("末值 ‖M^steps‖₂", f"{tg.final:.3g}"),
    ]

    notes: list[str] = []
    if tg.rho >= 1.0:
        level, code = "unstable", EXIT_STRUCTURAL
        verdict = f"ρ = {tg.rho:.4g} ≥ 1：谱本身就发散"
        notes.append("谱半径已经 ≥ 1，这一条与瞬态无关，先解决它。")
    elif tg.peak >= threshold:
        level, code = "transient", EXIT_GAP
        verdict = (f"谱说衰减（ρ={tg.rho:.4g}），实际先放大 {tg.peak:.3g} 倍"
                   f"（第 {tg.peak_step} 步）")
        notes.append(
            f"Kreiss 常数 K = {K:.4g}：非正规性把瞬态上界抬到了这个量级。"
            "「ρ < 1 所以稳定」在有限步内是**误导**的。")
        notes.append(
            "对策：减小 η（会同时压缩 ρ 和瞬态）、换变量做预条件/白化（改度量、降低非正规性），"
            "或者接受放大但把初始误差压小。")
    else:
        level, code = "ok", EXIT_OK
        verdict = (f"谱半径与瞬态一致衰减：ρ={tg.rho:.4g}，"
                   f"峰值 {tg.peak:.4g} < 阈值 {threshold:g}")
        notes.append(f"Kreiss 常数 K = {K:.4g}（接近 1 说明矩阵接近正规）。")

    rows.append(("谱 vs 实际", verdict))
    notes.append(
        "本模块用显式矩阵算精确 2-范数。维度很大时请改用随机探针估 ‖M^k v‖（那是下界）。")

    return NonNormalDiagnosis(
        title=title, rho=tg.rho, peak=tg.peak, peak_step=tg.peak_step, kreiss=K,
        sigma_min_probe=sigma_min, verdict=verdict, level=level, exit_code=code,
        rows=rows, notes=notes,
    )


def render(d: NonNormalDiagnosis, color: bool = False) -> str:
    """文本渲染，样式与 ``check`` 一致。"""
    from . import report

    tag = report.LEVEL_TAG.get(d.level, "[ ? ]")
    if color and d.level in report.LEVEL_COLOR:
        tag = report.LEVEL_COLOR[d.level] + tag + report.RESET
    lines = [report._wrap_dash(d.title), f"  判定: {tag}  {d.verdict}", ""]
    lines.append(report.render_rows(d.rows))
    if d.notes:
        lines += ["", "  提示:", report.render_notes(d.notes)]
    return "\n".join(lines)


def as_json(d: NonNormalDiagnosis) -> dict[str, Any]:
    return {
        "title": d.title,
        "level": d.level,
        "exit_code": d.exit_code,
        "verdict": d.verdict,
        "rho": d.rho,
        "peak": d.peak,
        "peak_step": d.peak_step,
        "kreiss": d.kreiss,
        "sigma_min_probe": d.sigma_min_probe,
        "rows": [[k, v] for k, v in d.rows],
        "notes": d.notes,
    }
