"""stability_lens.core — 纯数值内核（零第三方依赖）。

把「增量式算法 → ODE → 稳定性分析」流程里所有算得出来的部分做成函数：

    (4)  Hurwitz 判据 / 谱半径        eig_2x2, is_hurwitz, spectral_radius
    (5)  Lyapunov 证书                solve_lyapunov_2x2
    (6)  前向 Euler 绝对稳定域        euler_eta_max, euler_rho, euler_stable
    (6') Heavy-ball 的步长上界        heavy_ball_eta_max, heavy_ball_eigs
    (7)  常步长 LMS 的噪声地板        steady_variance, simulate_lms

本模块与 `incremental-ode/web/stability-core.js` 是同一套公式的两份**独立实现**：
对同一组参数，两边的解析结果逐位一致（见 `stability_lens.selftest` 与仓库 README 的交叉验证表）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

__all__ = [
    "EtaMax",
    "LyapunovSolution",
    "WindowStats",
    "eig_2x2",
    "spectral_radius",
    "is_hurwitz",
    "solve_lyapunov_2x2",
    "euler_multiplier",
    "euler_stable",
    "euler_rho",
    "euler_eta_max",
    "euler_effective_rate",
    "bisect_eta_max",
    "simulate_diagonal",
    "exact_flow_diagonal",
    "norm",
    "heavy_ball_matrix",
    "heavy_ball_eigs",
    "heavy_ball_rho",
    "heavy_ball_eta_max",
    "heavy_ball_disc",
    "continuous_damping",
    "simulate_heavy_ball",
    "steady_variance",
    "simulate_lms",
    "window_stats",
    "mat2_mul",
    "mat2_norm2",
    "transient_curve_2x2",
    "transient_peak_2x2",
    "kreiss_2x2",
    "heavy_ball_transient",
]

Matrix = Sequence[Sequence[float]]
Eigs = Sequence[complex]

# 发散保护阈值：超过它就认为已经爆掉，提前结束模拟
BLOWUP = 1e12


# ===========================================================================
# (4) 特征值与稳定性判据
# ===========================================================================
def eig_2x2(m: Matrix) -> tuple[complex, complex]:
    """2x2 实矩阵的解析特征值。``m = [[a, b], [c, d]]``。"""
    a, b = m[0][0], m[0][1]
    c, d = m[1][0], m[1][1]
    tr, det = a + d, a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        s = math.sqrt(disc)
        return complex((tr + s) / 2.0, 0.0), complex((tr - s) / 2.0, 0.0)
    s = math.sqrt(-disc)
    return complex(tr / 2.0, s / 2.0), complex(tr / 2.0, -s / 2.0)


def spectral_radius(eigs: Eigs) -> float:
    """``rho = max |lambda|``。"""
    return max(abs(z) for z in eigs)


def is_hurwitz(eigs: Eigs) -> bool:
    """所有特征值实部严格为负（连续时间渐近稳定）。"""
    return all(z.real < 0.0 for z in eigs)


# ===========================================================================
# (5) Lyapunov 证书
# ===========================================================================
@dataclass(frozen=True)
class LyapunovSolution:
    """``J' P + P J = -Q`` 的解。

    Attributes:
        P: 2x2 对称矩阵，按 ``((p11, p12), (p12, p22))`` 存储。
        positive_definite: ``P > 0`` 是否成立（即证书是否有效）。
        residual: ``||J'P + PJ + Q||_F``，数值健康度检查。
    """

    P: tuple[tuple[float, float], tuple[float, float]]
    positive_definite: bool
    residual: float


def solve_lyapunov_2x2(J: Matrix, Q: Matrix) -> LyapunovSolution | None:
    """解 2x2 Lyapunov 方程 ``J'P + PJ = -Q``（``P`` 对称，3 个未知量）。"""
    a, b = J[0][0], J[0][1]
    c, d = J[1][0], J[1][1]

    # 未知量 [p, q, r] 对应 P = [[p, q], [q, r]]
    A = [
        [2.0 * a, 2.0 * c, 0.0],
        [b, a + d, c],
        [0.0, 2.0 * b, 2.0 * d],
    ]
    rhs = [-Q[0][0], -Q[0][1], -Q[1][1]]
    sol = _solve3(A, rhs)
    if sol is None:
        return None

    p, q, r = sol
    P = ((p, q), (q, r))
    pd = p > 0.0 and p * r - q * q > 0.0

    jp = (
        (J[0][0] * p + J[1][0] * q, J[0][0] * q + J[1][0] * r),
        (J[0][1] * p + J[1][1] * q, J[0][1] * q + J[1][1] * r),
    )
    pj = (
        (p * J[0][0] + q * J[1][0], p * J[0][1] + q * J[1][1]),
        (q * J[0][0] + r * J[1][0], q * J[0][1] + r * J[1][1]),
    )
    res = math.sqrt(
        sum((jp[i][j] + pj[i][j] + Q[i][j]) ** 2 for i in range(2) for j in range(2))
    )
    return LyapunovSolution(P=P, positive_definite=pd, residual=res)


def _solve3(A: Matrix, y: Sequence[float]) -> list[float] | None:
    """3x3 高斯消元（部分选主元）；奇异返回 ``None``。"""
    M = [[A[i][0], A[i][1], A[i][2], y[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(3):
            if r == col:
                continue
            f = M[r][col] / M[col][col]
            if f == 0.0:
                continue
            for k in range(col, 4):
                M[r][k] -= f * M[col][k]
    return [M[i][3] / M[i][i] for i in range(3)]


# ===========================================================================
# (6) 前向 Euler 的绝对稳定域
# ===========================================================================
@dataclass(frozen=True)
class EtaMax:
    """步长上界 ``eta_max = min_i 2(-Re lambda_i) / |lambda_i|^2``。

    Attributes:
        eta_max: 稳定的步长上界；``hurwitz`` 为 False 时为 0。
        hurwitz: ODE 本身是否稳定（存在 ``Re lambda >= 0`` 时为 False）。
        limiting: 决定该上界的那个特征值。
        limiting_reason: ``"hurwitz-fail"`` 或 ``"eta"``。
    """

    eta_max: float
    hurwitz: bool
    limiting: complex | None = None
    limiting_reason: str = "eta"


def euler_multiplier(lam: complex, eta: float) -> complex:
    """前向 Euler 的乘子 ``1 + eta * lambda``。"""
    return 1.0 + eta * lam


def euler_stable(lam: complex, eta: float) -> bool:
    return abs(euler_multiplier(lam, eta)) < 1.0


def euler_rho(lams: Eigs, eta: float) -> float:
    """``rho(I + eta J) = max_i |1 + eta lambda_i|``。"""
    return max(abs(euler_multiplier(l, eta)) for l in lams)


def euler_eta_max(lams: Eigs) -> EtaMax:
    """由 ``|1 + eta*lambda| < 1  <=>  0 < eta < 2(-Re lambda)/|lambda|^2`` 得上界。

    ``J`` 必须是 Hurwitz 的，否则连续时间本身不稳定，返回 ``eta_max = 0``。
    """
    best = math.inf
    limiting: complex | None = None
    hurwitz = True
    for lam in lams:
        if not lam.real < 0.0:
            hurwitz = False
            continue
        v = 2.0 * (-lam.real) / (lam.real * lam.real + lam.imag * lam.imag)
        if v < best:
            best, limiting = v, lam
    if not hurwitz:
        return EtaMax(eta_max=0.0, hurwitz=False, limiting=None, limiting_reason="hurwitz-fail")
    return EtaMax(eta_max=best, hurwitz=True, limiting=limiting, limiting_reason="eta")


def euler_effective_rate(lam: complex, eta: float) -> float:
    """把离散步的有效连续衰减率还原出来：``-log|1 + eta*lambda| / eta``。

    小 ``eta`` 时它趋于 ``-Re lambda``；两者之差就是离散化带来的速率修正
    （一阶项恰为 ``eta * |lambda|^2 / 2``）。
    """
    m = abs(euler_multiplier(lam, eta))
    return math.inf if m == 0.0 else -math.log(m) / eta


def bisect_eta_max(rho_fn: Callable[[float], float], lo: float, hi: float, iters: int = 60) -> float:
    """数值求 ``rho(eta) = 1`` 的临界点，用来和解析上界互相验证。

    前提：``rho(lo) < 1 <= rho(hi)``。
    """
    if not (rho_fn(lo) < 1.0 <= rho_fn(hi)):
        return math.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rho_fn(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# 模拟：对角线性系统
# ===========================================================================
def simulate_diagonal(
    lams: Sequence[float], eta: float, steps: int, x0: Sequence[float]
) -> list[list[float]]:
    """``x_{k+1} = (I - eta * diag(lams)) x_k``，返回 ``[x_0, ..., x_K]``。

    ``|x| > 1e12`` 时提前返回（末元素即爆掉那一步）。
    """
    x = list(x0)
    out = [list(x)]
    for _ in range(steps):
        for i in range(len(x)):
            x[i] *= 1.0 - eta * lams[i]
        out.append(list(x))
        if any((not math.isfinite(v)) or abs(v) > BLOWUP for v in x):
            return out
    return out


def exact_flow_diagonal(lams: Sequence[float], t: float, x0: Sequence[float]) -> list[float]:
    """``x_i(t) = x0_i * exp(-lams_i * t)``。"""
    return [x0[i] * math.exp(-lams[i] * t) for i in range(len(lams))]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


# ===========================================================================
# (6') Heavy-ball / momentum
# ===========================================================================
def heavy_ball_matrix(eta: float, beta: float, lam: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """单模态的迭代矩阵 ``M = [[1 + beta - eta*lam, -beta], [1, 0]]``。"""
    return ((1.0 + beta - eta * lam, -beta), (1.0, 0.0))


def heavy_ball_eigs(eta: float, beta: float, lam: float) -> tuple[complex, complex]:
    return eig_2x2(heavy_ball_matrix(eta, beta, lam))


def heavy_ball_rho(eta: float, beta: float, lam: float) -> float:
    return spectral_radius(heavy_ball_eigs(eta, beta, lam))


def heavy_ball_eta_max(beta: float, lam_max: float) -> float:
    """Jury/Schur 判据：``|beta| < 1`` 且 ``0 < eta*lam < 2(1+beta)``。"""
    return 2.0 * (1.0 + beta) / lam_max


def heavy_ball_disc(eta: float, beta: float, lam: float) -> float:
    """判别式 ``tr^2 - 4*det``。``< 0`` 时两根共轭且 ``|mu| = sqrt(beta)``（与 eta 无关）。"""
    tr = 1.0 + beta - eta * lam
    return tr * tr - 4.0 * beta


def continuous_damping(eta: float, beta: float) -> float:
    """连续极限 ``x'' + a x' + grad f = 0`` 的阻尼 ``a = (1 - beta) / sqrt(eta)``。

    ``beta < 1`` 时 ``a > 0``，能量单调下降 —— **ODE 无条件稳定，与 eta 无关**。
    """
    return (1.0 - beta) / math.sqrt(eta)


def simulate_heavy_ball(
    eta: float, beta: float, lam: float, steps: int, x0: float = 1.0
) -> list[float]:
    """``x_{k+1} = x_k - eta*lam*x_k + beta*(x_k - x_{k-1})``，``x_{-1} = x_0 = x0``。"""
    out = [x0]
    xp = x = x0
    for _ in range(steps):
        xn = x - eta * lam * x + beta * (x - xp)
        xp, x = x, xn
        out.append(x)
        if (not math.isfinite(x)) or abs(x) > BLOWUP:
            break
    return out


# ===========================================================================
# (7) 噪声地板：标量 LMS / 随机逼近
# ===========================================================================
def steady_variance(eta: float, s2: float, sigma2: float) -> float:
    """常步长标量 LMS 的稳态方差。

    误差递推 ``e_{k+1} = (1 - eta*s2) e_k + eta*sqrt(s2)*eps``，``Var[eps] = sigma2``，则

        V = eta^2 s2 sigma2 / (1 - (1 - eta s2)^2) = eta sigma2 / (2 - eta s2) ~ eta sigma2 / 2

    —— **正比于 eta**，这就是常数步长的 ``O(sqrt(eta))`` 噪声地板。
    """
    d = 1.0 - (1.0 - eta * s2) ** 2
    if d <= 0.0:
        return math.inf
    return eta * eta * s2 * sigma2 / d


@dataclass(frozen=True)
class WindowStats:
    mean: float
    se: float
    n: int


def simulate_lms(
    eta_fn: Callable[[int], float],
    s2: float,
    sigma2: float,
    paths: int,
    steps: int,
    seed: int = 12345,
    x0: float | None = 1.0,
) -> list[float]:
    """返回每步的 MSD（``paths`` 条独立轨道的均值）。

    Args:
        eta_fn: ``k -> eta_k``，支持常数步长与衰减步长。
        paths: 独立轨道数。MSD 估计的标准差约为 ``V·sqrt(2/paths)``。
        steps: 步数，返回长度为 ``steps + 1`` 的序列。
        x0: 初始误差。``1.0``（默认）表示 ``e_0 = 1``；传 ``None`` 则从**平稳分布**
            ``N(0, V(eta_0))`` 抽样。

    Warning:
        从 ``e_0 = 1`` 出发时，初始条件的记忆以 ``(1 - eta·s²)^k`` 衰减 —— 在 ``eta`` 很小时
        要 ``~10/(eta·s²)`` 步才忘得掉。**窗口取得太短会把瞬态当成稳态，严重高估方差**
        （实测：``eta = 0.01``、只烧 200 步会高估 2.6 倍）。做稳态估计时请传 ``x0=None``，
        或者把 burn-in 取得足够长。
    """
    rng = random.Random(seed)
    sd = math.sqrt(sigma2)
    root_s2 = math.sqrt(s2)
    if x0 is None:
        v0 = steady_variance(eta_fn(0), s2, sigma2)
        s0 = math.sqrt(v0) if math.isfinite(v0) else 0.0
        e = [rng.gauss(0.0, s0) for _ in range(paths)]
        series = [sum(v * v for v in e) / paths]
    else:
        e = [float(x0)] * paths
        series = [float(x0) * float(x0)]
    for k in range(steps):
        eta = eta_fn(k)
        a = 1.0 - eta * s2
        g = eta * root_s2 * sd
        total = 0.0
        for i in range(paths):
            v = a * e[i] + g * rng.gauss(0.0, 1.0)
            e[i] = v
            total += v * v
        series.append(total / paths)
    return series


def window_stats(series: Sequence[float], start: int, stop: int) -> WindowStats:
    """一段序列的均值与标准误。

    注意：对 MSD 时间序列直接算 SE 会**忽略自相关而偏小**；要出真置信区间请用
    批量均值（batch means）或 HAC 估计。
    """
    seg = list(series[start:stop])
    n = len(seg)
    mean = sum(seg) / n
    var = sum((v - mean) ** 2 for v in seg) / (n - 1) if n > 1 else 0.0
    return WindowStats(mean=mean, se=math.sqrt(var / n), n=n)


# ===========================================================================
# 2x2 的瞬态增长（纯标准库实现，保住 core 的零依赖）
# ===========================================================================
# 非正规性的账：rho(M) < 1 只管 k -> inf，有限步内 ‖M^k‖ 可以先放大。
# 这里给一份 2x2 解析实现（不依赖 numpy）；任意 n 的 numpy 版在
# stability_lens.transient 里，两份实现会在 tests 里互相交叉验证。


def mat2_mul(a: Matrix, b: Matrix) -> tuple[tuple[float, float], tuple[float, float]]:
    """2x2 矩阵乘法。"""
    return ((a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
            (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]))


def mat2_norm2(a: Sequence[Sequence[complex]]) -> float:
    """2x2 的最大奇异值 ``‖a‖₂``（元素可正可复）。

    ``‖a‖₂ = sqrt(λ_max(aᴴa))``，而 2x2 Hermitian 矩阵的特征值有解析式。
    """
    a00, a01 = complex(a[0][0]), complex(a[0][1])
    a10, a11 = complex(a[1][0]), complex(a[1][1])
    g00 = abs(a00) ** 2 + abs(a10) ** 2
    g11 = abs(a01) ** 2 + abs(a11) ** 2
    g01 = a00.conjugate() * a01 + a10.conjugate() * a11
    tr = g00 + g11
    det = (g00 * g11 - abs(g01) ** 2).real
    disc = max(tr * tr - 4.0 * det, 0.0)
    return math.sqrt(max(0.5 * (tr + math.sqrt(disc)), 0.0))


def transient_curve_2x2(M: Matrix, steps: int = 60) -> list[float]:
    """``[‖M⁰‖, ‖M¹‖, …, ‖M^steps‖]``（精确 2-范数）。"""
    P = ((float(M[0][0]), float(M[0][1])), (float(M[1][0]), float(M[1][1])))
    out = [mat2_norm2(P)]
    for _ in range(steps):
        P = mat2_mul(P, M)
        out.append(mat2_norm2(P))
    return out


def transient_peak_2x2(M: Matrix, steps: int = 60) -> tuple[float, int]:
    """返回 ``(峰值, 峰值步)``。"""
    curve = transient_curve_2x2(M, steps)
    peak = max(curve)
    return peak, curve.index(peak)


def kreiss_2x2(
    M: Matrix,
    radii: Sequence[float] = (1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0,
                              100.0, 1000.0),
    n_angles: int = 96,
) -> float:
    """``K(M) = sup_{|z|>1} (|z| − 1)·‖(zI − M)⁻¹‖₂``（2x2 解析求逆 + 网格搜索）。"""
    a, b = complex(M[0][0]), complex(M[0][1])
    c, d = complex(M[1][0]), complex(M[1][1])
    best = 0.0
    for r in radii:
        for i in range(n_angles):
            theta = 2.0 * math.pi * i / n_angles
            z = complex(r * math.cos(theta), r * math.sin(theta))
            e00, e11 = z - a, z - d
            det = e00 * e11 - b * c
            if abs(det) < 1e-300:
                continue
            inv = ((e11 / det, -b / det), (-c / det, e00 / det))
            best = max(best, (r - 1.0) * mat2_norm2(inv))
    return best


def heavy_ball_transient(eta: float, beta: float, lam: float,
                         steps: int = 120) -> dict[str, float]:
    """Heavy-ball 伴随矩阵的瞬态指标。

    动量法是这个框架里瞬态放大的**唯一来源**：GD 的迭代矩阵 ``I − ηH`` 因 ``H`` 对称而
    正规，``ρ < 1`` 直接意味着误差单调下降；而动量的伴随矩阵
    ``[[1+β−ηλ, −β], [1, 0]]`` 非正规。

    两个量必须分清（实测：``β=0`` 时 ``state_peak`` 报 1.28，但那只是**状态表示**的假象）：

    * ``position_peak``：``max_k ‖e₁ᵀ M^k‖₂`` —— 最坏初始条件下**误差本身**的放大，
      这才是"误差会不会先涨后落"的答案。``β = 0`` 时它恒为 1。
    * ``state_peak``：``max_k ‖M^k‖₂`` —— 含动量记忆的整状态放大，是更保守的上界。
    """
    M = heavy_ball_matrix(eta, beta, lam)
    P = ((1.0, 0.0), (0.0, 1.0))
    state_peak = mat2_norm2(P)
    position_peak, position_step = 1.0, 0          # k=0：e₁ᵀI = (1,0)，范数 1
    for k in range(1, steps + 1):
        P = mat2_mul(P, M)
        state_peak = max(state_peak, mat2_norm2(P))
        row_norm = math.hypot(P[0][0], P[0][1])
        if row_norm > position_peak:
            position_peak, position_step = row_norm, k
    return {
        "rho": heavy_ball_rho(eta, beta, lam),
        "position_peak": position_peak,
        "position_peak_step": float(position_step),
        "state_peak": state_peak,
        "kreiss": kreiss_2x2(M),
    }
