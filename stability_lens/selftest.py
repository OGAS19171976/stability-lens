"""stability_lens.selftest — 可复现的数值证据表。

对同一组参数做 25 项断言，覆盖 A（线性迭代）/ B（Heavy-ball）/ C（噪声地板）三组。
这些参数与 `incremental-ode/web/test-core.js` 完全一致，因此两套独立实现（Python / JS）
的**解析行**应当逐位相同 —— 这就是跨语言交叉验证。

运行::

    python -m stability_lens selftest
    python -m stability_lens selftest --json
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from . import core
from .report import render_evidence_table

__all__ = ["Check", "run_checks", "render", "as_json", "MC_BUDGET"]

# 蒙特卡洛预算（纯 Python 循环，故刻意压小；加大预算只影响 MC 行的精度）
MC_BUDGET = {
    "variance_paths": 2000,
    "variance_steps": 400,
    "variance_window": (250, 400),
    "slope_paths": 600,
    "slope_steps": 300,
    "slope_window": (200, 300),
    "decay_paths": 600,
    "decay_steps": 2000,
}


@dataclass
class Check:
    ok: bool
    name: str
    theory: str
    measured: str
    relerr: str
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.ok else "FAIL"


def _num(v: float, digits: int = 6) -> str:
    if not math.isfinite(v):
        return str(v)
    if v != 0.0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def _close(name: str, measured: float, theory: float, tol: float, note: str = "") -> Check:
    err = abs(measured) if theory == 0 else abs(measured - theory) / abs(theory)
    return Check(ok=err < tol, name=name, theory=_num(theory), measured=_num(measured),
                 relerr=f"{err:.1e}", note=note)


def _ok(name: str, cond: bool, note: str = "", theory: str = "—", measured: str | None = None) -> Check:
    return Check(ok=bool(cond), name=name, theory=theory,
                 measured=measured if measured is not None else ("true" if cond else "false"),
                 relerr="—", note=note)


def run_checks(budget: dict[str, Any] | None = None) -> list[Check]:
    """跑完全部断言，返回结果列表（不打印）。"""
    b = dict(MC_BUDGET)
    if budget:
        b.update(budget)
    out: list[Check] = []

    # ---------------- 0. 基础 ----------------
    e1, e2 = core.eig_2x2([[0.0, -1.0], [1.0, 0.0]])
    out.append(_ok("eig_2x2: 旋转矩阵 → ±i", abs(e1.imag - 1) < 1e-15 and abs(e2.imag + 1) < 1e-15))

    lyap = core.solve_lyapunov_2x2([[-1.0, 0.0], [0.0, -4.0]], [[1.0, 0.0], [0.0, 1.0]])
    out.append(_ok("(5) J^T P + P J = −I 且 P > 0",
                   lyap is not None and lyap.positive_definite and lyap.residual < 1e-12,
                   theory="P = diag(0.5000, 0.1250)",
                   measured=f"P = diag({lyap.P[0][0]:.4f}, {lyap.P[1][1]:.4f})，残差 {lyap.residual:.1e}"))

    # ---------------- A. 线性增量迭代 ----------------
    lams = [1.0, 4.0]
    J = [complex(-v, 0.0) for v in lams]
    em = core.euler_eta_max(J)
    numeric = core.bisect_eta_max(lambda e: core.euler_rho(J, e), 1e-12, 2.0 / 4.0, 60)
    out.append(_close("A · η_max = 2/λ_max（解析 vs 二分）", numeric, em.eta_max, 1e-12, "λ = {1, 4}"))
    out.append(_close("A · η_max 数值", em.eta_max, 0.5, 1e-15))

    for eta, want in ((0.025, 0.975), (0.075, 0.925), (0.125, 0.875), (0.2, 0.8)):
        out.append(_close(f"A · ρ(I−ηA), η = {eta}", core.euler_rho(J, eta), want, 1e-14))

    def run(eta: float) -> float:
        traj = core.simulate_diagonal(lams, eta, 2000, [1.0, 1.0])
        return core.norm(traj[-1])

    lo, hi = run(0.99 * em.eta_max), run(1.01 * em.eta_max)
    out.append(_ok("A · 边界两侧（0.99 收敛 / 1.01 发散）", lo < 1e-8 and hi > 1e10,
                   measured=f"{lo:.2e} vs {hi:.2e}"))

    lc = complex(-1.0, 3.0)
    out.append(_close("A · 复 λ = −1±3i 的 η_max = 2/|λ|²", core.euler_eta_max([lc]).eta_max, 0.2, 1e-15))

    # ---------------- B. Heavy-ball ----------------
    lam_max, beta = 2.0, 0.5
    for bt in (0.0, 0.2, 0.5, 0.8, 0.9):
        th = core.heavy_ball_eta_max(bt, lam_max)
        num = core.bisect_eta_max(lambda e, _b=bt: core.heavy_ball_rho(e, _b, lam_max), 1e-12, 4.0, 70)
        out.append(_close(f"B · η_max(β={bt}) = 2(1+β)/λ_max", num, th, 1e-12))

    eta_c = 0.7
    out.append(_ok(f"B · η={eta_c} 落在复根区", core.heavy_ball_disc(eta_c, beta, lam_max) < 0))
    out.append(_close("B · 复根区 ρ = √β", core.heavy_ball_rho(eta_c, beta, lam_max), math.sqrt(beta), 1e-14))

    traj = core.simulate_heavy_ball(eta_c, beta, lam_max, 1800, 1.0)
    n0, n1 = abs(traj[600]), abs(traj[1800])
    meas = math.exp((math.log(n1) - math.log(n0)) / 1200.0)
    out.append(_close("B · 实测增长率 vs ρ（几何平均）", meas, core.heavy_ball_rho(eta_c, beta, lam_max), 1e-3))

    eta_gap = 1.5 * core.heavy_ball_eta_max(beta, lam_max)
    a = core.continuous_damping(eta_gap, beta)
    t2 = core.simulate_heavy_ball(eta_gap, beta, lam_max, 200, 1.0)
    blow = next((i for i, v in enumerate(t2) if abs(v) > core.BLOWUP), -1)
    out.append(_ok("B · GAP：ODE 阻尼 a > 0 而离散爆炸", a > 0 and 0 < blow < 60,
                   theory="a = 0.3333 > 0",
                   measured=f"第 {blow} 步 |x|>1e12，ρ={core.heavy_ball_rho(eta_gap, beta, lam_max):.4f}"))

    # ---------------- C. 噪声地板 ----------------
    eta_n, s2, sig2 = 0.05, 1.0, 1.0
    th = core.steady_variance(eta_n, s2, sig2)
    series = core.simulate_lms(lambda k: eta_n, s2, sig2,
                               b["variance_paths"], b["variance_steps"], seed=20240607, x0=None)
    w = core.window_stats(series, *b["variance_window"])
    out.append(_close("C · 稳态方差 V = ησ²/(2−ηs²) vs MC", w.mean, th, 0.05,
                      note=f"x0=平稳分布；±1.96SE = {1.96 * w.se:.2e}（时间序列自相关，SE 偏小）"))

    e_small = 0.002
    out.append(_close("C · 小 η 渐近 V/(ησ²/2) → 1",
                      core.steady_variance(e_small, s2, sig2) / (e_small * sig2 / 2.0), 1.0, 2e-3))

    pts = []
    for i, e in enumerate((0.01, 0.02, 0.04, 0.08)):
        # x0=None：从平稳分布出发。否则 eta=0.01 时初始条件要 ~10/(eta*s^2) 步才忘掉，
        # 短窗口会把瞬态当稳态（这正是本项目实际踩到的坑，见 core.simulate_lms 的 Warning）。
        ser = core.simulate_lms(lambda k, _e=e: _e, s2, sig2,
                                b["slope_paths"], b["slope_steps"], seed=500 + i, x0=None)
        pts.append((math.log(e), math.log(core.window_stats(ser, *b["slope_window"]).mean)))
    slope = (pts[3][1] - pts[0][1]) / (pts[3][0] - pts[0][0])
    out.append(_close("C · log-log 斜率 d log MSD / d log η", slope, 1.0, 0.06))

    decay = core.simulate_lms(lambda k: 2.0 / (k + 10.0), s2, sig2,
                              b["decay_paths"], b["decay_steps"], seed=909)
    floor = core.steady_variance(0.2, s2, sig2)
    last = decay[b["decay_steps"]]
    out.append(_ok("C · 衰减步长远低于同期的常步长噪声地板", last < 0.01 * floor,
                   measured=f"MSD({b['decay_steps']}) = {last:.2e}，地板 = {floor:.4f}"))
    out.append(_close("C · 衰减步长 MSD·k → c/2（1/k 律）", last * b["decay_steps"], 1.0, 0.35))

    def euler_err(eta: float, T: float = 1.0) -> float:
        x, n = 1.0, int(round(T / eta))
        for _ in range(n):
            x *= 1.0 - eta
        return abs(x - math.exp(-T))

    ratios = [euler_err(e) / e for e in (0.05, 0.01, 0.002)]
    out.append(_close("C · 前向 Euler 全局误差 = O(η)", ratios[2], ratios[1], 0.01,
                      note="err/η = " + " / ".join(f"{v:.5f}" for v in ratios)))

    return out


def render(checks: list[Check] | None = None, color: bool = False) -> str:
    checks = checks if checks is not None else run_checks()
    rows = [[c.status, c.name, c.theory, c.measured, c.relerr, c.note] for c in checks]
    body = render_evidence_table(rows, ["状态", "检验项", "理论值", "实测值", "相对误差", "备注"])
    npass = sum(1 for c in checks if c.ok)
    head = "stability-lens · 数值证据表（Python 独立实现，参数与 web/test-core.js 一致）"
    tail = f"{npass} passed, {len(checks) - npass} failed, {len(checks)} checks"
    return f"{head}\n{body}\n{tail}"


def as_json(checks: list[Check] | None = None) -> dict[str, Any]:
    checks = checks if checks is not None else run_checks()
    return {
        "passed": sum(1 for c in checks if c.ok),
        "failed": sum(1 for c in checks if not c.ok),
        "total": len(checks),
        "checks": [
            {"status": c.status, "name": c.name, "theory": c.theory, "measured": c.measured,
             "relerr": c.relerr, "note": c.note}
            for c in checks
        ],
    }
