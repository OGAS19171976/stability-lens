"""stability_lens.monitor — 训练过程中的在线稳定性监控（发散前告警）。

前面的模块回答"这个学习率上界是多少"，本模块回答**"现在这一版跑起来安不安全"**：

* 每到检查点，在当前 ``θ_k`` 上（不是只在 ``θ*`` 上）用**热启动的幂迭代**估一次 ``λ_max``；
* 由此得到**当前**的步长上界 ``η_max(θ_k) = 2(1+β)/λ_max(θ_k)``；
* 一旦 ``η > η_max(θ_k)`` 就告警 —— 这一次告警是在损失爆掉**之前**给出的。

热启动是让它可用的关键：幂迭代从上一次的极大特征向量出发，几次迭代就够，
不必每次从头收敛（冷启动要上百步）。

代价要说清楚：每次检查要 ``2 × power_iters`` 次梯度评估（FD 版 HVP）。
``check_every=50, power_iters=8`` 大约是训练步数的 32% 额外开销 ——
它是**诊断模式**，不是默认打开的东西。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .spectral import power_iteration

Array = np.ndarray
GradFn = Callable[[Array], tuple[float, Array]]
HvpFn = Callable[[Array, Array], Array]

__all__ = [
    "MonitorEvent",
    "MonitorReport",
    "StabilityMonitor",
    "run_with_monitor",
]


@dataclass
class MonitorEvent:
    """一次检查点的读数。"""

    step: int
    lambda_max: float
    eta_max: float
    ratio: float
    kind: str          # "ok" | "alarm" | "indefinite"
    message: str = ""


@dataclass
class MonitorReport:
    """一次监控会话的完整记录。"""

    eta: float
    beta: float
    steps_run: int
    events: list[MonitorEvent] = field(default_factory=list)
    lambda_max_trace: list[tuple[int, float]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    alarm_step: int | None = None
    blowup_step: int | None = None
    diverged: bool = False

    @property
    def alarmed(self) -> bool:
        return self.alarm_step is not None

    @property
    def lead_time(self) -> int | None:
        """告警比损失爆掉提前了多少步（爆掉之前告警才有意义）。"""
        if self.alarm_step is None or self.blowup_step is None:
            return None
        return self.blowup_step - self.alarm_step

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def render(self) -> str:
        head = (f"  η = {self.eta:.6g}（β = {self.beta:g}）跑了 {self.steps_run} 步，"
                f"检查 {len(self.events)} 次")
        if self.alarmed:
            line = (f"  告警@第 {self.alarm_step} 步"
                    + (f"，损失爆掉@第 {self.blowup_step} 步 → 提前 {self.lead_time} 步"
                       if self.blowup_step is not None else "，损失未爆（非线性饱和）"))
        else:
            line = "  未告警"
        return head + "\n" + line


class StabilityMonitor:
    """逐步估算 ``λ_max(θ_k)`` 并在越过步长上界时告警。"""

    def __init__(
        self,
        eta: float,
        beta: float = 0.0,
        hvp: HvpFn | None = None,
        check_every: int = 25,
        power_iters: int = 8,
        first_check_iters: int | None = None,
        safety: float = 1.0,
        blowup_loss: float | None = None,
        seed: int = 0,
    ) -> None:
        if hvp is None:
            raise ValueError("需要 hvp：一个 (theta, v) -> H(θ)v 的可调用对象")
        if check_every <= 0:
            raise ValueError("check_every 必须为正")
        self.eta = float(eta)
        self.beta = float(beta)
        self.hvp = hvp
        self.check_every = int(check_every)
        self.power_iters = int(power_iters)
        # 第一次检查没有可复用的特征向量，只能冷启动 —— 若预算与热启动相同，
        # Rayleigh 商偏低会让 η_max 被高估、**告警被推迟**（实测线性模型上推迟了 40 步）。
        # 所以第一次给更大的预算。
        self.first_check_iters = int(first_check_iters
                                     if first_check_iters is not None
                                     else max(power_iters, 60))
        self.safety = float(safety)
        self.blowup_loss = blowup_loss
        self._rng = np.random.default_rng(seed)
        self._vec: Array | None = None
        self._checks = 0
        self.report = MonitorReport(eta=self.eta, beta=self.beta, steps_run=0)

    # ------------------------------------------------------------------
    def _lambda_max(self, theta: Array) -> float:
        """热启动幂迭代估当前点的 λ_max（首次检查用更大的冷启动预算）。"""
        cold = self._vec is None
        v0 = self._vec if self._vec is not None else self._rng.normal(size=theta.size)
        iters = self.first_check_iters if cold else self.power_iters
        res = power_iteration(lambda w, t=theta: self.hvp(t, w), theta.size,
                              iters=iters, tol=0.0, v0=v0, keep_trace=False)
        self._vec = res.vec
        self._checks += 1
        return float(res.lam)

    def observe(self, step: int, theta: Array, loss: float,
                grad_norm: float | None = None) -> MonitorEvent | None:
        """每个训练步调用一次；只在检查点做幂迭代并可能返回一个事件。"""
        r = self.report
        r.steps_run = step + 1
        r.losses.append(float(loss))
        if grad_norm is not None:
            r.grad_norms.append(float(grad_norm))
        if self.blowup_loss is not None and r.blowup_step is None:
            if (not math.isfinite(loss)) or loss > self.blowup_loss:
                r.blowup_step = step
        if step % self.check_every != 0:
            return None

        lam = self._lambda_max(theta)
        r.lambda_max_trace.append((step, lam))
        if lam <= 0.0:
            event = MonitorEvent(step=step, lambda_max=lam, eta_max=float("inf"),
                                 ratio=0.0, kind="indefinite",
                                 message="λ_max ≤ 0：当前方向是负曲率/零曲率，局部判据不适用")
        else:
            eta_max = 2.0 * (1.0 + self.beta) / lam
            ratio = self.eta / eta_max
            if ratio > self.safety:
                event = MonitorEvent(
                    step=step, lambda_max=lam, eta_max=eta_max, ratio=ratio, kind="alarm",
                    message=(f"η={self.eta:.6g} 已越过 η_max(θ_{step})={eta_max:.6g}"
                             f"（{ratio:.3f} 倍）：按线性化判据这一步会放大，"
                             "继续跑大概率发散"),
                )
                if r.alarm_step is None:
                    r.alarm_step = step
            else:
                event = MonitorEvent(step=step, lambda_max=lam, eta_max=eta_max,
                                     ratio=ratio, kind="ok")
        r.events.append(event)
        return event


def run_with_monitor(
    grad_fn: GradFn,
    hvp_fn: HvpFn,
    theta0: Array,
    eta: float,
    steps: int = 400,
    beta: float = 0.0,
    blowup_loss: float | None = None,
    check_every: int = 25,
    power_iters: int = 8,
    safety: float = 1.0,
    seed: int = 0,
    stop_on_blowup: bool = False,
) -> MonitorReport:
    """跑一轮固定步长训练，同时在线监控。

    ``hvp_fn(theta, v)`` 是 Hessian-向量积。返回 :class:`MonitorReport`。
    """
    th = np.array(theta0, dtype=float)
    prev = th.copy()
    monitor = StabilityMonitor(eta=eta, beta=beta, hvp=hvp_fn, check_every=check_every,
                               power_iters=power_iters, safety=safety,
                               blowup_loss=blowup_loss, seed=seed)
    for k in range(steps):
        loss, g = grad_fn(th)
        monitor.observe(k, th, loss, float(np.linalg.norm(g)))
        if stop_on_blowup and monitor.report.blowup_step is not None:
            break
        nxt = th - eta * g + beta * (th - prev)
        prev, th = th, nxt
        if not np.all(np.isfinite(th)):
            break
    # 收尾：最后一步也要被记录
    loss, g = grad_fn(th)
    monitor.observe(min(steps, monitor.report.steps_run), th, loss, float(np.linalg.norm(g)))
    return monitor.report
