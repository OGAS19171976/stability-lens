"""``stability_lens.monitor`` 的测试。

大多数用例跑在**闭式二次目标**上：``λ_max = eigvalsh(A)[-1]`` 精确已知、
发散没有歧义，因此"该不该报警"有唯一正确答案。
"""

from __future__ import annotations

import unittest

import numpy as np

from stability_lens import datasets, experiment, monitor, spectral


def _quadratic(n=6, seed=0, lam_lo=0.5, lam_hi=2.0):
    """返回 (grad_fn, hvp, λ_max, θ0)，损失为 1/2 θᵀAθ。"""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.linspace(lam_lo, lam_hi, n)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    grad_fn = lambda th: (0.5 * float(th @ A @ th), A @ th)   # noqa: E731
    hvp = lambda th, v: A @ v                                  # noqa: E731
    return grad_fn, hvp, float(lam[-1]), rng.normal(size=n)


class TestMonitorBasics(unittest.TestCase):
    def test_requires_hvp(self):
        with self.assertRaises(ValueError):
            monitor.StabilityMonitor(eta=0.1)

    def test_rejects_bad_check_every(self):
        with self.assertRaises(ValueError):
            monitor.StabilityMonitor(eta=0.1, hvp=lambda t, v: v, check_every=0)

    def test_indefinite_curvature_is_reported_not_crashed(self):
        """负曲率时 λ_max < 0：应当报 indefinite，而不是拿它算 η_max。"""
        gram = lambda th: (0.5 * float(th @ th), th.copy())     # noqa: E731
        hvp = lambda th, v: -2.0 * v                            # noqa: E731
        rep = monitor.run_with_monitor(gram, hvp, np.ones(4), eta=0.1, steps=4,
                                       check_every=1)
        self.assertTrue(rep.events)
        self.assertEqual(rep.events[0].kind, "indefinite")
        self.assertIsNone(rep.alarm_step)

    def test_first_check_uses_larger_budget(self):
        """回归：冷启动预算若与热启动相同，Rayleigh 商偏低会让告警被推迟。"""
        grad_fn, hvp, lam_max, th0 = _quadratic(n=8, seed=1)
        eta_max = 2.0 / lam_max
        rep = monitor.run_with_monitor(grad_fn, hvp, th0, eta=1.01 * eta_max,
                                       steps=60, check_every=20, power_iters=8)
        self.assertEqual(rep.alarm_step, 0, "第一次检查就必须估准 λ_max 并报警")
        lam0 = rep.lambda_max_trace[0][1]
        self.assertLess(abs(lam0 - lam_max) / lam_max, 1e-3)

    def test_warm_start_converges_to_exact_lambda_max(self):
        """8 次热启动迭代大约给到 1e-5 相对精度 —— 判告警够用，但别拿它去发表 λ_max。

        幂迭代的收敛率是 (λ₂/λ₁)^k，所以"够不够准"只影响告警阈值的小数点后几位，
        不影响该不该报警这个二值判断。Rayleigh 商是 λ_max 的**下界**（未收敛时偏低）。
        """
        grad_fn, hvp, lam_max, th0 = _quadratic(n=8, seed=2)
        rep = monitor.run_with_monitor(grad_fn, hvp, th0, eta=0.5 * 2.0 / lam_max,
                                       steps=200, check_every=20, power_iters=8)
        for _, lam in rep.lambda_max_trace:
            self.assertLess(abs(lam - lam_max) / lam_max, 1e-4)
            self.assertLessEqual(lam, lam_max * (1 + 1e-9))


class TestAlarmDecision(unittest.TestCase):
    def setUp(self):
        self.grad_fn, self.hvp, self.lam_max, self.th0 = _quadratic(n=6, seed=3)
        self.eta_max = 2.0 / self.lam_max

    def _run(self, frac, steps=200):
        loss0 = self.grad_fn(self.th0)[0]
        return monitor.run_with_monitor(
            self.grad_fn, self.hvp, self.th0, eta=frac * self.eta_max, steps=steps,
            check_every=10, blowup_loss=4.0 * loss0)

    def test_stable_configurations_do_not_alarm(self):
        for frac in (0.5, 0.9, 0.99):
            rep = self._run(frac)
            self.assertFalse(rep.alarmed, f"frac={frac} 不应告警（误报）")
            self.assertTrue(np.isfinite(rep.final_loss))

    def test_unstable_configurations_alarm_immediately(self):
        for frac in (1.01, 1.1, 1.3):
            rep = self._run(frac)
            self.assertTrue(rep.alarmed, f"frac={frac} 应当告警")
            self.assertEqual(rep.alarm_step, 0, "二次目标上第 0 步就能判定")

    def test_alarm_precedes_blowup(self):
        rep = self._run(1.3)
        self.assertIsNotNone(rep.blowup_step)
        self.assertLess(rep.alarm_step, rep.blowup_step)
        self.assertGreater(rep.lead_time, 0)

    def test_report_render_mentions_alarm_and_lead(self):
        rep = self._run(1.3)
        text = rep.render()
        self.assertIn("告警@第 0 步", text)
        self.assertIn("提前", text)

    def test_safety_factor_delays_alarm(self):
        """safety > 1 时应当更保守（同样的 η 不再报）。"""
        rep = monitor.run_with_monitor(self.grad_fn, self.hvp, self.th0,
                                       eta=1.05 * self.eta_max, steps=60,
                                       check_every=10, safety=1.5)
        self.assertFalse(rep.alarmed)


class TestMonitorStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = datasets.make_synthetic(n=200, d=5, seed=0)
        cls.study = experiment.run_monitor_study(
            dataset=ds, h=5, fracs=(0.5, 1.3), steps=120, check_every=20,
            seed=0, with_mlp=True)

    def test_no_false_alarm_and_no_miss(self):
        self.assertEqual(self.study.false_alarms, 0)
        self.assertEqual(self.study.missed, 0)

    def test_linear_rows_match_theory(self):
        lin = {r.frac: r for r in self.study.rows if r.model == "linear"}
        self.assertFalse(lin[0.5].alarmed)
        self.assertTrue(lin[1.3].alarmed)
        self.assertEqual(lin[1.3].alarm_step, 0)
        self.assertGreater(lin[1.3].lead_time, 0)

    def test_mlp_from_theta_star_alarms_when_above_bound(self):
        rows = {(r.model, r.start, r.frac): r for r in self.study.rows}
        self.assertFalse(rows[("mlp", "θ*", 0.5)].alarmed)
        self.assertTrue(rows[("mlp", "θ*", 1.3)].alarmed)

    def test_render_contains_table_and_notes(self):
        text = experiment.render_monitor_study(self.study)
        for key in ("告警步", "爆掉步", "提前量", "误报", "漏报"):
            self.assertIn(key, text)
        self.assertNotIn("nan", text)

    def test_json_shape(self):
        payload = experiment.monitor_as_json(self.study)
        self.assertEqual(payload["false_alarms"], 0)
        self.assertEqual(len(payload["rows"]), len(self.study.rows))
        self.assertIn("start", payload["rows"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
