"""stability-lens 的估计器 / 训练 / 实验层测试。

覆盖：
* 解析梯度 vs 中心差分（独立对拍）；
* 线性模型：闭式 Hessian ``XᵀX/n`` 作为**不依赖差分**的真值，验证 HVP + 幂迭代；
* MLP：装配 Hessian 的谱 vs 幂迭代；
* 二次目标上的实测边界必须精确等于 ``2/λ_max``（这是整套测量装置的基准）；
* 饱和陷阱：大扰动会把发散误判成稳定（回归测试，防止以后改回去）。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from stability_lens import datasets, experiment, models, spectral, training


def _linear_problem(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.1 * rng.normal(size=n)
    X, y, _ = datasets.standardize(X, y)
    return X, y


class TestGradients(unittest.TestCase):
    def test_mlp_gradient_matches_finite_difference(self):
        X, y = _linear_problem()
        d = X.shape[1]
        h = 5
        theta = models.init_params(d, h, seed=3)
        _, g = models.mlp_loss_grad(theta, X, y, d, h)
        err = models.gradient_check(lambda t: models.mlp_loss(t, X, y, d, h), g, theta)
        self.assertLess(err, 1e-7)

    def test_linear_gradient_matches_finite_difference(self):
        X, y = _linear_problem()
        theta = np.zeros(X.shape[1])
        _, g = models.linear_loss_grad(theta, X, y)
        err = models.gradient_check(lambda t: models.linear_loss_grad(t, X, y)[0], g, theta)
        self.assertLess(err, 1e-7)

    def test_l2_shows_up_in_gradient(self):
        X, y = _linear_problem()
        d = X.shape[1]
        h = 4
        theta = models.init_params(d, h, seed=1)
        _, g0 = models.mlp_loss_grad(theta, X, y, d, h, l2=0.0)
        _, g1 = models.mlp_loss_grad(theta, X, y, d, h, l2=0.1)
        self.assertTrue(np.allclose(g1 - g0, 0.1 * theta, atol=1e-12))


class TestHVPAndSpectrum(unittest.TestCase):
    def test_hvp_matches_assembled_hessian(self):
        X, y = _linear_problem(n=120, d=6, seed=1)
        d = X.shape[1]
        h = 4
        theta = models.init_params(d, h, seed=2)
        H = models.mlp_hessian(theta, X, y, d, h)
        rng = np.random.default_rng(0)
        v = rng.normal(size=theta.size)
        self.assertLess(np.linalg.norm(models.mlp_hvp(theta, X, y, d, h, v) - H @ v)
                        / np.linalg.norm(H @ v), 1e-6)

    def test_assembled_hessian_is_symmetric(self):
        X, y = _linear_problem(n=120, d=6, seed=1)
        d, h = X.shape[1], 4
        theta = models.init_params(d, h, seed=2)
        H = models.mlp_hessian(theta, X, y, d, h)
        self.assertLess(np.linalg.norm(H - H.T) / np.linalg.norm(H), 1e-12)

    def test_power_iteration_matches_closed_form_linear_hessian(self):
        """线性模型的 Hessian 有闭式解 XᵀX/n —— 不依赖任何差分的独立真值。"""
        X, y = _linear_problem(n=300, d=10, seed=5)
        H = models.linear_hessian(X)
        truth = spectral.spectral_summary(H)["lam_max"]
        theta = np.zeros(X.shape[1])

        def hvp(v):
            eps = 1e-5
            gp = models.linear_loss_grad(theta + eps * v, X, y)[1]
            gm = models.linear_loss_grad(theta - eps * v, X, y)[1]
            return (gp - gm) / (2 * eps)

        res = spectral.power_iteration(hvp, X.shape[1], iters=500, tol=1e-13)
        self.assertLess(abs(res.lam - truth) / truth, 1e-6)
        self.assertGreater(res.lam, 0.0)
        self.assertIsNotNone(res.vec)
        self.assertAlmostEqual(float(np.linalg.norm(res.vec)), 1.0, places=12)

    def test_power_iteration_matches_assembled_mlp_hessian(self):
        X, y = _linear_problem(n=150, d=6, seed=7)
        d, h = X.shape[1], 5
        theta = models.init_params(d, h, seed=8)
        H = models.mlp_hessian(theta, X, y, d, h)
        truth = spectral.spectral_summary(H)["lam_max"]
        res = spectral.power_iteration(lambda v: models.mlp_hvp(theta, X, y, d, h, v),
                                       theta.size, iters=600, tol=1e-13)
        self.assertLess(abs(res.lam - truth) / abs(truth), 1e-5)

    def test_power_iteration_detects_dominant_mode(self):
        """对角矩阵：幂迭代必须收敛到最大对角元。"""
        diag = np.array([1.0, 3.0, 2.0])
        res = spectral.power_iteration(lambda v: diag * v, 3, iters=400, tol=1e-14)
        self.assertAlmostEqual(res.lam, 3.0, places=10)


class TestTraining(unittest.TestCase):
    def test_damped_newton_finds_quadratic_minimum(self):
        X, y = _linear_problem(n=200, d=6, seed=3)
        gf = lambda th: models.linear_loss_grad(th, X, y)  # noqa: E731
        hf = lambda th: models.linear_hessian(X)          # noqa: E731
        res = training.train_damped_newton(gf, hf, np.zeros(X.shape[1]), steps=30, tol=1e-12)
        theta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.assertLess(res.final_grad_norm, 1e-8)
        self.assertTrue(np.allclose(res.theta, theta_ls, atol=1e-8))

    def test_heavy_ball_iteration_matches_recurrence(self):
        """``train`` 的 heavy-ball 更新必须严格等于手写递推。"""
        gf = lambda th: models.linear_loss_grad(th, X, y)  # noqa: E731
        X, y = _linear_problem(seed=4)
        theta0 = np.zeros(X.shape[1])
        eta, beta = 0.3, 0.5
        got = training.train(gf, theta0, eta=eta, steps=5, beta=beta).theta
        th, prev = theta0.copy(), theta0.copy()
        for _ in range(5):
            _, g = gf(th)
            nxt = th - eta * g + beta * (th - prev)
            prev, th = th, nxt
        self.assertTrue(np.allclose(got, th, atol=1e-12))

    def test_measured_boundary_on_quadratic_is_exact(self):
        """二次目标上 ``η_max = 2/λ_max`` 是精确的 —— 整套测量装置的基准测试。"""
        X, y = _linear_problem(n=250, d=8, seed=11)
        H = models.linear_hessian(X)
        lam_max = spectral.spectral_summary(H)["lam_max"]
        theta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
        power = spectral.power_iteration(
            lambda v: H @ v, X.shape[1], iters=500, tol=1e-14)
        meas = training.boundary_from_prediction(
            lambda th: models.linear_loss_grad(th, X, y), theta_ls, 2.0 / lam_max,
            beta=0.0, bisect_iters=22, steps=150, direction=power.vec)
        self.assertTrue(meas["valid"])
        self.assertLess(abs(float(meas["eta_max"]) - 2.0 / lam_max) / (2.0 / lam_max), 0.01)

    def test_perturbation_size_truncates_growth(self):
        """回归测试：实测增长强依赖扰动幅度 —— 非线性饱和会把它截断。

        本项目实测过这个坑：扰动取 1e-3 时，MLP 的实测边界被判到 **145%** 偏差处
        （tanh 饱和后距离不再增长，后半段渐近倍率回落到 ≈1，被误判成"稳定"）；
        把扰动压到 1e-8、判据改用整段几何倍率之后，偏差回到 **0.01%**。
        所以 ``decay_test`` 的默认 ``delta_scale`` 是 1e-8，且**不**使用尾部倍率做判据。
        """
        X, y = _linear_problem(n=200, d=6, seed=13)
        d, h = X.shape[1], 8
        gf = lambda th: models.mlp_loss_grad(th, X, y, d, h, l2=1e-3)   # noqa: E731
        hf = lambda th: models.mlp_hessian(th, X, y, d, h, l2=1e-3)     # noqa: E731
        theta0 = models.init_params(d, h, seed=14)
        theta_star = training.train_damped_newton(gf, hf, theta0, steps=60, tol=1e-10).theta
        lam_max = spectral.spectral_summary(hf(theta_star))["lam_max"]
        power = spectral.power_iteration(
            lambda v: models.mlp_hvp(theta_star, X, y, d, h, v, l2=1e-3),
            theta_star.size, iters=400, tol=1e-12)
        eta_bad = 3.0 * (2.0 / lam_max)          # 明确在稳定域之外

        small = training.decay_test(gf, theta_star, eta_bad, steps=150,
                                    delta_scale=1e-8, direction=power.vec)
        self.assertFalse(small["decayed"], "线性区测量必须判为不稳定")
        self.assertGreater(small["rate"], 1.0)

        # 大扰动：同样判为不稳定，但增长被饱和截断，量级远低于线性理论的预言
        large = training.decay_test(gf, theta_star, eta_bad, steps=150,
                                    delta_scale=1e-3, direction=power.vec,
                                    blowup=1e6)
        linear_growth = abs(1.0 - eta_bad * lam_max) ** 150
        self.assertGreater(large["e_ratio"], 1e3, "大扰动应当先明显长起来")
        self.assertLess(large["e_ratio"], linear_growth * 1e-6,
                        "饱和把增长截断在线性理论预言的倍率之下")


class TestDatasets(unittest.TestCase):
    def test_synthetic_is_reproducible(self):
        a = datasets.make_synthetic(n=50, d=4, seed=1)
        b = datasets.make_synthetic(n=50, d=4, seed=1)
        self.assertTrue(np.allclose(a.X, b.X))
        self.assertTrue(np.allclose(a.y, b.y))

    def test_standardize_centers_and_scales(self):
        X, y = _linear_problem(seed=2)
        Xs, ys, _ = datasets.standardize(X * 5 + 3, y * 2 - 1)
        self.assertTrue(np.allclose(Xs.mean(axis=0), 0.0, atol=1e-12))
        self.assertTrue(np.allclose(Xs.std(axis=0), 1.0, atol=1e-12))
        self.assertAlmostEqual(float(ys.mean()), 0.0, places=12)

    def test_parse_dt_handles_both_formats(self):
        self.assertIsNotNone(datasets.parse_dt("2026-08-01 19:51:29"))
        self.assertIsNotNone(datasets.parse_dt("2026/08/01 08:34:14"))
        self.assertIsNone(datasets.parse_dt("not a date"))


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = datasets.make_synthetic(n=400, d=6, seed=0)
        cls.res = experiment.run_study(dataset=ds, h=6, betas=(0.0, 0.5),
                                       train_steps=60, test_steps=120,
                                       bisect_iters=10)

    def test_reaches_strict_local_minimum(self):
        self.assertGreater(self.res.lam_min, 0.0)
        self.assertLess(self.res.train_grad_norm, 1e-6)

    def test_gradient_check_passes(self):
        self.assertLess(self.res.grad_check, 1e-7)

    def test_power_iteration_matches_assembled_hessian(self):
        rel = abs(self.res.lam_max_power - self.res.lam_max_exact) / self.res.lam_max_exact
        self.assertLess(rel, 1e-6)

    def test_predicted_boundary_matches_measurement(self):
        for row in self.res.rows:
            self.assertTrue(row.valid, row.label)
            self.assertLess(row.rel_err, 0.05, f"{row.label}: 偏差 {row.rel_err:.2%}")

    def test_monotone_in_beta(self):
        self.assertLess(self.res.rows[0].eta_measured, self.res.rows[-1].eta_measured)

    def test_linear_reference_is_exact(self):
        li = self.res.linear
        self.assertLess(li["power_rel_err"], 1e-6)
        self.assertLess(li["rel_err"], 0.01)

    def test_render_contains_key_fields(self):
        text = experiment.render_study(self.res)
        for key in ("λ_max", "预测 η_max", "实测边界", "(1+β) 缩放检验", "线性模型对照"):
            self.assertIn(key, text)
        self.assertNotIn("nan", text)


class TestOptionalNumpy(unittest.TestCase):
    """numpy 只被 predict 需要：缺了它要给一句人话，其余子命令必须照常工作。"""

    LAZY = ("datasets", "experiment", "models", "spectral", "training")

    def _hide_lazy_modules(self):
        """把惰性导入的子模块从 sys.modules **和父包属性**上一起摘掉。

        只摘 sys.modules 是不够的：``from . import datasets`` 会先看父包上有没有
        这个属性，有就直接绑定、**不会**重新执行模块代码 —— 于是 ``import numpy``
        根本不会被触发，测试会假装通过。
        """
        import sys

        import stability_lens

        saved_mods = {m: sys.modules.pop(m) for m in list(sys.modules)
                      if m.startswith("stability_lens.")
                      and m.split(".", 1)[1] in self.LAZY}
        saved_attrs = {n: getattr(stability_lens, n) for n in self.LAZY
                       if hasattr(stability_lens, n)}
        for name in saved_attrs:
            delattr(stability_lens, name)
        return saved_mods, saved_attrs

    @staticmethod
    def _restore(saved_mods, saved_attrs):
        import sys

        import stability_lens

        sys.modules.update(saved_mods)
        for name, value in saved_attrs.items():
            setattr(stability_lens, name, value)

    def test_missing_numpy_gives_friendly_hint(self):
        import io
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from unittest import mock

        from stability_lens.cli import main as cli_main

        saved = self._hide_lazy_modules()
        err = io.StringIO()
        try:
            with mock.patch.dict(sys.modules, {"numpy": None}):
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    code = cli_main(["predict", "--quick"])
        finally:
            self._restore(*saved)

        self.assertEqual(code, 64)
        text = err.getvalue()
        self.assertIn("numpy", text)
        self.assertIn("pip install", text)
        self.assertNotIn("Traceback", text)

    def test_zero_dependency_paths_never_touch_numpy(self):
        """check / selftest 必须不依赖 numpy —— 这是「零依赖」声明的机械检查。"""
        import io
        import sys
        from contextlib import redirect_stdout
        from unittest import mock

        from stability_lens.cli import main as cli_main

        saved = self._hide_lazy_modules()
        try:
            with mock.patch.dict(sys.modules, {"numpy": None}):
                with redirect_stdout(io.StringIO()):
                    code_check = cli_main(["check", "--rule", "gd",
                                           "--spectrum", "1,4", "--eta", "0.6"])
                    code_self = cli_main(["selftest", "--fast"])
        finally:
            self._restore(*saved)

        self.assertEqual(code_check, 2)   # 缝隙
        self.assertEqual(code_self, 0)    # 25 项断言仍然全过


if __name__ == "__main__":
    unittest.main(verbosity=2)
