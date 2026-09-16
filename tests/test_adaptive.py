"""``stability_lens.adaptive`` 的测试。

关键是把 ``v̂`` 的**约定**钉住（单样本损失梯度，不除以 n），
以及把预条件后的谱与实测边界对上。
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

import numpy as np

from stability_lens import adaptive, datasets, models, spectral, training
from stability_lens.cli import main as cli_main


def _small_problem(n=60, d=5, h=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.2 * rng.normal(size=n)
    X, y, _ = datasets.standardize(X, y)
    theta = models.init_params(d, h, seed=seed + 1)
    return theta, X, y, d, h


class TestGradSqMoments(unittest.TestCase):
    def test_matches_brute_force_per_sample_gradients(self):
        """与"逐样本算梯度再平方平均"的暴力实现对拍 —— 钉住 v̂ 的约定。"""
        theta, X, y, d, h = _small_problem()
        W1, b1, W2, b2 = models.unpack(theta, d, h)
        n = X.shape[0]
        acc = np.zeros_like(theta)
        for i in range(n):
            xi, yi = X[i], y[i]
            a = np.tanh(W1 @ xi + b1)
            r = float(W2.ravel() @ a) + float(b2[0]) - yi
            gW1 = r * np.outer((1 - a ** 2) * W2.ravel(), xi)
            gb1 = r * (1 - a ** 2) * W2.ravel()
            gW2 = r * a
            gb2 = np.array([r])
            acc += models.pack(gW1, gb1, gW2.reshape(1, h), gb2) ** 2
        brute = acc / n
        fast = adaptive.mlp_grad_sq_moments(theta, X, y, d, h)
        self.assertTrue(np.allclose(fast, brute, rtol=1e-10, atol=1e-14))

    def test_moments_dominate_the_full_batch_gradient_squared(self):
        """``v̂ = E_i[g_i²] ≥ (E_i[g_i])² = (∇L)²``（Jensen），且通常严格更大。

        这条把"单样本二阶矩"与"全批梯度平方"两个容易混淆的量区分开：
        若 ``v̂`` 误写成后者，下面的严格不等式会失效。
        """
        theta, X, y, d, h = _small_problem()
        v = adaptive.mlp_grad_sq_moments(theta, X, y, d, h)
        _, grad = models.mlp_loss_grad(theta, X, y, d, h)
        self.assertTrue(np.all(v >= grad ** 2 - 1e-12))
        self.assertGreater(float(np.max(v - grad ** 2)), 1e-6)

    def test_l2_term_is_added(self):
        theta, X, y, d, h = _small_problem()
        v0 = adaptive.mlp_grad_sq_moments(theta, X, y, d, h, l2=0.0)
        v1 = adaptive.mlp_grad_sq_moments(theta, X, y, d, h, l2=0.1)
        self.assertTrue(np.allclose(v1 - v0, (0.1 * theta) ** 2, atol=1e-14))


class TestPreconditioner(unittest.TestCase):
    def test_formula_and_eps_cap(self):
        v = np.array([0.0, 1.0, 4.0])
        p = adaptive.diagonal_preconditioner(v, eps=1e-2)
        self.assertAlmostEqual(p[0], 1.0 / 1e-2)      # 上限就是 1/ε
        self.assertAlmostEqual(p[1], 1.0 / 1.01)
        self.assertAlmostEqual(p[2], 1.0 / 2.01)

    def test_rejects_nonpositive_eps(self):
        with self.assertRaises(ValueError):
            adaptive.diagonal_preconditioner(np.ones(3), eps=0.0)


class TestPreconditionedSpectrum(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        lam = np.linspace(0.5, 3.0, 7)
        self.H = (Q * lam) @ Q.T
        self.H = 0.5 * (self.H + self.H.T)
        self.p = np.exp(rng.normal(scale=0.7, size=7))     # 正的对角预条件子

    def test_matches_explicit_symmetric_product(self):
        sqrt_p = np.sqrt(self.p)
        M = (sqrt_p[:, None] * self.H) * sqrt_p[None, :]
        want = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
        res = adaptive.preconditioned_lambda_max(lambda v: self.H @ v, self.p,
                                                 iters=600, tol=1e-13)
        self.assertLess(abs(res.lam - want) / want, 1e-9)

    def test_identity_preconditioner_recovers_plain_lambda_max(self):
        want = spectral.spectral_summary(self.H)["lam_max"]
        res = adaptive.preconditioned_lambda_max(lambda v: self.H @ v, np.ones(7),
                                                 iters=600, tol=1e-13)
        self.assertLess(abs(res.lam - want) / want, 1e-9)

    def test_uniform_preconditioner_scales_the_spectrum(self):
        """p ≡ c 时 λ_max 恰好放大 c 倍。"""
        res = adaptive.preconditioned_lambda_max(lambda v: self.H @ v, np.full(7, 4.0),
                                                 iters=600, tol=1e-13)
        want = 4.0 * spectral.spectral_summary(self.H)["lam_max"]
        self.assertLess(abs(res.lam - want) / want, 1e-9)

    def test_eta_bound_arithmetic(self):
        b = adaptive.adam_eta_bound(4.0, 2.0)
        self.assertAlmostEqual(b["eta_max_adam"], 0.5)
        self.assertAlmostEqual(b["eta_max_gd"], 1.0)
        self.assertAlmostEqual(b["ratio"], 0.5)


class TestBoundaryMatchesTheory(unittest.TestCase):
    def test_preconditioned_iteration_boundary_on_quadratic(self):
        """二次目标上 ``η_max = 2/λ_max(PH)`` 是精确的 —— 端到端验证。"""
        rng = np.random.default_rng(3)
        n, dim = 200, 6
        X = rng.normal(size=(n, dim))
        y = X @ rng.normal(size=dim) + 0.1 * rng.normal(size=n)
        X, y, _ = datasets.standardize(X, y)
        H = models.linear_hessian(X)
        theta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)

        p = np.exp(rng.normal(scale=0.5, size=dim))
        sqrt_p = np.sqrt(p)
        M = (sqrt_p[:, None] * H) * sqrt_p[None, :]
        lam = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
        power = adaptive.preconditioned_lambda_max(lambda v: H @ v, p,
                                                   iters=500, tol=1e-13)
        self.assertLess(abs(power.lam - lam) / lam, 1e-9)

        eta_pred = 2.0 / lam
        precond_grad = lambda th: (models.linear_loss_grad(th, X, y)[0],
                                   p * models.linear_loss_grad(th, X, y)[1])
        meas = training.boundary_from_prediction(
            precond_grad, theta_ls, eta_pred, beta=0.0, bisect_iters=22,
            steps=150, direction=power.vec)
        self.assertTrue(meas["valid"])
        self.assertLess(abs(float(meas["eta_max"]) - eta_pred) / eta_pred, 0.01)


class TestAdaptiveStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = datasets.make_synthetic(n=250, d=5, seed=0)
        cls.study = adaptive.run_adaptive_study(dataset=ds, h=5, eps=1e-8,
                                                train_steps=60, test_steps=120,
                                                bisect_iters=10)

    def test_prediction_is_confirmed_by_measurement(self):
        self.assertTrue(self.study.measured["valid"])
        self.assertLess(self.study.measured["rel_err"], 0.05)

    def test_linear_closed_form_check_is_exact(self):
        lc = self.study.linear_check
        self.assertLess(lc["power_rel_err"], 1e-9)
        self.assertLess(lc["rel_err"], 0.01)

    def test_ratio_and_bounds_are_consistent(self):
        s = self.study
        self.assertAlmostEqual(s.ratio, s.eta_max_adam / s.eta_max_gd, places=12)
        self.assertAlmostEqual(s.eta_max_adam, 2.0 / s.lam_max_precond, places=12)
        self.assertAlmostEqual(s.eta_max_gd, 2.0 / s.lam_max_plain, places=12)

    def test_render_and_json(self):
        text = adaptive.render_adaptive_study(self.study)
        for key in ("λ_max", "η_max  Adam/RMSProp", "预测 vs 实测", "闭式对照"):
            self.assertIn(key, text)
        self.assertNotIn("nan", text)
        payload = json.loads(json.dumps(adaptive.as_json(self.study), ensure_ascii=False))
        self.assertIn("eta_max_adam", payload)
        self.assertIn("linear_check", payload)


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_quick_run_passes_the_gate(self):
        code, out = self.run_cli(["adaptive", "--quick"])
        self.assertEqual(code, 0, out[-600:])
        self.assertIn("η_max", out)

    def test_json_output(self):
        code, out = self.run_cli(["adaptive", "--quick", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertGreater(payload["eta_max_adam"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
