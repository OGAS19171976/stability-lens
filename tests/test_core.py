"""stability-lens 测试套件。

零第三方依赖即可运行（纯标准库 unittest）：::

    python -m unittest discover -s tests -v
    python -m pytest tests -q          # pytest 也能跑同一套

MC 相关的用例刻意用较小预算并把容差与蒙特卡洛误差挂钩。
"""

from __future__ import annotations

import io
import json
import math
import unittest
from contextlib import redirect_stdout

import stability_lens as sl
from stability_lens import core, selftest
from stability_lens.cli import EXIT_USAGE, main as cli_main
from stability_lens.diagnose import (
    EXIT_GAP,
    EXIT_OK,
    EXIT_STRUCTURAL,
    parse_complex,
    parse_spectrum,
)


class TestLinearAlgebra(unittest.TestCase):
    def test_eig_rotation(self):
        e1, e2 = core.eig_2x2([[0.0, -1.0], [1.0, 0.0]])
        self.assertAlmostEqual(e1.imag, 1.0, places=15)
        self.assertAlmostEqual(e2.imag, -1.0, places=15)

    def test_eig_real_split(self):
        e1, e2 = core.eig_2x2([[3.0, 0.0], [0.0, 1.0]])
        self.assertEqual(sorted([e1.real, e2.real]), [1.0, 3.0])

    def test_hurwitz(self):
        self.assertTrue(core.is_hurwitz([complex(-1, 3), complex(-1, -3)]))
        self.assertFalse(core.is_hurwitz([complex(0, 1), complex(0, -1)]))

    def test_lyapunov(self):
        sol = core.solve_lyapunov_2x2([[-1.0, 0.0], [0.0, -4.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertIsNotNone(sol)
        self.assertTrue(sol.positive_definite)
        self.assertLess(sol.residual, 1e-12)
        self.assertAlmostEqual(sol.P[0][0], 0.5, places=12)
        self.assertAlmostEqual(sol.P[1][1], 0.125, places=12)

    def test_lyapunov_non_hurwitz_has_no_pd_certificate(self):
        sol = core.solve_lyapunov_2x2([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertIsNotNone(sol)
        self.assertFalse(sol.positive_definite)


class TestEulerDomain(unittest.TestCase):
    lams = [complex(-1.0, 0.0), complex(-4.0, 0.0)]

    def test_eta_max_analytic_matches_bisection(self):
        th = core.euler_eta_max(self.lams).eta_max
        self.assertAlmostEqual(th, 0.5, places=15)
        num = core.bisect_eta_max(lambda e: core.euler_rho(self.lams, e), 1e-12, 0.5, 60)
        self.assertLess(abs(num - th) / th, 1e-12)

    def test_rho_values(self):
        for eta, want in ((0.025, 0.975), (0.075, 0.925), (0.125, 0.875), (0.2, 0.8)):
            self.assertAlmostEqual(core.euler_rho(self.lams, eta), want, places=14)

    def test_boundary_straddle(self):
        lo = core.norm(core.simulate_diagonal([1.0, 4.0], 0.99 * 0.5, 2000, [1.0, 1.0])[-1])
        hi = core.norm(core.simulate_diagonal([1.0, 4.0], 1.01 * 0.5, 2000, [1.0, 1.0])[-1])
        self.assertLess(lo, 1e-8)
        self.assertGreater(hi, 1e10)

    def test_complex_spectrum_bound(self):
        # |λ|² = 1 + 9 = 10 ⟹ η_max = 2 * 1 / 10 = 0.2
        self.assertAlmostEqual(core.euler_eta_max([complex(-1.0, 3.0)]).eta_max, 0.2, places=15)

    def test_non_hurwitz_reports_no_bound(self):
        em = core.euler_eta_max([complex(0.5, 0.0)])
        self.assertFalse(em.hurwitz)
        self.assertEqual(em.eta_max, 0.0)

    def test_effective_rate_second_order_correction(self):
        lam = complex(-1.0, 0.0)
        eta = 0.01
        rate = core.euler_effective_rate(lam, eta)
        # -log(1-η)/η = 1 + η/2 + η²/3 + O(η³)
        self.assertAlmostEqual(rate, 1.0 + eta / 2.0 + eta ** 2 / 3.0, places=6)
        self.assertGreater(rate, 1.0)


class TestHeavyBall(unittest.TestCase):
    lam_max = 2.0

    def test_eta_max_scales_linearly_with_beta(self):
        for beta, want in ((0.0, 1.0), (0.2, 1.2), (0.5, 1.5), (0.8, 1.8), (0.9, 1.9)):
            th = core.heavy_ball_eta_max(beta, self.lam_max)
            self.assertAlmostEqual(th, want, places=15)
            num = core.bisect_eta_max(
                lambda e, b=beta: core.heavy_ball_rho(e, b, self.lam_max), 1e-12, 4.0, 70)
            self.assertLess(abs(num - th) / th, 1e-12)

    def test_complex_region_rho_is_sqrt_beta(self):
        eta, beta = 0.7, 0.5
        self.assertLess(core.heavy_ball_disc(eta, beta, self.lam_max), 0.0)
        self.assertAlmostEqual(core.heavy_ball_rho(eta, beta, self.lam_max), math.sqrt(beta), places=14)

    def test_measured_growth_matches_rho(self):
        eta, beta = 0.7, 0.5
        traj = core.simulate_heavy_ball(eta, beta, self.lam_max, 1800, 1.0)
        meas = math.exp((math.log(abs(traj[1800])) - math.log(abs(traj[600]))) / 1200.0)
        self.assertLess(abs(meas - core.heavy_ball_rho(eta, beta, self.lam_max)) /
                        core.heavy_ball_rho(eta, beta, self.lam_max), 1e-3)

    def test_gap_continuous_stable_but_discrete_blows_up(self):
        beta = 0.5
        eta = 1.5 * core.heavy_ball_eta_max(beta, self.lam_max)
        self.assertAlmostEqual(core.continuous_damping(eta, beta), 1.0 / 3.0, places=6)
        self.assertGreater(core.heavy_ball_rho(eta, beta, self.lam_max), 1.0)
        traj = core.simulate_heavy_ball(eta, beta, self.lam_max, 200, 1.0)
        blow = next((i for i, v in enumerate(traj) if abs(v) > core.BLOWUP), -1)
        self.assertEqual(blow, 27)  # 与 JS 内核、与 README §6 记录一致


class TestNoiseFloor(unittest.TestCase):
    def test_closed_form_matches_definition(self):
        eta, s2, sig2 = 0.05, 1.0, 1.0
        v = core.steady_variance(eta, s2, sig2)
        exact = eta * eta * s2 * sig2 / (1.0 - (1.0 - eta * s2) ** 2)
        self.assertAlmostEqual(v, exact, places=15)
        self.assertAlmostEqual(v, eta * sig2 / (2.0 - eta * s2), places=15)

    def test_small_eta_asymptote(self):
        eta, s2, sig2 = 0.002, 1.0, 1.0
        ratio = core.steady_variance(eta, s2, sig2) / (eta * sig2 / 2.0)
        self.assertLess(abs(ratio - 1.0), 2e-3)

    def test_monte_carlo_matches_theory(self):
        eta, s2, sig2 = 0.05, 1.0, 1.0
        ser = core.simulate_lms(lambda k: eta, s2, sig2, 1500, 400, seed=20240607)
        w = core.window_stats(ser, 250, 400)
        th = core.steady_variance(eta, s2, sig2)
        # 容差与蒙特卡洛误差挂钩：SE ≈ sqrt(2/paths) / sqrt(窗口内有效独立样本数)
        tol = 5.0 * math.sqrt(2.0 / 1500) / math.sqrt(150 * 2 * eta * s2) + 0.005
        self.assertLess(abs(w.mean - th) / th, max(0.03, tol))

    def test_decay_schedule_beats_constant_floor(self):
        floor = core.steady_variance(0.2, 1.0, 1.0)
        decay = core.simulate_lms(lambda k: 2.0 / (k + 10.0), 1.0, 1.0, 400, 2000, seed=909)
        self.assertLess(decay[2000], 0.01 * floor)
        self.assertLess(abs(decay[2000] * 2000 - 1.0), 0.35)

    def test_stationary_init_removes_burnin_bias(self):
        """小 η + 短窗口：从 e_0 = 1 出发会把瞬态当稳态（高估），从平稳分布出发不会。

        这是本项目实际踩到的坑：初值记忆以 (1-ηs²)^k 衰减，η=0.01 时需要 ~10/(ηs²) 步。
        """
        eta, s2, sig2 = 0.01, 1.0, 1.0
        th = core.steady_variance(eta, s2, sig2)
        biased = core.window_stats(
            core.simulate_lms(lambda k: eta, s2, sig2, 2000, 200, seed=1), 100, 200).mean
        fair = core.window_stats(
            core.simulate_lms(lambda k: eta, s2, sig2, 2000, 200, seed=1, x0=None), 100, 200).mean
        self.assertGreater(biased, 1.5 * th)
        self.assertLess(abs(fair - th) / th, 0.10)

    def test_simulate_lms_is_reproducible(self):
        a = core.simulate_lms(lambda k: 0.05, 1.0, 1.0, 200, 100, seed=7)
        b = core.simulate_lms(lambda k: 0.05, 1.0, 1.0, 200, 100, seed=7)
        self.assertEqual(a, b)


class TestParsing(unittest.TestCase):
    def test_parse_complex_variants(self):
        cases = {
            "1": 1 + 0j, "-2.5": -2.5 + 0j, "3i": 3j, "-i": -1j, "i": 1j,
            "-1+3i": complex(-1, 3), "0.5-2j": complex(0.5, -2), "1+3i": complex(1, 3),
        }
        for text, want in cases.items():
            self.assertEqual(parse_complex(text), want, msg=text)

    def test_parse_spectrum(self):
        self.assertEqual(parse_spectrum("1,4"), [1 + 0j, 4 + 0j])
        self.assertEqual(parse_spectrum("-1+3i,-2"), [complex(-1, 3), -2 + 0j])

    def test_parse_spectrum_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_spectrum("  ")


class TestDiagnose(unittest.TestCase):
    def test_stable_case(self):
        d = sl.diagnose_euler("-1,-4", eta=0.45)
        self.assertEqual(d.exit_code, EXIT_OK)
        self.assertTrue(d.stable)
        self.assertAlmostEqual(d.eta_max, 0.5, places=12)

    def test_spectrum_is_jacobian_so_positive_is_structural(self):
        d = sl.diagnose_euler("1,4", eta=0.05)
        self.assertEqual(d.exit_code, EXIT_STRUCTURAL)
        self.assertFalse(d.hurwitz)
        # 给二次目标曲率的用户一个明确的路标
        self.assertTrue(any("--rule gd" in n for n in d.notes))

    def test_gap_case(self):
        d = sl.diagnose_heavy_ball(eta=2.25, beta=0.5, lambda_max=2.0)
        self.assertEqual(d.exit_code, EXIT_GAP)
        self.assertFalse(d.stable)
        self.assertAlmostEqual(d.eta_max, 1.5, places=12)
        self.assertTrue(any("构造性反例" in n for n in d.notes))

    def test_heavy_ball_stable_case(self):
        d = sl.diagnose_heavy_ball(eta=0.6, beta=0.5, lambda_max=2.0)
        self.assertEqual(d.exit_code, EXIT_OK)
        self.assertTrue(d.stable)

    def test_gd_rule_uses_curvature(self):
        d = sl.diagnose_gd("1,4", eta=0.45)
        self.assertEqual(d.exit_code, EXIT_OK)
        self.assertAlmostEqual(d.eta_max, 0.5, places=12)
        d2 = sl.diagnose_gd("1,4", eta=0.6)
        self.assertEqual(d2.exit_code, EXIT_GAP)

    def test_lms_reports_noise_floor(self):
        d = sl.diagnose_lms(eta=0.05)
        self.assertEqual(d.exit_code, EXIT_OK)
        self.assertIn("O(√η)", d.verdict)
        self.assertTrue(any("噪声地板" in n for n in d.notes))

    def test_lms_beyond_stability(self):
        d = sl.diagnose_lms(eta=3.0)
        self.assertEqual(d.exit_code, EXIT_STRUCTURAL)

    def test_diagnose_dispatcher(self):
        self.assertEqual(sl.diagnose("euler", spectrum="1,4", eta=0.45).rule, "euler")
        with self.assertRaises(ValueError):
            sl.diagnose("nope")

    def test_json_serializable(self):
        for d in (sl.diagnose_euler("1,4", eta=0.6),
                  sl.diagnose_heavy_ball(eta=2.25, beta=0.5, lambda_max=2.0),
                  sl.diagnose_lms(eta=0.05),
                  sl.diagnose_gd("1,4")):
            payload = sl.to_json(d)
            json.dumps(payload, allow_nan=False)  # 不允许 NaN/Infinity 漏出去
            self.assertIn("verdict", payload)


class TestSweep(unittest.TestCase):
    def test_sweep_euler_finds_boundary(self):
        res = sl.sweep_euler("-1,-4", 0.01, 1.0, n=11)
        self.assertAlmostEqual(res["eta_max"], 0.5, places=12)
        self.assertLess(abs(res["eta_max_numeric"] - 0.5) / 0.5, 1e-10)
        self.assertEqual(len(res["points"]), 11)

    def test_sweep_heavy_ball_finds_boundary(self):
        res = sl.sweep_heavy_ball(0.5, 2.0, 0.01, 3.0, n=9)
        self.assertAlmostEqual(res["eta_max"], 1.5, places=12)
        self.assertLess(abs(res["eta_max_numeric"] - 1.5) / 1.5, 1e-10)


class TestSelftest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checks = selftest.run_checks()

    def test_all_checks_pass(self):
        failed = [c.name for c in self.checks if not c.ok]
        self.assertEqual(failed, [], f"失败的断言: {failed}")

    def test_check_count(self):
        self.assertEqual(len(self.checks), 25)

    def test_render_contains_summary(self):
        text = selftest.render(self.checks)
        self.assertIn("25 passed, 0 failed", text)

    def test_json_shape(self):
        payload = selftest.as_json(self.checks)
        self.assertEqual(payload["total"], 25)
        self.assertEqual(payload["failed"], 0)


class TestNormalizeDashValues(unittest.TestCase):
    """改写规则必须**既能修 bug、又不误伤**。"""

    def test_rewrites_dash_leading_values(self):
        from stability_lens.cli import normalize_dash_values as norm
        self.assertEqual(norm(["--spectrum", "-1,-4"]), ["--spectrum=-1,-4"])
        self.assertEqual(norm(["--spectrum", "-1+3i,-2"]), ["--spectrum=-1+3i,-2"])
        self.assertEqual(norm(["check", "--spectrum", "-2"]), ["check", "--spectrum=-2"])

    def test_leaves_normal_values_alone(self):
        from stability_lens.cli import normalize_dash_values as norm
        self.assertEqual(norm(["--spectrum", "1,4"]), ["--spectrum", "1,4"])
        self.assertEqual(norm(["--spectrum=-1,-4"]), ["--spectrum=-1,-4"])
        # 别的选项不动 —— 它们的取值由 argparse 自己处理
        self.assertEqual(norm(["--eta", "-0.5"]), ["--eta", "-0.5"])
        self.assertEqual(norm([]), [])

    def test_does_not_swallow_a_missing_value(self):
        """用户漏写值时不能被"顺手修好"成别的意思。"""
        from stability_lens.cli import normalize_dash_values as norm
        self.assertEqual(norm(["--spectrum", "--eta", "0.5"]),
                         ["--spectrum", "--eta", "0.5"])
        self.assertEqual(norm(["--spectrum"]), ["--spectrum"])


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_check_gap_exit_code_and_json(self):
        code, out = self.run_cli(
            ["check", "--rule", "heavy-ball", "--beta", "0.5", "--lambda-max", "2", "--eta", "2.25", "--json"])
        self.assertEqual(code, EXIT_GAP)
        payload = json.loads(out)
        self.assertEqual(payload["level"], "gap")
        self.assertAlmostEqual(payload["eta_max"], 1.5, places=12)

    def test_check_stable_prints_eta_max(self):
        code, out = self.run_cli(["check", "--rule", "euler", "--spectrum", "-1,-4"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("0.5", out)

    def test_check_requires_spectrum(self):
        code, _ = self.run_cli(["check", "--rule", "euler"])
        self.assertEqual(code, EXIT_USAGE)

    def test_spectrum_with_leading_dash_is_accepted(self):
        """`--spectrum "-1,-4"` 必须在**所有受支持的 Python 上**都能用。

        这是真实的可用性 bug，不是测试写法问题：argparse 只把「纯负数」当值，
        而 `-1,-4` 带逗号，在 3.10~3.13 上会被当成选项、报
        "expected one argument"。Python 3.14 删掉了那个判定，所以在
        3.14 上本地全绿、只有 CI 的 3.10/3.12 会红。
        """
        code, out = self.run_cli(["check", "--rule", "euler", "--spectrum", "-1,-4"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("0.5", out)

    def test_spectrum_complex_with_leading_dash_is_accepted(self):
        """带虚部的写法（`-1+3i`）同样不匹配「纯负数」，一样要能过。"""
        code, _ = self.run_cli(["check", "--rule", "euler", "--spectrum", "-1+3i,-2"])
        self.assertEqual(code, EXIT_OK)

    def test_missing_spectrum_value_still_errors(self):
        """漏写取值时必须仍然报错，而不是被"顺手修好"成别的意思。

        `--spectrum` 后面跟的是另一个选项时，改写逻辑不会动它，
        argparse 于是照常报 usage 错误 —— 那是 `exit(2)`（抛 SystemExit），
        不是返回一个码。`--eta` 绝不能被当成谱值吞掉：那会把一个用法错误
        变成一个静默的错误结果，比直接报错糟得多。
        """
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["check", "--rule", "euler", "--spectrum", "--eta", "0.5"])
        self.assertEqual(ctx.exception.code, 2)

    def test_check_structural_exit_code(self):
        code, _ = self.run_cli(["check", "--rule", "euler", "--spectrum", "1,4", "--eta", "0.05"])
        self.assertEqual(code, EXIT_STRUCTURAL)

    def test_sweep_runs(self):
        code, out = self.run_cli(["sweep", "--rule", "heavy-ball", "--beta", "0.5", "--lambda-max", "2", "--n", "5"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("η_max", out)

    def test_selftest_fast(self):
        code, out = self.run_cli(["selftest", "--fast"])
        self.assertEqual(code, EXIT_OK, out[-800:])
        self.assertIn("passed", out)

    def test_module_entry_point_importable(self):
        import stability_lens.__main__ as m  # noqa: F401
        self.assertTrue(hasattr(m, "main"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
