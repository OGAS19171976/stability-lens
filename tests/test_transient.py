"""``stability_lens.transient`` 的测试。

核心命题：**``ρ(M) < 1`` 不保证 ‖M^k‖ 单调下降。**
所有断言都对着可手算的例子（Jordan 块、对角矩阵），不依赖近似。
"""

from __future__ import annotations

import io
import json
import math
import unittest
from contextlib import redirect_stdout

import numpy as np

from stability_lens import transient as T
from stability_lens import core as _core
from stability_lens.cli import main as cli_main

T.core = _core          # 便于在测试里直接引用 2x2 那套


class TestParsing(unittest.TestCase):
    def test_parse_matrix(self):
        M = T.parse_matrix("0.9,10;0,0.9")
        self.assertTrue(np.allclose(M, [[0.9, 10.0], [0.0, 0.9]]))

    def test_parse_matrix_space_and_pipe_separators(self):
        self.assertTrue(np.allclose(T.parse_matrix("1 2|3 4"), [[1, 2], [3, 4]]))

    def test_parse_matrix_rejects_non_square(self):
        with self.assertRaises(ValueError):
            T.parse_matrix("1,2,3;4,5,6")

    def test_iteration_matrix(self):
        A = np.array([[3.0, 12.0], [0.0, 3.0]])
        self.assertTrue(np.allclose(T.iteration_matrix(A, 0.1),
                                    [[0.7, -1.2], [0.0, 0.7]]))


class TestTransientCurve(unittest.TestCase):
    def test_curve_matches_explicit_matrix_powers(self):
        M = T.nonnormal_example(0.9, 10.0)
        curve = T.transient_curve(M, steps=12)
        P = np.eye(2)
        for k in range(13):
            if k:
                P = P @ M
            self.assertAlmostEqual(curve[k], float(np.linalg.norm(P, 2)), places=10)

    def test_jordan_block_amplifies_despite_rho_below_one(self):
        """经典反例：ρ = 0.9 < 1，但 ‖M^k‖ 先冲到 38 倍再衰减。"""
        res = T.transient_growth(T.nonnormal_example(0.9, 10.0), steps=60)
        self.assertAlmostEqual(res.rho, 0.9, places=12)
        self.assertGreater(res.peak, 30.0)
        self.assertGreater(res.peak_step, 0)
        self.assertLess(res.final, 2.0)                  # 最终还是衰减的
        self.assertTrue(res.contradicts_spectral_picture)

    def test_normal_matrix_decays_monotonically(self):
        res = T.transient_growth(np.diag([0.9, 0.8]), steps=40)
        self.assertAlmostEqual(res.peak, 1.0, places=12)
        self.assertEqual(res.peak_step, 0)
        self.assertFalse(res.contradicts_spectral_picture)
        curve = res.curve
        self.assertTrue(all(curve[i + 1] <= curve[i] + 1e-12 for i in range(len(curve) - 1)))

    def test_discretized_nonnormal_iteration_also_shows_it(self):
        A = np.array([[3.0, 12.0], [0.0, 3.0]])
        res = T.transient_growth(T.iteration_matrix(A, 0.1), steps=60)
        self.assertLess(res.rho, 1.0)
        self.assertGreater(res.peak, 1.5)


class TestKreissConstant(unittest.TestCase):
    def test_normal_matrix_has_kreiss_near_one(self):
        self.assertLess(abs(T.kreiss_constant(np.diag([0.9, 0.8])) - 1.0), 0.01)

    def test_nonnormal_matrix_has_large_kreiss(self):
        self.assertGreater(T.kreiss_constant(T.nonnormal_example(0.9, 10.0)), 20.0)

    def test_kreiss_sandwiches_the_peak(self):
        """Kreiss 矩阵定理：K(M) ≤ sup_k‖M^k‖ ≤ e·n·K(M)。"""
        M = T.nonnormal_example(0.9, 10.0)
        K = T.kreiss_constant(M)
        peak = T.transient_growth(M, steps=60).peak
        self.assertLessEqual(K, peak * (1 + 1e-9))
        self.assertLessEqual(peak, math.e * M.shape[0] * K * (1 + 1e-9))


class TestPseudospectrum(unittest.TestCase):
    def test_sigma_min_equals_distance_for_normal_matrix(self):
        """正规矩阵：σ_min(zI − M) 就是 z 到谱的距离。"""
        M = np.diag([0.9, 0.8])
        re, im, g = T.pseudospectrum_sigma_min(M, re_range=(0.9, 1.1), im_range=(-0.1, 0.1), n=21)
        val = g[np.argmin(abs(re - 1.0)), np.argmin(abs(im - 0.0))]
        self.assertAlmostEqual(float(val), 0.1, places=6)

    def test_pseudospectrum_bulges_for_nonnormal_matrix(self):
        """同样距离 0.1：非正规矩阵的 σ_min 小了约两个数量级 —— 伪谱鼓出来了。"""
        J = T.nonnormal_example(0.9, 10.0)
        N = np.diag([0.9, 0.8])
        re, im, gj = T.pseudospectrum_sigma_min(J, re_range=(0.9, 1.1), im_range=(-0.1, 0.1), n=21)
        rn, imn, gn = T.pseudospectrum_sigma_min(N, re_range=(0.9, 1.1), im_range=(-0.1, 0.1), n=21)
        vj = float(gj[np.argmin(abs(re - 1.0)), np.argmin(abs(im))])
        vn = float(gn[np.argmin(abs(rn - 1.0)), np.argmin(abs(imn))])
        self.assertLess(vj, vn / 50.0)
        self.assertAlmostEqual(vj, 1e-3, delta=2e-4)


class TestDiagnosis(unittest.TestCase):
    def test_transient_level(self):
        d = T.diagnose_nonnormal(T.nonnormal_example(0.9, 10.0))
        self.assertEqual(d.level, "transient")
        self.assertEqual(d.exit_code, 2)
        self.assertIn("先放大", d.verdict)

    def test_ok_level_for_normal_matrix(self):
        d = T.diagnose_nonnormal(np.diag([0.9, 0.8]))
        self.assertEqual(d.level, "ok")
        self.assertEqual(d.exit_code, 0)

    def test_unstable_level_when_rho_above_one(self):
        d = T.diagnose_nonnormal(np.diag([1.2, 0.9]))
        self.assertEqual(d.level, "unstable")
        self.assertEqual(d.exit_code, 3)

    def test_threshold_controls_the_level(self):
        M = T.iteration_matrix(np.array([[3.0, 12.0], [0.0, 3.0]]), 0.1)  # peak ≈ 1.83
        self.assertEqual(T.diagnose_nonnormal(M, threshold=10.0).level, "ok")
        self.assertEqual(T.diagnose_nonnormal(M, threshold=1.5).level, "transient")

    def test_render_and_json(self):
        d = T.diagnose_nonnormal(T.nonnormal_example())
        text = T.render(d)
        self.assertIn("瞬态", text)
        self.assertIn("Kreiss", text)
        payload = json.loads(json.dumps(T.as_json(d), ensure_ascii=False))
        self.assertEqual(payload["level"], "transient")
        self.assertAlmostEqual(payload["rho"], 0.9, places=12)


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_default_jordan_alarms(self):
        code, out = self.run_cli(["transient"])
        self.assertEqual(code, 2)
        self.assertIn("瞬态", out)

    def test_normal_example_is_clean(self):
        code, out = self.run_cli(["transient", "--example", "normal", "--no-notes"])
        self.assertEqual(code, 0)
        self.assertIn("Kreiss", out)

    def test_matrix_argument(self):
        code, _ = self.run_cli(["transient", "--matrix", "0.5,0;0,0.5"])
        self.assertEqual(code, 0)

    def test_rho_above_one_is_structural(self):
        code, _ = self.run_cli(["transient", "--matrix", "1.5,0;0,0.5"])
        self.assertEqual(code, 3)

    def test_bad_matrix_is_usage_error(self):
        code, _ = self.run_cli(["transient", "--matrix", "1,2,3;4,5,6"])
        self.assertEqual(code, 64)

    def test_json_output(self):
        code, out = self.run_cli(["transient", "--json"])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("kreiss", payload)
        self.assertIn("peak", payload)


class TestHeavyBallTransient(unittest.TestCase):
    """动量是这套框架里瞬态放大的唯一来源；且必须区分"状态范数"与"误差本身"。"""

    def test_gd_has_no_position_amplification(self):
        """β=0：位置放大恒为 1（GD 的误差单调下降）。

        但 ``max_k‖M^k‖`` 会报到 1.28 —— 那只是把二阶递推写成一阶状态带来的**表示假象**，
        所以两个量必须分开报，否则会给纯 GD 用户一个不存在的警告。
        """
        tr = T.core.heavy_ball_transient(0.9, 0.0, 2.0)
        self.assertAlmostEqual(tr["position_peak"], 1.0, places=12)
        self.assertEqual(tr["position_peak_step"], 0.0)
        self.assertGreater(tr["state_peak"], 1.2)

    def test_momentum_really_amplifies_the_error(self):
        tr = T.core.heavy_ball_transient(0.99 * 1.9, 0.9, 2.0)
        self.assertLess(tr["rho"], 1.0)
        self.assertGreater(tr["position_peak"], 4.0)

    def test_amplification_grows_with_eta(self):
        peaks = [T.core.heavy_ball_transient(f * 1.9, 0.9, 2.0)["position_peak"]
                 for f in (0.5, 0.9, 0.99)]
        self.assertEqual(peaks, sorted(peaks))
        self.assertAlmostEqual(peaks[0], 1.0, places=12)

    def test_amplification_grows_with_beta(self):
        peaks = [T.core.heavy_ball_transient(0.99 * (1 + b) * 2.0 / 2.0, b, 2.0)["position_peak"]
                 for b in (0.0, 0.5, 0.9)]
        self.assertLess(peaks[0], peaks[1])
        self.assertLess(peaks[1], peaks[2])

    def test_stdlib_2x2_matches_numpy_implementation(self):
        """纯标准库的 2x2 实现 vs numpy 的任意 n 实现 —— 两份独立代码互相验证。"""
        for beta in (0.0, 0.5, 0.9):
            for frac in (0.5, 0.9, 0.99):
                eta = frac * T.core.heavy_ball_eta_max(beta, 2.0)
                std = T.core.heavy_ball_transient(eta, beta, 2.0)
                Mnp = np.asarray(T.core.heavy_ball_matrix(eta, beta, 2.0), dtype=float)
                pos, _ = T.position_amplification(Mnp, steps=120)
                state = T.transient_growth(Mnp, steps=120).peak
                self.assertAlmostEqual(std["position_peak"], pos, places=10)
                self.assertAlmostEqual(std["state_peak"], state, places=10)
                self.assertAlmostEqual(std["kreiss"], T.kreiss_constant(Mnp), places=8)
                self.assertAlmostEqual(std["rho"], T.transient_growth(Mnp, steps=1).rho,
                                       places=12)

    def test_diagnose_heavy_ball_reports_the_transient(self):
        from stability_lens.diagnose import diagnose_heavy_ball

        d = diagnose_heavy_ball(eta=0.99 * 1.9, beta=0.9, lambda_max=2.0)
        labels = [k for k, _ in d.rows]
        self.assertTrue(any("误差瞬态峰值" in x for x in labels))
        self.assertTrue(any("整状态峰值" in x for x in labels))
        self.assertTrue(any("先放大" in n for n in d.notes))
        self.assertEqual(d.extra["transient"]["position_peak"],
                         T.core.heavy_ball_transient(0.99 * 1.9, 0.9, 2.0)["position_peak"])

    def test_diagnose_heavy_ball_gd_gets_no_transient_warning(self):
        from stability_lens.diagnose import diagnose_heavy_ball

        d = diagnose_heavy_ball(eta=0.9, beta=0.0, lambda_max=2.0)
        self.assertFalse(any("先放大" in n for n in d.notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
