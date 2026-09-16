"""``stability_lens.torch_backend`` 的测试。

torch 是可选依赖：没有它时整类测试 **skip**，但"缺依赖要给一句人话"这条仍然要测。
真正的数值验证（与 ``torch.autograd.functional.hessian`` 对拍）会在装了 torch
的环境里自动跑起来。
"""

from __future__ import annotations

import unittest

from stability_lens import torch_backend as tb

HAS_TORCH = tb.has_torch()


class TestWithoutTorch(unittest.TestCase):
    @unittest.skipIf(HAS_TORCH, "本机装了 torch，跳过缺依赖路径")
    def test_missing_torch_gives_a_human_message(self):
        with self.assertRaises(ImportError) as ctx:
            tb.lambda_max_torch(lambda: 0.0, [])
        self.assertIn("pip install torch", str(ctx.exception))

    def test_module_imports_without_torch(self):
        """模块本身必须能在没有 torch 的环境里 import（不能把 torch 写死在顶层）。"""
        self.assertIsInstance(HAS_TORCH, bool)

    def test_install_hint_mentions_optional(self):
        self.assertIn("可选依赖", tb.INSTALL_HINT)


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
class TestTorchBackend(unittest.TestCase):
    """装了 torch 才跑：数值上与 torch 自带的精确 Hessian 对拍。"""

    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch
        n, d, h = 40, 4, 3
        self.X = torch.randn(n, d, dtype=torch.float64)
        self.y = (self.X @ torch.randn(d, dtype=torch.float64)).unsqueeze(1)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d, h, dtype=torch.float64), torch.nn.Tanh(),
            torch.nn.Linear(h, 1, dtype=torch.float64),
        )
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.loss_fn = lambda: (  # noqa: E731
            (self.model(self.X) - self.y).pow(2).mean() / 2.0
        )

    def exact_lambda_max(self):
        """用 torch 自带的精确 Hessian 作为真值。

        关键是用 ``torch.func.functional_call`` 把参数**函数式**地替换掉：
        直接赋值 ``p.data`` 会切断 ``flat → loss`` 的计算图，
        那样算出来的 Hessian 全是零（这个坑我踩过一次）。
        """
        torch = self.torch
        flat0 = torch.cat([p.detach().reshape(-1) for p in self.params])

        def loss_from_flat(flat):
            offset, pd = 0, {}
            for name, p in self.model.named_parameters():
                pd[name] = flat[offset:offset + p.numel()].reshape(p.shape)
                offset += p.numel()
            out = torch.func.functional_call(self.model, pd, (self.X,))
            return (out - self.y).pow(2).mean() / 2.0

        H = torch.autograd.functional.hessian(loss_from_flat, flat0)
        return float(torch.linalg.eigvalsh(0.5 * (H + H.T))[-1])

    def test_lambda_max_matches_exact_hessian(self):
        want = self.exact_lambda_max()
        got = tb.lambda_max_torch(self.loss_fn, self.params, iters=300, tol=1e-10)
        self.assertLess(abs(got.lam - want) / want, 1e-6, f"{got.lam} vs {want}")

    def test_hvp_matches_finite_difference_of_gradient(self):
        torch = self.torch
        hvp = tb.make_hvp(self.loss_fn, self.params)
        flat0 = torch.cat([p.detach().reshape(-1) for p in self.params])
        dim = flat0.numel()
        gen = torch.Generator().manual_seed(1)
        v = torch.randn(dim, dtype=flat0.dtype, generator=gen)

        def grad_at(flat):
            offset = 0
            for p in self.params:
                p.data = flat[offset:offset + p.numel()].reshape(p.shape)
                offset += p.numel()
            loss = self.loss_fn()
            gs = torch.autograd.grad(loss, self.params)
            return torch.cat([g.reshape(-1) for g in gs]).detach()

        eps = 1e-6
        fd = (grad_at(flat0 + eps * v) - grad_at(flat0 - eps * v)) / (2 * eps)
        # 恢复参数（grad_at 改过 data）
        offset = 0
        for p in self.params:
            p.data = flat0[offset:offset + p.numel()].reshape(p.shape)
            offset += p.numel()
        got = hvp(v)
        self.assertLess(float((got - fd).norm() / fd.norm()), 1e-5)

    def test_eta_max_and_diagnosis(self):
        out = tb.eta_max_torch(self.loss_fn, self.params, beta=0.5)
        self.assertGreater(out["lambda_max"], 0.0)
        self.assertAlmostEqual(out["eta_max"], 2.0 * 1.5 / out["lambda_max"], places=10)

        d = tb.diagnose_torch(self.loss_fn, self.params, eta=out["eta_max"] * 0.5, beta=0.5)
        self.assertEqual(d.exit_code, 0)          # 半步长 → 稳定
        self.assertIn("PyTorch", d.title)
        self.assertTrue(any("误差瞬态峰值" in k for k, _ in d.rows))

    def test_unstable_eta_is_flagged(self):
        out = tb.eta_max_torch(self.loss_fn, self.params)
        d = tb.diagnose_torch(self.loss_fn, self.params, eta=out["eta_max"] * 1.5)
        self.assertEqual(d.exit_code, 2)          # 缝隙：越界


if __name__ == "__main__":
    unittest.main(verbosity=2)
