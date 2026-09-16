"""stability_lens.cli — 命令行入口。

::

    stability-lens check --rule heavy-ball --beta 0.5 --lambda-max 2 --eta 2.25
    stability-lens check --rule euler --spectrum "1,4" --eta 0.45
    stability-lens check --rule gd    --spectrum "1,4,16"
    stability-lens check --rule euler --spectrum "-1+3i,-2"
    stability-lens sweep --rule heavy-ball --beta 0.5 --lambda-max 2
    stability-lens selftest

退出码（可直接作为训练前的 gate）：``0`` 稳定 / 只给建议；``2`` 缝隙（连续稳定但离散不稳定）；
``3`` 结构性不稳定（ODE 本身不稳，调步长无用）；``1`` 自检失败或异常；
``64`` 参数用法错误或缺少可选依赖（如 predict 缺 numpy）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Sequence

from . import __version__
from . import report, selftest
from .diagnose import (
    EXIT_GAP,
    EXIT_OK,
    EXIT_STRUCTURAL,
    diagnose_euler,
    diagnose_gd,
    diagnose_heavy_ball,
    diagnose_lms,
    sweep_euler,
    sweep_heavy_ball,
)

EXIT_USAGE = 64

RULES = ("euler", "gd", "heavy-ball", "lms")

_NUMPY_HINT = (
    "该子命令需要 numpy，其余子命令（check / sweep / selftest）不需要。\n"
    "  pip install numpy\n"
    "  或 pip install \"stability-lens[numpy]\"\n"
    "原始错误：{exc}"
)


def force_utf8_stdio() -> None:
    """让 stdout/stderr 用 UTF-8，并尽力把 Windows 控制台也切到 UTF-8。

    Windows 默认代码页是 GBK，直接打印 ``η``/``ρ``/``√`` 这类字符会抛
    ``UnicodeEncodeError``。这里做两件事：把控制台输出代码页切到 65001，
    并把 Python 侧的编码改成 UTF-8（``errors="replace"`` 兜底，绝不因编码崩掉）。
    """
    if sys.platform == "win32":  # pragma: no cover - 平台相关
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

EPILOG = """\
例子:
  stability-lens check --rule heavy-ball --beta 0.5 --lambda-max 2 --eta 2.25   # 缝隙反例
  stability-lens check --rule euler --spectrum "1,4" --eta 0.45                 # 稳定
  stability-lens check --rule gd --spectrum "1,4,16"                            # 只求 η_max
  stability-lens predict --quick                                                # 训练稳定性诊断（快跑）
  stability-lens predict --csv data.csv --hidden 16 --betas 0,0.5,0.9          # 真实数据
  stability-lens monitor --quick                                                # 在线告警（快跑）
  stability-lens transient --matrix "0.9,10;0,0.9"                              # 瞬态放大体检
  stability-lens selftest                                                       # 数值证据表

退出码: 0 稳定 / 2 缝隙或瞬态放大 / 3 结构性不稳定 / 64 参数错误
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stability-lens",
        description="增量式算法的离散稳定性体检：步长上界 η_max、谱半径、Lyapunov 证书与噪声地板。",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"stability-lens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="输出 JSON（便于 CI 消费）")
        sp.add_argument("--color", action="store_true", help="彩色输出（ANSI）")

    c = sub.add_parser("check", help="对一个更新规则做一次稳定性体检")
    c.add_argument("--rule", choices=RULES, default="euler")
    c.add_argument("--spectrum", type=str, default=None,
                   help='特征值列表，逗号分隔。euler 解释为 λ(J)；gd 解释为曲率 λ(A)。如 "1,4" 或 "-1+3i,-2"')
    c.add_argument("--eta", type=float, default=None, help="要判定的步长；省略则只给 η_max")
    c.add_argument("--beta", type=float, default=0.9, help="heavy-ball 的动量系数")
    c.add_argument("--lambda-max", type=float, default=1.0, dest="lambda_max",
                   help="heavy-ball 的最大曲率 λ_max")
    c.add_argument("--s2", type=float, default=1.0, help="lms 的回归子二阶矩 s²")
    c.add_argument("--sigma2", type=float, default=1.0, help="lms 的噪声方差 σ²")
    c.add_argument("--no-notes", action="store_true", help="不打印提示")
    common(c)

    s = sub.add_parser("sweep", help="扫描 η，画出谱半径曲线并与解析 η_max 对照")
    s.add_argument("--rule", choices=("euler", "heavy-ball"), default="heavy-ball")
    s.add_argument("--spectrum", type=str, default=None)
    s.add_argument("--beta", type=float, default=0.5)
    s.add_argument("--lambda-max", type=float, default=1.0, dest="lambda_max")
    s.add_argument("--eta-lo", type=float, default=None, dest="eta_lo")
    s.add_argument("--eta-hi", type=float, default=None, dest="eta_hi")
    s.add_argument("--n", type=int, default=25)
    common(s)

    t = sub.add_parser("selftest", help="跑 25 项断言并打印数值证据表")
    t.add_argument("--fast", action="store_true", help="缩小蒙特卡洛预算（更快）")
    common(t)

    pr = sub.add_parser("predict", help="真实训练上的 LR 上限预测：估 λ_max → 预测 η_max → 实测对拍")
    pr.add_argument("--csv", type=str, default=None, help="数据表路径（默认用工作区的订单宽表）")
    pr.add_argument("--rows", type=int, default=1200, help="抽样行数")
    pr.add_argument("--hidden", type=int, default=12, help="隐藏层宽度")
    pr.add_argument("--betas", type=str, default="0,0.5,0.9", help="要检验的动量系数，逗号分隔")
    pr.add_argument("--l2", type=float, default=1e-3, help="权重衰减（保证 θ* 是严格局部极小）")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--quick", action="store_true",
                    help="小规模快跑（合成数据 + 少样本），用于冒烟验证")
    common(pr)

    mo = sub.add_parser("monitor", help="在线稳定性监控：训练中逐步估 λ_max(θ_k)，发散前告警")
    mo.add_argument("--csv", type=str, default=None, help="数据表；默认用工作区的订单宽表")
    mo.add_argument("--rows", type=int, default=1200, help="抽样行数")
    mo.add_argument("--hidden", type=int, default=12, help="隐藏层宽度")
    mo.add_argument("--fracs", type=str, default="0.5,0.9,1.01,1.3",
                    help="要检验的 η/η_max 取值，逗号分隔")
    mo.add_argument("--beta", type=float, default=0.0, help="动量系数")
    mo.add_argument("--steps", type=int, default=400)
    mo.add_argument("--check-every", type=int, default=20, dest="check_every",
                    help="每隔多少步做一次在线 λ_max 估计")
    mo.add_argument("--seed", type=int, default=0)
    mo.add_argument("--quick", action="store_true",
                    help="小规模快跑（合成数据 + 少样本）")
    common(mo)

    tr = sub.add_parser("transient", help="非正规性 / 瞬态增长：ρ<1 也可能先放大")
    tr.add_argument("--matrix", type=str, default=None,
                    help='迭代矩阵 M，行用 ";" 分隔，如 "0.9,10;0,0.9"')
    tr.add_argument("--example", choices=("jordan", "normal", "euler"), default=None,
                    help="不传 --matrix 时用内置例子：jordan（默认）/ normal / euler")
    tr.add_argument("--eta", type=float, default=0.1, help="--example euler 的步长")
    tr.add_argument("--steps", type=int, default=60, help="幂次上限 K")
    tr.add_argument("--threshold", type=float, default=10.0, help="放大多少倍算告警")
    tr.add_argument("--no-notes", action="store_true", help="不打印提示")
    common(tr)

    ad = sub.add_parser("adaptive", help="Adam/RMSProp 的逐坐标步长界（对角预条件后的谱）")
    ad.add_argument("--csv", type=str, default=None, help="数据表；默认用工作区的订单宽表")
    ad.add_argument("--rows", type=int, default=1200, help="抽样行数")
    ad.add_argument("--hidden", type=int, default=12, help="隐藏层宽度")
    ad.add_argument("--eps", type=float, default=1e-8, help="Adam 的分母保护项 ε")
    ad.add_argument("--seed", type=int, default=0)
    ad.add_argument("--quick", action="store_true", help="小规模快跑")
    common(ad)

    return p


def _diagnose_from_args(args: argparse.Namespace):
    if args.rule == "heavy-ball":
        return diagnose_heavy_ball(eta=args.eta, beta=args.beta, lambda_max=args.lambda_max)
    if args.rule == "lms":
        if args.eta is None:
            raise ValueError("lms 规则需要 --eta")
        return diagnose_lms(eta=args.eta, s2=args.s2, sigma2=args.sigma2)
    if args.spectrum is None:
        raise ValueError(f"{args.rule} 规则需要 --spectrum（如 \"1,4\" 或 \"-1+3i,-2\"）")
    if args.rule == "gd":
        return diagnose_gd(args.spectrum, eta=args.eta)
    return diagnose_euler(args.spectrum, eta=args.eta)


def cmd_check(args: argparse.Namespace) -> int:
    try:
        d = _diagnose_from_args(args)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.json:
        print(json.dumps(report.to_json(d), ensure_ascii=False, indent=2))
    else:
        print(report.render(d, color=args.color, show_notes=not args.no_notes))
        if d.exit_code == EXIT_OK and d.eta is None:
            print(f"\n  下一步: 把 η 控制在 {d.eta_max:.6g} 以下；`--eta <值>` 可直接判定某个具体步长。")
    return d.exit_code


def cmd_sweep(args: argparse.Namespace) -> int:
    try:
        if args.rule == "euler":
            if args.spectrum is None:
                raise ValueError('sweep --rule euler 需要 --spectrum，如 "1,4"')
            res = sweep_euler(args.spectrum, args.eta_lo or 0.01, args.eta_hi or 1.0, args.n)
        else:
            hi_default = 2.0 * (1.0 + args.beta) / args.lambda_max * 1.4
            res = sweep_heavy_ball(args.beta, args.lambda_max,
                                   args.eta_lo or 0.01, args.eta_hi or hi_default, args.n)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        payload = {
            "rule": args.rule,
            "eta_max": res["eta_max"],
            "eta_max_numeric": res["eta_max_numeric"],
            "points": [{"eta": e, "rho": r} for e, r in res["points"]],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(report._wrap_dash(f"η 扫描 · 谱半径 ρ" if args.rule == "euler"
                           else f"η 扫描 · Heavy-ball β={args.beta:g}, λ_max={args.lambda_max:g}"))
    for eta, rho in res["points"]:
        bar = "#" * min(60, int(round(rho * 20)))
        mark = "   <== 越过 1" if rho >= 1.0 else ""
        print(f"  η={eta:>12.6g}   ρ={rho:>12.8f}   {bar}{mark}")
    print("")
    print(f"  解析 η_max        = {res['eta_max']:.10g}")
    print(f"  数值 η_max（二分）= {res['eta_max_numeric']:.10g}   "
          f"相对误差 {abs(res['eta_max_numeric'] - res['eta_max']) / res['eta_max']:.1e}")
    return EXIT_OK


def cmd_selftest(args: argparse.Namespace) -> int:
    budget = None
    if args.fast:
        # 仅缩小蒙特卡洛预算；解析行不受影响。预算仍足以让 25 项断言全部通过。
        budget = {
            "variance_paths": 1000, "variance_steps": 300, "variance_window": (180, 300),
            "slope_paths": 300, "slope_steps": 250, "slope_window": (150, 250),
            "decay_paths": 250, "decay_steps": 2000,
        }
    checks = selftest.run_checks(budget)
    if args.json:
        print(json.dumps(selftest.as_json(checks), ensure_ascii=False, indent=2))
    else:
        rows = [[c.status, c.name, c.theory, c.measured, c.relerr, c.note] for c in checks]
        print("stability-lens · 数值证据表（Python 独立实现，参数与 web/test-core.js 一致）")
        print(report.render_evidence_table(
            rows, ["状态", "检验项", "理论值", "实测值", "相对误差", "备注"]))
        npass = sum(1 for c in checks if c.ok)
        print(f"{npass} passed, {len(checks) - npass} failed, {len(checks)} checks")
    return EXIT_OK if all(c.ok for c in checks) else 1


def cmd_predict(args: argparse.Namespace) -> int:
    # numpy 是 predict 的可选依赖（core / diagnose / selftest 都不需要它）。
    # 缺了它必须给出一句人话，而不是 ModuleNotFoundError 的 traceback。
    try:
        from . import datasets, experiment
    except ImportError as exc:
        print(_NUMPY_HINT.format(exc=exc), file=sys.stderr)
        return EXIT_USAGE

    betas = tuple(float(x) for x in str(args.betas).replace(";", ",").split(",") if x.strip())
    if args.quick:
        ds = datasets.make_synthetic(n=400, d=6, seed=args.seed)
        res = experiment.run_study(dataset=ds, h=6, betas=betas, seed=args.seed,
                                   l2=args.l2, train_steps=60, test_steps=120,
                                   bisect_iters=10)
    else:
        ds = datasets.load_order_amount(path=args.csv, max_rows=args.rows, seed=args.seed)
        res = experiment.run_study(dataset=ds, h=args.hidden, betas=betas, seed=args.seed,
                                   l2=args.l2)
    if args.json:
        print(json.dumps(experiment.as_json(res), ensure_ascii=False, indent=2))
    else:
        print(experiment.render_study(res))
    # 严格局部极小 + 预测与实测一致才算通过
    ok = res.is_local_min and all(r.valid and r.rel_err < 0.05 for r in res.rows)
    return EXIT_OK if ok else EXIT_GAP


def cmd_monitor(args: argparse.Namespace) -> int:
    try:
        from . import datasets, experiment
    except ImportError as exc:
        print(_NUMPY_HINT.format(exc=exc), file=sys.stderr)
        return EXIT_USAGE
    fracs = tuple(float(x) for x in str(args.fracs).replace(";", ",").split(",") if x.strip())
    if args.quick:
        ds = datasets.make_synthetic(n=300, d=5, seed=args.seed)
        study = experiment.run_monitor_study(
            dataset=ds, h=5, fracs=fracs, beta=args.beta, steps=min(args.steps, 120),
            check_every=args.check_every, seed=args.seed)
    else:
        ds = datasets.load_order_amount(path=args.csv, max_rows=args.rows, seed=args.seed)
        study = experiment.run_monitor_study(
            dataset=ds, h=args.hidden, fracs=fracs, beta=args.beta, steps=args.steps,
            check_every=args.check_every, seed=args.seed)
    if args.json:
        print(json.dumps(experiment.monitor_as_json(study), ensure_ascii=False, indent=2))
    else:
        print(experiment.render_monitor_study(study))
    return EXIT_OK if (study.false_alarms == 0 and study.missed == 0) else EXIT_GAP


def cmd_transient(args: argparse.Namespace) -> int:
    try:
        import numpy as np

        from . import transient as tr
    except ImportError as exc:
        print(_NUMPY_HINT.format(exc=exc), file=sys.stderr)
        return EXIT_USAGE

    try:
        if args.matrix:
            M = tr.parse_matrix(args.matrix)
            title = "非正规性 / 瞬态增长（给定 M）"
        elif args.example == "normal":
            M = np.diag([0.9, 0.8])
            title = "非正规性 / 瞬态增长（正规矩阵对照）"
        elif args.example == "euler":
            A = np.array([[3.0, 12.0], [0.0, 3.0]])
            M = tr.iteration_matrix(A, args.eta)
            title = f"非正规性 / 瞬态增长（M = I − {args.eta:g}A）"
        else:
            M = tr.nonnormal_example()
            title = "非正规性 / 瞬态增长（Jordan 块 [[0.9,10],[0,0.9]]）"
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return EXIT_USAGE

    d = tr.diagnose_nonnormal(M, threshold=args.threshold, steps=args.steps, title=title)
    if args.json:
        print(json.dumps(tr.as_json(d), ensure_ascii=False, indent=2))
    else:
        if args.no_notes:
            d.notes = []
        print(tr.render(d, color=args.color))
    return d.exit_code


def cmd_adaptive(args: argparse.Namespace) -> int:
    try:
        from . import adaptive, datasets
    except ImportError as exc:
        print(_NUMPY_HINT.format(exc=exc), file=sys.stderr)
        return EXIT_USAGE
    if args.quick:
        ds = datasets.make_synthetic(n=300, d=5, seed=args.seed)
        study = adaptive.run_adaptive_study(dataset=ds, h=5, eps=args.eps,
                                            train_steps=60, test_steps=120,
                                            bisect_iters=10, seed=args.seed)
    else:
        ds = datasets.load_order_amount(path=args.csv, max_rows=args.rows, seed=args.seed)
        study = adaptive.run_adaptive_study(dataset=ds, h=args.hidden, eps=args.eps,
                                            seed=args.seed)
    if args.json:
        print(json.dumps(adaptive.as_json(study), ensure_ascii=False, indent=2))
    else:
        print(adaptive.render_adaptive_study(study))
    # 门控：预测必须能被实测印证（否则说明局部判据在这里不适用）
    ok = bool(study.measured.get("valid")) and study.measured.get("rel_err", 1.0) < 0.05
    return EXIT_OK if ok else EXIT_GAP


# ---------------------------------------------------------------------------
# argparse 与「以 - 开头的取值」
# ---------------------------------------------------------------------------
# `--spectrum "-1,-4"` 是完全自然的写法，但它在 Python < 3.14 上会直接
# `error: argument --spectrum: expected one argument`。
#
# 原因是 argparse 只把「纯负数」字面量（`^-\d+$|^-\d*\.\d+$`）当成值；
# `-1,-4` 带逗号、`-1+3i` 带 i，都不匹配，于是被当成选项。
# **Python 3.14 把这个判定删掉了**（`argparse._negative_number_matcher`
# 已不存在），所以这个 bug 只在 3.10~3.13 上暴露 —— 本地 3.14 跑测试全绿，
# 是 CI 的 3.10/3.12 才把它照出来。
#
# 修法：参数进 argparse 之前，把 `--spectrum <以 - 开头的数值>` 改写成
# `--spectrum=<数值>`。等号形式无歧义，任何版本都认。
#
# 改写条件刻意收紧成「负号后跟数字或小数点」，因为要保住一个反例：
# 用户漏写值时（`--spectrum --eta 0.5`）必须仍然老实报错，
# 不能把 `--eta` 当成谱值吞掉、把一个用法错误变成一个静默的错误结果。
#
# 哪些选项要纳入：**取值天然可能以 `-` 开头的那些**。
# `--spectrum "-1,-4"` 之外还有 `--matrix "-1.5,0;0,0.5"`（同一个坑，
# `-1.5,0;0,0.5` 同样不匹配"纯负数"正则）。
# 数值型选项（`--eta -0.5`）本来就被 argparse 认，不必管；
# `--fracs` / `--betas` 的取值是正的比率，写负号属于用户笔误，让它报错更好。
_DASH_VALUE_OPTIONS = frozenset({"--spectrum", "--matrix"})
_DASH_VALUE_RE = re.compile(r"^-[.\d]")


def normalize_dash_values(argv: Sequence[str]) -> list[str]:
    """把 `--opt -1,-4` 改写成 `--opt=-1,-4`，规避旧版 argparse 的判定。"""
    items = list(argv)
    out: list[str] = []
    index = 0
    while index < len(items):
        token = items[index]
        if (token in _DASH_VALUE_OPTIONS and index + 1 < len(items)
                and _DASH_VALUE_RE.match(items[index + 1])):
            out.append(f"{token}={items[index + 1]}")
            index += 2
            continue
        out.append(token)
        index += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    force_utf8_stdio()
    parser = build_parser()
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    args = parser.parse_args(normalize_dash_values(raw))
    try:
        if args.command == "check":
            return cmd_check(args)
        if args.command == "sweep":
            return cmd_sweep(args)
        if args.command == "selftest":
            return cmd_selftest(args)
        if args.command == "predict":
            return cmd_predict(args)
        if args.command == "monitor":
            return cmd_monitor(args)
        if args.command == "transient":
            return cmd_transient(args)
        if args.command == "adaptive":
            return cmd_adaptive(args)
    except BrokenPipeError:  # pragma: no cover
        return EXIT_OK
    parser.error(f"未知子命令 {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
