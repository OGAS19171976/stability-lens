"""stability-lens — 增量式算法的离散稳定性体检。

把「算法 → ODE」流程里所有算得出来的部分做成一个零依赖的小库：

* **η_max**：前向 Euler 的绝对稳定域给出的解析步长上界 ``min 2(−Re λ)/|λ|²``；
* **谱半径**：``ρ(I + ηJ)``，以及它与数值二分临界点的互验；
* **Lyapunov 证书**：``J^T P + P J = −Q`` 的 2×2 解析解与正定性；
* **Heavy-ball**：``η_max = 2(1+β)/λ_max``、复根区 ``ρ = √β``、连续阻尼 ``a = (1−β)/√η``；
* **噪声地板**：常步长 LMS 的稳态方差 ``V = ησ²/(2 − ηs²) ∝ η``。

快速上手::

    >>> import stability_lens as sl
    >>> d = sl.diagnose_heavy_ball(eta=2.25, beta=0.5, lambda_max=2.0)
    >>> d.eta_max
    1.5
    >>> d.stable
    False
    >>> d.exit_code            # 2 = 缝隙：连续稳定，离散不稳定
    2

命令行::

    python -m stability_lens check --rule heavy-ball --beta 0.5 --lambda-max 2 --eta 2.25
    python -m stability_lens selftest

注意：``stability_lens.diagnose`` 这个名字在包顶层是**分发函数**；
子模块请用 ``from stability_lens.diagnose import diagnose_euler``。
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import core, report, selftest  # noqa: F401
from .core import __all__ as _core_all
from .core import *  # noqa: F401,F403
from .diagnose import __all__ as _diagnose_all
from .diagnose import *  # noqa: F401,F403
from .report import to_json  # noqa: F401

__all__ = [
    "__version__",
    "core",
    "report",
    "selftest",
    "to_json",
] + list(_core_all) + list(_diagnose_all)
