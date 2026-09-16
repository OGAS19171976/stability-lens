"""stability_lens.report — 把 :class:`~stability_lens.diagnose.Diagnosis` 渲染成文本。

只用标准库做对齐，不依赖任何终端探测库；颜色默认关闭（``color=True`` 时可开 ANSI）。
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any, Iterable, Sequence

from .diagnose import Diagnosis, fmt_c

__all__ = ["render", "render_rows", "render_notes", "render_evidence_table", "to_json", "LEVEL_TAG"]

LEVEL_TAG = {
    "ok": "[ 稳定 ]",
    "gap": "[ 缝隙 ]",
    "transient": "[ 瞬态 ]",
    "unstable": "[ 不稳定 ]",
    "info": "[ 建议 ]",
}
LEVEL_COLOR = {"ok": "\033[32m", "gap": "\033[33m", "transient": "\033[33m",
               "unstable": "\033[31m", "info": "\033[36m"}
RESET = "\033[0m"


def display_width(s: str) -> int:
    """终端显示宽度：CJK 全角字符算 2 列，其余算 1 列。

    ``str.ljust`` 按字符数补齐，含中文的标签会把后面的列推歪 —— 所以自己算宽度。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def pad_right(s: str, n: int) -> str:
    return s + " " * max(0, n - display_width(s))


def pad_left(s: str, n: int) -> str:
    return " " * max(0, n - display_width(s)) + s


def render_rows(rows: Sequence[tuple[str, str]], indent: int = 2) -> str:
    """两列读数字段的对齐渲染（按显示宽度对齐）。"""
    pad = " " * indent
    width = max((display_width(k) for k, _ in rows), default=0)
    return "\n".join(f"{pad}{pad_right(k, width)}  {v}" for k, v in rows)


def render_notes(notes: Sequence[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}· {n}" for n in notes)


def _wrap_dash(title: str, width: int = 78) -> str:
    fill = max(0, width - len(title) - 3)
    return f"-- {title} " + "-" * fill


def render(d: Diagnosis, color: bool = False, show_notes: bool = True) -> str:
    """完整渲染一次体检结果。"""
    tag = LEVEL_TAG.get(d.level, "[ ? ]")
    if color and d.level in LEVEL_COLOR:
        tag = LEVEL_COLOR[d.level] + tag + RESET

    lines = [_wrap_dash(d.title), f"  判定: {tag}  {d.verdict}", ""]
    lines.append(render_rows(d.rows))
    if show_notes and d.notes:
        lines.append("")
        lines.append("  提示:")
        lines.append(render_notes(d.notes))
    return "\n".join(lines)


def to_json(d: Diagnosis) -> dict[str, Any]:
    """结构化输出（可 JSON 序列化），供 CI / 其他工具消费。"""

    def conv(v: Any) -> Any:
        """递归把复数与 ±inf / NaN 变成 JSON 能表达的东西。"""
        if isinstance(v, complex):
            return [v.real, v.imag]
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v

    return {
        "rule": d.rule,
        "title": d.title,
        "verdict": d.verdict,
        "level": d.level,
        "exit_code": d.exit_code,
        "eta": conv(d.eta),
        "eta_max": conv(d.eta_max),
        "rho": conv(d.rho),
        "stable": d.stable,
        "hurwitz": d.hurwitz,
        "eigenvalues": [[z.real, z.imag] for z in d.eigenvalues],
        "rows": [[k, v] for k, v in d.rows],
        "notes": list(d.notes),
        "extra": {k: conv(v) for k, v in d.extra.items()},
    }


def render_evidence_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    """证据表（断言结果），列宽按**显示宽度**自适应。"""
    cols = len(headers)
    widths = [display_width(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], display_width(str(r[i])))
    sep = "=" * (sum(widths) + 3 * (cols - 1))
    lines = [sep]
    lines.append("   ".join(pad_right(str(headers[i]), widths[i]) for i in range(cols)))
    lines.append("-" * len(sep))
    for r in rows:
        cells = []
        for i in range(cols):
            cell = str(r[i])
            cells.append(pad_right(cell, widths[i]) if i < 2 else pad_left(cell, widths[i]))
        lines.append("   ".join(cells))
    lines.append(sep)
    return "\n".join(lines)
