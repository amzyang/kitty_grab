import os
import re
from typing import Any, Dict, List, Sequence

from kittens.tui.handler import result_handler
try:
    # For kitty v0.42+
    from kitty.typing_compat import BossType
except ModuleNotFoundError:
    # Fallback for older versions of kitty.
    from kitty.typing import BossType

import _grab_ui

# 预编译正则表达式（性能优化）
_SGR_CR_PATTERN = re.compile(r'\x1b\[[0-9;]*m\r')
_LONE_CR_PATTERN = re.compile(r'\r(?!\n)')


def main(args: List[str]) -> None:
    pass


@result_handler(no_ui=True)
def handle_result(args: List[str], data: Dict[str, Any], target_window_id: int, boss: BossType) -> None:
    window = boss.window_id_map.get(target_window_id)
    if window is None:
        return
    tab = window.tabref()
    if tab is None:
        return
    content = window.as_text(as_ansi=True, add_history=True,
                             add_wrap_markers=True)
    # convert all newlines to UNIX-style, but keep new-line wrap markers
    # '=65h' used as placeholder (looks like unused OSC)

    # Kitty wrap marker 格式分析（基于源代码 kitty/line.c）:
    # 1. 有 SGR 重置: 行内容 + \x1b[m + \r (当 ansibuf->len > 0)
    # 2. 无 SGR 重置: 行内容 + \r (当 ansibuf->len == 0，空行或只有空格)

    # 步骤1: 转换带 SGR 重置的 wrap marker（合并了 \r\n 和 \r 两种情况）
    content = _SGR_CR_PATTERN.sub('\x1b[=65h', content)

    # 步骤2: 转换不带 SGR 重置的 wrap marker（单独的 \r，但不是 \r\n 的一部分）
    content = _LONE_CR_PATTERN.sub('\x1b[=65h\n', content)

    # 步骤3: 清理剩余的 \r\n 和 \r
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    n_lines = content.count('\n')
    top_line = (n_lines - (window.screen.lines - 1) - window.screen.scrolled_by)
    boss._run_kitten(_grab_ui.__file__, args=[
        *args[1:],
        '--title={}'.format(window.title),
        '--cursor-x={}'.format(window.screen.cursor.x),
        '--cursor-y={}'.format(window.screen.cursor.y),
        '--top-line={}'.format(top_line)],
        input_data=content.encode('utf-8'),
        window=window)
