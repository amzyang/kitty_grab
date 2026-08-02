"""测试基础设施：在 kitty +runpy 环境中构造脱离 TUI 的 GrabHandler

运行方式见 tests/run.sh——测试必须通过 `kitty +runpy` 执行，
kitty.* 模块只在 kitty 自带的 Python 环境中可导入。

两个环境注意点：
- kitty 以 -O 运行 Python（__debug__=False），assert 会被剥离，
  断言一律用本模块的 check()；
- `kitty +runpy "<code>"` 内联代码有 exec 双命名空间问题（函数体
  看不到顶层 import），测试文件须经 runpy.run_path 执行。
"""
import os
import sys
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import _grab_ui  # noqa: E402
from kitten_options_types import defaults  # noqa: E402


def check(cond, msg):
    if not cond:
        raise RuntimeError('FAIL: {}'.format(msg))


def make_handler(lines, x=0, y=0, top=1, rows=36, cols=120, pad=0):
    """构造可直接调用动作方法的 GrabHandler。

    mock 掉 screen_size / cmd / print / quit_loop / _marker_cmd，
    复制结果落在 handler.result（confirm 后为 {'copy': ...}）。
    pad > 0 时用空行把 lines 补齐到 pad 行。
    """
    if pad:
        lines = lines + [''] * (pad - len(lines))
    args = SimpleNamespace(x=x, y=y, top_line=top,
                           copy_to='clipboard', title='t')
    h = _grab_ui.GrabHandler(args, defaults, lines)
    h._screen_size = SimpleNamespace(rows=rows, cols=cols)
    type(h).screen_size = property(lambda s: s._screen_size)
    h.quit_loop = lambda rc: None
    h.cmd = SimpleNamespace(set_window_title=lambda *a: None,
                            set_cursor_position=lambda *a: None)
    h.print = lambda *a, **k: None
    h._marker_cmd = lambda *a: None
    return h
