"""索引/滚动算法：真实实现 vs 朴素参照实现的随机等价

参照实现是行为规范（早期 O(n²)/逐步 while 版本的直接翻译），
优化后的实现必须与其在随机输入上逐一相等。
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, make_handler  # noqa: E402

from _grab_ui import WRAP_MARKER  # noqa: E402

random.seed(42)


# ---- _logical_line_map：朴素逐行扫描作为规范 ----

def naive_logical_map(has_wrap, n):
    m = {}
    current_start = 1
    for line_num in range(1, n + 1):
        if not (line_num > 1 and has_wrap[line_num - 1]):
            current_start = line_num
        end_line = line_num
        while end_line < n and has_wrap[end_line]:
            end_line += 1
        m[line_num] = (current_start, end_line)
    return m


for trial in range(60):
    n = random.randint(1, 60)
    wrap = [random.random() < 0.4 for _ in range(n)]
    lines = [('L{}'.format(i + 1)) + (WRAP_MARKER if wrap[i] else '')
             for i in range(n)]
    h = make_handler(lines, rows=1)
    has_wrap = {i + 1: wrap[i] for i in range(n)}
    check(h._logical_line_map == naive_logical_map(has_wrap, n),
          'logical map mismatch: n={} wrap={}'.format(n, wrap))
print('_logical_line_map vs 朴素规范 OK (60 组随机)')


# ---- _absolute_line_to_position：逐步 while 版本作为规范 ----

def naive_abs_pos(target_line, point_line, point_y, point_top, rows, nlines):
    new_y = point_y + (target_line - point_line)
    new_top_line = point_top
    while new_y < 0:
        new_y += 1
        new_top_line -= 1
    while new_y >= rows:
        new_y -= 1
        new_top_line += 1
    new_top_line = max(1, new_top_line)
    new_top_line = min(new_top_line, max(1, nlines - rows + 1))
    return (target_line - new_top_line, new_top_line)


for trial in range(300):
    rows = random.randint(1, 40)
    nlines = random.randint(rows, 200)
    top = random.randint(1, max(1, nlines - rows + 1))
    y = random.randint(0, rows - 1)
    target = random.randint(1, nlines)
    h = make_handler(['L{}'.format(i + 1) for i in range(nlines)],
                     x=0, y=y, top=top, rows=rows)
    p = h._absolute_line_to_position(target)
    ref = naive_abs_pos(target, y + top, y, top, rows, nlines)
    check((p.y, p.top_line) == ref,
          'abs pos mismatch: target={} y={} top={} rows={} n={} '
          'got={} ref={}'.format(target, y, top, rows, nlines,
                                 (p.y, p.top_line), ref))
print('_absolute_line_to_position vs 逐步规范 OK (300 组随机)')
