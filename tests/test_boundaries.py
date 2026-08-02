"""边界回归：缓冲区顶部按 k 不崩、word_end 跨行落点不偏移"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, make_handler  # noqa: E402

# --- up() 缓冲区顶部（历史 KeyError: 0 崩溃场景） ---
LINES = ['行{} 中文'.format(i) for i in range(1, 81)]

h = make_handler(LINES, x=0, y=0, top=1)
h.move('up')                                   # 曾在此 KeyError: 0
check(h.point.top_line == 1 and h.point.y == 0, 'top stays: {}'.format(h.point))

h = make_handler(LINES, x=0, y=0, top=2)
p = h.up()
check((p.y, p.top_line) == (0, 1), 'scroll up: {}'.format(p))

h = make_handler(LINES, x=0, y=5, top=3)
p = h.up()
check((p.y, p.top_line) == (4, 3), 'mid: {}'.format(p))

h = make_handler(LINES, x=0, y=2, top=1)
for _ in range(6):
    h.move('up')
check((h.point.y, h.point.top_line) == (0, 1), 'repeated k: {}'.format(h.point))
h.set_mode('visual')
for _ in range(3):
    h.move('up')
check((h.point.y, h.point.top_line) == (0, 1), 'visual k: {}'.format(h.point))
print('up() 顶部边界 OK')

# --- word_end 跨行落点（曾偏移一格；CJK 下偏两列） ---
h = make_handler(['abc def', 'xxx'], x=0, y=0, pad=40)
p = h.word_end()
check((p.line, p.x) == (1, 2), 'in-line e on c: {}'.format(p))

h = make_handler(['abc', 'def ghi'], x=2, y=0, pad=40)
p = h.word_end()
check((p.line, p.x) == (2, 2), 'cross-line e lands ON f: {}'.format(p))

h = make_handler(['abc', '中文词 tail'], x=2, y=0, pad=40)
p = h.word_end()
check((p.line, p.x) == (2, 4), 'cross-line e CJK: {}'.format(p))

h = make_handler(['abc', '   word here'], x=2, y=0, pad=40)
p = h.word_end()
check((p.line, p.x) == (2, 6), 'leading-space e on d: {}'.format(p))
print('word_end 跨行落点 OK')
