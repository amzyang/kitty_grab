"""行尾光标 vim 语义：h/l 按字符步进、不越行尾；G/h跨wrap/b回退落在末字符"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, make_handler  # noqa: E402

from _grab_ui import WRAP_MARKER  # noqa: E402

# ---- right()：不越过行尾最后一个字符 ----
h = make_handler(['abc', 'x'], x=1, y=0, pad=40)
p = h.right(); check(p.x == 2, 'l b->c: {}'.format(p.x))
h.point = p
p = h.right(); check(p.x == 2, 'l stays on c: {}'.format(p.x))

h = make_handler(['中文', 'x'], x=0, y=0, pad=40)
p = h.right(); check(p.x == 2, 'l 中->文: {}'.format(p.x))
h.point = p
p = h.right(); check(p.x == 2, 'l stays on 文: {}'.format(p.x))

# wrap 行末字符上 l 直接跳下一视觉行
h = make_handler(['abcd' + WRAP_MARKER, 'efg'], x=3, y=0, pad=40)
p = h.right(); check((p.line, p.x) == (2, 0), 'l wrap jump: {}'.format(p))

# 空行 l 停住
h = make_handler(['', 'x'], x=0, y=0, pad=40)
p = h.right(); check((p.line, p.x) == (1, 0), 'l empty stays: {}'.format(p))
print('right() vim 语义 OK')

# ---- right() block 模式 virtual edit 保持原行为 ----
h = make_handler(['中文', 'x'], x=2, y=0, pad=40)
h.set_mode('block')
p = h.right(); check(p.x == 4, 'block l 文->width: {}'.format(p.x))
h.point = p
p = h.right(); check(p.x == 5, 'block l virtual: {}'.format(p.x))
print('block virtual edit 保持 OK')

# ---- left()：按字符步进，与 right() 对称 ----
h = make_handler(['中文', 'x'], x=2, y=0, pad=40)
p = h.left(); check(p.x == 0, 'h 文->中 一步: {}'.format(p.x))
h = make_handler(['中文', 'x'], x=3, y=0, pad=40)   # 宽字符右半列（遗留位置）
p = h.left(); check(p.x == 0, 'h from mid-char col: {}'.format(p.x))
h = make_handler(['中文', 'x'], x=4, y=0, pad=40)   # 遗留过尾位置
p = h.left(); check(p.x == 2, 'h from past-end -> 文: {}'.format(p.x))
h = make_handler(['abc', 'x'], x=2, y=0, pad=40)
p = h.left(); check(p.x == 1, 'h ascii: {}'.format(p.x))

# h 跨 wrap 落在上一行末字符上，并与 l 往返对称
h = make_handler(['abcd' + WRAP_MARKER, 'efg'], x=0, y=1, pad=40)
p = h.left(); check((p.line, p.x) == (1, 3), 'h wrap -> last char: {}'.format(p))
h.point = p
p = h.right(); check((p.line, p.x) == (2, 0), 'h/l round trip: {}'.format(p))

# CJK wrap 行：宽 8，末字符 排 的起始列是 6（不是 width-1=7 的右半列）
h = make_handler(['中文混排' + WRAP_MARKER, '继续'], x=0, y=1, pad=40)
p = h.left(); check((p.line, p.x) == (1, 6), 'h CJK wrap: {}'.format(p))
h.point = p
p = h.right(); check((p.line, p.x) == (2, 0), 'h/l CJK round trip: {}'.format(p))
print('left() vim 语义 OK')

# ---- word_left 无 wrap 回退：落在上一行末字符 ----
h = make_handler(['abc', 'def'], x=0, y=1, pad=40)
p = h.word_left(); check((p.line, p.x) == (1, 2), 'b fallback: {}'.format(p))
print('word_left 回退 OK')

# ---- $ / ^$ / G：落在末字符的【起始列】，宽字符不得停在右半列 ----
# 'ab中文' 宽 6：文 占列 [4,5]，光标必须在 4
h = make_handler(['ab中文', 'x'], x=0, y=0, pad=40)
check(h.last().x == 4, '$ CJK on left half: {}'.format(h.last().x))
check(h.last_nonwhite().x == 4, '^$ CJK: {}'.format(h.last_nonwhite().x))

# 行尾带空白时 last_nonwhite 落在最后一个非空白字符的起始列
h = make_handler(['ab中文   ', 'x'], x=0, y=0, pad=40)
check(h.last_nonwhite().x == 4, '^$ trailing space: {}'.format(h.last_nonwhite().x))
check(h.last().x == 8, '$ trailing space (末字符是空格): {}'.format(h.last().x))

# ASCII 与空行
h = make_handler(['abc', 'x'], x=0, y=0, pad=40)
check(h.last().x == 2, '$ ascii: {}'.format(h.last().x))
h = make_handler(['', 'x'], x=0, y=0, pad=40)
check(h.last().x == 0, '$ empty line: {}'.format(h.last().x))

# 单个宽字符成行：光标在 0
h = make_handler(['中', 'x'], x=0, y=0, pad=40)
check(h.last().x == 0, '$ single wide char: {}'.format(h.last().x))

# $ 之后按 l 应停在原地（已在末字符上）
h = make_handler(['ab中文', 'x'], x=0, y=0, pad=40)
h.point = h.last()
check(h.right().x == 4, 'l after $ stays: {}'.format(h.right().x))

# G：末行 'end 中文' 宽 8，文 起始列 6
lines = ['first'] + [''] * 34 + ['end 中文']       # 恰 36 行，末行有内容
h = make_handler(lines, x=0, y=0)
p = h.bottom()
check(p.line == 36, 'G line: {}'.format(p))
check(p.x == 6, 'G x on 文 left half: {}'.format(p.x))

# block 模式虚拟位置切回 normal 时也 clamp 到末字符起始列
h = make_handler(['ab中文', 'x'], x=0, y=0, pad=40)
h.set_mode('block')
h.point = h.point._replace(x=9)                    # 虚拟位置
h.set_mode('normal')
check(h.point.x == 4, 'block->normal clamp: {}'.format(h.point.x))
print('$ / ^$ / G / block-clamp 落在宽字符左半列 OK')
