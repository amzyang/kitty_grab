"""模态与选区逻辑：set_mode toggle、LineRegion、wrap 合并复制、block 矩形、搜索、o"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, make_handler  # noqa: E402

import _grab_ui  # noqa: E402
from _grab_ui import WRAP_MARKER, Position  # noqa: E402

W = WRAP_MARKER
LINES = [
    'English line one words',
    '中文很长的一行会被折' + W,      # 行2: 逻辑行 2-3（wrap）
    '行继续内容尾部',                 # 行3
    'aaa bbb ccc',                   # 行4
    '中文字符列一二三',               # 行5
    'xx中文字符列四五六yy',           # 行6
]


def mk(x=0, y=0, top=1):
    return make_handler(LINES, x=x, y=y, top=top, pad=40)


# --- 1. set_mode toggle 语义（vim：同键退出，异键切换） ---
h = mk()
h.set_mode('visual');  check(h.mode == 'visual', 'v enter: ' + h.mode)
h.set_mode('visual');  check(h.mode == 'normal', 'v toggle: ' + h.mode)
h.set_mode('block');   check(h.mode == 'block', 'C-v enter: ' + h.mode)
h.set_mode('block');   check(h.mode == 'normal', 'C-v toggle: ' + h.mode)
h.set_mode('line');    check(h.mode == 'line', 'V enter: ' + h.mode)
h.set_mode('visual');  check(h.mode == 'visual', 'V->v switch: ' + h.mode)
h.set_mode('line');    check(h.mode == 'line', 'v->V switch: ' + h.mode)
h.set_mode('line');    check(h.mode == 'normal', 'V toggle: ' + h.mode)
h.set_mode('normal');  check(h.mode == 'normal', 'normal idempotent')
print('1. set_mode toggle OK')

# --- 2. line 模式 mark_type 与名称 ---
h = mk()
h.set_mode('line')
check(h.mark_type is _grab_ui.LineRegion, 'mark_type is LineRegion')
check(h.mark_type.name == 'line', 'name==line')
check(issubclass(_grab_ui.LineRegion, _grab_ui.StreamRegion), 'inheritance')
print('2. LineRegion (继承 StreamRegion) OK')

# --- 3. V 在 wrap 中文行上复制：合并为完整逻辑行，无换行 ---
h = mk(x=0, y=1)   # 光标在行2（wrap 行前半）
h.set_mode('line')
h.confirm()
expected = '中文很长的一行会被折行继续内容尾部'
check(h.result == {'copy': expected}, 'V wrap copy: {!r}'.format(h.result))
print('3. V wrap 合并复制 OK')

# --- 4. yy 在 wrap 行后半上：同样合并整个逻辑行 ---
h = mk(x=4, y=2)   # 光标在行3（wrap 行后半）
h.yank_line()
check(h.result == {'copy': expected}, 'yy wrap copy: {!r}'.format(h.result))
print('4. yy wrap 合并 OK')

# --- 5. block 矩形选择跨中文行 ---
h = mk(x=2, y=4)
h.set_mode('block')
h.mark = Position(2, 4, 1)          # 行5 col2
h.point = Position(7, 5, 1)         # 行6 col7
h.confirm()
check(h.result['copy'] == '文字符\n中文字',
      'block rect: {!r}'.format(h.result['copy']))
print('5. block 矩形 CJK OK')

# --- 6. 前向搜索中文 + n 跳转 ---
h = mk()
h.search_mode = 'forward'
h.search_query = '中文'
h._perform_search()
check(len(h.search_matches) == 3, 'match count: {}'.format(len(h.search_matches)))
cur = h.search_matches[h.current_match_index]
check((cur.line, cur.x) == (2, 0), 'first fwd match: {}'.format((cur.line, cur.x)))
h.search_next()
cur = h.search_matches[h.current_match_index]
check((cur.line, cur.x) == (5, 0), 'next match: {}'.format((cur.line, cur.x)))
print('6. 前向搜索中文 + n OK')

# --- 7. - / + 逻辑行移动（跨 wrap） ---
h = mk(x=3, y=3)   # 行4
p = h.line_up()
check(p.line == 2, 'line_up -> {}'.format(p.line))
h = mk(x=0, y=1)   # 行2（wrap 前半）
p = h.line_down()
check(p.line == 4, 'line_down -> {}'.format(p.line))
print('7. -/+ 跨 wrap 逻辑行 OK')

# --- 8. o 端点交换 ---
h = mk(x=0, y=4)
h.set_mode('visual')
h.point = Position(6, 5, 1)
h.toggle_selection_end()
check((h.point.line, h.point.x) == (5, 0), 'point after o: {}'.format(h.point))
check((h.mark.line, h.mark.x) == (6, 6), 'mark after o: {}'.format(h.mark))
print('8. o 端点交换 OK')

# --- 9. w/b/e 在中英混排上 ---
h = make_handler(['mixed 中文English混排 line tail'], pad=36)
p = h.word_right();  check(p.x == 6, 'w1 -> {}'.format(p.x))
h.point = p
p = h.word_right();  check(p.x == 22, 'w2 -> {}'.format(p.x))
h.point = p
p = h.word_left();   check(p.x == 6, 'b -> {}'.format(p.x))
h.point = p
p = h.word_end();    check(p.x == 19, 'e -> {}'.format(p.x))
print('9. w/b/e 中英混排 OK (6/22/6/19)')

# --- 10. Position 比较运算符（_key，按 (line, x) 而非 tuple 序） ---
a, b = Position(5, 0, 2), Position(0, 1, 2)   # a: line2 x5; b: line3 x0
check(a < b and b > a and a <= b and b >= a and a != b, 'lt/gt/le/ge/ne')
c = Position(5, 1, 1)                          # line2 x5 == a
check(a == c and a <= c and a >= c, 'eq cross-representation')
check(not (a < c) and not (a > c), 'strict')
print('10. Position 比较 OK')
