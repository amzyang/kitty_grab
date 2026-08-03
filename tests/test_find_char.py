"""f/F/t/T 与 ;/, 的 Neovim 语义：落点列、till 重复、CJK 左半列、跨行、yank 组合"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check, make_handler  # noqa: E402

import kitty.key_encoding as kk  # noqa: E402
from kitty.key_encoding import KeyEvent  # noqa: E402

from _grab_ui import WRAP_MARKER  # noqa: E402


def press(h, key, text=None):
    """走完整 on_key_event 分派路径（普通可打印键的 text 即 key 本身）"""
    h.on_key_event(KeyEvent(key=key, type=kk.PRESS,
                            text=key if text is None else text))


def find(h, direction, kind, ch, scope='line', repeat=False):
    return h._find_char_position(direction, kind, scope, ch, repeat)


# ---- 落点：f 落在目标上，t 落在目标前一格 ----
# 'hello world'：h0 e1 l2 l3 o4 _5 w6 o7 r8 l9 d10
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
check(find(h, 'forward', 'find', 'o').x == 4, 'fo -> 4')
check(find(h, 'forward', 'till', 'o').x == 3, 'to -> 3')

# 找不到目标返回 None（光标不动由 _apply_find 保证）
check(find(h, 'forward', 'find', 'z') is None, 'no match -> None')

# ---- till 重复：Neovim 默认 cpo 不含 ';'，重复不卡在原地 ----
h.point = h.point._replace(x=3)
check(find(h, 'forward', 'till', 'o', repeat=True).x == 6, 't; -> 6')
check(find(h, 'forward', 'till', 'o').x == 3, 't 首次原地不动')

# ---- 反向 F/T ----
h.point = h.point._replace(x=7)
check(find(h, 'backward', 'find', 'o').x == 4, 'Fo -> 4')
check(find(h, 'backward', 'till', 'o').x == 5, 'To -> 5')
h.point = h.point._replace(x=5)
check(find(h, 'backward', 'till', 'o', repeat=True) is None, 'T; 无更多目标')
print('f/F/t/T 基本落点 OK')

# ---- CJK：落点必须是目标字符的起始列，不能是「列 ±1」 ----
# '中文abc中文'：中0(列0) 文1(列2) a2(4) b3(5) c4(6) 中5(7) 文6(9)
h = make_handler(['中文abc中文', 'x'], x=0, y=0, pad=40)
check(find(h, 'forward', 'find', '中').x == 7, 'f中 -> 列 7')
check(find(h, 'forward', 'till', '中').x == 6, 't中 -> c 的列 6')
check(find(h, 'forward', 'find', '文').x == 2, 'f文 -> 列 2')

h.point = h.point._replace(x=9)
check(find(h, 'backward', 'find', '文').x == 2, 'F文 -> 列 2')
check(find(h, 'backward', 'till', '文').x == 4, 'T文 -> a 的列 4')

# T 落点本身是宽字符：'a中b' 中 a0(列0) 中1(列1) b2(列3)
# 用「目标列 - 1」会得到 2，即 中 的右半列，只显示半个光标
h = make_handler(['a中b', 'x'], x=3, y=0, pad=40)
check(find(h, 'backward', 'till', 'a').x == 1, 'T 落在宽字符左半列')
print('CJK 落点在宽字符左半列 OK')

# ---- 跨 wrap 折行（line scope 内的逻辑行延续）----
h = make_handler(['abc' + WRAP_MARKER, 'def'], x=0, y=0, pad=40)
p = find(h, 'forward', 'find', 'e')
check((p.line, p.x) == (2, 1), 'f 跨 wrap: {}'.format(p))
p = find(h, 'forward', 'till', 'e')
check((p.line, p.x) == (2, 0), 't 跨 wrap: {}'.format(p))
# till 落点回落到上一视觉行的末字符
p = find(h, 'forward', 'till', 'd')
check((p.line, p.x) == (1, 2), 't wrap 边界回落: {}'.format(p))

# CJK + wrap：'中文' + '混排' 平铺为 '中文混排'
h = make_handler(['中文' + WRAP_MARKER, '混排'], x=0, y=0, pad=40)
p = find(h, 'forward', 'find', '排')
check((p.line, p.x) == (2, 2), 'f CJK 跨 wrap: {}'.format(p))
p = find(h, 'forward', 'till', '混')
check((p.line, p.x) == (1, 2), 't CJK wrap 边界回落到 文: {}'.format(p))
print('跨 wrap 折行 OK')

# ---- buffer scope：跨真正的行边界 ----
h = make_handler(['abc', 'def'], x=0, y=0, pad=40)
check(find(h, 'forward', 'find', 'e') is None, 'line scope 不跨行')
p = find(h, 'forward', 'find', 'e', scope='buffer')
check((p.line, p.x) == (2, 1), 'buffer scope 跨行: {}'.format(p))
p = find(h, 'forward', 'till', 'd', scope='buffer')
check((p.line, p.x) == (1, 2), 'buffer till 回落到上一行末字符: {}'.format(p))

# 空行不含字符，平铺后应被跳过而不是被映射到
h = make_handler(['ab', '', 'cd'], x=0, y=0, pad=40)
p = find(h, 'forward', 'find', 'c', scope='buffer')
check((p.line, p.x) == (3, 0), 'buffer 跳过空行: {}'.format(p))
print('buffer scope 跨行 OK')

# ---- 按键路径：分派顺序、;/, 重复 ----
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'f')
check(h.pending_find == ('forward', 'find', 'buffer'), 'f 置 pending_find')
press(h, 'o')   # o 平时绑定 toggle_selection_end，此处必须被 pending_find 拦截
check(h.point.x == 4 and h.pending_find is None, 'f o 移动并清 pending')

press(h, ';')
check(h.point.x == 7, '; 同向重复: {}'.format(h.point))
press(h, ',')
check(h.point.x == 4, ', 反向重复: {}'.format(h.point))

# ; 对 t 走重复语义（不卡在原地）
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 't'); press(h, 'o')
check(h.point.x == 3, 't o -> 3: {}'.format(h.point))
press(h, ';')
check(h.point.x == 6, 't 之后 ; -> 6: {}'.format(h.point))

# 无文本的按键取消查找，光标不动
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'f')
press(h, 'ESCAPE', text='')
check(h.pending_find is None and h.point.x == 0, 'Escape 取消 f')
print('按键分派与 ;/, 重复 OK')

# ---- 与 yank operator 组合 ----
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'y')
check(h.pending_operator == 'y', 'y 置 operator')
press(h, 'f')
check(h.pending_operator == 'y' and h.pending_find is not None,
      'yf 保留 operator 并等字符')
press(h, 'o')
check(h.result == {'copy': 'hello'}, 'yfo 复制 hello: {!r}'.format(h.result))

# 查找失败：不复制、不退出、不移动，且撤销 operator
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'y'); press(h, 'f'); press(h, 'z')
check(h.result is None, '失败的 yf 不复制')
check(h.pending_operator is None, '失败的 yf 撤销 operator')
check(h.point.x == 0, '失败的 yf 不移动光标')

# yt 走 till 落点
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'y'); press(h, 't'); press(h, 'o')
check(h.result == {'copy': 'hell'}, 'yto 复制 hell: {!r}'.format(h.result))
print('yank + f/t 组合 OK')

# ---- visual 模式下 f 扩展选区 ----
h = make_handler(['hello world', 'x'], x=0, y=0, pad=40)
press(h, 'v')
press(h, 'f'); press(h, 'o')
check(h.point.x == 4 and h.mark is not None, 'visual f 扩展选区')
h.confirm()
check(h.result == {'copy': 'hello'}, 'visual f 选区: {!r}'.format(h.result))
print('visual 模式 f 扩展选区 OK')
