from base64 import b64encode
from functools import total_ordering
from itertools import takewhile
import json
import os.path
import re
import sys
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    NamedTuple, Optional, Set, Tuple, Type, Union)
import unicodedata

from kitty.boss import Boss
from kitty.cli import parse_args
from kitten_options_types import Options, defaults
from kitten_options_parse import create_result_dict, merge_result_dicts, parse_conf_item
from kitty.conf.utils import load_config as _load_config, parse_config_base, resolve_config
from kitty.constants import config_dir
from kitty.fast_data_types import truncate_point_for_length, wcswidth
import kitty.key_encoding as kk
from kitty.key_encoding import KeyEvent
from kitty.rgb import color_as_sgr
from kittens.tui.handler import Handler
from kittens.tui.loop import Loop


try:
    from kitty.clipboard import set_clipboard_string
except ImportError:
    from kitty.fast_data_types import set_clipboard_string


if TYPE_CHECKING:
    from typing_extensions import TypedDict
    ResultDict = TypedDict('ResultDict', {'copy': str})

# Line-wrapping 标记常量
WRAP_MARKER = '\x1b[=65h'

AbsoluteLine = int
ScreenLine = int
ScreenColumn = int
SelectionInLine = Union[Tuple[ScreenColumn, ScreenColumn],
                        Tuple[None, None]]


PositionBase = NamedTuple('Position', [
    ('x', ScreenColumn), ('y', ScreenLine), ('top_line', AbsoluteLine)])
class Position(PositionBase):
    """
    Coordinates of a cell.

    :param x: 0-based, left of window, to the right
    :param y: 0-based, top of window, down
    :param top_line: 1-based, start of scrollback, down
    """
    @property
    def line(self) -> AbsoluteLine:
        """
        Return 1-based absolute line number.
        """
        return self.y + self.top_line

    def moved(self, dx: int = 0, dy: int = 0,
              dtop: int = 0) -> 'Position':
        """
        Return a new position specified relative to self.
        """
        return self._replace(x=self.x + dx, y=self.y + dy,
                             top_line=self.top_line + dtop)

    def scrolled(self, dtop: int = 0) -> 'Position':
        """
        Return a new position equivalent to self
        but scrolled dtop lines.
        """
        return self.moved(dy=-dtop, dtop=dtop)

    def scrolled_up(self, rows: ScreenLine) -> 'Position':
        """
        Return a new position equivalent to self
        but with top_line as small as possible.
        """
        return self.scrolled(-min(self.top_line - 1,
                                  rows - 1 - self.y))

    def scrolled_down(self, rows: ScreenLine,
                      lines: AbsoluteLine) -> 'Position':
        """
        Return a new position equivalent to self
        but with top_line as large as possible.
        """
        return self.scrolled(min(lines - rows + 1 - self.top_line,
                                 self.y))

    def scrolled_towards(self, other: 'Position', rows: ScreenLine,
                         lines: Optional[AbsoluteLine] = None) -> 'Position':
        """
        Return a new position equivalent to self.
        If self and other fit within a single screen,
        scroll as little as possible to make both visible.
        Otherwise, scroll as much as possible towards other.
        """
        #  @ 
        #  .|   .    @|   .    .
        # |.|  |.   |.|  |.   |.|
        # |*|  |*|  |*|  |*|  |*|
        # |.   |.|  |.   |.|  |@|
        #  .    .|   .    @|   .
        #       @
        if other.line <= self.line - rows:         # above, unreachable
            return self.scrolled_up(rows)
        if other.line >= self.line + rows:         # below, unreachable
            assert lines is not None
            return self.scrolled_down(rows, lines)
        if other.line < self.top_line:             # above, reachable
            return self.scrolled(other.line - self.top_line)
        if other.line > self.top_line + rows - 1:  # below, reachable
            return self.scrolled(other.line - self.top_line - rows + 1)
        return self                                # visible

    def __str__(self) -> str:
        return '{},{}+{}'.format(self.x, self.y, self.top_line)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) < (other.line, other.x)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) <= (other.line, other.x)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) > (other.line, other.x)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) >= (other.line, other.x)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) == (other.line, other.x)

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.line, self.x) != (other.line, other.x)


def _span(line: AbsoluteLine, *lines: AbsoluteLine) -> Set[AbsoluteLine]:
    return set(range(min(line, *lines), max(line, *lines) + 1))


class Region:
    name = None  # type: Optional[str]
    uses_mark = False

    @staticmethod
    def line_inside_region(current_line: AbsoluteLine,
                           start: Position, end: Position) -> bool:
        """
        Return True if current_line is entirely inside the region
        defined by start and end.
        """
        return False

    @staticmethod
    def line_outside_region(current_line: AbsoluteLine,
                            start: Position, end: Position) -> bool:
        """
        Return True if current_line is entirely outside the region
        defined by start and end.
        """
        return current_line < start.line or end.line < current_line

    @staticmethod
    def adjust(start: Position, end: Position) -> Tuple[Position, Position]:
        """
        Return the normalized pair of markers
        equivalent to start and end. This is region-type-specific.
        """
        return start, end

    @staticmethod
    def selection_in_line(
            current_line: int, start: Position, end: Position,
            maxx: int) -> SelectionInLine:
        """
        Return bounds of the part of current_line
        that are within the region defined by start and end.
        """
        return None, None

    @staticmethod
    def lines_affected(mark: Optional[Position], old_point: Position,
                       point: Position) -> Set[AbsoluteLine]:
        """
        Return the set of lines (1-based, top of scrollback, down)
        that must be redrawn when point moves from old_point.
        """
        return set()

    @staticmethod
    def page_up(mark: Optional[Position], point: Position,
                rows: ScreenLine, lines: AbsoluteLine) -> Position:
        """
        Return the position page up from point.
        """
        #                          ........
        #                          ....$...|
        #  ........    ....$...|   ........|
        # |....$...|  |....^...|  |....^...|
        # |....^...|  |........|  |........
        # |........|  |........   |........
        #  ........    ........    ........
        if point.y > 0:
            return Position(point.x, 0, point.top_line)
        assert point.y == 0
        return Position(point.x, 0,
                        max(1, point.top_line - rows + 1))

    @staticmethod
    def page_down(mark: Optional[Position], point: Position,
                  rows: ScreenLine, lines: AbsoluteLine) -> Position:
        """
        Return the position page down from point.
        """
        #  ........    ........    ........
        # |........|  |........   |........
        # |....^...|  |........|  |........
        # |....$...|  |....^...|  |....^...|
        #  ........    ....$...|   ........|
        #                          ....$...|
        #                          ........
        maxy = rows - 1
        if point.y < maxy:
            return Position(point.x, maxy, point.top_line)
        assert point.y == maxy
        return Position(point.x, maxy,
                        min(lines - maxy, point.top_line + maxy))


class NoRegion(Region):
    name = 'unselected'
    uses_mark = False

    @staticmethod
    def line_outside_region(current_line: AbsoluteLine,
                            start: Position, end: Position) -> bool:
        return False


class MarkedRegion(Region):
    uses_mark = True

    # When a region is marked,
    # override page up and down motion
    # to keep as much region visible as possible.
    #
    # This means,
    # after computing the position in the usual way,
    # do the minimum possible scroll adjustment
    # to bring both mark and point on screen.
    # If that is not possible,
    # do the maximum possible scroll adjustment
    # towards mark
    # that keeps point on screen.
    @staticmethod
    def page_up(mark: Optional[Position], point: Position,
                rows: ScreenLine, lines: AbsoluteLine) -> Position:
        assert mark is not None
        return (Region.page_up(mark, point, rows, lines)
                .scrolled_towards(mark, rows, lines))

    @staticmethod
    def page_down(mark: Optional[Position], point: Position,
                  rows: ScreenLine, lines: AbsoluteLine) -> Position:
        assert mark is not None
        return (Region.page_down(mark, point, rows, lines)
                .scrolled_towards(mark, rows, lines))


class StreamRegion(MarkedRegion):
    name = 'stream'

    @staticmethod
    def line_inside_region(current_line: AbsoluteLine,
                           start: Position, end: Position) -> bool:
        return start.line < current_line < end.line

    @staticmethod
    def selection_in_line(
            current_line: AbsoluteLine, start: Position, end: Position,
            maxx: ScreenColumn) -> SelectionInLine:
        if StreamRegion.line_outside_region(current_line, start, end):
            return None, None
        return (start.x if current_line == start.line else 0,
                end.x + 1 if current_line == end.line else maxx)

    @staticmethod
    def lines_affected(mark: Optional[Position], old_point: Position,
                       point: Position) -> Set[AbsoluteLine]:
        return _span(old_point.line, point.line)


class ColumnarRegion(MarkedRegion):
    name = 'columnar'

    @staticmethod
    def adjust(start: Position, end: Position) -> Tuple[Position, Position]:
        return (start._replace(x=min(start.x, end.x)),
                end._replace(x=max(start.x, end.x)))

    @staticmethod
    def selection_in_line(
            current_line: AbsoluteLine, start: Position, end: Position,
            maxx: ScreenColumn) -> SelectionInLine:
        if ColumnarRegion.line_outside_region(current_line, start, end):
            return None, None
        return start.x, end.x + 1

    @staticmethod
    def lines_affected(mark: Optional[Position], old_point: Position,
                       point: Position) -> Set[AbsoluteLine]:
        assert mark is not None
        # If column changes, all lines change.
        if old_point.x != point.x:
            return _span(mark.line, old_point.line, point.line)
        # If point passes mark, all passed lines change except mark line.
        if old_point < mark < point or point < mark < old_point:
            return _span(old_point.line, point.line) - {mark.line}
        # If point moves away from mark,
        # all passed lines change except old point line.
        elif mark < old_point < point or point < old_point < mark:
            return _span(old_point.line, point.line) - {old_point.line}
        # Otherwise, point moves toward mark,
        # and all passed lines change except new point line.
        else:
            return _span(old_point.line, point.line) - {point.line}


class LineRegion(MarkedRegion):
    """行级选择区域（类似 vim 的 linewise visual mode，V 键）

    特性：
    - 总是选择完整的行（从列 0 到行尾）
    - 即使光标在行中间，也选择整行
    - 起始行和结束行都包含完整内容

    与 vim 的 linewise visual mode 对应：
    - 按 V 进入此模式
    - 选择区域总是整行高亮
    - 复制时获得完整的行内容
    """
    name = 'line'

    @staticmethod
    def line_inside_region(current_line: AbsoluteLine,
                           start: Position, end: Position) -> bool:
        """判断 current_line 是否完全在选区内部

        对于 LineRegion，与 StreamRegion 的行为相同
        """
        return start.line < current_line < end.line

    @staticmethod
    def selection_in_line(
            current_line: AbsoluteLine, start: Position, end: Position,
            maxx: ScreenColumn) -> SelectionInLine:
        """返回 current_line 中被选中的列范围

        LineRegion 的核心行为：总是返回完整行 (0, maxx)
        这确保了无论光标在行的哪个位置，都选择整行

        Args:
            current_line: 当前行号（1-based 绝对行号）
            start: 选区起始位置
            end: 选区结束位置
            maxx: 行的实际宽度（字符数，不是屏幕列数）

        Returns:
            (0, maxx): 选择整行
            (None, None): 行不在选区内
        """
        if LineRegion.line_outside_region(current_line, start, end):
            return None, None
        return (0, maxx)

    @staticmethod
    def lines_affected(mark: Optional[Position], old_point: Position,
                       point: Position) -> Set[AbsoluteLine]:
        """返回需要重绘的行集合

        LineRegion 的行为与 StreamRegion 相同：
        point 移动时，所有经过的行都需要重绘
        """
        return _span(old_point.line, point.line)


ActionName = str
ActionArgs = tuple
ShortcutMods = int
KeyName = str
Namespace = Any  # kitty.cli.Namespace (< 0.17.0)
OptionName = str
OptionValues = Dict[OptionName, Any]
TypeMap = Dict[OptionName, Callable[[Any], Any]]


def load_config(*paths: str, overrides: Optional[Iterable[str]] = None) -> Options:

    def parse_config(lines: Iterable[str]) -> Dict[str, Any]:
        ans: Dict[str, Any] = create_result_dict()
        parse_config_base(
            lines,
            parse_conf_item,
            ans,
        )
        return ans

    configs = list(resolve_config('/etc/xdg/kitty/grab.conf',
                                  os.path.join(config_dir, 'grab.conf'),
                                  config_files_on_cmd_line=[]))
    overrides = tuple(overrides) if overrides is not None else ()
    opts_dict, paths = _load_config(defaults, parse_config, merge_result_dicts, *configs, overrides=overrides)
    opts = Options(opts_dict)
    opts.config_paths = paths
    opts.config_overrides = overrides
    return opts


def unstyled(s: str) -> str:
    # 移除 SGR (Select Graphic Rendition) 序列
    s = re.sub(r'\x1b\[[0-9;:]*m', '', s)
    # 移除 OSC (Operating System Command) 序列，包括 shell integration
    s = re.sub(r'\x1b\](?:[^\x07\x1b]+|\x1b[^\\])*(?:\x1b\\|\x07)', '', s)
    # 额外清理各种可能的 shell integration 变种
    s = re.sub(r'\x1b\]133[^\x07\n]*\x07?', '', s)  # 清理 ]133 shell integration
    s = re.sub(r'\x1b\][0-9]+;[^\x07\n]*\x07?', '', s)  # 通用 OSC 序列清理
    # 移除 wrap marker 占位符（我们自定义的标记，用于标识 line-wrapping）
    s = s.replace(WRAP_MARKER, '')
    s = s.expandtabs()
    return s


def string_slice(s: str, start_x: ScreenColumn,
                 end_x: ScreenColumn) -> Tuple[str, bool]:
    prev_pos = (truncate_point_for_length(s, start_x - 1) if start_x > 0
                else None)
    start_pos = truncate_point_for_length(s, start_x)
    end_pos = truncate_point_for_length(s, end_x - 1) + 1
    return s[start_pos:end_pos], prev_pos == start_pos


DirectionStr = str
RegionTypeStr = str
ModeTypeStr = str


class GrabHandler(Handler):
    def __init__(self, args: Namespace, opts: Options,
                 lines: List[str]) -> None:
        super().__init__()
        self.args = args
        self.opts = opts
        self.lines = lines
        self.point = Position(args.x, args.y, args.top_line)
        self.mark = None           # type: Optional[Position]
        self.mark_type = NoRegion  # type: Type[Region]
        self.mode = 'normal'       # type: ModeTypeStr
        self.result = None         # type: Optional[ResultDict]

        # operator-pending 状态：等待 motion 输入
        # None 表示无 pending operator
        # 'y' 表示等待 motion 来执行 yank 操作
        self.pending_operator = None  # type: Optional[str]

        # g-pending 状态：等待 g 前缀键后的按键输入
        # False 表示无 pending g
        # True 表示等待 g 后的命令（如 gg, gj, gk 等）
        self.pending_g = False  # type: bool

        # Operating System Command (OSC); command number 52
        # c — clipboard
        # p — primary
        # s — secondary
        self.copy_to = {'primary': b'p', 'secondary': b's'}.get(args.copy_to, b'c')

        # 搜索状态
        self.search_mode = None           # type: Optional[str]  # None, 'forward', 'backward'
        self.search_query = ''            # type: str  # 当前输入的搜索词
        self.search_matches = []          # type: List[Position]  # 所有匹配位置列表
        self.current_match_index = -1     # type: int  # 当前匹配项索引

        for spec, action in self.opts.map:
            self.add_shortcut(action, spec)

        # 预处理所有行并缓存（性能优化）
        self._unstyled_cache = {}  # type: Dict[AbsoluteLine, str]
        self._width_cache = {}     # type: Dict[AbsoluteLine, int]
        self._has_wrap = {}        # type: Dict[AbsoluteLine, bool]

        # 第一阶段：预处理所有行
        for i, line in enumerate(lines):
            line_num = i + 1  # type: AbsoluteLine
            plain = unstyled(line)
            self._unstyled_cache[line_num] = plain
            self._width_cache[line_num] = wcswidth(plain)
            self._has_wrap[line_num] = WRAP_MARKER in line

        # 第二阶段：构建逻辑行边界索引（性能优化）
        self._logical_line_map = {}  # type: Dict[AbsoluteLine, Tuple[AbsoluteLine, AbsoluteLine]]

        current_start = 1
        for line_num in range(1, len(lines) + 1):
            # 检查上一行是否有 wrap marker
            if line_num > 1 and self._has_wrap[line_num - 1]:
                # 当前行是延续，不更新 current_start
                pass
            else:
                # 当前行是新逻辑行的开始
                current_start = line_num

            # 查找逻辑行的末尾
            end_line = line_num
            while end_line < len(lines) and self._has_wrap[end_line]:
                end_line += 1

            # 存储边界信息
            self._logical_line_map[line_num] = (current_start, end_line)

    def _start_end(self) -> Tuple[Position, Position]:
        start, end = sorted([self.point, self.mark or self.point])
        return self.mark_type.adjust(start, end)

    def _draw_line(self, current_line: AbsoluteLine) -> None:
        y = current_line - self.point.top_line  # type: ScreenLine
        line = self.lines[current_line - 1]
        clear_eol = '\x1b[m\x1b[K'
        sgr0 = '\x1b[m'

        plain = self._unstyled_cache[current_line]
        selection_sgr = '\x1b[38{};48{}m'.format(
            color_as_sgr(self.opts.selection_foreground),
            color_as_sgr(self.opts.selection_background))
        start, end = self._start_end()

        # 对于 LineRegion，展开范围用于显示判断
        if self.mark_type == LineRegion:
            expanded_start_line, expanded_end_line = self._expand_line_selection_for_wrap(
                start.line, end.line)
            # 创建临时 Position，调整 y 使得 line 属性 (y + top_line) 等于展开后的行号
            display_start = Position(start.x, expanded_start_line - start.top_line, start.top_line)
            display_end = Position(end.x, expanded_end_line - end.top_line, end.top_line)
        else:
            display_start, display_end = start, end

        # anti-flicker optimization
        if self.mark_type.line_inside_region(current_line, display_start, display_end):
            self.cmd.set_cursor_position(0, y)
            self.print('{}{}'.format(selection_sgr, plain),
                       end=clear_eol)
            return

        self.cmd.set_cursor_position(0, y)
        self.print('{}{}'.format(sgr0, line), end=clear_eol)

        if self.mark_type.line_outside_region(current_line, display_start, display_end):
            return

        start_x, end_x = self.mark_type.selection_in_line(
            current_line, display_start, display_end, self._width_cache[current_line])
        if start_x is None or end_x is None:
            return

        line_slice, half = string_slice(plain, start_x, end_x)
        self.cmd.set_cursor_position(start_x - (1 if half else 0), y)
        self.print('{}{}'.format(selection_sgr, line_slice), end='')

    def _update(self) -> None:
        self.cmd.set_window_title('Grab – {} {} {},{}+{} to {},{}+{}'.format(
            self.args.title,
            self.mark_type.name,
            getattr(self.mark, 'x', None), getattr(self.mark, 'y', None),
            getattr(self.mark, 'top_line', None),
            self.point.x, self.point.y, self.point.top_line))

        # 如果处于搜索输入模式，在屏幕底部显示搜索提示符
        if self.search_mode is not None:
            # 搜索提示符：'/' 表示向前搜索，'?' 表示向后搜索
            prompt = '/' if self.search_mode == 'forward' else '?'
            search_line = '{}{}'.format(prompt, self.search_query)

            # 在屏幕最后一行显示搜索提示
            bottom_y = self.screen_size.rows - 1
            self.cmd.set_cursor_position(0, bottom_y)

            # 使用反色高亮 (\x1b[7m)，不清除整行
            self.print('\x1b[7m{}\x1b[m'.format(search_line), end='')

            # 如果输入变短，清除多余的旧字符（使用 wcswidth 计算显示宽度）
            display_width = wcswidth(search_line)
            if hasattr(self, '_prev_search_len') and self._prev_search_len > display_width:
                spaces_to_clear = self._prev_search_len - display_width
                self.print(' ' * spaces_to_clear, end='')
            self._prev_search_len = display_width

            # 将光标定位到搜索查询的末尾（使用 wcswidth 计算显示宽度）
            cursor_x = display_width
            self.cmd.set_cursor_position(cursor_x, bottom_y)
        else:
            # 正常模式：将光标定位到 point 位置
            self.cmd.set_cursor_position(self.point.x, self.point.y)

    def _redraw_lines(self, lines: Iterable[AbsoluteLine]) -> None:
        for line in lines:
            self._draw_line(line)
        self._update()

    def _redraw(self) -> None:
        self._redraw_lines(range(
            self.point.top_line,
            self.point.top_line + self.screen_size.rows))

    def initialize(self) -> None:
        self.cmd.set_window_title('Grab – {}'.format(self.args.title))
        self.cmd.set_default_colors(cursor=self.opts.cursor)
        self._redraw()

    def perform_default_key_action(self, key_event: KeyEvent) -> bool:
        return False

    def _handle_search_input(self, key_event: KeyEvent) -> None:
        """处理搜索输入模式下的键盘事件"""
        if key_event.type not in [kk.PRESS, kk.REPEAT]:
            return

        key = key_event.key
        mods = key_event.mods

        # Enter 键：确认搜索
        if key == 'ENTER':
            if self.search_query:
                self._perform_search()
            else:
                # 空查询，取消搜索模式
                self.search_mode = None
                self._redraw()
            return

        # Escape 键：取消搜索
        if key == 'ESCAPE':
            self.search_mode = None
            self.search_query = ''
            # 清除 marker 高亮
            self._clear_search_marker()
            self._redraw()
            return

        # Backspace 键：删除最后一个字符
        if key == 'BACKSPACE':
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self._redraw()
            return

        # 可打印字符：添加到搜索查询
        # key_event.text 包含实际输入的字符（支持输入法一次输入多个字符）
        if key_event.text:
            # 只接受所有字符都是可打印的字符串
            if all(c.isprintable() for c in key_event.text):
                self.search_query += key_event.text
                self._redraw()

    def _handle_pending_operator(self, key_event: KeyEvent) -> None:
        """处理 operator-pending 模式下的键盘事件"""
        if key_event.type not in [kk.PRESS, kk.REPEAT]:
            return

        key = key_event.key
        mods = key_event.mods

        # Escape 键：取消 pending operator
        if key == 'ESCAPE':
            self.pending_operator = None
            return

        # 当前只支持 yank operator
        if self.pending_operator == 'y':
            # y + y = yank current line (特殊处理，使用 LineRegion)
            if key == 'y':
                self.yank_line()
                return

            # motion 键到方法名的映射表
            motion_map = {
                '$': 'last_nonwhite',    # y$: 复制到行尾
                '^': 'first_nonwhite',   # y^: 复制到第一个非空白字符
                '0': 'first',            # y0: 复制到行首
                'w': 'word_right',       # yw: 复制到下一个单词
                'b': 'word_left',        # yb: 复制到上一个单词
                'e': 'word_end',         # ye: 复制到单词末尾
                'h': 'left',             # yh: 复制左边一个字符
                'l': 'right',            # yl: 复制右边一个字符
                'j': 'down',             # yj: 复制当前行和下一行
                'k': 'up',               # yk: 复制当前行和上一行
            }

            # 查找对应的 motion 方法
            if key in motion_map:
                motion_method_name = motion_map[key]
                motion_method = getattr(self, motion_method_name)
                target_position = motion_method()
                self._execute_yank_motion(target_position)
                return

        # 其他按键：取消 pending operator（不识别的 motion）
        self.pending_operator = None

    def _handle_pending_g(self, key_event: KeyEvent) -> None:
        """处理 g 前缀键后的按键事件"""
        if key_event.type not in [kk.PRESS, kk.REPEAT]:
            return

        key = key_event.key

        # Escape 键：取消 pending g
        if key == 'ESCAPE':
            self.pending_g = False
            return

        # g + g = 跳转到顶部
        if key == 'g':
            self.move('top')
            self.pending_g = False
            return

        # g + j = 向下移动（在 kitty_grab 中等同于 j）
        if key == 'j':
            self.move('down')
            self.pending_g = False
            return

        # g + k = 向上移动（在 kitty_grab 中等同于 k）
        if key == 'k':
            self.move('up')
            self.pending_g = False
            return

        # 其他按键：取消 pending g（不识别的命令）
        self.pending_g = False

    def on_key_event(self, key_event: KeyEvent, in_bracketed_paste: bool = False) -> None:
        # 如果处于搜索输入模式，特殊处理键盘事件
        if self.search_mode is not None:
            self._handle_search_input(key_event)
            return

        # 如果处于 operator-pending 模式，特殊处理键盘事件
        if self.pending_operator is not None:
            self._handle_pending_operator(key_event)
            return

        # 如果处于 g-pending 模式，特殊处理键盘事件
        if self.pending_g:
            self._handle_pending_g(key_event)
            return

        action = self.shortcut_action(key_event)

        # 如果没有映射，且按下的是 'g' 键（无修饰符），进入 g-pending 状态
        if (action is None and
            key_event.type in [kk.PRESS, kk.REPEAT] and
            key_event.key == 'g' and
            key_event.mods == 0):
            self.pending_g = True
            return

        if (key_event.type not in [kk.PRESS, kk.REPEAT]
                or action is None):
            return
        self.perform_action(action)

    def perform_action(self, action: Tuple[ActionName, ActionArgs]) -> None:
        func, args = action
        getattr(self, func)(*args)

    def quit(self, *args: Any) -> None:
        # 退出时清除搜索 marker
        self._clear_search_marker()
        self.quit_loop(1)

    region_types = {'stream':   StreamRegion,
                    'line':     LineRegion,
                    'columnar': ColumnarRegion
                   }  # type: Dict[RegionTypeStr, Type[Region]]

    mode_types = {'normal': NoRegion,
                  'visual': StreamRegion,
                  'line':   LineRegion,
                  'block':  ColumnarRegion,
                  }  # type: Dict[ModeTypeStr, Type[Region]]

    def _ensure_mark(self, mark_type: Type[Region] = StreamRegion) -> None:
        need_redraw = mark_type is not self.mark_type
        self.mark_type = mark_type
        self.mark = (self.mark or self.point) if mark_type.uses_mark else None
        if need_redraw:
            self._redraw()

    def _scroll(self, dtop: int) -> None:
        rows = self.screen_size.rows
        new_point = self.point.moved(dtop=dtop)
        if not (0 < new_point.top_line <= 1 + len(self.lines) - rows):
            return
        self.point = new_point
        self._redraw()

    def scroll(self, direction: DirectionStr) -> None:
        self._scroll(dtop={'up': -1, 'down': 1}[direction])

    def left(self) -> Position:
        """向左移动一个字符（vim h 命令）

        如果在行首且上一行有 wrap marker，跳到上一行末尾（逻辑行延续）
        """
        if self.point.x > 0:
            return self.point.moved(dx=-1)

        # 在行首，检查上一行是否有 wrap marker
        if self.point.line > 1 and self._has_wrap_marker(self.point.line - 1):
            # 上一行有 wrap marker，当前行是延续，跳到上一行末尾
            prev_line = self._unstyled_cache[self.point.line - 1]
            return self._absolute_line_to_position(self.point.line - 1, x=wcswidth(prev_line))

        # 无 wrap 或已在第一行，停在当前位置
        return self.point

    def right(self) -> Position:
        """向右移动一个字符（vim l 命令）

        如果到达行尾且当前行有 wrap marker，跳到下一个视觉行的开始（逻辑行延续）
        """
        # 获取当前行的实际内容
        line = self._unstyled_cache[self.point.line]
        # 将显示列位置转换为字符串索引
        pos = truncate_point_for_length(line, self.point.x)

        # 检查是否还可以向右移动（未到达行尾）
        if pos < len(line):
            # 计算移动后的显示宽度
            new_x = wcswidth(line[:pos + 1])
            return Position(new_x, self.point.y, self.point.top_line)

        # 到达行尾，检查是否有 wrap marker
        if self._has_wrap_marker(self.point.line) and self.point.line < len(self.lines):
            # 有 wrap marker，跳到下一个视觉行的开始（逻辑行延续）
            return self._absolute_line_to_position(self.point.line + 1, x=0)

        # 无 wrap marker 或已到最后一行，停在当前位置
        return self.point

    def up(self) -> Position:
        return (self.point.moved(dy=-1) if self.point.y > 0 else
                self.point.moved(dtop=-1) if self.point.top_line > 0 else
                self.point)

    def down(self) -> Position:
        return (self.point.moved(dy=1)
                if self.point.y + 1 < self.screen_size.rows
                else self.point.moved(dtop=1)
                if self.point.line < len(self.lines)
                else self.point)

    def page_up(self) -> Position:
        return self.mark_type.page_up(
            self.mark, self.point, self.screen_size.rows,
            max(self.screen_size.rows, len(self.lines)))

    def page_down(self) -> Position:
        return self.mark_type.page_down(
            self.mark, self.point, self.screen_size.rows,
            max(self.screen_size.rows, len(self.lines)))

    def first(self) -> Position:
        """跳到逻辑行的开头（vim 0 命令）"""
        start_line = self._find_logical_line_start(self.point.line)
        return self._absolute_line_to_position(start_line, x=0)

    def first_nonwhite(self) -> Position:
        """跳到逻辑行的第一个非空白字符（vim ^ 命令）"""
        start_line = self._find_logical_line_start(self.point.line)
        line = self._unstyled_cache[start_line]
        prefix = ''.join(takewhile(str.isspace, line))
        return self._absolute_line_to_position(start_line, x=wcswidth(prefix))

    def last_nonwhite(self) -> Position:
        """返回当前逻辑行最后一个非空白字符的位置"""
        end_line = self._find_logical_line_end(self.point.line)
        line = self._unstyled_cache[end_line]
        suffix = ''.join(takewhile(str.isspace, reversed(line)))
        x = wcswidth(line[:len(line) - len(suffix)])
        return self._absolute_line_to_position(end_line, x=x)

    def last(self) -> Position:
        """跳到逻辑行的末尾（vim $ 命令）"""
        end_line = self._find_logical_line_end(self.point.line)
        line = self._unstyled_cache[end_line]
        return self._absolute_line_to_position(end_line, x=wcswidth(line))

    def top(self) -> Position:
        return Position(0, 0, 1)

    def bottom(self) -> Position:
        x = wcswidth(self._unstyled_cache[len(self.lines)])
        y = min(len(self.lines) - self.point.top_line,
                self.screen_size.rows - 1)
        return Position(x, y, len(self.lines) - y)

    def noop(self) -> Position:
        return self.point

    @property
    def _select_by_word_characters(self) -> str:
        return (self.opts.select_by_word_characters
                or (json.loads(os.getenv('KITTY_COMMON_OPTS', '{}'))
                    .get('select_by_word_characters', '@-./_~?&=%+#')))

    def _is_word_char(self, c: str) -> bool:
        return (unicodedata.category(c)[0] in 'LN'
                or c in self._select_by_word_characters)

    def _is_word_separator(self, c: str) -> bool:
        return (unicodedata.category(c)[0] not in 'LN'
                and c not in self._select_by_word_characters)

    # Line-wrapping 辅助方法

    def _has_wrap_marker(self, line_num: int) -> bool:
        """检查指定行是否有 wrap marker（优化版：使用预构建缓存）

        Args:
            line_num: 行号（1-based）

        Returns:
            是否有 wrap marker
        """
        return self._has_wrap.get(line_num, False)

    def _find_logical_line_start(self, from_line: int) -> int:
        """向上追溯找到逻辑行的起始行号（优化版：使用预构建索引）

        Args:
            from_line: 开始查找的行号（1-based）

        Returns:
            逻辑行的起始行号（1-based）
        """
        if from_line in self._logical_line_map:
            return self._logical_line_map[from_line][0]
        # 边界情况（理论上不应该发生）
        return from_line

    def _find_logical_line_end(self, from_line: int) -> int:
        """向下追溯找到逻辑行的末尾行号（优化版：使用预构建索引）

        Args:
            from_line: 开始查找的行号（1-based）

        Returns:
            逻辑行的末尾行号（1-based）
        """
        if from_line in self._logical_line_map:
            return self._logical_line_map[from_line][1]
        # 边界情况（理论上不应该发生）
        return from_line

    def _absolute_line_to_position(self, target_line: int, x: int = 0) -> Position:
        """将绝对行号转换为 Position（处理滚动和边界）

        Args:
            target_line: 目标绝对行号（1-based）
            x: x 坐标

        Returns:
            调整后的 Position
        """
        line_offset = target_line - self.point.line
        new_y = self.point.y + line_offset
        new_top_line = self.point.top_line

        # 向上滚动
        while new_y < 0:
            new_y += 1
            new_top_line -= 1

        # 向下滚动
        while new_y >= self.screen_size.rows:
            new_y -= 1
            new_top_line += 1

        # 边界检查
        new_top_line = max(1, new_top_line)
        max_top_line = max(1, len(self.lines) - self.screen_size.rows + 1)
        new_top_line = min(new_top_line, max_top_line)

        # 重新计算 y
        new_y = target_line - new_top_line

        return Position(x, new_y, new_top_line)

    def _is_word_split_at_wrap(self, line_num: int) -> bool:
        """检查指定行的 wrap 是否是单词分割

        Args:
            line_num: 当前行号（1-based），需要有 wrap marker

        Returns:
            是否是单词分割
        """
        if not self._has_wrap_marker(line_num):
            return False

        if line_num >= len(self.lines):
            return False

        current_line = self._unstyled_cache[line_num]
        next_line = self._unstyled_cache[line_num + 1]

        # 处理行尾留白（宽字符导致）：去除行尾空白后检查
        current_line_stripped = current_line.rstrip()
        current_ends_with_word = (len(current_line_stripped) > 0 and
                                 self._is_word_char(current_line_stripped[-1]))
        next_starts_with_word = len(next_line) > 0 and self._is_word_char(next_line[0])

        return current_ends_with_word and next_starts_with_word

    def _find_word_start_across_wraps(self, from_line: int) -> Tuple[int, int]:
        """跨 wrap 向上追溯找到单词的真正开始位置

        Args:
            from_line: 开始行号（1-based），该行应该是单词的一部分

        Returns:
            (line_num, pos): 单词开始的行号（1-based）和字符位置
        """
        target_line_num = from_line

        # 向上追溯，找到单词真正的开始行
        while target_line_num > 1:
            if not self._is_word_split_at_wrap(target_line_num - 1):
                break
            target_line_num -= 1

        # 在目标行中往回找单词开头
        target_line = self._unstyled_cache[target_line_num]
        # 从去除行尾空白后的位置开始，避免宽字符导致的留白
        pos = len(target_line.rstrip())
        while pos > 0 and self._is_word_char(target_line[pos - 1]):
            pos -= 1

        return (target_line_num, pos)

    def _find_next_word_end_from_line(self, start_line_idx: int) -> Optional[Tuple[int, int]]:
        """从指定行开始，查找下一个单词的末尾位置

        跳过空行和只有空白的行，找到第一个有单词的行。

        Args:
            start_line_idx: 开始查找的行索引（0-based，用于 self.lines）

        Returns:
            (line_idx, col_pos): 行索引（0-based）和列位置（字符串索引）
            如果没有找到返回 None
        """
        for line_idx in range(start_line_idx, len(self.lines)):
            line = self._unstyled_cache[line_idx + 1]
            # 跳过前导空白
            pos = 0
            while pos < len(line) and line[pos].isspace():
                pos += 1

            # 如果这行有单词
            if pos < len(line):
                # 移动到第一个单词的末尾
                pred = (self._is_word_char if self._is_word_char(line[pos])
                        else self._is_word_separator)
                while pos + 1 < len(line) and pred(line[pos + 1]):
                    pos += 1
                return (line_idx, pos)

        return None

    def word_left(self) -> Position:
        """向左移动到上一个单词开始（vim b 命令）

        支持跨越 wrap 的逻辑行延续和单词分割
        """
        line = self._unstyled_cache[self.point.line]
        pos = truncate_point_for_length(line, self.point.x)

        # 检查是否在行首且可能处于被分割单词的后半部分
        current_line_starts_with_word = pos == 0 and len(line) > 0 and self._is_word_char(line[0])

        if pos > 0:
            # 跳过空白字符
            while pos > 0 and line[pos - 1].isspace():
                pos -= 1

            # 跳过单词/分隔符
            if pos > 0:
                pred = self._is_word_char if self._is_word_char(line[pos - 1]) else self._is_word_separator
                new_pos = pos - len(''.join(takewhile(pred, reversed(line[:pos]))))

                # 检查是否到达行首且是单词分割
                if new_pos == 0 and self._is_word_split_at_wrap(self.point.line - 1):
                    # 向上追溯找到单词真正的开始
                    target_line, target_pos = self._find_word_start_across_wraps(self.point.line - 1)
                    target_line_content = self._unstyled_cache[target_line]
                    return self._absolute_line_to_position(target_line, x=wcswidth(target_line_content[:target_pos]))

                # 不是单词分割，正常返回
                return Position(wcswidth(line[:new_pos]), self.point.y, self.point.top_line)

        # 在行首，尝试跳到上一行
        if self.point.line <= 1:
            return self.point

        prev_line_raw = self.lines[self.point.line - 2]
        prev_line = self._unstyled_cache[self.point.line - 1]

        # 检查是否是单词分割（当前行首的单词延续）
        if self._is_word_split_at_wrap(self.point.line - 1):
            # 向上追溯找到单词真正的开始
            target_line, target_pos = self._find_word_start_across_wraps(self.point.line - 1)
            target_line_content = self._unstyled_cache[target_line]
            return self._absolute_line_to_position(target_line, x=wcswidth(target_line_content[:target_pos]))

        # 不是单词分割，正常跳到上一行的单词
        if WRAP_MARKER in prev_line_raw:
            # 有 wrap，在上一行查找单词
            pos = len(prev_line)
            while pos > 0 and prev_line[pos - 1].isspace():
                pos -= 1
            if pos > 0:
                pred = self._is_word_char if self._is_word_char(prev_line[pos - 1]) else self._is_word_separator
                pos = pos - len(''.join(takewhile(pred, reversed(prev_line[:pos]))))
            return self._absolute_line_to_position(self.point.line - 1, x=wcswidth(prev_line[:pos]))
        else:
            # 无 wrap，跳到上一行末尾
            return self._absolute_line_to_position(self.point.line - 1, x=wcswidth(prev_line))


    def word_right(self) -> Position:
        """移动到下一个单词开始（vim w 命令）

        支持跨越 wrap 的逻辑行延续和单词分割
        """
        line = self._unstyled_cache[self.point.line]
        pos = truncate_point_for_length(line, self.point.x)

        if pos < len(line):
            # 在当前行内移动
            pred = self._is_word_char if self._is_word_char(line[pos]) else self._is_word_separator
            new_pos = pos + len(''.join(takewhile(pred, line[pos:])))
            # 跳过空白字符，移动到下一个单词的开始
            while new_pos < len(line) and line[new_pos].isspace():
                new_pos += 1

            # 如果跳过空白后还在行内，返回该位置
            if new_pos < len(line):
                return Position(wcswidth(line[:new_pos]), self.point.y, self.point.top_line)

        # 到达行尾，检查是否还有下一行
        if self.point.line >= len(self.lines):
            return self.point

        # 跳到下一个视觉行
        next_line_num = self.point.line + 1
        next_line = self._unstyled_cache[next_line_num]

        # 如果有 wrap marker，检查是否是单词分割
        if self._has_wrap_marker(self.point.line):
            new_pos = 0
            if self._is_word_split_at_wrap(self.point.line):
                # 单词被分割，继续跳过下一行的单词字符（同一单词的延续）
                while new_pos < len(next_line) and self._is_word_char(next_line[new_pos]):
                    new_pos += 1

            # 跳过空白，找到下一个单词开始
            while new_pos < len(next_line) and next_line[new_pos].isspace():
                new_pos += 1

            return self._absolute_line_to_position(next_line_num, x=wcswidth(next_line[:new_pos]))

        # 无 wrap，返回下一行开始
        return self._absolute_line_to_position(next_line_num, x=0)

    def word_end(self) -> Position:
        """移动到当前/下一个单词的末尾（vim e 命令）

        行为：
        - 如果光标在单词中间或开始，移动到该单词末尾
        - 如果光标在单词末尾或空白，移动到下一个单词的末尾
        - 如果到达行尾，尝试跨行到下一行
        - 支持跨越 wrap 的逻辑行延续和单词分割
        """
        line = self._unstyled_cache[self.point.line]
        pos = truncate_point_for_length(line, self.point.x)

        # 辅助函数：处理跨行到下一行查找单词末尾
        def find_word_end_in_next_line() -> Optional[Position]:
            """跨行查找单词末尾，处理 wrap 和单词分割"""
            if self.point.line >= len(self.lines):
                return None

            next_line_num = self.point.line + 1
            next_line = self._unstyled_cache[next_line_num]

            if self._has_wrap_marker(self.point.line):
                new_pos = 0
                if self._is_word_split_at_wrap(self.point.line):
                    # 单词被分割，继续找到单词末尾
                    while new_pos < len(next_line) and self._is_word_char(next_line[new_pos]):
                        new_pos += 1
                    # new_pos 现在指向单词后的第一个字符，退一步指向单词末尾
                    if new_pos > 0:
                        new_pos -= 1
                    return self._absolute_line_to_position(next_line_num, x=wcswidth(next_line[:new_pos]))

                # 不是单词分割，跳过前导空白找下一个单词
                while new_pos < len(next_line) and next_line[new_pos].isspace():
                    new_pos += 1

                # 找到单词末尾
                if new_pos < len(next_line):
                    pred = self._is_word_char if self._is_word_char(next_line[new_pos]) else self._is_word_separator
                    while new_pos + 1 < len(next_line) and pred(next_line[new_pos + 1]):
                        new_pos += 1
                    return self._absolute_line_to_position(next_line_num, x=wcswidth(next_line[:new_pos]))

                # 下一行也是空的
                return self._absolute_line_to_position(next_line_num, x=0)

            # 无 wrap marker，使用原有逻辑查找下一个有单词的行
            result = self._find_next_word_end_from_line(self.point.line)
            if result is not None:
                target_line_idx, target_pos = result
                target_line = self._unstyled_cache[target_line_idx + 1]
                return self._absolute_line_to_position(target_line_idx + 1, x=wcswidth(target_line[:target_pos+1]))

            return None

        # 如果已经到达行尾，尝试跨行
        if pos >= len(line):
            result = find_word_end_in_next_line()
            return result if result is not None else self.point

        # 向前移动一个字符
        pos += 1

        # 如果超出行尾，尝试跨行
        if pos >= len(line):
            result = find_word_end_in_next_line()
            return result if result is not None else self.point

        # 如果在空白处，跳过所有空白
        while pos < len(line) and line[pos].isspace():
            pos += 1

        # 如果跳过空白后到达行尾，尝试跨行
        if pos >= len(line):
            result = find_word_end_in_next_line()
            if result is not None:
                return result
            # 没有找到，返回当前行末尾
            return Position(wcswidth(line), self.point.y, self.point.top_line)

        # 现在应该在单词字符或分隔符上，移动到该单词的末尾
        pred = self._is_word_char if self._is_word_char(line[pos]) else self._is_word_separator
        while pos + 1 < len(line) and pred(line[pos + 1]):
            pos += 1

        # 跨 wrap 处理：检查是否需要继续在下一行查找单词末尾
        current_abs_line = self.point.line
        current_y = self.point.y
        current_top = self.point.top_line

        while pos + 1 >= len(line):  # 到达行尾
            # 检查是否有 wrap marker
            if not self._has_wrap_marker(current_abs_line):
                break  # 没有 wrap，结束

            # 检查是否还有下一行
            if current_abs_line >= len(self.lines):
                break

            # 计算下一行的位置
            if current_y < self.screen_size.rows - 1:
                current_y += 1
            else:
                current_top += 1
            current_abs_line += 1

            # 获取下一行内容
            line = self._unstyled_cache[current_abs_line]

            # 检查下一行开头是否是同类字符（单词延续）
            if len(line) == 0 or not pred(line[0]):
                break  # 不是单词延续，结束

            # 继续在下一行查找单词末尾
            pos = 0
            while pos + 1 < len(line) and pred(line[pos + 1]):
                pos += 1

        return Position(wcswidth(line[:pos]), current_y, current_top)

    def _select(self, direction: DirectionStr,
                mark_type: Type[Region]) -> None:
        self._ensure_mark(mark_type)
        old_point = self.point
        self.point = (getattr(self, direction))()
        if self.point.top_line != old_point.top_line:
            self._redraw()
        else:
            self._redraw_lines(self.mark_type.lines_affected(
                self.mark, old_point, self.point))

    def move(self, direction: DirectionStr) -> None:
        self._select(direction, self.mode_types[self.mode])

    def select(self, region_type: RegionTypeStr,
               direction: DirectionStr) -> None:
        self._select(direction, self.region_types[region_type])

    def set_mode(self, mode: ModeTypeStr) -> None:
        self.mode = mode
        self._select('noop', self.mode_types[mode])

    def toggle_selection_end(self, same_line: str = '') -> None:
        """切换光标到选区的另一端（类似 vim 的 o/O 命令）

        Args:
            same_line: 非空字符串表示 O 命令（block 模式下只在同一行切换），
                      空字符串表示 o 命令（完全交换两个端点）
        """
        if self.mark is None:
            return

        old_point = self.point

        # O 命令在 block 模式下只交换列（x 坐标），保持行不变
        if same_line and self.mark_type == ColumnarRegion:
            # 只交换 x 坐标，保持 y 和 top_line 不变
            old_mark_x = self.mark.x
            self.mark = self.mark._replace(x=old_point.x)
            self.point = old_point._replace(x=old_mark_x)
        else:
            # o 命令或非 block 模式：完全交换 mark 和 point
            self.mark, self.point = self.point, self.mark

            # 确保新的 point 位置在屏幕可见范围内
            self.point = self.point.scrolled_towards(
                self.mark, self.screen_size.rows, len(self.lines))

        # 如果滚动位置改变，需要完全重绘
        if self.point.top_line != old_point.top_line:
            self._redraw()
        else:
            # 否则只重绘受影响的行
            self._redraw_lines(self.mark_type.lines_affected(
                self.mark, old_point, self.point))

    def _set_search_marker(self, query: str) -> None:
        """使用 Kitty marker 功能高亮搜索匹配项"""
        import subprocess
        # 使用 itext 类型进行大小写不敏感匹配
        # mark group 3 可以在 kitty.conf 中配置颜色
        subprocess.run(
            ['kitten', '@', 'create-marker', '--self=yes', 'itext', '3', query],
            capture_output=True,
            timeout=1,
            check=True
        )

    def _clear_search_marker(self) -> None:
        """移除 Kitty marker 高亮"""
        import subprocess
        subprocess.run(
            ['kitten', '@', 'remove-marker', '--self=yes'],
            capture_output=True,
            timeout=1,
            check=True
        )

    def search_start(self, direction: str) -> None:
        """进入搜索输入模式（vim / 或 ? 命令）"""
        # 清除之前的 marker（如果有）
        self._clear_search_marker()

        self.search_mode = direction
        self.search_query = ''
        self.search_matches = []
        self.current_match_index = -1
        self._prev_search_len = 1  # 初始长度为提示符 '/' 或 '?'
        self._redraw()

    def _perform_search(self) -> None:
        """执行搜索，找到所有匹配并跳转到第一个"""
        if not self.search_query:
            return

        self.search_matches = []

        # 使用正则引擎进行大小写不敏感搜索
        pattern = re.compile(re.escape(self.search_query), re.IGNORECASE)

        # 使用预处理的缓存（避免重复调用 unstyled）
        for line_num in range(1, len(self.lines) + 1):
            plain = self._unstyled_cache[line_num]

            # 使用 finditer 找到所有匹配（比循环 find 更高效）
            for match in pattern.finditer(plain):
                col = wcswidth(plain[:match.start()])
                match_pos = Position(col, 0, line_num)
                self.search_matches.append(match_pos)

        if not self.search_matches:
            # 没有找到匹配，退出搜索模式
            self.search_mode = None
            self._redraw()
            return

        # 根据搜索方向和当前光标位置，找到第一个匹配
        if self.search_mode == 'forward':
            # 向前搜索：找到第一个在当前光标之后的匹配
            for idx, match in enumerate(self.search_matches):
                if match.line > self.point.line or \
                   (match.line == self.point.line and match.x > self.point.x):
                    self.current_match_index = idx
                    break
            else:
                # 没有找到后面的，从头开始
                self.current_match_index = 0
        else:  # backward
            # 向后搜索：找到最后一个在当前光标之前的匹配
            for idx in range(len(self.search_matches) - 1, -1, -1):
                match = self.search_matches[idx]
                if match.line < self.point.line or \
                   (match.line == self.point.line and match.x < self.point.x):
                    self.current_match_index = idx
                    break
            else:
                # 没有找到前面的，从尾开始
                self.current_match_index = len(self.search_matches) - 1

        # 退出搜索输入模式（在跳转之前）
        self.search_mode = None

        # 使用 Kitty marker 高亮所有匹配项
        self._set_search_marker(self.search_query)

        # 跳转到匹配位置（会调用 _redraw()）
        self._jump_to_match(self.current_match_index)

    def _jump_to_match(self, match_index: int) -> None:
        """跳转到指定的匹配位置"""
        if not self.search_matches or match_index < 0 or match_index >= len(self.search_matches):
            return

        match = self.search_matches[match_index]
        abs_line = match.line  # 绝对行号（1-based）

        # 计算新的屏幕坐标
        # 尝试将匹配位置显示在屏幕中间
        rows = self.screen_size.rows
        target_y = rows // 2

        # 计算 top_line 使得 abs_line 显示在屏幕中间
        target_top_line = abs_line - target_y

        # 边界检查
        target_top_line = max(1, target_top_line)
        target_top_line = min(len(self.lines) - rows + 1, target_top_line)

        # 计算新的 y 坐标
        new_y = abs_line - target_top_line

        # 更新 point 位置
        self.point = Position(match.x, new_y, target_top_line)

        # 完全重绘屏幕
        self._redraw()

    def search_next(self) -> None:
        """跳转到下一个匹配（vim n 命令）"""
        if not self.search_matches:
            return

        # 循环到下一个匹配
        self.current_match_index = (self.current_match_index + 1) % len(self.search_matches)
        self._jump_to_match(self.current_match_index)

    def search_prev(self) -> None:
        """跳转到上一个匹配（vim N 命令）"""
        if not self.search_matches:
            return

        # 循环到上一个匹配
        self.current_match_index = (self.current_match_index - 1) % len(self.search_matches)
        self._jump_to_match(self.current_match_index)

    def start_yank(self) -> None:
        """进入 operator-pending 状态，等待 motion 输入（vim y 命令）

        在 normal 模式下：设置 pending_operator 等待 motion
        在 visual/line/block 模式下：复制选中内容并退出（等同于 confirm）
        """
        if self.mode == 'normal':
            # normal 模式：进入 operator-pending 状态
            self.pending_operator = 'y'
        else:
            # visual/line/block 模式：直接复制并退出
            self.confirm()

    def _execute_yank_motion(self, target_position: Position) -> None:
        """通用的 yank 执行方法（operator + motion 模式）

        实现方式：
        1. 保存当前位置作为 mark
        2. 设置 mark_type 为 StreamRegion
        3. 更新 point 到目标位置
        4. 清除 pending_operator 状态
        5. 调用 confirm() 复制并退出

        Args:
            target_position: motion 方法返回的目标位置
        """
        # 保存当前位置作为 mark
        self.mark = self.point
        self.mark_type = StreamRegion

        # 更新 point 到目标位置
        self.point = target_position

        # 清除 pending operator 状态
        self.pending_operator = None

        # 复制并退出
        self.confirm()

    def yank_line(self) -> None:
        """复制当前行并退出（vim yy 命令）

        实现方式：
        1. 临时设置 mark 和 mark_type 为 LineRegion
        2. 调用 confirm() 复制并退出
        """
        # 保存当前状态
        old_mark = self.mark
        old_mark_type = self.mark_type

        # 设置 mark 为当前行的开始位置（LineRegion 会自动选择整行）
        self.mark = Position(0, self.point.y, self.point.top_line)
        self.mark_type = LineRegion

        # 清除 pending operator 状态
        self.pending_operator = None

        # 调用 confirm() 复制并退出
        self.confirm()

    def yank_to_eol(self) -> None:
        """复制从光标位置到行尾并退出（Neovim Y 命令，等同于 y$）

        使用通用的 _execute_yank_motion() 方法，复用 y$ 的实现
        """
        self._execute_yank_motion(self.last_nonwhite())

    def _expand_line_selection_for_wrap(self, start_line: AbsoluteLine,
                                        end_line: AbsoluteLine) -> Tuple[AbsoluteLine, AbsoluteLine]:
        """展开选择范围以包含完整的逻辑行（处理 line-wrapping）

        检测被 wrap markers 分割的逻辑行，并展开选择范围。

        Args:
            start_line: 选择起始行号（1-based）
            end_line: 选择结束行号（1-based）

        Returns:
            (expanded_start, expanded_end): 展开后的行号范围
        """
        # 使用辅助方法简化实现
        expanded_start = self._find_logical_line_start(start_line)
        expanded_end = self._find_logical_line_end(end_line)

        return expanded_start, expanded_end

    def confirm(self, *args: Any) -> None:
        # 检查是否有实际选择
        # NoRegion 表示 normal 模式，没有选择任何内容
        if self.mark_type == NoRegion or self.mark is None:
            # 没有选择内容，直接退出而不复制（避免清空剪贴板）
            self.quit()
            return

        start, end = self._start_end()

        # 对于 LineRegion，展开选择范围以包含完整的逻辑行（处理 line-wrapping）
        if self.mark_type == LineRegion:
            start_line, end_line = self._expand_line_selection_for_wrap(
                start.line, end.line)
            # 创建临时 Position，调整 y 使得 line 属性 (y + top_line) 等于展开后的行号
            expanded_start = Position(start.x, start_line - start.top_line, start.top_line)
            expanded_end = Position(end.x, end_line - end.top_line, end.top_line)
        else:
            start_line, end_line = start.line, end.line
            expanded_start, expanded_end = start, end

        # 构建带 wrap 标记信息的列表
        lines_with_markers = []
        for line in range(start_line, end_line + 1):
            raw_line = self.lines[line - 1]
            has_wrap_marker = WRAP_MARKER in raw_line  # 在 unstyled() 之前检测
            plain = self._unstyled_cache[line]
            start_x, end_x = self.mark_type.selection_in_line(
                line, expanded_start, expanded_end, self._width_cache[line])
            if start_x is not None and end_x is not None:
                line_slice, _half = string_slice(plain, start_x, end_x)
                lines_with_markers.append((line_slice.rstrip(), has_wrap_marker))

        # 智能拼接：根据 wrap 标记决定是否插入换行符
        # 有 wrap 标记的行后面不加换行（继续拼接），无 wrap 标记的行后面加换行（逻辑行结束）
        parts = []
        for i, (text, has_wrap) in enumerate(lines_with_markers):
            parts.append(text)
            # 不是最后一行 且 当前行无 wrap 标记 → 加换行（逻辑行结束）
            if i < len(lines_with_markers) - 1 and not has_wrap:
                parts.append('\n')
            # 如果 has_wrap=True，什么都不加，下一行会直接拼接

        # 保留中间空行，清理首尾空行
        copied_text = ''.join(parts).strip()
        # 移除 wrap markers 以还原完整的逻辑行
        copied_text = copied_text.replace(WRAP_MARKER + '\n', '').replace(WRAP_MARKER, '')
        # 额外清理所有可能遗漏的 shell integration 和 ANSI 序列
        copied_text = re.sub(r'\x1b\]133[^\x07\n]*\x07?', '', copied_text)
        copied_text = re.sub(r'\x1b\][0-9]+[^\x07\n]*\x07?', '', copied_text)
        copied_text = re.sub(r'\x1b\[[0-9;:]*m', '', copied_text)  # 清理遗漏的 SGR
        self.result = {'copy': copied_text}
        self.quit_loop(0)


def main(args: List[str]) -> Optional['ResultDict']:

    def ospec() -> str:
        return '''
--copy-to
dest=copy_to
type=str
Copy to: 'clipboard' or 'primary'/selection or 'secondary' buffer


--cursor-x
dest=x
type=int
(Internal) Starting cursor column, 0-based.


--cursor-y
dest=y
type=int
(Internal) Starting cursor line, 0-based.


--top-line
dest=top_line
type=int
(Internal) Window scroll offset, 1-based.


--title
(Internal)'''

    try:
        args, _rest = parse_args(args[1:], ospec)
        tty = open(os.ctermid())
        lines = (sys.stdin.buffer.read().decode('utf-8')
                 .split('\n')[:-1])  # last line ends with \n, too
        sys.stdin = tty
        opts = load_config()
        handler = GrabHandler(args, opts, lines)
        loop = Loop()
        loop.loop(handler)
        if loop.return_code == 0 and 'copy' in handler.result:
            sys.stdout.buffer.write(b''.join((b'\x1b]52;', handler.copy_to, b';',
                                              b64encode(handler.result['copy'].encode('utf-8')),
                                              b'\x1b\\')))
        return {}
    except Exception as e:
        from kittens.tui.loop import debug
        from traceback import format_exc
        debug(format_exc())
        raise
