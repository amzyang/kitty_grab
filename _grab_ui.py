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
    s = re.sub(r'\x1b\[[0-9;:]*m', '', s)
    s = re.sub(r'\x1b\](?:[^\x07\x1b]+|\x1b[^\\])*(?:\x1b\\|\x07)', '', s)
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

    def _start_end(self) -> Tuple[Position, Position]:
        start, end = sorted([self.point, self.mark or self.point])
        return self.mark_type.adjust(start, end)

    def _draw_line(self, current_line: AbsoluteLine) -> None:
        y = current_line - self.point.top_line  # type: ScreenLine
        line = self.lines[current_line - 1]
        clear_eol = '\x1b[m\x1b[K'
        sgr0 = '\x1b[m'

        plain = unstyled(line)
        selection_sgr = '\x1b[38{};48{}m'.format(
            color_as_sgr(self.opts.selection_foreground),
            color_as_sgr(self.opts.selection_background))
        start, end = self._start_end()

        # anti-flicker optimization
        if self.mark_type.line_inside_region(current_line, start, end):
            self.cmd.set_cursor_position(0, y)
            self.print('{}{}'.format(selection_sgr, plain),
                       end=clear_eol)
            return

        self.cmd.set_cursor_position(0, y)
        self.print('{}{}'.format(sgr0, line), end=clear_eol)

        if self.mark_type.line_outside_region(current_line, start, end):
            return

        start_x, end_x = self.mark_type.selection_in_line(
            current_line, start, end, wcswidth(plain))
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

    def on_key_event(self, key_event: KeyEvent, in_bracketed_paste: bool = False) -> None:
        # 如果处于搜索输入模式，特殊处理键盘事件
        if self.search_mode is not None:
            self._handle_search_input(key_event)
            return

        # 如果处于 operator-pending 模式，特殊处理键盘事件
        if self.pending_operator is not None:
            self._handle_pending_operator(key_event)
            return

        action = self.shortcut_action(key_event)
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
        return self.point.moved(dx=-1) if self.point.x > 0 else self.point

    def right(self) -> Position:
        return (self.point.moved(dx=1)
                if self.point.x + 1 < self.screen_size.cols
                else self.point)

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
        return Position(0, self.point.y, self.point.top_line)

    def first_nonwhite(self) -> Position:
        line = unstyled(self.lines[self.point.line - 1])
        prefix = ''.join(takewhile(str.isspace, line))
        return Position(wcswidth(prefix), self.point.y, self.point.top_line)

    def last_nonwhite(self) -> Position:
        line = unstyled(self.lines[self.point.line - 1])
        suffix = ''.join(takewhile(str.isspace, reversed(line)))
        return Position(wcswidth(line[:len(line) - len(suffix)]),
                        self.point.y, self.point.top_line)

    def last(self) -> Position:
        return Position(self.screen_size.cols,
                        self.point.y, self.point.top_line)

    def top(self) -> Position:
        return Position(0, 0, 1)

    def bottom(self) -> Position:
        x = wcswidth(unstyled(self.lines[-1]))
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
            line = unstyled(self.lines[line_idx])
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
        if self.point.x > 0:
            line = unstyled(self.lines[self.point.line - 1])
            pos = truncate_point_for_length(line, self.point.x)
            if pos > 0:
                # Step 1: 先往回跳过所有空白字符
                while pos > 0 and line[pos - 1].isspace():
                    pos -= 1

                # Step 2: 如果还有字符，往回跳过整个单词/分隔符
                if pos > 0:
                    pred = (self._is_word_char if self._is_word_char(line[pos - 1])
                            else self._is_word_separator)
                    new_pos = pos - len(''.join(takewhile(pred, reversed(line[:pos]))))
                    return Position(wcswidth(line[:new_pos]),
                                    self.point.y, self.point.top_line)
        if self.point.y > 0:
            return Position(wcswidth(unstyled(self.lines[self.point.line - 2])),
                            self.point.y - 1, self.point.top_line)
        if self.point.top_line > 1:
            return Position(wcswidth(unstyled(self.lines[self.point.line - 2])),
                            self.point.y, self.point.top_line - 1)
        return self.point

    def word_right(self) -> Position:
        line = unstyled(self.lines[self.point.line - 1])
        pos = truncate_point_for_length(line, self.point.x)
        if pos < len(line):
            pred = (self._is_word_char if self._is_word_char(line[pos])
                    else self._is_word_separator)
            new_pos = pos + len(''.join(takewhile(pred, line[pos:])))
            # 跳过空白字符，移动到下一个单词的开始
            while new_pos < len(line) and line[new_pos].isspace():
                new_pos += 1
            return Position(wcswidth(line[:new_pos]),
                            self.point.y, self.point.top_line)
        if self.point.y < self.screen_size.rows - 1:
            return Position(0, self.point.y + 1, self.point.top_line)
        if self.point.top_line + self.point.y < len(self.lines):
            return Position(0, self.point.y, self.point.top_line + 1)
        return self.point

    def word_end(self) -> Position:
        """移动到当前/下一个单词的末尾（vim e 命令）

        行为：
        - 如果光标在单词中间或开始，移动到该单词末尾
        - 如果光标在单词末尾或空白，移动到下一个单词的末尾
        - 如果到达行尾，尝试跨行到下一行
        """
        line = unstyled(self.lines[self.point.line - 1])
        pos = truncate_point_for_length(line, self.point.x)

        # 如果已经到达行尾，尝试跨行
        if pos >= len(line):
            # 使用辅助方法查找下一个有单词的行
            result = self._find_next_word_end_from_line(self.point.line)  # 从下一行开始（0-based index）

            if result is not None:
                target_line_idx, target_pos = result
                target_abs_line = target_line_idx + 1  # 转为 1-based
                line_offset = target_abs_line - self.point.line

                # 计算新的 y 和 top_line
                new_y = self.point.y + line_offset
                new_top_line = self.point.top_line

                # 如果新的 y 超出屏幕底部，需要滚动
                while new_y >= self.screen_size.rows:
                    new_y -= 1
                    new_top_line += 1

                # 确保 top_line 不超过最大值
                max_top_line = max(1, len(self.lines) - self.screen_size.rows + 1)
                if new_top_line > max_top_line:
                    new_top_line = max_top_line
                    new_y = target_abs_line - new_top_line

                target_line = unstyled(self.lines[target_line_idx])
                return Position(wcswidth(target_line[:target_pos]), new_y, new_top_line)

            # 没有找到，返回当前位置
            return self.point

        # 向前移动一个字符
        pos += 1

        # 如果超出行尾，尝试跨行
        if pos >= len(line):
            # 使用辅助方法查找下一个有单词的行
            result = self._find_next_word_end_from_line(self.point.line)  # 从下一行开始（0-based index）

            if result is not None:
                target_line_idx, target_pos = result
                target_abs_line = target_line_idx + 1  # 转为 1-based
                line_offset = target_abs_line - self.point.line

                # 计算新的 y 和 top_line
                new_y = self.point.y + line_offset
                new_top_line = self.point.top_line

                # 如果新的 y 超出屏幕底部，需要滚动
                while new_y >= self.screen_size.rows:
                    new_y -= 1
                    new_top_line += 1

                # 确保 top_line 不超过最大值
                max_top_line = max(1, len(self.lines) - self.screen_size.rows + 1)
                if new_top_line > max_top_line:
                    new_top_line = max_top_line
                    new_y = target_abs_line - new_top_line

                target_line = unstyled(self.lines[target_line_idx])
                return Position(wcswidth(target_line[:target_pos]), new_y, new_top_line)

            # 没有找到，返回当前位置
            return self.point

        # 如果在空白处，跳过所有空白
        while pos < len(line) and line[pos].isspace():
            pos += 1

        # 如果跳过空白后到达行尾，尝试跨行
        if pos >= len(line):
            # 使用辅助方法查找下一个有单词的行
            result = self._find_next_word_end_from_line(self.point.line)  # 从下一行开始（0-based index）

            if result is not None:
                target_line_idx, target_pos = result
                target_abs_line = target_line_idx + 1  # 转为 1-based
                line_offset = target_abs_line - self.point.line

                # 计算新的 y 和 top_line
                new_y = self.point.y + line_offset
                new_top_line = self.point.top_line

                # 如果新的 y 超出屏幕底部，需要滚动
                while new_y >= self.screen_size.rows:
                    new_y -= 1
                    new_top_line += 1

                # 确保 top_line 不超过最大值
                max_top_line = max(1, len(self.lines) - self.screen_size.rows + 1)
                if new_top_line > max_top_line:
                    new_top_line = max_top_line
                    new_y = target_abs_line - new_top_line

                target_line = unstyled(self.lines[target_line_idx])
                return Position(wcswidth(target_line[:target_pos]), new_y, new_top_line)

            # 没有找到，返回当前行末尾
            return Position(wcswidth(line), self.point.y, self.point.top_line)

        # 现在应该在单词字符或分隔符上，移动到该单词的末尾
        pred = (self._is_word_char if self._is_word_char(line[pos])
                else self._is_word_separator)
        while pos + 1 < len(line) and pred(line[pos + 1]):
            pos += 1

        return Position(wcswidth(line[:pos]), self.point.y, self.point.top_line)

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
        query_lower = self.search_query.lower()

        # 遍历所有行查找匹配
        for line_idx, line in enumerate(self.lines):
            line_num = line_idx + 1  # 行号从1开始
            plain = unstyled(line)
            plain_lower = plain.lower()

            # 在当前行中查找所有匹配位置
            start_pos = 0
            while True:
                pos = plain_lower.find(query_lower, start_pos)
                if pos == -1:
                    break

                # 计算显示宽度（处理宽字符）
                col = wcswidth(plain[:pos])

                # 计算 Position（需要转换为屏幕坐标）
                # 暂时假设使用当前 top_line，后面会调整
                match_pos = Position(col, 0, line_num)
                self.search_matches.append(match_pos)

                start_pos = pos + 1

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

    def confirm(self, *args: Any) -> None:
        start, end = self._start_end()
        lines_list = [
            line_slice.rstrip()
            for line in range(start.line, end.line + 1)
            for plain in [unstyled(self.lines[line - 1])]
            for start_x, end_x in [self.mark_type.selection_in_line(
                line, start, end, wcswidth(plain))]
            if start_x is not None and end_x is not None
            for line_slice, _half in [string_slice(plain, start_x, end_x)]
        ]
        # 保留中间空行，清理首尾空行
        copied_text = '\n'.join(lines_list).strip()
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
