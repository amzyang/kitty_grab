from typing import Any, Callable, Iterable, Sequence, Tuple

from kitty.conf.utils import KittensKeyDefinition, parse_kittens_key

FuncArgsType = Tuple[str, Sequence[Any]]

try:
    from kitty.conf.utils import KeyFuncWrapper
    func_with_args = KeyFuncWrapper[FuncArgsType]()
except ImportError:
    from kitty.conf.utils import key_func
    func_with_args, args_funcs = key_func()
    func_with_args.args_funcs = args_funcs



def parse_map(val: str) -> Iterable[KittensKeyDefinition]:
    x = parse_kittens_key(val, func_with_args.args_funcs)
    if x is not None:
        yield x


def _choice(value: str, *allowed: str) -> str:
    result = value.lower()
    assert result in allowed
    return result


def parse_region_type(region_type: str) -> str:
    return _choice(region_type, 'stream', 'columnar', 'line')


def parse_direction(direction: str) -> str:
    return _choice(direction, 'left', 'right', 'up', 'down',
                   'page up', 'page down',
                   'first', 'first nonwhite',
                   'last nonwhite', 'last',
                   'top', 'bottom',
                   'word left', 'word right', 'word end',
                   'line up', 'line down').replace(' ', '_')


def parse_scroll_direction(direction: str) -> str:
    return _choice(direction, 'up', 'down')


def parse_mode(mode: str) -> str:
    return _choice(mode, 'normal', 'visual', 'line', 'block')


@func_with_args('move')
def move(func: Callable, direction: str) -> Tuple[Callable, str]:
    return func, parse_direction(direction)


@func_with_args('scroll')
def scroll(func: Callable, direction: str) -> Tuple[Callable, str]:
    return func, parse_scroll_direction(direction)


@func_with_args('select')
def select(func: Callable, args: str) -> Tuple[Callable, Tuple[str, str]]:
    region_type, direction = args.split(' ', 1)
    return func, (parse_region_type(region_type),
                  parse_direction(direction))


@func_with_args("set_mode")
def set_mode(func: Callable, mode: str) -> Tuple[Callable, str]:
    return func, parse_mode(mode)


@func_with_args("toggle_selection_end")
def toggle_selection_end(func: Callable, same_line: str = '') -> Tuple[Callable, Tuple[str,]]:
    return func, (same_line,)


@func_with_args('search_start')
def search_start(func: Callable, direction: str) -> Tuple[Callable, Tuple[str,]]:
    return func, (_choice(direction, 'forward', 'backward'),)

# 无参数的 action（search_next、start_yank 等）不需要注册：
# kitty 的 parse_kittens_func_args 对无参 action 直接返回 KeyAction(func, ())
