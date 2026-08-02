from typing import Any, Callable, Iterable, Sequence, Tuple

from kitty.conf.utils import (KeyFuncWrapper, KittensKeyDefinition, choices,
                              parse_kittens_key)

FuncArgsType = Tuple[str, Sequence[Any]]

func_with_args = KeyFuncWrapper[FuncArgsType]()


def parse_map(val: str) -> Iterable[KittensKeyDefinition]:
    x = parse_kittens_key(val, func_with_args.args_funcs)
    if x is not None:
        yield x


parse_region_type = choices('stream', 'columnar', 'line')

parse_scroll_direction = choices('up', 'down')

parse_mode = choices('normal', 'visual', 'line', 'block')

_direction = choices('left', 'right', 'up', 'down',
                     'page up', 'page down',
                     'first', 'first nonwhite',
                     'last nonwhite', 'last',
                     'top', 'bottom',
                     'word left', 'word right', 'word end',
                     'line up', 'line down')


def parse_direction(direction: str) -> str:
    return _direction(direction).replace(' ', '_')


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


_search_direction = choices('forward', 'backward')


@func_with_args('search_start')
def search_start(func: Callable, direction: str) -> Tuple[Callable, Tuple[str,]]:
    return func, (_search_direction(direction),)

# 无参数的 action（search_next、start_yank 等）不需要注册：
# kitty 的 parse_kittens_func_args 对无参 action 直接返回 KeyAction(func, ())
