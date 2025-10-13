# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

kitty_grab 是一个 Kitty 终端模拟器的 kitten（插件），用于实现键盘驱动的文本选择和复制功能。

**安装方式：**
```bash
cd ~/.config/kitty
git clone https://github.com/yurikhan/kitty_grab.git
```

在 `kitty.conf` 中绑定快捷键：
```
map Alt+Insert kitten kitty_grab/grab.py
```

**最低要求：** Kitty ≥ 0.21.2

**配置文件：** 复制 `grab-vim.conf.example` 到 `~/.config/kitty/grab.conf`（无需重启 Kitty，下次激活时生效）

## 核心架构

### 双阶段入口机制

1. **grab.py** - 第一阶段入口
   - 使用 `@result_handler(no_ui=True)` 装饰器
   - 从目标窗口捕获屏幕内容（包括历史滚动缓冲区）
   - 计算光标位置和 `top_line` 偏移量
   - 启动第二阶段 UI (`_grab_ui.py`)

2. **_grab_ui.py** - 主 UI 处理器
   - 继承 `kittens.tui.handler.Handler`
   - 实现完整的文本选择、光标移动、区域选择逻辑
   - 通过 OSC 52 转义序列将选中文本复制到剪贴板

### 坐标系统：Position 类

Position 是核心数据结构，表示屏幕上的一个单元格位置：

```python
Position(x, y, top_line)
```

- `x`: 屏幕列坐标（从0开始，从左到右）
- `y`: 屏幕行坐标（从0开始，从上到下）
- `top_line`: 滚动缓冲区的绝对行号（从1开始）

**关键方法：**
- `moved(dx, dy, dtop)` - 相对移动
- `scrolled(dtop)` - 滚动视图而不改变绝对位置
- `scrolled_towards(other, rows, lines)` - 智能滚动以同时显示两个位置

### 区域选择类型

三种区域类型实现不同的选择行为：

1. **NoRegion** - 无选择状态（normal 模式）
2. **StreamRegion** - 流式选择（visual 模式）
   - 包含起止位置之间的所有字符
   - 跨行时包含整行
3. **ColumnarRegion** - 列式选择（block 模式）
   - 矩形区域选择
   - 列边界由起止位置的 x 坐标最小值和最大值确定

每个 Region 类定义：
- `selection_in_line()` - 计算某行内的选择范围
- `lines_affected()` - 优化重绘，只更新受影响的行
- `page_up/page_down()` - 翻页时保持选区可见性

### 模态编辑

支持类 Vim 的模态编辑：

- **normal** 模式：无选择，光标移动
- **visual** 模式：流式选择
- **block** 模式：列式选择

快捷键：
- `v` 进入 visual 模式
- `Ctrl+v` 进入 block 模式
- `Ctrl+[` 或 `Escape` 返回 normal 模式
- `o` 切换光标到选区的对角（所有模式）
- `O` 在 block 模式下切换到同一行的另一个角，在 visual 模式下同 `o`

## 配置系统

配置使用 Kitty 的配置框架，涉及四个文件：

- **kitten_options_definition.py** - 定义选项和默认快捷键（手动编辑）
- **kitten_options_types.py** - 类型定义和默认快捷键映射（理论上自动生成，但实际需要手动编辑）
- **kitten_options_parse.py** - 解析配置项（自动生成）
- **kitten_options_utils.py** - 参数处理函数（手动编辑）

### 添加新快捷键的完整流程

**重要：** 虽然 `kitten_options_types.py` 头部标记为自动生成，但由于缺少 `gen-config.py` 工具，实际需要手动同步更新。

1. **定义快捷键** (`kitten_options_definition.py`)
   ```python
   map('Description', 'action_name key_name action_name arg1 arg2')
   ```

2. **添加参数处理函数** (`kitten_options_utils.py`)
   ```python
   @func_with_args("action_name")
   def action_name(func: Callable, arg1: str) -> Tuple[Callable, Tuple[str,]]:
       return func, (arg1,)
   ```

3. **手动更新类型文件** (`kitten_options_types.py`)
   在 `defaults.map` 列表末尾添加：
   ```python
   # action_name
   (ParsedShortcut(mods=0, key_name='key_name'), KeyAction('action_name', ('arg1',))),  # noqa
   ```

   修饰符常量：`mods=0`（无）、`mods=1`（Shift）、`mods=2`（Alt）、`mods=4`（Ctrl）

4. **实现处理器方法** (`_grab_ui.py` 中的 `GrabHandler` 类)
   ```python
   def action_name(self, arg1: str) -> None:
       # 实现逻辑
   ```

### 示例：toggle_selection_end

参考 `toggle_selection_end` 的实现（_grab_ui.py:653，kitten_options_utils.py:74，kitten_options_types.py:178）

配置文件位置：
- 系统级：`/etc/xdg/kitty/grab.conf`
- 用户级：`~/.config/kitty/grab.conf`

## OSC 52 剪贴板集成

使用 Operating System Command 52 将文本复制到剪贴板：

```python
# OSC 52 格式: \x1b]52;<target>;<base64_data>\x1b\\
# target: c=clipboard, p=primary, s=secondary
```

通过 `--copy-to` 参数选择目标缓冲区。

## Kitty 版本兼容性

**最低要求：** Kitty ≥ 0.21.2

**关键兼容性处理：**

```python
try:
    # Kitty v0.42+
    from kitty.typing_compat import BossType
except ModuleNotFoundError:
    # 旧版本 fallback
    from kitty.typing import BossType
```

## 文本处理

### 去除样式

`unstyled(s)` 函数处理终端文本：
- 移除 SGR (Select Graphic Rendition) 序列：`\x1b[...m`
- 移除 OSC 序列：`\x1b]...\x07` 或 `\x1b]...\x1b\\`
- 扩展制表符为空格（`expandtabs()`，默认 tab stop = 8）
  - 确保 `wcswidth()` 能正确计算显示宽度
  - 避免制表符导致的位置计算错位

### 字符串切片

`string_slice(s, start_x, end_x)` 处理宽字符（如中文）：
- 使用 `truncate_point_for_length()` 计算正确的字节偏移
- 返回切片和是否在宽字符中间开始的标志

### 单词边界识别

基于 Unicode 类别和配置的 `select_by_word_characters`：
- 字母和数字 (Unicode 类别 'L' 和 'N')
- 用户配置的特殊字符（默认从主 Kitty 配置继承）

### 复制时的文本清理

`confirm()` 方法在复制选中文本时会自动：
- 对每一行调用 `.rstrip()` 只去除行尾空白，保留行首缩进
- 保留中间的空行（代码块分隔），只清理首尾空行
- 使用 `'\n'.join(lines_list).strip()` 实现：先拼接所有行，再去除整体首尾空白

## 开发注意事项

1. **配置文件同步**：虽然 `kitten_options_types.py` 标记为自动生成，但实际需要手动更新以保持与 `kitten_options_definition.py` 同步（参见上文"添加新快捷键的完整流程"）

2. **性能优化**：`lines_affected()` 方法决定需要重绘的行，是性能关键
   - 只重绘受影响的行可以显著提高响应速度
   - 滚动时需要完全重绘（`_redraw()`）

3. **坐标计算注意事项**：
   - `top_line` 从 **1** 开始（绝对行号）
   - `x` 和 `y` 从 **0** 开始（屏幕相对坐标）
   - `Position.line` 属性返回绝对行号：`y + top_line`

4. **滚动边界检查**：
   - 最小 `top_line`: 1
   - 最大 `top_line`: `len(lines) - rows + 1`

5. **测试环境**：
   - 无法直接使用 `python3` 运行，因为 `kitty` 模块不在标准 Python 路径中
   - 必须通过 `kitten` 命令在 Kitty 终端中测试
   - 无自动化测试框架，需要手动测试所有功能

6. **参数类型**：
   - 快捷键系统传递的所有参数都是字符串类型
   - 布尔值需要检查字符串是否非空（`if same_line:` 而非 `if same_line == True:`）

## 配置示例

两种配置风格：

1. **标准模式** (`grab.conf.example`) - 使用箭头键和修饰键
2. **Vim 模式** (`grab-vim.conf.example`) - 使用 hjkl 和 Vim 风格快捷键

用户应将示例文件复制到 `~/.config/kitty/grab.conf` 并根据需要修改。
