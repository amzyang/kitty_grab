#!/bin/sh
# 运行全部自动化测试。
# 必须经 `kitty +runpy` 执行：kitty.* 模块只在 kitty 自带 Python 环境中可导入。
# 每个 test_*.py 是独立脚本（显式 check 断言，非 pytest——kitty 以 -O 运行，
# assert 会被剥离），任一文件抛异常即整体失败。
set -e
dir=$(cd "$(dirname "$0")" && pwd)

exec kitty +runpy "
import glob
import runpy
import sys

failed = []
for f in sorted(glob.glob('$dir/test_*.py')):
    name = f.rsplit('/', 1)[-1]
    print('==> ' + name)
    try:
        runpy.run_path(f, run_name='__main__')
    except Exception as e:
        failed.append(name)
        print('FAILED {}: {}'.format(name, e))
if failed:
    print('FAILED: ' + ', '.join(failed))
    sys.exit(1)
print('ALL TESTS PASSED')
"
