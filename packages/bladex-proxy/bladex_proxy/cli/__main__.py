"""`python -m bladex_proxy.cli` 入口（cli 成子包后 `-m` 需要 `__main__.py`；`bladex` console script 仍走 `cli:main`）。"""

from bladex_proxy.cli import main

raise SystemExit(main())
