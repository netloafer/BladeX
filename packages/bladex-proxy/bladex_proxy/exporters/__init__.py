"""具体 connector 实现（Beta T14 Obsidian / T15 PostgreSQL）。

工厂注册走 export_sync.register_exporter；build_exporters 按 type 惰性 import
（未用到的 connector 不引其依赖——postgres 需要 psycopg，obsidian 纯 stdlib）。
"""
