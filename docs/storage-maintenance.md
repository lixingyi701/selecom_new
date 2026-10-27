# 服务器存储维护

整理日期：2026-10-27。项目原工作区的 Git HEAD 为 `7e0f4da`，含既有暂存和未暂存修改；整理前保存代码/config 快照、两份补丁、原 Git index 及文件哈希。原工作区不执行 reset、stash、clean 或强制 checkout。

## 容量核查

按目录实际用量的近似值：项目约 134 GiB，数据约 78 GiB，基础模型约 43 GiB，预训练 checkpoint 约 5.6 GiB，results 约 4.2 GiB，首轮研究产物约 3.1 GiB。数据占用包含 NQ train/dev/eval Parquet、Stage1/Stage2 原始分片及已转换 JSONL；它们属于输入资产。

核查时系统 MemAvailable 约 114 GiB，磁盘剩余约 437 GiB。Linux buff/cache 约 109 GiB 是可回收系统页缓存，不能当作进程已占满内存。项目未发现大型 Arrow 或 tokenization 缓存；候选缓存合计约 138 KiB，其中 Python 字节码 16 文件、90,949 bytes。下载元数据/锁文件约 49 KiB，保留用于来源追溯。

## 目录与保留策略

| 路径类别 | 策略 |
| --- | --- |
| `baselineModel/`、`checkpoint/`、`data/` | 原始资产，保留，不移动训练依赖路径 |
| `results/`、`log/`、`research/repair-20261027/` | 保留有效权重、原始逐题结果、日志、配置与哈希 |
| `research/maintenance-20261027/` | 私有备份、维护前后清单、发布验证记录 |
| 干净发布工作副本 | 独立于原脏工作区，用于阅读新文档及复现代码；通过环境变量指向已有资产 |
| `__pycache__/*.pyc` | 仅在源文件存在、确认无占用且路径在项目内时逐文件清理 |
| `.cache/`、`*.lock` | 不按名字批量删除；下载 metadata 与共享模型缓存保留 |

具体清理结果见 [维护摘要](../experiments/repair_20261027/results/storage-cleanup.json)。本轮不运行 GPU 作业，不触及 GPU1–3，也不停止其他用户进程。未来仍保留至少 24 GiB 系统可用内存。
