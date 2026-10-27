# 2026-10-27 受控实验包

已完成：历史 checkpoint 审计、512 题检查、4,462 题统一重评、G0/G2 两个前端对照、generator-only 对照、512 题文档扰动。三个训练组均为 5,000 样本、79 步。聚合值和题目配对 bootstrap 区间见 [results/metrics.json](results/metrics.json)。

`scripts/` 为已执行脚本的可移植版本，`scripts/code/` 是实验实际使用的五个冻结模型/模板依赖，不导入仓库后续改动的 `model/` 或 `util/`。修改仅包含路径配置、来源文件定位及重跑覆盖保护，见 [source-manifest.json](source-manifest.json)。这些发布文件的 SHA 与原运行 manifest 中的 SHA 可以不同，原运行哈希仍保留。

| 入口 | 作用 |
| --- | --- |
| `run.sh check` | CPU/标准库资产、源码与路径检查；不加载模型 |
| `run.sh audit` | CPU checkpoint 对比及两次流式扫描，重建 5k 子集 |
| `run.sh evaluate-baseline` | 同一 G2 下原前端与历史 continue 的 4,462 题配对评测 |
| `run.sh train-front-g0` / `train-front-g2` | 冻结 G0/G2 更新前端 |
| `run.sh evaluate-fronts` | 固定 G2 评测两种新前端，并核对缓存基线 |
| `run.sh train-generator` / `evaluate-generator` | 冻结原前端，仅更新/评测 G2 LoRA |
| `run.sh intervention` | 原文档/人工错文档 gold 条件 NLL；不是错文档生成评测 |
| `verify_results.py --run-dir ...` | 不加载模型，从原始预测复核已发布的均值 |

执行顺序、环境版本、输入目录、预算及重跑限制见 [复现说明](../../docs/reproducibility.md)。本包不会自动排队执行整轮实验。原来的 3 组训练已结束。
