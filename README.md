# SeleCom 研究与受控验证

本仓库基于 [SeleCom 官方实现](https://github.com/yhliu7458/SeleCom)，保留既有 Stage1/Stage2/Stage3 代码，并记录对训练阶段交互、持续训练和证据利用的研究。本仓库是研究分支，原论文为 WWW 2026 的 [Rethinking Soft Compression in Retrieval-Augmented Generation](https://arxiv.org/abs/2602.15856)。

**2026-10-27：首轮受控调查已完成。适配后的 generator 改变了前端继续训练的作用，但现有证据不足以证明有害双端失配，也不支持必须优先更新前端。**

## 阅读入口

| 内容 | 入口 |
| --- | --- |
| 问题定义、经典问题归类与研究判断 | [研究说明](docs/research/overview.md) |
| 历史 stage1_continue 核查及全部实验结论 | [受控验证记录](docs/research/controlled-validation.md) |
| 最新相关工作、适用边界及已覆盖的思路 | [相关工作](docs/research/related-work.md) |
| 后续研究问题与候选监督 | [后续研究](docs/research/next-steps.md) |
| 环境、数据、checkpoint、执行顺序及结果身份 | [复现说明](docs/reproducibility.md) |
| 本轮脚本、冻结依赖、聚合指标和来源哈希 | [实验包](experiments/repair_20261027/README.md) |
| 历史设计稿、drift 诊断、旧 README | [历史资料](docs/archive/README.md) |
| 服务器目录和缓存维护规则 | [存储维护](docs/storage-maintenance.md) |

## 已观察到的结果

同一 5,000 条训练子集、79 次更新；NQ top1 共 4,462 题，greedy、batch4、最多 128 新 token。

| 模型 | EM，% | F1，% | 内容 NLL |
| --- | ---: | ---: | ---: |
| 原前端＋原 Stage2 generator | 38.0547 | 49.0096 | 1.58795 |
| 通过冻结 G0 更新前端，再搭配固定 G2 | 38.0547 | 49.0125 | 1.58496 |
| 通过冻结 G2 更新前端，再搭配固定 G2 | 38.0547 | 49.3642 | 1.57260 |
| 固定原前端，仅继续训练 G2 LoRA | 38.1667 | 49.5476 | 1.56256 |

G0/G2 分别是在同一模型中禁用/启用 Stage2 LoRA。generator-only 相对前端更新的 F1 差异区间包含 0，不能据点估计宣布普遍优胜。512 题文档扰动只测 gold 条件 NLL，没有测错文档自由生成。完整区间、配置和限制见验证记录与 [机器可读指标](experiments/repair_20261027/results/metrics.json)。

## 代码与运行

`main/`、`model/`、`util/` 保留历史实现。旧 launch 脚本有多卡和环境相关配置；本轮复现使用独立实验包及其冻结依赖，不修改旧脚本。新增入口只允许物理 GPU 0，使用预采样数据，保留至少 24 GiB 系统可用内存和 8 GiB GPU 空闲保护。

```bash
git clone https://github.com/lixingyi701/selecom_new.git
cd selecom_new
export SELECOM_ROOT=/path/to/selecom-assets
export SELECOM_RUN_DIR="$SELECOM_ROOT/research/reproduced-20261027"
export PYTHON_BIN=python
bash experiments/repair_20261027/run.sh check
```

检查命令不训练，也不下载模型。准备资产后的实际执行见复现说明。训练数据、逐题预测、日志和有效权重保存在实验存储中，不随 Git 提交；仓库保存汇总指标、身份哈希和复现代码。原始资料和既有未提交修改在服务器整理前已备份。

问题条件压缩设定保持不变。查询到来后仍需运行 selector/projector，效率比较必须计入在线压缩成本。首轮调查已停止，没有自动续训或待启动任务。

## 归属与许可

上游方法、图示及论文归原作者所有；历史 Stage3 设计稿是研究提案，不能视作已实现方法。代码沿用 [MIT License](LICENSE)。数据和模型遵循各自提供方的许可。
