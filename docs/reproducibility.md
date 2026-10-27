# 受控实验复现说明

本说明对应 2026-10-27 的首轮调查。单 seed/子集、NQ top1、一个预算点；三个训练组是匹配机制对照，不是旧双卡历史运行的逐比特复现。

## 实际环境与输入

已执行环境：Python 3.10，torch 2.6.0+cu124，transformers 4.57.6，peft 0.18.1，flash-attn 2.7.4.post1，numpy 2.2.6，datasets 4.8.4，accelerate 1.13.0，safetensors 0.8.0。版本快照见 [environment.json](../experiments/repair_20261027/environment.json)。基础版本说明不等于跨 GPU/驱动的逐比特保证。

资产根目录由 `SELECOM_ROOT` 指定；需要以下相对路径：

```text
baselineModel/Qwen3-Embedding-0.6B/
baselineModel/Qwen2.5-7B-Instruct/
checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector/
checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator/
data/stage1/stage1_train_data.jsonl
data/nq/eval/nq_eval.jsonl
results/stage1_continue/checkpoint-79/
```

模型及训练数据来自 [SeleCom 模型](https://huggingface.co/Ryan7458/Selecom/tree/main)、[训练数据](https://huggingface.co/datasets/Ryan7458/SeleCom_Training/tree/main)，基础模型分别为 Qwen3-Embedding-0.6B 与 Qwen2.5-7B-Instruct。NQ 评估文件应与已发布 manifest 的 SHA 一致，不能用另一个切分替代后仍声称复现本表。历史 continue checkpoint 是私有实验资产，没有打包进 Git；缺少它时无法完成历史成对重评，应明确报告缺失。

原 Stage1 文件 14,004,010 行，12,044,324,487 bytes。审计以 seed2025 重建 5,000 条并保存顺序和子集哈希，检查 max640 下问题和 8 个 ENCODE 全部保留；原历史运行没有数据哈希，因此历史来源不确定性仍存在。原始数据、逐题预测和有效权重保持在实验存储中，仓库不重复分发。

## 执行

```bash
export SELECOM_ROOT=/path/to/selecom-assets
export SELECOM_RUN_DIR="$SELECOM_ROOT/research/reproduced-20261027"
export PYTHON_BIN=python
RUN=experiments/repair_20261027/run.sh
bash "$RUN" check
bash "$RUN" audit
bash "$RUN" evaluate-baseline
bash "$RUN" train-front-g0
bash "$RUN" train-front-g2
bash "$RUN" evaluate-fronts
bash "$RUN" train-generator
bash "$RUN" evaluate-generator
bash "$RUN" intervention
```

每条单独执行，出现错误先检查对应日志。`SELECOM_RUN_DIR` 必须为新目录，入口拒绝使用原 `repair-20261027` 证据目录。路径可与代码仓库分离；冻结代码由脚本所在目录定位，原先的模型路径只由资产根目录定位。已完成训练自动跳过，存在 started 记录而未完成时拒绝重启。评估 manifest 不一致时拒绝覆盖；intervention 的未完成输出需先核查。

新 launcher 强制 `CUDA_VISIBLE_DEVICES=0`，不运行多卡分布式；OMP/MKL 为2、无数据 worker、离线加载。每次 GPU命令超时3小时。训练保留原微批4/global64、79步、峰值 LR5e-6、warmup50、AdamW beta0.9/0.999、eps1e-8、wd0、clip1、FP32可训练参数和优化器状态、BF16 autocast。冻结 generator 也显式 eval，固定 dropout；专用 embedding 冻结。

G0 是在同一个 PEFT 模型中禁用 Stage2 adapter，G2 为启用。G0/G2 前端最终都搭配原 G2 评估。generator-only 使用原前端，只更新161,480,704个LoRA参数；前端组更新654,504,960参数，容量不相同。

## 评测身份和验证

统一 greedy、batch4、max_new_tokens128、repetition_penalty1.05；NQ全部4,462题同序，EM/F1 对参考别名取最大，内容 NLL 使用第一个 gold 的正确 causal shift。新评估器复用基线前先逐字核对64题答案并检查 NLL。所有条件格式无效数为0。

本仓库的 [metrics.json](../experiments/repair_20261027/results/metrics.json) 保留原运行数据/模型/代码身份、bootstrap区间及资源记录。路径以环境变量占位符替代，数值与哈希不变。脚本便携化后的新哈希见 source-manifest，重跑会生成新的 manifest，不应把新源码冒充原运行的相同文件。

```bash
python experiments/repair_20261027/verify_results.py --run-dir /path/to/retained/repair-20261027
```

该命令纯 CPU/标准库读取现存逐题预测，核对全部均值；不重训、不生成，也不把题目 bootstrap 当作训练 seed 方差。归档区间为4,000次题目配对bootstrap、seed20261027、探索性且未作多重检验校正。

512题干预保留问题/gold并换成另一题不含参考字符串的文档，只测NLL。参考字符串不等于人工证据标注，gap增大不等于更忠实。完整报告包含失败原因：首次 safetensors 直接 CUDA adapter 加载失败，修复为 CPU 加载后迁移；该启动失败未产生预测。

已完成环节累计约2.58单GPU小时，训练约25.4分钟；峰值PyTorch allocated约30.14GiB。它是实验作业耗时，不是端到端在线RAG加速数据；后续效率比较必须计入selector/projector。
