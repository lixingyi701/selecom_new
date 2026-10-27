# 研究问题与当前判断

更新于 2026-10-27。已完成的证据见 [受控验证记录](controlled-validation.md)；本文将事实、解释和后续候选机制分开。

## 方法与实现纠正

系统为 `z = C_phi(q, d)`、`p(y) = G_theta(q, z)`，前端包含 selector 和 projector。Stage1 冻结的是已经预训练并经过指令训练的 generator，其梯度仍可经输入表示传给前端。训练损失覆盖答案序列 next-token CE。首个内容 token 的 KL 来自历史诊断脚本，并非训练只监督首词。

实际 selector prompt 按 Document、Question、Instruction、ENCODE 排列；末尾 ENCODE 可访问前面的文档和问题。当前历史 Stage3 forward 是 CE-only，设计稿里的 KL、attention/MSE/范数约束并未实现。原论文 Fig.8 的 selector-tuned Stage2 是直接联合更新的相关负面参照，不能推导所有联合优化均无效。[SeleCom 原文](https://arxiv.org/html/2602.15856v1)

## 对应的经典问题

**模块协同适配与接口兼容性。** 前端先适应旧 generator，generator 再适应固定前端，属于分阶段的分块优化。Stage2 不改变前端输出，改变的是 generator 对表示的解释，以及适合新 generator 的前端梯度。它与拆分共同适配特征后的优化困难相关；这种归类是对当前系统的解释，不是已经证明存在有害失配。[Yosinski 等，NIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html)

**任务相关的有损压缩。** 固定表示预算下，需要保留让特定 generator 回答当前问题的事实、关系和限定条件。前端丢失信息与后端未能读出信息，会产生相似的答案失败，却需要不同干预。率失真与信息瓶颈提供问题框架；硬压缩定理不能未经论证直接当作可训练软压缩的结论。[率失真框架，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html)、[QUITO-X，Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.362/)

**答案监督下的信用分配与捷径学习。** 正确答案可以来自文档、参数记忆或问题线索。CE 约束输出，却不唯一指定证据路径。新的 KL、状态监督或对比目标需要说明排除了哪种错误路径，不能仅以监督更丰富作为贡献。

## 已完成调查说明了什么

历史 continue 更新完整 encoder＋projector，冻结 Stage2 generator 和专用 embedding，使用 5,000 个旧 Stage1 样本、79 次更新。旧评测存在采样、batch 和随机种子混杂；统一重评后历史 F1 增益约 +0.417 个百分点，其题目 bootstrap 区间包含 0。

新匹配训练固定子集、顺序和更新预算。通过 G2 更新前端比通过 G0 获得更大的 NLL 改善，支持 generator 状态会改变前端继续训练的作用；仅更新 G2 LoRA 也获得同量级答案改善。前端与后端更新的直接 F1 比较区间包含 0。

这批数据没有带来新增原始样本。重复暴露、样本重新加权、额外更新以及参数块是否直接接受过这些监督，都可能贡献收益。Stage1 用过某个例子，不等于 generator 的可训练参数已经在相同条件下学过该例子。数据作用与接口条件性作用也不构成互斥解释。

原文档/错配文档诊断发现两端更新改变了不同的文档条件行为。较大的 NLL 差距可能包含正确文档收益与错误文档干扰，不能直接当作更好的忠实性。一次训练 seed、NQ top1、一个预算点和人工错配文档限制了外推；NQ 已用于诊断，后续应使用新的开发/最终评估划分。

## 保留的研究问题

生成器适配后，如何让查询条件软压缩保留正确证据的收益，同时抑制干扰证据，并确定应该更新前端还是后端？

联合训练、完整上下文 KL、序列蒸馏和通用相关性对比均已有近邻工作。后续贡献需要落在具体失败条件、端点选择或证据监督机制上，并超过 generator-only 同数据重放基线。[相关工作](related-work.md) 与 [后续研究](next-steps.md) 列出适用边界。
