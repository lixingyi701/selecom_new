# 相关工作与解法

核验日期：2026-10-27。下面记录本轮实际检查的原文与适用范围；预印本不当作已录用论文，其他任务上的有效性不当作 SeleCom 结果。旧研究记录中的“简单加 Stage3/KL”的定位已由本页和受控结果更新。

| 工作与版本 | 类型及已覆盖的方法 | 与当前问题的关系和边界 |
| --- | --- | --- |
| [How transferable are features in deep neural networks?，NIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html) | 表征迁移实证研究；拆分共同适配特征引起优化困难 | 协同适配的经典参照；视觉网络结论不直接证明本系统失配 |
| [Fundamental Limits of Prompt Compression，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html) | 理论与方法；黑盒硬 prompt 的率失真框架及 query-aware 压缩 | 提供任务相关失真视角；正式边界不是双端可训练软表示 |
| [QUITO-X，Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.362/) | 方法；信息瓶颈及与问题相关的信息保留 | “任务相关压缩”已有系统化表述 |
| [xRAG，2024，§3.3](https://arxiv.org/html/2405.13792) | 方法；固定 retriever/generator，projector 的答案 NLL＋完整文档教师 KL | 软压缩加 KL 已有明确先例；其表示预算和前端不同 |
| [PISCO，Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.800/) | 方法；序列级蒸馏训练压缩 RAG，无需人工答案标注 | 联合适配与教师生成答案不是空白；压缩不依赖查询 |
| [OSCAR，arXiv v2，2026-03-04](https://arxiv.org/html/2504.07109) | 方法；query-conditioned 在线软压缩、序列蒸馏、压缩器与 generator 联合训练，兼顾 reranking | 与直接联合训练高度接近；其冻结 generator 消融较弱，与 SeleCom Fig.8 共同说明结论依赖训练设置 |
| [RAM，arXiv v2，2026-02-27](https://arxiv.org/html/2602.01840) | 方法；问题相关混合压缩，LM 与问题—片段对比目标 | 相关性对比已有先例；片段相关不等于正确使用证据 |
| [DEX-Comp，arXiv v2，2026-09-28，§3.2–3.3](https://arxiv.org/html/2609.05152v2) | 方法；教师答对样本上的联合 KL 蒸馏，教师答错样本上的结果奖励 GRPO | “联合＋KL＋教师正确性筛选”已覆盖；采用问题无关离线压缩，缓存收益不能直接移用 |
| [SADA，ACL 2026，§3.3](https://aclanthology.org/2026.acl-long.1046/) | 方法；将上下文映射到动态 adapter，完整上下文 oracle 的隐藏状态和输出监督 | 提供过程监督参照；不是 SeleCom 式 soft-token 架构，也不证明逐层 MSE 普遍合适 |
| [On-Policy Distillation of Language Models，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) | 方法；GKD 在学生生成序列上获得教师反馈 | 处理训练轨迹与推理轨迹不一致；本轮 NLL/F1 不一致尚不能单独确诊 exposure bias |
| [Entropy-Aware On-Policy Distillation，ICML 2026](https://proceedings.mlr.press/v306/jin26e.html) | 方法；教师高熵时加入 forward KL，缓解 reverse KL 的模式选择与不稳定 | KL 方向与教师不确定性需要明确；实验以数学推理为主 |
| [CAFD，arXiv v2，2026-10-21](https://arxiv.org/abs/2609.39953) | 方法；固定多模态压缩流程，用完整上下文 self-teacher 指导压缩学生的 on-policy 轨迹，无需 gold/奖励 | 与后端适应压缩表示直接相关；从多模态迁移到 SeleCom 文本 QA 是待检验推论 |
| [Faithfulness-QA，arXiv v2，2026-04-29](https://arxiv.org/html/2604.25313v2) | 数据集；类型一致的答案实体替换，制造文档和参数记忆冲突 | 提供反事实数据构造参照；不能把数据集论文当作当前方法训练收益 |
| [KEEP-Bench，arXiv v1，2026-10-19](https://arxiv.org/html/2610.26827v1) | 基准；4,102 个样本，直接检查并行、条件、非均衡证据的命题保留 | 区分答案正确与压缩忠实；聚焦可直接检查的文本压缩，不能原样评估 soft embeddings |
| [Compression Is Not Evaluation-Neutral，arXiv v2，2026-10-02](https://arxiv.org/html/2606.21807v2) | 评测实证；固定压缩材料后比较 reader 升级，发现 reader 收益可能被压缩掩盖 | 支持区分完整流水线收益、reader 收益和压缩作用；主要研究文本压缩 |

## 将这些解法用于当前系统

联合或交替更新对应协同适配；小学习率、受限参数块及重放约束可作为稳定化选择，但没有理由预设联合一定优于 generator-only。

完整上下文蒸馏应对齐经过验证的冻结教师，而不是让新 generator 回到旧 generator 的分布。教师与学生使用相同答案前缀或明确的学生轨迹；教师答错时降低模仿或改变监督。教师答案正确仍不能证明其依赖了文档。

状态监督应选择可以对应的答案位置和少数层。完整文档与 soft prefix 的 token 数不同，不能直接把所有位置的隐藏状态逐一做 MSE；状态变化本身也不是有害证据。

反事实监督需要区分两类变化：保留正确支持、添加无关干扰时保持稳定；在明确“按文档作答”的合成任务中改变决定答案的事实时随之改变。删除唯一支持后的目标需要预先定义为证据不足、拒答或其他任务策略，不能继续强迫猜原答案。

本次检索没有建立“端点自适应选择机制已经具备独立新颖性”的结论。现有观察支持继续提出该问题；具体机制仍需与这些近邻工作逐项比较。
