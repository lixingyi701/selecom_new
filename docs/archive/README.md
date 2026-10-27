# 历史资料

这些资料用于追溯先前假设，不代表已确认的机制或当前最优方法。

- [旧 README](upstream-readme.md)：保留上游归属与旧使用说明。
- [Stage3 设计稿](proposals/stage3-design.pdf)：原仓库根目录和 util 中两份文件逐字节相同，现保留一份。其 KL、attention/MSE/范数项未在既有 Stage3 forward 中实现。
- `diagnostics/`：原 results 中的三个 drift 诊断文件，数值未改，机器路径以环境变量占位符替代。首内容 token KL 与缺少 measurement_strategy 的 KL 不能混为同一指标；任意阈值也不能证明有害失配。
- [历史 continue 源码记录](historical-continue/README.md)：服务器当时保留的脚本，按文本归档；其“已经失配”的注释是原假设，不是本轮结论。

原先跟踪的 `results/stage3/checkpoint-{625,1250,1875}` 只有配置、trainer state 和转换脚本，没有模型权重。当前 Git 树不再把这些片段显示为完整 checkpoint；原内容仍可从 [整理前提交](https://github.com/lixingyi701/selecom_new/tree/7e0f4da7871d5a70eb60a54fdfdfac0c219c9499/results/stage3) 获取。服务器上的有效 checkpoint 没有因此删除。

原论文 PDF 移至 [docs/papers/SeleCom.pdf](../papers/SeleCom.pdf)，图示与 MIT License 保持归属。
