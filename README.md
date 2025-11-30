# Streamlit 语义相似度匹配工具

一款基于向量检索与可选 LLM 语义增强的 Streamlit 应用，支持批量题目匹配、导出及多种相似度算法选择。

## 功能特性
- 🔍 **多算法相似度匹配**：支持余弦/欧几里得/点积相似度，侧边栏可选。
- 🤖 **可选 LLM 语义增强**：勾选即可调用 LLM 重新排序并给出理由，关闭可降低耗时与费用。
- ⚡ **并发向量化**：批量异步请求 Embedding，提升吞吐。
- 📊 **结果统计与导出**：内置匹配统计、Excel 下载与列宽优化。
- 🧩 **可调阈值与 Top-K**：侧边栏可实时调整过滤阈值与返回数量。

## 项目结构
```
.
├── app.py                     # Streamlit 入口
├── similarity_app/
│   ├── __init__.py
│   ├── config.py              # 配置与路径
│   ├── main.py                # 主界面与业务流程
│   ├── similarity_matcher.py  # 相似度计算与统计
│   └── services/
│       ├── __init__.py
│       ├── embedding_service.py
│       └── llm_service.py
├── requirements.txt
└── 题库/
    └── embeddings_cache.pkl   # 预置题库向量缓存（需自行准备）
```

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 将基础题库向量缓存放置到 `题库/embeddings_cache.pkl`（支持含 `embeddings` 和 `texts` 的字典格式，或兼容旧格式）。
3. 运行应用：`streamlit run app.py`
4. 在浏览器中按照提示上传 UTF-8 BOM 的 CSV（第一列为题目文本），启动匹配并下载 Excel 结果。

## 可选功能开关
- **相似度算法选择**：侧边栏下拉框选择余弦/欧几里得/点积。
- **Top-K 与阈值**：通过滑块调整返回数量和过滤阈值。
- **LLM 语义增强**：复选框控制是否调用 LLM（关闭后仅使用向量结果）。

## 配置说明
`similarity_app/config.py` 中定义了 API Key、模型、并发、阈值等参数，可按需修改。默认向量缓存路径为 `题库/embeddings_cache.pkl`（相对项目根目录）。

## 开发建议
- 在开启 VERBOSE 时可查看向量化与 LLM 请求的 Token 统计信息。
- 若使用代理或私有模型服务，可调整 `BASE_URL` 与对应模型名称。
