# -*- coding: utf-8 -*-
"""
项目配置文件 - Streamlit版本
"""

# API配置
API_KEY = "sk-trspbkqtlmeezqymfhuktmikyudbevnrqudyniabwwhzqaos"
BASE_URL = "https://api.siliconflow.cn/v1"

# Embedding模型配置
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_ENCODING_FORMAT = "base64"

# LLM模型配置
LLM_MODEL = "THUDM/GLM-4-9B-0414"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.7
LLM_TOP_K = 50
LLM_FREQUENCY_PENALTY = 0.5

# 文件路径配置 - Streamlit版本
import os
EMBEDDINGS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "题库", "embeddings_cache.pkl")

# 并发配置
# 基于API限制: L0级别 1000 RPM (每分钟请求数), 50000 TPM (每分钟token数)
# 1000 RPM = 16.67 RPS，考虑安全边际，设置为15 RPS
MAX_CONCURRENT_REQUESTS = 50  # 优化并发数，基于API限制
REQUEST_TIMEOUT = 30  # 超时时间
MAX_RETRIES = 3  # 最大重试次数

# 相似度配置
TOP_K_SIMILAR = 3
SIMILARITY_THRESHOLD = 0.7

# 输出配置
VERBOSE = False  # 设置为False减少输出信息，True显示详细信息