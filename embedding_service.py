# -*- coding: utf-8 -*-
"""
Embedding向量化服务模块
"""

import asyncio
import requests
import base64
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time
from config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, 
    EMBEDDING_ENCODING_FORMAT, MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT, VERBOSE
)


class EmbeddingService:
    """Embedding向量化服务类"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self.encoding_format = EMBEDDING_ENCODING_FORMAT
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.total_tokens = 0
        self.request_count = 0
        
    def _sync_get_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """同步获取单个文本的embedding向量"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": text,
            "encoding_format": self.encoding_format,
            "dimensions": self.dimensions
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers,
                json=data,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                result = response.json()
                self.total_tokens += result.get('usage', {}).get('total_tokens', 0)
                self.request_count += 1
                return result
            else:
                print(f"API请求失败: {response.status_code}, {response.text}")
                return None
                    
        except Exception as e:
            print(f"获取embedding时发生错误: {str(e)}")
            return None
    
    async def get_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """异步获取单个文本的embedding向量"""
        async with self.semaphore:
            return await asyncio.to_thread(self._sync_get_embedding, text)
    
    def decode_base64_embedding(self, base64_str: str) -> np.ndarray:
        """解码base64格式的embedding向量"""
        try:
            # 解码base64字符串
            decoded_bytes = base64.b64decode(base64_str)
            # 转换为float32数组
            embedding = np.frombuffer(decoded_bytes, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"解码embedding时发生错误: {str(e)}")
            return np.array([])
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量获取文本的embedding向量"""
        if VERBOSE:
            print(f"开始处理 {len(texts)} 个文本的向量化...")
        start_time = time.time()
        
        tasks = [self.get_embedding(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉失败的请求
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and result is not None:
                result['original_text'] = texts[i]
                result['index'] = i
                valid_results.append(result)
            else:
                print(f"第 {i+1} 个文本向量化失败")
        
        end_time = time.time()
        if VERBOSE:
            print(f"向量化完成! 耗时: {end_time - start_time:.2f}秒")
            print(f"成功处理: {len(valid_results)}/{len(texts)} 个文本")
            print(f"总Token使用量: {self.total_tokens}")
            print(f"API请求次数: {self.request_count}")
        
        return valid_results
    
    def process_embeddings(self, embedding_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理embedding结果，解码向量数据"""
        processed_results = []
        
        for result in embedding_results:
            try:
                # 获取embedding数据
                embedding_data = result['data'][0]['embedding']
                
                if self.encoding_format == "base64":
                    # 解码base64格式的embedding
                    embedding_vector = self.decode_base64_embedding(embedding_data)
                else:
                    # 直接使用数组格式的embedding
                    embedding_vector = np.array(embedding_data, dtype=np.float32)
                
                processed_result = {
                    'text': result['original_text'],
                    'index': result['index'],
                    'embedding': embedding_vector,
                    'model': result['model'],
                    'usage': result.get('usage', {})
                }
                
                processed_results.append(processed_result)
                
            except Exception as e:
                print(f"处理embedding结果时发生错误: {str(e)}")
                continue
        
        return processed_results
    
    def get_token_usage(self) -> Dict[str, int]:
        """获取token使用统计"""
        return {
            'total_tokens': self.total_tokens,
            'request_count': self.request_count,
            'avg_tokens_per_request': self.total_tokens // max(self.request_count, 1)
        }