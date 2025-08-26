# -*- coding: utf-8 -*-
"""
LLM模型调用服务模块
"""

import asyncio
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from config import (
    API_KEY, BASE_URL, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_TOP_P, LLM_TOP_K, LLM_FREQUENCY_PENALTY, MAX_CONCURRENT_REQUESTS, REQUEST_TIMEOUT, MAX_RETRIES, VERBOSE
)


class LLMService:
    """LLM模型调用服务类"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.top_p = LLM_TOP_P
        self.top_k = LLM_TOP_K
        self.frequency_penalty = LLM_FREQUENCY_PENALTY
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.total_tokens = 0
        self.request_count = 0
    
    def _sync_call_llm(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """同步调用LLM模型，带重试机制"""
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "frequency_penalty": self.frequency_penalty,
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.total_tokens += result.get('usage', {}).get('total_tokens', 0)
                    self.request_count += 1
                    return result
                else:
                    print(f"LLM API请求失败: HTTP {response.status_code} (尝试 {attempt + 1}/{MAX_RETRIES})")
                    print(f"错误详情: {response.text}")
                    if attempt == MAX_RETRIES - 1:  # 最后一次尝试
                        print(f"请求URL: {self.base_url}/chat/completions")
                        print(f"使用模型: {self.model}")
                    return None
                    
            except Exception as e:
                print(f"调用LLM时发生错误: {str(e)} (尝试 {attempt + 1}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES - 1:  # 最后一次尝试
                    print(f"错误类型: {type(e).__name__}")
                    print(f"请求URL: {self.base_url}/chat/completions")
                    print(f"使用模型: {self.model}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                else:
                    # 等待一段时间后重试
                    time.sleep(2 ** attempt)  # 指数退避
                    
        return None
    
    async def call_llm(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """异步调用LLM模型，使用asyncio.to_thread包装同步调用"""
        async with self.semaphore:
            return await asyncio.to_thread(self._sync_call_llm, messages)
    
    def create_similarity_prompt(self, query_text: str, candidate_texts: List[str]) -> List[Dict[str, str]]:
        """创建相似度判断的提示词"""
        candidate_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(candidate_texts)])
        
        prompt = f"""请分析以下查询文本与候选文本的语义相似度，并给出详细的相似度评分和理由。

查询文本：
```
{query_text}
```

候选文本：
{candidate_list}

请按照以下格式回答：
1. 对每个候选文本，给出0-1之间的相似度评分（保留4位小数）
2. 简要说明相似度评分的理由
3. 按相似度从高到低排序

回答格式示例：
候选文本1: 0.8500 - 理由：主题相同，都涉及职业变化问题
候选文本2: 0.6200 - 理由：有一定关联，但侧重点不同
候选文本3: 0.3100 - 理由：主题差异较大，仅有少量关键词重叠

排序结果：候选文本1 > 候选文本2 > 候选文本3"""
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的文本相似度分析专家，擅长分析文本的语义相似度。请客观、准确地评估文本之间的相似程度。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        return messages
    
    def parse_similarity_response(self, response_text: str, candidate_texts: List[str]) -> List[Tuple[str, float, str]]:
        """解析LLM返回的相似度评分"""
        results = []
        
        try:
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or '候选文本' not in line:
                    continue
                
                # 尝试提取相似度评分
                parts = line.split(':')
                if len(parts) >= 2:
                    # 提取评分部分
                    score_part = parts[1].strip()
                    
                    # 使用正则表达式提取数字评分
                    import re
                    score_match = re.search(r'(\d+\.?\d*)', score_part)
                    
                    if score_match:
                        try:
                            score = float(score_match.group(1))
                            # 确保评分在0-1范围内
                            if score > 1.0:
                                score = score / 10.0  # 如果是10分制，转换为1分制
                            
                            # 提取候选文本索引
                            candidate_idx_match = re.search(r'候选文本(\d+)', parts[0])
                            if candidate_idx_match:
                                candidate_idx = int(candidate_idx_match.group(1)) - 1
                                
                                if 0 <= candidate_idx < len(candidate_texts):
                                    candidate_text = candidate_texts[candidate_idx]
                                    
                                    # 提取理由
                                    reason = ''
                                    if '-' in score_part:
                                        reason = score_part.split('-', 1)[1].strip()
                                    elif '：' in score_part:
                                        reason = score_part.split('：', 1)[1].strip()
                                    
                                    results.append((candidate_text, score, reason))
                        
                        except (ValueError, IndexError) as e:
                            if VERBOSE:
                                print(f"解析评分时发生错误: {str(e)}, 行内容: {line}")
                            continue
                    else:
                        if VERBOSE:
                            print(f"未找到有效评分，行内容: {line}")
        
        except Exception as e:
            print(f"解析LLM响应时发生错误: {str(e)}")
        
        # 如果解析失败，返回默认评分
        if not results:
            if VERBOSE:
                print("LLM响应解析失败，使用默认评分")
                print(f"原始响应: {response_text[:200]}...")
            for i, text in enumerate(candidate_texts):
                results.append((text, 0.5, "LLM解析失败，使用默认评分"))
        
        # 按评分降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def evaluate_similarity_batch(self, 
                                       query_texts: List[str],
                                       candidate_texts_list: List[List[str]]) -> List[List[Tuple[str, float, str]]]:
        """批量评估相似度"""
        total_queries = len(query_texts)
        print(f"正在进行LLM辅助评估... (共{total_queries}个查询)")
        if VERBOSE:
            print(f"开始LLM批量相似度评估...")
            print(f"查询数量: {total_queries}")
        start_time = time.time()
        
        tasks = []
        
        for query_text, candidate_texts in zip(query_texts, candidate_texts_list):
            if candidate_texts:  # 只处理有候选文本的查询
                messages = self.create_similarity_prompt(query_text, candidate_texts)
                task = self.call_llm(messages)
                tasks.append((query_text, candidate_texts, task))
            else:
                tasks.append((query_text, [], None))
        
        # 并发执行所有任务
        valid_tasks = [task for _, _, task in tasks if task is not None]
        
        if valid_tasks:
            # 使用asyncio.gather实现真正的并发执行
            responses = await asyncio.gather(*valid_tasks, return_exceptions=True)
        else:
            responses = []
        
        # 处理结果
        results = []
        response_idx = 0
        
        for query_text, candidate_texts, task in tasks:
            if task is None:
                results.append([])
                continue
            
            try:
                response = responses[response_idx]
                response_idx += 1
                
                if isinstance(response, Exception):
                    print(f"LLM请求异常，查询: {query_text[:50]}... 错误: {str(response)}")
                    default_results = [(text, 0.5, "请求异常") for text in candidate_texts]
                    results.append(default_results)
                elif response and 'choices' in response:
                    response_text = response['choices'][0]['message']['content']
                    parsed_results = self.parse_similarity_response(response_text, candidate_texts)
                    results.append(parsed_results)
                else:
                    print(f"LLM评估失败，查询: {query_text[:50]}...")
                    default_results = [(text, 0.5, "LLM评估失败") for text in candidate_texts]
                    results.append(default_results)
            
            except Exception as e:
                print(f"处理LLM响应时发生错误: {str(e)}")
                default_results = [(text, 0.5, "处理错误") for text in candidate_texts]
                results.append(default_results)
        
        end_time = time.time()
        print(f"LLM辅助评估完成! 耗时: {end_time - start_time:.2f}秒")
        if VERBOSE:
            print(f"总Token使用量: {self.total_tokens}")
            print(f"API请求次数: {self.request_count}")
        
        return results
    
    async def enhance_similarity_results(self, 
                                       similarity_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float, str]]]:
        """使用LLM增强相似度结果"""
        enhanced_results = {}
        
        # 准备批量处理的数据
        query_texts = []
        candidate_texts_list = []
        
        for query_text, candidates in similarity_results.items():
            if candidates:
                query_texts.append(query_text)
                candidate_texts_list.append([text for text, _ in candidates])
            else:
                enhanced_results[query_text] = []
        
        if not query_texts:
            return enhanced_results
        
        # 批量调用LLM评估
        llm_results = await self.evaluate_similarity_batch(query_texts, candidate_texts_list)
        
        # 合并结果
        for i, query_text in enumerate(query_texts):
            enhanced_results[query_text] = llm_results[i]
        
        return enhanced_results
    
    def get_token_usage(self) -> Dict[str, int]:
        """获取token使用统计"""
        return {
            'total_tokens': self.total_tokens,
            'request_count': self.request_count,
            'avg_tokens_per_request': self.total_tokens // max(self.request_count, 1)
        }