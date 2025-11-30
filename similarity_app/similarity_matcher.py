# -*- coding: utf-8 -*-
"""
相似度计算和匹配模块
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine

from similarity_app.config import TOP_K_SIMILAR, SIMILARITY_THRESHOLD, VERBOSE


class SimilarityMatcher:
    """相似度匹配器类"""

    def __init__(self, top_k: int = TOP_K_SIMILAR, threshold: float = SIMILARITY_THRESHOLD):
        self.top_k = top_k
        self.threshold = threshold

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        try:
            vec1 = vec1.flatten()
            vec2 = vec2.flatten()

            if len(vec1) != len(vec2):
                print(f"警告: 向量长度不一致 {len(vec1)} vs {len(vec2)}")
                return 0.0

            if np.allclose(vec1, 0) or np.allclose(vec2, 0):
                return 0.0

            similarity = 1 - cosine(vec1, vec2)
            if np.isnan(similarity):
                return 0.0

            return float(similarity)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"计算余弦相似度时发生错误: {str(exc)}")
            return 0.0

    def euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的欧几里得相似度"""
        try:
            vec1 = vec1.flatten()
            vec2 = vec2.flatten()

            if len(vec1) != len(vec2):
                return 0.0

            distance = np.linalg.norm(vec1 - vec2)
            similarity = 1 / (1 + distance)
            return float(similarity)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"计算欧几里得相似度时发生错误: {str(exc)}")
            return 0.0

    def dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的点积相似度"""
        try:
            vec1 = vec1.flatten()
            vec2 = vec2.flatten()

            if len(vec1) != len(vec2):
                return 0.0

            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"计算点积相似度时发生错误: {str(exc)}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Dict[str, np.ndarray],
        similarity_method: str = "cosine",
    ) -> List[Tuple[str, float]]:
        """找到最相似的候选项"""
        similarities: List[Tuple[str, float]] = []

        if similarity_method == "cosine":
            similarity_func = self.cosine_similarity
        elif similarity_method == "euclidean":
            similarity_func = self.euclidean_similarity
        elif similarity_method == "dot_product":
            similarity_func = self.dot_product_similarity
        else:
            print(f"未知的相似度计算方法: {similarity_method}，使用余弦相似度")
            similarity_func = self.cosine_similarity

        for text, embedding in candidate_embeddings.items():
            similarity = similarity_func(query_embedding, embedding)
            if similarity >= self.threshold:
                similarities.append((text, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[: self.top_k]

    def batch_find_similar(
        self,
        query_embeddings: Dict[str, np.ndarray],
        candidate_embeddings: Dict[str, np.ndarray],
        similarity_method: str = "cosine",
    ) -> Dict[str, List[Tuple[str, float]]]:
        """批量查找相似项"""
        results: Dict[str, List[Tuple[str, float]]] = {}

        if VERBOSE:
            print("开始批量相似度匹配...")
            print(f"查询项数量: {len(query_embeddings)}")
            print(f"候选项数量: {len(candidate_embeddings)}")
            print(f"相似度计算方法: {similarity_method}")
            print(f"相似度阈值: {self.threshold}")
            print(f"返回Top-K: {self.top_k}")

        for i, (query_text, query_embedding) in enumerate(query_embeddings.items()):
            similar_items = self.find_most_similar(
                query_embedding,
                candidate_embeddings,
                similarity_method,
            )

            results[query_text] = similar_items

            if VERBOSE and ((i + 1) % 10 == 0 or (i + 1) == len(query_embeddings)):
                print(f"已处理: {i + 1}/{len(query_embeddings)} 个查询")

        if VERBOSE:
            print("批量相似度匹配完成!")
        return results

    def get_similarity_matrix(
        self,
        embeddings1: Dict[str, np.ndarray],
        embeddings2: Dict[str, np.ndarray] = None,
        similarity_method: str = "cosine",
    ) -> np.ndarray:
        """计算相似度矩阵"""
        if embeddings2 is None:
            embeddings2 = embeddings1

        texts1 = list(embeddings1.keys())
        texts2 = list(embeddings2.keys())
        matrix = np.zeros((len(texts1), len(texts2)))

        if similarity_method == "cosine":
            similarity_func = self.cosine_similarity
        elif similarity_method == "euclidean":
            similarity_func = self.euclidean_similarity
        elif similarity_method == "dot_product":
            similarity_func = self.dot_product_similarity
        else:
            similarity_func = self.cosine_similarity

        for i, text1 in enumerate(texts1):
            for j, text2 in enumerate(texts2):
                similarity = similarity_func(
                    embeddings1[text1],
                    embeddings2[text2],
                )
                matrix[i, j] = similarity

        return matrix

    def filter_by_threshold(self, similarities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """根据阈值过滤相似度结果"""
        return [(text, score) for text, score in similarities if score >= self.threshold]

    def get_statistics(self, similarity_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """获取相似度匹配统计信息"""
        total_queries = len(similarity_results)
        total_matches = sum(len(matches) for matches in similarity_results.values())

        all_similarities: List[float] = []
        for matches in similarity_results.values():
            all_similarities.extend([score for _, score in matches])

        avg_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0
        max_similarity = float(np.max(all_similarities)) if all_similarities else 0.0
        min_similarity = float(np.min(all_similarities)) if all_similarities else 0.0

        queries_with_matches = sum(1 for matches in similarity_results.values() if matches)
        match_rate = queries_with_matches / total_queries if total_queries > 0 else 0.0

        return {
            "total_queries": total_queries,
            "total_matches": total_matches,
            "queries_with_matches": queries_with_matches,
            "match_rate": round(match_rate, 4),
            "avg_matches_per_query": round(total_matches / total_queries, 2) if total_queries > 0 else 0.0,
            "avg_similarity": round(avg_similarity, 4),
            "max_similarity": round(max_similarity, 4),
            "min_similarity": round(min_similarity, 4),
            "similarity_threshold": self.threshold,
            "top_k": self.top_k,
        }
