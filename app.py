# -*- coding: utf-8 -*-
"""
Streamlit应用主文件 - 相似题目匹配系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import io

# 导入自定义模块
from embedding_service import EmbeddingService
from similarity_matcher import SimilarityMatcher
from llm_service import LLMService
from config import (
    EMBEDDINGS_CACHE_PATH, TOP_K_SIMILAR, SIMILARITY_THRESHOLD, 
    VERBOSE
)

# 定义相似度方法和输出列配置
SIMILARITY_METHOD = "cosine"
OUTPUT_COLUMNS = {
    'query': '题目',
    'similar_text_1': '相似题目1',
    'similarity_score_1': '相似度1',
    'similar_text_2': '相似题目2',
    'similarity_score_2': '相似度2',
    'similar_text_3': '相似题目3',
    'similarity_score_3': '相似度3'
}


class SimilarityMatchingApp:
    """相似题目匹配应用类"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.similarity_matcher = SimilarityMatcher()
        self.llm_service = LLMService()
        self.base_embeddings = None
        self.base_texts = None
        
    def load_base_embeddings(self) -> bool:
        """加载基础题库的向量数据"""
        try:
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 兼容多种缓存格式
                if isinstance(cache_data, dict):
                    # 目标格式: { 'embeddings': Dict[str, np.ndarray], 'texts': List[str] }
                    if 'embeddings' in cache_data and 'texts' in cache_data:
                        self.base_embeddings = cache_data['embeddings']
                        self.base_texts = cache_data['texts']
                        st.success(f"✅ 成功加载基础题库，共 {len(self.base_texts)} 条数据")
                        return True
                    
                    # 兼容格式: { 'embeddings': { id: { 'text': str, 'embedding': np.ndarray } } }
                    elif 'embeddings' in cache_data and isinstance(cache_data['embeddings'], dict):
                        raw = cache_data['embeddings']
                        reconstructed: Dict[str, np.ndarray] = {}
                        texts: List[str] = []
                        for _id, item in raw.items():
                            try:
                                # 情况A：item为包含text/embedding的字典
                                if isinstance(item, dict):
                                    text = item.get('text')
                                    emb = item.get('embedding')
                                    if text is None:
                                        # 兼容可能的键名
                                        text = item.get('content') or item.get('title') or item.get('question')
                                    if text is not None and emb is not None:
                                        if not isinstance(emb, np.ndarray):
                                            emb = np.array(emb, dtype=np.float32)
                                        reconstructed[text] = emb
                                        texts.append(text)
                                else:
                                    # 情况B：直接映射为 { 文本: 向量 }
                                    text = str(_id)
                                    emb = item
                                    if emb is not None:
                                        if not isinstance(emb, np.ndarray):
                                            emb = np.array(emb, dtype=np.float32)
                                        reconstructed[text] = emb
                                        texts.append(text)
                            except Exception:
                                continue
                        if reconstructed:
                            self.base_embeddings = reconstructed
                            self.base_texts = texts
                            st.success(f"✅ 成功加载基础题库，共 {len(self.base_texts)} 条数据")
                            return True
                        else:
                            st.error("❌ 缓存文件格式不正确（未解析到有效的文本与向量）")
                            return False
                    else:
                        st.error("❌ 缓存文件格式不正确")
                        return False
                else:
                    st.error("❌ 缓存文件格式不正确")
                    return False
            else:
                st.error(f"❌ 未找到基础题库向量文件: {EMBEDDINGS_CACHE_PATH}")
                return False
        except Exception as e:
            st.error(f"❌ 加载基础题库向量数据失败: {str(e)}")
            return False
    
    async def process_user_texts(self, user_texts: List[str]) -> List[np.ndarray]:
        """处理用户上传的文本，生成向量"""
        try:
            # 异步请求Embedding
            embedding_results = await self.embedding_service.get_embeddings_batch(user_texts)
            # 解码为numpy向量，并按原始顺序对齐
            processed = self.embedding_service.process_embeddings(embedding_results)
            processed.sort(key=lambda x: x.get('index', 0))
            embeddings = [item['embedding'] for item in processed]
            return embeddings
        except Exception as e:
            st.error(f"向量化处理失败: {str(e)}")
            return []
    
    def find_similar_texts(self, user_embeddings: List[np.ndarray], user_texts: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """查找相似文本"""
        similarity_results = {}
        
        for i, (user_embedding, user_text) in enumerate(zip(user_embeddings, user_texts)):
            try:
                # 使用相似度匹配器查找最相似的文本
                similar_items = self.similarity_matcher.find_most_similar(
                    query_embedding=user_embedding,
                    candidate_embeddings=self.base_embeddings,
                    similarity_method=SIMILARITY_METHOD
                )
                
                similarity_results[user_text] = similar_items
                
            except Exception as e:
                st.error(f"处理文本 '{user_text[:50]}...' 时发生错误: {str(e)}")
                similarity_results[user_text] = []
        
        return similarity_results
    
    async def enhance_with_llm(self, similarity_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float, str]]]:
        """使用LLM增强相似度结果"""
        try:
            enhanced_results = await self.llm_service.enhance_similarity_results(similarity_results)
            return enhanced_results
        except Exception as e:
            st.error(f"LLM增强处理失败: {str(e)}")
            # 返回默认增强结果
            enhanced_results = {}
            for query_text, candidates in similarity_results.items():
                enhanced_results[query_text] = [
                    (text, score, "LLM处理失败") for text, score in candidates
                ]
            return enhanced_results
    
    def create_output_dataframe(self, enhanced_results: Dict[str, List[Tuple[str, float, str]]]) -> pd.DataFrame:
        """创建输出DataFrame - 水平排列格式"""
        output_data = []
        
        for query_text, results in enhanced_results.items():
            # 创建一行数据，包含查询题目和最多3个相似题目
            row_data = {OUTPUT_COLUMNS['query']: query_text}
            
            # 填充相似题目和相似度（最多3个）
            for i in range(3):
                similar_key = f'similar_text_{i+1}'
                score_key = f'similarity_score_{i+1}'
                
                if i < len(results):
                    similar_text, similarity_score, _ = results[i]
                    row_data[OUTPUT_COLUMNS[similar_key]] = similar_text
                    row_data[OUTPUT_COLUMNS[score_key]] = round(similarity_score, 4)
                else:
                    # 如果没有足够的相似题目，填充空值
                    row_data[OUTPUT_COLUMNS[similar_key]] = ""
                    row_data[OUTPUT_COLUMNS[score_key]] = ""
            
            output_data.append(row_data)
        
        return pd.DataFrame(output_data)
    
    def create_excel_download(self, df: pd.DataFrame) -> bytes:
        """创建Excel文件用于下载"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='相似题目匹配结果')
            
            # 获取工作表并调整列宽
            worksheet = writer.sheets['相似题目匹配结果']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # 限制最大宽度
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output.getvalue()


def main():
    """主函数"""
    st.set_page_config(
        page_title="相似题目匹配系统",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 相似题目匹配系统")
    st.markdown("---")
    
    # 初始化应用
    if 'app' not in st.session_state:
        st.session_state.app = SimilarityMatchingApp()
    
    app = st.session_state.app
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 显示当前配置
        st.subheader("当前配置")
        st.write(f"**相似度方法**: {SIMILARITY_METHOD}")
        st.write(f"**TOP-K**: {TOP_K_SIMILAR}")
        st.write(f"**相似度阈值**: {SIMILARITY_THRESHOLD}")
        
        st.markdown("---")
        
        # 系统状态
        st.subheader("📊 系统状态")
        
        # 检查基础题库状态
        if app.load_base_embeddings():
            st.success(f"✅ 基础题库已加载 ({len(app.base_texts)} 条)")
            base_loaded = True
        else:
            st.error("❌ 基础题库加载失败")
            base_loaded = False
    
    # 主界面
    if not base_loaded:
        st.error("⚠️ 系统初始化失败，请检查基础题库文件是否存在")
        st.info(f"请确保文件存在: {EMBEDDINGS_CACHE_PATH}")
        return
    
    # 文件上传区域
    st.header("📁 上传新题库")
    
    uploaded_file = st.file_uploader(
        "选择CSV文件 (UTF-8 BOM编码，第一列为题目文本)",
        type=['csv'],
        help="请上传UTF-8 BOM编码的CSV文件，第一列应包含需要匹配的题目文本"
    )
    
    if uploaded_file is not None:
        try:
            # 读取CSV文件
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            if df.empty:
                st.error("上传的文件为空")
                return
            
            # 获取第一列数据
            first_column = df.iloc[:, 0]
            user_texts = first_column.dropna().astype(str).tolist()
            
            if not user_texts:
                st.error("第一列没有有效的文本数据")
                return
            
            st.success(f"✅ 文件上传成功，共读取到 {len(user_texts)} 条题目")
            
            # 显示预览
            with st.expander("📋 数据预览", expanded=False):
                preview_df = pd.DataFrame({
                    '序号': range(1, min(11, len(user_texts) + 1)),
                    '题目文本': user_texts[:10]
                })
                st.dataframe(preview_df, use_container_width=True)
                
                if len(user_texts) > 10:
                    st.info(f"仅显示前10条，总共{len(user_texts)}条")
            
            # 处理按钮
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 开始匹配", type="primary", use_container_width=True):
                    # 创建进度条和状态显示
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # 步骤1: 向量化用户文本
                        status_text.text("🔄 正在向量化用户文本...")
                        progress_bar.progress(20)
                        
                        # 使用asyncio运行异步函数
                        user_embeddings = asyncio.run(app.process_user_texts(user_texts))
                        
                        if not user_embeddings:
                            st.error("向量化失败，请检查网络连接和API配置")
                            return
                        
                        # 步骤2: 查找相似文本
                        status_text.text("🔍 正在查找相似文本...")
                        progress_bar.progress(40)
                        
                        similarity_results = app.find_similar_texts(user_embeddings, user_texts)
                        
                        # 步骤3: LLM增强
                        status_text.text("🤖 正在使用LLM增强结果...")
                        progress_bar.progress(60)
                        
                        enhanced_results = asyncio.run(app.enhance_with_llm(similarity_results))
                        
                        # 步骤4: 生成结果
                        status_text.text("📊 正在生成结果...")
                        progress_bar.progress(80)
                        
                        result_df = app.create_output_dataframe(enhanced_results)
                        
                        # 步骤5: 完成
                        status_text.text("✅ 处理完成！")
                        progress_bar.progress(100)
                        
                        # 显示结果
                        st.markdown("---")
                        st.header("📋 匹配结果")
                        
                        # 结果统计
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("查询题目数", len(user_texts))
                        
                        with col2:
                            # 计算总的匹配结果数（非空的相似题目）
                            total_matches = 0
                            for i in range(1, 4):
                                col_name = OUTPUT_COLUMNS[f'similar_text_{i}']
                                total_matches += len(result_df[result_df[col_name] != ""])
                            st.metric("匹配结果数", total_matches)
                        
                        with col3:
                            # 计算平均相似度（所有非空相似度的平均值）
                            all_scores = []
                            for i in range(1, 4):
                                score_col = OUTPUT_COLUMNS[f'similarity_score_{i}']
                                scores = result_df[result_df[score_col] != ""][score_col]
                                all_scores.extend(scores.tolist())
                            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                            st.metric("平均相似度", f"{avg_score:.3f}")
                        
                        with col4:
                            # 计算高质量匹配数（相似度>=0.7的匹配）
                            high_quality_matches = 0
                            for i in range(1, 4):
                                score_col = OUTPUT_COLUMNS[f'similarity_score_{i}']
                                high_quality_matches += len(result_df[(result_df[score_col] != "") & (result_df[score_col] >= 0.7)])
                            st.metric("高质量匹配", f"{high_quality_matches}")
                        
                        # 显示结果表格
                        st.subheader("详细结果")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # 下载按钮
                        excel_data = app.create_excel_download(result_df)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"相似题目匹配结果_{timestamp}.xlsx"
                        
                        st.download_button(
                            label="📥 下载Excel文件",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True
                        )
                        
                        # 显示token使用统计
                        if VERBOSE:
                            with st.expander("📊 处理统计", expanded=False):
                                token_usage = app.llm_service.get_token_usage()
                                embedding_usage = app.embedding_service.get_token_usage()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("LLM使用统计")
                                    st.write(f"总Token数: {token_usage['total_tokens']}")
                                    st.write(f"请求次数: {token_usage['request_count']}")
                                    st.write(f"平均Token/请求: {token_usage['avg_tokens_per_request']}")
                                
                                with col2:
                                    st.subheader("Embedding使用统计")
                                    st.write(f"总Token数: {embedding_usage['total_tokens']}")
                                    st.write(f"请求次数: {embedding_usage['request_count']}")
                                    st.write(f"平均Token/请求: {embedding_usage['avg_tokens_per_request']}")
                        
                        st.success("🎉 匹配完成！您可以下载结果文件。")
                        
                    except Exception as e:
                        st.error(f"处理过程中发生错误: {str(e)}")
                        import traceback
                        if VERBOSE:
                            st.error(f"详细错误信息: {traceback.format_exc()}")
        
        except Exception as e:
            st.error(f"读取文件时发生错误: {str(e)}")
            st.info("请确保文件是UTF-8 BOM编码的CSV格式")
    
    else:
        st.info("👆 请上传CSV文件开始匹配")
        
        # 显示使用说明
        with st.expander("📖 使用说明", expanded=True):
            st.markdown("""
            ### 📋 使用步骤
            
            1. **准备文件**: 确保您的CSV文件使用UTF-8 BOM编码
            2. **上传文件**: 点击上方的文件上传区域选择您的CSV文件
            3. **检查预览**: 确认第一列包含需要匹配的题目文本
            4. **开始匹配**: 点击"开始匹配"按钮
            5. **下载结果**: 处理完成后下载Excel格式的结果文件
            
            ### ⚙️ 系统特性
             
             - **智能匹配**: 结合向量相似度和LLM语义理解
             - **高效处理**: 支持批量并发处理
             - **水平排列**: 每行显示一个题目及其前3个最相似题目
             - **Excel导出**: 结果以Excel格式提供下载
            
            ### 📊 输出格式
             
             结果文件包含以下列：
             - **题目**: 您上传的原始题目
             - **相似题目1**: 最相似的题目
             - **相似度1**: 对应的相似度分数（0-1之间）
             - **相似题目2**: 第二相似的题目
             - **相似度2**: 对应的相似度分数
             - **相似题目3**: 第三相似的题目
             - **相似度3**: 对应的相似度分数
            """)


if __name__ == "__main__":
    main()