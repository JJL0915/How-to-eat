"""
检索优化模块"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(documents=self.chunks, k=5)

        logger.info("检索器设置完成")

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """按本次查询需要的数量执行 BM25 检索。"""
        original_k = self.bm25_retriever.k
        try:
            self.bm25_retriever.k = k
            return self.bm25_retriever.invoke(query)
        finally:
            self.bm25_retriever.k = original_k

    def _copy_document(self, doc: Document) -> Document:
        """复制文档对象，避免把本次检索分数写回共享文档元数据。"""
        return Document(page_content=doc.page_content, metadata=dict(doc.metadata))

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """

        candidate_k = max(top_k, 5)

        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vectorstore.similarity_search(query, k=candidate_k)
        bm25_docs = self._bm25_search(query, candidate_k)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank(
        self, vector_results: List[Document], bm25_results: List[Document], k: int = 60
    ) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """

        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRf分数
        for rank, doc in enumerate(vector_results, 1):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = self._copy_document(doc)
            # RRF公式: 1/(k+rank)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = hash(doc.page_content)
            doc_objects.setdefault(doc_id, self._copy_document(doc))

            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank}: RRF分数 = {rrf_score:.4f}")

        # 按最终的RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构造最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            # 将RRF分数添加到文档数据中
            doc.metadata["rrf_score"] = final_score
            reranked_docs.append(doc)
            logger.debug(
                f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}"
            )

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_results)}个文档, BM25检索{len(bm25_results)}个文档, 合并后{len(reranked_docs)}个文档"
        )

        return reranked_docs

    def preference_aware_search(
        self,
        query: str,
        preferences: Dict[str, Any] | None = None,
        filters: Dict[str, Any] | None = None,
        top_k: int = 3,
        candidate_k: int | None = None,
    ) -> List[Document]:
        """
        带偏好重排的检索。

        这里不会因为难度或分类直接丢弃候选，而是在扩大召回后加权重排，避免“简单”
        这类词把结果过早过滤空。
        """
        preferences = preferences or {}
        filters = filters or {}
        candidate_k = candidate_k or max(top_k * 8, 20)

        docs = self.hybrid_search(query, candidate_k)
        scored_docs = []

        for rank, doc in enumerate(docs):
            base_score = float(doc.metadata.get("rrf_score") or 0.0)
            filter_score = self._metadata_preference_score(doc, filters)
            preference_score = self._query_preference_score(doc, preferences)

            # 轻微保留原始排序，避免同分时结果抖动。
            total_score = base_score + filter_score + preference_score - rank * 0.000001
            doc.metadata["filter_preference_score"] = filter_score
            doc.metadata["query_preference_score"] = preference_score
            doc.metadata["final_retrieval_score"] = total_score
            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def _metadata_preference_score(
        self, doc: Document, filters: Dict[str, Any]
    ) -> float:
        """把明确元数据条件转成排序加分，而不是直接硬过滤。"""
        if not filters:
            return 0.0

        score = 0.0
        for key, value in filters.items():
            doc_value = doc.metadata.get(key)
            if isinstance(value, list):
                if doc_value in value:
                    score += 0.035
            elif doc_value == value:
                score += 0.035
        return score

    def _query_preference_score(
        self, doc: Document, preferences: Dict[str, Any]
    ) -> float:
        """根据查询偏好给候选文档加权。"""
        score = 0.0
        metadata = doc.metadata
        dish_name = metadata.get("dish_name", "")
        category = metadata.get("category", "")
        difficulty = metadata.get("difficulty", "")
        content = doc.page_content or ""
        searchable_text = f"{dish_name}\n{category}\n{difficulty}\n{content}"
        query_text = preferences.get("_query_text", "")
        focus_dish = preferences.get("focus_dish", "")
        target_dishes = preferences.get("target_dishes", []) or []

        if target_dishes:
            for target_dish in target_dishes:
                if dish_name == target_dish:
                    score += 0.15
                elif target_dish in dish_name or dish_name in target_dish:
                    score += 0.1

        if focus_dish:
            if dish_name == focus_dish:
                score += 0.18
            elif focus_dish in dish_name or dish_name in focus_dish:
                score += 0.12

        if len(dish_name) >= 3 and dish_name in query_text:
            score += 0.08

        if category in preferences.get("categories", []):
            score += 0.03

        difficulty_scores = preferences.get("difficulty_scores", {})
        score += float(difficulty_scores.get(difficulty, 0.0))

        for term in preferences.get("positive_terms", []):
            if term in dish_name:
                score += 0.025
            elif term in searchable_text:
                score += 0.008

        for term in preferences.get("negative_terms", []):
            if term in dish_name:
                score -= 0.04
            elif term in searchable_text:
                score -= 0.015

        return score

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """
        带元数据过滤的检索

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """

        # 先进行混合检索  多返回几条后面过滤
        docs = self.hybrid_search(query, top_k * 3)

        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs
