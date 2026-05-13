"""
索引构建模块
"""

import logging
from typing import Iterator, List, Set
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""

        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引

        Args:
            chunks: 文档块列表

        Returns:
            FAISS向量存储对象
        """
        logger.info(f"正在构建FAISS向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks, embedding=self.embeddings
        )

        logging.info(f"向量索引构建完成，包含{len(chunks)}个向量")
        return self.vectorstore

    def save_index(self):
        """保存向量索引到配置的路径"""
        logger.info(f"正在保存索引到本地...")

        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到:{self.index_save_path}")

    def load_index(self) -> FAISS:
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        logger.info(f"正在加载向量索引")

        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.warning(f"索引路径不存在:{self.index_save_path}")
            return None
        try:

            self.vectorstore = FAISS.load_local(
                folder_path=self.index_save_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"向量索引已从{self.index_save_path}加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败:{e}")
            return None

    def get_indexed_parent_ids(self) -> Set[str]:
        """获取当前 FAISS 索引中已存在的父文档 ID 集合。"""
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        parent_ids: Set[str] = set()
        for doc in self._iter_indexed_documents():
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_ids.add(parent_id)

        return parent_ids

    def _iter_indexed_documents(self) -> Iterator[Document]:
        """遍历 FAISS docstore 中的文档对象。"""
        if not self.vectorstore:
            return

        docstore = getattr(self.vectorstore, "docstore", None)
        if not docstore:
            return

        index_to_docstore_id = getattr(self.vectorstore, "index_to_docstore_id", {})
        if index_to_docstore_id and hasattr(docstore, "search"):
            for docstore_id in index_to_docstore_id.values():
                doc = docstore.search(docstore_id)
                if isinstance(doc, Document):
                    yield doc
            return

        docstore_dict = getattr(docstore, "_dict", None)
        if isinstance(docstore_dict, dict):
            for doc in docstore_dict.values():
                if isinstance(doc, Document):
                    yield doc

    def add_documents(self, new_chunks: List[Document]) -> int:
        """
        向现有索引添加新文档

        Args:
            new_chunks: 新的文档块列表

        Returns:
            实际添加到索引的文档块数量
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        if not new_chunks:
            logger.info("没有新文档需要添加到索引")
            return 0

        try:
            logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
            self.vectorstore.add_documents(new_chunks)
            logger.info("新文档添加完成")
            return len(new_chunks)
        except Exception as e:
            logger.warning(f"添加新文档到索引失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)
