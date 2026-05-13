"""
RAG系统主程序
"""

import sys
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
    ConversationMemory,
    QueryPreferenceExtractor,
    MenuSafetyGuard,
    RAGPipelineResult,
)

# .env 会在 config.py 导入时自动加载

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retriever_module = None
        self.generation_module = None
        self.conversation_memory = ConversationMemory(max_turns=6)
        self.preference_extractor = QueryPreferenceExtractor()
        self.menu_safety = MenuSafetyGuard()

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise ValueError(f"数据路径不存在:{self.config.data_path}")

        # 检查API密钥
        if not self.config.llm_api_key:
            raise ValueError("请在项目根目录的 .env 中设置 LLM_API_KEY")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1.初始化数据模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2.初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        # 3.初始化生成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1.尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
            self._sync_new_recipes_to_index(chunks)
            vectorstore = self.index_module.vectorstore
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2.加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3.文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4.构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5.保存索引
            print("保存向量索引")
            self.index_module.save_index()

        # 6.初始化检索模块
        print("初始化检索优化...")
        self.retriever_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def _sync_new_recipes_to_index(self, chunks: List[Any]) -> tuple[int, int]:
        """
        将新增菜谱增量追加到已加载的向量索引。

        只处理新增文件：如果已有菜谱内容被修改或删除，本方法不会更新旧向量。
        """
        indexed_parent_ids = self.index_module.get_indexed_parent_ids()
        if not indexed_parent_ids:
            print(
                "⚠️ 已加载的向量索引中没有 parent_id 元数据，跳过自动增量同步。"
                "如需同步新增菜谱，请删除 vector_index 后全量重建一次。"
            )
            return 0, 0

        current_parent_ids = {
            chunk.metadata.get("parent_id")
            for chunk in chunks
            if chunk.metadata.get("parent_id")
        }
        new_parent_ids = current_parent_ids - indexed_parent_ids

        if not new_parent_ids:
            print("未发现新增菜谱，跳过增量索引。")
            return 0, 0

        new_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("parent_id") in new_parent_ids
        ]
        new_dish_names = sorted(
            {
                chunk.metadata.get("dish_name", "未知菜品")
                for chunk in new_chunks
            }
        )
        preview_names = "、".join(new_dish_names[:5])
        if len(new_dish_names) > 5:
            preview_names += f" 等 {len(new_dish_names)} 道"

        print(
            f"发现 {len(new_parent_ids)} 篇新增菜谱"
            f"（{preview_names}），准备追加 {len(new_chunks)} 个文档块到向量索引..."
        )
        self.index_module.add_documents(new_chunks)
        print("保存增量向量索引...")
        self.index_module.save_index()
        print("✅ 增量索引同步完成！")
        return len(new_parent_ids), len(new_chunks)

    def ask_question(
        self,
        question: str,
        stream: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        result = self.run_question_pipeline(
            question,
            stream=stream,
            session_id=session_id,
            remember=True,
            verbose=True,
        )
        if stream and result.answer_stream is not None:
            return result.answer_stream
        return result.answer

    def run_question_pipeline(
        self,
        question: str,
        stream: bool = False,
        session_id: Optional[str] = None,
        *,
        remember: bool = True,
        verbose: bool = True,
    ) -> RAGPipelineResult:
        """执行当前生产问答链路，并返回可供评测复用的结构化结果。"""
        if not all([self.retriever_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        result = RAGPipelineResult(question=question)
        log = print if verbose else (lambda *args, **kwargs: None)

        log(f"\n❓ 用户问题: {question}")
        result.conversation_context = self._get_conversation_context(session_id)
        result.conversation_state = self._get_conversation_state(session_id)
        result.query_plan = self.generation_module.plan_query_with_memory(
            question,
            conversation_context=result.conversation_context,
            conversation_state=result.conversation_state,
        )
        result.processing_question = (
            result.query_plan.get("standalone_query") or question
        )
        result.route_type = result.query_plan.get("route_type", "general")
        result.answer_style = result.query_plan.get("answer_style", "basic")
        result.target_dishes = result.query_plan.get("target_dishes", [])
        result.focus_dish = result.query_plan.get("focus_dish", "")

        if result.query_plan.get("needs_clarification"):
            result.answer = result.query_plan.get("clarification") or "你指的是哪一道菜？"
            if remember:
                self._remember_answer(session_id, question, result.answer)
            return result

        is_comparison_query = (
            result.answer_style == "compare" or len(result.target_dishes) > 1
        )
        if result.processing_question != question:
            log(f"🧠 LLM 规划改写为: {result.processing_question}")
        log(f"🎯 查询类型: {result.route_type} | 回答风格: {result.answer_style}")
        if result.focus_dish:
            log(f"🎯 当前会话焦点菜品: {result.focus_dish}")
        if result.target_dishes:
            log(f"🎯 当前问题目标菜品: {', '.join(result.target_dishes)}")

        # 3.检索相关子块，并应用偏好重排
        log("🔍 检索相关文档...")
        (
            result.relevant_chunks,
            result.filters,
            result.preferences,
        ) = self._retrieve_relevant_chunks(
            result.processing_question,
            result.processing_question,
            focus_dish=result.focus_dish,
            target_dishes=result.target_dishes,
        )
        if result.filters:
            log(f"应用过滤条件: {result.filters}")
        if result.preferences:
            log(f"应用排序偏好: {result.preferences}")

        # 显示检索到的子块信息
        self._log_retrieved_chunks(result.relevant_chunks, verbose=verbose)

        # 4.检查是否找到相关内容
        if not result.relevant_chunks:
            result.answer = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            if remember:
                self._remember_answer(session_id, question, result.answer)
            return result

        # 5.根据路由类型选择回答方式
        if result.route_type == "list" or result.answer_style == "list":
            # 列表查询：直接返回菜品名称列表
            log("📋 生成菜品列表...")
            result.relevant_docs = self.data_module.get_parent_documents(
                result.relevant_chunks
            )
            self._log_parent_docs(result.relevant_docs, verbose=verbose, separator="：")

            if remember:
                self._update_conversation_focus(
                    session_id,
                    route_type=result.route_type,
                    relevant_docs=result.relevant_docs,
                    focus_dish="",
                    target_dishes=result.target_dishes,
                    comparison_query=is_comparison_query,
                )
            result.answer = self.generation_module.generate_list_answer(
                result.processing_question, result.relevant_docs
            )
            if remember:
                self._remember_answer(session_id, question, result.answer)
            return result
        else:
            # 详细查询:获取完整文档并生成详细回答
            log("获取完整文档...")
            result.relevant_docs = self.data_module.get_parent_documents(
                result.relevant_chunks
            )
            self._log_parent_docs(result.relevant_docs, verbose=verbose, separator=": ")

            if result.focus_dish and not is_comparison_query:
                result.relevant_docs = self._narrow_docs_to_focus(
                    result.relevant_docs, result.focus_dish
                )

            if result.route_type == "detail":
                refusal = self.menu_safety.build_out_of_menu_refusal(
                    result.processing_question,
                    result.relevant_docs,
                    self._get_known_dish_names(),
                )
                if refusal:
                    log("未命中本地菜单中的目标菜品，拒绝编造做法。")
                    result.refusal_reason = "目标菜品不在本地菜单中"
                    result.answer = refusal
                    if remember:
                        self._remember_answer(session_id, question, result.answer)
                    return result

            log("✍️ 生成详细回答...")
            if remember:
                self._update_conversation_focus(
                    session_id,
                    route_type=result.route_type,
                    relevant_docs=result.relevant_docs,
                    focus_dish=result.focus_dish,
                    target_dishes=result.target_dishes,
                    comparison_query=is_comparison_query,
                )
            result.generation_context = self._build_generation_context(
                result.conversation_context, result.focus_dish, result.target_dishes
            )
            if result.answer_style in {"direct", "compare"}:
                log("✍️ 按 LLM 规划生成直接/对比回答...")
                if stream:
                    result.answer_stream = (
                        self.generation_module.generate_direct_answer_stream(
                            result.processing_question,
                            result.relevant_docs,
                            conversation_context=result.generation_context,
                        )
                    )
                    if remember:
                        result.answer_stream = self._remember_stream(
                            session_id, question, result.answer_stream
                        )
                    return result

                result.answer = self.generation_module.generate_direct_answer(
                    result.processing_question,
                    result.relevant_docs,
                    conversation_context=result.generation_context,
                )
                if remember:
                    self._remember_answer(session_id, question, result.answer)
                return result

            # 根据 LLM 规划选择回答模式
            if result.answer_style == "step_by_step" or result.route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    result.answer_stream = (
                        self.generation_module.generate_step_by_step_answer_stream(
                            result.processing_question,
                            result.relevant_docs,
                            conversation_context=result.generation_context,
                        )
                    )
                    if remember:
                        result.answer_stream = self._remember_stream(
                            session_id, question, result.answer_stream
                        )
                    return result

                result.answer = self.generation_module.generate_step_by_step_answer(
                    result.processing_question,
                    result.relevant_docs,
                    conversation_context=result.generation_context,
                )
                if remember:
                    self._remember_answer(session_id, question, result.answer)
                return result
            else:
                # 一般查询使用基础回答模式
                if stream:
                    result.answer_stream = self.generation_module.generate_basic_answer_stream(
                        result.processing_question,
                        result.relevant_docs,
                        conversation_context=result.generation_context,
                    )
                    if remember:
                        result.answer_stream = self._remember_stream(
                            session_id, question, result.answer_stream
                        )
                    return result

                result.answer = self.generation_module.generate_basic_answer(
                    result.processing_question,
                    result.relevant_docs,
                    conversation_context=result.generation_context,
                )
                if remember:
                    self._remember_answer(session_id, question, result.answer)
                return result

    @staticmethod
    def empty_pipeline_result(question: str, answer: str = "") -> RAGPipelineResult:
        """构造失败兜底结果，避免评测脚本复制流水线字段默认值。"""
        return RAGPipelineResult(
            question=question,
            answer=answer,
            processing_question=question,
            route_type="",
            answer_style="",
        )

    def _log_retrieved_chunks(self, relevant_chunks: List[Any], verbose: bool) -> None:
        """输出检索到的子块摘要。"""
        if not verbose:
            return
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith("#"):
                    title_end = (
                        content_preview.find("\n")
                        if "\n" in content_preview
                        else len(content_preview)
                    )
                    section_title = content_preview[:title_end].replace("#", "").strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

    def _log_parent_docs(
        self, relevant_docs: List[Any], verbose: bool, separator: str
    ) -> None:
        """输出父文档摘要。"""
        if not verbose:
            return
        doc_names = [
            doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs
        ]
        if doc_names:
            print(f"找到文档{separator}{', '.join(doc_names)}")
        else:
            print(f"对应 {len(relevant_docs)} 个完整文档")

    def _get_conversation_context(self, session_id: Optional[str]) -> str:
        """读取当前会话最近对话；未提供 session_id 时保持单轮行为。"""
        if not session_id:
            return ""
        return self.conversation_memory.format_history(session_id)

    def _get_conversation_state(self, session_id: Optional[str]) -> Dict[str, Any]:
        """读取当前会话结构化状态；未提供 session_id 时保持空状态。"""
        if not session_id:
            return {}
        return self.conversation_memory.get_state(session_id)

    def _build_generation_context(
        self,
        conversation_context: str,
        focus_dish: str,
        target_dishes: List[str] | None = None,
    ) -> str:
        """为生成阶段组织包含焦点菜品的上下文说明。"""
        target_dishes = target_dishes or []
        if len(target_dishes) > 1:
            target_context = (
                f"当前问题目标菜品：{'、'.join(target_dishes)}\n"
                "用户正在比较这些菜品。请基于相关食谱信息对比用户问到的维度，"
                "不要只分析其中一道菜。"
            )
            if not conversation_context:
                return target_context
            return f"{target_context}\n{conversation_context}"

        if not focus_dish:
            return conversation_context

        focus_context = (
            f"当前焦点菜品：{focus_dish}\n"
            "用户正在追问这道菜。请优先只回答该菜品相关信息，"
            "不要并列展开其他候选菜谱。"
        )
        if not conversation_context:
            return focus_context
        return f"{focus_context}\n{conversation_context}"

    def _narrow_docs_to_focus(
        self, relevant_docs: List[Any], focus_dish: str
    ) -> List[Any]:
        """追问命中焦点菜品时，把生成上下文收窄到该菜品。"""
        if not focus_dish:
            return relevant_docs
        focus_docs = [
            doc
            for doc in relevant_docs
            if self._dish_matches_focus(doc.metadata.get("dish_name", ""), focus_dish)
        ]
        return focus_docs or relevant_docs

    def _update_conversation_focus(
        self,
        session_id: Optional[str],
        *,
        route_type: str,
        relevant_docs: List[Any],
        focus_dish: str = "",
        target_dishes: List[str] | None = None,
        comparison_query: bool = False,
    ) -> None:
        """根据本轮检索结果更新会话焦点状态。"""
        if not session_id:
            return

        target_dishes = target_dishes or []
        candidate_dishes = target_dishes or self._extract_unique_dish_names(relevant_docs)
        next_focus: Optional[str] = None
        if comparison_query or len(candidate_dishes) > 1 and target_dishes:
            next_focus = ""
        elif route_type == "list":
            next_focus = ""
        elif route_type == "detail":
            next_focus = focus_dish or (candidate_dishes[0] if candidate_dishes else "")
        elif focus_dish:
            next_focus = focus_dish

        self.conversation_memory.update_focus(
            session_id,
            focus_dish=next_focus,
            candidate_dishes=candidate_dishes,
            route_type=route_type,
        )

    def _extract_unique_dish_names(self, docs: List[Any]) -> List[str]:
        """从文档列表中提取去重菜名，并保持顺序。"""
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "")
            if dish_name and dish_name not in dish_names:
                dish_names.append(dish_name)
        return dish_names

    def _dish_matches_focus(self, dish_name: str, focus_dish: str) -> bool:
        """判断文档菜名是否匹配当前焦点菜品。"""
        if not dish_name or not focus_dish:
            return False
        return (
            dish_name == focus_dish
            or focus_dish in dish_name
            or dish_name in focus_dish
        )

    def _remember_answer(
        self, session_id: Optional[str], question: str, answer: str
    ) -> None:
        """在启用 session_id 时保存一轮完整问答。"""
        if not session_id:
            return
        self.conversation_memory.add_turn(session_id, question, answer)

    def _remember_stream(
        self,
        session_id: Optional[str],
        question: str,
        chunks: Iterable[str],
    ) -> Iterable[str]:
        """包装流式回答，结束后保存完整回答到会话记忆。"""
        if not session_id:
            return chunks

        def generate_with_memory():
            answer_chunks = []
            for chunk in chunks:
                answer_chunks.append(chunk)
                yield chunk
            answer = "".join(answer_chunks)
            self._remember_answer(session_id, question, answer)

        return generate_with_memory()

    def _get_known_dish_names(self) -> List[str]:
        """获取本地父文档中的菜名列表。"""
        if not self.data_module or not getattr(self.data_module, "documents", None):
            return []
        return [
            doc.metadata.get("dish_name", "")
            for doc in self.data_module.documents
            if doc.metadata.get("dish_name")
        ]

    def _retrieve_relevant_chunks(
        self,
        rewritten_query: str,
        original_query: str,
        focus_dish: str = "",
        target_dishes: List[str] | None = None,
    ) -> tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        """
        根据原始问题抽取偏好，并用改写后的问题执行检索。

        难度、做法等信息只参与排序，不再作为硬过滤条件。
        """
        filters = self.preference_extractor.extract_filters(original_query)
        preferences = self.preference_extractor.extract_preferences(original_query)
        if focus_dish:
            preferences["focus_dish"] = focus_dish
        if target_dishes:
            preferences["target_dishes"] = target_dishes

        if filters or preferences:
            search_preferences = dict(preferences)
            search_preferences["_query_text"] = original_query
            relevant_chunks = self.retriever_module.preference_aware_search(
                rewritten_query,
                preferences=search_preferences,
                filters=filters,
                top_k=self.config.top_k,
            )
        else:
            relevant_chunks = self.retriever_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        return relevant_chunks, filters, preferences

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        if not self.data_module or not getattr(self.data_module, "documents", None):
            raise ValueError("请先构建知识库")

        query = query.strip()
        dish_names: List[str] = []
        for doc in self.data_module.documents:
            if doc.metadata.get("category") != category:
                continue
            dish_name = doc.metadata.get("dish_name", "")
            if query and query not in dish_name and query not in doc.page_content:
                continue
            if dish_name and dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not self.data_module or not getattr(self.data_module, "documents", None):
            raise ValueError("请先构建知识库")

        for doc in self.data_module.documents:
            current_dish_name = doc.metadata.get("dish_name", "")
            if not self._dish_matches_focus(current_dish_name, dish_name):
                continue

            content = doc.page_content
            match = re.search(
                r"##\s*(?:🛒\s*)?(?:所需食材|食材|原料|配料)[\s\S]*?(?=\n##\s|\Z)",
                content,
            )
            if match:
                return match.group(0).strip()
            return f"找到「{current_dish_name}」，但菜谱中没有明确的食材章节。"

        return f"未找到「{dish_name}」的本地菜谱。"

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ["退出", "quit", "exit", ""]:
                    break

                # 询问是否使用流式输出
                stream_choice = (
                    input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                )
                use_stream = stream_choice != "n"

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用尝尝咸淡RAG系统！")
