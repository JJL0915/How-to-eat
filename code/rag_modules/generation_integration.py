"""
生成集成模块
"""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

BASIC_ANSWER_PROMPT = """
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

最近对话:
{conversation_context}

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。
最近对话只用于理解用户追问，不得当作菜谱事实；回答必须以相关食谱信息为依据。

回答:"""

DIRECT_ANSWER_PROMPT = """
你是一位专业的烹饪助手。用户正在追问某一道菜的具体信息，请根据食谱信息直接回答。

最近对话和当前焦点:
{conversation_context}

用户问题: {question}

相关食谱信息:
{context}

回答要求:
- 只回答用户这次问到的点，不要输出菜品介绍、完整食材表、完整制作步骤或制作技巧。
- 如果用户在比较多道菜，请分别依据各自食谱信息比较用户问到的维度，并给出明确结论。
- 如果用户问温度和时间，直接给出温度和时间；如果有预热、烘烤等多个阶段，按阶段简短列出。
- 最多 3 条要点或 3 句话。
- 严格依据相关食谱信息回答；原文没有明确说明时，直接说“食谱中没有明确说明”。

回答:"""

STEP_BY_STEP_ANSWER_PROMPT = """
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

最近对话:
{conversation_context}

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 严格依据“相关食谱信息”回答，不要补充上下文中没有的营养评价、适宜人群、食材、用量、时间或技巧
- 最近对话只用于理解用户追问，不得当作菜谱事实
- 优先使用最相关的菜谱信息，不要把其他相似菜谱的做法混入当前菜品
- 原文没有的信息请省略；原文标注为可选的内容，回答中也要明确标注为可选
- 重点突出实用性和可操作性，但不要为了完整结构强行填充内容

回答:"""


class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: str = "",
        base_url: str = "",
    ):
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            api_key: OpenAI 兼容接口 API 密钥
            base_url: OpenAI 兼容接口地址
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM:{self.model_name}")

        if not self.model_name:
            raise ValueError("请在项目根目录的 .env 中设置 LLM_MODEL")
        if not self.api_key:
            raise ValueError("请在项目根目录的 .env 中设置 LLM_API_KEY")
        if not self.base_url:
            raise ValueError("请在项目根目录的 .env 中设置 LLM_BASE_URL")
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.base_url,
            )

            logger.info("LLM初始化完成")
        except Exception as e:
            logger.warning(f"LLM初始化失败:{e}")

    def plan_query_with_memory(
        self,
        query: str,
        conversation_context: str = "",
        conversation_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        让 LLM 基于会话记忆规划本轮查询。

        返回结构化计划，代码只按计划执行检索和生成，不用关键词规则预判用户意图。
        """
        conversation_state = conversation_state or {}
        state_text = self._format_conversation_state(conversation_state)
        prompt = ChatPromptTemplate.from_template("""
你是中文菜谱 RAG 系统的对话理解与查询规划器。请结合最近对话和结构化会话状态，理解用户当前问题，并输出 JSON。

结构化会话状态:
{state_text}

最近对话:
{conversation_context}

当前用户问题:
{query}

你的任务:
1. 判断当前问题是否包含“这个、那个、它、刚才、上一个、第几个”等指代，并结合会话状态和最近对话消解指代。
2. 如果用户在比较多个菜品，standalone_query 必须包含所有被比较的菜名。
3. 判断本轮应该如何回答：
   - list: 用户要推荐或列出菜品
   - detail: 用户要某道菜的完整做法、步骤、食材或调料
   - general: 用户问技巧、比较、选择、局部信息或开放问题
4. 判断答案风格：
   - list: 菜名列表
   - step_by_step: 完整做法/步骤
   - direct: 只问局部信息，例如多久、多少度、多少克、需要什么调料
   - compare: 比较多个菜品，例如哪个更简单、区别是什么
   - basic: 其他一般回答
5. 如果无法判断“这个/那个”指哪一道菜，设置 needs_clarification=true 并写出 clarification。

请只返回 JSON，不要输出解释。格式如下:
{{
  "standalone_query": "消解指代后的完整问题",
  "route_type": "list/detail/general",
  "answer_style": "list/step_by_step/direct/compare/basic",
  "target_dishes": ["问题涉及的菜名，按用户意图排序"],
  "focus_dish": "如果只有一个明确焦点菜品则填写，否则为空字符串",
  "needs_clarification": false,
  "clarification": ""
}}

注意:
- 不要编造会话中没有出现、用户也没有提到的菜名。
- 如果用户说“这个和 X 哪个简单”，其中“这个”应解析为会话焦点菜品，并把两个菜名都放入 target_dishes。
- 如果当前问题已经完整，standalone_query 可以等于原问题。
""")

        chain = (
            {
                "query": lambda _: query,
                "conversation_context": lambda _: conversation_context or "无",
                "state_text": lambda _: state_text,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        raw_response = chain.invoke(query).strip()
        return self._parse_query_plan(raw_response, query)

    def _format_conversation_state(self, conversation_state: Dict[str, Any]) -> str:
        """格式化结构化会话状态。"""
        if not conversation_state:
            return "无"

        lines = []
        focus_dish = conversation_state.get("focus_dish")
        candidate_dishes = conversation_state.get("candidate_dishes") or []
        last_route_type = conversation_state.get("last_route_type")
        if focus_dish:
            lines.append(f"当前焦点菜品: {focus_dish}")
        if candidate_dishes:
            lines.append(f"上轮候选菜品: {'、'.join(candidate_dishes)}")
        if last_route_type:
            lines.append(f"上一轮问题类型: {last_route_type}")
        return "\n".join(lines) if lines else "无"

    def _parse_query_plan(
        self, raw_response: str, original_query: str
    ) -> Dict[str, Any]:
        """解析 LLM 返回的查询计划，失败时给出保守兜底。"""
        try:
            json_text = self._extract_json_object(raw_response)
            plan = json.loads(json_text)
        except Exception as exc:
            logger.warning(f"查询规划 JSON 解析失败: {exc}; 原始输出: {raw_response}")
            return self._fallback_query_plan(original_query)

        route_type = str(plan.get("route_type", "general")).strip().lower()
        if route_type not in {"list", "detail", "general"}:
            route_type = "general"

        answer_style = str(plan.get("answer_style", "basic")).strip().lower()
        if answer_style not in {"list", "step_by_step", "direct", "compare", "basic"}:
            answer_style = "basic"

        target_dishes = plan.get("target_dishes") or []
        if not isinstance(target_dishes, list):
            target_dishes = []
        target_dishes = [
            str(dish).strip() for dish in target_dishes if str(dish).strip()
        ]

        needs_clarification = bool(plan.get("needs_clarification", False))
        clarification = str(plan.get("clarification", "")).strip()
        standalone_query = (
            str(plan.get("standalone_query", "")).strip() or original_query
        )
        focus_dish = str(plan.get("focus_dish", "")).strip()

        return {
            "standalone_query": standalone_query,
            "route_type": route_type,
            "answer_style": answer_style,
            "target_dishes": target_dishes,
            "focus_dish": focus_dish,
            "needs_clarification": needs_clarification,
            "clarification": clarification,
        }

    def _extract_json_object(self, text: str) -> str:
        """从模型输出中提取 JSON 对象。"""
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
        if fenced:
            return fenced.group(1)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _fallback_query_plan(self, query: str) -> Dict[str, Any]:
        """查询规划失败时的保守兜底。"""
        return {
            "standalone_query": query,
            "route_type": "general",
            "answer_style": "basic",
            "target_dishes": [],
            "focus_dish": "",
            "needs_clarification": False,
            "clarification": "",
        }

    def _build_answer_chain(
        self,
        prompt_text: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ):
        """构造回答生成链，统一流式和非流式回答的 prompt。"""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(prompt_text)
        return (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "conversation_context": lambda _: conversation_context or "无",
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_basic_answer(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        chain = self._build_answer_chain(
            BASIC_ANSWER_PROMPT, context_docs, conversation_context
        )
        return chain.invoke(query)

    def generate_direct_answer(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ) -> str:
        """
        生成焦点追问的直接回答。

        适用于用户只问温度、时间、调料、用量等局部信息的场景。
        """
        chain = self._build_answer_chain(
            DIRECT_ANSWER_PROMPT, context_docs, conversation_context
        )
        return chain.invoke(query)

    def generate_step_by_step_answer(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        chain = self._build_answer_chain(
            STEP_BY_STEP_ANSWER_PROMPT, context_docs, conversation_context
        )
        return chain.invoke(query)

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """

        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 查询菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为你推荐:{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join(
                [f"{i+1}. {name}" for i, name in enumerate(dish_names)]
            )
        else:
            return (
                f"为您推荐以下菜品：\n"
                + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])])
                + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"
            )

    def generate_basic_answer_stream(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        chain = self._build_answer_chain(
            BASIC_ANSWER_PROMPT, context_docs, conversation_context
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_direct_answer_stream(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ):
        """
        生成焦点追问的直接回答 - 流式输出。
        """
        chain = self._build_answer_chain(
            DIRECT_ANSWER_PROMPT, context_docs, conversation_context
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(
        self,
        query: str,
        context_docs: List[Document],
        conversation_context: str = "",
    ):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        chain = self._build_answer_chain(
            STEP_BY_STEP_ANSWER_PROMPT, context_docs, conversation_context
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 6000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if "dish_name" in doc.metadata:
                metadata_info += f"{doc.metadata['dish_name']}"
            if "category" in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if "difficulty" in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"

            content = doc.page_content.strip()
            doc_header = f"{metadata_info}\n"
            remaining_length = max_length - current_length
            if remaining_length <= len(doc_header):
                break

            # 单篇父文档可能很长，保留排序靠前的内容并截断，而不是直接丢弃整篇文档。
            available_content_length = remaining_length - len(doc_header)
            truncated = False
            if len(content) > available_content_length:
                content = content[:available_content_length].rstrip()
                truncated = True

            # 构建文档文本
            doc_text = f"{doc_header}{content}"
            if truncated:
                doc_text += "\n...[内容已截断]"
            doc_text += "\n"

            context_parts.append(doc_text)
            current_length += len(doc_text)

        if not context_parts:
            return "暂无相关食谱信息"

        separator = "\n" + "=" * 50 + "\n"
        return separator.join(context_parts)
