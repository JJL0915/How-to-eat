"""
菜单安全校验模块。

负责判断 detail 类问题是否命中了本地菜单，避免用相似菜谱编造不存在菜品的做法。
"""

from __future__ import annotations

from typing import Any, Iterable, List


class MenuSafetyGuard:
    """菜单外菜品拒答策略。"""

    def build_out_of_menu_refusal(
        self,
        query: str,
        relevant_docs: List[Any],
        known_dish_names: Iterable[str],
    ) -> str:
        """
        对明确菜名的 detail 问题做菜单外拒答。

        如果用户问的是本地菜单没有的具体菜品，不用相似菜谱编造完整做法。
        """
        target_dish = self._extract_detail_target_dish(query)
        if not target_dish:
            return ""

        retrieved_dish_names = [
            doc.metadata.get("dish_name", "")
            for doc in relevant_docs
            if doc.metadata.get("dish_name")
        ]
        if self._has_supported_dish_match(
            target_dish,
            [*known_dish_names, *retrieved_dish_names],
        ):
            return ""

        return (
            f"抱歉，本地菜谱库里没有找到「{target_dish}」的可靠菜谱信息，"
            "我不能直接编造它的做法。你可以换一个已有菜名再问，"
            "或者让我推荐相近的本地菜谱。"
        )

    def _extract_detail_target_dish(self, query: str) -> str:
        """从明确做法类问题中抽取用户想问的目标菜名。"""
        cleaned_query = self._normalize_query_text(query)
        suffix_patterns = [
            "需要哪些调料",
            "需要什么调料",
            "需要哪些食材",
            "需要什么食材",
            "用什么调料",
            "放什么调料",
            "怎么制作",
            "如何制作",
            "制作方法",
            "制作步骤",
            "怎么做",
            "怎样做",
            "如何做",
            "做法",
        ]
        prefix_patterns = ["怎么做", "怎样做", "如何做", "怎么制作", "如何制作"]

        for pattern in suffix_patterns:
            if pattern in cleaned_query:
                candidate = cleaned_query.split(pattern, 1)[0]
                candidate = self._clean_dish_candidate(candidate)
                if candidate:
                    return candidate

        for pattern in prefix_patterns:
            if cleaned_query.startswith(pattern):
                candidate = cleaned_query[len(pattern) :]
                candidate = self._clean_dish_candidate(candidate)
                if candidate:
                    return candidate

        return ""

    def _normalize_query_text(self, query: str) -> str:
        """清理问题文本中不影响菜名判断的标点和口语前缀。"""
        cleaned = query.strip()
        for mark in ["？", "?", "。", "！", "!", "，", ","]:
            cleaned = cleaned.replace(mark, "")
        for prefix in ["请问", "我想问一下", "想问一下", "帮我看看"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        return cleaned.strip()

    def _clean_dish_candidate(self, candidate: str) -> str:
        """清理抽取出的菜名候选。"""
        cleaned = candidate.strip()
        for prefix in ["我想做", "想做", "我要做", "要做", "做一道", "做一个"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        for suffix in [
            "完整制作步骤",
            "详细制作步骤",
            "完整步骤",
            "详细步骤",
            "制作步骤",
            "完整做法",
            "详细做法",
            "这道菜",
            "这个菜",
            "这菜",
            "步骤",
            "做法",
        ]:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        return cleaned.strip()

    def _has_supported_dish_match(
        self, target_dish: str, candidate_names: Iterable[str]
    ) -> bool:
        """
        判断目标菜名是否被本地菜单支持。

        允许“红烧肉”匹配“简易红烧肉”这类更具体的本地菜名，但不允许
        “惠灵顿牛排”只因为包含“牛排”就通过。
        """
        for dish_name in candidate_names:
            if not dish_name:
                continue
            if target_dish == dish_name:
                return True
            if len(target_dish) >= 3 and target_dish in dish_name:
                return True

        return False
