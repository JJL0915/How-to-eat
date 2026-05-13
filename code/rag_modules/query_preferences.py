"""
查询偏好抽取模块。

这里的规则只用于检索排序和少量高置信过滤，不承担用户意图路由。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .data_preparation import DataPreparationModule


class QueryPreferenceExtractor:
    """从用户问题中抽取检索阶段使用的过滤条件和排序偏好。"""

    CATEGORY_ALIASES = {
        "荤菜": ["荤菜", "肉菜", "肉类", "鸡肉", "猪肉", "牛肉", "羊肉", "排骨"],
        "素菜": ["素菜", "蔬菜", "青菜", "不吃肉"],
        "汤品": ["汤品", "做汤", "喝汤", "汤", "羹"],
        "甜品": ["甜品", "甜点", "蛋糕"],
        "早餐": ["早餐", "早饭", "早点"],
        "主食": ["主食", "米饭", "面条", "面食", "饺子", "粥"],
        "水产": ["水产", "海鲜", "鱼", "虾", "蟹"],
        "调料": ["调料", "酱料", "料汁", "蘸料"],
        "饮品": ["饮品", "饮料", "喝的"],
        "半成品": ["半成品", "空气炸锅"],
    }

    SEASONING_DETAIL_PATTERNS = [
        "需要哪些调料",
        "需要什么调料",
        "用什么调料",
        "放什么调料",
        "哪些调料",
        "什么调料",
    ]

    METHOD_TERMS = [
        "空气炸锅",
        "微波炉",
        "凉拌",
        "清蒸",
        "水煮",
        "红烧",
        "糖醋",
        "清炒",
        "油炸",
        "炖菜",
        "煎蛋",
        "烤箱",
    ]

    NEGATIVE_PATTERNS = {
        "炒": ["不要炒", "不想炒", "别炒", "不炒", "非炒", "不以炒"],
        "辣": ["不要辣", "不吃辣", "别太辣", "不辣"],
        "油炸": ["不要油炸", "不想油炸", "别油炸"],
    }

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        从用户问题中提取高置信元数据条件。

        难度不在这里返回，避免“比较简单”这类表达被硬过滤成单一难度。
        """
        filters: Dict[str, Any] = {}
        for category in DataPreparationModule.get_supported_categories():
            if category in query:
                if category == "调料" and self.is_seasoning_detail_query(query):
                    continue
                filters["category"] = category
                break

        return filters

    def extract_preferences(self, query: str) -> Dict[str, Any]:
        """从用户问题中抽取排序偏好。"""
        preferences: Dict[str, Any] = {}

        categories = self.extract_category_preferences(query)
        if categories:
            preferences["categories"] = categories

        difficulty_scores = self.extract_difficulty_preferences(query)
        if difficulty_scores:
            preferences["difficulty_scores"] = difficulty_scores

        positive_terms = self.extract_positive_terms(query)
        if positive_terms:
            preferences["positive_terms"] = positive_terms

        negative_terms = self.extract_negative_terms(query)
        if negative_terms:
            preferences["negative_terms"] = negative_terms

        return preferences

    def extract_category_preferences(self, query: str) -> List[str]:
        """抽取用户对菜品分类的偏好。"""
        categories = []
        for category, aliases in self.CATEGORY_ALIASES.items():
            if any(alias in query for alias in aliases):
                categories.append(category)

        if self.is_seasoning_detail_query(query):
            categories = [category for category in categories if category != "调料"]

        return categories

    def is_seasoning_detail_query(self, query: str) -> bool:
        """判断用户是否在询问某道菜需要的调料，而不是要找调料类食谱。"""
        return any(pattern in query for pattern in self.SEASONING_DETAIL_PATTERNS)

    def extract_difficulty_preferences(self, query: str) -> Dict[str, float]:
        """抽取难度偏好分数。"""
        very_easy_terms = ["非常简单", "特别简单", "超简单", "最简单", "懒人", "新手", "入门"]
        easy_terms = [
            "简单",
            "容易",
            "好做",
            "快手",
            "省事",
            "方便",
            "轻松",
            "比较简单",
        ]

        if any(term in query for term in very_easy_terms):
            return {
                "非常简单": 0.022,
                "简单": 0.012,
                "中等": -0.004,
                "困难": -0.014,
                "非常困难": -0.02,
            }

        if any(term in query for term in easy_terms):
            return {
                "非常简单": 0.016,
                "简单": 0.016,
                "中等": 0.002,
                "困难": -0.012,
                "非常困难": -0.018,
            }

        if "中等" in query:
            return {
                "中等": 0.016,
                "简单": 0.006,
                "困难": 0.004,
            }

        if "困难" in query or "复杂" in query or "硬菜" in query:
            return {
                "困难": 0.016,
                "非常困难": 0.014,
                "中等": 0.004,
            }

        return {}

    def extract_positive_terms(self, query: str) -> List[str]:
        """抽取需要优先匹配的做法或关键词。"""
        return [term for term in self.METHOD_TERMS if term in query]

    def extract_negative_terms(self, query: str) -> List[str]:
        """抽取用户明确排除的做法或关键词。"""
        negative_terms = []
        for term, patterns in self.NEGATIVE_PATTERNS.items():
            if any(pattern in query for pattern in patterns):
                negative_terms.append(term)

        return negative_terms
