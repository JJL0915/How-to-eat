"""
会话记忆模块。

仅保存当前进程内的最近几轮对话，不做持久化。
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List, Optional


class ConversationMemory:
    """按 session_id 管理短期对话历史。"""

    def __init__(
        self,
        max_turns: int = 6,
        max_message_chars: int = 1200,
        max_context_chars: int = 3000,
    ):
        self.max_turns = max_turns
        self.max_messages = max_turns * 2
        self.max_message_chars = max_message_chars
        self.max_context_chars = max_context_chars
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """返回指定会话的消息副本。"""
        normalized_session_id = self._normalize_session_id(session_id)
        with self._lock:
            return list(self._sessions.get(normalized_session_id, []))

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """保存一轮用户问题和助手回答。"""
        normalized_session_id = self._normalize_session_id(session_id)
        user_message = self._trim_message(user_message)
        assistant_message = self._trim_message(assistant_message)
        if not user_message or not assistant_message:
            return

        with self._lock:
            messages = self._sessions.setdefault(normalized_session_id, [])
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
            if len(messages) > self.max_messages:
                del messages[: len(messages) - self.max_messages]

    def format_history(self, session_id: str) -> str:
        """将最近对话格式化为 prompt 可读的短文本。"""
        messages = self.get_messages(session_id)
        if not messages:
            return ""

        lines = []
        for message in messages:
            role = "用户" if message["role"] == "user" else "助手"
            content = message["content"].replace("\n", " ").strip()
            lines.append(f"{role}: {content}")

        history = "\n".join(lines)
        if len(history) <= self.max_context_chars:
            return history

        return history[-self.max_context_chars :].lstrip()

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """返回指定会话的结构化状态副本。"""
        normalized_session_id = self._normalize_session_id(session_id)
        with self._lock:
            state = self._states.get(normalized_session_id, {})
            return {
                "focus_dish": state.get("focus_dish", ""),
                "candidate_dishes": list(state.get("candidate_dishes", [])),
                "last_route_type": state.get("last_route_type", ""),
            }

    def update_focus(
        self,
        session_id: str,
        *,
        focus_dish: Optional[str] = None,
        candidate_dishes: Optional[List[str]] = None,
        route_type: str = "",
    ) -> None:
        """更新当前会话的焦点菜品和候选菜品列表。"""
        normalized_session_id = self._normalize_session_id(session_id)
        with self._lock:
            state = self._states.setdefault(normalized_session_id, {})
            if focus_dish is not None:
                state["focus_dish"] = focus_dish
            if candidate_dishes is not None:
                state["candidate_dishes"] = list(candidate_dishes)
            if route_type:
                state["last_route_type"] = route_type

    def clear(self, session_id: str) -> None:
        """清空指定会话。"""
        normalized_session_id = self._normalize_session_id(session_id)
        with self._lock:
            self._sessions.pop(normalized_session_id, None)
            self._states.pop(normalized_session_id, None)

    def _trim_message(self, message: str) -> str:
        """限制单条消息长度，避免 prompt 膨胀。"""
        message = (message or "").strip()
        if len(message) <= self.max_message_chars:
            return message
        return message[: self.max_message_chars].rstrip() + "...[已截断]"

    def _normalize_session_id(self, session_id: str) -> str:
        """规范化 session_id，避免空字符串产生散乱会话。"""
        session_id = (session_id or "").strip()
        return session_id or "default"
