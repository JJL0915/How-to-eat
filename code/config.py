"""
RAG系统配置文件
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


def _load_dotenv_file(dotenv_path: Path) -> None:
    """读取 .env 文件中的键值对配置，不覆盖系统已有环境变量。"""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_project_env() -> None:
    """优先加载项目根目录的 .env，兼容加载 code 目录下的 .env。"""
    config_dir = Path(__file__).resolve().parent
    project_root = config_dir.parent

    _load_dotenv_file(project_root / ".env")
    _load_dotenv_file(config_dir / ".env")


load_project_env()


def _env_str(*names: str, default: str = "") -> str:
    """按优先级读取第一个非空环境变量。"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    """按优先级读取整数环境变量。"""
    raw_value = _env_str(*names)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    """按优先级读取浮点数环境变量。"""
    raw_value = _env_str(*names)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # API 配置
    llm_api_key: str = field(default_factory=lambda: _env_str("LLM_API_KEY"))
    llm_base_url: str = field(
        default_factory=lambda: _env_str(
            "LLM_BASE_URL",
            default="https://api.deepseek.com",
        )
    )

    # 路径配置
    data_path: str = field(
        default_factory=lambda: _env_str(
            "RAG_DATA_PATH", "DATA_PATH", default="../data/cook"
        )
    )
    index_save_path: str = field(
        default_factory=lambda: _env_str(
            "RAG_INDEX_SAVE_PATH", "INDEX_SAVE_PATH", default="./vector_index"
        )
    )

    # 模型配置
    embedding_model: str = field(
        default_factory=lambda: _env_str(
            "EMBEDDING_MODEL", default="BAAI/bge-small-zh-v1.5"
        )
    )

    llm_model: str = field(
        default_factory=lambda: _env_str(
            "LLM_MODEL",
            default="deepseek-v4-flash",
        )
    )

    # 检索配置
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", "TOP_K", default=3))

    # 生成配置
    temperature: float = field(
        default_factory=lambda: _env_float(
            "LLM_TEMPERATURE", "TEMPERATURE", default=0.1
        )
    )
    max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", "MAX_TOKENS", default=2048)
    )

    def __post_init__(self):
        """初始化后的处理"""
        if not self.llm_api_key:
            self.llm_api_key = _env_str("LLM_API_KEY")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llm_api_key_configured": bool(self.llm_api_key),
        }


# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
