"""
RAG系统配置文件
"""

import os
from dataclasses import dataclass
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


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # API 配置
    api_key_env_name: str = "ALI_API_KEY"
    ali_api_key: str = ""

    # 路径配置
    data_path: str = "../data/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    llm_model: str = "qwen3.5-35b-a3b"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        if not self.ali_api_key:
            self.ali_api_key = os.environ.get(self.api_key_env_name, "")

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
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_env_name": self.api_key_env_name,
        }


# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
