"""配置管理"""

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from ..core.exceptions import ConfigException


class Config(BaseModel):
    """配置类"""

    # LLM配置
    model_id: str
    api_key: str
    base_url: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 60.0

    # 系统配置
    log_level: str = "WARNING"

    # 其他配置
    max_history_length: int = 100

    # 搜索工具配置
    tavily_api_key: str | None = None
    serpapi_api_key: str | None = None

    # 嵌入模型配置
    embed_model_name: str | None = None

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "Config":
        """从环境变量创建配置"""
        load_dotenv(dotenv_path=dotenv_path)

        model_id = os.getenv("LLM_MODEL_ID")
        if not model_id:
            raise ConfigException("环境变量 'LLM_MODEL_ID' 缺失，请检查配置。")
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ConfigException("环境变量 'LLM_API_KEY' 缺失，请检查配置。")
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            raise ConfigException("环境变量 'LLM_BASE_URL' 缺失，请检查配置。")

        max_tokens_env = os.getenv("MAX_TOKENS")

        return cls(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            log_level=os.getenv("LOG_LEVEL", "WARNING"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(max_tokens_env) if max_tokens_env is not None else None,
            timeout=float(os.getenv("TIMEOUT", "60.0")),
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            embed_model_name=os.getenv("EMBED_MODEL_NAME"),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()
