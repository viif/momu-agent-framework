import os
from unittest.mock import patch

import pytest

with patch("momu_agent.core.config.load_dotenv"):
    from momu_agent.core.config import Config

from momu_agent.core.exceptions import ConfigException


class TestConfig:
    """测试 Config 类"""

    def _mock_env(self, **kwargs):
        """
        辅助方法：模拟环境变量
        使用 patch.dict 临时修改 os.environ，测试结束后自动恢复
        """
        base_env = {
            "LLM_MODEL_ID": "qwen-turbo",
            "LLM_API_KEY": "sk-test123",
            "LLM_BASE_URL": "http://test.url",
        }
        base_env.update(kwargs)
        return patch.dict(os.environ, base_env, clear=True)

    def test_from_env_success(self):
        """测试当所有必填环境变量都存在时，配置加载成功"""
        with self._mock_env():
            config = Config.from_env()

            assert config.model_id == "qwen-turbo"
            assert config.api_key == "sk-test123"
            assert config.base_url == "http://test.url"
            # 检查默认值
            assert config.temperature == 0.7
            assert config.debug is False
            assert config.max_tokens is None
            assert config.timeout == 60.0
            assert config.max_history_length == 100

    def test_from_env_with_custom_params(self):
        """测试加载非默认的环境变量值"""
        with self._mock_env(
            TEMPERATURE="0.9",
            DEBUG="true",
            MAX_TOKENS="2048",
            LOG_LEVEL="DEBUG",
            TIMEOUT="120.5",
        ):
            config = Config.from_env()

            assert config.temperature == 0.9
            assert config.debug is True
            assert config.max_tokens == 2048
            assert config.log_level == "DEBUG"
            assert config.timeout == 120.5

    def test_from_env_missing_model_id(self):
        """测试缺失 LLM_MODEL_ID 时抛出异常"""
        with self._mock_env(LLM_MODEL_ID=""):
            with pytest.raises(ConfigException) as exc_info:
                Config.from_env()
            assert "LLM_MODEL_ID" in str(exc_info.value)

    def test_from_env_missing_api_key(self):
        """测试缺失 LLM_API_KEY 时抛出异常"""
        env_vars = {
            "LLM_MODEL_ID": "qwen",
            # "LLM_API_KEY": "missing",
            "LLM_BASE_URL": "http://test.url",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigException) as exc_info:
                Config.from_env()
            assert "LLM_API_KEY" in str(exc_info.value)

    def test_from_env_missing_base_url(self):
        """测试缺失 LLM_BASE_URL 时抛出异常"""
        env_vars = {
            "LLM_MODEL_ID": "qwen",
            "LLM_API_KEY": "sk-123",
            # "LLM_BASE_URL": "missing"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigException) as exc_info:
                Config.from_env()
            assert "LLM_BASE_URL" in str(exc_info.value)

    def test_to_dict(self):
        """测试转换为字典格式"""
        with self._mock_env():
            config = Config.from_env()
            config_dict = config.to_dict()

            assert isinstance(config_dict, dict)
            assert config_dict["model_id"] == "qwen-turbo"
            assert config_dict["api_key"] == "sk-test123"
            assert config_dict["timeout"] == 60.0
