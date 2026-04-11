import logging
from typing import Dict, Iterator, List, Optional, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .exceptions import LLMException

logger = logging.getLogger(__name__)


class MomuAgentLLM:
    """
    MomuAgent 统一 LLM 接口。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs,
    ):
        """
        初始化客户端。

        Args:
            model: 模型名称 (必填)。
            api_key: API 密钥 (必填)。
            base_url: 服务地址 (必填)。
            temperature: 生成温度。
            max_tokens: 最大生成长度。
            timeout: 超时时间（秒）。
            **kwargs: 传递给 OpenAI 客户端的其他参数。

        Raises:
            LLMException: 当必填参数缺失时抛出。
        """
        if not model:
            raise LLMException("初始化失败：必须提供 'model' 参数。")
        if not api_key:
            raise LLMException("初始化失败：必须提供 'api_key' 参数。")
        if not base_url:
            raise LLMException("初始化失败：必须提供 'base_url' 参数。")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs

        logger.debug(f"正在初始化 LLM 客户端: {self.model} @ {self.base_url}")

        try:
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            logger.info(f"✅ LLM 客户端初始化成功: {self.model}")
        except Exception as e:
            error_msg = f"LLM 客户端初始化失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg)

    def think(
        self, messages: List[Dict[str, str]], temperature: Optional[float] = None
    ) -> Iterator[str]:
        """
        流式思考接口。

        Args:
            messages: 消息列表。
            temperature: 覆盖默认温度。

        Yields:
            str: 生成的文本片段。

        Raises:
            LLMException: 当 API 调用失败时。
        """
        logger.info(f"🧠 开始流式调用模型: {self.model}")
        logger.debug(f"对话历史: {messages}")

        try:
            typed_messages = cast(List[ChatCompletionMessageParam], messages)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 处理流式响应
            full_content = ""
            for chunk in response:
                # 类型提示：ChatCompletionChunk
                content = chunk.choices[0].delta.content
                if content:
                    full_content += content
                    yield content

            logger.info("✅ 流式响应完成")
            return full_content

        except Exception as e:
            error_msg = f"流式调用失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg)

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        非流式调用接口。

        Args:
            messages: 消息列表。
        **kwargs: 覆盖初始化参数（如 temperature）。

        Returns:
            str: 完整的模型响应文本。

        Raises:
            LLMException: 当 API 调用失败时。
        """
        logger.info(f"➡️  开始非流式调用模型: {self.model}")

        try:
            typed_messages = cast(List[ChatCompletionMessageParam], messages)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMException("模型返回内容为空")
            logger.info("✅ 非流式响应完成")
            return content
        except Exception as e:
            error_msg = f"非流式调用失败: {str(e)}"
            logger.error(error_msg)
            raise LLMException(error_msg)
