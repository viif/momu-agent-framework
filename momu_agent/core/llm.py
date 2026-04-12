from typing import AsyncIterator, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..utils.logger import get_logger
from .exceptions import LLMException


class LLM:
    """
    MomuAgent 统一 LLM 接口
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int = 60,
        **kwargs,
    ):
        """
        初始化异步客户端
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

        self.logger = get_logger(__name__)

        self.logger.debug(
            f"🧠 正在初始化异步 LLM 客户端: {self.model} @ {self.base_url}"
        )

        try:
            self._client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            self.logger.info(f"🧠 异步 LLM 客户端初始化成功: {self.model}")
        except Exception as e:
            error_msg = f"🧠 LLM 客户端初始化失败: {str(e)}"
            self.logger.error(error_msg)
            raise LLMException(error_msg)

    async def think(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> AsyncIterator[str]:
        """
        异步流式思考接口

        Yields:
            str: 生成的文本片段
        """
        self.logger.info(f"🧠 开始异步流式调用模型: {self.model}")

        try:
            typed_messages = cast(list[ChatCompletionMessageParam], messages)

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            full_content = ""
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_content += content
                    yield content

            self.logger.info("🧠 异步流式响应完成")

        except Exception as e:
            error_msg = f"🧠 异步流式调用失败: {str(e)}"
            self.logger.error(error_msg)
            raise LLMException(error_msg)

    async def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        异步非流式调用接口

        Returns:
            str: 完整的模型响应文本
        """
        self.logger.info(f"🧠 开始异步非流式调用模型: {self.model}")

        try:
            typed_messages = cast(list[ChatCompletionMessageParam], messages)

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMException("模型返回内容为空")

            self.logger.info("🧠 异步非流式响应完成")
            return content

        except Exception as e:
            error_msg = f"🧠 异步非流式调用失败: {str(e)}"
            self.logger.error(error_msg)
            raise LLMException(error_msg)
