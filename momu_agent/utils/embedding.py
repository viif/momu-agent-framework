"""嵌入模块"""

import os
import threading
from typing import List, Optional, Union

from .config import Config
from .logger import get_logger

# 设定缓存目录
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.getcwd(), ".model_cache")


class EmbeddingModel:
    """嵌入模型基类"""

    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalTransformerEmbedding(EmbeddingModel):
    """
    本地 Transformer 嵌入实现
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.logger = get_logger(__name__)

        self._model = None
        self._dimension = None

        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "缺少可选依赖 'sentence-transformers'，请安装: pip install sentence-transformers"
            )

        self.logger.debug(f"正在加载模型: {self.model_name}")
        self.logger.debug(f"缓存路径: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")

        try:
            self._model = SentenceTransformer(self.model_name)

            test_vec = self._model.encode("test")
            self._dimension = len(test_vec)
            self.logger.info(f"模型加载成功，维度: {self._dimension}")

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"模型加载失败: {e}")

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        assert self._model is not None, "模型未加载成功"
        vecs = self._model.encode(inputs, convert_to_numpy=True)
        result = [v for v in vecs]

        return result[0] if single else result

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None


def create_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """
    工厂函数：创建具体的嵌入模型实例
    """
    return LocalTransformerEmbedding(model_name=model_name)


def get_text_embedder() -> EmbeddingModel:
    """
    获取全局共享的文本嵌入实例（线程安全单例）
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _lock:
        if _embedder is None:
            _embedder = create_embedding_model(
                model_name=Config.from_env().embed_model_name
            )
        return _embedder


def get_dimension(default: int = 384) -> int:
    """
    获取统一向量维度
    """
    try:
        return int(get_text_embedder().dimension)
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """
    强制重建嵌入实例。
    """
    global _embedder
    with _lock:
        _embedder = create_embedding_model(
            model_name=Config.from_env().embed_model_name
        )
        return _embedder
