import builtins
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import momu_agent.utils.embedding as embedding


class _FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy: bool = False):
        if isinstance(texts, str):
            return [1.0, 2.0, 3.0]
        return [
            [float(len(text)), float(len(text)) + 1, float(len(text)) + 2]
            for text in texts
        ]


@pytest.fixture(autouse=True)
def reset_embedder():
    embedding._embedder = None
    yield
    embedding._embedder = None


def test_local_transformer_embedding_encode_and_dimension():
    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        model = embedding.LocalTransformerEmbedding("fake-model")

    assert model.dimension == 3
    assert model.encode("hi") == [2.0, 3.0, 4.0]
    assert model.encode(["a", "abcd"]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_local_transformer_embedding_import_error():
    real_import = builtins.__import__

    def _mocked_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing dependency")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mocked_import):
        with pytest.raises(ImportError, match="sentence-transformers"):
            embedding.LocalTransformerEmbedding("fake-model")


def test_local_transformer_embedding_load_runtime_error():
    class _BrokenSentenceTransformer:
        def __init__(self, model_name: str):
            _ = model_name
            raise ValueError("boom")

    fake_module = SimpleNamespace(SentenceTransformer=_BrokenSentenceTransformer)
    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        with pytest.raises(RuntimeError, match="模型加载失败"):
            embedding.LocalTransformerEmbedding("fake-model")


def test_create_embedding_model_uses_local_transformer():
    with patch.object(embedding, "LocalTransformerEmbedding") as mocked_cls:
        instance = object()
        mocked_cls.return_value = instance

        result = embedding.create_embedding_model("my-model")

    assert result is instance
    mocked_cls.assert_called_once_with(model_name="my-model")


def test_get_text_embedder_returns_cached_instance():
    expected = object()
    fake_config = SimpleNamespace(embed_model_name="cached-model")

    with (
        patch.object(
            embedding.Config, "from_env", return_value=fake_config
        ) as mocked_cfg,
        patch.object(
            embedding,
            "create_embedding_model",
            return_value=expected,
        ) as mocked_create,
    ):
        first = embedding.get_text_embedder()
        second = embedding.get_text_embedder()

    assert first is expected
    assert second is expected
    mocked_cfg.assert_called_once_with()
    mocked_create.assert_called_once_with(model_name="cached-model")


def test_get_dimension_returns_embedder_dimension():
    fake_embedder = SimpleNamespace(dimension=768)

    with patch.object(embedding, "get_text_embedder", return_value=fake_embedder):
        assert embedding.get_dimension() == 768


def test_get_dimension_returns_default_on_error():
    with patch.object(
        embedding, "get_text_embedder", side_effect=RuntimeError("failed")
    ):
        assert embedding.get_dimension(default=256) == 256


def test_refresh_embedder_rebuilds_instance():
    old_instance = object()
    new_instance = object()
    embedding._embedder = old_instance
    fake_config = SimpleNamespace(embed_model_name="new-model")

    with (
        patch.object(
            embedding.Config, "from_env", return_value=fake_config
        ) as mocked_cfg,
        patch.object(
            embedding,
            "create_embedding_model",
            return_value=new_instance,
        ) as mocked_create,
    ):
        refreshed = embedding.refresh_embedder()

    assert refreshed is new_instance
    assert embedding._embedder is new_instance
    mocked_cfg.assert_called_once_with()
    mocked_create.assert_called_once_with(model_name="new-model")
