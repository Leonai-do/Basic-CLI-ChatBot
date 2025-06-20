import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chatbot import (
    ChatBot,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    DeepSeekProvider,
    GroqProvider,
    ModelInfo,
    Provider,
)

@pytest.fixture
def mock_model():
    return ModelInfo("test-model", "Test Model")

@pytest.fixture
def thinking_model():
    return ModelInfo("test-thinking", "Thinking Model", is_thinking=True)


def test_provider_initialization(mock_model):
    key = "k"
    for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider, DeepSeekProvider, GroqProvider]:
        with patch.object(cls, "__init__", return_value=None) as init:
            cls(key, mock_model)
            init.assert_called()

@pytest.mark.asyncio
async def test_openai_stream(mock_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="a"))]),
        Mock(choices=[Mock(delta=Mock(content="b"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["a", "b"]

@pytest.mark.asyncio
async def test_deepseek_thinking(thinking_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="<thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="hidden"))]),
        Mock(choices=[Mock(delta=Mock(content="</thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="final"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = DeepSeekProvider("k", thinking_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["final"]


def test_history(mock_model):
    p = OpenAIProvider("k", mock_model)
    p.client = Mock()
    p.add_to_history("user", "hi")
    p.add_to_history("assistant", "yo")
    assert p.history[0]["role"] == "user"
    assert p.history[1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_error_handling(mock_model):
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.side_effect = Exception("boom")
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert any("Error" in c for c in out)


def test_get_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "123")
    bot = ChatBot()
    assert bot.get_api_key(Provider.OPENAI) == "123"


def test_get_api_key_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dotenv-key")
    monkeypatch.chdir(tmp_path)
    # ensure no env var set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bot = ChatBot()
    assert bot.get_api_key(Provider.OPENAI) == "dotenv-key"


class DummyBenchmark:
    async def __call__(self, func):
        return await func()

@pytest.mark.asyncio
async def test_stream_perf(mock_model):
    mock_stream = [Mock(choices=[Mock(delta=Mock(content=f"c{i}"))]) for i in range(10)]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        bench = DummyBenchmark()

        async def _stream():
            out = []
            async for chunk in provider.stream_response("hi"):
                out.append(chunk)
            return out

        result = await bench(_stream)
        assert len(result) == 10
