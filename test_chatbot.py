from unittest.mock import Mock, patch

import anthropic # For spec
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
    PROVIDER_MODELS, # For Gemini test
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

@pytest.mark.asyncio
async def test_anthropic_stream(mock_model):
    mock_text_stream_content = ["Anthropic says ", "hello!"]

    # Mock for the context manager client.messages.stream(...)
    # The object returned by stream() is the context manager itself.
    # Its __enter__ method returns an object that has the text_stream attribute.
    mock_stream_context_instance = Mock()
    mock_stream_context_instance.text_stream = mock_text_stream_content # text_stream is iterable

    mock_anthropic_client = Mock(spec=anthropic.Anthropic)
    # Configure the .stream() method to return a context manager mock
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_stream_context_instance) # This is what 'stream' becomes in 'with ... as stream:'
    mock_cm.__exit__ = Mock(return_value=None)
    mock_anthropic_client.messages.stream.return_value = mock_cm

    with patch("anthropic.Anthropic", return_value=mock_anthropic_client) as mock_anthropic_init:
        provider = AnthropicProvider("anthropic_key", mock_model)
        mock_anthropic_init.assert_called_once_with(api_key="anthropic_key")

        output_chunks = []
        async for chunk in provider.stream_response("Test Anthropic"):
            output_chunks.append(chunk)

        assert output_chunks == ["Anthropic says ", "hello!"]
        assert provider.history == [
            {"role": "user", "content": "Test Anthropic"},
            {"role": "assistant", "content": "Anthropic says hello!"},
        ]
        expected_messages_to_anthropic = [{"role": "user", "content": "Test Anthropic"}]
        mock_anthropic_client.messages.stream.assert_called_once_with(
            model=mock_model.name,
            messages=expected_messages_to_anthropic,
            max_tokens=mock_model.max_tokens,
        )

@pytest.mark.asyncio
async def test_gemini_stream(mock_model):
    mock_chunk1 = Mock(text="Gemini says ")
    mock_chunk2 = Mock(text="hello!")
    mock_chunk_empty = Mock(text=None)
    mock_gemini_response_stream = [mock_chunk1, mock_chunk_empty, mock_chunk2]

    mock_chat_session = Mock() # send_message is not async itself
    mock_chat_session.send_message.return_value = mock_gemini_response_stream

    mock_generative_model = Mock()
    mock_generative_model.start_chat.return_value = mock_chat_session

    # Use the actual model name for Gemini from PROVIDER_MODELS
    gemini_model_info = PROVIDER_MODELS[Provider.GEMINI][0]

    with patch("google.generativeai.configure") as mock_configure, \
         patch("google.generativeai.GenerativeModel", return_value=mock_generative_model) as mock_gm_init:

        provider = GeminiProvider("gemini_key", gemini_model_info)

        mock_configure.assert_called_once_with(api_key="gemini_key")
        mock_gm_init.assert_called_once_with(gemini_model_info.name)
        provider.model_client.start_chat.assert_called_once_with(history=[])

        output_chunks = []
        async for chunk_text in provider.stream_response("Test Gemini"):
            output_chunks.append(chunk_text)

        assert output_chunks == ["Gemini says ", "hello!"]
        mock_chat_session.send_message.assert_called_once()
        call_args = mock_chat_session.send_message.call_args
        assert call_args[0][0] == "Test Gemini"
        assert call_args[1]['stream'] is True
        assert call_args[1]['generation_config'].max_output_tokens == gemini_model_info.max_tokens

        assert provider.history == [
            {"role": "user", "parts": ["Test Gemini"]},
            {"role": "model", "parts": ["Gemini says hello!"]},
        ]

@pytest.mark.asyncio
async def test_groq_stream(mock_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="Groq says "))]),
        Mock(choices=[Mock(delta=Mock(content="hi!"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]), # Test empty delta
    ]
    with patch("groq.Groq") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = GroqProvider("groq_key", mock_model)
        
        out = []
        async for chunk in provider.stream_response("Test Groq"):
            out.append(chunk)
        assert out == ["Groq says ", "hi!"]
        assert provider.history == [{"role": "user", "content": "Test Groq"}, {"role": "assistant", "content": "Groq says hi!"}]


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

@pytest.mark.asyncio
async def test_anthropic_error_handling(mock_model):
    mock_anthropic_client = Mock(spec=anthropic.Anthropic)
    # Make the stream call itself raise an exception
    mock_anthropic_client.messages.stream.side_effect = Exception("Anthropic API boom")

    with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
        provider = AnthropicProvider("anthropic_key", mock_model)
        out = []
        async for chunk in provider.stream_response("test message"):
            out.append(chunk)
        assert any("Error: Anthropic API boom" in c for c in out)
        assert provider.history == [{"role": "user", "content": "test message"}]

@pytest.mark.asyncio
async def test_gemini_error_handling(mock_model):
    mock_chat_session = Mock()
    mock_chat_session.send_message.side_effect = Exception("Gemini API boom")
    mock_generative_model = Mock()
    mock_generative_model.start_chat.return_value = mock_chat_session

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel", return_value=mock_generative_model):
        provider = GeminiProvider("gemini_key", mock_model)
        out = []
        async for chunk in provider.stream_response("test message"):
            out.append(chunk)
        assert any("Error: Gemini API boom" in c for c in out)

@pytest.mark.asyncio
async def test_groq_error_handling(mock_model):
    with patch("groq.Groq") as m:
        m.return_value.chat.completions.create.side_effect = Exception("Groq API boom")
        provider = GroqProvider("k", mock_model)
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert any("Error: Groq API boom" in c for c in out)

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

@pytest.mark.asyncio
async def test_deepseek_thinking_split_tags(thinking_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="<think"))]),
        Mock(choices=[Mock(delta=Mock(content="ing>hidden"))]),
        Mock(choices=[Mock(delta=Mock(content="</think"))]),
        Mock(choices=[Mock(delta=Mock(content="ing>"))]),
        Mock(choices=[Mock(delta=Mock(content="visible"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = DeepSeekProvider("k", thinking_model)
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["visible"]
