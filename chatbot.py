#!/usr/bin/env python3
"""CLI chatbot supporting multiple providers with streaming output."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from prompt_toolkit import prompt
from dotenv import load_dotenv

# Provider libraries. These are imported lazily in provider classes
import openai
import anthropic
import google.generativeai as genai
from groq import Groq

console = Console()

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"

@dataclass
class ModelInfo:
    name: str
    display_name: str
    is_thinking: bool = False
    max_tokens: int = 4096

PROVIDER_MODELS: dict[Provider, List[ModelInfo]] = {
    Provider.OPENAI: [
        ModelInfo("gpt-4o", "GPT-4o", max_tokens=128000),
        ModelInfo("gpt-4", "GPT-4"),
        ModelInfo("o3", "o3", is_thinking=True),
    ],
    Provider.ANTHROPIC: [
        ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
        ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    Provider.GEMINI: [
        ModelInfo("gemini-pro", "Gemini Pro"),
    ],
    Provider.DEEPSEEK: [
        ModelInfo("deepseek-chat", "DeepSeek Chat"),
        ModelInfo("deepseek-reasoner", "DeepSeek R1", is_thinking=True),
    ],
    Provider.GROQ: [
        ModelInfo("llama3-70b-8192", "Llama3 70B"),
    ],
}

class ChatbotProvider:
    def __init__(self, api_key: str, model: ModelInfo):
        self.api_key = api_key
        self.model = model
        self.history: List[dict[str, str]] = []

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class OpenAIProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover - network errors
            yield f"Error: {e}"

class AnthropicProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            messages = [m for m in self.history if m["role"] != "system"]
            with self.client.messages.stream(
                model=self.model.name,
                messages=messages,
                max_tokens=self.model.max_tokens,
            ) as stream:
                full = ""
                for text in stream.text_stream:
                    full += text
                    yield text
                self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GeminiProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model.name)
        self.chat = self.model_client.start_chat(history=[])

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        try:
            resp = self.chat.send_message(
                message,
                stream=True,
                generation_config=genai.types.GenerationConfig(max_output_tokens=self.model.max_tokens),
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class DeepSeekProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            hide = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if self.model.is_thinking:
                    if "<thinking>" in delta:
                        hide = True
                        continue
                    if "</thinking>" in delta:
                        hide = False
                        continue
                    if hide:
                        continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GroqProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = Groq(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class ChatBot:
    def __init__(self):
        self.provider: Optional[ChatbotProvider] = None
        self.provider_type: Optional[Provider] = None
        self.model: Optional[ModelInfo] = None

    def display_welcome(self) -> None:
        text = Text()
        text.append("Multi-Provider CLI Chatbot\n", style="bold blue")
        text.append("Supports OpenAI, Anthropic, Gemini, DeepSeek, Groq", style="green")
        console.print(Panel(text, title="Welcome", border_style="blue"))

    def select_provider_and_model(self) -> tuple[Provider, ModelInfo]:
        providers = list(Provider)
        console.print("\nProviders:", style="bold cyan")
        for i, p in enumerate(providers, 1):
            console.print(f"  {i}. {p.value}")
        while True:
            try:
                choice = int(prompt("Select provider: ")) - 1
                if 0 <= choice < len(providers):
                    provider = providers[choice]
                    break
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
            console.print("Invalid choice", style="red")

        models = PROVIDER_MODELS[provider]
        console.print(f"\nModels for {provider.value}:", style="bold cyan")
        for i, m in enumerate(models, 1):
            mark = " 🧠" if m.is_thinking else ""
            console.print(f"  {i}. {m.display_name}{mark}")
        while True:
            try:
                choice = int(prompt("Select model: ")) - 1
                if 0 <= choice < len(models):
                    model = models[choice]
                    break
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
            console.print("Invalid choice", style="red")
        return provider, model

    def get_api_key(self, provider: Provider) -> str:
        load_dotenv(dotenv_path=".env", override=False)
        env_name = f"{provider.value.upper()}_API_KEY"
        key = os.getenv(env_name)
        if key:
            console.print(
                f"Using {env_name} from environment or .env", style="green"
            )
            return key
        console.print(f"Enter your {provider.value.title()} API key:")
        key = prompt("API key: ", is_password=True)
        if not key:
            console.print("API key required", style="red")
            sys.exit(1)
        return key

    def create_provider(self, provider: Provider, key: str, model: ModelInfo) -> ChatbotProvider:
        mapping = {
            Provider.OPENAI: OpenAIProvider,
            Provider.ANTHROPIC: AnthropicProvider,
            Provider.GEMINI: GeminiProvider,
            Provider.DEEPSEEK: DeepSeekProvider,
            Provider.GROQ: GroqProvider,
        }
        cls = mapping[provider]
        return cls(key, model)

    async def chat_loop(self) -> None:
        console.print("\nType 'quit' to exit", style="green")
        if self.model and self.model.is_thinking:
            console.print("Thinking model active", style="yellow")
        while True:
            user = prompt("You: ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            if not user:
                continue
            console.print(Panel(Text(user, style="blue"), title="You", border_style="blue"))
            console.print("AI:", style="bold green")
            if self.model and self.model.is_thinking:
                with Live(Spinner("dots", text="Thinking...")):
                    await asyncio.sleep(0.5)
            out = Text()
            with Live(out, refresh_per_second=10) as live:
                async for chunk in self.provider.stream_response(user):
                    out.append(chunk, style="green")
                    live.update(out)
            console.print(Panel(out, title="Assistant", border_style="green"))

    async def run(self) -> None:
        self.display_welcome()
        self.provider_type, self.model = self.select_provider_and_model()
        key = self.get_api_key(self.provider_type)
        self.provider = self.create_provider(self.provider_type, key, self.model)
        console.print(f"Connected to {self.provider_type.value}", style="green")
        await self.chat_loop()

app = typer.Typer(help="CLI Chatbot")

@app.command()
def chat():
    chatbot = ChatBot()
    asyncio.run(chatbot.run())

if __name__ == "__main__":
    app()
