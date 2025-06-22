#!/usr/bin/env python3
"""CLI chatbot supporting multiple providers with streaming output."""

from __future__ import annotations

import asyncio
import os
import sys
import json
import aiohttp
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional
import re

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt, IntPrompt
from prompt_toolkit import prompt
from dotenv import load_dotenv

# Provider libraries
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
    MCP = "mcp"

@dataclass
class ModelInfo:
    name: str
    display_name: str
    is_thinking: bool = False
    max_tokens: int = 4096

PROVIDER_MODELS: dict[Provider, List[ModelInfo]] = {
    Provider.OPENAI: [
        ModelInfo("gpt-4o", "GPT-4o", max_tokens=128000),
        ModelInfo("gpt-4o-mini", "GPT-4o Mini", max_tokens=128000),
        ModelInfo("gpt-4", "GPT-4"),
        ModelInfo("o1", "o1", is_thinking=True),
        ModelInfo("o1-mini", "o1-mini", is_thinking=True),
    ],
    Provider.ANTHROPIC: [
        ModelInfo("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ModelInfo("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
        ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
        ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    Provider.GEMINI: [
        ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro"),
        ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash"),
        ModelInfo("gemini-pro", "Gemini Pro"),
    ],
    Provider.DEEPSEEK: [
        ModelInfo("deepseek-chat", "DeepSeek Chat"),
        ModelInfo("deepseek-reasoner", "DeepSeek R1", is_thinking=True),
    ],
    Provider.GROQ: [
        ModelInfo("llama-3.1-70b-versatile", "Llama 3.1 70B"),
        ModelInfo("llama-3.1-8b-instant", "Llama 3.1 8B"),
        ModelInfo("mixtral-8x7b-32768", "Mixtral 8x7B"),
    ],
    Provider.MCP: [
        ModelInfo("mcp-standard", "MCP Standard", max_tokens=4096),
        ModelInfo("mcp-advanced", "MCP Advanced", max_tokens=8192),
        ModelInfo("mcp-enterprise", "MCP Enterprise", max_tokens=32768),
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
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            # Handle o1 models differently (no streaming, no system messages)
            if self.model.name.startswith(("o1", "o3")):
                # o1 models don't support streaming
                response = await self.client.chat.completions.create(
                    model=self.model.name,
                    messages=[m for m in self.history if m["role"] != "system"],
                    max_completion_tokens=self.model.max_tokens,
                )
                content = response.choices[0].message.content or ""
                self.add_to_history("assistant", content)
                yield content
            else:
                stream = await self.client.chat.completions.create(
                    model=self.model.name,
                    messages=self.history,
                    stream=True,
                    max_tokens=self.model.max_tokens,
                )
                full = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        full += delta
                        yield delta
                self.add_to_history("assistant", full)
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            yield error_msg
        except openai.APIConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            yield error_msg
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            yield error_msg
        except openai.APIConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            yield error_msg

class AnthropicProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            messages = [m for m in self.history if m["role"] != "system"]
            async with self.client.messages.stream(
                model=self.model.name,
                messages=messages,
                max_tokens=self.model.max_tokens,
            ) as stream:
                full = ""
                async for text in stream.text_stream:
                    full += text
                    yield text
                self.add_to_history("assistant", full)
        except Exception as e:
            error_msg = f"Anthropic error: {str(e)}"
            yield error_msg

class GeminiProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model.name)
        self.chat = self.model_client.start_chat(history=[])

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        try:
            response = await asyncio.to_thread(
                self.chat.send_message,
                message,
                stream=True,
                generation_config=genai.types.GenerationConfig(max_output_tokens=self.model.max_tokens),
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            error_msg = f"Gemini error: {str(e)}"
            yield error_msg

class DeepSeekProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._thinking_buffer = ""
        self._inside_thinking = False

    def _process_thinking_content(self, delta: str) -> str:
        """Process delta content to hide thinking tags for thinking models."""
        if not self.model.is_thinking:
            return delta
        
        self._thinking_buffer += delta
        output = ""
        
        # Process the buffer to extract content outside thinking tags
        while self._thinking_buffer:
            if not self._inside_thinking:
                # Look for opening tag
                start_idx = self._thinking_buffer.find("<thinking>")
                if start_idx == -1:
                    # No opening tag found, output everything
                    output += self._thinking_buffer
                    self._thinking_buffer = ""
                else:
                    # Output content before opening tag
                    output += self._thinking_buffer[:start_idx]
                    self._thinking_buffer = self._thinking_buffer[start_idx + 10:]  # len("<thinking>") = 10
                    self._inside_thinking = True
            else:
                # Look for closing tag
                end_idx = self._thinking_buffer.find("</thinking>")
                if end_idx == -1:
                    # No closing tag found yet, discard buffer content
                    self._thinking_buffer = ""
                else:
                    # Skip content inside thinking tags
                    self._thinking_buffer = self._thinking_buffer[end_idx + 11:]  # len("</thinking>") = 11
                    self._inside_thinking = False
        
        return output

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        self._thinking_buffer = ""
        self._inside_thinking = False
        try:
            stream = await self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    processed_delta = self._process_thinking_content(delta)
                    if processed_delta:
                        full += processed_delta
                        yield processed_delta
            self.add_to_history("assistant", full)
        except Exception as e:
            error_msg = f"DeepSeek error: {str(e)}"
            yield error_msg

class MCPProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.base_url = os.getenv("MCP_BASE_URL", "https://api.mcp-server.com")
        self.session = aiohttp.ClientSession()
        
    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            payload = {
                "model": self.model.name,
                "messages": self.history,
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_tool",
                            "description": "Execute a tool via MCP protocol",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "tool_name": {"type": "string"},
                                    "parameters": {"type": "object"}
                                },
                                "required": ["tool_name"]
                            }
                        }
                    }
                ]
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            ) as response:
                async for line in response.content:
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
        except Exception as e:
            error_msg = f"MCP error: {str(e)}"
            yield error_msg

class GroqProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = Groq(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full += delta
                    yield delta
            self.add_to_history("assistant", full)
        except Exception as e:
            error_msg = f"Groq error: {str(e)}"
            yield error_msg

class ChatBot:
    def __init__(self):
        self.provider: Optional[ChatbotProvider] = None
        self.provider_type: Optional[Provider] = None
        self.model: Optional[ModelInfo] = None

    def display_welcome(self) -> None:
        console.print(Panel(
            "[bold blue]🤖 Multi-Provider CLI Chatbot\n\n"
            "📡 Supports: [green]OpenAI, Anthropic, Gemini, DeepSeek, Groq[/green]\n"
            "💬 Type your message, 'clear' to reset, or 'quit' to exit",
            title="Welcome to ChatBot",
            title_align="center",
            border_style="blue",
            padding=(1, 2)
        ))

    def select_provider_and_model(self) -> tuple[Provider, ModelInfo]:
        providers = list(Provider)
        console.print("\n📡 Available Providers:", style="bold cyan")
        for i, p in enumerate(providers, 1):
            console.print(f"  {i}. {p.value.title()}")
        
        provider_choice = IntPrompt.ask(
            "Select provider", 
            choices=[str(i) for i in range(1, len(providers) + 1)],
            default=1
        )
        provider = providers[provider_choice - 1]
        
        models = PROVIDER_MODELS[provider]
        console.print(f"\n🧠 Models for {provider.value.title()}:", style="bold cyan")
        for i, m in enumerate(models, 1):
            mark = " 🤔" if m.is_thinking else ""
            console.print(f"  {i}. {m.display_name}{mark}")
        
        model_choice = IntPrompt.ask(
            "Select model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default=1
        )
        model = models[model_choice - 1]
        
        return provider, model

    def get_api_key(self, provider: Provider) -> str:
        load_dotenv(dotenv_path=".env", override=False)
        env_name = f"{provider.value.upper()}_API_KEY"
        key = os.getenv(env_name)
        if key:
            console.print(
                f"✅ Using {env_name} from environment", style="green"
            )
            return key
        
        console.print(f"🔑 Enter your {provider.value.title()} API key:")
        key = Prompt.ask("API key", password=True)
        if not key:
            console.print("❌ API key required", style="bold red")
            sys.exit(1)
        return key

    def create_provider(self, provider: Provider, key: str, model: ModelInfo) -> ChatbotProvider:
        mapping = {
            Provider.OPENAI: OpenAIProvider,
            Provider.ANTHROPIC: AnthropicProvider,
            Provider.GEMINI: GeminiProvider,
            Provider.DEEPSEEK: DeepSeekProvider,
            Provider.GROQ: GroqProvider,
            Provider.MCP: MCPProvider,
        }
        cls = mapping[provider]
        return cls(key, model)

    async def chat_loop(self) -> None:
        console.print("\n💬 Chat started! Type 'quit', 'exit', or 'clear' to manage the session", style="green")
        if self.model and self.model.is_thinking:
            console.print("🤔 Thinking model active - responses may take longer", style="yellow")
        
        while True:
            try:
                # Get input directly without empty panel
                user_input = prompt("You 👤 > ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n👋 Goodbye!", style="yellow")
                break
                
            if user_input.lower() in {"quit", "exit", "q"}:
                console.print("👋 Goodbye!", style="yellow")
                break
            elif user_input.lower() == "clear":
                if self.provider:
                    self.provider.history.clear()
                    console.print("🧹 Chat history cleared!", style="yellow")
                else:
                    console.print("⚠️ No active provider to clear history", style="yellow")
                continue
            elif not user_input:
                continue
            
            # Only display user message in panel format
            console.print(
                Panel(
                    user_input,
                    title="You 👤",
                    title_align="left",
                    style="blue",
                    padding=(0, 2),
                    expand=False
                )
            )
            
            # Show thinking indicator for thinking models
            if self.model and self.model.is_thinking:
                with Live(Spinner("dots", text="🤔 Thinking..."), refresh_per_second=10):
                    await asyncio.sleep(0.5)
            
            if self.provider:
                # Create a panel for the assistant response
                panel_title = f"{self.model.display_name} 🤖"
                panel = Panel("", title=panel_title, title_align="left", 
                              style="green", padding=(0, 2), expand=False)
                
                try:
                    with Live(panel, console=console, refresh_per_second=10) as live:
                        full_response = ""
                        async for chunk in self.provider.stream_response(user_input):
                            full_response += chunk
                            # Format markdown and preserve newlines
                            formatted = full_response.replace('\n', '\n\n')
                            panel.renderable = formatted
                            live.update(panel)
                except KeyboardInterrupt:
                    console.print("\n⏹️ Response interrupted", style="yellow")
                except Exception as e:
                    console.print(f"\n❌ Error during response: {e}", style="bold red")
            else:
                console.print("⚠️ No active provider to generate response", style="yellow")

    async def run(self) -> None:
        try:
            self.display_welcome()
            self.provider_type, self.model = self.select_provider_and_model()
            key = self.get_api_key(self.provider_type)
            self.provider = self.create_provider(self.provider_type, key, self.model)
            console.print(f"🔗 Connected to {self.provider_type.value.title()} ({self.model.display_name})", style="green")
            await self.chat_loop()
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="yellow")
        except Exception as e:
            console.print(f"❌ Fatal error: {e}", style="bold red")
            sys.exit(1)

app = typer.Typer(help="Multi-Provider CLI Chatbot")

@app.command()
def chat():
    """Start the interactive chatbot CLI."""
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass  # nest_asyncio not required in all environments
    
    chatbot = ChatBot()
    asyncio.run(chatbot.run())

def main():
    import sys
    # If 'chat' command is provided, run the chat function directly
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        asyncio.run(ChatBot().run())
    else:
        app()

if __name__ == "__main__":
    main()
